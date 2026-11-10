"""Worker processes: play games with the current weights (training and eval), verify wins on the spot, and
replay saved games into training data (self-imitation)."""

import math
import os
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DISCARD, HAND_COL, PLAY, collate, game_config, rand_seed  # noqa: E402
from curriculum import CURRICULUM_ANTES  # noqa: E402
from explore import explore_sample  # noqa: E402
from model import EXPERT_HANDS, make_model, phase_member  # noqa: E402
from pins import (apply_pin, count_pin_move, early_pin, moe_min_plays, moe_pin_rate,  # noqa: E402
                  new_pin_stats, pin_vocab_id)
from rewards import REWARDS, build_shaping, shaping, step_discount  # noqa: E402


# ---------------- workers ----------------

def inject_at(args, kind):
    """Iterations where a fault is injected (testing the safety checks), e.g. --inject nan:3,crash:5"""
    out = set()
    for part in (args.inject or "").split(","):
        if part.startswith(kind + ":"):
            out.add(int(part.split(":")[1]))
    return out


def worker_main(wid, args, in_q, out_q):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    try:
        from sim import Sim, sim_kwargs
        sim = Sim(args.src, args.mods, mode=args.actions, enc=args.enc, **sim_kwargs(args))
    except Exception:
        out_q.put(("error", wid, traceback.format_exc()))
        return
    model = make_model(args, len(sim.vocab))
    model.eval()
    reward_fn = REWARDS[args.reward]
    rng = random.Random(f"{args.seed}-{wid}")
    out_q.put(("ready", wid, sim.vocab))
    crash_at = inject_at(args, "crash")
    while True:
        msg = in_q.get()
        if msg is None:
            return
        ticket = msg[5]
        try:
            out_q.put(("done", wid, play_games(sim, model, reward_fn, rng, args, wid, msg[:5], crash_at), ticket))
        except Exception:
            out_q.put(("error", wid, traceback.format_exc(), ticket))


def play_games(sim, model, reward_fn, rng, args, wid, msg, crash_at):
        # mode: "train" (sampled, steps kept for PPO), "eval" (sampled with a fixed RNG per seed, so
        # scores are comparable between rounds), "greedy" (most likely action; can loop forever
        # on free actions like dragging jokers, so it is a diagnostic only).
        state, seeds, mode, it, gen = msg
        if mode == "replay":
            return replay_games(sim, reward_fn, args, seeds)
        greedy = mode == "greedy"
        if getattr(sim, "phi_loaded", False):
            # v4: phi_build only feeds training's shaping; evals and tests read only the offer values
            sim.build_phi = mode == "train" and bool(getattr(args, "build_shaping", 0))
        if it in crash_at and gen == 0 and wid == 0 and mode == "train":
            raise RuntimeError(f"injected worker crash at iteration {it}")
        model.load_state_dict(state)
        trng = random.Random(f"{args.seed}-{wid}-{it}")      # reproducible sampling per worker+iteration
        pin_id0 = pin_vocab_id(sim.vocab, args.pin_hand)
        if args.moe:
            hand_ids = [pin_vocab_id(sim.vocab, h) for h in EXPERT_HANDS]
            base = model.base_rate.numpy().astype(np.float64)
            rates = [moe_pin_rate(h, args) * float(model.pin_scale[i]) for i, h in enumerate(EXPERT_HANDS)]
            # The early model's own pin (--moe-early-pin): its hand or any higher one, fading by the run's games.
            early_ids, early_rate = early_pin(sim.vocab, args)
            early_rate *= float(model.early_pin_scale[0])
            min_plays = np.array([moe_min_plays(h, args) for h in EXPERT_HANDS], dtype=np.float64)
        episodes = []
        for seed in seeds:
            start = seed if isinstance(seed, dict) else None      # v4 curriculum start (see curriculum_entry)
            seed = start["seed"] if start else (seed or rand_seed(rng))
            srng = random.Random(f"eval-{seed}") if mode == "eval" else trng
            # The pin's own coin, apart from the sampling RNG, so pinning doesn't shift eval sampling.
            prng = random.Random(f"pin-eval-{seed}" if mode == "eval" else f"pin-{args.seed}-{wid}-{it}-{seed}")
            pin_id, pin_rate = pin_id0, args.pin_rate
            if args.moe and early_ids:
                pin_id, pin_rate = early_ids, early_rate
            pst = new_pin_stats() if pin_id is not None else None
            early_pst = pst if args.moe else None
            member, routed, counts, handoff_at, rule_k = 0, None, np.zeros(len(EXPERT_HANDS)), None, None
            late, late_plays = np.zeros(len(EXPERT_HANDS)), 0     # plays after the handoff, by hand
            if start:
                back, stake, challenge = start["back"], start["stake"], start.get("challenge")
            else:
                back, stake, challenge = game_config(seed, args)
            o = sim.reset(seed, back, stake, challenge)
            t_replay, cur_info = 0.0, None
            if start:
                tr = time.time()
                o = sim.fast_forward(start["acts"], start["desc"], start.get("until_ante"))
                t_replay = time.time() - tr
                if o is None:
                    raise RuntimeError(f"curriculum start {start['from']} {seed} ante {start['ante']} did not replay")
                cur_info = {"ante": start["ante"], "from": start["from"], "start_step": sim.steps}
            shop_at = {}            # first shop of each curriculum ante: where this game could start others
            if pst is not None:
                apply_pin(o, pin_id, pin_rate, prng, pst)
            steps, done, tail, reached, disc, members = [], False, [], False, [], []
            t_model = t_sim = 0.0
            while not done and sim.steps < args.max_steps:
                tm = time.time()
                if args.moe:
                    o["member"] = member
                elif getattr(args, "phase_split", False):
                    st_now = sim.H.state_name()
                    o["member"] = phase_member(st_now)     # v4: table or shop network
                    members.append(o["member"])
                    a_now = sim.progress["ante"]
                    if mode == "train" and st_now == "SHOP" and a_now in CURRICULUM_ANTES and a_now not in shop_at:
                        shop_at[a_now] = sim.steps
                with torch.no_grad():
                    logits, v = model(*collate([o]))
                t_model += time.time() - tm
                temp = getattr(args, "eval_temp", 1.0) if mode == "eval" else 1.0
                p = torch.softmax(logits[0] / temp if temp != 1.0 else logits[0], -1).numpy().astype(np.float64)
                if not np.isfinite(p).all():
                    raise FloatingPointError("non-finite action probabilities")
                if greedy:
                    a = int(p.argmax())
                    logp = float(math.log(max(p[a], 1e-30)))
                elif mode == "train" and args.explore_hands > 0:
                    a, logp = explore_sample(p, o["act"], args.explore_hands, srng, o.get("allow"))
                else:
                    a = int(np.searchsorted(np.cumsum(p), srng.random() * p.sum()))
                    a = min(a, len(p) - 1)
                    logp = float(math.log(max(p[a], 1e-30)))
                if mode != "train" and sim.steps >= args.max_steps - 30:
                    tail.append((sim.H.state_name(), int(o["act"][a][0]), int(o["act"][a][1]),
                                 int(o["act"][a][2]), round(float(p[a]), 4), len(p)))
                ts = time.time()
                prev, cur, done = sim.step(a)
                if args.stop_ante and not done and cur["ante"] >= args.stop_ante \
                        and cur["bosses_beaten"] >= args.stop_ante - 1:
                    done, reached = True, True        # early-game model: its run ends here (see --stop-ante)
                g_t, l_t = step_discount(prev, cur, args)
                r = reward_fn(prev, cur) + args.shaping * shaping(prev, cur, g_t) \
                    + getattr(args, "build_shaping", 0) * build_shaping(prev, cur, g_t)
                steps.append((o, a, logp, float(v[0]), r, reward_fn(prev, cur)))
                disc.append((g_t, l_t, int(o["act"][a][0]) in (PLAY, DISCARD)))   # 3rd: a hand (v4b's hand clock)
                if pst is not None:
                    count_pin_move(o, a, pin_id, pst)
                if args.moe and handoff_at is not None and o["act"][a][0] == PLAY:
                    late_plays += 1
                    for k2, hid in enumerate(hand_ids):
                        late[k2] += int(o["act"][a][HAND_COL] == hid)
                if args.moe and member == 0 and handoff_at is None:
                    row = o["act"][a]
                    if row[0] == PLAY:
                        for k, hid in enumerate(hand_ids):
                            counts[k] += int(row[HAND_COL] == hid)
                    if not done and cur["ante"] >= args.moe_handoff and cur["bosses_beaten"] >= args.moe_handoff - 1:
                        # Handoff (the boss before --moe-handoff beaten): the expert for the hand this
                        # run leaned on most, relative to how often the early model plays it; in
                        # training games, --moe-random of them go to an expert picked at random.
                        # A hand with a minimum (--moe-min-plays) needs that many plays to be chosen.
                        score = np.where(counts >= min_plays, (counts + 1.0) / (base + 1.0), -np.inf)
                        rule_k = int(np.argmax(score))
                        force = getattr(args, "moe_force", None)
                        if force == -1:
                            # no handoff: the early head plays on (analysis control); handoff_at marks
                            # where the handoff would have been
                            handoff_at, routed = sim.steps, "none"
                        elif force is not None:
                            # analysis/paired_experts.py: every expert from the same handoff state
                            k, routed = int(force), "forced"
                        elif mode == "train" and prng.random() < args.moe_random:
                            k, routed = min(int(prng.random() * len(EXPERT_HANDS)), len(EXPERT_HANDS) - 1), "random"
                        else:
                            k, routed = rule_k, "rule"
                        if force != -1:
                            member, handoff_at = 1 + k, sim.steps
                            pin_id, pin_rate, pst = hand_ids[k], rates[k], new_pin_stats()
                        ro = getattr(args, "moe_rollout", 0)
                        if ro and mode == "eval":
                            # another play-out from the same handoff state: same moves up to here, its
                            # own sampling after (each rollout shares its RNG across experts)
                            srng = random.Random(f"eval-{seed}-rollout{ro}")
                if not done:
                    o = sim.observe()
                    if pst is not None:
                        apply_pin(o, pin_id, pin_rate, prng, pst)
                t_sim += time.time() - ts
            ep = {"seed": seed, "back": back, "stake": stake, "challenge": challenge,
                  "steps": steps if mode == "train" else None, "n": sim.steps,
                  "disc": disc if mode == "train" else None,
                  "progress": sim.progress, "capped": not done, "tail": tail if not done else None,
                  "return": sum(s[5] for s in steps),
                  # the exact game: seed + every action (index into the legal list, and as text)
                  "acts": list(sim.route), "desc": list(sim.desc), "final": sim.final_hash(), "verified": None,
                  "t_model": t_model, "t_sim": t_sim, "members": members if getattr(args, "phase_split", False) else None,
                  "curriculum": cur_info, "t_replay": t_replay, "shop_at": shop_at, "pin": pst if pst is not early_pst else None, "reached": reached,
                  "moe": ({"expert": EXPERT_HANDS[member - 1] if member else None, "routed": routed,
                           "rule_choice": EXPERT_HANDS[rule_k] if rule_k is not None else None,
                           "late_counts": late.tolist(), "late_plays": late_plays,
                           "early_counts": counts.tolist(), "handoff_step": handoff_at,
                  "early_pin": early_pst} if args.moe else None)}
            if ep["progress"]["won"] or ep["progress"].get("near_win"):
                ep["jokers_held"] = sim.env.jokers_held()      # before the replay check resets the sim
                if getattr(args, "verify_wins", True):     # off when playing a save: a replay would count twice
                    ep["verified"] = verify_route(sim, args, ep)
            episodes.append(ep)
        return episodes


def verify_route(sim, args, ep):
    """Plays a won game again from its saved seed and actions; True only if every step is the same
    move and the game ends in the same state. (The game is deterministic, so this must hold.)"""
    prog, n, final, problems = sim.replay(ep["seed"], ep["back"], ep["stake"], ep["acts"], ep["desc"], ep["challenge"])
    return not problems and n == ep["n"] and final == ep["final"] and prog == ep["progress"]


def replay_games(sim, reward_fn, args, routes):
    """Self-imitation: replays saved games into training data (observation, action, reward per step).
    Saved routes are exact, so this regenerates the very states the bot saw when it played them."""
    out = []
    for r in routes:
        o = sim.reset(r["seed"], *game_config(r["seed"], args))
        steps, done, bad, gam = [], False, None, []
        for k, a in enumerate(r["acts"]):
            if done or a >= len(o["act"]) or sim.H.action_desc(a + 1) != r["desc"][k]:
                bad = f"step {k} differs from the saved game"
                break
            prev, cur, done = sim.step(a)
            g_t = step_discount(prev, cur, args)[0]
            steps.append((o, a, reward_fn(prev, cur) + args.shaping * shaping(prev, cur, g_t)
                          + getattr(args, "build_shaping", 0) * build_shaping(prev, cur, g_t)))
            gam.append(g_t)
            if not done:
                o = sim.observe()
        if bad is None and sim.final_hash() != r["final"]:
            bad = "final state differs from the saved game"
        out.append({"key": r["key"], "steps": None if bad else steps, "gammas": None if bad else gam, "problem": bad})
    return out
