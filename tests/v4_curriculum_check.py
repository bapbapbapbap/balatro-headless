"""Pre-flight for model v4's curriculum (docs/hypotheses.md, 2026-11-09 amendment, change 1): starts reached by
replaying a saved game from its seed. Needs torch (it plays through train.play_games) and the real game.

  1. A published start (Sim.fast_forward to ante k) stops at the first move made in a SHOP at ante k,
     the same move an independent replay finds; from there the win's remaining moves are offered as the
     same moves and end in the same won game.
  2. Through the trainer (train.play_games with a curriculum start): the game is marked as curriculum with
     its start move, its trained steps begin there, and the run's own-start record (shop_at) points at
     the start's shop; a start built from it (curriculum_entry 'own', the moves up to that shop) reaches
     the identical game state (diff_check's complete dump).
  3. Held-out seeds (the eval 600 and every registered test tag) are never used as published starts.
  Reports the replay cost per start by ante, against the sim time of the published games themselves.

  python3 v4_curriculum_check.py <game source dir> [--routes 6]
"""
import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import diff_check  # noqa: E402
import train as T  # noqa: E402
from model import make_model  # noqa: E402
from sim import Sim, sim_kwargs  # noqa: E402

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def first_shop(src, r, antes):
    """Independent replay: the move index of the first move made in a SHOP at each ante, and the full game's end."""
    s = Sim(src, mode="ordered-capped", enc=2)
    s.reset(r["seed"], r["back"], r["stake"])
    at = {}
    for k, a in enumerate(r["acts"]):
        s.observe()
        if s.H.state_name() == "SHOP" and s.progress["ante"] in antes and s.progress["ante"] not in at:
            at[s.progress["ante"]] = k
        s.step(a)
    return at, s.final_hash(), s.progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=6)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    args = Namespace(src=a.src, mods=None, actions="ordered-capped", enc=2, d=32, layers=1, heads=2, phase_split=True,
                     moe=False, offer_values=True, build_shaping=1.0, shop_ent=None, ent=0.01, reward="blinds",
                     shaping=1.0, gamma=0.995, lam=0.95, gamma_blind=0.95, lam_blind=0.95, back="b_red", stake=1,
                     backs=[], stakes=[], challenges=[], challenge_share=0.25, max_steps=1500, stop_ante=0,
                     explore_hands=0.0, pin_hand=None, pin_rate=1.0, seed=7, eval_temp=1.0, verify_wins=True,
                     curriculum_published=300)

    banned = T.held_out_seeds()
    pub = T.published_starts(args, args.curriculum_published, banned)
    bad = [e["seed"] for e in pub[4] if e["seed"] in banned]
    if bad:
        fail(f"published starts on held-out seeds: {bad[:5]}")
    print(f"{len(pub[4])} published starts per ante, none on the {len(banned)} held-out seeds: "
          f"{'PASS' if not bad else 'FAIL'}", flush=True)
    picks = [pub[4][i] for i in np.linspace(0, len(pub[4]) - 1, a.routes).astype(int)]

    # 1. published starts
    replay_t = {k: [] for k in T.CURRICULUM_ANTES}
    sim_t = []
    s = Sim(a.src, mode="ordered-capped", enc=2, **sim_kwargs(args))
    for e in picks:
        r = {"seed": e["seed"], "back": e["back"], "stake": e["stake"], "acts": e["acts"], "desc": e["desc"]}
        t0 = time.time()
        at, end_hash, end_prog = first_shop(a.src, r, T.CURRICULUM_ANTES)
        sim_t.append((time.time() - t0, len(r["acts"])))
        for k in T.CURRICULUM_ANTES:
            if k not in at:
                fail(f"{e['seed']}: never in a shop at ante {k}")
                continue
            s.reset(e["seed"], e["back"], e["stake"])
            t1 = time.time()
            o = s.fast_forward(e["acts"], e["desc"], k)
            replay_t[k].append(time.time() - t1)
            if o is None or s.steps != at[k]:
                fail(f"{e['seed']} ante {k}: start at move {s.steps}, the first shop of ante {k} is move {at[k]}")
                continue
            for j in range(at[k], len(e["acts"])):
                if s.H.action_desc(e["acts"][j] + 1) != e["desc"][j]:
                    fail(f"{e['seed']} ante {k}: move {j} differs after the start")
                    break
                _, _, done = s.step(e["acts"][j])
                if not done:
                    s.observe()
            else:
                if s.final_hash() != end_hash or not s.progress["won"]:
                    fail(f"{e['seed']} ante {k}: the win's end differs after the start")
    print(f"published starts stop at the first shop of ante 4/5/6 and play on to the same win: "
          f"{'PASS' if not FAILS else 'FAIL'} ({len(picks)} wins x 3 antes)", flush=True)

    # 2. through the trainer, and an own start from it
    sim = Sim(a.src, mode="ordered-capped", enc=2, **sim_kwargs(args))
    model = make_model(args, len(sim.vocab))
    model.eval()
    ok = 0
    for e in picks[:3]:
        k = 5
        ep = T.play_games(sim, model, T.REWARDS["blinds"], random.Random(0), args, 0,
                          (model.state_dict(), [dict(e, ante=k, until_ante=k)], "train", 0, 0), set())[0]
        c = ep["curriculum"]
        if not c or c["ante"] != k or c["from"] != "published":
            fail(f"{e['seed']}: not marked as a curriculum game: {c}")
            continue
        start = c["start_step"]
        if len(ep["steps"]) != ep["n"] - start or ep["acts"][:start] != e["acts"][:start]:
            fail(f"{e['seed']}: trained steps don't begin at the start (move {start})")
            continue
        if ep["shop_at"].get(k) != start:
            fail(f"{e['seed']}: shop_at {ep['shop_at']} doesn't point at the start's shop (move {start})")
            continue
        own = T.curriculum_entry("own", ep, k, ep["acts"][:start], ep["desc"][:start])
        # each in a fresh runtime: card ids (sort_id) keep counting across games in one runtime
        s1, s2 = (Sim(a.src, mode="ordered-capped", enc=2, **sim_kwargs(args)) for _ in range(2))
        for x in (s1, s2):
            x.env.lua.execute(diff_check.DUMP, x.H)
        s1.reset(e["seed"], e["back"], e["stake"])
        s1.fast_forward(e["acts"], e["desc"], k)
        d_pub = s1.H.state_dump()
        s2.reset(own["seed"], own["back"], own["stake"])
        if s2.fast_forward(own["acts"], own["desc"], None) is None:
            fail(f"{e['seed']}: the own start didn't replay")
            continue
        if s2.H.state_dump() != d_pub:
            x, y = diff_check.first_diff(d_pub, s2.H.state_dump())
            fail(f"{e['seed']}: own start differs from the published start: {x[:2]} vs {y[:2]}")
            continue
        ok += 1
    print(f"trainer: curriculum games marked, trained from the start, own starts reach the same state: "
          f"{'PASS' if ok == 3 else 'FAIL'}", flush=True)

    per_move = sum(t for t, _ in sim_t) / sum(n for _, n in sim_t)
    print("replay cost per start: " + "; ".join(f"ante {k} {np.mean(v):.2f} s" for k, v in replay_t.items() if v)
          + f" (the published games: {1000 * per_move:.1f} ms per move, {np.mean([n for _, n in sim_t]):.0f} moves per win)")
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")


if __name__ == "__main__":
    main()
