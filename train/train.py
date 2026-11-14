"""The trainer: PPO with parallel workers on the real game (docs/model_v1.md for the base design).

  python3 train.py --src ~/balatro/local/source --run <run> ...      (normally via train/launch.sh <run>)

W worker processes each own a sim and a copy of the model. Every iteration they play games with the
current weights; the learner runs PPO and sends the new weights back. Checkpoints, config and metrics go
to runs/<run>/. Nothing here prefers one outcome over another except --reward; the hand pins (expert runs)
limit which plays the bot may make and leave the reward as it is.

This file is main(): setup, the training loop, evals, checkpoints and stops. The parts, by capability:
  options.py     every command-line setting          common.py      seeds, batching, action codes
  rollout.py     workers playing games               ppo.py         advantages and the PPO update
  pool.py        the worker pool, safety exceptions  curriculum.py  v4's mid-run starts
  pins.py        hand pins (experts, MoE)            moe.py         mixture-of-experts loading
  explore.py     hand-type exploration (off since v4b)
  credit.py      routed credit (v4)  rewards.py  reward, shaping, discount clocks  model.py  the networks
"""
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import EXPERT_HANDS, make_model, param_count, warm_start_state  # noqa: E402
from rewards import REWARDS, build_shaping, shaping, step_discount  # noqa: E402,F401
# The trainer's parts, by capability. Imported here too so `import train` keeps exposing them (play.py,
# analysis/, tests/ and tools/ use train.<name>).
from common import (DISCARD, HAND_COL, PLAY, SEED_CHARS, collate, eval_seeds, game_config,  # noqa: E402,F401
                    rand_seed, same_config)
from curriculum import CURRICULUM_ANTES, curriculum_entry, held_out_seeds, published_starts  # noqa: E402,F401
from explore import explore_sample  # noqa: E402,F401
from moe import enc_compatible, load_moe, load_weights_into  # noqa: E402,F401
from pins import (HAND_RANK, apply_pin, count_pin_move, early_pin, moe_min_plays, moe_pin_rate,  # noqa: E402,F401
                  new_pin_stats, pin_summary, pin_vocab_id)
from pool import Pool, Stop, WorkerFailure, disk_free_gb, mem_available_gb  # noqa: E402,F401
from ppo import (discounted, gae, parse_schedule, per_step, phase_ent, ppo_update, schedule_at,  # noqa: E402,F401
                 taken_logp)
from options import build_parser  # noqa: E402
from rollout import inject_at, play_games, replay_games, verify_route, worker_main  # noqa: E402,F401


# ---------------- bookkeeping ----------------

def find_info(src):
    d = os.path.abspath(src)
    for _ in range(4):
        p = os.path.join(d, "info.txt")
        if os.path.exists(p):
            return open(p, encoding="utf-8", errors="replace").read()
        d = os.path.dirname(d)
    return ""


def provenance(args, vocab):
    info = find_info(args.src)
    exe = next((l.split(":", 1)[1].strip() for l in info.splitlines() if "sha256" in l), "unknown")
    try:
        commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {"config": vars(args), "commit": commit, "balatro_exe_sha256": exe,
            "modded": bool(args.mods), "python": platform.python_version(), "torch": torch.__version__,
            "vocab_sha256": hashlib.sha256("\n".join(vocab).encode()).hexdigest(), "host": platform.node()}


def summarize(episodes):
    n = len(episodes)
    antes = {}
    for e in episodes:
        a = e["progress"]["ante"]
        antes[a] = antes.get(a, 0) + 1
    return {"games": n,
            "wins": sum(e["progress"]["won"] for e in episodes),
            # near wins: reached the ante-8 boss and lost it; final_boss = wins + near wins
            "near_wins": sum(bool(e["progress"].get("near_win")) for e in episodes),
            "final_boss": sum(bool(e["progress"].get("final_boss")) for e in episodes),
            "reached": sum(e.get("reached", False) for e in episodes),
            "blinds_mean": float(np.mean([e["progress"]["blinds_beaten"] for e in episodes])),
            "bosses_mean": float(np.mean([e["progress"]["bosses_beaten"] for e in episodes])),
            "return_mean": float(np.mean([e["return"] for e in episodes])),
            "decisions_mean": float(np.mean([e["n"] for e in episodes])),
            # jokers sold per game, from the bot's own moves (v4's failure; v4b's --stop-sells)
            "sells_mean": float(np.mean([sum(d.startswith("sell_joker") for d in
                                             e["desc"][(e.get("curriculum") or {}).get("start_step", 0):])
                                         for e in episodes])),
            # v4c's measures (hypotheses-v4c.md): swaps and rerolls per game, from the bot's own moves
            "swaps_mean": float(np.mean([sum(d.startswith("swap ") for d in
                                             e["desc"][(e.get("curriculum") or {}).get("start_step", 0):])
                                         for e in episodes])),
            "rerolls_mean": float(np.mean([sum(d == "reroll" for d in
                                               e["desc"][(e.get("curriculum") or {}).get("start_step", 0):])
                                           for e in episodes])),
            "capped": sum(e["capped"] for e in episodes),
            "final_ante": dict(sorted(antes.items())),
            **pin_summary(episodes)}


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if str(device).startswith("cuda") and args.gpu_mem_fraction > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        except Exception as ex:
            print(f"could not cap GPU memory: {ex}", flush=True)
    out = args.out or os.path.join(HERE, "runs", args.run)
    os.makedirs(out, exist_ok=True)
    log_alerts = []

    def alert(msg):
        line = f"ALERT {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        log_alerts.append(line)
        del log_alerts[:-30]
        print(line, flush=True)

    # Size the pool to the memory actually free (resource allocation).
    avail = mem_available_gb()
    fit = int(max(1, (avail - args.min_mem_gb - 1.0) // args.worker_mem_gb))
    if fit < args.workers:
        alert(f"{avail:.1f} GB free fits {fit} workers, not {args.workers}; using {fit}")
        args.workers = fit
    if disk_free_gb(out) < args.min_disk_gb:
        raise SystemExit(f"only {disk_free_gb(out):.1f} GB free on disk; need {args.min_disk_gb}")

    stop_requested = []
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append("interrupted (Ctrl+C)"))
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append("terminated"))

    pool = Pool(args)
    pool.start()
    vocab = pool.vocab

    if args.moe:
        assert not (args.stop_ante or args.bc or args.sil_buffer or args.pin_hand or args.init_weights), \
            "--moe runs whole games with its own pins; --stop-ante, --bc, --sil-buffer, --pin-hand and --init-weights are off"
    assert args.lam_hand == 1.0 or args.phase_split, "--lam-hand is model v4b's: it needs --phase-split"
    assert not (args.curriculum or args.offer_values or args.build_shaping) or args.phase_split, \
        "--curriculum, --offer-values and --build-shaping are model v4's: they need --phase-split"
    if args.phase_split:
        assert not (args.moe or args.stop_ante or args.bc or args.sil_buffer or args.pin_hand or args.init_weights), \
            "--phase-split (v4) trains two networks from random weights on whole games; --moe, --stop-ante, --bc, " \
            "--sil-buffer, --pin-hand and --init-weights are off"
        parse_schedule(args.shop_ent or args.ent)
    model = make_model(args, len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_it, resumed = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        assert ck["vocab"] == vocab, "checkpoint vocabulary differs from this game"
        arch = ck.get("arch", {})
        for k, v in (("d", args.d), ("layers", args.layers), ("heads", args.heads), ("enc", args.enc), ("actions", args.actions), ("moe", bool(args.moe)), ("phase_split", bool(args.phase_split)),
                     ("offer_values", bool(args.offer_values))):
            have = arch.get(k, {"enc": 1, "actions": "subsets", "moe": False, "phase_split": False,
                                "offer_values": False}.get(k))
            if k == "actions" and (have, v) == ("ordered", "ordered-capped"):
                # Same model and moves; only long chains of joker drags stop being offered.
                alert("continuing an 'ordered' checkpoint with 'ordered-capped' actions (joker-drag cap on)")
                continue
            assert have == v, f"checkpoint was trained with {k}={have}, this run has {k}={v}"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_it = ck["iteration"] + 1
        if args.reset_lr:
            for g in opt.param_groups:
                g["lr"] = args.lr
        resumed = ck
    elif args.warm_start:
        # model v4c (docs/hypotheses-v4c.md): v4b's weights, a fresh optimizer, round 0, this run's own evals
        ck = torch.load(args.warm_start, map_location=device, weights_only=False)
        assert ck["vocab"] == vocab, "checkpoint vocabulary differs from this game"
        arch = ck.get("arch", {})
        for k, v in (("d", args.d), ("layers", args.layers), ("heads", args.heads), ("enc", args.enc),
                     ("phase_split", bool(args.phase_split)), ("offer_values", bool(args.offer_values))):
            assert arch.get(k) == v, f"warm-start checkpoint has {k}={arch.get(k)}, this run has {k}={v}"
        have = arch.get("actions")
        assert have == args.actions or (have, args.actions) == ("ordered-capped", "ordered-capped-swap"), \
            f"warm-start checkpoint has actions={have}, this run has actions={args.actions}"
        model.load_state_dict(warm_start_state(ck["model"], model.state_dict()))
        alert(f"warm start from {args.warm_start} (its round {ck.get('iteration')}, its best eval "
              f"{ck.get('best_eval', -1):.2f} blinds/game): fresh optimizer, round 0"
              + ("; swap rows start as buy's" if have != args.actions else ""))
    elif args.init_weights:
        w = torch.load(args.init_weights, map_location=device, weights_only=True)
        arch = w.get("arch", {})
        for k, v in (("d", args.d), ("layers", args.layers), ("heads", args.heads)):
            assert arch.get(k) == v, f"weights file has {k}={arch.get(k)}, this run has {k}={v}"
        assert enc_compatible(arch.get("enc"), args.enc), f"weights file has enc={arch.get('enc')}, this run has enc={args.enc}"
        have = arch.get("actions")
        assert have == args.actions or (have, args.actions) == ("ordered", "ordered-capped"), \
            f"weights file has actions={have}, this run has actions={args.actions}"
        load_weights_into(model, w["state_dict"], w["vocab"], vocab)
        print(f"starting from weights {os.path.basename(args.init_weights)} (model {w.get('model_version')}), "
              f"fresh optimizer, round 0", flush=True)
    elif args.moe:
        src = load_moe(model, args, vocab, device)
        print(f"MoE (shared trunk, {1 + len(EXPERT_HANDS)} heads): trunk from {src['trunk']}; expert heads from solo runs: "
              f"{src['experts_from_solo_runs'] or 'none'}; handoff at ante {args.moe_handoff}, "
              f"{args.moe_random:.0%} random handoffs in training, pins annealed to 0 over "
              f"{args.moe_anneal_games:,} games and not before each expert's {args.moe_min_games:,}, early head {'trained' if args.moe_train_early else 'frozen'}"
              f"{', its returns cut at the handoff' if args.moe_cut else ''}"
              f"{f'; early pin {args.moe_early_pin} (that hand or higher), fading with the run' if args.moe_early_pin else ''}"
              f"{f'; router minimum plays {args.moe_min_plays}' if args.moe_min_plays else ''}"
              f"; stall clock starts once every pin is 0", flush=True)
    prov = provenance(args, vocab)
    json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=2)
    print(f"model {param_count(model):,} params on {device}; {args.workers} workers; reward '{args.reward}' "
          f"(shaping {args.shaping}); {args.back} stake {args.stake}; {'modded' if args.mods else 'unmodded'}; "
          f"{avail:.1f} GB free; run dir {out}", flush=True)
    metrics = open(os.path.join(out, "metrics.jsonl"), "a")
    esd = eval_seeds(args.eval_games)
    nan_at = inject_at(args, "nan")

    def weights():
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def run_round(seeds, mode, it):
        for attempt in (1, 2):
            try:
                return pool.play(weights(), seeds, mode, it, args.round_timeout,
                                 lambda: stop_requested[0] if stop_requested else None)
            except WorkerFailure as e:
                alert(f"workers failed in round {it} (attempt {attempt}): {str(e)[-1500:]}")
                pool.stop()
                if attempt == 2:
                    raise Stop("workers failed twice in a row")
                try:
                    pool.start()
                except WorkerFailure as e2:
                    raise Stop(f"workers could not restart: {e2}")

    def save(it, tag):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it, "vocab": vocab,
              "provenance": prov, "arch": {"d": args.d, "layers": args.layers, "heads": args.heads, "enc": args.enc, "actions": args.actions, "moe": bool(args.moe),
                       "phase_split": bool(args.phase_split), "offer_values": bool(args.offer_values)},
              "best_eval": best_eval, "evals_since_best": evals_since_best, "games_total": games_total, "wins_total": wins_total,
              "decisions_total": decisions_total,
              "lr": opt.param_groups[0]["lr"], "sil_buffer": sil_buffer, "retry_pools": pools, "arms": arms,
              "beaten_seeds": sorted(beaten), "curriculum_own": {str(a): v for a, v in cur_own.items()}}
        d = out
        if tag.startswith("ckpt_") and spilling():
            d = os.path.join(args.spill_dir, "runs", args.run)
            os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{tag}.pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(d, f"{tag}.pt"))      # never leaves a half-written file

    health = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    drive_note = {}

    def usable(d):
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, f".write_test_{os.getpid()}")      # per process: two runs may check at once
            open(t, "w").write("ok"); os.remove(t)
            return True
        except OSError:
            return False

    def spilling():
        """True when the main drive is low and the spill drive is writable."""
        if not args.spill_dir or disk_free_gb(out) >= args.spill_at_gb:
            return False
        ok = usable(args.spill_dir)
        if drive_note.get("spill") != ok:
            drive_note["spill"] = ok
            alert(f"main drive under {args.spill_at_gb} GB free; " + (f"writing games and checkpoints to {args.spill_dir}"
                  if ok else f"spill drive {args.spill_dir} not writable, staying on the main drive"))
        return ok

    def mirror_win(line, name="wins.jsonl"):
        if not args.mirror_dir:
            return
        try:
            os.makedirs(args.mirror_dir, exist_ok=True)
            with open(os.path.join(args.mirror_dir, name), "a") as f:
                f.write(line)
            if drive_note.get("mirror") is False:
                alert(f"mirror {args.mirror_dir} writable again")
            drive_note["mirror"] = True
        except OSError as ex:
            if drive_note.get("mirror") is not False:
                alert(f"couldn't copy a win to {args.mirror_dir} ({ex}); the main copy is saved")
            drive_note["mirror"] = False

    def write_health(**kw):
        health.update(kw, alerts=log_alerts[-10:], mem_available_gb=round(mem_available_gb(), 2),
                      disk_free_gb=round(disk_free_gb(out), 1), lr=opt.param_groups[0]["lr"],
                      updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = os.path.join(out, ".health.json.tmp")
        json.dump(health, open(tmp, "w"), indent=2)
        os.replace(tmp, os.path.join(out, "health.json"))

    t_start = time.time()
    games_total, wins_total, strikes, it = 0, 0, 0, start_it
    decisions_total = 0         # training decisions (both networks): v4's shop entropy schedule
    evals_since_best = 0        # --stall-evals
    best_eval, best_time = -1.0, time.time()
    train_hist, best_roll = [], -1.0
    if resumed:
        best_eval = resumed.get("best_eval", -1.0)
        evals_since_best = resumed.get("evals_since_best", 0)
        games_total = resumed.get("games_total", 0)
        decisions_total = resumed.get("decisions_total", 0)
        wins_total = resumed.get("wins_total", 0)
        alert(f"resumed from {args.resume} at round {start_it} (best eval {best_eval:.2f} blinds/game, "
              f"{games_total:,} games and {wins_total:,} wins so far)")
    sil_buffer = list(resumed.get("sil_buffer", [])) if resumed else []
    if args.sil_buffer > 0 and args.sil_seed:
        have, added = {r["seed"] for r in sil_buffer}, 0
        for path in args.sil_seed:
            if not os.path.exists(path):
                continue
            for line in open(path):
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("mode") != "train" or not r.get("verified") or not same_config(r, args)
                        or r.get("actions", "subsets") != args.actions or r.get("enc", 1) != args.enc
                        or r["seed"] in have):
                    continue
                have.add(r["seed"])
                sil_buffer.append({"key": f"seed-{r['seed']}", "seed": r["seed"], "acts": r["acts"], "desc": r["desc"],
                                   "final": r["final"], "return": r["return"], "won": True, "it": r["iteration"]})
                added += 1
        sil_buffer.sort(key=lambda r: (r["return"], r["it"]), reverse=True)
        del sil_buffer[args.sil_buffer:]
        alert(f"self-imitation buffer filled with {added} saved wins ({sum(r['won'] for r in sil_buffer)} wins in a buffer of {len(sil_buffer)})")
    sil_rng = random.Random(f"sil-{args.seed}-{start_it}")
    games_dir = os.path.join(out, "games")
    if args.save_games != "none":
        os.makedirs(games_dir, exist_ok=True)
        with open(os.path.join(games_dir, "meta.jsonl"), "a") as f:   # what every saved game was played on
            f.write(json.dumps({"from_iteration": start_it, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "back": args.back, "stake": args.stake, "backs": args.backs, "stakes": args.stakes,
                                "challenges": args.challenges, "challenge_share": args.challenge_share,
                                "mode": args.actions, "enc": args.enc,
                                "profile": "unlocked", "fast": True, "provenance": prov}) + "\n")
    saved = {"games": 0, "wins": 0, "wins_verified": 0}
    if args.mirror_dir:
        alert(f"wins also copied to {args.mirror_dir}" if usable(args.mirror_dir)
              else f"mirror {args.mirror_dir} not writable now; wins stay on the main drive until it is")
    # retry pools: ante -> {seed: {"attempts", "best_blinds", "first_it"}}; arms: wins and games per arm
    ARMS = ["fresh"] + [str(a) for a in range(1, 9)]
    pools = {a: dict(v) for a, v in (resumed.get("retry_pools", {}) if resumed else {}).items()}
    for a in ARMS[1:]:
        pools.setdefault(a, {})
    arms = dict(resumed.get("arms", {})) if resumed else {}
    for a in ARMS:
        arms.setdefault(a, {"games": 0, "wins": 0})
    beaten = set(resumed.get("beaten_seeds", [])) if resumed else set()
    arm_rng = random.Random(f"arms-{args.seed}-{start_it}")
    # v4 curriculum: published starts (fixed at launch) and this run's own (the most recent --curriculum-keep per ante)
    cur_pub, cur_own = {a: [] for a in CURRICULUM_ANTES}, {a: [] for a in CURRICULUM_ANTES}
    cur_rng = random.Random(f"curriculum-{args.seed}-{start_it}")
    if args.curriculum > 0:
        cur_pub = published_starts(args, args.curriculum_published, held_out_seeds())
        for a, v in ((resumed or {}).get("curriculum_own") or {}).items():
            cur_own[int(a)] = list(v)
        print(f"curriculum: {args.curriculum:.0%} of training games start at the first shop of ante "
              f"{'/'.join(map(str, CURRICULUM_ANTES))}, from {len(cur_pub[CURRICULUM_ANTES[0]])} published wins and "
              f"this run's games ({sum(len(v) for v in cur_own.values())} kept so far)", flush=True)

    def add_curriculum(seeds):
        """Puts --curriculum of the round's games on curriculum starts, in fresh-game slots: an ante drawn
        uniformly among antes with starts, then a start drawn uniformly from that ante's."""
        if args.curriculum <= 0:
            return seeds
        n = sum(len(w) for w in seeds)
        want = int(round(n * args.curriculum))
        slots = [(w, i) for i in range(max(len(x) for x in seeds)) for w in range(len(seeds))
                 if i < len(seeds[w]) and seeds[w][i] is None]
        for w, i in slots[:want]:
            antes = [a for a in CURRICULUM_ANTES if cur_pub[a] or cur_own[a]]
            if not antes:
                break
            a = cur_rng.choice(antes)
            pool = cur_pub[a] + cur_own[a]
            seeds[w][i] = pool[cur_rng.randrange(len(pool))]
        return seeds

    def harvest_starts(eps):
        """This run's games as curriculum starts: each game's moves up to the first shop of each ante."""
        if args.curriculum <= 0:
            return
        for e in eps:
            if e.get("curriculum"):
                continue
            for a, i in (e.get("shop_at") or {}).items():
                cur_own[a].append(curriculum_entry("own", e, a, e["acts"][:i], e["desc"][:i]))
                del cur_own[a][:-args.curriculum_keep]

    def train_seeds():
        """Splits this round's games across the arms. Fresh keeps at least --fresh-min; every other
        slot goes to the arm with the highest draw from Beta(1 + wins, prior + losses), among arms
        with seeds waiting. Within an arm, seeds with the fewest attempts go first."""
        n = args.workers * args.games_per_worker
        if not args.retry:
            return [[None] * args.games_per_worker for _ in range(args.workers)], {}
        floor = int(math.ceil(n * args.fresh_min))
        left = {a: len(pools[a]) for a in ARMS[1:]}
        alloc = {a: 0 for a in ARMS}
        alloc["fresh"] = floor
        for _ in range(n - floor):
            best, bv = "fresh", -1.0
            for a in ARMS:
                if a != "fresh" and alloc[a] >= left[a]:
                    continue
                st = arms[a]
                v = arm_rng.betavariate(1 + st["wins"], args.arm_prior + st["games"] - st["wins"])
                if v > bv:
                    best, bv = a, v
            alloc[best] += 1
        source = {}
        flat = []
        for a in ARMS[1:]:
            order = sorted(pools[a].items(), key=lambda kv: (kv[1]["attempts"], -kv[1]["best_blinds"]))
            for sd, _ in order[:alloc[a]]:
                flat.append(sd)
                source[sd] = a
        flat += [None] * (n - len(flat))
        return [flat[w::args.workers] for w in range(args.workers)], source

    def update_retries(it, eps, source):
        for e in eps:
            sd, p = e["seed"], e["progress"]
            arm = source.get(sd, "fresh")
            e["arm"] = arm
            arms[arm]["games"] += 1
            arms[arm]["wins"] += int(p["won"])
            if arm != "fresh":
                pools[arm].pop(sd, None)
            if p["won"]:
                beaten.add(sd)
                continue
            if sd in beaten:
                continue
            prev = None if arm == "fresh" else e.get("_prev_entry")
            attempts = (prev or {}).get("attempts", 0) + (0 if arm == "fresh" else 1)
            if arm != "fresh" and attempts >= args.retry_max:
                continue
            # best ante reached so far; a Hieroglyph voucher can end a game at ante 0, kept with ante 1
            a = str(max(int(arm) if arm != "fresh" else 1, min(max(p["ante"], 1), 8)))
            pools[a][sd] = {"attempts": attempts, "best_blinds": max(p["blinds_beaten"], (prev or {}).get("best_blinds", 0)),
                            "first_it": (prev or {}).get("first_it", it)}
            if len(pools[a]) > args.retry_pool:     # full: drop the most-tried, then the oldest
                worst = max(pools[a].items(), key=lambda kv: (kv[1]["attempts"], -kv[1]["first_it"]))[0]
                del pools[a][worst]

    def record(it, mode, eps):
        """Writes each game exactly. Wins also go to wins.jsonl, flagged if they didn't replay."""
        if args.save_games == "none":
            return
        keep = [e for e in eps if args.save_games == "all" or e["progress"]["won"] or e["progress"].get("near_win")]
        recs = [{"iteration": it, "mode": mode, "seed": e["seed"], "back": e["back"], "stake": e["stake"],
                 **({"challenge": e["challenge"]} if e.get("challenge") else {}),
                 **({"jokers_held": e["jokers_held"]} if e.get("jokers_held") is not None else {}),
                 "found_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "n": e["n"], "capped": e["capped"], "return": e["return"], "progress": e["progress"],
                 "verified": e["verified"], "final": e["final"], "acts": e["acts"], "desc": e["desc"],
                 "retry": e.get("retry", False), "arm": e.get("arm"), "actions": args.actions, "enc": args.enc,
                 **({"curriculum": e["curriculum"]} if e.get("curriculum") else {}),
                 **({"moe": e.get("moe")} if args.moe else {}),
                 **({"stop_ante": args.stop_ante, "reached": e.get("reached", False)} if args.stop_ante else {}),
                 **({"pin_hand": args.pin_hand, "pin_rate": args.pin_rate, "pin": e.get("pin")} if args.pin_hand else {})}
                for e in keep]
        if recs:
            gd = os.path.join(args.spill_dir, "runs", args.run, "games") if spilling() else games_dir
            os.makedirs(gd, exist_ok=True)
            path = os.path.join(gd, f"r{it:05d}_{mode}.jsonl.gz")
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, path)
            saved["games"] += len(recs)
        for r in recs:
            if r.get("curriculum"):
                # a curriculum game began on another game's moves: never a win of this run's (v4 registration);
                # its wins are kept apart, replayed from the seed like any other
                if r["progress"]["won"]:
                    line = json.dumps(r | {"commit": prov["commit"][:12], "win_check": 2}) + "\n"
                    with open(os.path.join(out, "curriculum_wins.jsonl"), "a") as f:
                        f.write(line)
                    mirror_win(line, "curriculum_wins.jsonl")
                    saved["curriculum_wins"] = saved.get("curriculum_wins", 0) + 1
                continue
            if r["progress"]["won"]:
                # win_check 2: won by the game's own test (G.GAME.won and not GAME_OVER, sim.py);
                # wins saved before 2026-11-03 lack it and are replayed again before publishing.
                line = json.dumps(r | {"commit": prov["commit"][:12], "win_check": 2}) + "\n"
                with open(os.path.join(out, "wins.jsonl"), "a") as f:
                    f.write(line)
                mirror_win(line)
                saved["wins"] += 1
                if r["verified"]:
                    saved["wins_verified"] += 1
                else:
                    alert(f"won game (seed {r['seed']}, round {it}) did NOT replay exactly; saved, flagged unverified")
            elif r["progress"].get("near_win"):
                # Near wins (reached the ante-8 boss, lost it) are kept like wins, in near_wins.jsonl.
                with open(os.path.join(out, "near_wins.jsonl"), "a") as f:
                    f.write(json.dumps(r | {"commit": prov["commit"][:12], "win_check": 2}) + "\n")
                saved["near_wins"] = saved.get("near_wins", 0) + 1

    def add_to_buffer(it, eps):
        """Keeps the highest-return training games (return under the reward of record), as routes."""
        if args.sil_buffer <= 0:
            return
        for k, e in enumerate(eps):
            if e["capped"]:
                continue
            sil_buffer.append({"key": f"{it}-{k}", "seed": e["seed"], "acts": e["acts"], "desc": e["desc"],
                               "final": e["final"], "return": e["return"], "won": e["progress"]["won"], "it": it})
        sil_buffer.sort(key=lambda r: (r["return"], r["it"]), reverse=True)
        del sil_buffer[args.sil_buffer:]

    def sil_data(it):
        if args.sil_buffer <= 0 or not sil_buffer:
            return None, {}
        pick = [sil_buffer[sil_rng.randrange(len(sil_buffer))] for _ in range(args.sil_games)]
        split = [pick[w::args.workers] for w in range(args.workers)]
        got = run_round(split, "replay", it)
        obs, act, ret, bad = [], [], [], set()
        for g in got:
            if g["problem"]:
                bad.add(g["key"])
                alert(f"saved game {g['key']} no longer replays ({g['problem']}); dropped from the buffer")
                continue
            for (o, a, _), R in zip(g["steps"], discounted([x[2] for x in g["steps"]], g.get("gammas") or args.gamma)):
                obs.append(o); act.append(a); ret.append(R)
        if bad:
            sil_buffer[:] = [r for r in sil_buffer if r["key"] not in bad]
        if not obs:
            return None, {}
        info = {"buffer": len(sil_buffer), "buffer_wins": sum(r["won"] for r in sil_buffer),
                "buffer_min_return": min(r["return"] for r in sil_buffer),
                "replayed_games": len(got) - len(bad), "replayed_steps": len(obs)}
        return (obs, np.array(act), np.array(ret, np.float32)), info

    rates = []                     # decisions per second of play, for the throttling flag
    reason = "finished"

    def bc_phase():
        """Behavior cloning: replays the --bc games into exact training data (observation, the move
        that was made, the reward) and trains the policy to make that move (cross-entropy) and the
        critic to predict the game's actual discounted return. Runs before RL on a fresh model."""
        routes = []
        for path in args.bc:
            for line in open(path):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # a file still being written can end mid-line
                if (r.get("actions", "subsets") != args.actions or r.get("enc", 1) != args.enc
                        or not same_config(r, args)):
                    continue
                routes.append({"key": f"bc-{len(routes)}", "seed": r["seed"], "acts": r["acts"], "desc": r["desc"],
                               "final": r["final"]})
        if not routes:
            alert("behavior cloning: no usable games in --bc; skipped")
            return
        chunk = args.bc_chunk or 4 * args.workers
        total = int(len(routes) * args.bc_epochs)
        rng = random.Random(f"bc-{args.seed}")
        order = []
        while len(order) < total:
            o = list(range(len(routes)))
            rng.shuffle(o)
            order += o
        order = order[:total]
        alert(f"behavior cloning: {len(routes):,} games x {args.bc_epochs} epochs = {total:,} game replays")
        model.train()
        done, t0, stats = 0, time.time(), {"loss": 0.0, "acc": 0.0, "v": 0.0, "n": 0}
        for c in range(0, total, chunk):
            if stop_requested:
                raise Stop(stop_requested[0])
            pick = [routes[i] for i in order[c:c + chunk]]
            got = run_round([pick[w::args.workers] for w in range(args.workers)], "replay", -1)
            obs, act, ret = [], [], []
            for g in got:
                if g["problem"]:
                    alert(f"behavior cloning: game {g['key']} didn't replay ({g['problem']}); skipped")
                    continue
                for (o, a, _), R in zip(g["steps"], discounted([x[2] for x in g["steps"]], g.get("gammas") or args.gamma)):
                    obs.append(o); act.append(a); ret.append(R)
            idx = np.argsort([len(o["act"]) for o in obs])
            batches = [idx[i:i + args.minibatch] for i in range(0, len(idx), args.minibatch)]
            rng.shuffle(batches)
            for b in batches:
                batch = collate([obs[i] for i in b], device)
                logits, value = model(*batch)
                a = torch.tensor([act[i] for i in b], device=device)
                R = torch.tensor([ret[i] for i in b], device=device)
                ce = torch.nn.functional.cross_entropy(logits, a)
                v_loss = 0.5 * (value - R).pow(2).mean()
                loss = ce + args.vf * v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                with torch.no_grad():
                    stats["loss"] += ce.item(); stats["v"] += v_loss.item(); stats["n"] += 1
                    stats["acc"] += (logits.argmax(-1) == a).float().mean().item()
            done += len(pick)
            if stats["n"] and (c // chunk) % 10 == 0:
                n = stats["n"]
                line = (f"[bc] {done:,}/{total:,} games  move loss {stats['loss'] / n:.3f}  "
                        f"same move as saved {stats['acc'] / n:.1%}  value loss {stats['v'] / n:.2f}  "
                        f"{(time.time() - t0) / 60:.1f} min")
                print(line, flush=True)
                metrics.write(json.dumps({"bc": {"games": done, "ce": stats["loss"] / n, "acc": stats["acc"] / n,
                                                 "v_loss": stats["v"] / n}}) + "\n")
                metrics.flush()
                write_health(status="running: behavior cloning", bc_games=done, bc_total=total)
                stats = {"loss": 0.0, "acc": 0.0, "v": 0.0, "n": 0}
        model.eval()
        save(-1, "bc")
        alert(f"behavior cloning done: {total:,} game replays in {(time.time() - t0) / 60:.1f} min; saved bc.pt")
    try:
        if args.bc and resumed is None:
            bc_phase()
        pending = None
        save_at = sorted(int(x) for x in str(args.save_at_games).replace(" ", "").split(",") if x)
        # resumed runs: what's already on disk is done
        save_at = [g for g in save_at if not os.path.exists(os.path.join(out, f"games_{g}.pt"))]
        force_eval = False
        pins0_done = os.path.exists(os.path.join(out, "pins0.pt")) or not (args.moe and args.eval_at_pins_zero)
        pins0_evaled = os.path.exists(os.path.join(out, "pins0.pt"))
        for it in range(start_it, args.iterations):
            if stop_requested:
                raise Stop(stop_requested[0])
            hours = (time.time() - t_start) / 3600
            if hours > args.hours:
                raise Stop(f"time limit ({args.hours} h)")
            if args.max_games and games_total >= args.max_games:
                raise Stop(f"reached {games_total:,} training games (limit {args.max_games:,})")
            if args.max_wins and wins_total >= args.max_wins:
                raise Stop(f"reached {wins_total:,} won games (limit {args.max_wins:,})")
            if mem_available_gb() < args.min_mem_gb:
                raise Stop(f"free memory below {args.min_mem_gb} GB ({mem_available_gb():.2f} GB)")
            if disk_free_gb(out) < args.min_disk_gb:
                raise Stop(f"free disk below {args.min_disk_gb} GB")

            if it % args.eval_every == 0 or force_eval:
                te = time.time()
                split = [esd[w::args.workers] for w in range(args.workers)]
                eval_eps = run_round(split, "eval", it)
                ev = summarize(eval_eps)
                record(it, "eval", eval_eps)
                beaten.update(e["seed"] for e in eval_eps if e["progress"]["won"])
                print(f"[eval {it}] {json.dumps(ev)}", flush=True)
                metrics.write(json.dumps({"iteration": it, "eval": ev, "eval_seconds": round(time.time() - te, 1)}) + "\n")
                metrics.flush()
                save(it, f"ckpt_{it:05d}")
                if force_eval:
                    save(it, "pins0")
                    alert(f"every MoE pin is 0 (at {games_total:,} games): eval {it} is the first without pins, "
                          f"{ev['blinds_mean']:.3f} blinds/game, {ev['wins']} wins; weights in pins0.pt")
                    force_eval = False
                    pins0_evaled = True
                score = (ev["reached"] / max(ev["games"], 1) if args.best_by == "reached" else
                         ev["wins"] / max(ev["games"], 1) + ev["blinds_mean"] / 1e4 if args.best_by == "wins" else
                         ev["blinds_mean"])
                if score > best_eval:
                    best_eval = score
                    evals_since_best = 0
                    save(it, "best")
                else:
                    evals_since_best += 1
                if args.stop_sells and ev["sells_mean"] > args.stop_sells:
                    raise Stop(f"eval {it} sold {ev['sells_mean']:.2f} jokers per game (--stop-sells {args.stop_sells}: "
                               f"v4's failure is back)")
                if args.stall_evals and evals_since_best >= args.stall_evals:
                    raise Stop(f"no new best eval in {evals_since_best} evals (best {best_eval:.3f}; --stall-evals {args.stall_evals})")
                capped = [e for e in eval_eps if e["capped"]][:20]
                if capped:
                    with open(os.path.join(out, "eval_capped.jsonl"), "a") as f:
                        for e in capped:
                            f.write(json.dumps({"iteration": it, "seed": e["seed"], "progress": e["progress"],
                                                "last_moves": e["tail"]}) + "\n")
                health["last_eval"] = ev
                health["best_eval_blinds"] = best_eval

            t0 = time.time()
            if pending is None:
                seeds, source = train_seeds()
                seeds = add_curriculum(seeds)
                prev_entries = {sd: dict(pools[a][sd]) for sd, a in source.items()}
                eps = run_round(seeds, "train", it)
            else:
                # sent last round, before the update: played with weights one update old
                ticket, seeds, source, prev_entries = pending
                pending = None
                try:
                    eps = pool.collect(ticket, args.round_timeout, lambda: stop_requested[0] if stop_requested else None)
                except WorkerFailure as e:
                    alert(f"workers failed in overlapped round {it}: {str(e)[-1500:]}; restarting and replaying the round")
                    pool.stop()
                    try:
                        pool.start()
                    except WorkerFailure as e2:
                        raise Stop(f"workers could not restart: {e2}")
                    eps = run_round(seeds, "train", it)
            t1 = time.time()
            for e in eps:
                e["retry"] = not e.get("curriculum") and e["seed"] in source
                e["_prev_entry"] = None if e.get("curriculum") else prev_entries.get(e["seed"])
                if e.get("curriculum"):
                    e["arm"] = "curriculum"
            cur_eps = [e for e in eps if e.get("curriculum")]
            # curriculum games train but are in no stat, arm, retry pool or tally (v4 registration)
            update_retries(it, [e for e in eps if not e.get("curriculum")], source)
            record(it, "train", eps)
            harvest_starts(eps)
            if args.moe:
                # Router base rate: the early model's plays of each expert's hand in antes 1-3, averaged
                # over games that reached the handoff (moving average). Pins anneal by games played.
                hand = [e["moe"]["early_counts"] for e in eps if e["moe"]["expert"]]
                with torch.no_grad():
                    if hand:
                        m = torch.tensor(np.mean(hand, 0), dtype=torch.float32, device=model.base_rate.device)
                        model.base_rate.copy_(m if float(model.base_rate.sum()) == 0 else 0.95 * model.base_rate + 0.05 * m)
                    for e in eps:
                        if e["moe"]["expert"]:
                            model.expert_games[EXPERT_HANDS.index(e["moe"]["expert"])] += 1
                    # Each expert's pin fades with the run's games, but not before the expert has had
                    # its own minimum of games (--moe-min-games; chud: a minimum amount of training).
                    run_frac = min(1.0, (games_total + len(eps)) / max(args.moe_anneal_games, 1))
                    own = (model.expert_games / max(args.moe_min_games, 1)).clamp(max=1.0)
                    model.pin_scale.copy_(1.0 - torch.minimum(own, torch.full_like(own, run_frac)))
                    model.early_pin_scale.fill_(1.0 - run_frac)
            sil, sil_info = sil_data(it)
            add_to_buffer(it, eps)       # after sampling: this round's games are already in the PPO batch
            if args.overlap and not stop_requested:
                # Send the next round now, so the workers play while the learner updates.
                nseeds, nsource = train_seeds()
                nseeds = add_curriculum(nseeds)
                nprev = {sd: dict(pools[a][sd]) for sd, a in nsource.items()}
                pending = (pool.send(weights(), nseeds, "train", it + 1), nseeds, nsource, nprev)

            # Update guard: keep a copy, reject the update if it is non-finite or moves too far.
            snap_m = {k: v.detach().clone() for k, v in model.state_dict().items()}
            snap_o = copy.deepcopy(opt.state_dict())
            args.decisions_done = decisions_total     # the shop's entropy schedule (v4) counts training decisions
            st = ppo_update(model, opt, eps, args, device, sil)
            decisions_total += sum(len(e["steps"]) for e in eps)
            if it in nan_at:
                with torch.no_grad():
                    next(model.parameters()).fill_(float("nan"))
            finite = all(torch.isfinite(p).all() for p in model.parameters()) and \
                all(math.isfinite(v) for v in st.values())
            moved = st["kl_update"] if args.target_kl else abs(st["kl"])   # this update's own change
            if not finite or moved > args.max_kl:
                model.load_state_dict(snap_m)
                opt.load_state_dict(snap_o)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                strikes += 1
                why = "non-finite loss or weights" if not finite else f"kl {moved:.3f} > {args.max_kl}"
                alert(f"update {it} rejected ({why}); "
                      f"reverted, learning rate now {opt.param_groups[0]['lr']:.2e} (strike {strikes}/{args.max_strikes})")
                if strikes >= args.max_strikes:
                    raise Stop(f"{strikes} rejected updates in a row")
            else:
                strikes = 0
            gpu = {}
            if str(device).startswith("cuda"):
                # GPU memory around the update. Near the card's size, WSL's driver can spill to shared
                # system memory and updates slow ~10x (moe2, 2026-11-04: 46 s -> 3-9 min per update at
                # 5.9 of 6.1 GB). Releasing the allocator's unused cache each round keeps headroom.
                try:
                    tot = torch.cuda.get_device_properties(0).total_memory
                    gpu = {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                           "total_gb": round(tot / 2**30, 2)}
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    gpu["reserved_after_gb"] = round(torch.cuda.memory_reserved() / 2**30, 2)
                except Exception as ex:           # bookkeeping only; never stop training for it
                    gpu = {"error": str(ex)[:100]}
            t2 = time.time()

            games_total += len(eps)
            while save_at and games_total >= save_at[0]:
                g = save_at.pop(0)
                save(it, f"games_{g}")
                alert(f"saved games_{g}.pt at round {it}, {games_total:,} training games")
            if not pins0_done and float(torch.cat([model.pin_scale, model.early_pin_scale]).max()) == 0:
                pins0_done = force_eval = True          # the next round starts with the eval
            stop_pins0 = args.moe and args.stop_after_pins0 and pins0_evaled and not save_at
            fresh = [e for e in eps if not e["retry"] and not e.get("curriculum")]
            s = summarize(fresh or [e for e in eps if not e.get("curriculum")] or eps)   # the learning curve stays on fresh seeds only
            sr = summarize([e for e in eps if e["retry"]]) if source else None
            round_alloc = {a: sum(1 for e in eps if e["arm"] == a) for a in ARMS}
            wins_total += sum(e["progress"]["won"] for e in eps if not e.get("curriculum"))
            cur_rec = None
            if args.curriculum > 0:
                # the curriculum's cost: replaying to each start, against all of this round's play time
                t_all = sum(e.get("t_model", 0) + e.get("t_sim", 0) + e.get("t_replay", 0) for e in eps)
                t_rep = sum(e.get("t_replay", 0) for e in eps)
                cur_rec = {"games": len(cur_eps), "wins": sum(e["progress"]["won"] for e in cur_eps),
                           "by_ante": {str(a): sum(e["curriculum"]["ante"] == a for e in cur_eps) for a in CURRICULUM_ANTES},
                           "replay_seconds": round(t_rep, 2), "replay_share_of_play": round(t_rep / max(t_all, 1e-9), 4),
                           "own_starts": {str(a): len(v) for a, v in cur_own.items()}}
            train_hist.append(s["blinds_mean"])
            roll = float(np.mean(train_hist[-10:]))
            if args.moe and float(torch.cat([model.pin_scale, model.early_pin_scale]).max()) > 0:
                best_time = time.time()     # the stall clock starts once every MoE pin has faded to 0
            if len(train_hist) >= 10 and roll > best_roll + 0.05:
                best_roll, best_time = roll, time.time()
            elif not args.stall_evals and len(train_hist) >= 10 and (time.time() - best_time) / 3600 > args.stall_hours:
                raise Stop(f"training score flat for {args.stall_hours} h (best 10-round average "
                           f"{best_roll:.2f} blinds/game)")
            decisions = sum(e["n"] for e in eps)
            ms_model = 1000 * sum(e.get("t_model", 0) for e in eps) / max(decisions, 1)   # per decision, in a worker
            ms_sim = 1000 * sum(e.get("t_sim", 0) for e in eps) / max(decisions, 1)
            # Decisions per second of the whole round (wait for games + update). With overlap the workers
            # play during the previous update, so the wait alone can be ~0 s and gave rates like 77,000/s.
            rate = decisions / max(t2 - t0, 1e-6)
            rates.append((time.time() - t_start, rate, ms_sim, t2 - t1))
            early = [r for r in rates if r[0] < 1800]
            if (time.time() - t_start) > 3600 and early:
                med = lambda k: sorted(r[k] for r in early)[len(early) // 2]
                if ms_sim > 1.6 * med(2):
                    alert(f"sim time {ms_sim:.1f} ms per decision is over 1.6x the first half hour's "
                          f"({med(2):.1f} ms); likely CPU heat throttling")
                if t2 - t1 > 3 * med(3) and t2 - t1 > 60:
                    alert(f"model update took {t2 - t1:.0f} s, over 3x the first half hour's ({med(3):.0f} s); "
                          f"GPU memory peak {gpu.get('peak_reserved_gb', '?')} of {gpu.get('total_gb', '?')} GB reserved")
            if st["entropy"] < 0.05:
                alert(f"policy entropy {st['entropy']:.3f}: the policy has become nearly deterministic")
            rec = {**({"moe_pin_scale": dict(zip(EXPERT_HANDS, [round(float(x), 3) for x in model.pin_scale])),
                       "moe_early_pin_scale": round(float(model.early_pin_scale[0]), 3),
                      "moe_expert_games": dict(zip(EXPERT_HANDS, [int(x) for x in model.expert_games]))} if args.moe else {}),
                   "iteration": it, "train": s, "ppo": st, "wins_total": wins_total, "play_seconds": round(t1 - t0, 1),
                   "update_seconds": round(t2 - t1, 1), "gpu_memory": gpu, "games_per_hour": round(len(eps) / (t2 - t0) * 3600),
                   "decisions_per_second": round(rate, 1), "games_total": games_total, "decisions_total": decisions_total,
                   "hours": round((time.time() - t_start) / 3600, 3), "sil": sil_info, "saved": dict(saved), "ms_model": round(ms_model, 2), "ms_sim": round(ms_sim, 2),
                   "retry": sr, "alloc": round_alloc, "arms": {a: dict(v) for a, v in arms.items()},
                   "pools": {a: len(v) for a, v in pools.items()}, "beaten_seeds": len(beaten),
                   **({"curriculum": cur_rec} if cur_rec else {})}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            ent_txt = (f" (table {st['entropy_table']:.2f}, shop {st['entropy_shop']:.2f} at coef {st['ent_coef_shop']:.4f})"
                       if "entropy_shop" in st else "")
            print(f"[it {it}] blinds {s['blinds_mean']:.2f} wins {s['wins']}/{s['games']} (total {wins_total:,}) "
                  f"near wins {s['near_wins']} (reached the final boss {s['final_boss']}) ante {s['final_ante']} "
                  f"ent {st['entropy']:.2f}{ent_txt} kl {st['kl']:.4f} moved {st.get('kl_update', 0):.4f}{' (stopped early at ' + str(st['minibatches']) + ' minibatches)' if st.get('early_stop') else ''} | {rec['games_per_hour']} games/h "
                  f"{rate:.0f} dec/s (play {rec['play_seconds']}s, update {rec['update_seconds']}s; per decision: model {ms_model:.1f} ms, sim {ms_sim:.1f} ms)"
                  + (f" | sil buffer {sil_info['buffer']} ({sil_info['buffer_wins']} wins, min return {sil_info['buffer_min_return']:.0f}) "
                     f"pos {st.get('sil_pos', 0):.2f}" if sil_info else "")
                  + (f" | retry {sr['games']} games ante {sr['final_ante']} wins {sr['wins']}" if sr else "")
                  + " | alloc " + " ".join(f"{a[0] if a == 'fresh' else a}:{round_alloc[a]}" for a in ARMS if round_alloc[a])
                  + f" | waiting {sum(len(v) for v in pools.values())} beaten seeds {len(beaten)}"
                  + (f" | curriculum {cur_rec['games']} games (replay {cur_rec['replay_share_of_play']:.0%} of play)" if cur_rec else "")
                  + f" | saved {saved['games']:,} games, {saved['wins_verified']}/{saved['wins']} wins verified", flush=True)
            save(it, "latest")
            write_health(iteration=it, hours=rec["hours"], games_total=games_total, wins_total=wins_total, last_train=s,
                         decisions_per_second=rec["decisions_per_second"], saved=dict(saved), sil=sil_info,
                         last_retry=sr, alloc=round_alloc, arms={a: dict(v) for a, v in arms.items()},
                         pools={a: len(v) for a, v in pools.items()}, beaten_seeds=len(beaten))
            if stop_pins0:
                raise Stop(f"every pin at 0: the pins-zero eval is done (pins0.pt) and the checkpoints are saved "
                           f"(--stop-after-pins0), at {games_total:,} games")
    except Stop as e:
        reason = str(e)
    except Exception:
        reason = "crashed: " + traceback.format_exc()[-1500:]
    finally:
        try:
            save(it, "final")
        except Exception as e:
            reason += f" (final save failed: {e})"
        alert(f"stopped: {reason}")
        write_health(status=f"stopped: {reason}", iteration=it)
        pool.stop()
        print("done", flush=True)


if __name__ == "__main__":
    main()
