"""Paired expert comparison for the mixture of experts (docs/hypotheses.md: I2's test, the Full House
completion criterion, section J). Three questions (chud, 2026-11-05): does one strategy win the most;
does the handoff change play at all; is the handoff at the right ante.

The router gives each expert different games, so an expert's own record mixes the expert with the
games it was handed. Here every expert plays from the same positions: for each seed, the early head
plays to the ante-4 handoff exactly as in an eval (fixed sampling RNG per seed), then the game is
played out once with each of the 6 experts forced (train.py `moe_force`). With --rollouts R > 1,
each expert plays R times from that state; rollout r uses its own sampling after the handoff, shared
by every expert (common random numbers), so the moves up to the handoff are identical in every game
of a seed. That is checked: a game whose moves before the handoff differ from the seed's first
game is marked prefix_ok false.

- **Control:** "no handoff" lets the early head play the whole game from the same state. If the
  experts don't beat it, the handoff isn't adding anything.
- **Strategy:** each game records its plays after the handoff by hand type, so the summary shows
  whether the experts actually play differently from the same positions.
- **Handoff ante:** --handoff-antes 3,4,5 repeats everything with the handoff at each ante (moe2 trained
  at 4). The router's own choice at each ante is scored too.

  python3 paired_experts.py WEIGHTS.pt --src ~/balatro/local/source --seeds 300 --rollouts 2 \\
      --workers 4 --out paired.jsonl

WEIGHTS is a trainer checkpoint (runs/<run>/pins0.pt, best.pt, ckpt_*.pt): it carries the run's
settings, so the early head, router and pins play as they did in that run. Seeds are drawn from
--seed-tag, never the training eval's 600 (those are excluded). Output: one line per seed; a summary
is printed at the end (and with --summary FILE, for an existing output, without playing).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))

G = {}


def init(weights, src, max_steps):
    import torch
    torch.set_num_threads(1)
    from argparse import Namespace
    from model import MoE
    from sim import Sim
    import train as T
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    cfg.update(src=src, verify_wins=False, save_games="none")
    if max_steps:
        cfg["max_steps"] = max_steps
    args = Namespace(**cfg)
    if not args.moe:
        raise SystemExit("not a mixture-of-experts checkpoint")
    sim = Sim(args.src, args.mods, mode=args.actions, enc=args.enc)
    if list(sim.vocab) != list(ck["vocab"]):
        raise SystemExit("checkpoint vocabulary differs from this game's")
    model = MoE(len(sim.vocab), d=args.d, layers=args.layers, heads=args.heads, enc=args.enc)
    model.load_state_dict(ck["model"])
    model.eval()
    G.update(T=T, args=args, sim=sim, model=model, state=model.state_dict(),
             reward=T.REWARDS[args.reward], it=ck.get("iteration", 0),
             pins=[round(float(x), 3) for x in model.pin_scale] + [round(float(model.early_pin_scale[0]), 3)])


def play(seed, expert, rollout, ante):
    T, a = G["T"], G["args"]
    a.moe_force, a.moe_rollout, a.moe_handoff = expert, rollout, ante
    ep = T.play_games(G["sim"], G["model"], G["reward"], random.Random(0), a, 0,
                      (G["state"], [seed], "eval", G["it"], 0), set())[0]
    m = ep["moe"] or {}
    return {"won": bool(ep["progress"]["won"]), "near_win": bool(ep["progress"].get("near_win")),
            "blinds": ep["progress"]["blinds_beaten"], "ante": ep["progress"]["ante"],
            "return": ep["return"], "capped": ep["capped"], "handoff_step": m.get("handoff_step"),
            "rule_choice": m.get("rule_choice"), "late_counts": m.get("late_counts"), "late_plays": m.get("late_plays"),
            "prefix": ep["acts"][:m["handoff_step"]] if m.get("handoff_step") else None}


NO = "no handoff"
KEEP = ("won", "near_win", "blinds", "ante", "return", "capped", "late_counts", "late_plays")


def one_seed(job):
    seed, rollouts, antes = job
    hands = G["T"].EXPERT_HANDS
    t0 = time.time()
    rec = {"seed": seed, "pins": G["pins"], "iteration": G["it"], "by_ante": {}}
    for ante in antes:
        first = play(seed, -1, 0, ante)             # the control: also says whether the handoff is reached
        if first["handoff_step"] is None:
            rec["by_ante"][str(ante)] = {"handoff": False, "ante": first["ante"], "blinds": first["blinds"]}
            continue
        ref, ok = first["prefix"], True
        res = {NO: [{x: first[x] for x in KEEP}]}
        for r in range(1, rollouts):
            g = play(seed, -1, r, ante)
            ok &= g["prefix"] == ref
            res[NO].append({x: g[x] for x in KEEP})
        for k, h in enumerate(hands):
            res[h] = []
            for r in range(rollouts):
                g = play(seed, k, r, ante)
                ok &= g["prefix"] == ref
                res[h].append({x: g[x] for x in KEEP})
        rec["by_ante"][str(ante)] = {"handoff": True, "handoff_step": first["handoff_step"],
                                     "rule_choice": first["rule_choice"], "results": res, "prefix_ok": ok}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def one_seed_safe(job):
    """one_seed, but an error on a seed comes back as text in the record. An exception object from the
    game (a lupa LuaError) can't be unpickled in the main process, which doesn't load lupa: the pool's
    result thread died on one at seed ~150 of the first 2,000-seed run and the run hung. The seed's sim
    is rebuilt after an error, in case the error left it in a bad state."""
    try:
        return one_seed(job)
    except Exception:
        err = traceback.format_exc()[-4000:]
        try:
            from sim import Sim
            a = G["args"]
            G["sim"] = Sim(a.src, a.mods, mode=a.actions, enc=a.enc)
        except Exception:
            err += "\n(rebuilding the sim also failed: " + traceback.format_exc()[-1000:] + ")"
        return {"seed": job[0], "error": err}


def summarize(recs, out=sys.stdout):
    P = lambda *x: print(*x, file=out)
    errs = [r for r in recs if r.get("error")]
    if errs:
        P(f"{len(errs)} seed(s) ended in an error and are left out: " + ", ".join(r["seed"] for r in errs))
        recs = [r for r in recs if not r.get("error")]
    antes = sorted({a for r in recs for a in r.get("by_ante", {})}, key=int)
    for ante in antes:
        rows = [r["by_ante"][ante] for r in recs if ante in r.get("by_ante", {})]
        hr = [x for x in rows if x.get("handoff")]
        P(f"\n=== handoff at ante {ante}: {len(rows)} seeds, {len(hr)} reached it, "
          f"prefix mismatches {sum(1 for x in hr if not x.get('prefix_ok'))}, pins {recs[0].get('pins')}")
        if not hr:
            continue
        opts = list(hr[0]["results"])
        hands = [o for o in opts if o != NO]
        P(f"{'player after the handoff':26s} {'win rate':>9s} {'wins':>5s} {'near':>5s} {'blinds/g':>9s} {'best on seeds':>14s}")
        best = defaultdict(float)
        for x in hr:
            w = {o: np.mean([g["won"] for g in x["results"][o]]) for o in opts}
            top = max(w.values())
            if top > 0:
                ws = [o for o in opts if w[o] == top]
                for o in ws:
                    best[o] += 1 / len(ws)
        for o in opts:
            gs = [g for x in hr for g in x["results"][o]]
            wins = sum(g["won"] for g in gs)
            P(f"{o:26s} {wins / len(gs):9.2%} {wins:5d} {sum(g['near_win'] for g in gs):5d} "
              f"{np.mean([g['blinds'] for g in gs]):9.2f} {best[o]:14.1f}")
        rr = [g for x in hr for g in x["results"][x["rule_choice"]]] if hr[0].get("rule_choice") else []
        if rr:
            P(f"{'router (its own choice)':26s} {np.mean([g['won'] for g in rr]):9.2%} {sum(g['won'] for g in rr):5d} "
              f"{sum(g['near_win'] for g in rr):5d} {np.mean([g['blinds'] for g in rr]):9.2f}")
        oracle = sum(1 for x in hr if any(g["won"] for o in hands for g in x["results"][o]))
        P(f"seeds some expert won: {oracle} of {len(hr)} (a perfect router's ceiling at this handoff)")
        P("\nplays after the handoff, share by hand (same positions for every row):")
        P(f"{'':26s} " + " ".join(f"{h[:10]:>10s}" for h in hands))
        for o in opts:
            gs = [g for x in hr for g in x["results"][o]]
            tot = sum(g["late_plays"] or 0 for g in gs) or 1
            sh = np.sum([g["late_counts"] or [0] * len(hands) for g in gs], 0) / tot
            P(f"{o:26s} " + " ".join(f"{v:10.1%}" for v in sh))
        P("\nby the router's own choice (the seeds it would send to each expert): top 3 by win rate")
        for c in hands:
            sub = [x for x in hr if x.get("rule_choice") == c]
            if sub:
                rates = {o: np.mean([g["won"] for x in sub for g in x["results"][o]]) for o in opts}
                order = sorted(opts, key=lambda o: -rates[o])
                P(f"  routed to {c:16s} ({len(sub):4d} seeds): " + ", ".join(f"{o} {rates[o]:.1%}" for o in order[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--src")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-tag", default="paired-v1", help="seeds come from random.Random(tag)")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--handoff-antes", default="4", help="comma-separated antes to hand off at, e.g. 3,4,5")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="paired.jsonl")
    ap.add_argument("--summary", help="only summarize this existing output file")
    a = ap.parse_args()
    if a.summary:
        summarize([json.loads(l) for l in open(a.summary) if l.strip()])
        return
    if not (a.weights and a.src):
        ap.error("WEIGHTS and --src are needed to play")
    sys.path.insert(0, os.path.join(HERE, "..", "train"))
    from train import eval_seeds, rand_seed
    banned = set(eval_seeds(600))
    rng = random.Random(a.seed_tag)
    seeds = []
    while len(seeds) < a.seeds:
        s = rand_seed(rng)
        if s not in banned and s not in seeds:
            seeds.append(s)
    done = set()
    if os.path.exists(a.out):                      # resume: skip seeds already in the file
        done = {json.loads(l)["seed"] for l in open(a.out) if l.strip()}
    antes = [int(x) for x in a.handoff_antes.split(",") if x]
    todo = [(s, a.rollouts, antes) for s in seeds if s not in done]
    print(f"{len(seeds)} seeds ({len(done)} already done); per seed and handoff ante {antes}: "
          f"{a.rollouts} rollout(s) x (6 experts + no handoff); {a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=init, initargs=(a.weights, a.src, a.max_steps)) as pool, \
            open(a.out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(one_seed_safe, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec.get("error"):
                print(f"seed {rec['seed']}: error, recorded and skipped:\n{rec['error'].strip().splitlines()[-1]}", flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M:%S')}  {i} of {len(todo)} seeds  "
                      f"{i / max(time.time() - t0, 1e-9) * 3600:,.0f} seeds/h", flush=True)
    summarize([json.loads(l) for l in open(a.out) if l.strip()])


if __name__ == "__main__":
    main()
