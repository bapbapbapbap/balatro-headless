"""Plays a set of weights as hand-focus bots, one per poker hand, and reports how each does.

A hand-focus bot is the same model with one rule added to the game's move list (env.lua, "hand
focus"): whenever a play of its hand is available (and the blind would let it score), plays of
other hands aren't offered. Discards, shop, packs and everything else are left to the model.
The reward isn't involved: this only plays games.

    python3 focus_eval.py WEIGHTS.pt --src ~/balatro/local/source [--hands all] [--games 200] [--jobs 12]
                          [--out focus_eval.json]

--hands: "all" (the 12 hands plus "none", the model unrestricted) or a comma list, e.g.
         "none,Flush,Pair". "none" with the training eval's seeds and sampling should give back the
         eval score the weights were picked by (1.0: 12.79 blinds/game, 5 wins in 200), a check
         that this script plays like training's eval.
--games: how many of the training eval's fixed seeds to play (same seeds, same sampling RNG per
         seed as train.py's eval, so every hand is scored on the same games).

Untrained, the model doesn't steer discards or the shop toward the hand; it only has to play it
when it can. For a bot that learns to build toward its hand, train one with
`train.py --focus "<hand>"` (train/run_focus.sh).
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_w = {}


def _init(weights, src, max_steps):
    import torch
    from model import Net
    from sim import Sim
    torch.set_num_threads(1)
    w = torch.load(weights, map_location="cpu", weights_only=True)
    arch = w["arch"]
    sim = Sim(src, fast=True, mode=arch["actions"], enc=arch["enc"])
    if sim.vocab != w["vocab"]:
        raise SystemExit("these weights were trained on a different game version (vocabulary differs)")
    model = Net(len(w["vocab"]), d=arch["d"], layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model.load_state_dict(w["state_dict"])
    model.eval()
    hand_of = {i: v[len("hand:"):] for i, v in enumerate(sim.vocab) if v.startswith("hand:")}
    _w.update(sim=sim, model=model, hand_of=hand_of, max_steps=max_steps, torch=torch)


def _play(job):
    hand, seed = job
    from train import collate
    sim, model, hand_of, torch = _w["sim"], _w["model"], _w["hand_of"], _w["torch"]
    focus = None if hand == "none" else hand
    sim.focus = sim.env.focus = focus
    sim.H.focus_hand = focus
    rng = random.Random(f"eval-{seed}")             # train.py's eval sampling, per seed
    t0 = time.time()
    o = sim.reset(seed, "b_red", 1)
    played = collections.Counter()
    play_decisions = focus_available = 0
    done = False
    while not done and sim.steps < _w["max_steps"]:
        act = o["act"]
        plays = act[:, 0] == 1
        if plays.any():
            play_decisions += 1
            types = {hand_of.get(int(t)) for t in act[plays, 8]}
            if focus is not None and types == {focus}:
                focus_available += 1
        with torch.no_grad():
            logits, _ = model(*collate([o]))
        p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
        a = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), len(p) - 1)
        if act[a, 0] == 1:
            played[hand_of.get(int(act[a, 8]), "?")] += 1
        _, _, done = sim.step(a)
        if not done:
            o = sim.observe()
    prog = sim.progress
    return {"hand": hand, "seed": seed, "won": prog["won"], "blinds": prog["blinds_beaten"], "ante": prog["ante"],
            "steps": sim.steps, "capped": not done, "played": dict(played), "play_decisions": play_decisions,
            "focus_available": focus_available, "seconds": round(time.time() - t0, 1)}


def summary(hand, games):
    n = len(games)
    plays = collections.Counter()
    for g in games:
        plays.update(g["played"])
    total = sum(plays.values()) or 1
    dec = sum(g["play_decisions"] for g in games) or 1
    return {"hand": hand, "games": n, "wins": sum(g["won"] for g in games),
            "blinds_mean": float(np.mean([g["blinds"] for g in games])) if n else 0.0,
            "ante_mean": float(np.mean([g["ante"] for g in games])) if n else 0.0,
            "focus_share_of_plays": None if hand == "none" else plays[hand] / total,
            "focus_available_share": None if hand == "none" else sum(g["focus_available"] for g in games) / dec,
            "capped": sum(g["capped"] for g in games),
            "plays": {k: v / total for k, v in plays.most_common()}}


def main():
    from train import eval_seeds          # also puts harness/ on the path
    from env import HANDS
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--src", required=True)
    ap.add_argument("--hands", default="all")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-steps", type=int, default=1500, help="as train.py (run_v2.sh doesn't change it)")
    ap.add_argument("--out", default="focus_eval.json")
    a = ap.parse_args()

    hands = ["none", *HANDS] if a.hands == "all" else [h.strip() for h in a.hands.split(",")]
    for h in hands:
        if h != "none" and h not in HANDS:
            raise SystemExit(f"unknown hand {h!r}; use 'none' or one of: {', '.join(HANDS)}")
    seeds = eval_seeds(200)[:a.games] if a.games <= 200 else eval_seeds(a.games)
    jobs = [(h, s) for s in seeds for h in hands]     # interleaved, so partial results cover every hand
    print(f"{len(hands)} bots x {len(seeds)} seeds = {len(jobs)} games on {a.jobs} processes", flush=True)

    res = collections.defaultdict(list)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs, _init, (a.weights, a.src, a.max_steps)) as pool:
        for k, g in enumerate(pool.imap_unordered(_play, jobs), 1):
            res[g["hand"]].append(g)
            if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} games, {time.time() - t0:.0f}s", flush=True)

    rows = [summary(h, res[h]) for h in hands]
    print(f"\n{'bot':<16}{'games':>6}{'wins':>6}{'blinds':>8}{'ante':>6}{'plays of it':>13}{'available':>11}"
          f"  most played")
    for r in rows:
        top = ", ".join(f"{k} {v:.0%}" for k, v in list(r["plays"].items())[:3])
        f = "" if r["focus_share_of_plays"] is None else f"{r['focus_share_of_plays']:.0%}"
        av = "" if r["focus_available_share"] is None else f"{r['focus_available_share']:.0%}"
        print(f"{r['hand']:<16}{r['games']:>6}{r['wins']:>6}{r['blinds_mean']:>8.2f}{r['ante_mean']:>6.2f}"
              f"{f:>13}{av:>11}  {top}")
    print("\nplays of it: share of the bot's plays that were its hand. available: share of play decisions "
          "where its hand could be played (and so was the only kind of play offered).")
    with open(a.out, "w") as fh:
        json.dump({"weights": os.path.basename(a.weights), "seeds": seeds, "summary": rows,
                   "games": [g for h in hands for g in res[h]]}, fh, indent=1)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
