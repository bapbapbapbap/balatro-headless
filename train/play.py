"""Plays one seed with released weights and prints every move.

usage: python3 play.py WEIGHTS.pt --src ~/balatro/local/source --seed ABCD1234 [--back b_red] [--stake 1]
                       [--sample-seed 0] [--max-steps 1500] [--quiet] [--save route.jsonl]

The bot sees only what a player sees and picks each move by sampling its policy with a fixed RNG
(as in training evals; always taking the single most likely move can loop on free joker drags).
Same weights + seed + --sample-seed = the same game. With --save, the game is written as a route
that train/verify_games.py can replay.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net  # noqa: E402
from sim import Sim  # noqa: E402
from train import collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--src", required=True, help="game source extracted from your own copy")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--back", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save")
    a = ap.parse_args()

    w = torch.load(a.weights, map_location="cpu", weights_only=True)
    arch = w["arch"]
    sim = Sim(a.src, fast=True, mode=arch["actions"], enc=arch["enc"])
    if sim.vocab != w["vocab"]:
        raise SystemExit("these weights were trained on a different game version (vocabulary differs)")
    model = Net(len(w["vocab"]), d=arch["d"], layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model.load_state_dict(w["state_dict"])
    model.eval()
    print(f"model {w['model_version']} ({sum(p.numel() for p in model.parameters()):,} params), "
          f"seed {a.seed}, {a.back}, stake {a.stake}", flush=True)

    rng = random.Random(f"play-{a.sample_seed}-{a.seed}")
    o = sim.reset(a.seed, a.back, a.stake)
    done = False
    while not done and sim.steps < a.max_steps:
        with torch.no_grad():
            logits, _ = model(*collate([o]))
        p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
        i = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), len(p) - 1)
        desc = sim.H.action_desc(i + 1)
        state = sim.H.state_name()
        _, prog, done = sim.step(i)
        if not a.quiet and not desc.startswith("move_joker"):
            print(f"{sim.steps:4d}  ante {prog['ante']}  {state:<16} {desc:<28} p={p[i]:.2f}")
        if not done:
            o = sim.observe()
    prog = sim.progress
    print(f"result: {'WON' if prog['won'] else 'lost'} at ante {prog['ante']}, {prog['blinds_beaten']} blinds beaten, "
          f"{sim.steps} moves" + ("" if done else " (move cap reached)"))
    if a.save:
        rec = {"seed": a.seed, "back": a.back, "stake": a.stake, "actions": arch["actions"], "enc": arch["enc"],
               "acts": list(sim.route), "desc": list(sim.desc), "final": sim.final_hash(), "n": sim.steps,
               "progress": prog, "mode": "play", "model_version": w["model_version"]}
        with open(a.save, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"route saved to {a.save}")


if __name__ == "__main__":
    main()
