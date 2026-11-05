"""Plays one seed with released weights and prints every move.

usage: python3 play.py WEIGHTS.pt --src ~/balatro/local/source --seed ABCD1234 [--back b_red] [--stake 1]
                       [--sample-seed 0] [--max-steps 1500] [--quiet] [--save route.jsonl]

The bot sees only what a player sees and picks each move by sampling its policy with a fixed RNG
(as in training evals; always taking the single most likely move can loop on free joker drags).
Same weights + seed + --sample-seed = the same game. With --save, the game is written as a route
that train/verify_games.py can replay.

Mixture-of-experts weights (arch "moe", e.g. balatro-bot-moe2-pins0.pt) play through the trainer's own
eval code (train.play_games): the early-game head to the ante-4 handoff, the router's expert after it,
with the run's settings as stored in the file. The game is the one that model's evals play on that
seed (fixed sampling per seed), so --sample-seed has no effect, and move probabilities aren't shown.
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
    if arch.get("moe"):
        return play_moe(w, a)
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


def play_moe(w, a):
    """A mixture of experts plays one seed exactly as in its training evals (see the docstring)."""
    from argparse import Namespace
    from model import MoE
    import train as T
    torch.set_num_threads(1)      # as the trainer's workers: same arithmetic, so the same sampled moves
    cfg = dict(w["trained"]["provenance"]["config"])
    cfg.update(src=a.src, mods=None, back=a.back, stake=a.stake, backs=[], stakes=[], challenges=[],
               max_steps=a.max_steps, verify_wins=False, save_games="none")
    args = Namespace(**cfg)
    arch = w["arch"]
    sim = Sim(a.src, fast=True, mode=arch["actions"], enc=arch["enc"])
    if list(sim.vocab) != list(w["vocab"]):
        raise SystemExit("these weights were trained on a different game version (vocabulary differs)")
    model = MoE(len(w["vocab"]), d=arch["d"], layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model.load_state_dict(w["state_dict"])
    model.eval()
    pins = [float(x) for x in model.pin_scale] + [float(model.early_pin_scale[0])]
    print(f"model {w['model_version']} (mixture of experts, {sum(p.numel() for p in model.parameters()):,} params, "
          f"pins {'all 0' if not any(pins) else pins}), seed {a.seed}, {a.back}, stake {a.stake}", flush=True)
    ep = T.play_games(sim, model, T.REWARDS[args.reward], random.Random(0), args, 0,
                      (model.state_dict(), [a.seed], "eval", w["trained"].get("iteration") or 0, 0), set())[0]
    m = ep["moe"] or {}
    k = m.get("handoff_step")
    if not a.quiet:
        for i, desc in enumerate(ep["desc"], 1):
            if k and i == k + 1:
                print(f"---- handoff to the {m['expert']} expert ({m['routed']}) ----")
            if not desc.startswith("move_joker"):
                print(f"{i:4d}  {'early' if not k or i <= k else m['expert']:<16} {desc}")
    prog = ep["progress"]
    print(f"result: {'WON' if prog['won'] else 'lost'} at ante {prog['ante']}, {prog['blinds_beaten']} blinds beaten, "
          f"{ep['n']} moves" + (" (move cap reached)" if ep["capped"] else "") +
          (f"; handed to {m['expert']} at move {k}" if k else "; no handoff"))
    if a.save:
        rec = {"seed": a.seed, "back": ep["back"], "stake": ep["stake"], "challenge": ep["challenge"],
               "actions": arch["actions"], "enc": arch["enc"], "acts": ep["acts"], "desc": ep["desc"],
               "final": ep["final"], "n": ep["n"], "progress": prog, "mode": "play",
               "model_version": w["model_version"], "moe": m}
        with open(a.save, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"route saved to {a.save}")


if __name__ == "__main__":
    main()
