"""Pre-flight check for joint MoE training (train.py --moe; docs/hypotheses.md section I).

Builds the MoE from the early-game model and the 8 experts, plays games through the trainer's own
play_games, and checks:
  1. each game's decisions are made by member 0 (early model) up to the handoff and by one expert
     after it, and a game routed at the handoff names that expert;
  2. after the handoff, every bound pin decision took an allowed move;
  3. each game replays move for move from its saved route on a fresh sim.

usage: python3 tests/moe_check.py --src ~/balatro/local/source --early E.pt --experts "Pair=P.pt" ...
                                  [--games 4] [--stake 1] [--back b_red]
Last line is PASS or FAIL.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from model import EXPERT_HANDS, MoE  # noqa: E402
from rewards import REWARDS  # noqa: E402
from sim import Sim  # noqa: E402
from train import PLAY, HAND_COL, play_games  # noqa: E402


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck, (ck["model"] if "model" in ck else ck["state_dict"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--early", required=True)
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--back", default="b_red")
    a = ap.parse_args()

    ck, sd = load(a.early)
    sim = Sim(a.src, fast=True, mode="ordered-capped", enc=2)
    fresh = Sim(a.src, fast=True, mode="ordered-capped", enc=2)
    assert list(ck["vocab"]) == sim.vocab, "early model's vocabulary differs from this game"
    model = MoE(len(sim.vocab), d=192, layers=4, heads=6, enc=2)
    model.members[0].load_state_dict(sd)
    given = dict(x.rsplit("=", 1) for x in a.experts)
    for i, h in enumerate(EXPERT_HANDS):
        model.members[1 + i].load_state_dict(load(given[h])[1])
    model.base_rate.fill_(1.0)
    model.eval()
    args = argparse.Namespace(seed=0, pin_hand=None, pin_rate=1.0, moe=True, moe_handoff=4, moe_random=0.5,
                              moe_pin=["Pair=0.5"], moe_train_early=0, explore_hands=0.0, back=a.back,
                              stake=a.stake, max_steps=1500, shaping=1.0, gamma=0.995, stop_ante=0)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    fails, handoffs = [], 0
    for mode in ("train", "eval"):
        seeds = [f"MOECHK{mode[0].upper()}{g}" for g in range(a.games // 2)]
        eps = play_games(sim, model, REWARDS["blinds"], random.Random(0), args, 0, (state, seeds, mode, 0, 0), set())
        for e in eps:
            m = e["moe"]
            if mode == "train":
                members = [s[0]["member"] for s in e["steps"]]
                k = m["handoff_step"]
                if m["expert"] is None:
                    if any(members):
                        fails.append(f"{e['seed']}: an expert played but no handoff was recorded")
                else:
                    handoffs += 1
                    want = 1 + EXPERT_HANDS.index(m["expert"])
                    if any(members[:k]) or any(x != want for x in members[k:]):
                        fails.append(f"{e['seed']}: members don't switch once, at step {k}, to {m['expert']}")
                    for s in e["steps"][k:]:
                        o, act = s[0], s[1]
                        if "allow" in o and not o["allow"][act]:
                            fails.append(f"{e['seed']}: pinned decision took a disallowed move")
            elif m["expert"]:
                handoffs += 1
                if m["routed"] != "rule":
                    fails.append(f"{e['seed']}: eval game routed '{m['routed']}', should be by rule")
            _, n2, f2, problems = fresh.replay(e["seed"], a.back, a.stake, e["acts"], e["desc"])
            if problems or n2 != e["n"] or f2 != e["final"]:
                fails.append(f"{e['seed']}: route does not replay ({problems[:1] or 'final state differs'})")
            print(f"{mode} {e['seed']}: {e['n']} moves, ante {e['progress']['ante']}, expert {m['expert']} ({m['routed']}), "
                  f"early plays {m['early_counts']}, pin {e['pin']}")
    if handoffs == 0:
        fails.append("no game reached the handoff: routing was not checked")
    for f in fails[:10]:
        print("  " + f)
    print("PASS" if not fails else f"FAIL ({len(fails)} problems)")


if __name__ == "__main__":
    main()
