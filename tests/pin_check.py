"""Pre-flight check for the hand pin (train.py apply_pin; docs/hypotheses.md section I).

Plays a few games with released weights and a pin, on the real sim, and checks:
  1. whenever the pin binds, the move taken is a play of the pinned type or a non-play move;
  2. the action mask the model sees (collate) is exactly the pin's allow-list;
  3. each game replays move for move from its saved route on a fresh sim (the pin leaves the
     game's legal list alone, so routes stay exact).
Pair at rate 1.0 is the default: a pair is on offer at most play decisions, so the pin binds often.

usage: python3 tests/pin_check.py WEIGHTS.pt --src ~/balatro/local/source [--hand Pair] [--rate 1.0]
                                  [--games 3] [--max-steps 400] [--allow-unbound]
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
from model import Net  # noqa: E402
from sim import Sim  # noqa: E402
from train import PLAY, HAND_COL, apply_pin, collate, count_pin_move, new_pin_stats, pin_vocab_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--src", required=True)
    ap.add_argument("--hand", default="Pair")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--actions", default="ordered-capped")
    ap.add_argument("--allow-unbound", action="store_true",
                    help="pass even if the pin never bound (rare hands in a short check)")
    a = ap.parse_args()

    w = torch.load(a.weights, map_location="cpu", weights_only=True)
    arch = w["arch"]
    sim = Sim(a.src, fast=True, mode=a.actions, enc=arch["enc"])
    fresh = Sim(a.src, fast=True, mode=a.actions, enc=arch["enc"])
    assert sim.vocab == list(w["vocab"]), "weights vocabulary differs from this game"
    model = Net(len(w["vocab"]), d=arch["d"], layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model.load_state_dict(w["state_dict"])
    model.eval()
    pin_id = pin_vocab_id(sim.vocab, a.hand)

    fails, total = [], new_pin_stats()
    for g in range(a.games):
        seed = f"PINCHK{g:02d}"
        rng, prng = random.Random(f"pin-check-{g}"), random.Random(f"pin-check-coin-{g}")
        st = new_pin_stats()
        o = sim.reset(seed, "b_red", 1)
        apply_pin(o, pin_id, a.rate, prng, st)
        done = False
        while not done and sim.steps < a.max_steps:
            batch = collate([o])
            m = len(o["act"])
            if "allow" in o and not np.array_equal(batch[5][0, :m].numpy(), o["allow"]):
                fails.append(f"{seed} step {sim.steps}: model mask differs from the pin's allow-list")
            with torch.no_grad():
                logits, _ = model(*batch)
            p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
            i = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), len(p) - 1)
            row = o["act"][i]
            if "allow" in o and row[0] == PLAY and row[HAND_COL] != pin_id:
                fails.append(f"{seed} step {sim.steps}: pin bound but played '{sim.H.action_desc(i + 1)}'")
            count_pin_move(o, i, pin_id, st)
            _, _, done = sim.step(i)
            if not done:
                o = sim.observe()
                apply_pin(o, pin_id, a.rate, prng, st)
        route, desc, final, n = list(sim.route), list(sim.desc), sim.final_hash(), sim.steps
        _, n2, final2, problems = fresh.replay(seed, "b_red", 1, route, desc)
        if problems or n2 != n or final2 != final:
            fails.append(f"{seed}: route does not replay ({problems[:1] or 'final state differs'})")
        print(f"{seed}: {n} moves, ante {sim.progress['ante']}, pin {st}")
        for k in total:
            total[k] += st[k]

    if total["bound"] == 0 and not a.allow_unbound:
        fails.append(f"the pin never bound in {a.games} games: nothing was checked")
    for f in fails[:10]:
        print("  " + f)
    print(f"pin '{a.hand}' rate {a.rate}: {total}")
    print("PASS" if not fails else f"FAIL ({len(fails)} problems)")


if __name__ == "__main__":
    main()
