"""Balatro Lab's bot on OUR evaluation seeds: the 600 every one of our runs is scored on (train/common.py
eval_seeds, fixed RNG "balatro-eval-v1"), which Balatro Lab never saw or tuned on.

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e): balatro.run.Run
(seeded, Red Deck, White Stake, to ante 8, as its scripts/eval_value.py) and balatro.strategy.ValueBot (`--bot base`).

Why: Balatro Lab's 70.2% is on its seed set 90210, the same 480 seeds its bot was hand-tuned and evolved on (its
README's ladder is measured there), so that number can carry tuning to those seeds. Its second set (72.5%) is
held out; ours is a third, chosen before we knew of Balatro Lab. Reports wins and blinds beaten per game, the
number our evals report (v4b best.pt: 13.40 blinds/game on these 600 seeds).

    python3 eval_on_our_seeds.py --lab <balatro-lab clone> --games 600 --workers 8 --out lab_on_ours.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
from common import eval_seeds  # noqa: E402

LAB_DIR = None


def init(lab):
    os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")      # as Balatro Lab's eval_value.py sets it
    sys.path.insert(0, lab)


def play(seed):
    from balatro.run import Run
    from balatro.strategy import ValueBot
    t = time.time()
    res = Run(ValueBot(), game_seed=seed, deck="Red Deck", max_ante=8).play()
    return {"seed": seed, "won": bool(res.won), "ante": res.ante, "blinds": res.blinds_beaten,
            "progress": res.progress, "seconds": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="lab_on_ours.jsonl")
    a = ap.parse_args()
    seeds = eval_seeds(600)[:a.games]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["seed"] for l in open(a.out)}
    todo = [s for s in seeds if s not in done]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=init, initargs=(a.lab,)) as pool, open(a.out, "a") as out:
        for k, r in enumerate(pool.imap_unordered(play, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if k % 24 == 0 or k == len(todo):
                rows = [json.loads(l) for l in open(a.out)]
                w = sum(r["won"] for r in rows)
                b = sum(r["blinds"] for r in rows) / len(rows)
                print(f"{time.strftime('%H:%M:%S')}  {len(rows)}/{len(seeds)} games: won {w} "
                      f"({100 * w / len(rows):.1f}%), {b:.2f} blinds/game, {(time.time() - t0) / 60:.0f} min",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
