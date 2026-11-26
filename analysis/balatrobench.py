"""A trainer checkpoint under BalatroBench community-board conditions, off-board.

Usage (from the repo root, venv active):
    python3 analysis/balatrobench.py --weights /mnt/c/Users/<you>/balatro-work/runs/v6dt/best.pt \
        --out v6dt_t1.jsonl [--no-skip 1]

Their setup: Red deck, White stake, seeds AAAAAAA..EEEEEEE, 3 runs per seed, temperature 1.0,
ranked by mean final round reached (24 = all 8 antes).

Edits nothing in the repo. Imports tools/eval_ladder.py and calls its own init() and one_seed()
in one process (no worker pool). Writes only to the --out file.

Run 1 per seed uses the normal eval sampling stream (random.Random("eval-<seed>"), rollout.py
line 95), so it matches eval_ladder seed for seed. That stream is keyed by seed alone, so runs 2
and 3 would be copies; for them the stream is renamed "eval-<seed>-r2" / "-r3" inside rollout
only. The game's own RNG is untouched: every run of a seed sees the same deck, shops and bosses,
as on BalatroBench.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from argparse import Namespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_ladder as EL  # noqa: E402
import rollout  # noqa: E402

SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]

RUN = {"n": 1, "renamed": 0}
_Random = random.Random


class RunRandom(_Random):
    def __init__(self, x=None):
        if RUN["n"] > 1 and isinstance(x, str) and x.startswith("eval-") and "-rollout" not in x:
            x = f"{x}-r{RUN['n']}"
            RUN["renamed"] += 1
        super().__init__(x)


shim = types.ModuleType("random")
shim.__dict__.update(random.__dict__)
shim.Random = RunRandom
rollout.random = shim  # only rollout's own random.Random(...) calls see the rename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/balatro/local/source"))
    ap.add_argument("--weights", required=True, help="trainer checkpoint, e.g. .../runs/v6dt/best.pt")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-skip", type=int, default=0)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    if not os.path.exists(o.weights):
        sys.exit(f"no weights at {o.weights}")
    if os.path.exists(o.out):
        sys.exit(f"{o.out} exists; pick a new name (nothing overwritten)")

    a = Namespace(src=o.src, weights=o.weights, policy="model", init_seed=0, temp=o.temp,
                  survive=0, shun_stickers=0, no_skip=o.no_skip, back="b_red", stake=1,
                  games=len(SEEDS), workers=1, out=o.out, summary=None)
    EL.init(a)

    recs = []
    with open(o.out, "w") as f:
        sha = hashlib.sha256(open(o.weights, "rb").read()).hexdigest()
        run_dir = os.path.basename(os.path.dirname(os.path.abspath(o.weights)))
        f.write(json.dumps({"header": {"bench": "balatrobench-community",
                                       "weights": f"runs/{run_dir}/{os.path.basename(o.weights)}",
                                       "sha256": sha,
                                       "temp": o.temp, "runs": o.runs, "no_skip": o.no_skip,
                                       "back": "b_red", "stake": 1, "seeds": SEEDS}}) + "\n")
        for run in range(1, o.runs + 1):
            RUN["n"], RUN["renamed"] = run, 0
            for s in SEEDS:
                t = time.time()
                rec = EL.one_seed(s)
                rec.update(run=run, secs=round(time.time() - t, 1), stream_renamed=RUN["renamed"])
                f.write(json.dumps(rec) + "\n")
                f.flush()
                recs.append(rec)
                print(json.dumps(rec), flush=True)
            if run > 1 and RUN["renamed"] == 0:
                print(f"WARNING run {run}: sampling stream never renamed, games may repeat run 1",
                      flush=True)

    ok = [r for r in recs if not r.get("error")]
    print(f"\n{len(ok)}/{len(recs)} games without error; wins {sum(r['won'] for r in ok)}")
    print("record keys:", sorted(ok[0].keys()) if ok else "-")
    for s in SEEDS:
        rows = [r for r in ok if r["seed"] == s]
        print(s, [(r["run"], r.get("ante"), r.get("blinds"), r["won"]) for r in rows])


if __name__ == "__main__":
    main()
