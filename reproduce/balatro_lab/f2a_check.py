"""F2a (docs/hypotheses-v6.md, 2026-11-19, fix F2): at Red/White the teacher's moves must be identical with and without
F2. Compares, seed by seed, the games of a run made with F2 against the same seeds' saved games made before it.

Ours (Claude's, reviewed by chud).

    python3 f2a_check.py --before <teach_v6 dir> --after <dir of the re-run seeds>

A game counts as identical when every decision has the same legal-move text, the same teacher move and the same move
played. The first difference per game is printed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys


def games(d):
    rows = []
    for f in sorted(os.listdir(d)):
        if f.startswith("games_w") and f.endswith(".jsonl"):
            rows += [json.loads(l) for l in open(os.path.join(d, f))]
    return [r for r in rows if "error" not in r]


def moves(d, seed):
    with gzip.open(os.path.join(d, "games", f"{seed}.pkl.gz"), "rb") as f:
        return [(x.get("desc"), x.get("a"), x.get("played")) for x in pickle.load(f)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    a = ap.parse_args()
    same = n = 0
    for r in games(a.after):
        seed = r["seed"]
        if not os.path.exists(os.path.join(a.before, "games", f"{seed}.pkl.gz")):
            print(f"{seed}: no saved game before F2")
            continue
        x, y = moves(a.before, seed), moves(a.after, seed)
        n += 1
        k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if k is None and len(x) == len(y):
            same += 1
        else:
            k = min(len(x), len(y)) if k is None else k
            print(f"{seed}: differs at decision {k} of {len(x)}/{len(y)}: before {x[k] if k < len(x) else None}, "
                  f"after {y[k] if k < len(y) else None}")
    print(f"F2a: {same} of {n} games identical move for move")
    return 0 if n and same == n else 1


if __name__ == "__main__":
    sys.exit(main())
