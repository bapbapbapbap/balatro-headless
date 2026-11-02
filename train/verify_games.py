"""Replays saved games in a fresh game runtime and checks each is reproduced exactly: the same move
at every step (index and text), the same number of steps, the same final state and outcome.

    python3 verify_games.py --src DIR <run dir or files...> [--wins-only] [--full] [--fresh-each]

--full runs the untouched game (no visual stripping), the reference the --fast sim is checked
against. --fresh-each boots a new runtime for every game (default: one per file, reset between).
Exit code 0 only if every game checked replays exactly.
"""
import argparse
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import Sim  # noqa: E402


def records(paths, wins_only):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, "wins.jsonl")] if wins_only else sorted(glob.glob(os.path.join(p, "games", "*.jsonl.gz")))
        else:
            files.append(p)
    for f in files:
        if not os.path.exists(f):
            continue
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if not wins_only or r["progress"]["won"]:
                        yield f, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--wins-only", action="store_true")
    ap.add_argument("--full", action="store_true", help="untouched game, visuals not stripped")
    ap.add_argument("--fresh-each", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sim, cur, n, bad, wins = None, None, 0, 0, 0
    for f, r in records(args.paths, args.wins_only):
        if sim is None or args.fresh_each or f != cur:
            sim, cur = Sim(args.src, fast=not args.full), f
        prog, steps, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"])
        if not problems:
            if steps != r["n"]:
                problems.append(f"{steps} steps, saved {r['n']}")
            if final != r["final"]:
                problems.append("final state differs")
            if prog != r["progress"]:
                problems.append(f"outcome {prog} vs saved {r['progress']}")
        n += 1
        wins += bool(r["progress"]["won"])
        if problems:
            bad += 1
            print(f"MISMATCH {os.path.basename(f)} round {r['iteration']} {r['mode']} seed {r['seed']}: {problems[:3]}")
        if args.limit and n >= args.limit:
            break
    print(f"{n} games replayed ({wins} wins), {bad} mismatches, {'full' if args.full else 'fast'} sim"
          + (", fresh runtime per game" if args.fresh_each else ""))
    print("PASS" if n and not bad else "FAIL")
    raise SystemExit(0 if n and not bad else 1)


if __name__ == "__main__":
    main()
