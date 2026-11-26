"""Each training win's attempt number on its seed: 1 = won the first time the run saw the seed (cold),
k = won on the k-th game on that seed in this run.

Games saved from 2026-11-22 on carry an `attempt` field. Older runs don't (v6g launched without it), so the
count is rebuilt from the run's saved games (every train game is in games/r*_train.jsonl.gz with its seed).
Needs --save-games all (the default); with only wins saved, older runs can't be counted.

    python3 tools/win_attempts.py <work folder>/runs/<run>
"""
import collections
import glob
import gzip
import json
import os
import sys


def main(run):
    hist = collections.defaultdict(list)      # seed -> train games in play order: (final, arm, return)
    fresh = retry = 0
    for p in sorted(glob.glob(os.path.join(run, "games", "r*_train.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                if d.get("curriculum"):
                    continue
                hist[d["seed"]].append((d["final"], d.get("arm"), d.get("return")))
                fresh += d.get("arm") == "fresh"
                retry += d.get("arm") not in ("fresh", None)
    wins = [json.loads(l) for l in open(os.path.join(run, "wins.jsonl"))]
    wins = [w for w in wins if w.get("mode") == "train"]
    cold = 0
    for w in wins:
        games = hist.get(w["seed"], [])
        idx = next((i for i, g in enumerate(games) if g[0] == w["final"]), None)
        rebuilt = idx + 1 if idx is not None else None
        att = w.get("attempt", rebuilt)
        if w.get("attempt") is not None and rebuilt is not None and rebuilt != w["attempt"]:
            print(f"  note: {w['seed']} recorded attempt {w['attempt']}, saved games say {rebuilt}")
        cold += att == 1
        before = [g[2] for g in games[:idx]] if idx is not None else []
        print(f"it {w['iteration']:>5}  {w['seed']}  attempt {att if att is not None else '?':>3}  arm {w.get('arm')}"
              f"  earlier blinds {before}")
    print(f"{len(wins)} wins: {cold} cold (attempt 1) from {fresh:,} fresh games, "
          f"{len(wins) - cold} on a retry from {retry:,} retry games")


if __name__ == "__main__":
    main(sys.argv[1])
