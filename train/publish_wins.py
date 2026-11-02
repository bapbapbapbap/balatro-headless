"""Builds the public record of beaten seeds: one proof per seed, each replayed before it is kept.

    python3 publish_wins.py --src DIR --out wins/ <wins.jsonl ...> [--full]

A proof is the seed, the setup and every move. The game is deterministic, so replaying the moves
from the seed reproduces the run exactly; verify_games.py does that and checks every move, the
step count, the final state and the win. Writes:
  <out>/routes.jsonl       one line per beaten seed (the proof)
  <out>/beaten_seeds.csv   seed, deck, stake, moves, how it was found, game build
Seeds already in <out>/routes.jsonl are kept; new ones are added.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import Sim  # noqa: E402

KEEP = ("seed", "back", "stake", "actions", "enc", "acts", "desc", "final", "n", "progress", "mode", "iteration",
        "retry", "arm", "commit", "converted_from")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true", help="replay on the untouched game (slower)")
    ap.add_argument("--game", default="Balatro 1.0.1o-FULL (Balatro.exe sha256 "
                    "0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47), unmodded, all unlocked")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rpath = os.path.join(args.out, "routes.jsonl")
    have = {}
    if os.path.exists(rpath):
        for line in open(rpath):
            if line.strip():
                r = json.loads(line)
                have[(r["seed"], r["back"], r["stake"])] = r
    cands = {}
    for f in args.files:
        if not os.path.exists(f):
            print(f"skip {f}: not found")
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["progress"]["won"] or not r.get("verified", True):
                continue
            k = (r["seed"], r["back"], r["stake"])
            if k in have:
                continue
            if k not in cands or r["n"] < cands[k]["n"]:      # several wins on one seed: keep the shortest
                cands[k] = dict(r, source=os.path.join(os.path.basename(os.path.dirname(os.path.abspath(f))), os.path.basename(f)))
    sims, added, bad = {}, 0, 0
    for k, r in sorted(cands.items()):
        mode, enc = r.get("actions", "subsets"), r.get("enc", 1)
        if (mode, enc) not in sims:
            sims[(mode, enc)] = Sim(args.src, fast=not args.full, mode=mode, enc=enc)
        prog, n, final, problems = sims[(mode, enc)].replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"])
        if problems or n != r["n"] or final != r["final"] or not prog["won"]:
            bad += 1
            print(f"NOT PUBLISHED {r['seed']}: replay {'won' if prog['won'] else 'lost'} {problems[:1]}")
            continue
        have[k] = {x: r[x] for x in KEEP if x in r} | {"source": r["source"], "game": args.game}
        added += 1
    with open(rpath, "w") as f:
        for k in sorted(have):
            f.write(json.dumps(have[k]) + "\n")
    with open(os.path.join(args.out, "beaten_seeds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "deck", "stake", "moves", "found_in", "round", "retry", "action_list", "game"])
        for k in sorted(have):
            r = have[k]
            w.writerow([r["seed"], r["back"], r["stake"], r["n"], r.get("mode"), r.get("iteration"),
                        bool(r.get("retry")), r.get("actions", "subsets"), r["game"]])
    print(f"{added} seeds added after replaying exactly, {bad} not published; {len(have)} beaten seeds in {args.out}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
