"""Replays saved games in a fresh game runtime and checks each is reproduced exactly: the same move
at every step (index and text), the same number of steps, the same final state and outcome.

    python3 verify_games.py --src DIR <run dir or files...> [--wins-only] [--full] [--fresh-each]

--full runs the untouched game (no visual stripping), the reference the --fast sim is checked
against. --fresh-each boots a new runtime for every game (default: one per file, reset between).
--save checks a whole save (tools/play_save.py): one new profile, every run in file order, each
one selectable on the profile as it stood then, and at the end the game's own record against 100%
(a Gold Stake win with every joker, Gold Stake on every deck, every challenge).
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
    ap.add_argument("--save", action="store_true", help="the file is one save, played in order from a new profile")
    args = ap.parse_args()
    if args.save:
        return check_save(args)
    sim, cur, n, bad, wins = None, None, 0, 0, 0
    for f, r in records(args.paths, args.wins_only):
        kind = (f, r.get("actions", "subsets"), r.get("enc", 1))      # saved games before v2 are subsets/enc 1
        if sim is None or args.fresh_each or kind != cur:
            sim, cur = Sim(args.src, fast=not args.full, mode=kind[1], enc=kind[2]), kind
        prog, steps, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
        if not problems:
            if steps != r["n"]:
                problems.append(f"{steps} steps, saved {r['n']}")
            if final != r["final"]:
                problems.append("final state differs")
            # Compare the fields the saved game recorded (games saved before 2026-11-03 lack
            # final_boss / near_win). A final-boss loss saved as a win shows up here as won False.
            got = {k: prog.get(k) for k in r["progress"]}
            if got != r["progress"]:
                problems.append(f"outcome {got} vs saved {r['progress']}")
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


def check_save(args):
    sim, n, bad, locked = None, 0, 0, 0
    for f, r in records(args.paths, False):
        kind = (r.get("actions", "subsets"), r.get("enc", 1))
        if sim is None:
            sim, first = Sim(args.src, fast=not args.full, mode=kind[0], enc=kind[1], save_mode=True), kind
        assert kind == first, "one save is played with one action list and encoding"
        opts = sim.env.save_options()
        if r.get("challenge") and r["challenge"] not in opts["challenges"] or \
                not r.get("challenge") and opts["decks"].get(r["back"], 0) < r["stake"]:
            locked += 1
            print(f"LOCKED run {n + 1} seed {r['seed']}: {r.get('challenge') or r['back']} stake {r['stake']} "
                  f"wasn't selectable on the profile at that point")
        prog, steps, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
        if not problems and (steps != r["n"] or final != r["final"] or {k: prog.get(k) for k in r["progress"]} != r["progress"]):
            problems.append("steps, final state or outcome differs")
        n += 1
        if problems:
            bad += 1
            print(f"MISMATCH run {n} seed {r['seed']}: {problems[:3]}")
    if sim is None:
        print("FAIL (empty save)")
        raise SystemExit(1)
    env, rec = sim.env, sim.env.progress_record()
    decks, jokers, chs = env.deck_keys(), env.joker_keys(), env.challenge_ids()
    gd = [d for d in decks if rec["decks"].get(d, 0) >= 8]
    gj = [j for j in jokers if rec["jokers"].get(j, 0) >= 8]
    dc = [c for c in chs if c in rec["challenges"]]
    done, total = len(gd) + len(gj) + len(dc), len(decks) + len(jokers) + len(chs)
    print(f"{n} runs replayed in order on one new profile, {bad} mismatches, {locked} not selectable when played")
    print(f"game's record: Gold Stake on {len(gd)}/{len(decks)} decks, with {len(gj)}/{len(jokers)} jokers; "
          f"{len(dc)}/{len(chs)} challenges: {done}/{total} ({100 * done / max(total, 1):.2f}%)")
    ok = not bad and not locked
    print(("SAVE 100% PROVEN" if ok and done == total else "PASS (save replays; not 100% yet)") if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
