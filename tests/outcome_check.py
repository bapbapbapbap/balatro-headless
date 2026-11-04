"""Cross-check: the game's own outcome signals against the harness's old counters, step by step.

train/sim.py reads outcomes from the game itself (2026-11-03): a win is the game running
win_game(); a blind beaten is the game marking it 'Defeated' in round_resets.blind_states. The old
counters (cash-out screen entries; the corrected win test G.GAME.won and not GAME_OVER) still run
alongside with Sim.compare on. This replays saved games and fails on any step where the two
disagree, or where a replay doesn't reproduce its saved game.

usage: python3 tests/outcome_check.py --src ~/balatro/local/source FILE.jsonl[.gz] ... [--per-file N] [--jobs 4]
Last line is PASS or FAIL.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
_SIMS = {}


def _check(job):
    src, r = job
    from sim import Sim
    k = (r.get("actions", "subsets"), r.get("enc", 1))
    if k not in _SIMS:
        _SIMS[k] = Sim(src, fast=True, mode=k[0], enc=k[1])
        _SIMS[k].compare = True
    sim = _SIMS[k]
    sim.mismatches = []
    prog, n, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
    exact = not problems and n == r["n"] and final == r["final"]
    skips = sum(d.split()[0] == "skip_blind" for d in r["desc"])
    return r["seed"], exact, sim.mismatches[:2], prog, skips


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--per-file", type=int, default=0, help="random sample per file (0 = all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    rng = random.Random(0)
    games = []
    for f in a.files:
        rs = load(f)
        if a.per_file and len(rs) > a.per_file:
            rs = rng.sample(rs, a.per_file)
        games += rs
    print(f"replaying {len(games)} games with both outcome counters ({a.jobs} processes)", flush=True)
    bad, wins, near, skips, blinds = [], 0, 0, 0, 0
    with mp.Pool(a.jobs) as pool:
        for seed, exact, mm, prog, sk in pool.imap_unordered(_check, [(a.src, r) for r in games], chunksize=2):
            if not exact or mm:
                bad.append((seed, "not exact" if not exact else mm))
            wins += prog["won"]; near += bool(prog.get("near_win")); skips += sk; blinds += prog["blinds_beaten"]
    print(f"{len(games)} games: {wins} wins, {near} near wins, {blinds} blinds beaten, {skips} blinds skipped")
    for b in bad[:10]:
        print("  MISMATCH", b)
    print("PASS" if not bad else f"FAIL ({len(bad)} games)")


if __name__ == "__main__":
    main()
