"""Recount the published wins with the corrected win check (2026-11-03).

Balatro 1.0.1o sets G.GAME.won = true at the end of the ante-8 boss round whether or not the boss
was beaten (end_round, functions/state_events.lua). On a loss the game goes to GAME_OVER and never
calls win_game(). The harness read G.GAME.won alone, so a run that ran out of hands on the final
boss was counted, published and verified as a win. train/sim.py now uses the game's own test
(G.GAME.won and not GAME_OVER).

This replays every route in wins/routes.jsonl with the corrected check and splits them:
  wins/routes.jsonl        kept: routes that replay to a real win (the boss beaten, the game's win)
  wins/near_misses.jsonl       removed: routes that replay to a loss on the final boss, with the final
                           state, chips and target, so anyone can check the removal
  wins/beaten_seeds.csv    rewritten from the kept routes
Then run train/tally.py to update the counts shown in the README.

Nothing is removed unless its replay is exact (same moves, same step count, same final state);
a route that doesn't replay exactly stops the script without writing anything.

usage: python3 tools/recount_wins.py --src ~/balatro/local/source [--jobs 12] [--full] [--dry-run]
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
MODEL = {"v1": "0.1", "low": "0.1", "high": "0.1", "off": "0.1", "on": "0.1", "v2": "0.2", "v2b": "0.3"}
_SIMS = {}


def _replay(job):
    src, full, r = job
    from sim import Sim
    k = (r.get("actions", "subsets"), r.get("enc", 1))
    if k not in _SIMS:
        _SIMS[k] = Sim(src, fast=not full, mode=k[0], enc=k[1])
    sim = _SIMS[k]
    prog, n, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
    G = sim.env.lua.globals().G
    end = {"state": sim.H.state_name(), "ante": int(G.GAME.round_resets.ante),
           "chips": float(G.GAME.chips or 0), "target": float(G.GAME.blind.chips or 0),
           "blind": str(G.GAME.blind.name), "hands_left": int(G.GAME.current_round.hands_left),
           "game_won_flag": bool(G.GAME.won)}
    exact = not problems and n == r["n"] and final == r["final"]
    return r["seed"], r["back"], r["stake"], r.get("challenge"), exact, bool(prog["won"]), end, problems[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--full", action="store_true", help="replay on the untouched game (slower)")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="first N routes only (a test; implies --dry-run)")
    a = ap.parse_args()
    if a.limit:
        a.dry_run = True

    wins = os.path.join(ROOT, "wins")
    routes = [json.loads(l) for l in open(os.path.join(wins, "routes.jsonl")) if l.strip()]
    if a.limit:
        routes = routes[:a.limit]
    by_key = {(r["seed"], r["back"], r["stake"], r.get("challenge")): r for r in routes}
    print(f"replaying {len(routes)} published routes with the corrected win check ({a.jobs} processes)", flush=True)
    res = {}
    with mp.Pool(a.jobs) as pool:
        for i, (seed, back, stake, ch, exact, won, end, prob) in enumerate(
                pool.imap_unordered(_replay, [(a.src, a.full, r) for r in routes], chunksize=4)):
            res[(seed, back, stake, ch)] = (exact, won, end, prob)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(routes)}", flush=True)

    inexact = [(k, v) for k, v in res.items() if not v[0]]
    if inexact:
        for k, v in inexact[:10]:
            print(f"  NOT EXACT {k}: {v[3]}")
        raise SystemExit(f"{len(inexact)} routes didn't replay exactly; nothing written")

    keep = [by_key[k] for k in sorted(res) if res[k][1]]
    drop = [by_key[k] for k in sorted(res) if not res[k][1]]
    odd = [k for k in res if not res[k][1] and res[k][2]["state"] != "GAME_OVER"]
    tab = collections.Counter()
    for r in routes:
        tab[(MODEL.get((r.get("source") or "").split("/")[0], "?"), res[(r["seed"], r["back"], r["stake"], r.get("challenge"))][1])] += 1
    print(f"real wins: {len(keep)}   losses on the final boss counted as wins: {len(drop)}   of {len(routes)}")
    for m in sorted({m for m, _ in tab}):
        print(f"  model {m}: {tab[(m, True)]} real, {tab[(m, False)]} not won")
    if odd:
        print(f"  {len(odd)} not-won routes didn't end in GAME_OVER (e.g. {odd[0]}): check before trusting the split")
    if a.dry_run:
        print("dry run: nothing written")
        return

    with open(os.path.join(wins, "near_misses.jsonl"), "w") as f:
        for r in drop:
            k = (r["seed"], r["back"], r["stake"], r.get("challenge"))
            f.write(json.dumps(dict(r, recount={"won": False, "end": res[k][2],
                                                "why": "lost the ante-8 boss; G.GAME.won was set by end_round anyway"})) + "\n")
    tmp = os.path.join(wins, "routes.jsonl.tmp")
    with open(tmp, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(wins, "routes.jsonl"))
    with open(os.path.join(wins, "beaten_seeds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "deck", "stake", "moves", "found_in", "round", "retry", "action_list", "game"])
        for r in keep:
            w.writerow([r["seed"], r["back"], r["stake"], r["n"], r.get("mode"), r.get("iteration"),
                        bool(r.get("retry")), r.get("actions", "subsets"), r.get("game")])
    print(f"wrote wins/routes.jsonl ({len(keep)}), wins/near_misses.jsonl ({len(drop)}), wins/beaten_seeds.csv")


if __name__ == "__main__":
    main()
