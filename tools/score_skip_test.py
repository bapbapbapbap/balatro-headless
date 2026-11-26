"""Scores the skip test (docs/hypotheses-v6.md, section added 2026-11-23 after chud's 11:50 CST message): v6dt at
White, temperature 0, with skipping removed, paired seed by seed with v6dt as it plays (skips allowed) and with v6g.

    python3 tools/score_skip_test.py --noskip v6dt_t0_s1_noskip.jsonl --v6dt dagger/twin_t0.jsonl \\
        --v6g gold/v6g_t0_s1.jsonl

Prints the wins of each, the paired z of no-skip v6dt against v6dt, the share of v6g's drop (273 -> 174) that
removing skips reproduces = (v6dt wins - no-skip wins) / (v6dt wins - v6g wins), the filter's counts (decisions
where a skip was removed, and how many of those changed v6dt's move), and S1's verdict. Standard library only.
"""
import argparse
import json
import math


def load(path):
    return {r["seed"]: r for r in map(json.loads, open(path)) if "seed" in r and not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noskip", required=True)
    ap.add_argument("--v6dt", required=True)
    ap.add_argument("--v6g", required=True)
    a = ap.parse_args()
    ns, dt, g = load(a.noskip), load(a.v6dt), load(a.v6g)
    seeds = sorted(set(ns) & set(dt) & set(g))
    if len(seeds) != 600:
        print(f"WARNING: {len(seeds)} paired seeds, not 600")
    w = {k: sum(d[s]["won"] for s in seeds) for k, d in (("noskip", ns), ("v6dt", dt), ("v6g", g))}
    b = sum(ns[s]["won"] and not dt[s]["won"] for s in seeds)
    c = sum(dt[s]["won"] and not ns[s]["won"] for s in seeds)
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    ante = {k: sum(d[s]["ante"] for s in seeds) / len(seeds) for k, d in (("noskip", ns), ("v6dt", dt), ("v6g", g))}
    filt = sum(ns[s].get("no_skip", {}).get("filtered", 0) for s in seeds)
    chg = sum(ns[s].get("no_skip", {}).get("changed", 0) for s in seeds)
    print(f"{len(seeds)} seeds | wins: v6dt {w['v6dt']}, v6dt without skips {w['noskip']}, v6g {w['v6g']}")
    print(f"  mean ante reached: v6dt {ante['v6dt']:.2f}, without skips {ante['noskip']:.2f}, v6g {ante['v6g']:.2f}")
    print(f"  without skips vs v6dt: discordant {b} / {c}, z {z:+.2f}")
    print(f"  filter: {filt} decisions had a skip removed; {chg} of them changed v6dt's move ({chg / len(seeds):.2f} per game)")
    drop = w["v6dt"] - w["v6g"]
    share = (w["v6dt"] - w["noskip"]) / drop if drop else float("nan")
    print(f"  share of v6g's White drop reproduced by removing skips: {share:.0%}")
    s1 = 175 <= w["noskip"] <= 257
    print(f"  S1 (no-skip v6dt wins 175-257: skipping explains part of the drop, not all): {'MET' if s1 else 'NOT MET'}")
    if chg == 0:
        print("  CHECK FAILED: the filter never changed a move; the test did not run as registered")


if __name__ == "__main__":
    main()
