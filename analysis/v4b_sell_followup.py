"""What v4b's shop-network selling cost, and whether the critic's rise after a sell is money (diagnostic,
2026-11-11; Claude's (Claude Fable 5.2), reviewed by chud).

Reads the output of the two scripts that replayed v4b's games; plays nothing and needs no game source or torch:
  v4b_drop.py         -> v4b_drop.jsonl            (the 600 eval games of rounds 240, 260, 280)
  v4b_shop_credit.py  -> v4b_shop_credit*.jsonl    (one line per shop decision at a sell-legal state)
and prints the tables behind docs/hypotheses-v4b.md, "why v4b's shop network sold, and what it cost":

  (1) eval games, per round: sells per blind played, free-slot share, and for antes 2-5 the jokers held at the
      ante's first blind and the share of games reaching the ante that end in it, split by whether the game had a
      free-slot sell in an earlier ante. A free-slot sell: jokers held before the sell < 5 (Red Deck's slots).
  (2) training games, per checkpoint: the shop critic's value V0 at SHOP states regressed on money, jokers held,
      a free slot and ante dummies (antes 1-4; these are sell-legal states only, so at least one joker is held);
      then, for joker sells by free slot / slots full: V1 - V0 (the critic's value at the shop network's next
      decision minus its value at the sell), td, raw advantage, the sell value, its correlation with V1 - V0, and
      V1 - V0 by money held before the sell. Control: the same for consumable sells.
  (3) checkpoints 220-260 pooled: td on a sell by the joker sold and by jokers held before a free-slot sell.

Normalized advantages (adv_n) are not in the jsonl; they are in v4b_shop_credit.py's own printed tables.

usage (any Python 3 with numpy):
  python3 analysis/v4b_sell_followup.py --drop WORK/v4b_drop.jsonl \
      --credit WORK/v4b_shop_credit_c.jsonl WORK/v4b_shop_credit_b.jsonl
Files may be .jsonl or .jsonl.gz. With several --credit files a checkpoint is read from the first file that has it.
"""
import argparse
import collections
import gzip
import json

import numpy as np

SLOTS = 5


def lines(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ols(cols, y):
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return b, 1 - r.var() / y.var()


def mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def eval_games(path):
    R = collections.defaultdict(list)
    for d in lines(path):
        R[d["round"]].append(d)
    print(f"(1) eval games: {path}")
    for r in sorted(R):
        G = R[r]
        bad = sum(1 for d in G if d["mismatch"])
        blinds = sum(len(d["blinds"]) for d in G)
        S = [s for d in G for s in d["sells"]]
        free = [s for s in S if s["jokers_before"] < SLOTS]
        print(f"round {r}: {len(G)} games ({bad} mismatched), blinds/game {mean([d['return'] for d in G]):.2f}, "
              f"sells/game {len(S) / len(G):.2f}, per blind played {len(S) / blinds:.3f}, free-slot share "
              f"{len(free) / max(1, len(S)):.2f}, free-slot sells/game {len(free) / len(G):.2f}")
        for a in (2, 3, 4, 5):
            reach = [d for d in G if any(b["ante"] == a for b in d["blinds"])]
            if not reach:
                continue
            jk = lambda d: len(next(b for b in d["blinds"] if b["ante"] == a)["jokers"])
            sold = lambda d: any(s["ante"] < a and s["jokers_before"] < SLOTS for s in d["sells"])
            die = lambda L: mean([d["ante_end"] == a and not d["won"] for d in L])
            A, B = [d for d in reach if sold(d)], [d for d in reach if not sold(d)]
            print(f"  ante {a}: reached {len(reach):3d}, jokers at its first blind {mean([jk(d) for d in reach]):.2f}, "
                  f"end here {die(reach):.2f} | earlier free-slot sell: n={len(A):3d} jokers {mean([jk(d) for d in A]):.2f} "
                  f"end here {die(A):.2f} | none: n={len(B):3d} jokers {mean([jk(d) for d in B]):.2f} end here {die(B):.2f}")
    print()


def credit(paths):
    rows, src = collections.defaultdict(list), {}
    for p in paths:
        for d in lines(p):
            if src.setdefault(d["ckpt"], p) == p:
                rows[d["ckpt"]].append(d)
    print("(2) shop decisions at sell-legal states, per checkpoint")
    for ck in sorted(rows):
        R = rows[ck]
        S = [d for d in R if d["state"] == "SHOP" and d["ante"] <= 4]
        y = np.array([d["V0"] for d in S])
        b, r2 = ols([np.array([d["money"] for d in S]), np.array([d["jokers"] for d in S], float),
                     np.array([d["jokers"] < d["slots"] for d in S], float)]
                    + [np.array([d["ante"] == a for d in S], float) for a in (2, 3, 4)], y)
        sells = [d for d in R if d["verb"] == "sell_joker"]
        print(f"\nckpt {ck} ({src[ck]}): {len(R)} decisions, {len(sells)} joker sells")
        print(f"  V0 at SHOP, antes 1-4 (n={len(S)}, R2 {r2:.2f}): per $1 {b[1]:+.3f}, per joker held {b[2]:+.3f}, "
              f"free slot {b[3]:+.3f}")
        for name, sel in (("free slot ", [d for d in sells if d["jokers"] < d["slots"]]),
                          ("slots full", [d for d in sells if d["jokers"] >= d["slots"]])):
            if len(sel) < 8:
                print(f"  sells, {name}: n={len(sel)} (too few)")
                continue
            dv = np.array([d["V1"] - d["V0"] for d in sel])
            val = np.array([d["dmoney"] for d in sel])
            mo = np.array([d["money"] for d in sel])
            print(f"  sells, {name}: n={len(sel):3d} P(move) {mean([d['p'] for d in sel]):.3f} | V1-V0 mean {dv.mean():+.3f} "
                  f"median {np.median(dv):+.3f} share>0 {np.mean(dv > 0):.2f} | td {mean([d['td'] for d in sel]):+.3f} "
                  f"adv {mean([d['adv'] for d in sel]):+.3f} | sell value ${val.mean():.1f}, corr(V1-V0, value) "
                  f"{np.corrcoef(dv, val)[0, 1]:+.2f}")
            parts = []
            for lo, hi in ((-999, 0), (0, 4), (4, 10), (10, 999)):
                k = (mo >= lo) & (mo < hi)
                if k.sum() >= 5:
                    parts.append(f"${lo if lo > -999 else '<0'}{'-' + str(hi) if lo > -999 and hi < 999 else '+' if hi == 999 else ''}"
                                 f": n={k.sum()} {dv[k].mean():+.3f}")
            print("      V1-V0 by money before the sell: " + "; ".join(parts))
        sc = [d for d in R if d["verb"] == "sell_consumable"]
        if sc:
            print(f"  control, consumable sells: n={len(sc)} V1-V0 {mean([d['V1'] - d['V0'] for d in sc]):+.3f}, "
                  f"value ${mean([d['dmoney'] for d in sc]):.1f}")
    P = [d for ck in (220, 240, 260) for d in rows.get(ck, []) if d["verb"] == "sell_joker"]
    if not P:
        return
    print(f"\n(3) checkpoints 220-260 pooled: {len(P)} joker sells, td by the joker sold")
    by = collections.defaultdict(list)
    for d in P:
        by[d["key"]].append(d["td"])
    big = {k: v for k, v in by.items() if len(v) >= 8}
    m = np.array([np.mean(v) for v in big.values()])
    allv = np.array([d["td"] for d in P])
    between = sum(len(v) * (np.mean(v) - allv.mean()) ** 2 for v in by.values()) / ((allv - allv.mean()) ** 2).sum()
    print(f"  {len(by)} jokers sold; {len(big)} with 8+ sells, of which {np.mean(m > 0):.2f} have a mean td > 0 "
          f"(min {m.min():+.2f}, median {np.median(m):+.2f}, max {m.max():+.2f})")
    print(f"  share of td's variance between jokers: {between:.2f} (about {(len(by) - 1) / len(P):.2f} if the joker did not matter)")
    for k in sorted(by, key=lambda k: (-len(by[k]), k))[:14]:
        print(f"    {k:24s} n={len(by[k]):3d} td {np.mean(by[k]):+.2f} share>0 {np.mean(np.array(by[k]) > 0):.2f}")
    print("  free-slot sells by jokers held before the sell:")
    for j in range(1, SLOTS):
        v = [d["td"] for d in P if d["jokers"] < d["slots"] and d["jokers"] == j]
        if v:
            print(f"    {j} held: n={len(v):3d} td {np.mean(v):+.2f} share>0 {np.mean(np.array(v) > 0):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drop", help="v4b_drop.py's output")
    ap.add_argument("--credit", nargs="+", default=[], help="v4b_shop_credit.py's --out files")
    a = ap.parse_args()
    if a.drop:
        eval_games(a.drop)
    if a.credit:
        credit(a.credit)


if __name__ == "__main__":
    main()
