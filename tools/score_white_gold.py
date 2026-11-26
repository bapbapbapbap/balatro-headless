"""Scores chud's White-vs-Gold prediction (docs/hypotheses-v6.md, section added 2026-11-22 after chud's 23:12
CST message) from four eval_ladder outputs on the 600 eval seeds at temperature 0, paired seed by seed.

    python3 tools/score_white_gold.py --white-new v6g_t0_s1.jsonl --white-old dagger/twin_t0.jsonl \\
        --gold-new v6g_t0_s8.jsonl --gold-old gold/v6dt_t0_s8.jsonl

Prints, in this order, exactly what the registration commits to:
  1. White: wins new vs old, discordant seeds (new won / old won), paired z = (b - c) / sqrt(b + c),
     one-sided exact McNemar p, and the verdict: MET if z > 2 and new > old, else NOT MET.
  2. Gold: the same numbers (no verdict here; Gold is scored by G3-G5).
  3. The compression factor: White gain / Gold gain in blinds per game (wins at Gold are too few for a ratio
     of wins to mean anything), and the same ratio in wins when the Gold win gain is not 0.
Standard library only, so it runs anywhere the files are.
"""
import argparse
import json
import math
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "seed" in r and not r.get("error"):
            out[r["seed"]] = r
    return out


def mcnemar_p(b, c):
    """P(X >= b), X ~ Binomial(b + c, 1/2): the new model wins more discordant seeds than chance."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def paired(name, new_path, old_path):
    new, old = load(new_path), load(old_path)
    seeds = sorted(set(new) & set(old))
    if len(seeds) != 600:
        print(f"{name}: WARNING {len(seeds)} paired seeds, not 600 (new {len(new)}, old {len(old)})")
    wn = sum(new[s]["won"] for s in seeds)
    wo = sum(old[s]["won"] for s in seeds)
    b = sum(new[s]["won"] and not old[s]["won"] for s in seeds)
    c = sum(old[s]["won"] and not new[s]["won"] for s in seeds)
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    bn = sum(new[s]["blinds"] for s in seeds) / len(seeds)
    bo = sum(old[s]["blinds"] for s in seeds) / len(seeds)
    print(f"{name}: {len(seeds)} seeds | wins new {wn} vs old {wo} ({wn - wo:+d}) | discordant: new only {b}, "
          f"old only {c} | z {z:+.2f}, one-sided p {mcnemar_p(b, c):.4f} | blinds/game new {bn:.2f} vs old {bo:.2f} "
          f"({bn - bo:+.2f})")
    return dict(wins=wn - wo, blinds=bn - bo, z=z)


def main():
    ap = argparse.ArgumentParser()
    for k in ("white-new", "white-old", "gold-new", "gold-old"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    w = paired("White", a.white_new, a.white_old)
    met = w["z"] > 2 and w["wins"] > 0
    print(f"  chud's prediction (White, paired z > 2): {'MET' if met else 'NOT MET'}")
    g = paired("Gold ", a.gold_new, a.gold_old)
    cb = f"{w['blinds'] / g['blinds']:.2f}" if g["blinds"] else "undefined (no Gold gain)"
    cw = f"{w['wins'] / g['wins']:.2f}" if g["wins"] else "undefined (no Gold win gain)"
    print(f"Compression factor, White gain / Gold gain: blinds/game {cb}; wins {cw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
