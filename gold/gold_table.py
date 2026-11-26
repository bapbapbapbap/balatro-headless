"""Win rates for v6g best.pt, from the eval_ladder files: every deck at Gold, plus Red at White.
Each rate counts every game in the file (they're the first N eval seeds; verify_gold.py checks that).

  python3 gold_table.py [--gold DIR] [--md out.md]

Two kinds of rate per deck:
  first 600   the registered 600-seed sweep, played on all 15 decks: the only like-for-like comparison.
  all games   everything played. crack_t0.sh stopped extending a deck once it had a win, so these
              lean high and their n differs by deck (600 to 18,000); read them with their n and interval.
"""
import argparse
import json
import math
import os
import statistics as st

DECKS = ["red", "blue", "yellow", "green", "black", "magic", "nebula", "ghost", "abandoned", "checkered",
         "zodiac", "painted", "anaglyph", "plasma", "erratic"]
ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=os.path.join(os.environ.get("BALATRO_WORK", "."), "gold"))
ap.add_argument("--md")
a = ap.parse_args()


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else max(0.0, c - h)), min(1.0, c + h)


def load(name):
    p = os.path.join(a.gold, name)
    if not os.path.exists(p):
        return None
    games, seen = [], set()
    for l in open(p):
        if l.startswith('{"seed"'):
            g = json.loads(l)
            if not g.get("error") and g["seed"] not in seen:
                seen.add(g["seed"])
                games.append(g)
    return games


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n:,} = {100 * k / n:.3g}% (95% {100 * lo:.2g}-{100 * hi:.2g}%)" if n else "-"


def rate_cell(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.3g}% ({k}/{n:,}; {100 * lo:.2g}-{100 * hi:.2g})"


rows = []
for d in DECKS:
    t0 = load(f"v6g_t0_s8_{d}.jsonl") or (load("v6g_t0_s8.jsonl") if d == "red" else None) or []
    t1 = load(f"v6g_t1_s8_{d}.jsonl") or []
    f600 = t0[:600]
    rows.append({
        "deck": d.capitalize(),
        "first600": rate_cell(sum(g["won"] for g in f600), len(f600)),
        "all_t0": rate_cell(sum(g["won"] for g in t0), len(t0)),
        "t1": rate_cell(sum(g["won"] for g in t1), len(t1)) if t1 else "",
        "ante8": f"{100 * sum(g['ante'] >= 8 for g in t0) / len(t0):.2g}%" if t0 else "-",
        "mean_ante": f"{st.mean(g['ante'] for g in t0):.2f}" if t0 else "-",
        "_sort": st.mean(g["ante"] for g in f600) if f600 else 0,
    })

white = load("v6g_t0_s1.jsonl") or []
out = ["# v6g best.pt (round 660): win rates", "",
       "Trained on the Red Deck only: imitation + RL at White stake (v6dt), then RL at Gold stake (v6g, "
       "warm-started from v6dt). Every other deck is zero-shot. One game per seed, the fixed eval seed list; "
       "t0 = temperature 0 (always the most likely move).", "",
       "## Red Deck, White Stake (training deck)", "",
       f"- v6g best.pt, t0: {pct(sum(g['won'] for g in white), len(white))}" if white else "- (no White file)",
       "- v6dt best.pt (before Gold training), t0: 273/600 = 45.5% (95% 41.6-49.5%)", "",
       "## Gold Stake, all 15 decks (t0 unless marked)", "",
       "| Deck | Win rate, first 600 seeds | Win rate, all t0 games | t1 games | Reached ante 8 (t0) | Mean ante (t0) |",
       "|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: -r["_sort"]):
    out.append(f"| {r['deck']} | {r['first600']} | {r['all_t0']} | {r['t1']} | {r['ante8']} | {r['mean_ante']} |")
tot = [load(f"v6g_t0_s8_{d}.jsonl") or (load("v6g_t0_s8.jsonl") if d == "red" else None) or [] for d in DECKS]
f6 = [g for t in tot for g in t[:600]]
out += ["", f"First 600 seeds, all 15 decks pooled: {pct(sum(g['won'] for g in f6), len(f6))}.",
        "Rows sorted by mean ante over the first 600 seeds. 'All t0 games' stopped at a deck's first win "
        "(crack_t0.sh), so those rates lean high. Red's Gold win (WCRKZKR1) came from the training run's own "
        "eval at temperature 1, not from these t0 files."]
s = "\n".join(out)
print(s)
if a.md:
    open(a.md, "w").write(s + "\n")
