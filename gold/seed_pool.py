"""Seed pool: do best.pt's Gold wins land on the same seeds across decks, and how wide is the band of seeds it wins?

  python3 seed_pool.py [--gold DIR] [--md out.md] [--csv out.csv] [--boot 4000]

Reads the t0 eval files v6g_t0_s8_<deck>.jsonl (Red: the new file plus the old v6g_t0_s8.jsonl; even_9000.sh
checks they replay move for move). One game per seed per deck. Works on unequal n: each pair of decks is compared
only on the seeds both played, so it can run before even_9000.sh finishes. Reads files only; changes nothing.

Model (rough): every deck draws its wins uniformly from one shared band of seeds, a fraction f of all seeds.
Then P(decks i and j both win a seed) = p_i p_j / f, p = a deck's win rate. With no shared band (f = 1, wins
independent across decks) the expected number of shared-win pairs is E1 = sum over deck pairs of
overlap_ij * p_i * p_j. Observing O pairs gives f = E1 / O. The 95% interval is a Poisson bootstrap over seeds,
which keeps a 3-deck seed's 3 pairs together.

What f is: how wide the band of seeds that best.pt at t0 can win is, so p_d / f is its win rate inside the band.
It is the rough ceiling for this policy, NOT the fraction of seeds the game lets anyone win: a stronger player
wins seeds outside the band, and only a stronger policy or search raises that floor.
"""
import argparse
import itertools
import json
import math
import os
import random
from collections import defaultdict

DECKS = ["red", "blue", "yellow", "green", "black", "magic", "nebula", "ghost", "abandoned", "checkered",
         "zodiac", "painted", "anaglyph", "plasma", "erratic"]
ALTERED = {"abandoned", "checkered", "erratic"}  # starting cards differ from the standard 52
# Win seeds shared across decks before even_9000.sh (STATUS 11-24), with the decks that had won them.
# Red's WCRKZKR1 win was at t1 (training eval); its t0 result is reported separately.
REGISTERED = {"B88LZ88C": {"anaglyph", "magic", "nebula"}, "HTBDF155": {"checkered", "nebula", "painted"},
              "9QGDSETV": {"checkered", "magic"}, "T4IKVQ7T": {"anaglyph", "nebula"},
              "WCRKZKR1": {"abandoned", "red"}}
PREDICTION = 10  # Claude's, registered in chat 11-24 ~10:37: >= 10 of the other 63 deck/seed cells win at t0

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=os.path.join(os.environ.get("BALATRO_WORK", "."), "gold"))
ap.add_argument("--md")
ap.add_argument("--csv")
ap.add_argument("--boot", type=int, default=4000)
a = ap.parse_args()


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    q = k / n
    c = (q + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else max(0.0, c - h)), min(1.0, c + h)


def load(name):
    out, path = {}, os.path.join(a.gold, name)
    if os.path.exists(path):
        for line in open(path):
            if line.startswith('{"seed"'):
                g = json.loads(line)
                if not g.get("error"):
                    out.setdefault(g["seed"], bool(g["won"]))
    return out


G = {}
for d in DECKS:
    g = load(f"v6g_t0_s8_{d}.jsonl")
    if d == "red":
        for s, w in load("v6g_t0_s8.jsonl").items():
            g.setdefault(s, w)
    if g:
        G[d] = g
decks = [d for d in DECKS if d in G]
n = {d: len(G[d]) for d in decks}
W = {d: {s for s, w in G[d].items() if w} for d in decks}
won_by = defaultdict(set)
for d in decks:
    for s in W[d]:
        won_by[s].add(d)
pairs = list(itertools.combinations(decks, 2))
ov = {(i, j): len(G[i].keys() & G[j].keys()) for i, j in pairs}


def family(i, j):
    k = (i in ALTERED) + (j in ALTERED)
    return ["standard-standard", "standard-altered", "altered-altered"][k]


def e1_o(weight):
    """Expected pairs with no shared band (E1) and observed pairs (O), overall and by family."""
    pp = {d: sum(weight.get(s, 0) for s in W[d]) / n[d] for d in decks}
    E, O = defaultdict(float), defaultdict(float)
    for i, j in pairs:
        e = ov[i, j] * pp[i] * pp[j]
        E["all"] += e
        E[family(i, j)] += e
    for s, ds in won_by.items():
        for i, j in itertools.combinations(sorted(ds, key=DECKS.index), 2):
            O["all"] += weight.get(s, 0)
            O[family(i, j)] += weight.get(s, 0)
    return E, O


def poisson1(rng):
    k, t, L = 0, rng.random(), math.exp(-1)
    while t > L:
        k += 1
        t *= rng.random()
    return k


E, O = e1_o({s: 1 for s in won_by})
f = E["all"] / O["all"] if O["all"] else float("inf")
rng = random.Random(0)
boot = []
for _ in range(a.boot):
    e, o = e1_o({s: poisson1(rng) for s in won_by})
    boot.append(e["all"] / o["all"] if o["all"] else float("inf"))
boot.sort()
f_lo, f_hi = boot[int(0.025 * a.boot)], boot[int(0.975 * a.boot) - 1]

# Seeds won by 2+ decks: observed vs expected with independent decks (Poisson binomial over the decks that played it).
played_by = defaultdict(list)
for d in decks:
    for s in G[d]:
        played_by[s].append(d)
exp2 = 0.0
for s, ds in played_by.items():
    q0 = math.prod(1 - len(W[d]) / n[d] for d in ds)
    q1 = sum(len(W[d]) / n[d] * math.prod(1 - len(W[e]) / n[e] for e in ds if e != d) for d in ds)
    exp2 += 1 - q0 - q1
multi = {s: ds for s, ds in won_by.items() if len(ds) >= 2}


def pct(x):
    return "inf" if x == float("inf") else f"{100 * x:.2g}%"


tw, tn = sum(len(W[d]) for d in decks), sum(n.values())
out = ["# Seed pool: best.pt (v6g r660) Gold wins across decks, t0", "",
       f"{len(decks)} decks, {tn:,} games, {tw} wins; common seeds played by every deck: "
       f"{len(set.intersection(*(set(G[d]) for d in decks))):,}.", "",
       "## Do wins share seeds?", "",
       f"- Seeds won by 2+ decks: **{len(multi)}** observed, {exp2:.3f} expected if decks won independently.",
       f"- Shared-win deck pairs: **{O['all']:.0f}** observed, {E['all']:.3f} expected (E1).", "",
       "| Seed | Decks that won it |", "|---|---|"]
for s, ds in sorted(multi.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    out.append(f"| {s} | {', '.join(sorted(ds, key=DECKS.index))} |")
out += ["", "By deck family (altered = Abandoned, Checkered, Erratic: starting cards differ from the standard 52):", "",
        "| Pairs | Observed | Expected (independent) | Enrichment O/E1 |", "|---|---|---|---|"]
for k in ["standard-standard", "standard-altered", "altered-altered"]:
    out.append(f"| {k} | {O[k]:.0f} | {E[k]:.3f} | {O[k] / E[k]:.0f}x |" if E[k] else f"| {k} | {O[k]:.0f} | 0 | - |")
out += ["", "## Band width f and the ceiling", "",
        f"**f = {pct(f)} of seeds (95% {pct(f_lo)}-{pct(f_hi)}, Poisson bootstrap over seeds, {a.boot:,} draws).** "
        "Model: one band shared by all decks, every seed in it equally winnable. f is the rough ceiling for this "
        "policy, not the game's solvable fraction (see the docstring).", "",
        "| Deck | Games | Wins | Win rate (95%) | Win rate inside the band, p / f |", "|---|---|---|---|---|"]
for d in sorted(decks, key=lambda d: -len(W[d]) / n[d]):
    k, m = len(W[d]), n[d]
    lo, hi = wilson(k, m)
    inside = (f"{pct(k / m / f)} ({pct(k / m / f_hi)}-{pct(min(1.0, k / m / f_lo))})"
              if f != float("inf") and k else "-")
    out.append(f"| {d.capitalize()} | {m:,} | {k} | {100 * k / m:.3g}% ({100 * lo:.2g}-{100 * hi:.2g}%) | {inside} |")
out.append("")
out.append("The interval on p / f carries f's uncertainty only, not p's.")

out += ["", "## Registered prediction (Claude's, 11-24 ~10:37)", "",
        f"The 5 seeds shared before even_9000.sh, on the decks that hadn't won them (63 cells): prediction >= {PREDICTION} win.", ""]
played = won = 0
new = []
for s, winners in REGISTERED.items():
    for d in decks:
        if d in winners or s not in G[d]:
            continue
        played += 1
        if G[d][s]:
            won += 1
            new.append(f"{s} {d}")
verdict = ("MET" if won >= PREDICTION else "NOT MET (so far)" if played < 63 else "NOT MET")
out.append(f"- Played {played}/63, won {won}: **{verdict}**." + (f" New wins: {', '.join(new)}." if new else ""))
if "red" in G and "WCRKZKR1" in G["red"]:
    out.append(f"- Red at t0 on WCRKZKR1 (its win was at t1): {'won' if G['red']['WCRKZKR1'] else 'lost'}.")
s = "\n".join(out)
print(s)
if a.md:
    open(a.md, "w").write(s + "\n")
if a.csv:
    with open(a.csv, "w") as fh:
        fh.write("seed,n_decks,decks\n")
        for sd, ds in sorted(won_by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            fh.write(f"{sd},{len(ds)},{' '.join(sorted(ds, key=DECKS.index))}\n")
