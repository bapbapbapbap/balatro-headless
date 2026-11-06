"""Scores a paired_experts.py output against the registered predictions C1-C5 (docs/hypotheses.md,
2026-11-05 ~15:45 CST) and answers chud's three questions (one best strategy; does the handoff change
play; which handoff ante). Needs only numpy.

  python3 score_paired.py paired_moe2.jsonl [--seed-tag paired-v1] [--n 2000]

C1-C5 are scored at handoff ante 4 on all seeds and on the first 200 of the seed list (the 200 they
were registered against; the file is written in completion order, so the list is rebuilt from the tag).

Handoff ante. paired_experts.py's per-ante rows are on different seeds (ante 5 only counts seeds that
reach ante 5), and restricting to the seeds common to all antes doesn't fix it: those are the seeds on
which the early head itself reached the last ante, which is the no-handoff game, so the earlier
handoffs are compared only on seeds where the early head is known to survive. Here each handoff ante
is scored as a policy over every seed: a seed that never reaches the handoff keeps the early head's
game (identical for every player). That is the comparison that says which ante to hand off at.
"""
import argparse
import itertools
import json
import math
import random

import numpy as np

SEED_CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"        # train.py
HANDS = ["Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House"]   # model.EXPERT_HANDS
NO = "no handoff"
PLAYERS = [NO] + HANDS


def rand_seed(rng):
    return "".join(rng.choice(SEED_CHARS) for _ in range(8))


def seed_list(tag, n):
    """paired_experts.py's seed list, without importing torch."""
    rng = random.Random("balatro-eval-v1")
    banned = {rand_seed(rng) for _ in range(600)}
    rng, seeds = random.Random(tag), []
    while len(seeds) < n:
        s = rand_seed(rng)
        if s not in banned and s not in seeds:
            seeds.append(s)
    return seeds


def g(x, o):
    return x["results"][o][0]


def ci(d):
    return 1.96 * d.std(ddof=1) / math.sqrt(len(d))


def p_norm(d):
    """two-sided p for mean(d) = 0 (normal approximation; n is in the hundreds or more)."""
    z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return math.erfc(abs(z) / math.sqrt(2))


def p_binom(k, n):
    """exact two-sided sign test (McNemar on discordant pairs)."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-15))


def chi2_sf(x, k):
    """survival function of chi-square with k degrees of freedom (k odd or even, series form)."""
    if k % 2 == 0:
        return math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(k // 2))
    s = math.erfc(math.sqrt(x / 2))
    t = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for i in range(1, (k + 1) // 2):
        s += t
        t *= x / (2 * i + 1)
    return s


def score_c(recs, label):
    hr = [r["by_ante"]["4"] for r in recs if r["by_ante"]["4"].get("handoff")]
    B = {o: np.array([g(x, o)["blinds"] for x in hr], float) for o in PLAYERS}
    W = {o: np.array([g(x, o)["won"] for x in hr], float) for o in PLAYERS}
    rt = np.array([g(x, x["rule_choice"])["blinds"] for x in hr], float)
    emean = np.mean([B[h] for h in HANDS], 0)
    share = {o: np.sum([g(x, o)["late_counts"] for x in hr], 0) / sum(g(x, o)["late_plays"] for x in hr)
             for o in PLAYERS}
    print(f"\n## C1-C5, handoff at ante 4, {label}: {len(hr)} of {len(recs)} seeds reached it")
    for o in PLAYERS:
        print(f"  {o:16s} blinds/game {B[o].mean():6.3f}   wins {int(W[o].sum()):3d}")
    print(f"  {'router':16s} blinds/game {rt.mean():6.3f}")
    lead = []
    for k, h in enumerate(HANDS):
        other = max(share[o][k] for o in PLAYERS if o != h)
        lead.append(share[h][k] > other)
        print(f"  C1 {h:16s} own {share[h][k]:7.3%}  best other {other:7.3%}  {'leads' if lead[-1] else '-'}")
    print(f"  C1: {sum(lead)} of 6 -> {'PASS' if sum(lead) >= 3 else 'FALSIFIED'}")
    d = emean - B[NO]
    print(f"  C2: expert mean - no handoff {d.mean():+.3f} (95% CI ±{ci(d):.3f}) -> {'PASS' if d.mean() >= 0.5 else 'FALSIFIED'}")
    m = {h: B[h].mean() for h in HANDS}
    hi, lo = max(m, key=m.get), min(m, key=m.get)
    print(f"  C3: best {hi} {m[hi]:.3f} - worst {lo} {m[lo]:.3f} = {m[hi] - m[lo]:.3f} "
          f"(paired ±{ci(B[hi] - B[lo]):.3f}) -> {'PASS' if m[hi] - m[lo] < 1.0 else 'FALSIFIED'}")
    d = rt - emean
    print(f"  C4: router - expert mean {d.mean():+.3f} (±{ci(d):.3f}) -> {'PASS' if d.mean() < 0.3 else 'FALSIFIED'}")
    rank = sorted(HANDS, key=lambda h: -m[h])
    print(f"  C5: Full House is #{rank.index('Full House') + 1} of 6 by blinds/game -> "
          f"{'PASS' if rank.index('Full House') >= 2 else 'FALSIFIED'}")


def best_strategy(recs):
    print("\n## Q1: does one expert win the most? (seeds that reached the handoff)")
    for ante in ("3", "4", "5"):
        hr = [r["by_ante"][ante] for r in recs if r["by_ante"][ante].get("handoff")]
        W = np.array([[g(x, h)["won"] for h in HANDS] for x in hr], int)
        B = np.array([[g(x, h)["blinds"] for h in HANDS] for x in hr], float)
        k, T, L = len(HANDS), W.sum(0), W.sum(1)
        q = (k - 1) * (k * (T ** 2).sum() - T.sum() ** 2) / (k * T.sum() - (L ** 2).sum())
        sig = []
        for i, j in itertools.combinations(range(k), 2):
            p = p_norm(B[:, i] - B[:, j])
            if p * 15 < 0.05:
                sig.append(f"{HANDS[i]} vs {HANDS[j]} {(B[:, i] - B[:, j]).mean():+.2f}")
        hi, lo = int(T.argmax()), int(T.argmin())
        n10 = int(((W[:, hi] == 1) & (W[:, lo] == 0)).sum())
        n01 = int(((W[:, hi] == 0) & (W[:, lo] == 1)).sum())
        print(f"  ante {ante} ({len(hr)} seeds): wins " + ", ".join(f"{h} {t}" for h, t in zip(HANDS, T)) +
              f"; Cochran's Q {q:.2f}, p {chi2_sf(q, k - 1):.2f}; most vs fewest wins ({HANDS[hi]} vs {HANDS[lo]}) "
              f"{n10}:{n01}, p {p_binom(n10, n10 + n01):.3f} uncorrected; blinds pairs past Bonferroni (15): "
              f"{', '.join(sig) or 'none'}")


def play_mix(recs):
    print("\n## Q2: plays after the handoff, share by hand (ante 4, same positions for every row)")
    hr = [r["by_ante"]["4"] for r in recs if r["by_ante"]["4"].get("handoff")]
    print(f"  {'':16s} " + " ".join(f"{h[:10]:>10s}" for h in HANDS))
    for o in PLAYERS:
        sh = np.sum([g(x, o)["late_counts"] for x in hr], 0) / sum(g(x, o)["late_plays"] for x in hr)
        print(f"  {o:16s} " + " ".join(f"{v:10.2%}" for v in sh))


def handoff_ante(recs):
    print("\n## Q3: handoff ante as a policy over every seed (unreached seeds keep the early head's game)")

    def val(r, ante, who, key):
        x = r["by_ante"][ante]
        if not x.get("handoff"):
            return float(x["blinds"]) if key == "blinds" else 0.0    # ended before the handoff: not a win
        if who == "expert mean":
            return float(np.mean([g(x, h)[key] for h in HANDS]))
        if who == "router":
            return float(g(x, x["rule_choice"])[key])
        return float(g(x, who)[key])

    never = np.array([val(r, "3", NO, "blinds") for r in recs])
    assert all((never == np.array([val(r, a, NO, "blinds") for r in recs])).all() for a in ("4", "5")), \
        "the no-handoff game should be the same at every ante"
    print(f"  never hand off: {never.mean():.3f} blinds/seed, "
          f"{int(sum(val(r, '3', NO, 'won') for r in recs))} wins of {len(recs)}")
    for who in ("expert mean", "router"):
        v = {a: np.array([val(r, a, who, "blinds") for r in recs]) for a in ("3", "4", "5")}
        w = {a: sum(val(r, a, who, "won") for r in recs) for a in ("3", "4", "5")}
        print(f"  {who}: " + "; ".join(f"ante {a} {v[a].mean():.3f} blinds/seed, {w[a]:.1f} wins" for a in v))
        for a, b in (("4", "3"), ("5", "4"), ("5", "3")):
            d = v[a] - v[b]
            print(f"    ante {a} - ante {b}: {d.mean():+.3f} ± {ci(d):.3f}")
    for a in ("3", "4", "5"):
        print(f"  reached the ante-{a} handoff: {sum(r['by_ante'][a].get('handoff', False) for r in recs)} of {len(recs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--seed-tag", default="paired-v1")
    ap.add_argument("--n", type=int, default=0, help="seeds in the run (default: lines in the file)")
    a = ap.parse_args()
    recs = {}
    for line in open(a.file):
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                recs[r["seed"]] = r
    order = [s for s in seed_list(a.seed_tag, a.n or len(recs)) if s in recs]
    if len(order) != len(recs):
        print(f"warning: {len(recs) - len(order)} seeds in the file aren't in the '{a.seed_tag}' list")
    allr = [recs[s] for s in order]
    score_c(allr, f"all {len(allr)} seeds")
    score_c(allr[:200], "first 200 of the seed list (registered)")
    best_strategy(allr)
    play_mix(allr)
    handoff_ante(allr)


if __name__ == "__main__":
    main()
