"""Pattern analysis of replayed games (output of replay_events.py).

usage: python3 analyze.py --wins W1.jsonl [W2.jsonl ...] [--late late_events.jsonl]
Every rate is compared with a baseline from the bot's own data or from the options it had, so a
pattern is only reported as strategy when it beats what chance or habit would give.
"""
import argparse
import collections
import json
import math
import random
import statistics as st

CONTAINS = {
    "Pair": {"Pair", "Two Pair", "Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind", "Flush House", "Flush Five"},
    "Three of a Kind": {"Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind", "Flush House", "Flush Five"},
    "Two Pair": {"Two Pair", "Full House", "Flush House"},
    "Straight": {"Straight", "Straight Flush"},
    "Flush": {"Flush", "Straight Flush", "Flush House", "Flush Five"},
    "Four of a Kind": {"Four of a Kind", "Five of a Kind", "Flush Five"},
}
HAND_JOKERS = {
    "j_jolly": "Pair", "j_sly": "Pair", "j_duo": "Pair",
    "j_zany": "Three of a Kind", "j_wily": "Three of a Kind", "j_trio": "Three of a Kind",
    "j_mad": "Two Pair", "j_clever": "Two Pair",
    "j_crazy": "Straight", "j_devious": "Straight", "j_order": "Straight",
    "j_droll": "Flush", "j_crafty": "Flush", "j_tribe": "Flush",
    "j_family": "Four of a Kind",
}


def load(paths):
    out, seen, bad = [], set(), 0
    for p in paths:
        for l in open(p):
            d = json.loads(l)
            if d["problem"]:
                bad += 1
                print("REPLAY PROBLEM", d["seed"], d["problem"])
                continue
            k = (d["seed"], d.get("iteration"), d.get("retry"), d.get("n"))
            if k in seen:
                continue
            seen.add(k)
            out.append(d)
    return out, bad


def jl(x):
    return x if isinstance(x, list) else []


def binom_sf(k, n, p):
    if k <= 0:
        return 1.0
    lp = lambda i: math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log(1 - p)
    return min(1.0, sum(math.exp(lp(i)) for i in range(k, n + 1)))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def plays(g):
    return [e for e in g["events"] if e["t"] == "play"]


def section(t):
    print(f"\n## {t}")


def hand_mix(G):
    section("0. Hand mix")
    top = collections.Counter()
    allp = collections.Counter()
    for g in G:
        c = collections.Counter(e["hand"] for e in plays(g))
        if c:
            top[c.most_common(1)[0][0]] += 1
            allp.update(c)
    tot = sum(allp.values())
    print("most-played hand per game:", ", ".join(f"{h} {v}" for h, v in top.most_common()))
    print("share of all plays:", ", ".join(f"{h} {pct(v, tot)}" for h, v in allp.most_common()))
    off = collections.Counter()
    took = collections.Counter()
    for g in G:
        for e in plays(g):
            b = e.get("best_hud_hand")
            if b in ("Full House", "Four of a Kind", "Flush", "Straight", "Straight Flush"):
                off[b] += 1
                took[b] += e["hand"] == b
    for h in off:
        print(f"  {h} was the best hand on offer {off[h]} times; played it {took[h]} ({pct(took[h], off[h])})")


def hand_focus(G):
    section("1. Hand focus: is its most-leveled hand the hand it plays most?")
    match = n = 0
    pairs = []
    for g in G:
        c = collections.Counter(e["hand"] for e in plays(g))
        if not c:
            continue
        top = c.most_common(1)[0][0]
        hands = g["final"]["hands"]
        lv = max(hands, key=lambda h: hands[h]["level"])
        if hands[lv]["level"] > 1:
            n += 1
            match += lv == top
            pairs.append((top, lv))
    rnd = random.Random(0)
    base = []
    for _ in range(2000):
        lv = [p[1] for p in pairs]
        rnd.shuffle(lv)
        base.append(sum(a == b for (a, _), b in zip(pairs, lv)) / len(pairs))
    obs = match / max(n, 1)
    p = (sum(b >= obs for b in base) + 1) / (len(base) + 1)
    print(f"most-leveled == most-played: {match}/{n} = {pct(match, n)} "
          f"(shuffled across games: {pct(st.mean(base), 1)}; permutation p = {p:.4f})")


def planet_picks(G):
    section("2. Planet picks: offered >=2 planets, does it take its main hand's planet?")
    hit = n = 0
    exp = 0.0
    for g in G:
        played = collections.Counter()
        for e in g["events"]:
            if e["t"] == "play":
                played[e["hand"]] += 1
            if e["t"] == "pick" and e.get("planet") and played:
                opts = {o["planet"] for o in jl(e.get("options")) if o.get("planet")}
                if len(opts) < 2:
                    continue
                fav = played.most_common(1)[0][0]
                if fav not in opts:
                    continue
                n += 1
                exp += 1 / len(opts)
                hit += e["planet"] == fav
    if n:
        print(f"choices: {n}; took its main hand's planet {hit} = {pct(hit, n)}; uniform pick gives {pct(exp, n)}; "
              f"one-sided binomial p = {binom_sf(hit, n, exp / n):.2g}")


def level_then_play(G):
    section("3. Level, then play: does using a planet shift play toward that hand?")
    before, after = [], []
    for g in G:
        evs = g["events"]
        for i, e in enumerate(evs):
            if e["t"] in ("pick", "buy_and_use", "use_consumable") and e.get("planet"):
                h = e["planet"]
                pre = [x["hand"] for x in evs[:i] if x["t"] == "play"][-20:]
                post = [x["hand"] for x in evs[i + 1:] if x["t"] == "play"][:20]
                if len(pre) >= 5 and len(post) >= 5:
                    before.append(sum(x == h for x in pre) / len(pre))
                    after.append(sum(x == h for x in post) / len(post))
    if before:
        d = [a - b for a, b in zip(after, before)]
        se = st.pstdev(d) / math.sqrt(len(d))
        print(f"planets used: {len(d)}; plays of that hand, 20 before {pct(st.mean(before), 1)}, "
              f"20 after {pct(st.mean(after), 1)} ({100 * st.mean(d):+.1f} pts, 95% +/-{196 * se:.1f})")


def greedy(G):
    section("4. Does it just play the highest-HUD hand on offer?")
    tot = same = 0
    by = collections.Counter()
    for g in G:
        for e in plays(g):
            tot += 1
            if e["hand"] == e.get("best_hud_hand"):
                same += 1
            else:
                by[(e.get("best_hud_hand"), e["hand"])] += 1
    print(f"plays: {tot}; played the best hand type on offer: {pct(same, tot)}")
    print("most common departures (best -> played):", "; ".join(f"{a} -> {b} {v}" for (a, b), v in by.most_common(6)))


def joker_hand(G):
    section("5. Joker-hand synergy: holding a hand-type joker, does it play that hand more?")
    rows = collections.defaultdict(lambda: [0, 0, 0, 0])
    for g in G:
        held = set()
        for e in g["events"]:
            if e["t"] == "play":
                cur = set(jl(e.get("jokers")))
                for j, T in HAND_JOKERS.items():
                    hit = e["hand"] in CONTAINS[T]
                    r = rows[j]
                    if j in cur:
                        r[2] += hit; r[3] += 1
                    elif j not in held:
                        r[0] += hit; r[1] += 1
                held |= cur
    for j, (bh, bn, ah, an) in sorted(rows.items(), key=lambda x: -x[1][3]):
        if an >= 30:
            print(f"  {j[2:]} ({HAND_JOKERS[j]}): before buying {pct(bh, bn)} -> while held {pct(ah, an)} (n held {an})")
    m = n = 0
    for g in G:
        played = collections.Counter()
        for e in g["events"]:
            if e["t"] == "play":
                played[e["hand"]] += 1
            if e["t"] == "buy" and e.get("key") in HAND_JOKERS and sum(played.values()) >= 10:
                T = HAND_JOKERS[e["key"]]
                m += sum(v for h, v in played.items() if h in CONTAINS[T]) / sum(played.values())
                n += 1
    if n:
        print(f"hand-type jokers bought after 10+ plays: {n}; avg share of its earlier plays containing that hand: {pct(m, n)}")


def ever(gs):
    out = []
    for g in gs:
        s = set(jl(g["final"]["jokers"]))
        for e in g["events"]:
            s |= set(jl(e.get("jokers")))
        out.append(s)
    return out


def joker_lineups(G, L):
    section("6. Jokers: final lineups, wins vs recent losses, pairs above chance")
    fin = [set(jl(g["final"]["jokers"])) for g in G]
    c = collections.Counter(j for s in fin for j in s)
    print("most common in winning final lineups:", ", ".join(f"{j[2:]} {pct(v, len(fin))}" for j, v in c.most_common(15)))
    lost = [g for g in L if not g["final"]["won"] and g["ante"] >= 4]
    if lost:
        aw, al = ever(G), ever(lost)
        cw = collections.Counter(j for s in aw for j in s)
        cl = collections.Counter(j for s in al for j in s)
        rows = []
        for j in set(cw) | set(cl):
            a, b = cw[j] / len(aw), cl[j] / len(al)
            if cw[j] + cl[j] >= 40:
                rows.append((a / max(b, 1e-9), j, a, b))
        rows.sort(reverse=True)
        print(f"held at any point, wins (n={len(aw)}) vs recent losses reaching ante 4+ (n={len(al)}):")
        print("  over-represented in wins:", "; ".join(f"{j[2:]} {100*a:.0f}% vs {100*b:.0f}%" for _, j, a, b in rows[:10]))
        print("  under-represented in wins:", "; ".join(f"{j[2:]} {100*a:.0f}% vs {100*b:.0f}%" for _, j, a, b in rows[-8:]))
    pc = collections.Counter()
    for s in fin:
        for a in s:
            for b in s:
                if a < b:
                    pc[(a, b)] += 1
    n = len(fin)
    rows = sorted(((v * n / (c[a] * c[b]), a, b, v) for (a, b), v in pc.items() if v >= 15), reverse=True)
    print("pairs in winning lineups with the highest lift (>=15 games):")
    for lift, a, b, v in rows[:10]:
        print(f"  {a[2:]} + {b[2:]}: {v} games, lift {lift:.2f}")


BOSS_TESTS = {
    "bl_psychic": ("plays with 5 cards (fewer scores nothing)", lambda e: e["n"] == 5),
    "bl_eye": ("plays repeating a hand type this round (scores nothing)", lambda e: e["played_this_round"] > 0),
    "bl_ox": ("plays of the run's most-played hand (money set to $0)", lambda e: e["hand"] == e["most_played"]),
}


def bosses(G):
    section("7. Boss blinds: does it play around them?")
    rows = collections.defaultdict(list)
    normal = []
    for g in G:
        for e in plays(g):
            b = e.get("blind") or {}
            if b.get("boss") and not b.get("disabled"):
                rows[b.get("key")].append(e)
            elif b.get("key") in ("bl_small", "bl_big"):
                normal.append(e)
    rate = lambda es, f: sum(1 for e in es if f(e)) / max(len(es), 1)
    for key, (label, f) in BOSS_TESTS.items():
        es = rows.get(key, [])
        if es:
            print(f"  {key[3:]}: {label}: {pct(rate(es, f), 1)} under the boss vs {pct(rate(normal, f), 1)} on Small/Big ({len(es)} plays)")
    es = rows.get("bl_mouth", [])
    if es:
        bad = tot = 0
        first = prev = None
        for e in es:
            if prev is None or e["hands_left"] >= prev:
                first = e["hand"]
            else:
                tot += 1; bad += e["hand"] != first
            prev = e["hands_left"]
        print(f"  mouth: later plays of a different type than the first (score nothing): {pct(bad, tot)} ({tot})")
    print(f"  zero-scoring plays: Small/Big {pct(rate(normal, lambda e: e['gained'] == 0), 1)}; by boss:")
    zs = sorted(((rate(es, lambda e: e["gained"] == 0), k, len(es)) for k, es in rows.items()), reverse=True)
    for z, k, n in zs[:12]:
        print(f"    {k[3:]}: {pct(z, 1)} ({n} plays)")
    for k in ("bl_club", "bl_goad", "bl_window", "bl_head", "bl_plant", "bl_pillar"):
        es = rows.get(k)
        if es:
            print(f"  {k[3:]}: debuffed cards per play {st.mean(e['debuffed'] for e in es):.2f} vs Small/Big "
                  f"{st.mean(e['debuffed'] for e in normal):.2f} ({len(es)} plays)")


def economy(G):
    section("8. Economy")
    by = collections.defaultdict(list)
    rr = collections.defaultdict(list)
    for g in G:
        n = 0
        for e in g["events"]:
            if e["t"] == "reroll":
                n += 1
            if e["t"] == "end_shop":
                by[e["ante"]].append(e["money"]); rr[e["ante"]].append(n); n = 0
    for a in sorted(by):
        m = by[a]
        print(f"  ante {a}: money leaving shop median ${st.median(m):.0f}, >=$25 {pct(sum(x >= 25 for x in m), len(m))}, rerolls/shop {st.mean(rr[a]):.2f}")
    sk = sum(1 for g in G for e in g["events"] if e["t"] == "skip_blind")
    sel = sum(1 for g in G for e in g["events"] if e["t"] == "select_blind" and e.get("on_deck") in ("Small", "Big"))
    print(f"  skipped Small/Big blinds: {sk} of {sk + sel}")
    vc = collections.Counter(v for g in G for v in jl(g["final"]["vouchers"]))
    print("  vouchers in wins:", ", ".join(f"{v[2:]} {pct(c, len(G))}" for v, c in vc.most_common(8)))


def drags(G):
    section("9. Free actions")
    tot = sum(len(g["events"]) for g in G)
    mj = sum(1 for g in G for e in g["events"] if e["t"] == "move_joker")
    print(f"  joker drags: {pct(mj, tot)} of all moves ({mj / len(G):.0f} per game)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", nargs="+", required=True)
    ap.add_argument("--late", nargs="*", default=[])
    a = ap.parse_args()
    G, bad = load(a.wins)
    G = [g for g in G if g["final"]["won"]]
    L, badl = load(a.late) if a.late else ([], 0)
    print(f"# Patterns in {len(G)} won games ({bad} failed to replay)"
          + (f"; contrast: {len(L)} recent games, {sum(g['final']['won'] for g in L)} won" if L else ""))
    hand_mix(G); hand_focus(G); planet_picks(G); level_then_play(G); greedy(G); joker_hand(G)
    joker_lineups(G, L); bosses(G); economy(G); drags(G)


if __name__ == "__main__":
    main()
