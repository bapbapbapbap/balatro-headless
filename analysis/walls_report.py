"""Report on analysis/walls.py's output (2026-11-13). Plain python, no sim, no torch.

Money at death and at shop exits against the interest cap, rerolls, what was left on the shelf; jokers, xMult
jokers and main-hand level for games that died vs passed the same blind; boss deaths by boss; score/target at
the death blind. XM is an approximate list of xMult joker keys (editions such as Polychrome are not seen).
usage: python3 analysis/walls_report.py walls.jsonl
"""
import json, sys, collections, statistics as st
XM = set("""j_stencil j_loyalty_card j_cavendish j_card_sharp j_madness j_vampire j_hologram j_baron j_obelisk
j_photograph j_ancient j_ramen j_acrobat j_campfire j_throwback j_bloodstone j_glass j_flower_pot j_seeing_double
j_duo j_trio j_family j_order j_tribe j_drivers_license j_caino j_triboulet j_yorick j_idol j_steel_joker
j_lucky_cat j_constellation j_hit_the_road j_blackboard j_baseball""".split())
G = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "walls.jsonl")]
G = [g for g in G if not g["mismatch"]]
kind = lambda b: "small" if b == "bl_small" else "big" if b == "bl_big" else "boss"
maxlvl = lambda s: max([int(x.split(":")[1]) for x in s.split(",") if x] or [1])
pct = lambda a, b: f"{100*a/b:.1f}% ({a}/{b})" if b else "-"
mean = lambda xs: f"{st.mean(xs):.2f}" if xs else "-"
lost = [g for g in G if not g["won"]]
print(f"games {len(G)}, wins {len(G)-len(lost)}")

# ---- blinds with outcome; cap at each blind = cap at the shop just before it (25 before any shop)
rows = []
for g in G:
    caps = [25.0] + [s["cap"] for s in g["shops"]]
    for i, b in enumerate(g["blinds"]):
        died = (not g["won"]) and i == len(g["blinds"]) - 1
        cap = caps[min(i, len(caps) - 1)] if i else 25.0
        cap = g["shops"][i-1]["cap"] if i and i-1 < len(g["shops"]) else cap
        rows.append(dict(g=g, i=i, ante=b["ante"], kind=kind(b["blind"]), key=b["blind"], died=died,
                         dollars=b["dollars"], cap=cap, nj=len(b["jokers"]), slots=b["joker_slots"],
                         nx=sum(j in XM for j in b["jokers"]), lvl=maxlvl(b["levels"]),
                         ratio=b["score"]/b["target"] if b["target"] else 0))

print("\n== H1 hoarding ==")
d48 = [r for r in rows if r["died"] and r["ante"] >= 4]
print("deaths a4-8 holding > cap at the death blind:", pct(sum(r["dollars"] > r["cap"] for r in d48), len(d48)))
d13 = [r for r in rows if r["died"] and r["ante"] <= 3]
print("deaths a1-3 holding > cap:", pct(sum(r["dollars"] > r["cap"] for r in d13), len(d13)))
print("money at death by ante (median / mean / share>cap):")
for a in range(1, 9):
    xs = [r["dollars"] for r in rows if r["died"] and r["ante"] == a]
    if xs: print(f"  a{a} n={len(xs):4d} median ${st.median(xs):.0f} mean ${st.mean(xs):.1f} over-cap "
                 f"{pct(sum(r['dollars']>r['cap'] for r in rows if r['died'] and r['ante']==a), len(xs))}")
print("same blind, died vs passed: share over cap / mean excess over cap / mean $")
for a in range(4, 9):
    for k in ("big", "boss"):
        D = [r for r in rows if r["ante"] == a and r["kind"] == k and r["died"]]
        P = [r for r in rows if r["ante"] == a and r["kind"] == k and not r["died"]]
        f = lambda R: (pct(sum(r["dollars"] > r["cap"] for r in R), len(R)), mean([max(0, r["dollars"]-r["cap"]) for r in R]), mean([r["dollars"] for r in R]))
        if D: print(f"  a{a} {k:5s} died {f(D)}  passed {f(P)}")
S = [s for g in G for s in g["shops"]]
print("\nshop exits by ante: n, over cap, rerolls/visit, [over-cap exits: rerolls/visit, affordable joker + free slot]")
for a in range(1, 9):
    A = [s for s in S if s["ante"] == a]
    O = [s for s in A if s["dollars"] > s["cap"]]
    aff = sum(any(k == "Joker" and c <= s["dollars"] for _, k, c in s["shelf"]) and s["jokers"] < s["joker_slots"] for s in O)
    if A: print(f"  a{a} n={len(A):5d} over-cap {pct(len(O), len(A))} rr {sum(s['rerolls'] for s in A)/len(A):.3f} | "
                f"rr {sum(s['rerolls'] for s in O)/max(1,len(O)):.3f}, affordable+free {pct(aff, len(O))}")
O = [s for s in S if s["dollars"] > s["cap"]]
print("all over-cap exits: mean excess", mean([s["dollars"] - s["cap"] for s in O]),
      "| free slot", pct(sum(s["jokers"] < s["joker_slots"] for s in O), len(O)),
      "| zero rerolls", pct(sum(s["rerolls"] == 0 for s in O), len(O)),
      "| could afford >=3 rerolls", pct(sum(s["dollars"] - s["cap"] >= 3*s["reroll_cost"] for s in O), len(O)))
caps = collections.Counter(g["end"]["cap"] for g in G); print("cap at game end:", dict(caps))
vc = collections.Counter(v for g in G for v in g["end"]["vouchers"]); print("vouchers:", vc.most_common(12))
W = [g for g in G if g["won"]]; print("money at win: median", st.median([g["end"]["dollars"] for g in W]) if W else "-")

print("\n== H2 build, died vs passed at the same blind: jokers / xMult jokers / max hand level / free slots ==")
for a in range(1, 9):
    for k in ("big", "boss"):
        D = [r for r in rows if r["ante"] == a and r["kind"] == k and r["died"]]
        P = [r for r in rows if r["ante"] == a and r["kind"] == k and not r["died"]]
        if len(D) >= 15:
            print(f"  a{a} {k:5s} died n={len(D):3d} {mean([r['nj'] for r in D])} / {mean([r['nx'] for r in D])} / {mean([r['lvl'] for r in D])} / {mean([r['slots']-r['nj'] for r in D])}"
                  f"   passed n={len(P):4d} {mean([r['nj'] for r in P])} / {mean([r['nx'] for r in P])} / {mean([r['lvl'] for r in P])} / {mean([r['slots']-r['nj'] for r in P])}")

print("\n== H3 boss deaths by boss (antes 1-3 | 4-8): n, mean score/target, boss death rate when faced ==")
for lo, hi in ((1, 3), (4, 8)):
    D = [r for r in rows if r["died"] and r["kind"] == "boss" and lo <= r["ante"] <= hi]
    F = collections.Counter(r["key"] for r in rows if r["kind"] == "boss" and lo <= r["ante"] <= hi)
    c = collections.Counter(r["key"] for r in D)
    print(f" antes {lo}-{hi}: {len(D)} boss deaths; top 4 share {pct(sum(v for _, v in c.most_common(4)), len(D))}")
    for k, v in c.most_common(10):
        print(f"   {k:16s} {v:4d}  ratio {mean([r['ratio'] for r in D if r['key']==k])}  death rate {pct(v, F[k])}")

print("\n== H4 margin at the death blind (score/target) ==")
for lo, hi in ((1, 3), (4, 8)):
    D = [r["ratio"] for r in rows if r["died"] and lo <= r["ante"] <= hi]
    b = collections.Counter(min(int(x*10), 9) for x in D)
    print(f" antes {lo}-{hi} n={len(D)} median {st.median(D):.2f}  >=0.8 {pct(sum(x>=0.8 for x in D), len(D))}  <0.5 {pct(sum(x<0.5 for x in D), len(D))}")
    print("   deciles:", [b[i] for i in range(10)])
