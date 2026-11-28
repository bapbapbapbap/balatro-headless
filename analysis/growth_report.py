"""Report for analysis/growth_check.py: per joker type, observed per-round growth vs build_phi.lua GROWTH, and
the 6-round projection vs what happened."""
import json, sys, statistics as st
from collections import defaultdict
d = json.load(open(sys.argv[1]))
G, R = d["growth"], d["routes"]
inst = defaultdict(list)            # (route, name, sid, path) -> [(round, v)]
for k, rows in R.items():
    for rd, n, sid, p, v in rows:
        inst[(k, n, sid, p)].append((rd, v))
by = defaultdict(lambda: {"rates": [], "rounds": 0, "proj": []})
for (k, n, sid, p), ser in inst.items():
    ser.sort()
    rate_g = G[n][p]; floor = G[n].get("floor")
    b = by[(n, p)]
    if len(ser) >= 2:
        b["rates"].append((ser[-1][1] - ser[0][1]) / (ser[-1][0] - ser[0][0]))
    b["rounds"] += len(ser)
    at = dict(ser)
    for rd, v in ser:
        if rd + 6 in at:
            proj = v + 6 * rate_g
            if floor is not None and ((rate_g < 0 and proj < floor) or (rate_g > 0 and proj < floor)):
                proj = floor
            b["proj"].append((v, proj, at[rd + 6]))
rows = sorted(by.items(), key=lambda x: -x[1]["rounds"])
print(f"{'joker':16s} {'field':12s} {'rounds':>6s} {'inst':>4s} {'GROWTH':>7s} {'obs med':>8s} {'obs mean':>8s} {'ratio':>6s} {'n6':>4s} {'proj gain':>9s} {'real gain':>9s}")
for (n, p), b in rows:
    rg = G[n][p]
    med = st.median(b["rates"]) if b["rates"] else float("nan")
    mean = st.mean(b["rates"]) if b["rates"] else float("nan")
    pg = st.mean([pr - v for v, pr, a in b["proj"]]) if b["proj"] else float("nan")
    ag = st.mean([a - v for v, pr, a in b["proj"]]) if b["proj"] else float("nan")
    print(f"{n:16s} {p:12s} {b['rounds']:6d} {len(b['rates']):4d} {rg:7.3f} {med:8.3f} {mean:8.3f} {mean / rg if rg else 0:6.2f} {len(b['proj']):4d} {pg:9.3f} {ag:9.3f}")
