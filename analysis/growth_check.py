"""Usage: python3 analysis/growth_check.py <repo> <game source> <out.json> [stakes, e.g. 8 | comma-separated games .jsonl.gz]
(docs/hypotheses-v6.md, 2026-11-27 15:40 growth sections; report: analysis/growth_report.py <out.json>)

phi_build's growth projection vs what scaling jokers actually did in the bot's own won routes.
At the first move of every round (SELECTING_HAND after a blind is selected), every held joker in GROWTH: its
field value. Per joker instance (sort_id): projected value 6 rounds on (v + rate*6, floored) vs the actual value
6 rounds later if still held."""
import json, os, re, sys
from multiprocessing import Pool
ROOT, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
STAKES = sys.argv[4] if len(sys.argv) > 4 else ""
sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "harness"))
from sim import Sim

lua_src = open(os.path.join(ROOT, "harness", "build_phi.lua")).read()
block = lua_src[lua_src.index("local GROWTH = {"):lua_src.index("}\n", lua_src.index("local GROWTH = {")) + 1]
GROWTH = {}
for name, body in re.findall(r'\["([^"]+)"\]\s*=\s*\{([^}]*)\}', block):
    d = {}
    for k, v in re.findall(r'(?:\["([^"]+)"\]|(\w+))\s*=\s*(-?[\d.]+)', body) and [(a or b, c) for a, b, c in re.findall(r'(?:\["([^"]+)"\]|(\w+))\s*=\s*(-?[\d.]+)', body)]:
        d[k] = float(v)
    GROWTH[name] = d

READ = r"""local names = ...
local out = {}
for _, c in ipairs(G.jokers.cards) do
  local paths = names[c.ability.name]
  if paths then
    for p in string.gmatch(paths, '[^,]+') do
      local a, b = p:match('^([^.]+)%.(.+)$')
      local v
      if a then v = c.ability[a] and c.ability[a][b] else v = c.ability[p] end
      if type(v) == 'number' then out[#out + 1] = c.ability.name .. '|' .. c.sort_id .. '|' .. p .. '|' .. v end
    end
  end
end
return table.concat(out, ';')"""


def one(ir):
    i, r = ir
    s = Sim(SRC, mode="ordered-capped", enc=2)
    s.reset(r["seed"], r["back"], r["stake"])
    names = s.env.lua.table_from({k: ",".join(p for p in v if p != "floor") for k, v in GROWTH.items()})
    rows, last_round = [], None
    for act in r["acts"]:
        if s.H.state_name() == "SELECTING_HAND":
            rd = int(s.env.lua.eval("G.GAME.round"))
            if rd != last_round:
                last_round = rd
                got = s.env.lua.execute(READ, names)
                for x in (got.split(";") if got else []):
                    n, sid, p, v = x.split("|")
                    rows.append((rd, n, int(sid), p, float(v)))
        s.observe()
        _, _, done = s.step(act)
        if done:
            break
    return f"{i}/{r['seed']}/{r['back']}/{r['stake']}", rows


if __name__ == "__main__":
    if STAKES.endswith(".gz") or "," in STAKES and STAKES.split(",")[0].endswith(".gz"):
        import gzip
        rs = [json.loads(l) for f in STAKES.split(",") for l in gzip.open(f, "rt")]
    else:
        rs = [json.loads(l) for l in open(os.path.join(ROOT, "wins", "routes.jsonl")) if '"ordered-capped"' in l]
        if STAKES:
            rs = [r for r in rs if str(r["stake"]) in STAKES.split(",")]
    with Pool(int(os.environ.get("W", "2"))) as p:
        res = dict(p.map(one, list(enumerate(rs)), chunksize=4))
    json.dump({"growth": GROWTH, "routes": res}, open(OUT, "w"))
    print(len(res), "routes")
