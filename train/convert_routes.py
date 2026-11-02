"""Translates saved games from the v1 action list (subsets) into the v2 one (ordered), so a v2 model
can learn from wins found by v1. The game is replayed move for move: at each step the v1 move is
read from the v1 list and its exact equivalent is taken from the v2 list (a play keeps the lead it
had, the leftmost scoring card; Death keeps copying the rightmost card). A converted game is kept
only if it ends the same way, and is written with its v2 actions, v2 final-state hash and the v1
record it came from.

    python3 convert_routes.py --src DIR wins.jsonl [more.jsonl ...] --out wins_v2.jsonl [--wins-only]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import Sim  # noqa: E402

FIND = r"""
local H = ...
local function same(x, y)
  x, y = x or {}, y or {}
  if #x ~= #y then return false end
  for i = 1, #x do if x[i] ~= y[i] then return false end end
  return true
end
-- index (1-based) in the current (ordered) list of the action equivalent to v1 action a
function H.find_v2(a)
  local lead = nil
  if a.type == "play" then lead = H.scoring_positions(a.cards)[1] end
  for k, b in ipairs(H._encoded_actions) do
    if b.type == a.type and b.i == a.i and b.j == a.j and same(b.cards, a.cards) then
      if b.lead then
        if b.lead == lead then return k end
      elseif b.src then
        if b.src == a.cards[#a.cards] then return k end
      else
        return k
      end
    end
  end
  return nil
end
function H.v1_action(i) return H._encoded_actions[i] end
"""


def convert(sim, r):
    H = sim.H
    sim.reset(r["seed"], r["back"], r["stake"])
    for k, a1 in enumerate(r["acts"]):
        H.mode = "subsets"
        H.encode()
        if H.action_desc(a1 + 1) != r["desc"][k]:
            return None, f"step {k}: v1 move '{r['desc'][k]}' not where it was saved"
        a = H.v1_action(a1 + 1)
        H.mode = "ordered"
        sim.observe()
        j = H.find_v2(a)
        if j is None:
            return None, f"step {k}: no v2 equivalent of '{r['desc'][k]}'"
        _, _, done = sim.step(j - 1)
        if done and k != len(r["acts"]) - 1:
            return None, f"game ended at step {k}, {len(r['acts']) - k - 1} moves early"
    if sim.progress != r["progress"]:
        return None, f"ended {sim.progress}, saved {r['progress']}"
    out = {k: v for k, v in r.items() if k not in ("acts", "desc", "final")}
    out.update(acts=list(sim.route), desc=list(sim.desc), final=sim.final_hash(), actions="ordered", enc=2,
               converted_from={"actions": r.get("actions", "subsets"), "enc": r.get("enc", 1), "final": r["final"]})
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wins-only", action="store_true")
    args = ap.parse_args()
    sim = Sim(args.src, mode="ordered", enc=2)
    sim.env.lua.execute(FIND, sim.H)
    done, bad, seen = 0, 0, set()
    with open(args.out, "w") as fo:
        for f in args.files:
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if args.wins_only and not r["progress"]["won"]:
                    continue
                if r.get("actions", "subsets") != "subsets" or r.get("enc", 1) != 1:
                    continue
                key = (r["seed"], tuple(r["acts"]))
                if key in seen:
                    continue
                seen.add(key)
                out, why = convert(sim, r)
                if out is None:
                    bad += 1
                    print(f"NOT CONVERTED seed {r['seed']} ({r.get('mode')} round {r.get('iteration')}): {why}")
                    continue
                fo.write(json.dumps(out) + "\n")
                done += 1
    print(f"{done} games converted, {bad} not")
    raise SystemExit(0 if done and not bad else 1)


if __name__ == "__main__":
    main()
