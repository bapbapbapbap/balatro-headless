"""Differential accuracy check: a reference sim and a candidate sim play the same seeds with the
same actions; after every action their complete game state must be identical.

  reference: the game as loaded, nothing stripped (closest to the real game)
  candidate: --fast (visual stripping) plus any optimization under test

"Complete game state" is everything game logic can read: G.STATE, all of G.GAME (RNG streams,
money, round state, hand levels, used vouchers, pools, ...), every card in every area (deck order,
hand order, jokers, consumables, discard, play, shop, vouchers, boosters, pack) with its full
ability table, edition, seal, debuff, facing and selection, the blind, the tags, and the legal
actions offered. Visual-only fields (drawn positions, velocities, juice, shaders) are left out.

Actions come from a seeded driver: mostly the fuzz greedy player (so runs get deep), otherwise
uniform over action types. Both sims must offer the same legal actions at every step.

usage: python3 diff_check.py --src DIR [--runs 50] [--mode controls|subsets] [--candidate fast]
"""
import argparse
import collections
import random
import sys
import time

from env import BalatroEnv
import fuzz

DUMP = r"""
local H = ...
local VISUAL = { T = true, VT = true, CT = true, velocity = true, role = true, alignment = true,
  children = true, states = true, juice = true, pinch = true, parallax = true, shadow_parrallax = true,
  layered_parallax = true, hover_tilt = true, tilt_var = true, ambient_tilt = true, FRAME = true,
  static_rotation = true, offset = true, click_offset = true, hover_offset = true, original_T = true,
  zoom = true, shadow_height = true, mid = true, container = true, parent = true, area = true,
  ID = true, created_on_pause = true, last_moved = true, last_aligned = true, NEW_ALIGNMENT = true,
  sprite_pos = true, temp_edition = true, debuff_sprite = true, hover_state = true, draw_steps = true,
  -- Checked by reading every use in the source (vanilla 1.0.1o): written and read only by drawing
  -- or animation code, never by game logic.
  STATIONARY = true,     -- Moveable:move internals (engine/moveable.lua)
  CALCING = true,        -- read only by the debug outline draw (engine/node.lua)
  handname_text = true,  -- the hand-name label's text (UI func hand_text_UI_set + its DynaText)
  -- Display strings written only by the HUD's UI callbacks (chip_UI_set, hand_chip_UI_set,
  -- hand_mult_UI_set, hand_chip_total_UI_set) and read only by HUD text. An audit of all 57 UI
  -- callbacks found no other game state they write during a run.
  chips_text = true, chip_text = true, mult_text = true, chip_total_text = true,
}

local function fmt(v)
  local t = type(v)
  if t == "number" then return string.format("%.17g", v) end
  if t == "string" then return string.format("%q", v) end
  return tostring(v)
end

-- Plain data, recursively, sorted keys. Objects (tables with a metatable) are written as their
-- class name plus identity where it matters, never walked, so visuals can't leak in.
local function plain(x, out, path, seen, depth)
  if type(x) ~= "table" then out[#out + 1] = path .. "=" .. fmt(x); return end
  if seen[x] then out[#out + 1] = path .. "=<ref " .. seen[x] .. ">"; return end
  if getmetatable(x) then
    out[#out + 1] = path .. "=<obj" .. (x.sort_id and (" " .. x.sort_id) or (x.key and (" " .. tostring(x.key))) or "") .. ">"
    return
  end
  if depth > 12 then out[#out + 1] = path .. "=<deep>"; return end
  seen[x] = path
  local keys = {}
  for k in pairs(x) do keys[#keys + 1] = k end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  for _, k in ipairs(keys) do
    if not VISUAL[k] and type(x[k]) ~= "function" and type(x[k]) ~= "userdata" then
      plain(x[k], out, path .. "." .. tostring(k), seen, depth + 1)
    end
  end
end

local function card(c, out, path)
  out[#out + 1] = path .. ".sort_id=" .. c.sort_id
  out[#out + 1] = path .. ".center=" .. tostring(c.config.center.key)
  out[#out + 1] = path .. ".card=" .. tostring(c.config.card and c.config.card.name)
  plain(c.base, out, path .. ".base", {}, 0)
  plain(c.ability, out, path .. ".ability", {}, 0)
  plain(c.edition, out, path .. ".edition", {}, 0)
  for _, k in ipairs({ "seal", "debuff", "facing", "highlighted", "cost", "sell_cost", "pinned",
                       "added_to_deck", "base_cost", "extra_cost", "playing_card", "getting_sliced",
                       "destroyed", "shattered", "removed", "rank", "bypass_discovery_center",
                       "discovered", "unlocked" }) do
    out[#out + 1] = path .. "." .. k .. "=" .. fmt(c[k])
  end
end

local AREAS = { "deck", "hand", "jokers", "consumeables", "discard", "play", "shop_jokers",
                "shop_vouchers", "shop_booster", "pack_cards", "vouchers" }

function H.state_dump()
  local out = { "STATE=" .. tostring(G.STATE), "STATE_COMPLETE=" .. tostring(G.STATE_COMPLETE) }
  local saved = {}
  for _, k in ipairs({ "blind", "selected_back", "tags" }) do saved[k] = G.GAME[k]; G.GAME[k] = nil end
  plain(G.GAME, out, "GAME", {}, 0)
  for k, v in pairs(saved) do G.GAME[k] = v end
  local b = G.GAME.blind
  if b then
    local bb = {}
    for k, v in pairs(b) do if type(v) ~= "table" and type(v) ~= "function" and not VISUAL[k] then bb[k] = v end end
    bb.key = b.config and b.config.blind and b.config.blind.key
    plain(bb, out, "BLIND", {}, 0)
  end
  for i, t in ipairs(G.GAME.tags or {}) do
    out[#out + 1] = "TAG" .. i .. "=" .. tostring(t.key)
    plain(t.ability, out, "TAG" .. i .. ".ability", {}, 0)
  end
  for _, a in ipairs(AREAS) do
    local area = G[a]
    if area and area.cards then
      out[#out + 1] = a .. ".n=" .. #area.cards .. " limit=" .. tostring(area.config.card_limit)
        .. " hl=" .. #(area.highlighted or {}) .. " sort=" .. tostring(area.config.sort)
      for i, c in ipairs(area.cards) do card(c, out, a .. "[" .. i .. "]") end
    else
      out[#out + 1] = a .. "=nil"
    end
  end
  local ids = {}
  for _, c in ipairs(G.playing_cards or {}) do ids[#ids + 1] = c.sort_id end
  out[#out + 1] = "playing_cards=" .. table.concat(ids, ",")
  return table.concat(out, "\n")
end
"""


def make(src, fast, mode):
    env = BalatroEnv(src, fast=fast, mode=mode)
    env.lua.execute(fuzz.LUA, env.H)
    env.lua.execute(DUMP, env.H)
    return env


def first_diff(a, b, n=8):
    la, lb = a.split("\n"), b.split("\n")
    sa, sb = set(la), set(lb)
    only_a = [x for x in la if x not in sb][:n]
    only_b = [x for x in lb if x not in sa][:n]
    return only_a, only_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--mode", choices=["controls", "subsets"], default="subsets")
    ap.add_argument("--rng", type=int, default=1)
    ap.add_argument("--greedy", type=float, default=0.85)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--candidate", default="fast", help="fast | fast+<opt>,<opt> (see OPTIMIZATIONS)")
    ap.add_argument("--encode", action="store_true",
                    help="run the learner's encoder (encode.lua) before every comparison, candidate only: "
                         "proves encoding never changes the game")
    args = ap.parse_args()

    ref = make(args.src, False, args.mode)
    cand = make(args.src, True, args.mode)
    for opt in args.candidate.split("+")[1:]:
        for o in opt.split(","):
            cand.lua.execute(OPTIMIZATIONS[o], cand.H)
    if args.encode:
        import os
        cand.lua.execute(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode.lua")).read(), cand.H)
    rng = random.Random(args.rng)
    t0, steps, bad, states = time.time(), 0, [], collections.Counter()
    for r in range(args.runs):
        seed = "".join(rng.choice("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8))
        ref.reset(seed)
        cand.reset(seed)
        plan, ok = [], True
        while ok and not ref.done and ref.steps < args.max_steps:
            if args.encode:
                cand.H.encode()
            da, db = ref.H.state_dump(), cand.H.state_dump()
            if da != db:
                a, b = first_diff(da, db)
                bad.append((seed, ref.steps, ref.state()))
                print(f"{seed} step {ref.steps} ({ref.state()}): STATE DIFFERS\n  ref only:  {a}\n  cand only: {b}")
                ok = False
                break
            acts = ref.legal_actions()
            if acts != cand.legal_actions():
                bad.append((seed, ref.steps, ref.state()))
                ca = cand.legal_actions()
                print(f"{seed} step {ref.steps} ({ref.state()}): LEGAL ACTIONS DIFFER\n"
                      f"  ref only:  {sorted(set(acts) - set(ca), key=str)[:8]}\n"
                      f"  cand only: {sorted(set(ca) - set(acts), key=str)[:8]}")
                ok = False
                break
            states[ref.state()] += 1
            a = None
            if plan:
                a = plan.pop(0)
            elif ref.state() == "SELECTING_HAND" and rng.random() < args.greedy:
                plan = fuzz.greedy_plan(ref)
                a = plan.pop(0) if plan else None
            if a is None and ref.state() == "BLIND_SELECT" and rng.random() < args.greedy:
                a = ("select_blind", None, None, ())
            if a is None or a not in set(acts):
                plan = []
                by = collections.defaultdict(list)
                for x in acts:
                    by[x[0]].append(x)
                a = rng.choice(by[rng.choice(sorted(by))])
            ref.step(a)
            cand.step(a)
            steps += 1
        if ok:
            da, db = ref.H.state_dump(), cand.H.state_dump()
            if da != db:
                bad.append((seed, ref.steps, "final"))
                print(f"{seed} final state differs: {first_diff(da, db)}")
        o = ref.observe()
        print(f"{seed}  ante {o['ante']}  {o['state']:10s} {ref.steps:4d} steps  {'OK' if ok else 'DIFF'}", flush=True)
    el = time.time() - t0
    print(f"\ncandidate '{args.candidate}' vs reference, mode {args.mode}: {args.runs} runs, {steps} steps "
          f"compared, {len(bad)} runs differ  ({el:.0f}s)")
    print("decisions compared by state:", dict(states.most_common()))
    print("PASS" if not bad else "FAIL")
    sys.exit(0 if not bad else 1)


# Candidate optimizations, each loaded into the candidate sim only. Kept only if diff_check passes.
OPTIMIZATIONS = {}

if __name__ == "__main__":
    main()
