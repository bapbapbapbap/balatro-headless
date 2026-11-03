"""Checks the hand-focus rule (env.lua, "hand focus") in the real game, for every poker hand.

usage: python3 tests/focus_check.py <game source dir> [--seeds 3] [--modes ordered,controls] [--hands all]

Plays seeded games with a random agent (leaning toward plays and discards so it gets through
rounds) under each focus hand and checks, at every decision:
  1. Listing the moves changes nothing in the game (hand/joker order, selection, sort, and the
     blind's `triggered` flag, which the blind check touches).
  2. Away from a play decision, or with no focus, the move list is exactly the unfocused one.
  3. At a play decision: every non-play move is still offered, in the same order; if some play of
     the focus hand is available, the plays offered are exactly those, in the same order;
     otherwise every play is offered. (Controls mode: the one play move is offered iff the
     selection is the focus hand or no selection could be.)
  4. Independent of the rule's own code: after the bot makes a play the rule kept as the focus
     hand, the game's own record of the hand just played (G.GAME.last_hand_played) is that hand.
  5. Determinism: one game replayed with the same moves ends in the same state.
Prints ALL PASS or the failures.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
sys.path.insert(0, os.path.join(HERE, "..", "harness", "vendor"))
from env import HANDS, BalatroEnv  # noqa: E402

LUA = r"""
local H = ...
local function key(a)
  return table.concat({ a.type, tostring(a.i), tostring(a.j), table.concat(a.cards or {}, ","),
                        tostring(a.lead), tostring(a.src) }, "|")
end
local function cards_at(idxs) local o = {}; for _, i in ipairs(idxs or {}) do o[#o + 1] = G.hand.cards[i] end; return o end
-- Returns "ok" or a failure message, plus whether the focus hand was available.
function H.t_focus_check()
  local b = G.GAME.blind
  local fp, trig = H.fingerprint(), b and b.triggered
  local full, got = H.unfocused_legal_actions(), H.legal_actions()
  if H.fingerprint() ~= fp then return "listing moves changed the game", false end
  if b and b.triggered ~= trig then return "listing moves changed blind.triggered", false end
  local expect, avail = {}, false
  local selecting = H.focus_hand and G.STATE == G.STATES.SELECTING_HAND
  if selecting and H.mode == "controls" then
    local n, lim = #G.hand.cards, math.min(5, G.hand.config.highlighted_limit, #G.hand.cards)
    -- brute force over every selection, independently of focus_reachable()
    local function rec(start, cur)
      if avail then return end
      if #cur >= 1 then
        local forced_ok = true
        for i, c in ipairs(G.hand.cards) do
          if c.ability.forced_selection then
            local inc = false
            for _, j in ipairs(cur) do if j == i then inc = true end end
            forced_ok = forced_ok and inc
          end
        end
        if forced_ok and H.is_focus_play(cards_at(cur)) then avail = true; return end
      end
      if #cur == lim then return end
      for i = start, n do cur[#cur + 1] = i; rec(i + 1, cur); cur[#cur] = nil end
    end
    rec(1, {})
    local sel_ok = H.is_focus_play(G.hand.highlighted)
    for _, a in ipairs(full) do
      if a.type ~= "play" or sel_ok or not avail then expect[#expect + 1] = key(a) end
    end
  elseif selecting then
    for _, a in ipairs(full) do
      if a.type == "play" and H.is_focus_play(cards_at(a.cards)) then avail = true end
    end
    for _, a in ipairs(full) do
      if a.type ~= "play" or not avail or H.is_focus_play(cards_at(a.cards)) then expect[#expect + 1] = key(a) end
    end
  else
    for _, a in ipairs(full) do expect[#expect + 1] = key(a) end
  end
  if #expect ~= #got then return ("expected %d moves, got %d"):format(#expect, #got), avail end
  for k, a in ipairs(got) do
    if key(a) ~= expect[k] then return ("move %d: expected %s, got %s"):format(k, expect[k], key(a)), avail end
  end
  return "ok", avail
end
function H.t_last_hand() return G.GAME.last_hand_played end
function H.t_selection_is_focus() return H.is_focus_play(G.hand.highlighted) end
"""


def play(env, seed, rng, max_steps, fails, tag):
    """Random game with every check; returns the moves taken and the final fingerprint."""
    H = env.H
    env.reset(seed, "b_red", 1)
    moves, focus_plays, avail_n = [], 0, 0
    while not env.done and env.steps < max_steps:
        res, avail = H.t_focus_check()
        if res != "ok":
            fails.append(f"{tag} {seed} step {env.steps} ({env.state()}): {res}")
            return moves, None, focus_plays, avail_n
        avail_n += bool(avail)
        acts = env.legal_actions()
        plays = [a for a in acts if a[0] == "play"]
        discards = [a for a in acts if a[0] == "discard"]
        toggles = [a for a in acts if a[0] == "toggle"]
        r = rng.random()
        if plays and r < 0.5:
            a = rng.choice(plays)
        elif discards and r < 0.7:
            a = rng.choice(discards)
        elif toggles and r < 0.95:
            a = rng.choice(toggles)
        else:
            a = rng.choice([x for x in acts if x[0] not in ("move_joker", "move_hand")] or acts)
        check_hand = False
        if a[0] == "play" and env.focus:
            if env.H.mode == "controls":
                check_hand = bool(H.t_selection_is_focus())
            else:
                # with the focus hand available, every play offered is one; is this one?
                check_hand = bool(avail)
        env.step(a)
        moves.append(a)
        if check_hand:
            focus_plays += 1
            if H.t_last_hand() != env.focus:
                fails.append(f"{tag} {seed} step {env.steps}: kept as {env.focus}, game scored {H.t_last_hand()}")
    return moves, H.fingerprint() + f"|{env.observe()['chips']}|{env.observe()['money']}", focus_plays, avail_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="ordered,controls")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--hands", default="all", help='"all", or a comma list of hands (no focus is always checked too)')
    a = ap.parse_args()
    hands = HANDS if a.hands == "all" else tuple(h.strip() for h in a.hands.split(","))
    for h in hands:
        if h not in HANDS:
            raise SystemExit(f"unknown hand {h!r}; one of: {', '.join(HANDS)}")
    fails = []
    for mode in a.modes.split(","):
        env = BalatroEnv(a.src, mode=mode)
        env.lua.execute(LUA, env.H)
        for focus in (None, *hands):
            env.focus = focus
            env.H.focus_hand = focus
            tp = ta = 0
            for k in range(a.seeds):
                seed = f"FOCUS{k:03d}"
                rng = random.Random(f"{mode}-{focus}-{k}")
                moves, fp, tp_, ta_ = play(env, seed, rng, a.max_steps, fails, f"[{mode} {focus}]")
                tp, ta = tp + tp_, ta + ta_
                if fp is not None and k == 0:      # determinism: same moves, same end
                    env.reset(seed, "b_red", 1)
                    for m in moves:
                        env.step(m)
                    fp2 = env.H.fingerprint() + f"|{env.observe()['chips']}|{env.observe()['money']}"
                    if fp2 != fp:
                        fails.append(f"[{mode} {focus}] {seed}: replay ended in a different state")
            print(f"{mode:9s} {str(focus):16s} focus available at {ta:4d} decisions, {tp:4d} focus plays checked", flush=True)
    for f in fails[:30]:
        print("FAIL", f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
