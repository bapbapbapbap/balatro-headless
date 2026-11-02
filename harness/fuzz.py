"""Fuzzes env.lua: plays many runs and fails loudly if an offered action crashes, never settles,
or leaves the game unchanged, or if an observation leaks anything a player can't see.
Test driver only; its hand choice is a crutch to reach shops and packs, not a strategy.

  - SELECTING_HAND: with probability --greedy, plays the subset with the best base score
    (hand-type chips + scoring cards' chips) x mult, ignoring jokers. In controls mode that's
    a sequence of toggles then Play. Otherwise random.
  - BLIND_SELECT: with probability --greedy, takes the blind.
  - Everywhere else: picks an action type uniformly, then an action of that type uniformly.

Leak checks on every observation: no seed or RNG state anywhere in it; every face-down hand
card or joker in the game shows up as face_down with nothing else; deck lists are sets (sorted),
not draw order.

usage: python3 fuzz.py --src DIR [--runs 30] [--greedy 0.85] [--mode controls|subsets]
"""
import argparse
import collections
import json
import random
import time
import traceback

from env import BalatroEnv

LUA = r"""
local H = ...
function H.best_subset()
  local n, best, bs = #G.hand.cards, nil, -1
  local function rec(start, cur)
    if #cur >= 1 then
      local cards = {}
      for _, i in ipairs(cur) do cards[#cards + 1] = G.hand.cards[i] end
      local text, _, _, scoring = G.FUNCS.get_poker_hand_info(cards)
      local h = G.GAME.hands[text]
      if h then
        local chips = h.chips
        for _, c in ipairs(scoring) do if not c.debuff then chips = chips + c.base.nominal end end
        if chips * h.mult > bs then bs = chips * h.mult; best = { unpack(cur) } end
      end
    end
    if #cur == 5 then return end
    for i = start, n do cur[#cur + 1] = i; rec(i + 1, cur); cur[#cur] = nil end
  end
  rec(1, {})
  return best
end
function H.hidden_counts()
  local h, j = 0, 0
  for _, c in ipairs(G.hand.cards) do if c.facing == "back" then h = h + 1 end end
  for _, c in ipairs(G.jokers.cards) do if c.facing == "back" then j = j + 1 end end
  return h, j
end
function H.selected()
  local out = {}
  for i, c in ipairs(G.hand.cards) do if c.highlighted then out[#out + 1] = i end end
  return out
end
"""

LEAK_KEYS = {"seed", "hashed_seed", "pseudorandom", "pools", "rng"}


def check_leaks(env, o):
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k in LEAK_KEYS:
                    raise AssertionError(f"observation leaks '{k}'")
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(o)
    hh, hj = env.H.hidden_counts()
    for name, lst, n in (("hand", o["hand"], hh), ("jokers", o["jokers"], hj)):
        masked = [c for c in lst if c.get("face_down")]
        if len(masked) != n:
            raise AssertionError(f"{n} face-down {name} cards in game, {len(masked)} masked in observation")
        for c in masked:
            if set(c) - {"face_down", "highlighted"}:
                raise AssertionError(f"face-down {name} card shows {sorted(set(c) - {'face_down', 'highlighted'})}")
    for name in ("deck", "full_deck"):
        keys = [json.dumps([c.get("suit"), c.get("id"), c.get("enhancement"), c.get("edition"), c.get("seal")])
                for c in o[name]]
        if any(k is None for k in keys):
            raise AssertionError(name)
    return True


def fingerprint(env):
    o = env.observe()
    return (env.H.fingerprint(), o["state"], o["money"], o["chips"], len(o["hand"]), len(o["deck"]),
            len(o["jokers"]), len(o["consumables"]), len(o.get("shop") or []), len(o.get("pack") or []),
            o["hands_left"], o["discards_left"], o["reroll_cost"], tuple(o["tags"]),
            tuple(sorted(c["key"] for c in o["consumables"])), o.get("pack_choices"), o["blind_on_deck"])


def lua_list(t):
    return [t[i] for i in range(1, len(t) + 1)] if t is not None else []


def greedy_plan(env):
    best = env.H.best_subset()
    if best is None:
        return []
    want = set(lua_list(best))
    if env.H.mode == "subsets":
        return [("play", None, None, tuple(sorted(want)))]
    have = set(lua_list(env.H.selected()))
    plan = [("toggle", i, None, ()) for i in sorted(have - want)]   # deselect first (limit is 5)
    plan += [("toggle", i, None, ()) for i in sorted(want - have)]
    return plan + [("play", None, None, ())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--greedy", type=float, default=0.85)
    ap.add_argument("--rng", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--mode", choices=["controls", "subsets"], default="controls")
    args = ap.parse_args()

    env = BalatroEnv(args.src, mode=args.mode)
    env.lua.execute(LUA, env.H)
    rng = random.Random(args.rng)
    taken, noop, failures, antes = collections.Counter(), collections.Counter(), [], collections.Counter()
    t0, steps, checks = time.time(), 0, 0
    for r in range(args.runs):
        seed = "".join(rng.choice("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8))
        o = env.reset(seed)
        plan = []
        try:
            while not env.done and env.steps < args.max_steps:
                checks += check_leaks(env, o)
                acts = env.legal_actions()
                if not acts:
                    raise RuntimeError(f"no legal actions in {env.state()}")
                legal = set(acts)
                a = None
                if plan:
                    a = plan.pop(0)
                    if a not in legal:
                        raise AssertionError(f"planned {a} not legal in {env.state()}")
                elif env.state() == "SELECTING_HAND" and rng.random() < args.greedy:
                    plan = greedy_plan(env)
                    a = plan.pop(0) if plan else None
                    if a is not None and a not in legal:
                        raise AssertionError(f"planned {a} not legal in {env.state()}")
                if a is None and env.state() == "BLIND_SELECT" and rng.random() < args.greedy:
                    a = ("select_blind", None, None, ())
                if a is None:
                    by = collections.defaultdict(list)
                    for x in acts:
                        by[x[0]].append(x)
                    a = rng.choice(by[rng.choice(sorted(by))])
                before = fingerprint(env)
                o = env.step(a)
                taken[a[0]] += 1
                if fingerprint(env) == before:
                    noop[a[0]] += 1
                    if noop[a[0]] <= 3:
                        print(f"  no-op: {seed} step {env.steps} {a} in {before[1]}")
            checks += check_leaks(env, o)
            antes[o["ante"]] += 1
            print(f"{seed}  ante {o['ante']}  {'WON ' if env.won else ''}{o['state']:10s} {env.steps} steps")
        except Exception as ex:
            failures.append((seed, env.steps, repr(ex)))
            print(f"{seed}  FAILED at step {env.steps}: {ex}")
            traceback.print_exc(limit=3)
        steps += env.steps
    el = time.time() - t0
    print(f"\nmode {args.mode}: {args.runs} runs, {steps} steps, {el:.0f}s ({el / max(steps, 1) * 1000:.0f} ms/step)")
    print("final antes:", dict(sorted(antes.items())))
    print("actions taken:", dict(taken.most_common()))
    print("no-op actions:", dict(noop) or "none")
    print(f"leak checks passed: {checks}")
    print("failures:", failures or "none")


if __name__ == "__main__":
    main()
