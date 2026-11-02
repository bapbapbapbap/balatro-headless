"""Checks the ordered mode's extra actions do what they say, in the real game.
Test-only state setup (adding a joker/consumable/voucher) uses the game's own create_card/add_to_deck.

usage: python3 tests/ordered_semantics.py <game source dir>"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harness')); sys.path.insert(0, os.path.join(HERE, '..', 'harness', 'vendor'))
from env import BalatroEnv
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/balatro/local/source')
ok = True
def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (("  " + info) if info else ""))
    ok &= bool(cond)

env = BalatroEnv(SRC, mode="ordered"); lua = env.lua
lua.execute('''
local H = ...
function H.t_add_joker(key) local c = create_card('Joker', G.jokers, nil, nil, nil, nil, key); c:add_to_deck(); G.jokers:emplace(c); H.settle(); return c end
function H.t_add_cons(key) local c = create_card('Tarot', G.consumeables, nil, nil, nil, nil, key); c:add_to_deck(); G.consumeables:emplace(c); H.settle(); return c end
-- record the order cards are played in, and the card Death copies
H.t_played = nil
local orig_play = G.FUNCS.play_cards_from_highlighted
G.FUNCS.play_cards_from_highlighted = function(e)
  local hl = {}
  for _, c in ipairs(G.hand.highlighted) do hl[#hl + 1] = c end
  table.sort(hl, function(a, b) return a.T.x < b.T.x end)
  H.t_played = {}
  for i, c in ipairs(hl) do H.t_played[i] = c.sort_id end
  return orig_play(e)
end
function H.t_ids(idxs) local o = {}; for i, p in ipairs(idxs) do o[i] = G.hand.cards[p].sort_id end; return o end
function H.t_card(pos) local c = G.hand.cards[pos]; return c.sort_id, c.base.value, c.base.suit end
function H.t_find(sort_id) for _, c in ipairs(G.playing_cards) do if c.sort_id == sort_id then return c end end end
''', env.H)
H = env.H
L = lambda t: [t[i] for i in range(1, len(t) + 1)] if t is not None else []

# 1. Lead card: Hanging Chad held -> plays with >=2 scoring cards come once per scoring card as lead
env.reset("LEADTEST1")
env.step(("select_blind", None, None, ()))
H.t_add_joker("j_hanging_chad")
acts = env.legal_actions()
leads = [a for a in acts if a[0] == "play" and len(a) == 6]
check("lead variants offered with Hanging Chad", len(leads) > 0, f"{len(leads)} lead plays")
pairs = {}
for a in leads: pairs.setdefault(a[3], []).append(a[4])
multi = [(cards, ls) for cards, ls in pairs.items() if len(ls) >= 2]
cards, ls = multi[0]
lead = ls[-1]                      # not the default (leftmost) lead
want_first = H.t_card(lead)[0]
selected_ids = set(L(H.t_ids(lua.table(*cards))))
env.step(("play", None, None, cards, lead, None))
played = L(H.t_played)
check("lead card scores first", played[0] == want_first, f"lead pos {lead} sort_id {want_first}, played order {played}")
check("same cards played", set(played) == selected_ids)
# no lead variants when order can't matter (no Hanging Chad, plain cards)
env.reset("LEADTEST2"); env.step(("select_blind", None, None, ()))
check("no lead variants when order can't matter", not any(len(a) == 6 for a in env.legal_actions() if a[0] == "play"))

# 2. Death: both directions offered; the chosen src is what the other card becomes
for which in (0, 1):
    env.reset("DEATHTEST"); env.step(("select_blind", None, None, ()))
    H.t_add_cons("c_death")
    deaths = [a for a in env.legal_actions() if a[0] == "use_consumable" and len(a) == 6]
    # pick a pair with different ranks
    pick = None
    for a in deaths:
        c1, c2 = a[3]
        if H.t_card(c1)[1] != H.t_card(c2)[1]: pick = a[3]; break
    variants = [a for a in deaths if a[3] == pick]
    check(f"Death offered both ways (case {which})", len(variants) == 2)
    a = variants[which]
    src = a[5]; other = [p for p in pick if p != src][0]
    sid_src, rank_src, suit_src = H.t_card(src); sid_other = H.t_card(other)[0]
    env.step(a)
    co = H.t_find(sid_other); cs = H.t_find(sid_src)
    check(f"Death copies the chosen card (case {which})", co.base.value == rank_src and co.base.suit == suit_src and cs.base.value == rank_src,
          f"src {rank_src} of {suit_src} -> other now {co.base.value} of {co.base.suit}")

# 3. Boss reroll: offered with Retcon and $10, changes the boss, costs $10; not offered without
env.reset("BOSSTEST")
check("no boss reroll without the voucher", not any(a[0] == "reroll_boss" for a in env.legal_actions()))
lua.execute("G.GAME.used_vouchers['v_retcon'] = true; G.GAME.dollars = 20")
acts = env.legal_actions()
check("boss reroll offered with Retcon and $20", any(a[0] == "reroll_boss" for a in acts))
before = lua.eval("G.GAME.round_resets.blind_choices.Boss"); money = lua.eval("G.GAME.dollars")
env.step(("reroll_boss", None, None, ()))
after = lua.eval("G.GAME.round_resets.blind_choices.Boss")
check("boss changed and $10 spent", after != before and lua.eval("G.GAME.dollars") == money - 10, f"{before} -> {after}, ${money} -> ${lua.eval('G.GAME.dollars')}")
check("still selecting a blind after reroll", env.state() == "BLIND_SELECT")
print("ALL PASS" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
