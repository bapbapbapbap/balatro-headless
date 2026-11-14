"""Lockstep twin: the same seed and the same actions in the game (harness) and in Balatro Lab's simulator,
compared at every decision, until the first divergence.

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e):
balatro.run.Run (seed-exact mode, `game_seed=`), its balatro.agent.Agent callback interface, and
balatro.bots.estimate (only to choose hands). Nothing of Balatro Lab is copied here.

Balatro Lab's run asks its agent for each decision (skip a blind? round action, shop action, pack action). Here
that agent runs in a thread and blocks; the harness driver decides, plays the action in the game, and hands the
same action to the twin. Before each handover both states are compared: money, ante, jokers, consumables, tags,
hand levels, deck size, and per decision type the hand, the blind, the shop (cards, editions, prices, packs,
voucher, reroll cost) or the open pack. A seed stops at its first difference (after that the two runs are not the
same run), which is logged with both sides.

The driver's policy only has to cover the game: hands chosen as in validate_scoring.py; skips some Small/Big
blinds; buys jokers and planets, sometimes vouchers, opens Buffoon and Celestial packs, sometimes rerolls; uses
planets. It avoids what Balatro Lab says it does not make seed-exact (Standard packs, Arcana/Spectral targets).

    python3 twin.py --lab <balatro-lab clone> --src <game source> --seeds 50 --out twin.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import traceback
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))
sys.path.insert(0, HERE)

from env import BalatroEnv  # noqa: E402  (the harness loads lupa, vendor/ first: LuaJIT 2.0.5)

LUAJIT = sys.modules["lupa.luajit20"].LuaRuntime().eval("jit.version")
if LUAJIT != "LuaJIT 2.0.5" and os.environ.get("ALLOW_STOCK_LUAJIT") != "1":
    # Another LuaJIT orders pairs() differently: To Do List's and the Orbital Tag's hands then differ from the game's
    raise SystemExit(f"{LUAJIT}, not the game's LuaJIT 2.0.5: build or link harness/vendor (README), "
                     "or set ALLOW_STOCK_LUAJIT=1 to accept those two differences")
from state import LAB, ENHANCEMENT, edition_of, joker_state, lab_run, load_lab, lua_list  # noqa: E402,F401

TIMEOUT = 60
# Jokers the driver doesn't buy: chance effects Balatro Lab rolls with its own randomness (its docs list in-round
# chances as not seed-exact), and To Do List, whose hand is drawn in pairs() order (exact only on LuaJIT 2.0.5,
# which a stock lupa isn't). Orbital Tag has the same pairs() issue; skips that would take it are avoided too.
AVOID = {"Business Card", "Reserved Parking", "Hallucination", "8 Ball", "Misprint", "Bloodstone", "Space Joker",
         "Lucky Cat", "Gros Michel", "Cavendish", "To Do List", "Vagabond", "Superposition", "Seance",
         "Cartomancer", "Sixth Sense", "Gift Card", "Matador", "Oops! All 6s", "Wheel of Fortune", "Mr. Bones",
         "Invisible Joker", "Certificate", "Marble Joker", "DNA", "Riff-Raff", "Burnt Joker", "Midas Mask",
         "Vampire", "Hiker", "Rough Gem", "Trading Card", "Ancient Joker",
         "The Idol", "Mail-In Rebate", "Castle", "Glass Joker", "Bull", "Bootstraps"}


# ---- the twin ---------------------------------------------------------------------------------------------
class Twin:
    """Balatro Lab's Run in a thread; each decision it asks for arrives as (kind, run, extra) on `asks`."""

    def __init__(self, seed):
        from balatro.agent import Agent
        from balatro.run import Run
        twin = self
        self.asks, self.answers = queue.Queue(), queue.Queue()

        class Relay(Agent):
            def _ask(self, kind, run, extra=None):
                twin.asks.put((kind, run, extra))
                return twin.answers.get(timeout=TIMEOUT)

            def skip_blind(self, run, blind, tag):
                return self._ask("blind", run, (blind, tag))

            def reroll_boss(self, run, boss):
                return False

            def round_action(self, run):
                return self._ask("round", run)

            def shop_action(self, run, shop):
                return self._ask("shop", run, shop)

            def pack_action(self, run, pack):
                return self._ask("pack", run, pack)

        self.run = Run(Relay(), game_seed=seed, deck="Red Deck")

        def go():
            try:
                res = self.run.play()
                self.asks.put(("end", self.run, res))
            except Exception:
                self.asks.put(("crash", self.run, traceback.format_exc(limit=4)))

        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()

    def next(self):
        return self.asks.get(timeout=TIMEOUT)

    def answer(self, action):
        self.answers.put(action)


# ---- what both sides show --------------------------------------------------------------------------------
def sig_card(rank, suit, enh, ed, seal):
    return f"{rank}{suit[0]}{'/' + enh if enh else ''}{'/' + ed if ed else ''}{'/' + seal if seal else ''}"


def g_card(c):
    return sig_card(int(c.base.id), str(c.base.suit), ENHANCEMENT.get(c.config.center.key), edition_of(c),
                    str(c.seal) if c.seal else None)


def l_card(c):
    return sig_card(c.rank, c.suit, c.enhancement, c.edition, c.seal)


def g_name(c):
    center = c.config.center
    s = str(center.set)
    if s == "Booster":
        return str(center.name)
    if s in ("Default", "Enhanced"):
        return g_card(c)
    return str(LOC[s][str(center.key)].name)


LOC = {}


def game_view(G, kind):
    v = {"money": int(G.GAME.dollars), "ante": int(G.GAME.round_resets.ante),
         "jokers": [(g_name(c), edition_of(c)) for c in lua_list(G.jokers.cards)],
         "consumables": [g_name(c) for c in lua_list(G.consumeables.cards)],
         "levels": {str(k): int(h.level) for k, h in G.GAME.hands.items() if int(h.level) != 1},
         "deck_size": len(lua_list(G.playing_cards))}
    v["joker_state"] = [norm(joker_state(g_name(c), c.ability, G)) for c in lua_list(G.jokers.cards)]
    if kind == "blind":
        rr = G.GAME.round_resets
        on = str(G.GAME.blind_on_deck)
        v["tag"] = str(LOC["Tag"][str(rr.blind_tags[on])].name) if rr.blind_tags[on] else None
        v["boss"] = str(LOC["Blind"][str(rr.blind_choices.Boss)].name)
    if kind == "round":
        cr = G.GAME.current_round
        v.update(hand=sorted(g_card(c) for c in lua_list(G.hand.cards)), hands_left=int(cr.hands_left),
                 discards_left=int(cr.discards_left), blind_chips=int(G.GAME.blind.chips),
                 score=int(G.GAME.chips),
                 next=[g_card(c) for c in reversed(lua_list(G.deck.cards))][:12])
    if kind == "shop":
        v.update(cards=[(g_name(c), edition_of(c), int(c.cost)) for c in lua_list(G.shop_jokers.cards)],
                 packs=[(g_name(c), int(c.cost)) for c in lua_list(G.shop_booster.cards)],
                 vouchers=[(g_name(c), int(c.cost)) for c in lua_list(G.shop_vouchers.cards)],
                 reroll_cost=int(G.GAME.current_round.reroll_cost))
    if kind == "pack":
        v["pack"] = [g_name(c) if str(c.config.center.set) not in ("Default", "Enhanced") else g_card(c)
                     for c in lua_list(G.pack_cards.cards)]
    return v


def lab_view(run, kind, extra):
    from balatro.cards import Card
    from balatro.shop import cost_of
    v = {"money": int(run.money), "ante": int(run.ante),
         "jokers": [(j.name, j.edition) for j in run.jokers],
         "consumables": [c.name for c in run.consumables],
         "levels": {k: int(n) for k, n in run.hand_levels.items() if int(n) != 1},
         "deck_size": len(run.deck)}
    v["joker_state"] = [norm({k: getattr(j, k, None) for k in JSTATE.get(j.name, ())}) for j in run.jokers]
    if kind == "blind":
        blind, tag = extra
        v["tag"] = tag.name if tag else None
        v["boss"] = run.ante_boss.name if run.ante_boss else None
    if kind == "round":
        v.update(hand=sorted(l_card(c) for c in run.hand), hands_left=int(run.hands_left),
                 discards_left=int(run.discards_left), blind_chips=int(run.blind.chips), score=int(run.round_score),
                 next=[l_card(c) for c in reversed(run.draw_pile)][:12])
    if kind == "shop":
        shop = extra
        v.update(cards=[(o.name if not isinstance(o.item, Card) else l_card(o.item),
                         getattr(o.item, "edition", None), cost_of(run, o)) for o in shop.cards],
                 packs=[(o.name, cost_of(run, o)) for o in shop.packs],
                 vouchers=[(o.name, cost_of(run, o)) for o in shop.vouchers],
                 reroll_cost=int(shop.reroll_cost))
    if kind == "pack":
        v["pack"] = [x.name if not isinstance(x, Card) else l_card(x) for x in extra.cards]
    return v


def norm(d):
    return {k: (round(float(v), 6) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
            for k, v in d.items() if v is not None}


class _Probe:
    """Records which keys joker_state reads, so the twin side asks its jokers for the same ones."""
    def __getattr__(self, k):
        return self
    def __getitem__(self, k):
        return self
    def __int__(self):
        return 0
    def __float__(self):
        return 0.0
    def __str__(self):
        return ""


JSTATE = {'Ice Cream': ('chips',), 'Popcorn': ('mult',), 'Red Card': ('mult',), 'Ride the Bus': ('value',), 'Runner': ('value',), 'Square Joker': ('value',), 'Green Joker': ('value',), 'To Do List': ('target',), 'Mail-In Rebate': ('rank',), 'Obelisk': ('xmult',), 'Ancient Joker': ('suit',), 'Campfire': ('xmult',), 'Wee Joker': ('chips',), 'Hit the Road': ('xmult',), 'Invisible Joker': ('rounds',), 'Canio': ('xmult',), 'Yorick': ('xmult', 'discarded'), 'Turtle Bean': ('hand_size_bonus',), 'The Idol': ('rank', 'suit'), 'Lucky Cat': ('xmult',), 'Seltzer': ('hands',), 'Vampire': ('xmult',), 'Spare Trousers': ('mult',), 'Loyalty Card': ('since',), 'Constellation': ('xmult',), 'Hologram': ('xmult',), 'Glass Joker': ('xmult',), 'Flash Card': ('mult',), 'Ramen': ('xmult',), 'Castle': ('chips', 'suit'), 'Ceremonial Dagger': ('mult',), 'Madness': ('xmult',), 'Rocket': ('dollars',)}   # the keys state.joker_state() carries over, per joker


def diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


# ---- the driver -------------------------------------------------------------------------------------------
def resync_hand(G, run):
    """Give the twin the game's hand and draw pile (same cards, matched by signature). False if they don't match."""
    pool = {}
    for c in run.hand + run.draw_pile + [c for c in run.deck if c not in run.hand and c not in run.draw_pile]:
        pool.setdefault(l_card(c), []).append(c)
    try:
        hand = [pool[g_card(c)].pop() for c in lua_list(G.hand.cards)]
        pile = [pool[g_card(c)].pop() for c in lua_list(G.deck.cards)]
    except (KeyError, IndexError):
        return False
    run.hand, run.draw_pile = hand, pile
    return True


def order_like_game(G, run):
    """Put the twin's hand in the game's left-to-right order (the game keeps it sorted by rank)."""
    pos = {g_card(c): k for k, c in enumerate(lua_list(G.hand.cards))}
    run.hand.sort(key=lambda c: pos.get(l_card(c), 99))


def lab_indexes(G, run, positions):
    """Harness hand positions (1-based) -> Balatro Lab hand indexes, matched by card, not by position."""
    hand = lua_list(G.hand.cards)
    free = list(range(len(run.hand)))
    out = []
    for p in positions:
        s = g_card(hand[p - 1])
        k = next(k for k in free if l_card(run.hand[k]) == s)
        free.remove(k)
        out.append(k)
    return sorted(out)


def choose(env, G, acts, pick, estimate):
    """The driver's action: (harness action, twin action or None, twin decision kind or None)."""
    types = {x[0] for x in acts}
    first = lambda t: next(x for x in acts if x[0] == t)  # noqa: E731
    if "play" in types:
        run = lab_run(G)
        plays = [x for x in acts if x[0] == "play"]
        best, best_v = plays[0], -1
        for x in plays:
            played = [run.hand[i - 1] for i in x[3]]
            held = [c for k, c in enumerate(run.hand) if k + 1 not in x[3]]
            v = estimate(run, played, held, sure=True)
            if v > best_v:
                best, best_v = x, v
        need = float(G.GAME.blind.chips) - float(G.GAME.chips)
        discards = [x for x in acts if x[0] == "discard"]
        if discards and best_v * int(G.GAME.current_round.hands_left) < need:
            keep = set(best[3])
            rest = sorted((i for i in range(1, len(run.hand) + 1) if i not in keep),
                          key=lambda i: run.hand[i - 1].rank)[:5]
            d = next((x for x in discards if set(x[3]) == set(rest)), None)
            if rest and d:
                return d, "discard", "round"
        return best, "play", "round"
    if "select_blind" in types:
        if str(G.GAME.blind_on_deck) == "Boss":
            return first("select_blind"), None, None
        rr = G.GAME.round_resets
        tag = rr.blind_tags[str(G.GAME.blind_on_deck)]
        if "skip_blind" in types and str(tag) != "tag_orbital" and pick.random() < 0.15:
            return first("skip_blind"), True, "blind"
        return first("select_blind"), False, "blind"
    if "cash_out" in types:
        return first("cash_out"), None, None
    if "pick" in types or "skip_pack" in types:
        picks = [x for x in acts if x[0] == "pick" and not x[3]
                 and g_name(G.pack_cards.cards[x[1]]) not in AVOID]
        if picks:
            return picks[0], ("pick", picks[0][1] - 1, []), "pack"
        return first("skip_pack"), ("skip",), "pack"
    # shop (and planets held anywhere)
    uses = [x for x in acts if x[0] == "use_consumable" and not x[3]
            and str(G.consumeables.cards[x[1]].ability.set) == "Planet"]
    if uses:
        return uses[0], ("use", uses[0][1] - 1, []), "shop"
    if "end_shop" not in types:
        return pick.choice(acts), None, None
    shop_set = lambda x: str(G.shop_jokers.cards[x[1]].ability.set)  # noqa: E731
    buys = [x for x in acts if x[0] == "buy" and shop_set(x) in ("Joker", "Planet")
            and g_name(G.shop_jokers.cards[x[1]]) not in AVOID]
    if buys and pick.random() < 0.7:
        x = pick.choice(buys)
        return x, ("buy", "card", x[1] - 1), "shop"
    if "redeem" in types and pick.random() < 0.5:
        x = first("redeem")
        return x, ("buy", "voucher", x[1] - 1), "shop"
    opens = [x for x in acts if x[0] == "open_pack"
             and any(k in str(G.shop_booster.cards[x[1]].config.center.name) for k in ("Buffoon", "Celestial"))]
    if opens and pick.random() < 0.5:
        x = pick.choice(opens)
        return x, ("buy", "pack", x[1] - 1), "shop"
    if "reroll" in types and int(G.GAME.dollars) > 12 and pick.random() < 0.25:
        return first("reroll"), ("reroll",), "shop"
    return first("end_shop"), ("leave",), "shop"


def play_seed(env, seed, pick, estimate):
    env.reset(seed=seed, back="b_red", stake=1)
    G = env.lua.globals().G
    if not LOC:
        L = G.localization.descriptions
        for s in ("Joker", "Tarot", "Planet", "Spectral", "Voucher", "Tag", "Blind"):
            LOC[s] = L[s]
    twin = Twin(seed)
    decisions = Counter()
    trail = []
    try:
        for step in range(4000):
            if env.done:
                kind, run, extra = twin.next()
                ok = kind == "end"
                same = ok and bool(extra.won) == bool(G.GAME.won) and int(extra.ante) == int(G.GAME.round_resets.ante)
                return {"seed": seed, "result": ("both ended" if same else "ended differently") if ok
                        else f"game over, twin asks {kind}", "ante": int(G.GAME.round_resets.ante),
                        "won": bool(G.GAME.won), "twin_won": bool(extra.won) if ok else None,
                        "decisions": dict(decisions)}
            acts = env.legal_actions()
            h_act, t_act, t_kind = choose(env, G, acts, pick, estimate)
            if t_kind is not None:
                kind, run, extra = twin.next()
                if kind != t_kind:
                    return {"seed": seed, "result": "diverged", "where": f"game at {t_kind}, twin at {kind}",
                            "ante": int(G.GAME.round_resets.ante), "decisions": dict(decisions),
                            "game": game_view(G, t_kind), "twin": lab_view(run, kind, extra) if kind in
                            ("blind", "round", "shop", "pack") else str(extra)[:2000]}
                gv, lv = game_view(G, t_kind), lab_view(run, kind, extra)
                d = diff(gv, lv)
                if d and kind == "round" and set(d) <= {"hand", "next"} and run.blind.boss is not None \
                        and run.blind.boss.name == "The Hook" and resync_hand(G, run):
                    decisions["hook_resync"] += 1
                    d = diff(game_view(G, t_kind), lab_view(run, kind, extra))
                if d:
                    return {"seed": seed, "result": "diverged", "where": t_kind, "action_next": list(h_act[:2]),
                            "ante": gv["ante"], "decisions": dict(decisions), "diff": d, "trail": trail[-12:]}
                decisions[t_kind] += 1
                if t_act in ("play", "discard"):
                    order_like_game(G, run)    # left-to-right order scores (Hanging Chad, Photograph...)
                    t_act = (t_act, lab_indexes(G, run, h_act[3]))
                twin.answer(t_act)
            if h_act[0] in ("play", "discard"):
                trail.append([h_act[0], sorted(g_card(lua_list(G.hand.cards)[i - 1]) for i in h_act[3])])
            elif h_act[0] in ("buy", "open_pack", "pick", "redeem", "use_consumable"):
                area = {"buy": "shop_jokers", "open_pack": "shop_booster", "pick": "pack_cards",
                        "redeem": "shop_vouchers", "use_consumable": "consumeables"}[h_act[0]]
                c = lua_list(G[area].cards)[h_act[1] - 1]
                trail.append([h_act[0], g_name(c) if str(c.config.center.set) not in ("Default", "Enhanced")
                              else g_card(c)])
            else:
                trail.append([h_act[0]])
            env.step(h_act)
        return {"seed": seed, "result": "step cap", "decisions": dict(decisions)}
    except queue.Empty:
        return {"seed": seed, "result": "twin timeout", "decisions": dict(decisions)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rng", type=int, default=90210)
    ap.add_argument("--out", default="twin.jsonl")
    a = ap.parse_args()
    load_lab(a.lab)
    from balatro.bots import estimate
    chars = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    r = random.Random(a.rng)
    seeds = ["".join(r.choice(chars) for _ in range(8)) for _ in range(a.seeds)]
    env = BalatroEnv(a.src, mode="subsets")
    pick = random.Random(1)
    with open(a.out, "w") as out:
        for seed in seeds:
            try:
                rec = play_seed(env, seed, pick, estimate)
            except Exception:
                rec = {"seed": seed, "result": "driver error", "error": traceback.format_exc(limit=4)}
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            print(seed, rec["result"], rec.get("where", ""), rec.get("decisions", ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
