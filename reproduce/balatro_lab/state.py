"""Harness state -> a Balatro Lab Run.

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e), imported
from a separate clone: `load_lab(dir)` puts it on the path. Nothing of Balatro Lab is copied here.

`lab_run(G)` reads the game's own state (G = the Lua globals' G from a harness runtime) during a round and
returns a balatro.run.Run holding the same deck, hand, draw pile, jokers (with their counters), hand levels,
money, round counters and blind. `unmapped` on the result lists what couldn't be carried over.
"""
from __future__ import annotations

import sys
from collections import Counter

LAB = {}


def load_lab(lab_dir: str) -> None:
    """Import Balatro Lab from its own clone (once)."""
    if LAB:
        return
    sys.path.insert(0, lab_dir)
    from balatro.run import Run, Blind
    from balatro.cards import Card
    from balatro.jokers import CATALOG
    from balatro.items import make, BOSSES, Boss
    LAB.update(Run=Run, Blind=Blind, Card=Card, CATALOG=CATALOG, make=make, BOSSES=BOSSES, Boss=Boss)


ENHANCEMENT = {"m_bonus": "Bonus", "m_mult": "Mult", "m_wild": "Wild", "m_glass": "Glass", "m_steel": "Steel",
               "m_stone": "Stone", "m_gold": "Gold", "m_lucky": "Lucky"}
EDITION = (("foil", "Foil"), ("holo", "Holographic"), ("polychrome", "Polychrome"), ("negative", "Negative"))
KIND = {"Small": 0, "Big": 1, "Boss": 2}


def lua_list(t):
    return [t[i] for i in range(1, len(t) + 1)] if t is not None else []


def edition_of(card):
    e = card.edition
    if e is None:
        return None
    for k, name in EDITION:
        if e[k]:
            return name
    return None


def lab_card(card, uid):
    ability = card.ability
    key = card.config.center.key
    c = LAB["Card"](rank=int(card.base.id), suit=str(card.base.suit),
                    enhancement=ENHANCEMENT.get(key), edition=edition_of(card),
                    seal=str(card.seal) if card.seal else None,
                    bonus_chips=int(ability.perma_bonus or 0) if ability else 0,
                    debuffed=bool(card.debuff), face_down=card.facing == "back", uid=uid)
    return c


def _x(ability):
    return float(ability.x_mult or 1)


def joker_state(name, ability, G):
    """The joker's counters, as Balatro Lab's class for it keeps them (only jokers whose class has state)."""
    cr = G.GAME.current_round
    ex = ability.extra
    return {
        "Ice Cream": lambda: {"chips": ex.chips},
        "Popcorn": lambda: {"mult": ability.mult},
        "Red Card": lambda: {"mult": ability.mult},
        "Ride the Bus": lambda: {"value": ability.mult},
        "Runner": lambda: {"value": ex.chips},
        "Square Joker": lambda: {"value": ex.chips},
        "Green Joker": lambda: {"value": ability.mult},
        "To Do List": lambda: {"target": str(ability.to_do_poker_hand)},
        "Mail-In Rebate": lambda: {"rank": int(cr.mail_card.id)},
        "Obelisk": lambda: {"xmult": _x(ability)},
        "Ancient Joker": lambda: {"suit": str(cr.ancient_card.suit)},
        "Campfire": lambda: {"xmult": _x(ability)},
        "Wee Joker": lambda: {"chips": ex.chips},
        "Hit the Road": lambda: {"xmult": _x(ability)},
        "Invisible Joker": lambda: {"rounds": ability.invis_rounds},
        "Canio": lambda: {"xmult": float(ability.caino_xmult or 1)},
        "Yorick": lambda: {"xmult": _x(ability),
                           "discarded": int((_x(ability) - 1) * 23 + (23 - (ability.yorick_discards or 23)))},
        "Turtle Bean": lambda: {"hand_size_bonus": ex.h_size},
        "The Idol": lambda: {"rank": int(cr.idol_card.id), "suit": str(cr.idol_card.suit)},
        "Lucky Cat": lambda: {"xmult": _x(ability)},
        "Seltzer": lambda: {"hands": ex},
        "Vampire": lambda: {"xmult": _x(ability)},
        "Spare Trousers": lambda: {"mult": ability.mult},
        "Loyalty Card": lambda: {"since": ability.hands_played_at_create},
        "Constellation": lambda: {"xmult": _x(ability)},
        "Hologram": lambda: {"xmult": _x(ability)},
        "Glass Joker": lambda: {"xmult": _x(ability)},
        "Flash Card": lambda: {"mult": ability.mult},
        "Ramen": lambda: {"xmult": _x(ability)},
        "Castle": lambda: {"chips": ex.chips, "suit": str(cr.castle_card.suit)},
        "Ceremonial Dagger": lambda: {"mult": ability.mult},
        "Madness": lambda: {"xmult": _x(ability)},
        "Rocket": lambda: {"dollars": ex.dollars},
    }.get(name, dict)()


STAKES = ("White Stake", "Red Stake", "Green Stake", "Black Stake", "Blue Stake", "Purple Stake", "Orange Stake",
          "Gold Stake")


def run_args(G):
    """Fix F2 (docs/hypotheses-v6.md, 2026-11-19): the game's own deck and stake, by Balatro Lab's names, for every Run
    the bridge builds. Before F2 every Run was Red Deck at White Stake, whatever the game was playing. The game's
    Back.name is the deck's English name (back.lua), which is Balatro Lab's; G.GAME.stake is 1-8 in the game's order."""
    back = G.GAME.selected_back
    deck = str(back.name) if back is not None and back.name else "Red Deck"
    return {"deck": deck, "stake": STAKES[int(G.GAME.stake or 1) - 1]}


def lab_run(G):
    """A Balatro Lab Run matching the harness's current round."""
    L = G.localization.descriptions
    Run = LAB["Run"]
    run = Run(None, **run_args(G))
    run.unmapped = []
    # cards: one Lab Card per game card, shared between deck / hand / draw pile like the game's
    by_id, uid = {}, 0
    for card in lua_list(G.playing_cards):
        uid += 1
        by_id[card.sort_id] = lab_card(card, uid)
    run.deck = list(by_id.values())
    run.next_uid = uid + 1
    run.hand = [by_id[c.sort_id] for c in lua_list(G.hand.cards)]
    run.draw_pile = [by_id[c.sort_id] for c in lua_list(G.deck.cards)]   # Lab draws from the end, as the game does
    # jokers
    run.jokers = []
    for card in lua_list(G.jokers.cards):
        key = card.config.center.key
        name = str(L.Joker[key].name)
        cls = LAB["CATALOG"].get(name)
        if cls is None:
            run.unmapped.append(f"joker {name}")
            continue
        j = cls(edition_of(card))
        for k, v in joker_state(name, card.ability, G).items():
            setattr(j, k, v)
        j.eternal, j.perishable, j.rental = bool(card.ability.eternal), bool(card.ability.perishable), \
            bool(card.ability.rental)
        j.disabled = bool(card.debuff)
        run.jokers.append(j)
    run.joker_slots = int(G.jokers.config.card_limit)
    # vouchers
    run.vouchers = [str(L.Voucher[k].name) for k, v in G.GAME.used_vouchers.items() if v]
    if lua_list(G.consumeables.cards):
        run.unmapped.append("consumables (not carried over)")
    # run counters
    hands = G.GAME.hands
    run.hand_levels = {str(k): int(v.level) for k, v in hands.items()}
    run.hand_plays = Counter({str(k): int(v.played) for k, v in hands.items()})
    # hand types played this round, in the order first played (Balatro Lab's The Mouth reads the first key); the
    # game keeps only the locked type (blind.only_hand), so that one goes first
    played_now = {str(k): int(v.played_this_round) for k, v in hands.items() if int(v.played_this_round)}
    first = G.GAME.blind.only_hand
    order = sorted(played_now, key=lambda h: h != (str(first) if first else None))
    run.round_hand_types = Counter({h: played_now[h] for h in order})
    run.hands_played = int(G.GAME.hands_played)
    run.money = int(G.GAME.dollars)
    run.ante = int(G.GAME.round_resets.ante)
    run.skips = int(G.GAME.skips or 0)
    usage = G.GAME.consumeable_usage_total
    if usage is not None:
        run.consumables_used = Counter({"Tarot": int(usage.tarot or 0), "Planet": int(usage.planet or 0),
                                        "Spectral": int(usage.spectral or 0)})
    cr = G.GAME.current_round
    run.hands_left, run.discards_left = int(cr.hands_left), int(cr.discards_left)
    run.discards_used = int(cr.discards_used or 0)
    run.in_round = True
    # blind
    b = G.GAME.blind
    kind = KIND.get(str(G.GAME.blind_on_deck), 0)
    boss = None
    if kind == 2:
        boss = LAB["make"](LAB["BOSSES"], LAB["Boss"], str(L.Blind[b.config.blind.key].name))
        boss.disabled = bool(b.disabled)
    run.blind = LAB["Blind"](str(b.name), kind, int(b.chips), int(b.dollars or 0), boss)
    run.ante_boss = boss
    run.round_score = int(G.GAME.chips)
    return run
