"""Where do our bot and Balatro Lab's bot choose differently? (disagreement mining, step 1: find them)

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e): its seeded
Run (through twin.py) and its own bot, balatro.strategy.ValueBot (the bot behind its 70% runs, `--bot base`).
Nothing of Balatro Lab is copied here.

For each seed:
1. Our bot plays the game in the harness exactly as in its evals (train.play_games, the trainer's own code), which
   gives its moves (the route).
2. The route is replayed in the harness while Balatro Lab's seeded run follows it in lockstep (twin.py), compared
   at every decision. A seed's comparison ends at the first difference the twin can't re-sync (The Hook's 2 cards
   are re-synced; other chance effects end it), since after that the twin is in another game.
3. At every decision point of the twin, a copy of the twin's state is handed to Balatro Lab's bot, which says what
   it would do there. Its answer and our bot's move are both written down by what they mean ("buy Hologram",
   "play 13H 13S 2C 2S 14C", "skip"), along with what each sold or used first.

Output: one line per decision point: seed, move number, ante, decision type, ours, theirs, agree. Step 2 of the
mining (which of the two choices wins more often, played on from that point) reads these lines.

    python3 mine.py --weights v4b_best.pt --lab <balatro-lab clone> --src <game source> --seeds 20 --out mine.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import random
import sys
import time
import traceback
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
import twin as TW  # noqa: E402  (sets up harness paths and the lupa shim)
from twin import (Twin, diff, g_card, g_name, game_view, l_card, lab_view, lua_list,  # noqa: E402
                  resync_hand)

FREE = {"sell_joker", "sell_consumable", "use_consumable"}
REAL_KIND = {"play": "round", "discard": "round", "end_shop": "shop", "buy": "shop", "buy_and_use": "shop",
             "redeem": "shop", "open_pack": "shop", "reroll": "shop", "pick": "pack", "skip_pack": "pack"}


# ---- our bot ----------------------------------------------------------------------------------------------
def load_ours(weights, src):
    import torch
    from model import make_model
    from sim import Sim, sim_kwargs
    import train as T
    torch.set_num_threads(1)
    w = torch.load(weights, map_location="cpu", weights_only=True)
    cfg = dict(w["trained"]["provenance"]["config"])
    cfg.update(src=src, mods=None, back="b_red", stake=1, backs=[], stakes=[], challenges=[], max_steps=1500,
               verify_wins=False, save_games="none")
    args = Namespace(**cfg)
    arch = w["arch"]
    sim = Sim(src, fast=True, mode=arch["actions"], enc=arch["enc"], **sim_kwargs(args))
    model = make_model(args, len(w["vocab"]))
    model.load_state_dict(w["state_dict"])
    model.eval()
    it = w["trained"].get("iteration") or 0

    def route(seed):
        ep = T.play_games(sim, model, T.REWARDS[args.reward], random.Random(0), args, 0,
                          (model.state_dict(), [seed], "eval", it, 0), set())[0]
        return ep["acts"], ep["desc"], ep["progress"]
    return sim, route


def parse_desc(desc):
    t = desc.split()
    kind, i, j, cards, lead = t[0], None, None, [], None
    rest = t[1:]
    if kind in ("play", "discard"):
        cards = [int(x) for x in rest[0].split(",")] if rest else []
        rest = rest[1:]
    elif rest and rest[0] not in ("lead", "src"):
        a = rest[0]
        if ">" in a:
            i, j = (int(x) for x in a.split(">"))
        else:
            i = int(a)
        rest = rest[1:]
        if rest and rest[0] not in ("lead", "src"):
            cards = [int(x) for x in rest[0].split(",")]
            rest = rest[1:]
    if "lead" in rest:
        lead = int(rest[rest.index("lead") + 1])
    return kind, i, j, cards, lead


# ---- what a move means ------------------------------------------------------------------------------------
def ours_meaning(G, kind, i, cards):
    hand = lua_list(G.hand.cards)
    if kind in ("play", "discard"):
        return f"{kind} " + " ".join(sorted(g_card(hand[p - 1]) for p in cards))
    area = {"buy": "shop_jokers", "buy_and_use": "shop_jokers", "redeem": "shop_vouchers", "open_pack": "shop_booster",
            "pick": "pack_cards", "sell_joker": "jokers", "sell_consumable": "consumeables",
            "use_consumable": "consumeables"}.get(kind)
    word = {"buy": "buy", "buy_and_use": "buy_use", "redeem": "redeem", "open_pack": "open", "pick": "pick",
            "sell_joker": "sell", "sell_consumable": "sell_c", "use_consumable": "use", "end_shop": "leave",
            "skip_pack": "skip", "reroll": "reroll", "select_blind": "select", "skip_blind": "skip"}[kind]
    if area:
        c = lua_list(G[area].cards)[i - 1]
        name = g_name(c) if str(c.config.center.set) not in ("Default", "Enhanced") else g_card(c)
        return f"{word} {name}"
    return word


def theirs_meaning(run, extra, a):
    from balatro.cards import Card
    k = a[0] if a else None
    if k in ("play", "discard"):
        return f"{k} " + " ".join(sorted(l_card(run.hand[p]) for p in run.selection(a[1])))
    if k == "buy":
        where, i = (a[1], a[2]) if len(a) >= 3 else ("card", a[1])
        offers = {"card": extra.cards, "pack": extra.packs, "voucher": extra.vouchers}[where]
        o = offers[i]
        name = l_card(o.item) if isinstance(o.item, Card) else o.name
        return {"card": "buy", "pack": "open", "voucher": "redeem"}[where] + " " + name
    if k == "buy_use":
        return "buy_use " + extra.cards[a[1]].name
    if k == "pick":
        x = extra.cards[a[1]]
        return "pick " + (l_card(x) if isinstance(x, Card) else x.name)
    if k == "sell_joker":
        return "sell " + run.jokers[a[1]].name
    if k == "sell_consumable":
        return "sell_c " + run.consumables[a[1]].name
    if k == "use":
        return "use " + run.consumables[a[1]].name
    return {"reroll": "reroll", "leave": "leave", "skip": "skip"}.get(k, str(k))


def ask_theirs(bot, kind, run, extra):
    """Balatro Lab's bot at a copy of this decision point: what it sells/uses first, then its move."""
    agent = run.agent
    run.agent = None                       # the relay holds locks; a copy doesn't need it
    try:
        r, x = copy.deepcopy((run, extra))
    finally:
        run.agent = agent
    r.agent = bot
    if kind == "blind":
        blind, tag = x
        return [], "skip" if bot.skip_blind(r, blind, tag) else "select"
    decide = {"round": bot.round_action, "shop": bot.shop_action, "pack": bot.pack_action}[kind]
    frees = []
    for _ in range(30):
        a = decide(r, x) if kind != "round" else decide(r)
        if a and a[0] in ("order_jokers", "order_hand", "sort_hand"):
            r.free_action(a)
            continue
        if a and a[0] in ("use", "sell_joker", "sell_consumable"):
            frees.append(theirs_meaning(r, x, a))
            r.free_action(a)
            continue
        return frees, theirs_meaning(r, x, a)
    return frees, "?"


# ---- lockstep helpers -------------------------------------------------------------------------------------
def order_jokers_like_game(G, run):
    pos = {}
    for k, c in enumerate(lua_list(G.jokers.cards)):
        pos.setdefault((g_name(c), TW.edition_of(c)), []).append(k)
    def key(j):
        lst = pos.get((j.name, j.edition))
        return lst.pop(0) if lst else 99
    run.jokers.sort(key=key)


def order_hand_like_game(G, run, cards=None, lead=None):
    """The twin's hand in the game's order, after the env's lead move (lead dragged to the first selected slot)."""
    sigs = [g_card(c) for c in lua_list(G.hand.cards)]
    if lead and cards and lead != cards[0]:
        s = sigs.pop(lead - 1)
        sigs.insert(cards[0] - 1, s)
    pos = {}
    for k, s in enumerate(sigs):
        pos.setdefault(s, []).append(k)
    run.hand.sort(key=lambda c: pos[l_card(c)].pop(0) if pos.get(l_card(c)) else 99)
    return sigs


def lab_item(c, uid):
    """A Balatro Lab object for a game card in a pack or the shop."""
    from balatro.jokers import CATALOG
    from balatro.shop import CONSUMABLES
    s = str(c.config.center.set)
    if s in ("Default", "Enhanced"):
        return TW.LAB["Card"](rank=int(c.base.id), suit=str(c.base.suit), enhancement=TW.ENHANCEMENT.get(
            c.config.center.key), edition=TW.edition_of(c), seal=str(c.seal) if c.seal else None, uid=uid)
    if s == "Joker":
        return CATALOG[g_name(c)](TW.edition_of(c))
    return CONSUMABLES[g_name(c)]()


def resync_pack(G, run, extra):
    items = []
    for c in lua_list(G.pack_cards.cards):
        items.append(lab_item(c, run.next_uid))
        run.next_uid += 1
    extra.cards = items


def resync_shop_cards(G, run, extra):
    """Only when the differing shop slots hold playing cards: those Balatro Lab draws with its own randomness."""
    from balatro.cards import Card
    game = lua_list(G.shop_jokers.cards)
    if len(game) != len(extra.cards):
        return False
    for c, o in zip(game, extra.cards):
        playing = str(c.config.center.set) in ("Default", "Enhanced")
        if playing != isinstance(o.item, Card):
            return False
        if playing and g_card(c) != l_card(o.item):
            o.item = lab_item(c, run.next_uid)
            run.next_uid += 1
    return True


def lab_targets(G, run, cards, sigs=None):
    sigs = sigs or [g_card(c) for c in lua_list(G.hand.cards)]
    free = list(range(len(run.hand)))
    out = []
    for p in cards:
        k = next(k for k in free if l_card(run.hand[k]) == sigs[p - 1])
        free.remove(k)
        out.append(k)
    return sorted(out)


def twin_action(G, run, kind, i, cards, lead):
    if kind in ("play", "discard"):
        sigs = order_hand_like_game(G, run, cards, lead)
        return (kind, lab_targets(G, run, cards, sigs))
    x = (i or 1) - 1
    if kind in ("buy_and_use", "pick", "use_consumable"):
        tg = lab_targets(G, run, cards) if cards else []
        return {"buy_and_use": ("buy_use", x, tg), "pick": ("pick", x, tg), "use_consumable": ("use", x, tg)}[kind]
    return {"end_shop": ("leave",), "reroll": ("reroll",), "skip_pack": ("skip",), "buy": ("buy", "card", x),
            "redeem": ("buy", "voucher", x), "open_pack": ("buy", "pack", x), "sell_joker": ("sell_joker", x),
            "sell_consumable": ("sell_consumable", x)}[kind]


# ---- one seed ---------------------------------------------------------------------------------------------
def mine_seed(sim, route_of, seed, bot_factory, out):
    acts, descs, prog = route_of(seed)
    sim.reset(seed, "b_red", 1)
    G = sim.env.lua.globals().G
    if not TW.LOC:
        L = G.localization.descriptions
        for s in ("Joker", "Tarot", "Planet", "Spectral", "Voucher", "Tag", "Blind"):
            TW.LOC[s] = L[s]
    tw = Twin(seed)
    bot = bot_factory()
    pending = []                 # our free moves made where the twin isn't asking (cash-out, blind select)
    point = None                 # the open decision point: theirs (asked once), ours (accumulating)
    n_points = n_agree = resyncs = pack_resyncs = shop_resyncs = hand_resyncs = 0
    end = "route ended"

    def close(ours_real):
        nonlocal point, n_points, n_agree
        p = point
        agree = p["theirs"] == ours_real and sorted(p["theirs_free"]) == sorted(p["ours_free"])
        n_points += 1
        n_agree += agree
        out.write(json.dumps({"seed": seed, "move": p["move"], "ante": p["ante"], "kind": p["kind"],
                              "money": p["money"], "ours": ours_real, "ours_free": p["ours_free"],
                              "theirs": p["theirs"], "theirs_free": p["theirs_free"], "agree": agree,
                              "theirs_s": p["sec"]}) + "\n")
        point = None

    try:
        for k, (a, desc) in enumerate(zip(acts, descs)):
            sim.H.encode()                         # sets the action list the route's index points into
            kind, i, j, cards, lead = parse_desc(desc)
            if kind == "move_joker" or kind in ("sort_rank", "sort_suit", "move_hand"):
                sim.step(a)
                continue
            twin_kind = None
            if kind in REAL_KIND:
                twin_kind = REAL_KIND[kind]
            elif kind == "skip_blind" or (kind == "select_blind" and str(G.GAME.blind_on_deck) != "Boss"):
                twin_kind = "blind"
            elif kind in FREE:
                st = str(G.STATE)
                twin_kind = {str(G.STATES.SHOP): "shop", str(G.STATES.SELECTING_HAND): "round",
                             str(G.STATES.TAROT_PACK): "pack", str(G.STATES.PLANET_PACK): "pack",
                             str(G.STATES.SPECTRAL_PACK): "pack", str(G.STATES.STANDARD_PACK): "pack",
                             str(G.STATES.BUFFOON_PACK): "pack"}.get(st)
                if twin_kind is None:              # cash-out or blind select: hand it over at the next ask
                    pending.append((kind, i, cards))
                    sim.step(a)
                    continue
            if twin_kind is None:                  # cash_out, the Boss's select
                sim.step(a)
                continue
            got, run, extra = tw.next()
            while pending and got in ("round", "shop", "pack", "blind"):
                pk, pi, pc = pending.pop(0)
                order_jokers_like_game(G, run)
                tw.answer(twin_action(G, run, pk, pi, pc, None))
                got, run, extra = tw.next()
            if got != twin_kind:
                end = f"twin at {got}, game at {twin_kind}"
                break
            order_jokers_like_game(G, run)
            if got == "pack" and lua_list(G.hand.cards) and getattr(extra.pack, "kind", "") in ("Arcana", "Spectral"):
                if resync_hand(G, run):             # the hand an Arcana/Spectral pack deals: Balatro Lab
                    pack_resyncs += 1               # shuffles it with its own randomness
            d = diff(game_view(G, got), lab_view(run, got, extra))
            if d and got == "pack" and set(d) == {"pack"}:     # pack contents are random draws; Balatro Lab's
                resync_pack(G, run, extra)                     # differ (Standard always, Celestial sometimes)
                pack_resyncs += 1
                d = diff(game_view(G, got), lab_view(run, got, extra))
            if d and got == "shop" and set(d) == {"cards"} and resync_shop_cards(G, run, extra):
                shop_resyncs += 1                              # playing cards in the shop (Magic Trick)
                d = diff(game_view(G, got), lab_view(run, got, extra))
            if d and got == "round" and set(d) <= {"hand", "next"} and run.blind.boss is not None \
                    and run.blind.boss.name == "The Hook" and resync_hand(G, run):
                resyncs += 1
                d = diff(game_view(G, got), lab_view(run, got, extra))
            if d:
                end = f"diverged at {got}: " + json.dumps(d, default=str)[:300]
                break
            if point is None:                      # a new decision point: ask their bot once
                t0 = time.time()
                tf, tr = ask_theirs(bot, got, run, extra)
                point = {"move": k, "ante": int(run.ante), "kind": got, "money": int(run.money), "theirs": tr,
                         "theirs_free": tf, "ours_free": [], "sec": round(time.time() - t0, 2)}
            if got == "blind":
                close("skip" if kind == "skip_blind" else "select")
                tw.answer(kind == "skip_blind")
            else:
                meaning = ours_meaning(G, kind, i, cards)
                if kind in FREE:
                    point["ours_free"].append(meaning)
                else:
                    close(meaning)
                try:
                    act = twin_action(G, run, kind, i, cards, lead)
                except StopIteration:              # a target card isn't in the twin's hand: a random card
                    if not resync_hand(G, run):    # effect (Familiar, Immolate, Sigil...) went another way
                        end = f"hand differs at {kind} with targets"
                        break
                    hand_resyncs += 1
                    act = twin_action(G, run, kind, i, cards, lead)
                tw.answer(act)
            sim.step(a)
    except queue.Empty:
        end = "twin timeout"
    except Exception:
        end = "error: " + traceback.format_exc(limit=3)
    return {"seed": seed, "won": bool(prog.get("won")), "ante": prog.get("ante"), "blinds": prog.get("blinds_beaten"),
            "points": n_points, "agree": n_agree, "hook_resyncs": resyncs, "pack_resyncs": pack_resyncs, "shop_resyncs": shop_resyncs, "hand_resyncs": hand_resyncs, "moves": len(acts), "end": end}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first", type=int, default=0, help="skip this many seeds of the set (to split work)")
    ap.add_argument("--rng", type=int, default=90210)
    ap.add_argument("--out", default="mine.jsonl")
    ap.add_argument("--workers", type=int, default=1, help="split the seeds over this many processes")
    a = ap.parse_args()
    if a.workers > 1:                          # one process per slice, each with its own output, merged after
        import subprocess
        per = -(-a.seeds // a.workers)
        procs, parts = [], []
        for w in range(a.workers):
            first, n = a.first + w * per, min(per, a.seeds - w * per)
            if n <= 0:
                break
            part = a.out.replace(".jsonl", f"_part{w}.jsonl")
            parts.append(part)
            procs.append(subprocess.Popen([sys.executable, __file__, "--weights", a.weights, "--lab", a.lab,
                                           "--src", a.src, "--seeds", str(n), "--first", str(first),
                                           "--rng", str(a.rng), "--out", part]))
        for pr in procs:
            pr.wait()
        for suffix in ("", "_seeds"):
            with open(a.out.replace(".jsonl", f"{suffix}.jsonl"), "a") as out:
                for part in parts:
                    f = part.replace(".jsonl", f"{suffix}.jsonl")
                    if os.path.exists(f):
                        out.write(open(f).read())
                        os.remove(f)
        return 0
    os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")      # as Balatro Lab's eval_value.py sets it
    TW.TIMEOUT = 3600                                       # their bot's boss-round look-ahead can take minutes
    TW.load_lab(a.lab)
    from balatro.strategy import ValueBot
    chars = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    r = random.Random(a.rng)
    seeds = ["".join(r.choice(chars) for _ in range(8)) for _ in range(a.first + a.seeds)][a.first:]
    sim, route_of = load_ours(a.weights, a.src)
    with open(a.out, "a") as out, open(a.out.replace(".jsonl", "_seeds.jsonl"), "a") as sout:
        for seed in seeds:
            rec = mine_seed(sim, route_of, seed, ValueBot, out)
            out.flush()
            sout.write(json.dumps(rec) + "\n")
            sout.flush()
            print(json.dumps(rec)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
