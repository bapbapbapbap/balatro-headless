"""The bridge (model v6, stage 0; docs/hypotheses-v6.md): Balatro Lab's bot deciding on our game's own state.

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e), imported from
a separate clone (state.load_lab): its Run, Shop, Offer, OpenPack, PACKS and item registries, and its value bot
(balatro.strategy.ValueBot) as the teacher. Nothing of Balatro Lab is copied here.

R's twin (twin.py) ran Balatro Lab's own seeded run beside the game and was lost in most games once a chance effect
went its own way. The bridge has no run of its own to lose: at every decision, `Shadow.state(G)` builds a fresh
Balatro Lab Run from what the game shows (state.lab_run for the round, extended to every decision type here), plus
the decision's own object: (blind, tag) for the blind choice, a Shop for the shop, an OpenPack for a pack. A few run
totals the game does not keep (rerolls, packs skipped, cards sold, discards) are counted by `Shadow.after(...)` from
the moves played; the rest is read from the game each time.

The teacher sees only what a player sees: the run carries no game seed (`seedrng` None), its random streams are
fresh, and the bot's own sampling is keyed on the run's seed string and counters, as in its own games.

    sh = Shadow(seed); kind = decision_kind(G, state_name)
    run, extra = sh.state(G, kind)            # kind: "round", "shop", "pack", "blind", "boss"
    move = teacher_move(bot, kind, run, extra) # Balatro Lab's action tuple (reorders skipped, counted)
    ...play the matching harness move (rank.to_desc)..., then sh.after(G_before_info, desc)
"""
from __future__ import annotations

import copy
from collections import Counter

import state as S
from state import LAB, edition_of, lua_list

PACK_STATES = {"TAROT_PACK": "Arcana", "PLANET_PACK": "Celestial", "SPECTRAL_PACK": "Spectral",
               "STANDARD_PACK": "Standard", "BUFFOON_PACK": "Buffoon"}
KIND = {"Small": 0, "Big": 1, "Boss": 2}
_LAB2 = {}


def _lab():
    """Balatro Lab names beyond state.py's (after state.load_lab)."""
    if not _LAB2:
        from balatro.items import CONSUMABLES, TAGS, Tag, VOUCHERS, Voucher, make
        from balatro.shop import PACKS, Offer, OpenPack, Shop
        from balatro.jokers import CATALOG
        _LAB2.update(CONSUMABLES=CONSUMABLES, TAGS=TAGS, Tag=Tag, VOUCHERS=VOUCHERS, Voucher=Voucher, make=make,
                     PACKS=PACKS, Offer=Offer, OpenPack=OpenPack, Shop=Shop, CATALOG=CATALOG)
    return _LAB2


def decision_kind(G, state_name):
    if state_name == "SELECTING_HAND":
        return "round"
    if state_name == "SHOP":
        return "shop"
    if state_name in PACK_STATES:
        return "pack"
    if state_name == "BLIND_SELECT":
        return "boss" if str(G.GAME.blind_on_deck) == "Boss" else "blind"
    if state_name == "ROUND_EVAL":
        return "cash"
    return None


def name_of(G, card):
    """A game card's name as Balatro Lab names it (twin.g_name, without twin's module state)."""
    L = G.localization.descriptions
    center = card.config.center
    s = str(center.set)
    if s == "Booster":
        return str(center.name)
    return str(L[s][str(center.key)].name)


class Shadow:
    """One game's bridge: run totals the game doesn't keep, and a fresh Balatro Lab state at each decision."""

    def __init__(self, seed: str):
        self.seed = seed
        self.n = Counter()             # rerolls, packs_skipped, cards_sold, discards_made, cards_discarded, shops
        self.pack_picks = 0            # picks made in the open pack
        self.pack_first = None         # (pack_size, pack_choices) when the open pack was first seen
        self.in_shop = False
        self.in_cash = False
        self.round_kind = 0
        self.last_round = None         # what Balatro Lab's run still holds from the round just won (blind, counts)
        self.unmapped = Counter()      # what couldn't be carried over, by name
        self.uid = {}                  # game card (sort_id) -> Balatro Lab uid, numbered as its own runs number them
        self.next_uid = 1

    def uids(self, G):
        """Card identities as Balatro Lab numbers them: the starting deck in its standard order (suits Spades,
        Hearts, Clubs, Diamonds, ranks 2-14), then each card added since, in the order it appeared. The bot keys
        its sampling on uids, so the same numbering gives the same samples as in its own runs."""
        cards = lua_list(G.playing_cards)
        if not self.uid:
            order = {s: i for i, s in enumerate(("Spades", "Hearts", "Clubs", "Diamonds"))}
            first = sorted(cards, key=lambda c: (order.get(str(c.base.suit), 9), int(c.base.id)))
            for c in first:
                self.uid[int(c.sort_id)] = self.next_uid
                self.next_uid += 1
        for c in sorted(cards, key=lambda c: int(c.sort_id)):
            if int(c.sort_id) not in self.uid:
                self.uid[int(c.sort_id)] = self.next_uid
                self.next_uid += 1
        return [self.uid[int(c.sort_id)] for c in cards]

    # ---- the run as the game shows it -------------------------------------------------------------------------
    def base_run(self, G, in_round: bool):
        X = _lab()
        L = G.localization.descriptions
        if in_round:
            run = S.lab_run(G)                     # deck, hand, draw pile, jokers, levels, counters, blind
            # the game keeps every card in the draw pile face down (facing "back"); state.lab_card reads that as
            # Balatro Lab's face_down, which means a card dealt face down by a boss and blinds the bot to the
            # whole deck. Only cards in hand can be face down in that sense (The House, Wheel, Mark, Fish).
            in_hand = {id(c) for c in run.hand}
            for c in run.deck:
                if id(c) not in in_hand:
                    c.face_down = False
            for c, u in zip(run.deck, self.uids(G)):    # lab_run builds the deck in G.playing_cards' order
                c.uid = u
            run.next_uid = self.next_uid
            # Cerulean Bell (fix F1, docs/hypotheses-v6.md 2026-11-19): the game marks the card it forces into every
            # play and discard (ability.forced_selection); Balatro Lab's CeruleanBell keeps it as `forced`, which its
            # Run.selection adds to the bot's picks. Without it the teacher's hands leave the card out and none match.
            boss = run.blind.boss if run.blind is not None else None
            if boss is not None and boss.name == "Cerulean Bell" and not boss.disabled:
                hand = lua_list(G.hand.cards)
                boss.forced = next((run.hand[i] for i, c in enumerate(hand)
                                    if c.ability and c.ability.forced_selection), None)
        else:
            run = LAB["Run"](None, deck="Red Deck")
            run.unmapped = []
            by_id = {}
            for card, u in zip(lua_list(G.playing_cards), self.uids(G)):
                by_id[card.sort_id] = S.lab_card(card, u)
                by_id[card.sort_id].face_down = False
            self.by_sort = by_id
            run.deck = list(by_id.values())
            run.next_uid = self.next_uid
            run.hand, run.draw_pile = [], []
            run.jokers = []
            for card in lua_list(G.jokers.cards):
                name = name_of(G, card)
                cls = LAB["CATALOG"].get(name)
                if cls is None:
                    run.unmapped.append(f"joker {name}")
                    continue
                j = cls(edition_of(card))
                for k, v in S.joker_state(name, card.ability, G).items():
                    setattr(j, k, v)
                j.eternal, j.perishable, j.rental = bool(card.ability.eternal), bool(card.ability.perishable), \
                    bool(card.ability.rental)
                run.jokers.append(j)
            hands = G.GAME.hands
            run.hand_levels = {str(k): int(v.level) for k, v in hands.items()}
            run.hand_plays = Counter({str(k): int(v.played) for k, v in hands.items()})
            run.hands_played = int(G.GAME.hands_played)
            run.money = int(G.GAME.dollars)
            run.ante = int(G.GAME.round_resets.ante)
            run.skips = int(G.GAME.skips or 0)
            usage = G.GAME.consumeable_usage_total
            if usage is not None:
                run.consumables_used = Counter({"Tarot": int(usage.tarot or 0), "Planet": int(usage.planet or 0),
                                                "Spectral": int(usage.spectral or 0)})
            run.in_round = False
            run.blind = None
            run.round_score = 0
        run.seed = self.seed
        run.seedrng = None
        # vouchers: their own effects first (whatever the game doesn't show), then the game's numbers over them
        run.vouchers = []
        for k, v in G.GAME.used_vouchers.items():
            if v:
                name = str(L.Voucher[str(k)].name)
                run.vouchers.append(name)
                try:
                    X["make"](X["VOUCHERS"], X["Voucher"], name).apply(run)
                except Exception:
                    self.unmapped[f"voucher {name}"] += 1
        run.ante = int(G.GAME.round_resets.ante)
        run.money = int(G.GAME.dollars)
        # consumables held
        run.consumables = []
        for c in lua_list(G.consumeables.cards):
            name = name_of(G, c)
            cls = X["CONSUMABLES"].get(name)
            if cls is None:
                self.unmapped[f"consumable {name}"] += 1
                continue
            run.consumables.append(cls(edition_of(c)))
        if "consumables (not carried over)" in run.unmapped:
            run.unmapped.remove("consumables (not carried over)")
        # slots, hands, discards, hand size: the game's own numbers, less what Balatro Lab adds on its own
        neg_j = sum(j.edition == "Negative" for j in run.jokers)
        neg_c = sum(c.edition == "Negative" for c in run.consumables)
        run.joker_slots = int(G.jokers.config.card_limit) - neg_j
        run.consumable_slots = int(G.consumeables.config.card_limit) - neg_c
        act = [j for j in run.jokers if not j.disabled]
        rr = G.GAME.round_resets
        run.base_hands = int(rr.hands) - sum(j.hands_bonus for j in act)
        run.base_discards = int(rr.discards) - sum(j.discards_bonus for j in act)
        delta = 0
        if in_round:
            delta = int(rr.temp_handsize or 0)
            if run.blind is not None and run.blind.boss is not None and run.blind.boss.name == "The Manacle" \
                    and not run.blind.boss.disabled:
                delta -= 1
        run.hand_size_delta = delta
        run.base_hand_size = int(G.hand.config.card_limit) - delta - sum(j.hand_size_bonus for j in act)
        # economy and shop odds
        run.interest_cap = int(G.GAME.interest_cap or 25) // 5
        run.interest_enabled = not bool(G.GAME.modifiers.no_interest)
        run.discount_percent = int(G.GAME.discount_percent or 0)
        run.edition_rate = float(G.GAME.edition_rate or 1)
        run.reroll_base = int(rr.reroll_cost or 5)
        run.shop_card_slots = int(G.GAME.shop.joker_max or 2)
        # tags held, the ante's boss, skip tags and voucher
        run.tags = [t for t in (self.tag(G, str(t.key)) for t in lua_list(G.GAME.tags)) if t is not None]
        bc = rr.blind_choices
        run.ante_boss = LAB["make"](LAB["BOSSES"], LAB["Boss"], str(L.Blind[str(bc.Boss)].name)) if bc.Boss else None
        if in_round and run.blind is not None and run.blind.boss is not None:
            run.ante_boss = run.blind.boss
        run.ante_tags = []
        for i, on in enumerate(("Small", "Big")):
            key = rr.blind_tags[on] if rr.blind_tags else None
            t = self.tag(G, str(key)) if key else None
            if t is not None and t.name == "Orbital Tag":
                oc = G.GAME.orbital_choices
                oc = oc[int(rr.ante)] if oc else None
                if oc and oc[on]:
                    t.hand = str(oc[on])
            run.ante_tags.append(t)
        cr = G.GAME.current_round
        v = cr.voucher
        run.ante_voucher = str(L.Voucher[str(v)].name) if v and not G.GAME.used_vouchers[str(v)] else None
        run.vouchers_rolled = max(1, run.ante)
        run.boss_rerolled = bool(rr.boss_rerolled)
        # run totals
        run.rounds_played = max(0, int(G.GAME.round) - (1 if in_round else 0))
        run.unused_discards = int(G.GAME.unused_discards or 0)
        rs = G.GAME.round_scores
        run.rerolls_done = int(rs.times_rerolled.amt)
        run.cards_played = int(rs.cards_played.amt)
        run.cards_discarded = int(rs.cards_discarded.amt)
        run.cards_bought = int(rs.cards_purchased.amt)
        run.best_hand = int(rs.hand.amt)
        run.packs_skipped = self.n["packs_skipped"]
        run.cards_sold = self.n["cards_sold"]
        run.discards_made = self.n["discards_made"]
        run.shops_seen = self.n["shops"]
        run.blinds_beaten = self.n["blinds_beaten"]
        lh = G.GAME.last_hand_played
        run.last_hand = str(lh) if lh else None
        run.boss_used = Counter()
        for k, v in (G.GAME.bosses_used or {}).items():
            if v and int(v) > 0:
                run.boss_used[str(L.Blind[str(k)].name)] = int(v)
        run.played_this_ante = {u for c, u in zip(lua_list(G.playing_cards), self.uids(G))
                                if c.ability and c.ability.played_this_ante}
        run.deck.sort(key=lambda c: c.uid)        # Balatro Lab keeps its deck in uid order; its samples follow it
        if not in_round and self.last_round is not None:
            lr = self.last_round
            run.blind = run.make_blind(lr["kind"])
            run.hands_left, run.discards_left, run.discards_used = lr["hands_left"], lr["discards_left"], lr["discards_used"]
            run.round_score = lr["score"]
            run.round_hand_types = Counter(lr["types"])
        run.extinct = {"Gros Michel"} if G.GAME.pool_flags.gros_michel_extinct else set()
        ltp = G.GAME.last_tarot_planet
        run.last_tarot_planet = self.center_name(G, str(ltp)) if ltp else None
        usage = G.GAME.consumeable_usage
        run.planets_used = set()
        if usage:
            for k, v in usage.items():
                c = G.P_CENTERS[str(k)]
                if c and str(c.set) == "Planet" and v and int(v.count or 0) > 0:
                    run.planets_used.add(self.center_name(G, str(k)))
        return run

    @staticmethod
    def center_name(G, key):
        c = G.P_CENTERS[key]
        return str(G.localization.descriptions[str(c.set)][key].name) if c else key

    def tag(self, G, key):
        X = _lab()
        L = G.localization.descriptions
        try:
            name = str(L.Tag[key].name)
        except Exception:
            self.unmapped[f"tag {key}"] += 1
            return None
        if name not in X["TAGS"]:
            self.unmapped[f"tag {name}"] += 1
            return None
        return X["make"](X["TAGS"], X["Tag"], name)

    # ---- each decision's state --------------------------------------------------------------------------------
    def state(self, G, kind, state_name=None):
        if kind == "round":
            return self.base_run(G, True), None
        run = self.base_run(G, False)
        if kind in ("blind", "boss"):
            k = KIND[str(G.GAME.blind_on_deck)]
            run.blind = run.make_blind(k)
            return run, (run.blind, run.ante_tags[k] if k < 2 else None)
        if kind == "shop":
            return run, self.shop(G, run)
        if kind == "pack":
            return run, self.pack(G, run, PACK_STATES[state_name])
        if kind == "cash":
            return run, None
        raise ValueError(kind)

    def shop(self, G, run):
        X = _lab()
        sh = X["Shop"].__new__(X["Shop"])
        sh.run = run
        sh.rng = run.rng["shop"]
        cr = G.GAME.current_round
        sh.rerolls = int(cr.reroll_cost_increase or 0)
        sh.free_rerolls = int(cr.free_rerolls or 0)
        temp = G.GAME.round_resets.temp_reroll_cost
        sh.reroll_base = int(temp) if temp is not None else None
        sh.cards, sh.packs, sh.vouchers = [], [], []
        for c in lua_list(G.shop_jokers.cards):
            o = self.offer(G, run, c)
            if o is not None:
                sh.cards.append(o)
        for c in lua_list(G.shop_booster.cards):
            name = name_of(G, c)
            p = X["PACKS"].get(name)
            if p is None:
                self.unmapped[f"pack {name}"] += 1
                continue
            sh.packs.append(X["Offer"]("Pack", p, free=int(c.cost) == 0))
        for c in lua_list(G.shop_vouchers.cards):
            sh.vouchers.append(X["Offer"]("Voucher", name_of(G, c), free=int(c.cost) == 0))
        run.current_shop = sh
        return sh

    def offer(self, G, run, c):
        X = _lab()
        s = str(c.config.center.set)
        free = int(c.cost) == 0 and not (s == "Planet" and run.has_joker("Astronomer"))
        if s in ("Default", "Enhanced"):
            card = S.lab_card(c, run.next_uid)
            run.next_uid += 1
            card.debuffed = card.face_down = False
            return X["Offer"]("Playing card", card, free)
        name = name_of(G, c)
        if s == "Joker":
            cls = X["CATALOG"].get(name)
            if cls is None:
                self.unmapped[f"joker {name}"] += 1
                return None
            j = cls(edition_of(c))
            j.eternal, j.perishable, j.rental = bool(c.ability.eternal), bool(c.ability.perishable), \
                bool(c.ability.rental)
            return X["Offer"]("Joker", j, free)
        cls = X["CONSUMABLES"].get(name)
        if cls is None:
            self.unmapped[f"{s} {name}"] += 1
            return None
        return X["Offer"](s, cls(edition_of(c)), free)

    def pack(self, G, run, kind):
        X = _lab()
        size, choices = self.pack_first or (int(G.GAME.pack_size), int(G.GAME.pack_choices))
        p = next((p for p in X["PACKS"].values() if p.kind == kind and p.shown == size and p.chosen == choices), None)
        if p is None:
            p = next((p for p in X["PACKS"].values() if p.kind == kind and p.shown == size), None)
            self.unmapped[f"pack {kind} {size}/{choices}"] += 1
        cards = []
        for c in lua_list(G.pack_cards.cards):
            s = str(c.config.center.set)
            if s in ("Default", "Enhanced"):
                card = S.lab_card(c, run.next_uid)
                run.next_uid += 1
                card.debuffed = card.face_down = False
                cards.append(card)
                continue
            name = name_of(G, c)
            cls = (X["CATALOG"] if s == "Joker" else X["CONSUMABLES"]).get(name)
            if cls is None:
                self.unmapped[f"{s} {name}"] += 1
                continue
            cards.append(cls(edition_of(c)))
        op = X["OpenPack"](p, cards, int(G.GAME.pack_choices))
        run.open_pack_state = op
        if kind in ("Arcana", "Spectral"):
            run.hand = [self.by_sort.get(c.sort_id) or S.lab_card(c, 0) for c in lua_list(G.hand.cards)]
        return op

    # ---- run totals, counted from the moves played ------------------------------------------------------------
    def before(self, G, kind):
        """Call at every decision, before the move: notes entering a shop or a pack, and a won round."""
        if kind == "cash" and not self.in_cash:
            self.n["blinds_beaten"] += 1
            cr = G.GAME.current_round
            hands = G.GAME.hands
            self.last_round = {"kind": self.round_kind,
                               "hands_left": int(cr.hands_left), "discards_left": int(cr.discards_left),
                               "discards_used": int(cr.discards_used or 0), "score": int(G.GAME.chips),
                               "types": {str(k): int(v.played_this_round) for k, v in hands.items()
                                         if int(v.played_this_round)}}
        self.in_cash = kind == "cash"
        if kind == "round":
            self.round_kind = KIND.get(str(G.GAME.blind_on_deck), 0)
        if kind == "shop" and not self.in_shop:
            self.n["shops"] += 1
        self.in_shop = kind == "shop" or (kind == "pack" and self.in_shop)
        if kind == "pack":
            if self.pack_first is None:
                self.pack_first = (int(G.GAME.pack_size), int(G.GAME.pack_choices))
                self.pack_picks = 0
        else:
            self.pack_first = None

    def after(self, desc_kind, ncards=0):
        """Call after a move is chosen (harness move kind, cards selected)."""
        if desc_kind == "reroll":
            self.n["rerolls"] += 1
        elif desc_kind in ("sell_joker", "sell_consumable"):
            self.n["cards_sold"] += 1
        elif desc_kind == "discard":
            self.n["discards_made"] += 1
            self.n["cards_discarded"] += ncards
        elif desc_kind == "pick":
            self.pack_picks += 1
        elif desc_kind == "skip_pack" and self.pack_picks == 0:
            self.n["packs_skipped"] += 1


REORDERS = ("order_jokers", "order_hand", "sort_hand")


def teacher_move(bot, kind, run, extra, counts=None, drags=True):
    """Balatro Lab's bot's next move on this state, as its action tuple. A joker reorder that changes the order
    becomes one drag ("move_joker", from, to; 1-based), our action list's way of reordering; hand reorders are
    applied to the state and counted (a play's lead card carries the order that scores)."""
    run.agent = bot
    if kind == "blind":
        blind, tag = extra
        return ("skip_blind",) if bot.skip_blind(run, blind, tag) else ("select_blind",)
    if kind == "cash":
        return ("cash_out",)
    if kind == "boss":
        boss = run.ante_boss
        can = run.has_voucher("Retcon") or (run.has_voucher("Director's Cut") and not run.boss_rerolled)
        if can and run.money - 10 >= run.debt_limit and bot.reroll_boss(run, boss):
            return ("reroll_boss",)
        return ("select_blind",)
    decide = {"round": lambda: bot.round_action(run), "shop": lambda: bot.shop_action(run, extra),
              "pack": lambda: bot.pack_action(run, extra)}[kind]
    for _ in range(30):
        a = decide()
        if a and a[0] == "order_jokers":
            before = list(run.jokers)
            run.free_action(a)
            after = list(run.jokers)
            if after != before:
                if counts is not None:
                    counts["order_jokers"] += 1
                if drags:
                    # one drag at a time, as our action list has it: the first joker out of place is dragged to
                    # its place (the bot is asked again after it, on the game's new order)
                    k = next(k for k in range(len(after)) if after[k] is not before[k])
                    m = next(m for m in range(len(before)) if before[m] is after[k])
                    return ("move_joker", m + 1, k + 1)
            continue
        if a and a[0] in REORDERS:
            if counts is not None:
                counts[a[0]] += 1
            run.free_action(a)
            continue
        return a
    return None
