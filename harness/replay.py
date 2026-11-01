"""Replay an Antelytics run log through the headless game and diff state at every decision.

Each log node holds the state *before* its action (except where Antelytics writes after,
see NOTES below). For every node: compare sim vs log, then apply the logged action.

usage: python3 replay.py <run.json.gz> --src DIR [--mods DIR] [--fast] [--quiet] [--stop-on-mismatch]

NOTES on Antelytics quirks handled here:
  - tag_added / skip_blind_tag nodes are written after the skip, so their tags already
    include the new tag. Tags are only compared on other nodes.
  - Boss-blind auto-discards (auto_source, e.g. The Hook) are logged as their own nodes,
    sometimes before the play that caused them. They're consequences, not decisions.
  - The 'small' and 'big' RNG streams are drawn by Steamodded; compared only on a modded sim.
"""
import argparse
import gzip
import json
import time

from run import make_runtime

EDITIONS = {"holo": "holographic"}
PACK_STATES = {"TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK", "SMODS_BOOSTER_OPENED"}
INFO_ONLY = {"tag_added", "tag_triggered", "end_of_round", "shop_entered"}
# Antelytics writes these nodes after (or partway through) the game's own processing of the
# action rather than before it: end of round mid-cleanup, purchases after the card moves but
# before money updates, and uses/sells/rerolls after they've happened.
MID_TRANSITION = {"end_of_round", "buy_joker", "buy_consumable", "buy_voucher",
                  "use_consumable", "sell_joker", "sell_consumable", "reroll_shop"}
POST_ACTION_TAGS = {"tag_added", "skip_blind_tag", "tag_triggered"}


def nkey(k):
    """Sort key for card/item tuples that may contain None."""
    return tuple("" if x is None else str(x) for x in k)


def lua_list(t):
    return [t[i] for i in range(1, len(t) + 1)] if t is not None else []


# ---------- card identity ----------

def sim_edition(c):
    e = c.edition
    if e is None:
        return None
    for k in ("foil", "holo", "polychrome", "negative"):
        if e[k]:
            return EDITIONS.get(k, k)
    return None


def sim_card_key(c):
    enh = c.config.center.key
    enh = None if enh == "c_base" else enh[2:]
    return (c.base.value, c.base.suit, enh, sim_edition(c), c.seal)


def log_card_key(c):
    ed = c.get("edition")
    return (c["rank"], c["suit"], c.get("enhancement"), None if ed in (None, "base") else ed, c.get("seal"))


def sim_item_key(c):
    return (c.config.center.key, sim_edition(c))


def log_item_key(j):
    ed = j.get("edition")
    return (j["id"], None if ed in (None, "base") else ed)


class Replayer:
    def __init__(self, log, src, profile=None, mods=None, fast=False):
        self.log = log
        self.lua, self.H = make_runtime(src, mods)
        self.modded = bool(mods)
        self.H.start_up()
        if fast:
            self.H.strip_visuals()
        self.G = self.lua.globals().G
        self.tutorial = log["seed"] == "TUTORIAL"
        # Unlock state changes which jokers/tags/vouchers can appear. Default: Unlock All
        # profile; a TUTORIAL-seed run can only come from a fresh profile.
        self.profile = profile or ("fresh" if self.tutorial else "unlocked")
        if self.profile == "unlocked":
            self.H.unlock_all()
        if self.tutorial:
            self._tutorial_setup()
        self.H.start_run(log["deck_back"], log["stake"], log["seed"])
        self.F = self.G.FUNCS
        self.cmap = {}        # log sort_id -> sim card
        self.mapped = set()   # sim sort_ids already mapped
        self.reorders = 0
        self.last_pack_pick = None
        self.next_node = None

    # The tutorial run uses forced shop/voucher/tags that the tutorial overlay later clears.
    # We don't run the overlay, so clear them at the same points it would.
    def _tutorial_setup(self):
        lua = self.lua
        self.G.SETTINGS.tutorial_progress = lua.table(
            forced_shop=lua.table("j_joker", "c_empress"),
            forced_voucher="v_grabber",
            forced_tags=lua.table("tag_handy", "tag_garbage"),
            hold_parts=lua.table(), completed_parts=lua.table())

    def tutorial_tick(self):
        tp = self.G.SETTINGS.tutorial_progress
        if not self.tutorial or tp is None:
            return
        st = self.H.state_name()
        if st == "BLIND_SELECT" and self.G.GAME.round > 0:
            tp.forced_tags = None
        if st == "SHOP":
            tp.forced_voucher = None

    # ---------- mapping log cards to sim cards ----------
    def map_initial_deck(self, node0):
        log_deck = [c for c in node0["state"]["full_deck"] if c.get("area", "deck") == "deck"]
        for lc, sc in zip(log_deck, lua_list(self.G.deck.cards)):
            self.cmap[lc["sort_id"]] = sc
            self.mapped.add(sc.sort_id)

    # Identical copies (Cryptid, DNA) can't be told apart by rank/suit/enhancement, but the
    # game numbers cards in creation order and the sim does too, so twins pair up by sort_id
    # order. Rebuilt each node from the logged deck; enhancements and suits change mid-run.
    def sync_map(self, node):
        from collections import defaultdict
        logs, sims = defaultdict(list), defaultdict(list)
        for c in node["state"]["full_deck"]:
            logs[log_card_key(c)].append(c["sort_id"])
        for c in lua_list(self.G.playing_cards):
            sims[sim_card_key(c)].append(c)
        for k, ids in logs.items():
            sc = sims.get(k, [])
            if len(sc) == len(ids):
                for lid, c in zip(sorted(ids), sorted(sc, key=lambda c: c.sort_id)):
                    self.cmap[lid] = c
                    self.mapped.add(c.sort_id)

    def resolve(self, lc, area=None):
        sc = self.cmap.get(lc["sort_id"])
        if sc is not None and sc.area is not None and (area is None or sc.area == area):
            return sc
        if area is not None:
            want = log_card_key(lc)
            for c in lua_list(area.cards):
                if sim_card_key(c) == want and not c.highlighted:
                    self.cmap[lc["sort_id"]] = c
                    return c
        want = log_card_key(lc)
        pool = lua_list(self.G.hand.cards) + lua_list(self.G.playing_cards)
        for c in pool:
            if c.sort_id not in self.mapped and sim_card_key(c) == want:
                self.cmap[lc["sort_id"]] = c
                self.mapped.add(c.sort_id)
                return c
        for c in lua_list(self.G.hand.cards):  # fall back: a card in hand with matching key
            if sim_card_key(c) == want:
                return c
        raise LookupError(f"no sim card for {want} (log sort_id {lc['sort_id']})")

    # ---------- state ----------
    def sim_state(self):
        G = self.G
        return {
            "ante": G.GAME.round_resets.ante,
            "money": G.GAME.dollars,
            "score": G.GAME.chips,
            "hands_remaining": G.GAME.current_round.hands_left,
            "discards_remaining": G.GAME.current_round.discards_left,
            "hand": sorted((sim_card_key(c) for c in lua_list(G.hand.cards)), key=nkey),
            "deck": [sim_card_key(c) for c in lua_list(G.deck.cards)],
            "jokers": [sim_item_key(c) for c in lua_list(G.jokers.cards)],
            "consumables": sorted((sim_item_key(c) for c in lua_list(G.consumeables.cards)), key=nkey),
            "tags": [t.key for t in lua_list(G.GAME.tags)],
            "hand_levels": {k: v.level for k, v in G.GAME.hands.items()},
            "rng": {k: v for k, v in G.GAME.pseudorandom.items() if k not in ("seed", "hashed_seed")},
        }

    @staticmethod
    def log_state(s):
        fd = s["full_deck"]
        return {
            "ante": s["ante"],
            "money": s["money"],
            "score": s["score"],
            "hands_remaining": s["hands_remaining"],
            "discards_remaining": s["discards_remaining"],
            "hand": sorted((log_card_key(c) for c in fd if c.get("area") == "hand"), key=nkey),
            "deck": [log_card_key(c) for c in fd if c.get("area", "deck") == "deck"],
            "jokers": [log_item_key(j) for j in sorted(s["jokers"], key=lambda j: j.get("slot", 0))],
            "consumables": sorted((log_item_key(c) for c in s["consumables"]), key=nkey),
            "tags": [t["id"] for t in s["tags"]],
            "hand_levels": {k: v["level"] for k, v in s["hand_levels"].items()},
            "rng": s["pools"],
        }

    def diff(self, node):
        sim, log = self.sim_state(), self.log_state(node["state"])
        t, loc = node["action"]["type"], node["state"]["location"]
        if t in MID_TRANSITION or node["action"].get("auto_source"):
            return None  # Antelytics snapshot taken mid-animation; nothing stable to compare
        keys = ["money", "jokers", "consumables", "hand_levels"]
        if t not in POST_ACTION_TAGS:
            keys.append("tags")
        if loc in ("blind_select", "playing_blind"):
            keys.append("ante")   # the logged ante lags in round_eval/shop
        if loc == "playing_blind":
            keys += ["score", "hands_remaining", "discards_remaining", "hand"]
        out = [f"{k}: sim={sim[k]} log={log[k]}" for k in keys if sim[k] != log[k]]
        # Between rounds the deck's order doesn't matter: every shuffle sorts by sort_id first.
        if loc == "playing_blind" and sim["deck"] != log["deck"]:
            out.append(f"deck differs (sim {len(sim['deck'])} cards, log {len(log['deck'])})")
        for k, v in log["rng"].items():
            if k in ("small", "big") and not self.modded:
                continue  # drawn by Steamodded; absent from an unmodded sim
            sv = sim["rng"].get(k)
            if sv is None:
                out.append(f"rng stream {k}: missing in sim")
            elif abs(sv - v) > 1e-12:
                out.append(f"rng stream {k}: sim={sv} log={v}")
        return out

    # Players drag jokers into new orders; Antelytics records the order (slot) but not the drag.
    def sync_joker_order(self, node):
        want = [log_item_key(j) for j in sorted(node["state"]["jokers"], key=lambda j: j.get("slot", 0))]
        have = lua_list(self.G.jokers.cards)
        if [sim_item_key(c) for c in have] == want or len(have) != len(want):
            return
        used, order = set(), []
        for k in want:
            idx = next((i for i, c in enumerate(have) if i not in used and sim_item_key(c) == k), None)
            if idx is None:
                return
            used.add(idx)
            order.append(have[idx])
        self.G.jokers.cards = self.lua.table(*order)
        self.G.jokers.set_ranks(self.G.jokers)
        self.reorders += 1

    # ---------- actions ----------
    def e(self, card):
        return self.lua.table(config=self.lua.table(ref_table=card))

    @staticmethod
    def find(area, key, edition=None):
        for c in (lua_list(area.cards) if area is not None else []):
            if c.config.center.key == key and (edition is None or sim_edition(c) == edition):
                return c
        return None

    def highlight(self, log_cards):
        cards = []
        for lc in log_cards:
            c = self.resolve(lc, self.G.hand)
            if not c.highlighted:
                self.G.hand.add_to_highlighted(self.G.hand, c, True)
            cards.append(c)
        self.order_in_hand(cards)

    # Left-to-right order in hand matters (played-card order; Death converts the left card
    # into the right one). The player may have dragged or suit-sorted the hand; the log lists
    # selected cards left to right, so put them in that order within the slots they occupy.
    def order_in_hand(self, cards):
        hand = lua_list(self.G.hand.cards)
        ids = {x.sort_id for x in cards}
        pos = sorted(i for i, c in enumerate(hand) if c.sort_id in ids)
        if len(pos) != len(cards):
            return
        for p, c in zip(pos, cards):
            hand[p] = c
        self.G.hand.cards = self.lua.table(*hand)
        self.G.hand.set_ranks(self.G.hand)
        self.G.hand.align_cards(self.G.hand)

    def apply(self, node):
        a, G, F = node["action"], self.G, self.F
        t = a["type"]
        if t in INFO_ONLY:
            return "info"
        if t == "skip_blind_tag":
            F.skip_blind(self.lua.table(UIBox=G.blind_select_opts[a["blind_slot"]]))
        elif t == "select_blind":
            box = G.blind_select_opts[a["blind_slot"]]
            F.select_blind(box.get_UIE_by_ID(box, "select_blind_button"))
        elif t in ("discard", "play_hand"):
            if a.get("auto_source"):
                return "info"
            self.highlight(a["cards"])
            if t == "discard":
                F.discard_cards_from_highlighted(None)
            else:
                F.play_cards_from_highlighted(None)
        elif t == "blind_beaten":
            F.cash_out(self.lua.table(config=self.lua.table()))
        elif t == "ending_shop":
            F.toggle_shop(None)
        elif t == "reroll_shop":
            F.reroll_shop(None)
        elif t in ("buy_joker", "buy_consumable"):
            key = a.get("joker_id") or a.get("consumable_id")
            ed = a.get("edition")
            c = self.find(G.shop_jokers, key, None if ed in (None, "base") else ed) or self.find(G.shop_jokers, key)
            if c is None:
                raise LookupError(f"{key} not in shop")
            nxt = self.next_node["action"] if self.next_node else {}
            if t == "buy_consumable" and nxt.get("type") == "use_consumable" and nxt.get("consumable_id") == key:
                # The shop's "Buy and Use" button; Antelytics logs it as a buy then a use.
                self.highlight(nxt.get("target_cards") or [])
                e = self.e(c)
                e.config.id = "buy_and_use"
                F.buy_from_shop(e)
                self.last_pack_pick = key   # so the logged use that follows is skipped
                return "acted"
            F.buy_from_shop(self.e(c))
        elif t == "buy_voucher":
            c = self.find(G.shop_vouchers, a["voucher_id"])
            if c is None:
                raise LookupError(f"{a['voucher_id']} not in shop")
            F.use_card(self.e(c))
        elif t == "open_pack":
            c = self.find(G.shop_booster, a["pack_id"])
            if c is None:
                raise LookupError(f"{a['pack_id']} not in shop")
            F.use_card(self.e(c))
        elif t == "select_from_pack":
            c = self.find(G.pack_cards, a["selected_card_id"])
            desc = a.get("selected_description") or {}
            if c is None and "rank" in desc:   # Standard pack: a playing card, matched by rank/suit/etc.
                want = log_card_key(desc)
                c = next((x for x in lua_list(G.pack_cards.cards) if sim_card_key(x) == want), None)
            if c is None:
                raise LookupError(f"{a['selected_card_id']} not in pack")
            # A tarot/spectral picked from a pack is used on the spot; Antelytics logs its
            # targets on the use_consumable node that follows, so select them first.
            nxt = self.next_node["action"] if self.next_node else {}
            if nxt.get("type") == "use_consumable" and nxt.get("consumable_id") == a["selected_card_id"]:
                self.highlight(nxt.get("target_cards") or [])
            F.use_card(self.e(c))
            self.last_pack_pick = a["selected_card_id"]
            return "acted"
        elif t == "ending_pack":
            if self.H.state_name() not in PACK_STATES:
                return "info"  # pack already closed itself after the last pick
            F.skip_booster(None)
        elif t == "use_consumable":
            c = self.find(G.consumeables, a["consumable_id"])
            if c is None and self.last_pack_pick == a["consumable_id"]:
                self.last_pack_pick = None
                return "info"  # the pack pick already used it
            if c is None:
                raise LookupError(f"{a['consumable_id']} not held")
            self.highlight(a.get("target_cards") or [])
            F.use_card(self.e(c))
        elif t in ("sell_joker", "sell_consumable"):
            area = G.jokers if t == "sell_joker" else G.consumeables
            c = self.find(area, a.get("joker_id") or a.get("consumable_id"))
            if c is None:
                raise LookupError("card to sell not found")
            F.sell_card(self.e(c))
        else:
            raise NotImplementedError(t)
        self.last_pack_pick = None
        return "acted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--src", required=True,
                    help="game source; for a modded game, vanilla source overlaid with Lovely's dump")
    ap.add_argument("--mods", help="Mods folder with Steamodded (omit for the unmodded game)")
    ap.add_argument("--quiet", action="store_true", help="only print mismatching nodes")
    ap.add_argument("--stop-on-mismatch", action="store_true")
    ap.add_argument("--fast", action="store_true", help="strip visual-only work (see boot.lua strip_visuals)")
    ap.add_argument("--profile", choices=["unlocked", "fresh"], help="default: fresh for TUTORIAL seed, else unlocked")
    args = ap.parse_args()

    log = json.load(gzip.open(args.log))
    t0 = time.time()
    R = Replayer(log, args.src, args.profile, args.mods, args.fast)
    nodes = sorted(log["nodes"], key=lambda n: n["index"])
    R.map_initial_deck(nodes[0])
    print(f"run {log['run_id']}  seed {log['seed']}  {log['deck_back']}  stake {log['stake']}  "
          f"{len(nodes)} decisions  outcome {log['outcome']}")

    bad, frames, acted, first_bad, stopped, skipped = 0, 0, 0, None, None, 0
    for i, node in enumerate(nodes):
        a = node["action"]
        R.next_node = nodes[i + 1] if i + 1 < len(nodes) else None
        R.sync_joker_order(node)
        R.sync_map(node)
        problems = R.diff(node)
        if problems is None:
            skipped += 1
            problems = []
        elif problems:
            bad += 1
            first_bad = node["index"] if first_bad is None else first_bad
        if problems or not args.quiet:
            print(f"[{node['index']:3d}] {a['type']:18s} {'OK' if not problems else 'MISMATCH'}")
            for p in problems:
                print("        " + p)
        if problems and args.stop_on_mismatch:
            stopped = node["index"]
            break
        try:
            r = R.apply(node)
        except (LookupError, NotImplementedError) as ex:
            print(f"[{node['index']:3d}] cannot apply {a['type']}: {ex}; stopping")
            stopped = node["index"]
            break
        if r == "acted":
            acted += 1
            n = R.H.settle()
            if n < 0:
                print(f"[{node['index']:3d}] game never settled after {a['type']}; stopping")
                stopped = node["index"]
                break
            frames += n
            R.tutorial_tick()

    el = time.time() - t0
    end = R.sim_state()
    print(f"\nfinal sim: state {R.H.state_name()}  ante {end['ante']}  money {end['money']}  "
          f"won {bool(R.G.GAME.won)}  (log outcome: {log['outcome']}, final ante {log['final_ante']})")
    print(f"profile: {R.profile}   game: {'Steamodded ' + str(R.lua.eval('SMODS.version')) if args.mods else 'unmodded'}")
    print(f"mismatching nodes: {bad}/{len(nodes) - skipped} compared" + (f"  (first at {first_bad})" if first_bad is not None else "")
          + f"; {skipped} mid-transition snapshots not compared")
    if stopped is not None:
        print(f"stopped at node {stopped}")
    print(f"joker reorders applied from log: {R.reorders}")
    print(f"{acted} actions, {frames} frames, {el:.1f}s total ({el / max(acted, 1) * 1000:.0f} ms/action)")


if __name__ == "__main__":
    main()
