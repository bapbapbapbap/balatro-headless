"""Checks env.lua against a logged run.

--mode controls (default): drives the whole run through env actions in controls mode. Joker drags
  become move_joker, hand order becomes move_hand, selections become toggles, then play / discard
  / use / pick. Nothing is set directly. Every action must be in env.legal_actions() when taken,
  and the state is compared with the log at every decision, as replay.py does.
--mode subsets: replays with replay.py's own actions and checks the logged action is among the
  subsets-mode legal actions at every decision.

usage: python3 check_env.py <run.json.gz> --src DIR [--mods DIR] [--fast] [--mode controls|subsets]
"""
import argparse
import collections
import gzip
import json
import os

from replay import Replayer, lua_list, log_card_key, sim_card_key, log_item_key, sim_item_key

HERE = os.path.dirname(os.path.abspath(__file__))
PACKS = ("TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK")
INFO = ("tag_added", "tag_triggered", "end_of_round", "shop_entered")


class Illegal(Exception):
    pass


# Hand positions (1-based) of logged cards, in the log's left-to-right order. Identical copies
# (Cryptid) are interchangeable: each takes its mapped slot if free, else any identical one.
def hand_positions(R, cards, ordered=False):
    hand = lua_list(R.G.hand.cards)
    used, out = set(), []
    for lc in cards:
        c = R.resolve(lc, R.G.hand)
        pos = next((i for i, h in enumerate(hand) if h.sort_id == c.sort_id and i not in used), None)
        if pos is None:
            want = log_card_key(lc)
            pos = next(i for i, h in enumerate(hand) if i not in used and sim_card_key(h) == want)
        used.add(pos)
        out.append(pos + 1)
    return tuple(out) if ordered else tuple(sorted(out))


def index_of(area, key):
    for i, c in enumerate(lua_list(area.cards) if area is not None else []):
        if c.config.center.key == key:
            return i + 1
    return None


def wait_block_play(R):
    G, n = R.G, 0
    while R.H.state_name() == "SELECTING_HAND" and G.GAME.blind and G.GAME.blind.block_play and n < 600:
        R.lua.globals().HEADLESS_ADVANCE_CLOCK(1 / 30)
        G.update(G, 1 / 30)
        n += 1


# ---------------- controls mode: drive everything through env actions ----------------

class Driver:
    def __init__(self, R):
        self.R, self.H, self.G = R, R.H, R.G
        self.steps = collections.Counter()

    def do(self, t, i=None, j=None):
        legal = {}
        raw = self.H.legal_actions()
        for k in range(1, len(raw) + 1):
            legal[(raw[k].type, raw[k].i, raw[k].j)] = raw[k]
        a = legal.get((t, i, j))
        if a is None:
            types = collections.Counter(k[0] for k in legal)
            raise Illegal(f"{(t, i, j)} not legal in {self.H.state_name()}; offered {dict(types)}")
        if self.H.step(a) < 0:
            raise RuntimeError(f"never settled after {(t, i, j)}")
        self.steps[t] += 1

    def reorder(self, area, kind, want_ids):
        """Realize an order (list of sort_ids) with move actions, selection-sort style."""
        for k, sid in enumerate(want_ids):
            have = [c.sort_id for c in lua_list(area.cards)]
            if have[k] == sid:
                continue
            src = have.index(sid, k)
            self.do(kind, src + 1, k + 1)

    def sync_jokers(self, node):
        want = [log_item_key(j) for j in sorted(node["state"]["jokers"], key=lambda j: j.get("slot", 0))]
        have = lua_list(self.G.jokers.cards)
        if [sim_item_key(c) for c in have] == want or len(have) != len(want):
            return
        used, ids = set(), []
        for key in want:
            idx = next((i for i, c in enumerate(have) if i not in used and sim_item_key(c) == key), None)
            if idx is None:
                return
            used.add(idx)
            ids.append(have[idx].sort_id)
        self.reorder(self.G.jokers, "move_joker", ids)

    def select(self, log_cards):
        """Put the logged cards in the log's left-to-right order within the slots they occupy
        (as replay.order_in_hand does), then select exactly them."""
        if not log_cards:
            targets = []
        else:
            pos = hand_positions(self.R, log_cards, ordered=True)
            hand = lua_list(self.G.hand.cards)
            targets = [hand[p - 1] for p in pos]
            ids = [c.sort_id for c in hand]
            for slot, c in zip(sorted(pos), targets):
                ids[slot - 1] = c.sort_id
            self.reorder(self.G.hand, "move_hand", ids)
        want = {c.sort_id for c in targets}
        for k, c in enumerate(lua_list(self.G.hand.cards)):     # deselect first: limit is 5
            if c.highlighted and c.sort_id not in want and not c.ability.forced_selection:
                self.do("toggle", k + 1)
        for k, c in enumerate(lua_list(self.G.hand.cards)):
            if not c.highlighted and c.sort_id in want:
                self.do("toggle", k + 1)

    def apply(self, node, nxt):
        a, G, R = node["action"], self.G, self.R
        t = a["type"]
        nt = nxt["action"] if nxt else {}
        if t in INFO or a.get("auto_source"):
            return
        if t == "skip_blind_tag":
            self.do("skip_blind")
        elif t == "select_blind":
            self.do("select_blind")
        elif t in ("play_hand", "discard"):
            self.select(a["cards"])
            self.do("play" if t == "play_hand" else "discard")
        elif t == "blind_beaten":
            self.do("cash_out")
        elif t == "ending_shop":
            self.do("end_shop")
        elif t == "reroll_shop":
            self.do("reroll")
        elif t in ("buy_joker", "buy_consumable"):
            key = a.get("joker_id") or a.get("consumable_id")
            ed = a.get("edition")
            c = R.find(G.shop_jokers, key, None if ed in (None, "base") else ed) or R.find(G.shop_jokers, key)
            i = next(k + 1 for k, x in enumerate(lua_list(G.shop_jokers.cards)) if x.sort_id == c.sort_id)
            if t == "buy_consumable" and nt.get("type") == "use_consumable" and nt.get("consumable_id") == key:
                self.do("buy_and_use", i)
                R.last_pack_pick = key     # the logged use that follows is this same card
                return
            else:
                self.do("buy", i)
        elif t == "buy_voucher":
            self.do("redeem", index_of(G.shop_vouchers, a["voucher_id"]))
        elif t == "open_pack":
            self.do("open_pack", index_of(G.shop_booster, a["pack_id"]))
        elif t == "select_from_pack":
            c = R.find(G.pack_cards, a["selected_card_id"])
            desc = a.get("selected_description") or {}
            if c is None and "rank" in desc:
                want = log_card_key(desc)
                c = next((x for x in lua_list(G.pack_cards.cards) if sim_card_key(x) == want), None)
            if c is None:
                raise LookupError(f"{a['selected_card_id']} not in pack")
            if nt.get("type") == "use_consumable" and nt.get("consumable_id") == a["selected_card_id"]:
                self.select(nt.get("target_cards") or [])
            i = next(k + 1 for k, x in enumerate(lua_list(G.pack_cards.cards)) if x.sort_id == c.sort_id)
            self.do("pick", i)
            R.last_pack_pick = a["selected_card_id"]
            return
        elif t == "ending_pack":
            if self.H.state_name() in PACKS:
                self.do("skip_pack")
        elif t == "use_consumable":
            i = index_of(G.consumeables, a["consumable_id"])
            if i is None and R.last_pack_pick == a["consumable_id"]:
                R.last_pack_pick = None
                return
            if i is None:
                raise LookupError(f"{a['consumable_id']} not held")
            self.select(a.get("target_cards") or [])
            self.do("use_consumable", i)
        elif t == "sell_joker":
            self.do("sell_joker", index_of(G.jokers, a["joker_id"]))
        elif t == "sell_consumable":
            self.do("sell_consumable", index_of(G.consumeables, a["consumable_id"]))
        else:
            raise NotImplementedError(t)
        R.last_pack_pick = None


def run_controls(R, nodes):
    D = Driver(R)
    state_bad, decisions, stopped = 0, 0, None
    for k, node in enumerate(nodes):
        nxt = nodes[k + 1] if k + 1 < len(nodes) else None
        R.next_node = nxt
        try:
            D.sync_jokers(node)
            R.sync_map(node)
            wait_block_play(R)
            problems = [p for p in (R.diff(node) or []) if R.modded or "missing in sim" not in p]
            if problems:
                state_bad += 1
                print(f"[{node['index']:3d}] state mismatch: {problems[:3]}")
            before = sum(D.steps.values())
            D.apply(node, nxt)
            decisions += sum(D.steps.values()) > before
        except (Illegal, LookupError, NotImplementedError, RuntimeError) as ex:
            print(f"[{node['index']:3d}] {node['action']['type']}: {ex}; stopping")
            stopped = node["index"]
            break
        R.tutorial_tick()
    print(f"\ncontrols mode: {decisions} logged decisions driven through {sum(D.steps.values())} env actions"
          + (f"; STOPPED at node {stopped}" if stopped is not None else ""))
    print(f"state mismatches: {state_bad}")
    print("env actions used:", dict(D.steps.most_common()))
    print(f"final: {R.H.state_name()}  ante {R.G.GAME.round_resets.ante}  won {bool(R.G.GAME.won)}")
    return stopped is None and state_bad == 0


# ---------------- subsets mode: logged action must be in the legal set ----------------

def expected(R, node, nxt):
    a, G = node["action"], R.G
    t = a["type"]
    if t in INFO or a.get("auto_source"):
        return None
    simple = {"skip_blind_tag": "skip_blind", "select_blind": "select_blind", "blind_beaten": "cash_out",
              "ending_shop": "end_shop", "reroll_shop": "reroll"}
    if t in simple:
        return (simple[t], None, ())
    if t in ("play_hand", "discard"):
        return ("play" if t == "play_hand" else "discard", None, hand_positions(R, a["cards"]))
    if t in ("buy_joker", "buy_consumable"):
        key = a.get("joker_id") or a.get("consumable_id")
        nt = nxt["action"] if nxt else {}
        kind = "buy_and_use" if (t == "buy_consumable" and nt.get("type") == "use_consumable"
                                 and nt.get("consumable_id") == key) else "buy"
        return (kind, index_of(G.shop_jokers, key), None)
    if t == "buy_voucher":
        return ("redeem", index_of(G.shop_vouchers, a["voucher_id"]), None)
    if t == "open_pack":
        return ("open_pack", index_of(G.shop_booster, a["pack_id"]), None)
    if t == "select_from_pack":
        i = index_of(G.pack_cards, a["selected_card_id"])
        if i is None:
            return ("pick", "playing card", None)
        nt = nxt["action"] if nxt else {}
        cards = ()
        if nt.get("type") == "use_consumable" and nt.get("consumable_id") == a["selected_card_id"]:
            cards = hand_positions(R, nt.get("target_cards") or [])
        return ("pick", i, cards)
    if t == "ending_pack":
        return ("skip_pack", None, ()) if R.H.state_name() in PACKS else None
    if t == "use_consumable":
        i = index_of(G.consumeables, a["consumable_id"])
        if i is None:
            return None
        return ("use_consumable", i, hand_positions(R, a.get("target_cards") or []))
    if t == "sell_joker":
        return ("sell_joker", index_of(G.jokers, a["joker_id"]), None)
    if t == "sell_consumable":
        return ("sell_consumable", index_of(G.consumeables, a["consumable_id"]), None)
    raise NotImplementedError(t)


def legal(R):
    raw = R.H.legal_actions()
    out = []
    for i in range(1, len(raw) + 1):
        a = raw[i]
        cards = tuple(a.cards[j] for j in range(1, len(a.cards) + 1)) if a.cards else ()
        out.append((a.type, a.i, cards))
    return out


def matches(exp, acts):
    t, i, cards = exp
    return any(a[0] == t and (i == "playing card" or i is None or a[1] == i) and (cards is None or a[2] == cards)
               for a in acts)


def run_subsets(R, nodes):
    checked, missing, sizes, state_bad = 0, [], collections.defaultdict(list), 0
    for k, node in enumerate(nodes):
        R.next_node = nodes[k + 1] if k + 1 < len(nodes) else None
        R.sync_joker_order(node)
        R.sync_map(node)
        wait_block_play(R)
        problems = [p for p in (R.diff(node) or []) if R.modded or "missing in sim" not in p]
        if problems:
            state_bad += 1
            print(f"[{node['index']:3d}] state mismatch: {problems[:3]}")
        exp = expected(R, node, R.next_node)
        if exp is not None:
            acts = legal(R)
            checked += 1
            sizes[R.H.state_name()].append(len(acts))
            if not matches(exp, acts):
                missing.append(node["index"])
                print(f"[{node['index']:3d}] {node['action']['type']}: expected {exp} not legal; "
                      f"offered {dict(collections.Counter(a[0] for a in acts))}")
        if R.apply(node) == "acted" and R.H.settle() < 0:
            print(f"[{node['index']:3d}] never settled; stopping")
            break
        R.tutorial_tick()
    print(f"\nsubsets mode: {checked} logged decisions checked, {len(missing)} not in the legal set"
          + (f" (nodes {missing[:10]})" if missing else ""))
    print(f"state mismatches: {state_bad}")
    print("legal actions offered, by state (min / median / max):")
    for st, v in sorted(sizes.items()):
        v = sorted(v)
        print(f"  {st:16s} n={len(v):3d}  {v[0]} / {v[len(v) // 2]} / {v[-1]}")
    return not missing and state_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--src", required=True)
    ap.add_argument("--mods")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--profile", choices=["unlocked", "fresh"])
    ap.add_argument("--mode", choices=["controls", "subsets"], default="controls")
    args = ap.parse_args()

    log = json.load(gzip.open(args.log))
    R = Replayer(log, args.src, args.profile, args.mods, args.fast)
    R.H.mode = args.mode
    R.lua.execute(open(os.path.join(HERE, "env.lua")).read(), R.H)
    nodes = sorted(log["nodes"], key=lambda n: n["index"])
    R.map_initial_deck(nodes[0])
    print(f"run {log['run_id']}  seed {log['seed']}  {log['deck_back']}  stake {log['stake']}  {len(nodes)} nodes")
    ok = (run_controls if args.mode == "controls" else run_subsets)(R, nodes)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
