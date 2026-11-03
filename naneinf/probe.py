"""Checks community naneinf seeds' openings on the pinned game (1.0.1o, unmodded, all unlocked).

For each seed and deck, plays a few fixed opening lines and records what the player would see:
skip tags, pack contents (and the legendary a Soul card turns into), shop jokers, vouchers,
boosters, and the shop after rerolls. Plays are a plain best-HUD-hand player, only to get past
the first blinds; the point is the seed's contents, not the play.

usage: python3 probe.py SEEDS.json OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from sim import Sim  # noqa: E402

SRC = os.environ.get("BALATRO_SRC", os.path.expanduser("~/balatro/local/source"))
INTEREST = {"j_perkeo", "j_triboulet", "j_chicot", "j_yorick", "j_caino", "j_baron", "j_mime", "j_blueprint",
            "j_brainstorm", "j_invisible", "j_showman", "c_soul", "c_ankh", "c_cryptid", "c_deja_vu",
            "c_ectoplasm", "c_hex", "v_overstock_norm", "v_blank", "j_dna"}


class P:
    def __init__(self):
        self.s = Sim(SRC, fast=True, mode="ordered", enc=2)
        self.H, self.G = self.s.H, self.s.env.lua.globals().G

    def keys(self, area):
        a = getattr(self.G, area, None)
        if not a or not a.cards:
            return []
        out = []
        for i in range(1, len(a.cards) + 1):
            c = a.cards[i]
            k = c.config.center.key
            if c.ability.set in ("Default", "Enhanced"):
                k = f"{c.base.value} of {c.base.suit}" + (f" {k}" if k != "c_base" else "") + (f" {c.seal} seal" if c.seal else "")
            ed = c.edition
            if ed:
                for e in ("negative", "polychrome", "holo", "foil"):
                    if ed[e]:
                        k += f" ({e})"
            out.append(k)
        return out

    def acts(self):
        self.s.observe()
        a = self.H._encoded_actions
        return [a[i] for i in range(1, len(a) + 1)]

    def do(self, pred):
        for i, a in enumerate(self.acts()):
            if pred(a):
                self.s.step(i)
                return True
        return False

    def state(self):
        return self.H.state_name()

    def reset(self, seed, back):
        self.s.reset(seed, back, 1)
        self.log = []

    def tag_on_deck(self):
        rr = self.G.GAME.round_resets
        deck = self.G.GAME.blind_on_deck
        return rr.blind_tags[deck] if rr.blind_tags else None

    def take_pack(self, prefer_soul=True):
        """In an open pack: record contents; take The Soul if there (record the legendary), else skip."""
        cont = self.keys("pack_cards")
        note = {"pack": cont}
        if prefer_soul and "c_soul" in cont:
            i = cont.index("c_soul") + 1
            before = set(self.keys("jokers"))
            if self.do(lambda a: a.type == "pick" and a.i == i):
                note["soul_gave"] = [k for k in self.keys("jokers") if k not in before]
        while self.state().endswith("PACK"):
            if not self.do(lambda a: a.type == "skip_pack"):
                if not self.do(lambda a: a.type == "pick"):
                    break
        return note

    def play_blind(self):
        """Best-HUD-hand player until the blind is over. Returns True if it reached the shop."""
        for _ in range(60):
            st = self.state()
            if st == "ROUND_EVAL":
                self.do(lambda a: a.type == "cash_out")
                continue
            if st == "SHOP":
                return True
            if st != "SELECTING_HAND":
                return False
            o = self.s.observe()
            best, bi = -1, None
            for i, row in enumerate(o["act"]):
                if row[0] == 1 and int(row[16]) + int(row[17]) > best:
                    best, bi = int(row[16]) + int(row[17]), i
            if bi is None:
                return False
            self.s.step(bi)
        return self.state() == "SHOP"

    def shop(self):
        return {"jokers": self.keys("shop_jokers"), "vouchers": self.keys("shop_vouchers"),
                "boosters": self.keys("shop_booster"), "money": self.G.GAME.dollars}


def line_skips(p, seed, back):
    """Skip Small and Big; take what the tags give; play the Boss; look at the shop."""
    p.reset(seed, back)
    out = {}
    for name in ("small", "big"):
        tag = p.tag_on_deck()
        if not p.do(lambda a: a.type == "skip_blind"):
            break
        out[f"skip_{name}_tag"] = tag
        if p.state().endswith("PACK"):
            out[f"skip_{name}_tag_pack"] = p.take_pack()
    out["jokers_after_skips"] = p.keys("jokers")
    p.do(lambda a: a.type == "select_blind")
    if p.play_blind():
        out["shop_after_boss"] = p.shop()
    else:
        out["boss"] = "lost or stuck at " + p.state()
    return out


def line_play_small(p, seed, back):
    """Play the Small Blind; look at the first shop, each of its packs, and up to 3 rerolls."""
    def to_shop():
        p.reset(seed, back)
        p.do(lambda a: a.type == "select_blind")
        return p.play_blind()
    out = {}
    if not to_shop():
        return {"small": "lost or stuck"}
    out["shop1"] = p.shop()
    nb = len(out["shop1"]["boosters"])
    for b in range(1, nb + 1):
        to_shop()
        if p.do(lambda a, b=b: a.type == "open_pack" and a.i == b):
            out[f"shop1_pack{b}"] = p.take_pack()
    to_shop()
    rr = []
    for _ in range(3):
        if not p.do(lambda a: a.type == "reroll"):
            break
        rr.append(p.keys("shop_jokers"))
    out["shop1_rerolls"] = rr
    # Big Blind's skip tag from the first shop's point of view
    p.do(lambda a: a.type == "end_shop")
    out["big_skip_tag"] = p.tag_on_deck()
    return out


def flag(d):
    hits = set()
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for k in INTEREST:
                if k in x:
                    hits.add(k)
    walk(d)
    return sorted(hits)


def main():
    seeds = json.load(open(sys.argv[1]))
    p = P()
    res = []
    for s in seeds:
        r = dict(s)
        for name, f in (("skip_skip", line_skips), ("play_small", line_play_small)):
            try:
                r[name] = f(p, s["seed"], s["back"])
            except Exception as e:      # a line that the seed doesn't allow (e.g. no money) is data
                r[name] = {"error": str(e)[-300:]}
        r["seen"] = flag({k: r[k] for k in ("skip_skip", "play_small")})
        res.append(r)
        print(s["seed"], s["back"], r["seen"], flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
