"""Summarize mine.py's output: where our bot and Balatro Lab's bot choose differently, by decision type.

Ours (Claude's, reviewed by chud). Reads only mine.py's jsonl; no Balatro Lab code.

    python3 mine_report.py mine*.jsonl
"""
import json
import sys
from collections import Counter, defaultdict


def verb(meaning):
    return meaning.split(" ", 1)[0] if meaning else "?"


def shop_class(meaning, name_kind):
    v = verb(meaning)
    if v == "buy":
        return "buy " + name_kind(meaning.split(" ", 1)[1])
    return {"open": "open pack", "redeem": "redeem voucher", "buy_use": "buy+use", "reroll": "reroll",
            "leave": "leave"}.get(v, v)


PLANETS = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Planet X",
           "Ceres", "Eris"}


def name_kind(name):
    if name in PLANETS:
        return "planet"
    if name.startswith("The ") or name in ("Judgement", "Justice", "Strength", "Temperance", "Death"):
        return "tarot"
    if name[:-1].isdigit() or "/" in name or (name[:2].isdigit()):
        return "playing card"
    return "joker"


def main():
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
    seeds = {r["seed"] for r in rows}
    print(f"{len(rows)} decision points, {len(seeds)} seeds; agree {sum(r['agree'] for r in rows)} "
          f"({100 * sum(r['agree'] for r in rows) / max(1, len(rows)):.0f}%)\n")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for kind in ("blind", "round", "shop", "pack"):
        rs = by.get(kind, [])
        if not rs:
            continue
        a = sum(r["agree"] for r in rs)
        print(f"== {kind}: {len(rs)} points, agree {a} ({100 * a / len(rs):.0f}%)")
        if kind == "round":
            m = Counter((verb(r["ours"]), verb(r["theirs"])) for r in rs)
            print("   (ours, theirs):", dict(m))
            same_verb = [r for r in rs if verb(r["ours"]) == verb(r["theirs"]) == "play" and not r["agree"]]
            sub = sum(set(r["ours"].split()[1:]) >= set(r["theirs"].split()[1:]) or
                      set(r["theirs"].split()[1:]) >= set(r["ours"].split()[1:]) for r in same_verb)
            print(f"   both play, different cards: {len(same_verb)} (one set contains the other, i.e. kickers: {sub})")
            uses = sum(bool(r["theirs_free"]) for r in rs)
            ours_uses = sum(bool(r["ours_free"]) for r in rs)
            print(f"   uses/sells a card first: theirs {uses}, ours {ours_uses}")
        elif kind == "shop":
            m = Counter((shop_class(r["ours"], name_kind), shop_class(r["theirs"], name_kind)) for r in rs
                        if not r["agree"])
            for (o, t), n in m.most_common(14):
                print(f"   ours {o:<16} theirs {t:<16} {n}")
            tf = Counter(verb(x) for r in rs for x in r["theirs_free"])
            of = Counter(verb(x) for r in rs for x in r["ours_free"])
            print(f"   sells/uses first: theirs {dict(tf)}, ours {dict(of)}")
            buys_t = Counter(r["theirs"].split(" ", 1)[1] for r in rs if verb(r["theirs"]) in ("buy", "redeem"))
            buys_o = Counter(r["ours"].split(" ", 1)[1] for r in rs if verb(r["ours"]) in ("buy", "redeem"))
            print("   theirs buys most:", buys_t.most_common(10))
            print("   ours buys most:  ", buys_o.most_common(10))
        else:
            m = Counter((verb(r["ours"]), verb(r["theirs"])) for r in rs if not r["agree"])
            print("   disagreements (ours, theirs):", dict(m))
        print()


if __name__ == "__main__":
    main()
