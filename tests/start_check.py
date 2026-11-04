"""Checks that every deck and every challenge starts as itself, and that encoding 3 shows it.

For each deck: a run starts, the observation names that deck, and the encoding-3 tokens include
the deck's token. For each challenge: the run names the challenge, uses stake 1, holds at least the
challenge's starting jokers, and the encoding includes the challenge's token. Also lists every
place the pinned source reads GAME.seeded (save mode clears that flag; see boot.lua start_run).

usage: python3 tests/start_check.py --src ~/balatro/local/source
Last line is PASS or FAIL.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from sim import Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    sim = Sim(a.src, mode="ordered-capped", enc=3)
    env, G = sim.env, sim.env.lua.globals().G
    fails = []
    for deck in env.deck_keys():
        o = sim.reset("STARTCHK", deck, 1)
        seen = env.observe().get("back")
        if seen != deck:
            fails.append(f"deck {deck}: run reports deck {seen}")
        elif sim.vocab.index(deck) not in set(int(x) for x in o["vocab"]):
            fails.append(f"deck {deck}: no deck token in encoding 3")
        print(f"deck {deck}: {'ok' if not fails or not fails[-1].startswith('deck ' + deck) else 'FAIL'}")
    for ch in env.challenge_ids():
        o = sim.reset("STARTCHK", "b_red", 1, ch)
        ob = env.observe()
        spec = next(c for c in sim.env.lua.globals().G.CHALLENGES.values() if c.id == ch)
        want_jokers = len(spec.jokers) if spec.jokers else 0
        problems = []
        if ob.get("challenge") != ch:
            problems.append(f"run reports challenge {ob.get('challenge')}")
        if ob.get("stake") != 1:
            problems.append(f"stake {ob.get('stake')}")
        if len(ob.get("jokers") or []) < want_jokers:
            problems.append(f"{len(ob.get('jokers') or [])} jokers, challenge starts with {want_jokers}")
        tok = "challenge:" + ch
        if tok not in sim.vocab or sim.vocab.index(tok) not in set(int(x) for x in o["vocab"]):
            problems.append("no challenge token in encoding 3")
        fails += [f"challenge {ch}: {p}" for p in problems]
        print(f"challenge {ch} (deck {ob.get('back')}): {'ok' if not problems else 'FAIL ' + '; '.join(problems)}")
    print("\nplaces the pinned source reads GAME.seeded (save mode clears it after a run starts):")
    for f in sorted(glob.glob(os.path.join(a.src, "**", "*.lua"), recursive=True)):
        for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
            if "GAME.seeded" in line:
                print(f"  {os.path.relpath(f, a.src)}:{i}: {line.strip()[:110]}")
    for f in fails[:20]:
        print("  " + f)
    print("PASS" if not fails else f"FAIL ({len(fails)} problems)")


if __name__ == "__main__":
    main()
