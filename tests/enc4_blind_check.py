"""Encoding 4's checks (docs/hypotheses-v6.md, 2026-11-27 20:14 section): the blinds as the player sees them.

B1, requirement: replaying routes at encoding 4, the requirement each choice's BLIND token carries (feature 14) in the
    shop and at blind select is the game's own G.GAME.blind.chips once that blind is selected. Mismatches listed.
B2, effects: BLIND_TEXT (harness/encode.lua) names every blind in G.P_BLINDS exactly once; each boss's effect text
    and the slots set for it are printed for review.
B3, no change below encoding 4: encodings 1-3 give identical tokens with this code and with --ref (a checkout
    without encoding 4), route by route, move by move.

  python3 tests/enc4_blind_check.py <game source> [--routes 12] [--stake 8] [--ref <other repo>]
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SLOT_NAMES = {17: "one hand", 18: "no discards", 19: "Clubs debuffed", 20: "Spades debuffed", 21: "Hearts debuffed",
              22: "Diamonds debuffed", 23: "faces debuffed", 24: "drawn face down", 25: "-1 hand size",
              26: "must play 5", 27: "hand type limited", 28: "hand weakened", 29: "money lost", 30: "hand disrupted",
              31: "cards debuffed by play/sale", 32: "jokers disrupted"}


def routes(stake, n):
    rs = [json.loads(l) for l in open(os.path.join(ROOT, "wins", "routes.jsonl"))
          if '"ordered-capped"' in l and f'"stake": {stake}' in l]
    step = max(1, len(rs) // n)
    return rs[::step][:n]


def slog(x):
    return math.log(1 + x) / 10


def b1(src, rs):
    sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "harness"))
    from sim import Sim
    bad, checked = [], 0
    for r in rs:
        s = Sim(src, mode="ordered-capped", enc=4)
        s.reset(r["seed"], r["back"], r["stake"])
        L = s.env.lua
        seen = {}                                  # slot -> requirement shown, from the last shop / blind select
        for k, act in enumerate(r["acts"]):
            st = s.H.state_name()
            ob = s.observe()
            if st in ("SHOP", "BLIND_SELECT"):
                on = L.eval("G.GAME.blind_on_deck or ''")
                amt = L.eval(f"G.GAME.round_resets.blind_choices['{on}'] and "
                             f"get_blind_amount(G.GAME.round_resets.blind_ante) * G.P_BLINDS[G.GAME.round_resets.blind_choices['{on}']].mult"
                             f" * G.GAME.starting_params.ante_scaling or -1") if on else -1
                # the token carries it: some BLIND token (kind 4) has feature 14 = slog(amt)
                feats = ob["feat"][ob["kind"] == 4]
                if amt and amt > 0 and not any(abs(float(f[13]) - slog(amt)) < 2e-3 for f in feats):
                    bad.append((r["seed"], k, st, "token lacks requirement", amt))
                if on:
                    seen[on] = (amt, st, k)
            was = st
            _, _, done = s.step(act)
            if done:
                break
            if was == "BLIND_SELECT" and s.H.state_name() == "SELECTING_HAND":
                on = L.eval("G.GAME.blind and G.GAME.blind:get_type() or ''")
                chips = L.eval("G.GAME.blind.chips")
                if on in seen:
                    checked += 1
                    if abs(seen[on][0] - chips) > 1e-6 * max(1, chips):
                        bad.append((r["seed"], k, on, f"shown {seen[on][0]} ({seen[on][1]} move {seen[on][2]})", chips))
    print(f"B1: {checked} blinds checked on {len(rs)} routes, {len(bad)} mismatches")
    for x in bad[:20]:
        print("   ", x)
    return not bad


def b2(src):
    sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "harness"))
    from sim import Sim
    s = Sim(src, mode="ordered-capped", enc=4)
    s.reset("AAAAAAAA", "b_red", 1)
    L = s.env.lua
    keys = L.eval("(function() local t = {} for k in pairs(G.P_BLINDS) do t[#t + 1] = k end table.sort(t) "
                  "return table.concat(t, ',') end)()").split(",")
    have = L.execute("local H = ...; local t = {} for k in pairs(H.BLIND_TEXT) do t[#t + 1] = k end table.sort(t) "
                     "return table.concat(t, ',')", s.H).split(",")
    ok = sorted(keys) == sorted(have)
    print(f"B2: G.P_BLINDS {len(keys)}, BLIND_TEXT {len(have)}: {'same keys' if ok else 'DIFFERENT'}")
    if not ok:
        print("   missing:", sorted(set(keys) - set(have)), "extra:", sorted(set(have) - set(keys)))
    for k in keys:
        text = L.eval(f"(function() local d = G.localization.descriptions.Blind['{k}'] "
                      f"return d and table.concat(d.text or {{}}, ' ') or '' end)()")
        slots = L.execute(f"local H = ...; local t = {{}} for _, i in ipairs(H.BLIND_TEXT['{k}'] or {{}}) do t[#t + 1] = i end "
                          f"return table.concat(t, ',')", s.H)
        mult = L.eval(f"G.P_BLINDS['{k}'].mult")
        names = ", ".join(SLOT_NAMES[int(i)] for i in slots.split(",") if i) or "-"
        print(f"   {k:16s} x{mult:<4g} {text[:52]:52s} -> {names}")
    return ok


def token_hashes(root, src, rs, enc):
    code = f"""
import sys, json, hashlib
sys.path.insert(0, {os.path.join(root, 'train')!r}); sys.path.insert(0, {os.path.join(root, 'harness')!r})
from sim import Sim
out = []
for r in json.loads(sys.stdin.read()):
    s = Sim({src!r}, mode="ordered-capped", enc={enc})
    s.reset(r["seed"], r["back"], r["stake"])
    h = hashlib.sha256()
    for a in r["acts"]:
        ob = s.observe()
        for k in ("kind", "vocab", "feat", "act"):
            h.update(ob[k].tobytes())
        _, _, done = s.step(a)
        if done:
            break
    out.append(h.hexdigest())
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", code], input=json.dumps(rs), capture_output=True, text=True, cwd=root)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def b3(src, rs, ref):
    ok = True
    for enc in (1, 2, 3):
        a, b = token_hashes(ROOT, src, rs, enc), token_hashes(ref, src, rs, enc)
        same = sum(x == y for x, y in zip(a, b))
        print(f"B3: encoding {enc}: {same} of {len(rs)} routes identical to --ref")
        ok &= same == len(rs)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=12)
    ap.add_argument("--stake", type=int, default=8)
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    rs = routes(a.stake, a.routes)
    ok = b2(a.src)
    ok &= b1(a.src, rs)
    if a.ref:
        ok &= b3(a.src, rs[:3], a.ref)
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
