"""Pre-flight for model v4's build potential and offer values (harness/build_phi.lua; docs/hypotheses.md, v4's
2026-11-09 amendment, changes 2 and 3, and the definitions entry). No torch needed. Replays published wins on
the real game, side by side with and without phi_build, and checks at every move:

  1. Computing it changes nothing: the complete game state (diff_check's dump: all of G.GAME, the random
     state included, every card area) and the observation are identical to a run without it.
  2. The cached value equals a fresh computation (the cache key holds everything phi_build reads); the same
     for every offer value, every 10th move.
  3. The same state gives the same value: in another runtime that replays to the same move with an empty
     cache, and with the deck in another order (draw order is hidden; phi_build reads the deck as a set).
  4. Bounded in [0, 3], 0 at the end of the game; the shaping sums to -phi(start) over a game at gamma 1.
  5. An offer's value is what taking it does: after buying or picking a joker or planet, or redeeming a
     voucher, phi_build moves by exactly that offer's value (the price paid included: Bull and Bootstraps
     read money).
  Reports the sim cost per move: without, with offer values only (evals), with both (training).

  python3 v4_phi_check.py <game source dir> [--routes 6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import diff_check  # noqa: E402
from sim import Sim  # noqa: E402

FAILS = []
OFFER_TYPES = ("buy", "buy_and_use", "pick", "redeem")


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def routes(n):
    rs = [json.loads(l) for l in open(os.path.join(ROOT, "wins", "routes.jsonl")) if '"ordered-capped"' in l]
    return [rs[i] for i in np.linspace(0, len(rs) - 1, n).astype(int)]


def with_dump(sim):
    sim.env.lua.execute(diff_check.DUMP, sim.H)
    return sim


def realized_check(sim, desc, before, after, offers, money_read):
    """offers: {action text: value x 10000} at the move; before/after: phi_build around it."""
    t = desc.split(" ")[0]
    if t not in OFFER_TYPES or desc not in offers:
        return None
    if t == "buy" and sim.H._bought_planet:
        return None       # a planet bought into a slot is valued as used; phi moves when it is used
    want = offers[desc] / 10000.0
    return abs((after - before) - want) <= 0.5e-4 + 1e-7, want, after - before      # stored rounded to 1e-4


def check_route(src, r, k_cross):
    a = with_dump(Sim(src, mode="ordered-capped", enc=2))
    b = with_dump(Sim(src, mode="ordered-capped", enc=2, build_phi=True, offer_values=True))
    oa, ob = a.reset(r["seed"], r["back"], r["stake"]), b.reset(r["seed"], r["back"], r["stake"])
    start = b.progress["phi_build"]
    total, real_ok, real_n, real_bad, cross = 0.0, 0, 0, [], None
    for k, act in enumerate(r["acts"]):
        st = b.H.state_name()
        if a.H.state_dump() != b.H.state_dump():
            x, y = diff_check.first_diff(a.H.state_dump(), b.H.state_dump())
            return fail(f"{r['seed']} move {k} ({st}): phi_build changed the game: {x[:3]} vs {y[:3]}")
        if not (np.array_equal(oa["act"], ob["act"][:, :18]) and np.array_equal(oa["feat"], ob["feat"])
                and np.array_equal(oa["vocab"], ob["vocab"])):
            return fail(f"{r['seed']} move {k}: observation differs apart from the offer column")
        offers = {b.H.action_desc(i + 1): int(v) for i, v in enumerate(ob["act"][:, 18]) if v}
        phi_now = b.progress["phi_build"]
        cached = b.H.build_power()[0]
        if cached != phi_now:
            return fail(f"{r['seed']} move {k}: progress phi_build {phi_now} != build_power() {cached}")
        fresh = b.H.build_power_fresh()[0]
        if cached != fresh:
            return fail(f"{r['seed']} move {k} ({st}): cached phi_build {cached} != fresh {fresh} (key misses something)")
        if not 0.0 <= phi_now <= 3.0:
            return fail(f"{r['seed']} move {k}: phi_build {phi_now} out of [0, 3]")
        if k % 10 == 0 and ob["act"][:, 18].any():
            b.H._phi_cache = None
            again = np.array(b.H.offer_values().split(), dtype=np.int64)
            if not np.array_equal(again, ob["act"][:, 18]):
                return fail(f"{r['seed']} move {k}: offer values cached {list(ob['act'][:, 18])} != fresh {list(again)}")
        if k % 25 == 0:
            rev = "local H = ...; local t = G.playing_cards; local n = #t; for i = 1, math.floor(n / 2) do t[i], t[n - i + 1] = t[n - i + 1], t[i] end"
            v0 = b.H.build_power_fresh()[0]
            b.env.lua.execute(rev, b.H)
            v1 = b.H.build_power_fresh()[0]
            b.env.lua.execute(rev, b.H)
            if v0 != v1:
                return fail(f"{r['seed']} move {k}: phi_build depends on the deck's order: {v0} vs {v1}")
        if k == k_cross:
            cross = (phi_now, list(ob["act"][:, 18]))
        money = bool(b.env.lua.eval("(function() for _, c in ipairs(G.jokers.cards) do "
                                    "if c.ability.name == 'Bull' or c.ability.name == 'Bootstraps' then return true end end "
                                    "return false end)()"))
        desc = b.H.action_desc(act + 1)
        b.H._bought_planet = desc.startswith("buy ") and bool(b.env.lua.eval(
            f"(function() local c = G.shop_jokers and G.shop_jokers.cards[{int(desc.split()[1])}]; "
            "return c ~= nil and c.ability.set == 'Planet' end)()"))
        _, ca, done_a = a.step(act)
        _, cb, done_b = b.step(act)
        rc = realized_check(b, desc, phi_now, cb["phi_build"], offers, money)
        if rc is not None:
            real_n += 1
            if rc[0]:
                real_ok += 1
            else:
                real_bad.append(f"move {k} {desc}: offer {rc[1]:+.4f}, phi moved {rc[2]:+.6f}")
        total += 1.0 * cb["phi_build"] - phi_now
        if not done_b:
            oa, ob = a.observe(), b.observe()
    if not (done_b and b.progress["won"]):
        return fail(f"{r['seed']}: the published win didn't win with phi_build on")
    if b.progress["phi_build"] != 0.0:
        return fail(f"{r['seed']}: phi_build not 0 at the end")
    if abs(total + start) > 1e-9:
        return fail(f"{r['seed']}: shaping over the game {total} != -phi(start) {-start}")
    # another runtime, empty cache, same state
    c = Sim(src, mode="ordered-capped", enc=2, build_phi=True, offer_values=True)
    o = c.reset(r["seed"], r["back"], r["stake"])
    for act in r["acts"][:k_cross]:
        c.step(act)
        o = c.observe()
    if (c.progress["phi_build"], list(o["act"][:, 18])) != cross:
        return fail(f"{r['seed']} move {k_cross}: another runtime gives {c.progress['phi_build']} vs {cross[0]}")
    if real_bad:
        fail(f"{r['seed']}: {len(real_bad)} of {real_n} offers didn't move phi_build by their value: {real_bad[:3]}")
    print(f"{r['seed']}: {len(r['acts'])} moves, phi_build {start:.3f} at the start; offers taken {real_n}, "
          f"value = change {real_ok}", flush=True)
    return real_n, real_ok


def cost(src, rs):
    out = {}
    for name, kw in (("without", {}), ("offer values only (evals)", {"offer_values": True}),
                     ("phi_build and offer values (training)", {"build_phi": True, "offer_values": True})):
        s = Sim(src, mode="ordered-capped", enc=2, **kw)
        t, n = time.time(), 0
        for r in rs:
            s.reset(r["seed"], r["back"], r["stake"])
            for act in r["acts"]:
                s.observe()
                s.step(act)
                n += 1
        out[name] = 1000 * (time.time() - t) / n
    base = out["without"]
    print("sim cost per move: " + "; ".join(f"{k} {v:.1f} ms ({v / base - 1:+.0%})" if k != "without" else f"{k} {v:.1f} ms"
                                            for k, v in out.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=6)
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()
    rs = routes(a.routes)
    n = ok = 0
    for i, r in enumerate(rs):
        got = check_route(a.src, r, k_cross=len(r["acts"]) // 2 + i)
        if got:
            n, ok = n + got[0], ok + got[1]
    print(f"phi_build: game unchanged, cache = fresh, deck order and runtime don't matter, bounded, telescopes: "
          f"{'PASS' if not FAILS else 'FAIL'}; offers taken {n}, value = change in {ok}")
    if not a.no_cost:
        cost(a.src, rs)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")


if __name__ == "__main__":
    main()
