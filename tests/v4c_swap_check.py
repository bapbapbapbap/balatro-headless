"""Pre-flight for model v4c (docs/hypotheses-v4c.md, 2026-11-13 ~18:45 CST): the swap move and the offer values
for swaps, rerolls and Buffoon/Celestial packs (harness/env.lua H.swaps, harness/build_phi.lua pool_value).
Replays published ordered-capped wins on the real game and checks:

  1. The swap action list changes nothing else: replayed on ordered-capped-swap with phi_build and every offer
     value on, beside ordered-capped with nothing, the complete game state (diff_check's dump: all of G.GAME,
     the random state included, every card area) is identical at every move. Every saved move has the same
     index and text, swaps are listed after every other move, and the published win still wins.
  2. A swap is a sale then a buy: at states where swaps are offered, taking "swap i>j" in one runtime and
     "sell_joker i" then "buy j" in another ends in the same complete game state.
  3. A swap's offer value is what it does: phi_build moves by exactly the swap's value (boards without
     Campfire, which grows when a card is sold).
  4. A pool card is the shop's card: a base-edition joker on the shelf, valued as the pool's card for the same
     joker, gives the same value as the shelf card itself (To Do List excepted: its hand is drawn at random).
  5. Reroll and pack values are fixed by the state (the same in a new runtime with an empty cache), in [0, 3],
     and not all 0.
  6. The warm start (needs torch): a v4b-shaped model's weights load into the swap model, every weight unchanged,
     the swap row a copy of buy's, and on actions with no swap the two models give the same logits and values.
  Reports the sim cost per move against ordered-capped with the same values on (the registration's gate is
  read from the smoke run's own sim time).

  python3 v4c_swap_check.py <game source dir> [--routes 3] [--swaps 4]
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
SWAP = "ordered-capped-swap"


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def routes(n):
    rs = [json.loads(l) for l in open(os.path.join(ROOT, "wins", "routes.jsonl")) if '"ordered-capped"' in l]
    return [rs[i] for i in np.linspace(0, len(rs) - 1, n).astype(int)]


def with_dump(sim):
    sim.env.lua.execute(diff_check.DUMP, sim.H)
    return sim


def descs(sim, o):
    return [sim.H.action_desc(i + 1) for i in range(len(o["act"]))]


def lua_bool(sim, expr):
    return bool(sim.env.lua.eval(f"(function() {expr} end)()"))


def at(src, r, k, **kw):
    """A fresh swap-list runtime at move k of route r, with the dump loaded."""
    s = with_dump(Sim(src, mode=SWAP, enc=2, **kw))
    s.reset(r["seed"], r["back"], r["stake"])
    o = s.fast_forward(r["acts"][:k], r["desc"][:k]) if k else s.observe()
    if o is None:
        raise RuntimeError(f"{r['seed']}: couldn't replay to move {k}")
    return s, o


def check_replay(src, r):
    """1, and the states (move numbers) where swaps are offered, with the reroll/pack values seen."""
    a = with_dump(Sim(src, mode="ordered-capped", enc=2))
    b = with_dump(Sim(src, mode=SWAP, enc=2, build_phi=True, offer_values=True))
    oa, ob = a.reset(r["seed"], r["back"], r["stake"]), b.reset(r["seed"], r["back"], r["stake"])
    swap_at, pool_vals, done_b = [], [], False
    for k, act in enumerate(r["acts"]):
        if a.H.state_dump() != b.H.state_dump():
            x, y = diff_check.first_diff(a.H.state_dump(), b.H.state_dump())
            return fail(f"{r['seed']} move {k} ({b.H.state_name()}): the swap list or its values changed the game: {x[:3]} vs {y[:3]}")
        da, db = descs(a, oa), descs(b, ob)
        if db[:len(da)] != da:
            return fail(f"{r['seed']} move {k}: the swap list reorders other moves")
        extra = db[len(da):]
        if any(not d.startswith("swap ") for d in extra) or any(d.startswith("swap ") for d in da):
            return fail(f"{r['seed']} move {k}: moves past ordered-capped's aren't all swaps: {extra[:4]}")
        if not np.array_equal(oa["act"], ob["act"][:len(da), :18]):
            return fail(f"{r['seed']} move {k}: action rows differ apart from the swaps and the offer column")
        if extra:
            swap_at.append(k)
        for i, d in enumerate(db):
            if d == "reroll" or d.startswith("open_pack"):
                pool_vals.append((k, d, int(ob["act"][i, 18])))
        _, _, _ = a.step(act)
        _, _, done_b = b.step(act)
        if not done_b:
            oa, ob = a.observe(), b.observe()
    if not (done_b and b.progress["won"]):
        return fail(f"{r['seed']}: the published win didn't win on the swap list")
    print(f"{r['seed']}: {len(r['acts'])} moves replayed unchanged on {SWAP}; swaps offered at {len(swap_at)} states",
          flush=True)
    return swap_at, pool_vals


def check_swaps(src, r, swap_at, n):
    """2 and 3 at up to n of the states where swaps are offered."""
    picks = [swap_at[i] for i in np.linspace(0, len(swap_at) - 1, min(n, len(swap_at))).astype(int)] if swap_at else []
    done = 0
    for k in picks:
        c, oc = at(src, r, k, build_phi=True, offer_values=True)
        dc = descs(c, oc)
        sw = [i for i, d in enumerate(dc) if d.startswith("swap ")]
        i = sw[(k * 7) % len(sw)]
        d = dc[i]
        ji, sj = d.split(" ")[1].split(">")
        value = int(oc["act"][i, 18]) / 10000.0
        phi0 = c.progress["phi_build"]
        campfire = lua_bool(c, "for _, x in ipairs(G.jokers.cards) do if x.ability.name == 'Campfire' then return true end end return false")
        c.step(i)
        phi1 = c.progress["phi_build"]
        if int(c.H.swap_partial or 0):
            fail(f"{r['seed']} move {k}: {d} sold but couldn't buy")
            continue
        e, oe = at(src, r, k, build_phi=True, offer_values=True)
        de = descs(e, oe)
        e.step(de.index(f"sell_joker {ji}"))
        oe = e.observe()
        de = descs(e, oe)
        if f"buy {sj}" not in de:
            fail(f"{r['seed']} move {k}: after sell_joker {ji}, buy {sj} isn't offered: {de[:8]}")
            continue
        e.step(de.index(f"buy {sj}"))
        if c.H.state_dump() != e.H.state_dump():
            x, y = diff_check.first_diff(c.H.state_dump(), e.H.state_dump())
            fail(f"{r['seed']} move {k}: {d} != sell then buy: {x[:3]} vs {y[:3]}")
            continue
        if not campfire and abs((phi1 - phi0) - value) > 0.5e-4 + 1e-7:
            fail(f"{r['seed']} move {k}: {d} valued {value:+.4f}, phi_build moved {phi1 - phi0:+.6f}")
            continue
        done += 1
    print(f"{r['seed']}: {done} of {len(picks)} swaps = sell then buy, value = change", flush=True)


POOL_FIDELITY = r"""
local H = ...
-- each base-edition shelf joker's buy value, beside the same joker as a pool card: {real, pool} pairs
local out = {}
if #G.jokers.cards >= G.jokers.config.card_limit then return out end
local base = H.build_key_base()
local now = H.build_power(nil, base)
for _, c in ipairs(G.shop_jokers and G.shop_jokers.cards or {}) do
  if c.ability.set == "Joker" and not c.edition and c.ability.name ~= "To Do List" then
    local real = H.build_power({ joker = c, cost = c.cost }, base) - now
    local p = H.pool_card(c.config.center.key)
    Card.set_cost(p)
    local pool = H.build_power({ joker = p, cost = c.cost }, base) - now
    out[#out + 1] = { c.ability.name, real, pool }
  end
end
return out
"""


def check_pool_cards(src, r):
    """4 at every shop state with a free slot along the route."""
    s = with_dump(Sim(src, mode=SWAP, enc=2, build_phi=True, offer_values=True))
    s.reset(r["seed"], r["back"], r["stake"])
    fn = s.env.lua.execute("return function(H) " + POOL_FIDELITY.replace("local H = ...\n", "") + " end")
    n, bad = 0, []
    for k, act in enumerate(r["acts"]):
        if s.H.state_name() == "SHOP":
            before = s.H.state_dump()
            for row in fn(s.H).values():
                name, real, pool = row[1], row[2], row[3]
                n += 1
                if abs(real - pool) > 1e-12:
                    bad.append(f"move {k} {name}: shelf {real:+.6f}, pool card {pool:+.6f}")
            if s.H.state_dump() != before:
                return fail(f"{r['seed']} move {k}: building pool cards changed the game")
        s.observe()
        _, _, done = s.step(act)
        if done:
            break
    if bad:
        fail(f"{r['seed']}: {len(bad)} of {n} pool cards differ from the shelf card: {bad[:3]}")
    print(f"{r['seed']}: {n} shelf jokers valued as pool cards, {n - len(bad)} equal", flush=True)


def check_pool_values(src, r, pool_vals):
    """5: bounds, and the same value in a new runtime with an empty cache."""
    vals = [v for _, _, v in pool_vals]
    if any(v < 0 or v > 30000 for v in vals):
        fail(f"{r['seed']}: a reroll/pack value outside [0, 3]: {[x for x in pool_vals if x[2] < 0 or x[2] > 30000][:3]}")
    if pool_vals:
        k, d, v = pool_vals[len(pool_vals) // 2]
        s, o = at(src, r, k, build_phi=True, offer_values=True)
        again = int(o["act"][descs(s, o).index(d), 18])
        if again != v:
            fail(f"{r['seed']} move {k}: {d} valued {v} on the replay, {again} in a new runtime")
    nz = sum(1 for v in vals if v)
    print(f"{r['seed']}: {len(vals)} reroll/pack values, {nz} nonzero, max {max(vals, default=0) / 10000:.4f}", flush=True)
    return nz


def check_warm_start():
    """6 (torch)."""
    try:
        import torch
    except ImportError:
        print("warm start: torch not installed here, skipped (the launch's own pre-flight has it)")
        return
    from argparse import Namespace
    from model import BUY, SWAP as SWAP_T, make_model, warm_start_state
    torch.manual_seed(0)
    cfg = dict(phase_split=True, d=32, layers=1, heads=2, enc=2, offer_values=True)
    old = make_model(Namespace(actions="ordered-capped", **cfg), 300)
    new = make_model(Namespace(actions=SWAP, **cfg), 300)
    sd = warm_start_state(old.state_dict(), new.state_dict())
    new.load_state_dict(sd)
    for k, v in old.state_dict().items():
        w = new.state_dict()[k]
        if k.endswith("type_emb.weight"):
            if not (torch.equal(w[:SWAP_T], v) and torch.equal(w[SWAP_T], v[BUY])):
                return fail(f"warm start: {k}: old rows changed or swap row isn't buy's")
        elif not torch.equal(w, v):
            return fail(f"warm start: {k} changed")
    B, T, A = 3, 7, 5
    kind = torch.randint(1, 16, (B, T)); vocab = torch.randint(1, 300, (B, T)); feat = torch.randn(B, T, 32)
    tm = torch.ones(B, T, dtype=torch.bool)
    act = torch.full((B, A, 19), -1, dtype=torch.long)
    act[..., 0] = torch.randint(1, SWAP_T, (B, A)); act[..., 1] = torch.randint(0, T, (B, A))
    for c in (8, 16, 17, 18):
        act[..., c] = torch.randint(0, 50, (B, A))
    am = torch.ones(B, A, dtype=torch.bool); mem = torch.tensor([0, 1, 1])
    with torch.no_grad():
        l0, v0 = old(kind, vocab, feat, tm, act, am, mem)
        l1, v1 = new(kind, vocab, feat, tm, act, am, mem)
    if not (torch.equal(l0, l1) and torch.equal(v0, v1)):
        return fail("warm start: without swaps, the swap model's logits/values differ from the old model's")
    print("warm start: every weight unchanged, swap row = buy row, same logits and values without swaps", flush=True)


def cost(src, rs):
    out = {}
    for name, mode in (("ordered-capped", "ordered-capped"), (SWAP, SWAP)):
        s = Sim(src, mode=mode, enc=2, build_phi=True, offer_values=True)
        t, n = time.time(), 0
        for r in rs:
            s.reset(r["seed"], r["back"], r["stake"])
            for act in r["acts"]:
                s.observe()
                s.step(act)
                n += 1
        out[name] = 1000 * (time.time() - t) / n
    print(f"sim cost per move (phi_build and offer values on): ordered-capped {out['ordered-capped']:.1f} ms, "
          f"{SWAP} {out[SWAP]:.1f} ms ({out[SWAP] / out['ordered-capped']:.2f}x)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--swaps", type=int, default=4, help="swap states checked per route (2 and 3)")
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()
    rs = routes(a.routes)
    nonzero = 0
    for r in rs:
        got = check_replay(a.src, r)
        if not got:
            continue
        swap_at, pool_vals = got
        check_swaps(a.src, r, swap_at, a.swaps)
        check_pool_cards(a.src, r)
        nonzero += check_pool_values(a.src, r, pool_vals)
    if not nonzero:
        fail("no reroll or pack value was ever nonzero")
    check_warm_start()
    if not a.no_cost:
        cost(a.src, rs)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")


if __name__ == "__main__":
    main()
