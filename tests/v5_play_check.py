"""Pre-flight for model v5's play values (harness/build_phi.lua H.play_scores, H.play_values; registered in
docs/hypotheses-v5.md, 2026-11-15, "Checks before the smoke test"). Replays published wins on the real game,
side by side with v4b's observation (phi_build and offer values) and v5's (the same plus play values):

  1. Computing play values changes nothing: the complete game state (diff_check's dump) and every column of the
     observation v4b sees are identical at every move.
  2. S, the score a play is valued at, equals the chips the game scores when the play is made (the game's own
     hand_chips * mult, captured from evaluate_play's check_and_set_high_score), for every play with nothing
     the registration leaves out: not DNA (its copy isn't made), no chance in the score (Space Joker, Bloodstone,
     Misprint, a Lucky Card played), not The Hook. (Until 2026-11-27 the jokers acting before scoring were left out
     too: Spare Trousers, Square Joker, Runner, Midas Mask, Vampire, To Do List, Ride the Bus, Obelisk, Green Joker.) At least 98% exact; every miss is listed.
  3. The warm-started v5 model (train/model.py warm_start_state) gives the same move probabilities as v4b's
     checkpoint at every move of the routes (largest difference under 1e-5). Needs torch and --warm.
  4. Sim time per move, v5 over v4b on the same routes: reported; over 2x fails (the launch stops for chud).

  python3 tests/v5_play_check.py <game source dir> [--routes 3] [--warm <v4b training checkpoint>]
"""
import argparse
import json
import math
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
V4B = {"build_phi": True, "offer_values": True}
V5 = {"build_phi": True, "offer_values": True, "play_values": True}
# What score() can't match exactly: DNA (its copy isn't made), chance (Space Joker, Bloodstone, Misprint). The jokers
# acting before scoring came in with score()'s before step (docs/hypotheses-v6.md, 2026-11-27 15:40 section).
LEFT_OUT_JOKERS = {"DNA", "Space Joker", "Bloodstone", "Misprint"}
CAPTURE = """local H = ...
H._cap = nil
if not H._cap_wrapped then
  H._cap_wrapped = true
  local f = check_and_set_high_score
  check_and_set_high_score = function(score, amt)
    if score == 'hand' then H._cap = (H._cap or 0) + math.floor(amt) end
    if f then return f(score, amt) end
  end
end"""


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def routes(n):
    rs = [json.loads(l) for l in open(os.path.join(ROOT, "wins", "routes.jsonl")) if '"ordered-capped"' in l]
    return [rs[i] for i in np.linspace(0, len(rs) - 1, n).astype(int)]


def with_dump(sim):
    sim.env.lua.execute(diff_check.DUMP, sim.H)
    return sim


def left_out(sim, act):
    """Why this play's score is outside check 2 (None = inside)."""
    lua = sim.env.lua
    names = set(lua.eval("(function() local t = {} for _, j in ipairs(G.jokers.cards) do "
                         "t[#t + 1] = j.ability.name end return table.concat(t, '|') end)()").split("|"))
    hit = sorted(names & LEFT_OUT_JOKERS)
    if hit:
        return hit[0]
    if lua.eval("G.GAME.blind and G.GAME.blind.name or ''") == "The Hook":
        return "The Hook"
    a = sim.H._encoded_actions[act + 1]
    for p in (a.cards or {}).values():
        if lua.eval(f"G.hand.cards[{int(p)}].ability.name") == "Lucky Card":
            return "Lucky Card"
    return None


def check_route(src, r, models=None):
    a = with_dump(Sim(src, mode="ordered-capped", enc=2, **V4B))
    b = with_dump(Sim(src, mode="ordered-capped", enc=2, **V5))
    oa, ob = a.reset(r["seed"], r["back"], r["stake"]), b.reset(r["seed"], r["back"], r["stake"])
    b.env.lua.execute(CAPTURE, b.H)
    plays = exact = 0
    misses, outside, worst = [], {}, 0.0
    done_b = False
    for k, act in enumerate(r["acts"]):
        st = b.H.state_name()
        if a.H.state_dump() != b.H.state_dump():
            x, y = diff_check.first_diff(a.H.state_dump(), b.H.state_dump())
            return fail(f"{r['seed']} move {k} ({st}): play values changed the game: {x[:3]} vs {y[:3]}")
        if not (np.array_equal(oa["act"], ob["act"][:, :19]) and np.array_equal(oa["feat"], ob["feat"])
                and np.array_equal(oa["vocab"], ob["vocab"]) and np.array_equal(oa["kind"], ob["kind"])):
            return fail(f"{r['seed']} move {k}: v4b's part of the observation differs")
        if ob["act"].shape[1] != 21:
            return fail(f"{r['seed']} move {k}: action rows are {ob['act'].shape[1]} wide, not 21")
        if models is not None:
            worst = max(worst, prob_diff(models, oa, ob, st))
        desc = b.H.action_desc(act + 1)
        S = None
        if desc.startswith("play"):
            S = b.H.play_scores()[act + 1]
            if S is None:
                return fail(f"{r['seed']} move {k}: a play ({desc}) got no score")
            why = left_out(b, act)
            if b.H.state_dump() != a.H.state_dump():
                return fail(f"{r['seed']} move {k}: play_scores changed the game")
        else:
            pv = ob["act"][act, 19:21]
            if pv.any():
                return fail(f"{r['seed']} move {k}: a move that isn't a play ({desc}) has play values {list(pv)}")
        b.H._cap = None
        _, cb, done_b = b.step(act)
        a.step(act)
        if S is not None:
            got = b.H._cap or 0
            if why:
                outside[why] = outside.get(why, 0) + 1
            else:
                plays += 1
                if math.floor(S) == got:
                    exact += 1
                else:
                    misses.append(f"{r['seed']} move {k} {desc}: S {math.floor(S)}, game {got}")
        if not done_b:
            oa, ob = a.observe(), b.observe()
    if not (done_b and b.progress["won"]):
        return fail(f"{r['seed']}: the published win didn't win with play values on")
    print(f"{r['seed']}: {len(r['acts'])} moves; plays checked {plays}, S = game's chips {exact}; outside check 2: "
          f"{dict(sorted(outside.items())) or 'none'}" + (f"; largest probability difference {worst:.2e}" if models else ""),
          flush=True)
    return plays, exact, misses, worst


def load_models(path):
    import torch
    from types import SimpleNamespace
    from model import make_model, warm_start_state
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    base = dict(phase_split=True, offer_values=True, d=arch["d"], layers=arch["layers"], heads=arch["heads"],
                enc=arch["enc"])
    old = make_model(SimpleNamespace(**base), len(ck["vocab"]))
    old.load_state_dict(ck["model"])
    new = make_model(SimpleNamespace(play_values=True, **base), len(ck["vocab"]))
    new.load_state_dict(warm_start_state(ck["model"], new.state_dict()))
    old.eval(), new.eval()
    return old, new


def prob_diff(models, oa, ob, st):
    import torch
    from common import collate
    from model import phase_member
    m = phase_member(st)
    out = []
    for net, o in zip(models, (oa, ob)):
        o = dict(o, member=m)
        with torch.no_grad():
            logits, _ = net(*collate([o]))
        out.append(torch.softmax(logits[0, :len(o["act"])].double(), -1).numpy())
    return float(np.abs(out[0] - out[1]).max())


def cost(src, rs):
    out = {}
    for name, kw in (("v4b", V4B), ("v5", V5)):
        s = Sim(src, mode="ordered-capped", enc=2, **kw)
        t, n = time.time(), 0
        for r in rs:
            s.reset(r["seed"], r["back"], r["stake"])
            for act in r["acts"]:
                s.observe()
                s.step(act)
                n += 1
        out[name] = 1000 * (time.time() - t) / n
    ratio = out["v5"] / out["v4b"]
    print(f"check 4, sim time per move: v4b {out['v4b']:.1f} ms, v5 {out['v5']:.1f} ms ({ratio:.2f}x)", flush=True)
    if ratio > 2.0:
        fail(f"v5's sim is {ratio:.2f}x v4b's (over 2x: the launch stops for chud to decide)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--warm", help="v4b's training checkpoint (runs/v4b/best.pt): check 3")
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()
    rs = routes(a.routes)
    models = None
    if a.warm:
        if not os.path.exists(a.warm):
            fail(f"--warm {a.warm}: no such checkpoint (v5 warm-starts from it)")
        else:
            models = load_models(a.warm)
    plays = exact = 0
    misses, worst = [], 0.0
    for r in rs:
        got = check_route(a.src, r, models)
        if got:
            plays, exact, misses, worst = plays + got[0], exact + got[1], misses + got[2], max(worst, got[3])
    print(f"check 1, game and v4b's observation unchanged: {'PASS' if not FAILS else 'FAIL'}")
    share = exact / plays if plays else 0.0
    print(f"check 2, S = the game's chips: {exact} of {plays} plays exact ({share:.1%})")
    for m in misses:
        print("  miss:", m)
    if plays == 0 or share < 0.98:
        fail(f"S matched the game's chips on {share:.1%} of plays (registered: at least 98%)")
    if models is not None:
        print(f"check 3, warm start plays as v4b: largest probability difference {worst:.2e}")
        if worst >= 1e-5:
            fail(f"warm-started v5's probabilities differ from v4b's by {worst:.2e} (registered: under 1e-5)")
    else:
        print("check 3 skipped (no --warm)")
    if not a.no_cost:
        cost(a.src, rs)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")


if __name__ == "__main__":
    main()
