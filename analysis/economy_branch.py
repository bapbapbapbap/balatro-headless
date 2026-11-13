"""Why v4b hoards: branch rich late shops three ways and play each out (diagnostic, 2026-11-13).

The question (chud, 2026-11-13 ~11:30): what keeps v4b from spending idle money late. Four explanations were
open (docs/hypotheses-v4b.md, after the walls diagnosis):
  A. exploration collapse: P(reroll) is near 0, so the few rerolls sampled can't move the policy;
  B. the signal says no: a reroll the bot does take doesn't pay under what it is trained on;
  C. spending doesn't pay here (red deck, white stake, the "blinds" reward), even spent well;
  D. mechanical: reroll is not offered when it should be.

From saved eval games, one shop exit per game is picked (an `end_shop` at ante >= --min-ante with money over
the interest cap; chosen with a fixed RNG per seed). At that state the checkpoint's shop network gives
P(reroll), P(end_shop), P(any buy), ... and its critic value V; phi_build is read too. Then three branches:
  end    the move the bot made (end_shop);
  reroll one reroll, then the bot plays on (in the shop too);
  spend  a scripted spender, then the bot plays on. Keeps the interest cap as a reserve and spends the rest:
         buys an offer with positive offer value (a joker into a free slot, a planet, used at once), swaps
         a held joker for a shop joker when phi_build gains at least --min-gain, redeems a voucher whose
         offer value is not negative (so not Hieroglyph / Petroglyph), opens Celestial packs and Buffoon packs
         (with a free slot) and picks the best positive offer value, rerolls while it can afford to, then
         ends the shop. Its judgment is phi_build's, the same numbers the bot sees as offer values.
Each branch's opening moves are fixed (the game is deterministic), so every play-out replays the saved game
from its seed to the branch point (Sim.fast_forward, every move checked against its saved text), takes them,
and plays on with the checkpoint, sampled at temperature 1.0 (training's). Play-out p of every branch uses the
same sampling RNG ("econ-<seed>-<p>"), so the branches are compared on common random numbers.

Recorded per branch: blinds beaten after the branch point, the return the trainer optimizes from there
(the reward, discounted by its own per-step discounts: --gamma-blind per blind passed), wins, where it ends;
and the one-step view the learner has of the branch's opening moves: delta = r + phi_build' - phi_build +
V' - V (the potential-based shaping's term, no blind passed inside a shop so gamma = 1). The shaping
telescopes over a whole play-out to -phi_build at the shared branch point, so it cancels between branches
and the play-out returns leave it out. D is counted on every qualifying shop exit met, chosen or not.

Reading (registered nowhere yet; chud decides): P(reroll) ~ 0 and the reroll branch's return higher -> A;
reroll no better but spend better -> B; neither better -> C. The summary prints the numbers, not a verdict.

usage (WSL, venv active, from the repo; the checkpoint must be a trainer checkpoint, it carries the settings):
  python3 analysis/economy_branch.py /mnt/c/Users/Jacobi/balatro-work/runs/v4b/best.pt \\
      --src ~/balatro/local/source --games /mnt/c/Users/Jacobi/balatro-work/runs/v4b/games \\
      --rounds 1840,1860,1880 --playouts 8 --workers 12 --out econ_v4b.jsonl        # rerun to resume
  python3 analysis/economy_branch.py --summary econ_v4b.jsonl
Smoke: add --max-games 3 --playouts 2 --workers 3 (prints seconds per game, to size the full run).
Read-only on the run: writes only --out.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(os.getcwd(), "train"))   # run from the repo with the script elsewhere

BRANCHES = ("end", "reroll", "spend")

PROBE = r"""
local H = ...
function H.econ_info()
  local g = G.GAME
  local cr = g.current_round or {}
  return g.dollars or 0, g.interest_cap or 25, cr.reroll_cost or 0, H.state_name(), g.round_resets.ante,
         #G.jokers.cards, G.jokers.config.card_limit or 0
end
function H.econ_card(area, i)
  local a = G[area]
  local c = a and a.cards and a.cards[i]
  if not c then return "", "", 0 end
  return c.config.center.key, (c.ability and c.ability.set) or "", c.cost or 0
end
-- Jokers whose effect phi_build reads from the game's own counters (hand size, discards, slots, odds):
-- taking one out of G.jokers.cards would not take its effect out, so they are never swapped out.
local NO_SWAP = { ["Juggler"] = true, ["Drunkard"] = true, ["Turtle Bean"] = true, ["Troubadour"] = true,
                  ["Stuntman"] = true, ["Oops! All 6s"] = true, ["Merry Andy"] = true }
-- The best swap: sell held joker i, buy shop joker j. phi_build of the board without i plus j, minus now.
-- budget: what the spender may spend over its reserve (net of the sale). Returns i, j, gain, net cost.
function H.econ_best_swap(budget, min_gain)
  local now = H.build_power(nil, H.build_key_base())
  local held = G.jokers.cards
  local bi, bj, bg, bn = 0, 0, min_gain, 0
  for i, h in ipairs(held) do
    local a = h.ability or {}
    local ok = not a.eternal and not NO_SWAP[a.name] and not (h.edition and h.edition.negative)
               and (a.h_size or 0) == 0 and (a.d_size or 0) == 0
    if ok then
      local rest = {}
      for k, c in ipairs(held) do if k ~= i then rest[#rest + 1] = c end end
      for j, c in ipairs(G.shop_jokers and G.shop_jokers.cards or {}) do
        local net = (c.cost or 0) - (h.sell_cost or 0)
        if c.ability.set == "Joker" and net <= budget then
          G.jokers.cards = rest
          local okc, v = pcall(function() return H.build_power({ joker = c, cost = net }, H.build_key_base()) end)
          G.jokers.cards = held
          if not okc then error(v) end
          if v - now > bg then bi, bj, bg, bn = i, j, v - now, net end
        end
      end
    end
  end
  return bi, bj, bg, bn
end
"""

G = {}


def load(path, src):
    import torch
    from argparse import Namespace
    from model import make_model
    from sim import Sim, sim_kwargs
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    cfg.update(src=src, verify_wins=False, save_games="none", eval_temp=1.0)
    a = Namespace(**cfg)
    if not getattr(a, "phase_split", False):
        raise SystemExit(f"{path}: not a phase-split (v4) checkpoint; this script reads the shop network")
    sim = Sim(a.src, a.mods, mode=a.actions, enc=a.enc, **sim_kwargs(a))
    if list(sim.vocab) != list(ck["vocab"]):
        raise SystemExit(f"{path}: checkpoint vocabulary differs from this game's")
    sim.build_phi = False            # as in evals: offer values on, shaping's per-move phi_build off
    sim.env.lua.execute(PROBE, sim.H)
    model = make_model(a, len(sim.vocab))
    model.load_state_dict(ck["model"])
    model.eval()
    return a, sim, model, ck.get("iteration", 0)


def init(path, src):
    import torch
    torch.set_num_threads(1)
    G["args"], G["sim"], G["model"], G["it"] = load(path, src)


def info(H):
    d, cap, rc, st, ante, nj, lim = H.econ_info()
    return {"dollars": float(d), "cap": float(cap), "reroll_cost": float(rc), "state": st, "ante": int(ante),
            "jokers": int(nj), "slots": int(lim)}


def descs(H, o):
    return [H.action_desc(i + 1) for i in range(len(o["act"]))]


def policy(o, sim):
    """The acting network's probabilities (temperature 1) and value at this observation."""
    import torch
    from common import collate
    from model import phase_member
    o["member"] = phase_member(sim.H.state_name())
    with torch.no_grad():
        logits, v = G["model"](*collate([o]))
    p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
    if not np.isfinite(p).all():
        raise FloatingPointError("non-finite action probabilities")
    return p, float(v[0])


def phi_now(H):
    return float(H.build_power(None, H.build_key_base())[0])


def take(sim, o, d):
    """Takes the move whose text is d; returns the next observation (None at the end)."""
    ds = descs(sim.H, o)
    if d not in ds:
        raise RuntimeError(f"move '{d}' not offered: {ds}")
    _, _, done = sim.step(ds.index(d))
    return None if done else sim.observe()


def is_pack(st):
    return st.endswith("_PACK") or st == "SMODS_BOOSTER_OPENED"


def spender(sim, o, min_gain, max_moves=60):
    """The scripted spender, from a SHOP state until it ends the shop. Returns the moves' texts and a tally."""
    H = sim.H
    moves, tally = [], {"buy": 0, "swap": 0, "redeem": 0, "open_pack": 0, "pick": 0, "reroll": 0}
    pending = None
    for _ in range(max_moves):
        s = info(H)
        st, ds = s["state"], descs(H, o)
        ov = o["act"][:, -1].astype(np.float64) / 10000.0      # offer values (sim offer_values=True)
        budget = s["dollars"] - s["cap"]
        pick = None
        if is_pack(st):
            best = [(ov[i], d) for i, d in enumerate(ds) if d.startswith("pick ") and ov[i] > 0]
            pick = max(best)[1] if best else ("skip_pack" if "skip_pack" in ds else None)
            if pick is None:
                raise RuntimeError(f"pack with no pick and no skip: {ds}")
            tally["pick"] += pick.startswith("pick ")
        elif st == "SHOP":
            if pending:
                pick, pending = (pending if pending in ds else None), None
            if pick is None:                                   # 1. buys with a positive value
                best = []
                for i, d in enumerate(ds):
                    verb = d.split()[0]
                    if verb in ("buy", "buy_and_use") and ov[i] > 0:
                        key, kind, cost = H.econ_card("shop_jokers", int(d.split()[1]))
                        if cost <= budget and (kind == "Joker" and verb == "buy" or kind == "Planet" and verb == "buy_and_use"):
                            best.append((ov[i], d))
                if best:
                    pick = max(best)[1]
                    tally["buy"] += 1
            if pick is None and s["jokers"] >= s["slots"]:     # 2. a swap phi_build says is worth it
                i, j, gain, net = H.econ_best_swap(budget, min_gain)
                if i > 0 and f"sell_joker {i}" in ds:
                    pick, pending = f"sell_joker {i}", f"buy {j}"
                    tally["swap"] += 1
            if pick is None:                                   # 3. vouchers phi_build doesn't count against
                for i, d in enumerate(ds):
                    if d.startswith("redeem ") and ov[i] >= 0:
                        _, _, cost = H.econ_card("shop_vouchers", int(d.split()[1]))
                        if cost <= budget:
                            pick = d
                            tally["redeem"] += 1
                            break
            if pick is None:                                   # 4. Celestial packs, Buffoon packs with a free slot
                for d in ds:
                    if d.startswith("open_pack "):
                        key, _, cost = H.econ_card("shop_booster", int(d.split()[1]))
                        if cost <= budget and (key.startswith("p_celestial")
                                               or key.startswith("p_buffoon") and s["jokers"] < s["slots"]):
                            pick = d
                            tally["open_pack"] += 1
                            break
            if pick is None and "reroll" in ds and s["reroll_cost"] <= budget:   # 5. reroll
                pick = "reroll"
                tally["reroll"] += 1
            if pick is None:
                pick = "end_shop"
        else:
            break                                              # out of the shop
        moves.append(pick)
        o = take(sim, o, pick)
        if pick == "end_shop" or o is None:
            break
    else:
        moves.append("end_shop")                               # move cap: leave the shop
        o = take(sim, o, "end_shop")
    return moves, tally, o


def to_branch(sim, g, k, prefix):
    """Fresh replay of the saved game to move k, then the branch's fixed opening moves (as texts)."""
    sim.reset(g["seed"], g["back"], g["stake"], g.get("challenge"))
    o = sim.fast_forward(g["acts"][:k], g["desc"][:k], None)
    if o is None:
        raise RuntimeError("saved game did not replay to the branch point")
    for d in prefix:
        o = take(sim, o, d)
    return o


def playout(sim, o, srng):
    """The checkpoint plays on from here, sampled at temperature 1. Returns blinds beaten, discounted return."""
    from rewards import REWARDS, step_discount
    a, reward_fn = G["args"], REWARDS[G["args"].reward]
    ret, disc, done = 0.0, 1.0, o is None
    while not done and sim.steps < a.max_steps:
        p, _ = policy(o, sim)
        i = min(int(np.searchsorted(np.cumsum(p), srng.random() * p.sum())), len(p) - 1)
        prev, cur, done = sim.step(i)
        g_t, _ = step_discount(prev, cur, a)
        ret += disc * reward_fn(prev, cur)
        disc *= g_t
        if not done:
            o = sim.observe()
    pr = sim.progress
    return {"blinds": pr["blinds_beaten"], "ret": round(ret, 4), "won": bool(pr["won"]), "ante": pr["ante"],
            "capped": not done}


def walk(sim, g, min_ante):
    """The saved game move by move (each checked against its text): every end_shop at ante >= min_ante with
    money over the interest cap, as its move index, and at each whether reroll was offered (D)."""
    H = sim.H
    o = sim.reset(g["seed"], g["back"], g["stake"], g.get("challenge"))
    exits, d_count = [], []
    for k, (a, d) in enumerate(zip(g["acts"], g["desc"])):
        if a >= len(o["act"]) or H.action_desc(a + 1) != d:
            raise RuntimeError(f"step {k}: saved '{d}' differs from the sim")
        if d == "end_shop":
            s = info(H)
            if s["state"] == "SHOP" and s["ante"] >= min_ante and s["dollars"] > s["cap"]:
                d_count.append({"ante": s["ante"], "reroll_offered": "reroll" in descs(H, o),
                                "affordable": s["reroll_cost"] <= s["dollars"]})
                exits.append(k)
        _, _, done = sim.step(a)
        if done:
            break
        o = sim.observe()
    return exits, d_count


def one_game(job):
    rnd, g, playouts, min_ante, min_gain = job
    sim, H = G["sim"], G["sim"].H
    t0 = time.time()
    rec = {"round": rnd, "seed": g["seed"], "error": None}
    try:
        # 1. walk the saved game: every qualifying shop exit (and D at each), then pick one
        sim.offer_values = False             # the walk needs only the legal moves (fast_forward does the same)
        try:
            exits, d_count = walk(sim, g, min_ante)
        finally:
            sim.offer_values = True
        rec["d"] = d_count
        if not exits:
            rec["chosen"] = None
            rec["seconds"] = round(time.time() - t0, 1)
            return rec
        k = random.Random(f"econ-pick-{g['seed']}").choice(exits)
        rec["chosen"] = k
        # 2. the branch point: the bot's view of it
        o = to_branch(sim, g, k, [])
        p, v = policy(o, sim)
        ds = descs(H, o)
        mass = {}
        for i, d in enumerate(ds):
            verb = d.split()[0]
            mass[verb] = mass.get(verb, 0.0) + float(p[i])
        s0 = info(H)
        phi0 = phi_now(H)
        rec["at"] = {**s0, "V": round(v, 4), "phi": round(phi0, 4),
                     "p": {x: round(y, 5) for x, y in sorted(mass.items())}}
        # 3. each branch's fixed opening, its one-step view (delta), then the play-outs
        rec["branches"] = {}
        for b in BRANCHES:
            o = to_branch(sim, g, k, [])
            if b == "end":
                prefix, tally = ["end_shop"], None
                o = take(sim, o, "end_shop")
            elif b == "reroll":
                if "reroll" not in descs(H, o):
                    rec["branches"][b] = {"skipped": "reroll not offered"}
                    continue
                prefix, tally = ["reroll"], None
                o = take(sim, o, "reroll")
            else:
                prefix, tally, o = spender(sim, o, min_gain)
            # no blind is beaten or passed inside a shop: r = 0 and gamma = 1 for the opening moves
            _, v1 = policy(o, sim)
            phi1 = phi_now(H)
            br = {"prefix": prefix, "tally": tally, "V1": round(v1, 4), "phi1": round(phi1, 4),
                  "delta": round(phi1 - phi0 + v1 - v, 4), "dollars_after": info(H)["dollars"], "plays": []}
            base = sim.progress["blinds_beaten"]
            for j in range(playouts):
                o = to_branch(sim, g, k, prefix)
                res = playout(sim, o, random.Random(f"econ-{g['seed']}-{j}"))
                res["blinds"] -= base
                br["plays"].append(res)
            rec["branches"][b] = br
    except Exception:
        rec["error"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------- summary (no game, no torch)

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(recs):
    errs = [r for r in recs if r.get("error")]
    ok = [r for r in recs if not r.get("error")]
    states = [r for r in ok if r.get("chosen") is not None and all(
        b in r.get("branches", {}) and "plays" in r["branches"][b] for b in BRANCHES)]
    print(f"games {len(recs)}, errors {len(errs)}, with a qualifying exit {sum(r.get('chosen') is not None for r in ok)}, "
          f"all three branches played {len(states)}")
    for r in errs[:3]:
        print(f"  error, seed {r['seed']}: {r['error'].strip().splitlines()[-1]}")
    # D: is reroll offered at rich late exits?
    dd = [d for r in ok for d in r.get("d", [])]
    if dd:
        off = np.mean([d["reroll_offered"] for d in dd])
        aff = [d for d in dd if d["affordable"]]
        off_aff = np.mean([d["reroll_offered"] for d in aff]) if aff else float("nan")
        print(f"\nD. qualifying exits {len(dd)}: reroll offered {off:.1%}; where affordable ({len(aff)}) {off_aff:.1%}")
    if not states:
        return
    # A: the policy at the branch point
    pr = np.array([r["at"]["p"].get("reroll", 0.0) for r in states])
    pe = np.array([r["at"]["p"].get("end_shop", 0.0) for r in states])
    pb = np.array([r["at"]["p"].get("buy", 0.0) + r["at"]["p"].get("buy_and_use", 0.0) for r in states])
    print(f"\nA. at the branch point (n={len(states)}, temperature 1): P(reroll) median {np.median(pr):.4f}, "
          f"mean {pr.mean():.4f}, under 0.01 in {np.mean(pr < 0.01):.0%}, under 0.05 in {np.mean(pr < 0.05):.0%}; "
          f"P(end_shop) median {np.median(pe):.3f}; P(buy) median {np.median(pb):.4f}")
    a = [r["at"] for r in states]
    print(f"   ante {np.mean([x['ante'] for x in a]):.2f}, money ${np.median([x['dollars'] for x in a]):.0f} "
          f"(cap ${np.median([x['cap'] for x in a]):.0f}), jokers {np.mean([x['jokers'] for x in a]):.2f}/"
          f"{np.mean([x['slots'] for x in a]):.2f}")
    # per branch
    print("\nbranch    blinds after   discounted return   wins     delta (one-step view)")
    per = {}
    for b in BRANCHES:
        bl = np.array([np.mean([p["blinds"] for p in r["branches"][b]["plays"]]) for r in states])
        rt = np.array([np.mean([p["ret"] for p in r["branches"][b]["plays"]]) for r in states])
        wn = np.mean([p["won"] for r in states for p in r["branches"][b]["plays"]])
        de = np.array([r["branches"][b]["delta"] for r in states])
        per[b] = (bl, rt)
        print(f"{b:8s}  {bl.mean():8.3f}       {rt.mean():8.3f}          {wn:6.2%}   {de.mean():+.4f}")
    print("\npaired, branch minus end (95% bootstrap over states)")
    for b in ("reroll", "spend"):
        for name, idx in (("blinds", 0), ("return", 1)):
            d = per[b][idx] - per["end"][idx]
            lo, hi = boot_ci(d)
            print(f"  {b:6s} {name:6s} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    t = {}
    for r in states:
        for x, y in (r["branches"]["spend"]["tally"] or {}).items():
            t[x] = t.get(x, 0) + y
    n = len(states)
    print("\nspend branch, moves per state: " + ", ".join(f"{x} {y / n:.2f}" for x, y in t.items())
          + f"; money left ${np.median([r['branches']['spend']['dollars_after'] for r in states]):.0f} (median)")
    caps = sum(p["capped"] for r in states for b in BRANCHES for p in r["branches"][b]["plays"])
    if caps:
        print(f"(play-outs cut at max_steps: {caps})")
    sec = [r["seconds"] for r in recs if r.get("seconds")]
    if sec:
        print(f"seconds per game: median {np.median(sec):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", help="trainer checkpoint (best.pt)")
    ap.add_argument("--src")
    ap.add_argument("--games", help="the run's games/ folder")
    ap.add_argument("--rounds", default="1840,1860,1880")
    ap.add_argument("--playouts", type=int, default=8)
    ap.add_argument("--min-ante", type=int, default=5)
    ap.add_argument("--min-gain", type=float, default=0.02, help="phi_build gain a swap needs (0.25 = one doubling)")
    ap.add_argument("--max-games", type=int, default=0, help="per round (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="econ.jsonl")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summarize([json.loads(x) for x in open(a.summary) if x.strip()])
        return
    if not (a.ckpt and a.src and a.games):
        ap.error("the checkpoint, --src and --games are needed to play")
    jobs = []
    for r in [int(x) for x in a.rounds.split(",")]:
        with gzip.open(os.path.join(a.games, f"r{r:05d}_eval.jsonl.gz"), "rt") as f:
            games = [json.loads(line) for line in f]
        if a.max_games:
            games = games[:a.max_games]
        jobs += [(r, g, a.playouts, a.min_ante, a.min_gain) for g in games]
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            if x.strip():
                rec = json.loads(x)
                if not rec.get("error"):
                    done.add((rec["round"], rec["seed"]))
    jobs = [j for j in jobs if (j[0], j[1]["seed"]) not in done]
    print(f"{len(jobs)} games to branch ({len(done)} done), {a.playouts} play-outs x {len(BRANCHES)} branches, "
          f"{a.workers} workers", flush=True)
    t0, n, bad = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, initializer=init, initargs=(a.ckpt, a.src)) as pool, \
            open(a.out, "a") as out:
        for rec in pool.imap_unordered(one_game, jobs, chunksize=1):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            bad += rec["error"] is not None
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} games, {bad} errors, {time.time() - t0:.0f} s", flush=True)
    print(f"done -> {a.out}; summary: python3 analysis/economy_branch.py --summary {a.out}")


if __name__ == "__main__":
    main()
