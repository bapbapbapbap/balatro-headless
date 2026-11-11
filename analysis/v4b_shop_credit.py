"""Why v4b's shop network is credited for selling jokers (diagnostic, 2026-11-10; Claude's, reviewed by chud).

Diagnosis 2 (claude/STATUS.md, 21:05-21:30) found eval falling 8.29 -> 7.20 at r280 because the shop network
sells staple jokers between blinds, 82% of them with a free slot. This asks why its credit favors that. Same
method as v4_sells.py: replay saved training games on the sim under a checkpoint, rebuild every step's training
reward (blinds + shaping + build shaping, blind clock) and the routed advantages exactly as training forms them
(credit.routed_gae, lam_hand on the table network only). Forward passes are batched per game.

For each shop-network decision it keeps: place (game state), ante, jokers held vs slots (free slot or full),
money, the move, P(move), P(any sell), and the credit split into parts:
  V0   = the shop critic's value at the decision
  V1   = its value at the shop network's next decision (what routed credit bootstraps from)
  R    = discounted reward until then, split into blinds / shaping / build shaping
  td   = R + Gam*V1 - V0         carry = adv - td  (the later decisions' share through lambda)
Tables: (A) sells by place x slot; (B) sells vs the other moves at the same sell-legal states, by ante x slot;
(C) every move at sell-legal shop states by verb x slot; (D) the shop critic by jokers held, antes 1-3 at the
shop: mean V0 vs mean target (adv + V0), the number the critic is trained toward; (E) which jokers it sells.
If V0 is flat in jokers held while the target rises, the critic underrates jokers and a sell costs nothing in
td; if R_build on sells is ~0 or positive, build shaping doesn't charge for the lost joker.

usage (WSL, venv active, from the repo; finds checkpoints and games under the work folder or the spill drive):
  python3 analysis/v4b_shop_credit.py --src ~/balatro/local/source --run v4b --its 160,240,260,280 --window 5 \
      --out /mnt/c/Users/Jacobi/balatro-work/v4b_shop_credit.jsonl
--window W: the games of rounds it-W+1..it under checkpoint it. Curriculum games are skipped. --out writes one
JSON line per shop decision at a sell-legal state (optional); nothing else is written.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from common import collate  # noqa: E402
from credit import routed_gae  # noqa: E402
from model import PHASE_NAMES, make_model, phase_member  # noqa: E402
from rewards import REWARDS, build_shaping, shaping, step_discount  # noqa: E402
from sim import Sim  # noqa: E402

SELL = 14                         # sell_joker's action type code (as v4_sells.py)
SHOP = PHASE_NAMES.index("shop")
ROOTS = ["/mnt/c/Users/Jacobi/balatro-work/runs", "/mnt/f/balatro data/runs"]


def lua_num(sim, expr, cast=float, bad=float("nan")):
    try:
        return cast(sim.env.lua.eval(expr))
    except Exception:
        return bad


def find(run, name, roots):
    for r in roots:
        for p in (os.path.join(r, run, name), os.path.join(r, run, "games", name)):
            if os.path.exists(p):
                return p
    return None


def load(ckpt_path, src, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    arch = ck["arch"]
    mcfg = SimpleNamespace(phase_split=True, offer_values=arch["offer_values"], d=arch["d"],
                           layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model = make_model(mcfg, len(ck["vocab"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    sim = Sim(src, None, mode=cfg["actions"], enc=arch["enc"], build_phi=True, offer_values=arch["offer_values"])
    assert sim.vocab == ck["vocab"], "game vocabulary differs from the checkpoint's"
    return model, sim, cfg


def forward(model, obs, device, chunk=256):
    """Log-probs (per observation, over its own actions) and values, batched."""
    lps, vs = [], []
    with torch.no_grad():
        for i in range(0, len(obs), chunk):
            part = obs[i:i + chunk]
            logits, v = model(*collate(part, device))
            lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
            vs.extend(v.float().cpu().numpy().tolist())
            for k, o in enumerate(part):
                lps.append(lp[k, :len(o["act"])])
    return lps, vs


def replay(model, sim, cfg, ep, device):
    """One game -> list of per-decision dicts with credit parts, or None if it didn't replay."""
    args = SimpleNamespace(**cfg)
    reward_fn = REWARDS[cfg["reward"]]
    sh, bsh = cfg.get("shaping", 1.0), cfg.get("build_shaping", 0)
    o = sim.reset(ep["seed"], ep["back"], ep["stake"], ep.get("challenge"))
    obs, steps = [], []
    done = False
    for k, a in enumerate(ep["acts"]):
        if done or a >= len(o["act"]) or sim.H.action_desc(a + 1) != ep["desc"][k]:
            return None
        state = sim.H.state_name()
        o["member"] = phase_member(state)
        desc = ep["desc"][k]
        verb = desc.split()[0]
        jk, slots = lua_num(sim, "#G.jokers.cards", int, -1), lua_num(sim, "G.jokers.config.card_limit", int, -1)
        m0 = lua_num(sim, "G.GAME.dollars")
        key = None
        if verb == "sell_joker":
            try:
                key = str(sim.env.lua.eval(f"G.jokers.cards[{int(desc.split()[1])}].config.center.key"))
            except Exception:
                key = "?"
        prev, cur, done = sim.step(a)
        g, l = step_discount(prev, cur, args)
        rb = reward_fn(prev, cur)
        rs = sh * shaping(prev, cur, g)
        rp = bsh * build_shaping(prev, cur, g)
        steps.append(dict(k=k, a=a, verb=verb, state=state, member=o["member"], ante=prev["ante"], jokers=jk,
                          slots=slots, money=m0, dmoney=lua_num(sim, "G.GAME.dollars") - m0, key=key,
                          rb=rb, rs=rs, rp=rp, g=g, l=l, hand=int(o["act"][a][0]) in (1, 2),
                          sell_legal=bool((o["act"][:, 0] == SELL).any())))
        obs.append(o)
        if not done:
            o = sim.observe()
    if not done:                                   # capped: bootstrap from the last state, as train.py does
        o["member"] = phase_member(sim.H.state_name())
        obs.append(o)
    lps, vs = forward(model, obs, device)
    n = len(steps)
    R = [s["rb"] + s["rs"] + s["rp"] for s in steps]
    G = [s["g"] for s in steps]
    L = [s["l"] for s in steps]
    own = [s["member"] for s in steps]
    lh = cfg.get("lam_hand", 1.0)
    if lh < 1.0:
        L = {0: [x * (lh if s["hand"] else 1.0) for x, s in zip(L, steps)], 1: L}
    adv = routed_gae(R, vs, G, L, own)
    tail = vs[n] if len(vs) > n else 0.0
    nxt = {}                                       # next own decision of each decision
    last = {}
    for t in reversed(range(n)):
        nxt[t] = last.get(own[t], n)
        last[own[t]] = t
    for t, s in enumerate(steps):
        end = nxt[t]
        rb = rs = rp = 0.0
        gam = 1.0
        for j in range(t, end):
            rb += gam * steps[j]["rb"]; rs += gam * steps[j]["rs"]; rp += gam * steps[j]["rp"]
            gam *= G[j]
        v1 = vs[end] if end < n else tail
        td = rb + rs + rp + gam * v1 - vs[t]
        lp = lps[t]
        act = obs[t]["act"]
        s.update(V0=vs[t], V1=v1, Gam=gam, Rb=rb, Rs=rs, Rp=rp, td=td, adv=float(adv[t]), carry=float(adv[t]) - td,
                 p=float(np.exp(lp[s["a"]])), p_sell=float(np.exp(lp[act[:, 0] == SELL]).sum()),
                 gap=end - t, blind_between=gam < 1.0)
    return steps


def bucket_ante(a):
    return str(a) if a <= 4 else "5+"


def report(it, rows, games, secs):
    shop = [r for r in rows if r["member"] == SHOP]
    adv = np.array([r["adv"] for r in shop])
    mu, sd = adv.mean(), adv.std() + 1e-8
    for r in shop:
        r["adv_n"] = (r["adv"] - mu) / sd
    sells = [r for r in shop if r["verb"] == "sell_joker"]
    legal = [r for r in shop if r["sell_legal"]]
    free = lambda r: r["jokers"] < r["slots"]
    print(f"\n=== ckpt {it}: {games} games, {len(rows)} decisions ({len(shop)} shop), {len(sells)} shop sells, {secs:.0f} s")

    def line(lbl, ix, extra=""):
        if not ix:
            return
        f = lambda k: np.mean([r[k] for r in ix])
        print(f"{lbl:28} {len(ix):6d} {f('p'):6.3f} {f('Rb'):7.3f} {f('Rs'):7.3f} {f('Rp'):7.3f} {f('V0'):7.3f} "
              f"{f('V1'):7.3f} {f('td'):7.3f} {f('carry'):7.3f} {f('adv'):7.3f} {f('adv_n'):6.2f} "
              f"{np.mean([r['adv'] > 0 for r in ix]):5.2f}{extra}")
    hdr = (f"{'':28} {'n':>6} {'P':>6} {'R_bl':>7} {'R_sh':>7} {'R_bld':>7} {'V0':>7} {'V1':>7} {'td':>7} "
           f"{'carry':>7} {'adv':>7} {'adv_n':>6} {'>0':>5}")

    print("\n(A) shop sells by place x slot")
    print(hdr)
    for st in sorted({r["state"] for r in sells}):
        for fr in (True, False):
            line(f"{st[:18]} {'free' if fr else 'full'}", [r for r in sells if r["state"] == st and free(r) == fr])
    line("ALL sells", sells, f"  blind between: {np.mean([r['blind_between'] for r in sells]) if sells else 0:.2f}")

    print("\n(B) sells vs other moves at sell-legal shop states, by ante x slot")
    print(hdr + "   P(any sell)")
    for an in ("1", "2", "3", "4", "5+"):
        for fr in (True, False):
            at = [r for r in legal if bucket_ante(r["ante"]) == an and free(r) == fr]
            s = [r for r in at if r["verb"] == "sell_joker"]
            o = [r for r in at if r["verb"] != "sell_joker"]
            ps = f"   {np.mean([r['p_sell'] for r in at]):.4f}" if at else ""
            line(f"ante {an} {'free' if fr else 'full'} sell", s, ps)
            line(f"ante {an} {'free' if fr else 'full'} other", o)

    print("\n(C) every move at sell-legal shop states, by verb x slot")
    print(hdr)
    by = collections.defaultdict(list)
    for r in legal:
        by[(r["verb"], free(r))].append(r)
    for (v, fr), ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
        line(f"{v[:22]} {'free' if fr else 'full'}", ix)

    print("\n(D) shop critic by jokers held, at the SHOP, antes 1-3: V0 vs its target (adv+V0), money")
    print(f"{'ante':>4} {'jokers':>6} {'n':>6} {'V0':>7} {'target':>7} {'gap':>7} {'money':>7}")
    at_shop = [r for r in shop if r["state"] == "SHOP" and r["ante"] <= 3]
    for an in (1, 2, 3):
        for j in range(0, 6):
            ix = [r for r in at_shop if r["ante"] == an and min(r["jokers"], 5) == j]
            if ix:
                v0 = np.mean([r["V0"] for r in ix]); tg = np.mean([r["V0"] + r["adv"] for r in ix])
                print(f"{an:4d} {('5+' if j == 5 else str(j)):>6} {len(ix):6d} {v0:7.3f} {tg:7.3f} {tg - v0:7.3f} "
                      f"{np.nanmean([r['money'] for r in ix]):7.1f}")

    print("\n(E) jokers sold (top 12): n, free share, mean adv, mean td, mean R_bld")
    ks = collections.defaultdict(list)
    for r in sells:
        ks[r["key"]].append(r)
    for k, ix in sorted(ks.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {str(k):22} {len(ix):5d} {np.mean([free(r) for r in ix]):5.2f} {np.mean([r['adv'] for r in ix]):7.3f} "
              f"{np.mean([r['td'] for r in ix]):7.3f} {np.mean([r['Rp'] for r in ix]):7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", default="v4b")
    ap.add_argument("--its", default="160,240,260,280")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--roots", default=",".join(ROOTS), help="comma-separated runs/ folders to search")
    ap.add_argument("--max-games", type=int, default=0, help="per checkpoint; 0 = all in the window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(4)
    roots = a.roots.split(",")
    out = open(a.out, "a") if a.out else None         # appends: a rerun of later checkpoints keeps earlier lines
    sys.stdout.reconfigure(line_buffering=True)        # through tee: every line reaches the file at once
    print(f"device {a.device}; roots {roots}")
    for it in [int(x) for x in a.its.split(",")]:
        ck = find(a.run, f"ckpt_{it:05d}.pt", roots)
        files = [find(a.run, f"r{k:05d}_train.jsonl.gz", roots) for k in range(it - a.window + 1, it + 1)]
        files = [f for f in files if f]
        if not ck or not files:
            print(f"\n=== ckpt {it}: checkpoint {'found' if ck else 'MISSING'}, {len(files)} game files; skipped")
            continue
        print(f"\n--- ckpt {it}: {ck}\n    games: {', '.join(os.path.basename(f) for f in files)}")
        model, sim, cfg = load(ck, a.src, a.device)
        rows, games, skipped, t0 = [], 0, 0, time.time()
        for f in files:
            for line in gzip.open(f, "rt"):
                ep = json.loads(line)
                if ep.get("mode", "train") != "train" or ep.get("curriculum"):
                    continue
                steps = replay(model, sim, cfg, ep, a.device)
                if steps is None:
                    skipped += 1
                    continue
                rows.extend(steps)
                games += 1
                if games % 25 == 0:
                    gpu = torch.cuda.memory_allocated() / 2**30 if a.device.startswith("cuda") else 0.0
                    print(f"    {games} games, {len(rows)} decisions, {time.time() - t0:.0f} s, "
                          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB, "
                          f"GPU {gpu:.2f} GB", flush=True)
                if out:
                    for s in steps:
                        if s["member"] == SHOP and s["sell_legal"]:
                            out.write(json.dumps({"ckpt": it, "seed": ep["seed"], "iteration": ep.get("iteration"),
                                                  **{k: v for k, v in s.items() if k not in ("hand",)}}) + "\n")
                    out.flush()
                if a.max_games and games >= a.max_games:
                    break
            if a.max_games and games >= a.max_games:
                break
        if skipped:
            print(f"    {skipped} games did not replay (skipped)")
        report(it, rows, games, time.time() - t0)
        sys.stdout.flush()
        del model, sim, rows
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if out:
        out.close()


if __name__ == "__main__":
    main()
