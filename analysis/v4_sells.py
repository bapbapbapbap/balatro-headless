"""Why v4's table network sells its jokers mid-blind (diagnostic, 2026-11-10).

Replays saved v4 training games on the sim (phi_build and offer values on, as in training), runs a
checkpoint on every decision, rebuilds each step's training reward (blinds + both shaping terms, blind
clock) and the routed advantages exactly as train.ppo_update forms them (credit.routed_gae), and prints,
per move type and network: count, mean reward on the step, mean change in phi_build, mean advantage
(raw and per-network normalized), share with advantage > 0. Sells get extra detail (money, joker count).
Also counts taken moves whose float32 probability is 0 (a worker would store log(1e-30) for them).

usage (WSL, venv active, from the repo):
  python3 analysis/v4_sells.py --src ~/balatro/local/source \
      --pair RUNS/v4/ckpt_00120.pt:RUNS/v4/games/r00120_train.jsonl.gz \
      --pair RUNS/v4/ckpt_00300.pt:RUNS/v4/games/r00300_train.jsonl.gz
Curriculum games are skipped (they start mid-run); --curriculum reads only them, from their start. Read-only: writes nothing.
"""
import argparse
import collections
import gzip
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from credit import routed_gae  # noqa: E402
from model import PHASE_NAMES, make_model, phase_member  # noqa: E402
from rewards import REWARDS, build_shaping, shaping, step_discount  # noqa: E402
from sim import Sim  # noqa: E402
from train import collate  # noqa: E402

FLOOR = math.log(1e-30)


def money(sim):
    try:
        return float(sim.env.lua.globals().G.GAME.dollars)
    except Exception:
        return float("nan")


def n_jokers(sim):
    try:
        return int(sim.env.lua.eval("#G.jokers.cards"))
    except Exception:
        return -1


def run_pair(ckpt_path, games_path, src, max_games, which="own"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    arch = ck["arch"]
    mcfg = SimpleNamespace(phase_split=True, offer_values=arch["offer_values"], d=arch["d"],
                           layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    model = make_model(mcfg, len(ck["vocab"]))
    model.load_state_dict(ck["model"])
    model.eval()
    args = SimpleNamespace(**cfg)
    sim = Sim(src, None, mode=cfg["actions"], enc=arch["enc"], build_phi=True, offer_values=arch["offer_values"])
    assert sim.vocab == ck["vocab"], "game vocabulary differs from the checkpoint's"
    reward_fn = REWARDS[cfg["reward"]]

    rows = []          # one per decision: (verb, member, r, dphi_build, adv, logp, dmoney, jokers_before, ante)
    games = 0
    t0 = time.time()
    for line in gzip.open(games_path, "rt"):
        ep = json.loads(line)
        cur = ep.get("curriculum")
        if ep.get("mode", "train") != "train" or bool(cur) != (which == "curriculum"):
            continue
        o = sim.reset(ep["seed"], ep["back"], ep["stake"], ep.get("challenge"))
        st = cur["start_step"] if cur else 0
        if st:                                # curriculum: the replayed start is not the bot's play
            o = sim.fast_forward(ep["acts"][:st], ep["desc"][:st])
            if o is None:
                print(f"  {ep['seed']}: curriculum start did not replay; game skipped")
                continue
        R, V, G, L, own, info = [], [], [], [], [], []
        done = False
        for k, a in enumerate(ep["acts"]):
            if k < st:
                continue
            if done or a >= len(o["act"]) or sim.H.action_desc(a + 1) != ep["desc"][k]:
                print(f"  {ep['seed']}: replay diverged at step {k}; game skipped")
                R = None
                break
            o["member"] = phase_member(sim.H.state_name())
            with torch.no_grad():
                logits, v = model(*collate([o]))
            lp_all = torch.log_softmax(logits[0], -1)
            p32 = torch.softmax(logits[0], -1)[a].item()
            m0, j0 = money(sim), n_jokers(sim)
            prev, cur, done = sim.step(a)
            g, l = step_discount(prev, cur, args)
            r = reward_fn(prev, cur) + cfg.get("shaping", 1.0) * shaping(prev, cur, g) \
                + cfg.get("build_shaping", 0) * build_shaping(prev, cur, g)
            R.append(r); V.append(float(v[0])); G.append(g); L.append(l); own.append(o["member"])
            info.append((ep["desc"][k].split()[0], o["member"], r,
                         cur.get("phi_build", 0.0) - prev.get("phi_build", 0.0),
                         float(lp_all[a]) if p32 > 0 else FLOOR, money(sim) - m0, j0, prev["ante"], p32 == 0.0,
                         float(lp_all[a])))
            if not done:
                o = sim.observe()
        if R is None:
            continue
        if not done:                       # capped game: bootstrap from the last state, as train.py does
            o["member"] = phase_member(sim.H.state_name())
            with torch.no_grad():
                V.append(float(model(*collate([o]))[1][0]))
        adv = routed_gae(R, V, G, L, own)
        for x, ad in zip(info, adv):
            rows.append(x[:4] + (float(ad),) + x[4:])
        games += 1
        if max_games and games >= max_games:
            break
    print(f"\n=== {os.path.basename(ckpt_path)} on {os.path.basename(games_path)} ({which} games): {games} games, "
          f"{len(rows)} decisions, {time.time() - t0:.0f} s")
    report(rows)


def report(rows):
    adv = np.array([r[4] for r in rows])
    mem = np.array([r[1] for r in rows])
    norm = np.zeros_like(adv)
    for m in set(mem.tolist()):
        s = mem == m
        norm[s] = (adv[s] - adv[s].mean()) / (adv[s].std() + 1e-8)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[(PHASE_NAMES[r[1]], r[0])].append(i)
    print(f"{'net':5} {'move':16} {'n':>6} {'reward':>8} {'dphi_b':>8} {'adv':>8} {'adv_n':>7} {'adv>0':>6} {'p=0':>5}")
    for (net, verb), ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ix = np.array(ix)
        rr = np.array([rows[i][2] for i in ix]); dp = np.array([rows[i][3] for i in ix])
        z = np.array([rows[i][9] for i in ix])
        print(f"{net:5} {verb:16} {len(ix):6d} {rr.mean():8.3f} {dp.mean():8.3f} {adv[ix].mean():8.3f} "
              f"{norm[ix].mean():7.3f} {(adv[ix] > 0).mean():6.2f} {z.sum():5d}")
    sells = [r for r in rows if r[0] == "sell_joker" and r[1] == 0]
    if sells:
        print(f"\ntable sells: {len(sells)}; mean money +{np.nanmean([s[6] for s in sells]):.2f}, "
              f"jokers held before {np.mean([s[7] for s in sells]):.2f}, mean ante {np.mean([s[8] for s in sells]):.2f}")
        print("  dphi_build of sells: min {:.3f}  median {:.3f}  max {:.3f}; share > 0: {:.2f}".format(
            *np.percentile([s[3] for s in sells], [0, 50, 100]), np.mean([s[3] > 0 for s in sells])))
    z = [r for r in rows if r[9]]
    print(f"\ntaken moves with float32 probability 0 (stored at the log(1e-30) floor): {len(z)} of {len(rows)}; "
          f"their true log-prob, median {np.median([r[10] for r in z]) if z else float('nan'):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pair", action="append", required=True, help="CKPT.pt:GAMES.jsonl.gz")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--curriculum", action="store_true", help="curriculum games only (from their start)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    for p in a.pair:
        ck, gm = p.rsplit(":", 1)
        run_pair(ck, gm, a.src, a.max_games, "curriculum" if a.curriculum else "own")


if __name__ == "__main__":
    main()
