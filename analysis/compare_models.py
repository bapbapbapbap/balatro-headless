"""Two trainer checkpoints, paired on the same seeds: model v3's A/B checks and its final test
(docs/hypotheses.md, 2026-11-05 ~21:19 CST and the stopping-rule amendment ~21:24).

  python3 compare_models.py A.pt B.pt --src ~/balatro/local/source --seed-tag v3-ab --seeds 2000 \\
      --workers 12 --out ab_check1.jsonl [--look 1] [--eval-temp 1.0]
  python3 compare_models.py --summary ab_check1.jsonl --look 1

Each checkpoint plays each seed once exactly as its own evals do (train.play_games, mode "eval":
sampled with a fixed RNG per seed; MoE checkpoints play through their router and handoff). Seeds
come from --seed-tag and never include the eval 600 or another registered tag's seeds. The output
resumes (one line per seed). The summary gives the paired difference in blinds per game (A - B),
wins and the exact one-sided McNemar test, how each model's games end by ante, and, with --look k,
the O'Brien-Fleming boundary for check k of 3 (|z| > 3.47, 2.45, 2.00; overall two-sided alpha 0.05).
Each game's moves are saved, so anything else (money held at ante 6, for V3) can be read by replay.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
TAGS = ("paired-v1", "v3-ab", "v3-test", "v4-test")        # registered seed lists, kept disjoint
OBF = {1: 3.471, 2: 2.454, 3: 2.004}             # O'Brien-Fleming, 3 equally spaced looks, two-sided 0.05
G = {}


def seed_list(tag, n, rand_seed, banned):
    rng, out = random.Random(tag), []
    while len(out) < n:
        s = rand_seed(rng)
        if s not in banned and s not in out:
            out.append(s)
    return out


def load(path, src, temp):
    import torch
    from argparse import Namespace
    from model import make_model
    from sim import Sim
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    cfg.update(src=src, verify_wins=False, save_games="none", eval_temp=temp)
    a = Namespace(**cfg)
    sim = Sim(a.src, a.mods, mode=a.actions, enc=a.enc)
    if list(sim.vocab) != list(ck["vocab"]):
        raise SystemExit(f"{path}: checkpoint vocabulary differs from this game's")
    model = make_model(a, len(sim.vocab))
    model.load_state_dict(ck["model"])
    model.eval()
    return {"args": a, "sim": sim, "model": model, "state": model.state_dict(), "it": ck.get("iteration", 0)}


def init(paths, src, temp):
    import torch
    torch.set_num_threads(1)
    import train as T
    from rewards import REWARDS
    G["T"], G["R"] = T, REWARDS
    G["m"] = [load(p, src, temp) for p in paths]


def one_seed(seed):
    try:
        T, out = G["T"], {"seed": seed}
        for name, m in zip("AB", G["m"]):
            t0 = time.time()
            ep = T.play_games(m["sim"], m["model"], G["R"][m["args"].reward], random.Random(0), m["args"], 0,
                              (m["state"], [seed], "eval", m["it"], 0), set())[0]
            p = ep["progress"]
            out[name] = {"won": bool(p["won"]), "near_win": bool(p.get("near_win")), "blinds": p["blinds_beaten"],
                         "ante": p["ante"], "capped": ep["capped"], "acts": ep["acts"],
                         # W4a (docs/hypotheses.md 2026-11-09): this game's play time, model + sim, in one process
                         "seconds": round(time.time() - t0, 3)}
        return out
    except Exception:
        return {"seed": seed, "error": traceback.format_exc()[-3000:]}


def p_mcnemar_one_sided(b, c):
    """P(X >= b) for X ~ Binomial(b + c, 1/2): A wins more of the discordant seeds than chance."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def summarize(recs, look=None, names=("A", "B")):
    recs = [r for r in recs if not r.get("error")]
    n = len(recs)
    bA = np.array([r["A"]["blinds"] for r in recs], float)
    bB = np.array([r["B"]["blinds"] for r in recs], float)
    wA = np.array([r["A"]["won"] for r in recs], int)
    wB = np.array([r["B"]["won"] for r in recs], int)
    d = bA - bB
    se = d.std(ddof=1) / math.sqrt(n)
    z = d.mean() / se
    b, c = int(((wA == 1) & (wB == 0)).sum()), int(((wA == 0) & (wB == 1)).sum())
    print(f"{n} seeds")
    print(f"  {names[0]}: {bA.mean():.3f} blinds/game, {wA.sum()} wins ({wA.mean():.2%}), near wins "
          f"{sum(r['A']['near_win'] for r in recs)}")
    print(f"  {names[1]}: {bB.mean():.3f} blinds/game, {wB.sum()} wins ({wB.mean():.2%}), near wins "
          f"{sum(r['B']['near_win'] for r in recs)}")
    print(f"  blinds A - B: {d.mean():+.3f} (SE {se:.3f}, 95% CI {d.mean() - 1.96 * se:+.3f} to {d.mean() + 1.96 * se:+.3f}), "
          f"z = {z:+.2f}, paired SD {d.std(ddof=1):.2f}, correlation {np.corrcoef(bA, bB)[0, 1]:.2f}")
    print(f"  wins: A only {b}, B only {c}; one-sided exact McNemar (A > B) p = {p_mcnemar_one_sided(b, c):.4f}")
    for name, k in zip(names, "AB"):
        ends = {}
        for r in recs:
            g = r[k]
            if not g["won"]:
                ends[g["ante"]] = ends.get(g["ante"], 0) + 1
        mid = sum(v for a, v in ends.items() if 4 <= a <= 6)
        print(f"  {name} losses by ante: " + ", ".join(f"{a}:{ends[a]}" for a in sorted(ends))
              + f"  (antes 4-6: {mid}, {mid / n:.1%} of games)")
    if all("seconds" in r["A"] and "seconds" in r["B"] for r in recs):
        w4a(recs, names)
    if look:
        bound = OBF[look]
        verdict = (f"STOP: |z| {abs(z):.2f} > {bound:.2f}, keep {'A' if z > 0 else 'B'}" if abs(z) > bound else
                   (f"no boundary crossed at the last check: a tie, keep v3" if look == 3 else
                    f"continue: |z| {abs(z):.2f} <= {bound:.2f}"))
        print(f"  check {look} of 3 (O'Brien-Fleming): {verdict}")


def w4a(recs, names=("A", "B"), resamples=10000):
    """W4a (docs/hypotheses.md, 2026-11-09): wins per sim-hour, A over B, with the one-sided 95% lower bound
    from a paired bootstrap over seeds (10,000 resamples, fixed RNG so the bound is reproducible)."""
    wA = np.array([r["A"]["won"] for r in recs], float)
    wB = np.array([r["B"]["won"] for r in recs], float)
    sA = np.array([r["A"]["seconds"] for r in recs], float)
    sB = np.array([r["B"]["seconds"] for r in recs], float)
    rate = lambda w, s: w.sum() / max(s.sum() / 3600, 1e-12)
    rA, rB = rate(wA, sA), rate(wB, sB)
    print(f"  W4a wins per sim-hour: {names[0]} {rA:.3f} ({sA.sum() / 3600:.2f} h, {sA.mean():.2f} s/game), "
          f"{names[1]} {rB:.3f} ({sB.sum() / 3600:.2f} h, {sB.mean():.2f} s/game)")
    if wB.sum() == 0:
        print("  W4a ratio: undefined (B won no game)")
        return
    rng = np.random.default_rng(20261109)
    n, ratios = len(recs), []
    for _ in range(resamples):
        i = rng.integers(0, n, n)
        b = rate(wB[i], sB[i])
        ratios.append(rate(wA[i], sA[i]) / b if b > 0 else np.inf)
    lo = float(np.percentile(ratios, 5))
    ratio = rA / rB
    tier = ("met strongly" if lo >= 2 else "met" if ratio >= 2 and lo > 1 else "not met")
    print(f"  W4a ratio {names[0]}/{names[1]} = {ratio:.2f}, one-sided 95% lower bound {lo:.2f} -> W4a {tier}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--src")
    ap.add_argument("--seed-tag", default="v3-ab", choices=TAGS[1:])
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--look", type=int, choices=[1, 2, 3])
    ap.add_argument("--out", default="compare.jsonl")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summarize([json.loads(l) for l in open(a.summary) if l.strip()], a.look)
        return
    if not (a.a and a.b and a.src):
        ap.error("A.pt, B.pt and --src are needed to play")
    from train import eval_seeds, rand_seed
    banned = set(eval_seeds(600))
    for t in TAGS:
        if t != a.seed_tag:
            banned |= set(seed_list(t, 2000, rand_seed, set(eval_seeds(600))))
    seeds = seed_list(a.seed_tag, a.seeds, rand_seed, banned)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["seed"] for l in open(a.out) if l.strip() and not json.loads(l).get("error")}
    todo = [s for s in seeds if s not in done]
    print(f"A = {a.a}\nB = {a.b}\n{len(seeds)} seeds from '{a.seed_tag}' ({len(done)} already done), "
          f"eval temperature {a.eval_temp}, {a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=init, initargs=((a.a, a.b), a.src, a.eval_temp)) as pool, open(a.out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(one_seed, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec.get("error"):
                print(f"seed {rec['seed']}: error, skipped: {rec['error'].strip().splitlines()[-1]}", flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M:%S')}  {i} of {len(todo)}  {i / max(time.time() - t0, 1e-9) * 3600:,.0f} seeds/h", flush=True)
    summarize([json.loads(l) for l in open(a.out) if l.strip()], a.look)


if __name__ == "__main__":
    main()
