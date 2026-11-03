"""Re-solve: plays released weights on every published won seed K times, and on as many fresh
control seeds K times, to measure each seed's win probability under one fixed player.

    # play (resumable: rerun the same command to continue after a stop)
    python3 resolve.py play ../weights/balatro-bot-1.0.pt --src ~/balatro/local/source \
        --routes ../wins/routes.jsonl --k 10 --jobs 12 --out resolve_1.0.jsonl --routes-out resolve_1.0_wins.jsonl
    # summarize
    python3 resolve.py summary resolve_1.0.jsonl --routes ../wins/routes.jsonl --k 10 --ranked resolve_1.0_ranked.csv

The question: is a win a property of the seed, or of the moves the policy happened to sample?
If 1.0 wins the published seeds at about its fresh-seed rate (~2.5%), the seed carried nothing;
if well above, winnable seeds are a stable fact about the game and the seeds can be ranked.

Moves are sampled exactly as in train/play.py (RNG "play-{sample_seed}-{seed}"), so any game here
can be watched again with `play.py WEIGHTS --seed S --sample-seed J`. Won games are also written
as routes (--routes-out) that train/verify_games.py replays.

Control seeds come from a fixed RNG ("resolve-control-v1"), excluding every published seed and the
200 evaluation seeds. Exposure: 1.0 is round 810 of the 0.3 run, so it trained on most published
seeds' games. The 0.3 wins found after round 810 on a fresh seed (not a retry) were played by later
weights and are unseen by 1.0; the summary reports them separately.
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "train")
sys.path.insert(0, TRAIN)

MODEL = {"v1": "0.1", "low": "0.1", "high": "0.1", "off": "0.1", "on": "0.1", "v2": "0.2", "v2b": "0.3"}
SNAPSHOT_ROUND = {"1.0": 810}       # round of the 0.3 run each released model was taken at


def load_routes(path):
    out = []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            run = r.get("source", "").split("/")[0]
            out.append({"seed": r["seed"], "back": r["back"], "stake": r["stake"], "run": run,
                        "model": MODEL.get(run, run or "unknown"), "retry": bool(r.get("retry")),
                        "iteration": r.get("iteration"), "found_blinds": r["progress"]["blinds_beaten"]})
    return out


def control_seeds(n, exclude):
    from train import eval_seeds, rand_seed
    exclude = set(exclude) | set(eval_seeds(200))
    rng = random.Random("resolve-control-v1")
    out = []
    while len(out) < n:
        s = rand_seed(rng)
        if s not in exclude:
            exclude.add(s)
            out.append(s)
    return out


# ---------------- play ----------------

_w = {}


def worker_init(weights, src):
    import torch
    torch.set_num_threads(1)
    from model import Net
    from sim import Sim
    w = torch.load(weights, map_location="cpu", weights_only=True)
    arch = w["arch"]
    sim = Sim(src, fast=True, mode=arch["actions"], enc=arch["enc"])
    if sim.vocab != w["vocab"]:
        raise SystemExit("these weights were trained on a different game version (vocabulary differs)")
    net = Net(len(w["vocab"]), d=arch["d"], layers=arch["layers"], heads=arch["heads"], enc=arch["enc"])
    net.load_state_dict(w["state_dict"])
    net.eval()
    _w.update(sim=sim, net=net, arch=arch, version=w["model_version"])


def play_one(job):
    import numpy as np
    import torch
    from train import collate
    seed, back, stake, j, group, max_steps = job
    sim, net = _w["sim"], _w["net"]
    rng = random.Random(f"play-{j}-{seed}")
    o = sim.reset(seed, back, stake)
    done = False
    while not done and sim.steps < max_steps:
        with torch.no_grad():
            logits, _ = net(*collate([o]))
        p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
        i = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), len(p) - 1)
        _, _, done = sim.step(i)
        if not done:
            o = sim.observe()
    prog = sim.progress
    rec = {"seed": seed, "back": back, "stake": stake, "sample_seed": j, "group": group,
           "won": prog["won"], "blinds": prog["blinds_beaten"], "ante": prog["ante"], "steps": sim.steps,
           "capped": not done, "model_version": _w["version"]}
    route = None
    if prog["won"]:
        route = {"seed": seed, "back": back, "stake": stake, "actions": _w["arch"]["actions"],
                 "enc": _w["arch"]["enc"], "acts": list(sim.route), "desc": list(sim.desc),
                 "final": sim.final_hash(), "n": sim.steps, "progress": prog, "mode": "resolve",
                 "iteration": None, "sample_seed": j, "model_version": _w["version"]}
    return rec, route


def cmd_play(a):
    import multiprocessing as mp
    worker_init(a.weights, a.src)       # check the weights and the game here, before starting workers
    _w.clear()
    routes = load_routes(a.routes)
    won = [(r["seed"], r["back"], r["stake"]) for r in routes]
    if a.limit:
        won = won[:a.limit]
    n_ctrl = len(won) if a.control < 0 else a.control
    ctrl = [(s, "b_red", 1) for s in control_seeds(n_ctrl, [r["seed"] for r in routes])]
    jobs = [(s, b, st, j, g, a.max_steps) for g, seeds in (("won", won), ("control", ctrl))
            for s, b, st in seeds for j in range(a.k)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["seed"], r["sample_seed"], r["group"]))
    jobs = [x for x in jobs if (x[0], x[3], x[4]) not in done]
    si, sn = map(int, a.shard.split("/"))
    jobs = jobs[si::sn]
    # interleave the two groups so a partial run is still a fair comparison
    jobs.sort(key=lambda x: (x[3], x[0][::-1]))
    print(f"{len(won)} published seeds, {len(ctrl)} control seeds, k={a.k}: "
          f"{len(done)} games already done, {len(jobs)} to play on {a.jobs} processes", flush=True)
    n = w = 0
    with open(a.out, "a") as f, (open(a.routes_out, "a") if a.routes_out else open(os.devnull, "w")) as fr, \
            mp.get_context("spawn").Pool(a.jobs, initializer=worker_init, initargs=(a.weights, a.src)) as pool:
        for rec, route in pool.imap_unordered(play_one, jobs, chunksize=4):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if route:
                fr.write(json.dumps(route) + "\n")
                fr.flush()
            n += 1
            w += rec["won"]
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} played, {w} won", flush=True)
    print(f"done: {n} played, {w} won")


# ---------------- summary ----------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def rate_line(name, games):
    n, k = len(games), sum(g["won"] for g in games)
    if not n:
        return f"  {name:<34} no games"
    lo, hi = wilson(k, n)
    b = sum(g["blinds"] for g in games) / n
    return f"  {name:<34} {k:5d}/{n:<6d} = {100 * k / n:5.2f}%  (95% {100 * lo:.2f}–{100 * hi:.2f}), blinds/game {b:.2f}"


def per_seed(games):
    d = collections.defaultdict(list)
    for g in games:
        d[g["seed"]].append(g)
    return d


def dispersion(seeds, k):
    """Are per-seed win counts more spread than Binomial(k, mean)? Index D = sum (x - kp)^2 / (kp(1-p)),
    about chi-squared with n-1 df if seeds don't matter; normal approximation for the p-value."""
    xs = [sum(g["won"] for g in gs) for gs in seeds.values() if len(gs) == k]
    n = len(xs)
    if n < 2:
        return None
    p = sum(xs) / (n * k)
    if p in (0, 1):
        return None
    D = sum((x - k * p) ** 2 for x in xs) / (k * p * (1 - p))
    z = (D - (n - 1)) / math.sqrt(2 * (n - 1))
    return n, p, D / (n - 1), norm_sf(z), collections.Counter(xs)


def boot_diff(a, b, reps=2000):
    """Seed-level bootstrap of mean per-seed win rate, a minus b."""
    rng = random.Random("resolve-boot")
    ra = [sum(g["won"] for g in gs) / len(gs) for gs in a.values()]
    rb = [sum(g["won"] for g in gs) / len(gs) for gs in b.values()]
    m = lambda xs: sum(xs) / len(xs)
    ds = sorted(m(rng.choices(ra, k=len(ra))) - m(rng.choices(rb, k=len(rb))) for _ in range(reps))
    return m(ra) - m(rb), ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def cmd_summary(a):
    games = [json.loads(l) for l in open(a.results) if l.strip()]
    info = {r["seed"]: r for r in load_routes(a.routes)}
    version = games[0]["model_version"] if games else "?"
    snap = SNAPSHOT_ROUND.get(version)
    W = [g for g in games if g["group"] == "won"]
    C = [g for g in games if g["group"] == "control"]
    print(f"# Re-solve with model {version}: {len(games)} games, k={a.k} per seed")
    print(f"(capped at the move limit: {sum(g['capped'] for g in games)})\n")
    print("## Win rate per game")
    print(rate_line("control (fresh seeds)", C))
    print(rate_line("published won seeds", W))
    for m in sorted({info[g["seed"]]["model"] for g in W if g["seed"] in info}):
        print(rate_line(f"  first won by model {m}", [g for g in W if info.get(g["seed"], {}).get("model") == m]))
    print(rate_line("  found on a fresh seed", [g for g in W if not info.get(g["seed"], {}).get("retry")]))
    print(rate_line("  found on a retry", [g for g in W if info.get(g["seed"], {}).get("retry")]))
    if snap is not None:
        unseen = lambda r: r["run"] == "v2b" and not r["retry"] and (r["iteration"] or 0) > snap
        print(rate_line(f"  unseen by {version} (0.3, fresh, round > {snap})",
                        [g for g in W if g["seed"] in info and unseen(info[g["seed"]])]))
        print(rate_line(f"  possibly trained on by {version}",
                        [g for g in W if g["seed"] in info and not unseen(info[g["seed"]])]))

    sw, sc = per_seed(W), per_seed(C)
    print("\n## Per seed")
    for name, s in (("control", sc), ("published", sw)):
        if not s:
            continue
        any_win = sum(any(g["won"] for g in gs) for gs in s.values())
        print(f"  {name}: {len(s)} seeds, won at least once by {version}: {any_win} ({100 * any_win / len(s):.1f}%)")
        d = dispersion(s, a.k)
        if d:
            n, p, ratio, pv, hist = d
            exp = [n * math.comb(a.k, x) * p ** x * (1 - p) ** (a.k - x) for x in range(a.k + 1)]
            print(f"    wins out of {a.k} per seed (observed / expected if seeds didn't matter):")
            print("    " + "  ".join(f"{x}: {hist.get(x, 0)}/{exp[x]:.0f}" for x in range(a.k + 1)))
            print(f"    dispersion {ratio:.2f} (1 = binomial), p = {pv:.2g} (normal approx.; full-k seeds only)")
    if sw and sc:
        d, lo, hi = boot_diff(sw, sc)
        print(f"\n## Published minus control, mean per-seed win rate: {100 * d:+.2f} points "
              f"(seed bootstrap 95% {100 * lo:+.2f} to {100 * hi:+.2f})")

    if a.ranked:
        rows = []
        for s, gs in list(sw.items()) + list(sc.items()):
            r = info.get(s, {})
            rows.append({"seed": s, "group": gs[0]["group"], "games": len(gs), "wins": sum(g["won"] for g in gs),
                         "mean_blinds": round(sum(g["blinds"] for g in gs) / len(gs), 2),
                         "first_won_by": r.get("model", ""), "retry": r.get("retry", ""),
                         "found_round": r.get("iteration", "")})
        rows.sort(key=lambda r: (-r["wins"] / r["games"], -r["mean_blinds"], r["seed"]))
        with open(a.ranked, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nranked seeds written to {a.ranked}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("weights")
    p.add_argument("--src", required=True, help="game source extracted from your own copy")
    p.add_argument("--routes", default=os.path.join(HERE, "..", "wins", "routes.jsonl"))
    p.add_argument("--k", type=int, default=10, help="games per seed")
    p.add_argument("--control", type=int, default=-1, help="control seeds (default: as many as published)")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--out", required=True)
    p.add_argument("--routes-out", help="write won games as replayable routes")
    p.add_argument("--limit", type=int, default=0, help="first N published seeds only (for a test)")
    p.add_argument("--shard", default="0/1", help="I/N: play every Nth remaining game, from I")
    s = sub.add_parser("summary")
    s.add_argument("results")
    s.add_argument("--routes", default=os.path.join(HERE, "..", "wins", "routes.jsonl"))
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--ranked", help="CSV of seeds, easiest first")
    a = ap.parse_args()
    cmd_play(a) if a.cmd == "play" else cmd_summary(a)


if __name__ == "__main__":
    main()
