"""One model on the eval seeds at any deck, stake and eval temperature (docs/hypotheses-v6.md, 2026-11-20: M1,
the stake sweeps of roadmap step 3, the no-ought floors N0/N1).

  python3 tools/eval_ladder.py --src ~/balatro/local/source --weights runs/v6/best.pt --out m1_t0.jsonl \\
      [--temp 0] [--back b_red] [--stake 1] [--games 600] [--workers 12] [--policy model|uniform|untrained]
  python3 tools/eval_ladder.py --summary m1_t0.jsonl

Each game is played exactly as a run's eval plays it (train.play_games, mode "eval": a fixed sampling RNG per
seed), so at temperature 1, Red deck, White stake the checkpoint's own eval is reproduced seed for seed. The seeds
are the eval seeds (common.eval_seeds): --games 600 is the registered set, a larger n extends it (its first 600
are the same). --policy uniform plays a legal move uniformly at random (N0: no preference); untrained plays a
randomly initialized network of the checkpoint's architecture (N1; --init-seed). The checkpoint gives the game
settings (encoding, action mode, offer and play values) in every case.

Output: one line per game (resumes), with the moves, so every win is a route that replays (wins/README.md).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
G = {}


def build(a):
    import torch
    from argparse import Namespace
    from model import make_model
    from sim import Sim, sim_kwargs
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    cfg = dict(ck["provenance"]["config"])
    cfg.update(src=a.src, verify_wins=False, save_games="none", eval_temp=a.temp, back=a.back, stake=a.stake,
               backs=[], stakes=[], challenges=[])
    args = Namespace(**cfg)
    sim = Sim(args.src, args.mods, mode=args.actions, enc=args.enc, **sim_kwargs(args))
    if list(sim.vocab) != list(ck["vocab"]):
        raise SystemExit(f"{a.weights}: checkpoint vocabulary differs from this game's")
    if a.policy == "uniform":
        model = Uniform()
    else:
        if a.policy == "untrained":
            torch.manual_seed(a.init_seed)
        model = make_model(args, len(sim.vocab))
        if a.policy == "model":
            model.load_state_dict(ck["model"])
    model.eval()
    return {"args": args, "sim": sim, "model": model, "state": model.state_dict(), "it": ck.get("iteration", 0)}


def Uniform():
    import torch

    class _Uniform(torch.nn.Module):
        """Every legal move equally likely; the value is 0 (N0)."""
        def forward(self, kind, vocab, feat, tok_mask, act, act_mask, *member):
            logits = torch.zeros(act_mask.shape, dtype=torch.float32).masked_fill(~act_mask, float("-inf"))
            return logits, torch.zeros(act_mask.shape[0])
    return _Uniform()


def init(a):
    import torch
    torch.set_num_threads(1)
    import train as T
    from rewards import REWARDS
    G["T"], G["R"], G["m"] = T, REWARDS, build(a)


def one_seed(seed):
    try:
        T, m = G["T"], G["m"]
        t0 = time.time()
        ep = T.play_games(m["sim"], m["model"], G["R"][m["args"].reward], random.Random(0), m["args"], 0,
                          (m["state"], [seed], "eval", m["it"], 0), set())[0]
        p = ep["progress"]
        return {"seed": seed, "won": bool(p["won"]), "near_win": bool(p.get("near_win")),
                "blinds": p["blinds_beaten"], "ante": p["ante"], "capped": bool(ep["capped"]),
                "seconds": round(time.time() - t0, 2), "acts": ep["acts"]}
    except Exception:
        return {"seed": seed, "error": traceback.format_exc()[-3000:]}


def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    head = recs[0] if recs and "run" in recs[0] else {}
    games = [r for r in recs if "seed" in r and not r.get("error")]
    errors = sum(1 for r in recs if r.get("error"))
    n, w = len(games), sum(r["won"] for r in games)
    if not n:
        print(f"{path}: no games yet ({errors} errors)")
        return
    p = w / n
    lo, hi = wilson(w, n)
    ends = {}
    for r in games:
        if not r["won"]:
            ends[r["ante"]] = ends.get(r["ante"], 0) + 1
    print(f"{path}: {head.get('run', '')}")
    print(f"  {n} games ({errors} errors): won {w} ({p:.1%}, 95% {lo:.1%}-{hi:.1%}), "
          f"{sum(r['blinds'] for r in games) / n:.2f} blinds/game, near wins {sum(r['near_win'] for r in games)}, "
          f"capped {sum(r['capped'] for r in games)}, {sum(r['seconds'] for r in games) / n:.1f} s/game")
    print("  losses by ante: " + ", ".join(f"{k}:{ends[k]}" for k in sorted(ends)))
    if w == 0:
        print(f"  no win: the win rate is below {3 / n:.2%} (95%, rule of three)")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--weights", help="a trainer checkpoint (runs/<run>/best.pt): the model and the game settings")
    ap.add_argument("--policy", default="model", choices=["model", "uniform", "untrained"])
    ap.add_argument("--init-seed", type=int, default=0, help="--policy untrained: the initialization's seed")
    ap.add_argument("--temp", type=float, default=1.0, help="eval temperature; 0 = the most likely move")
    ap.add_argument("--back", default="b_red")
    ap.add_argument("--stake", type=int, default=1, help="1 White ... 8 Gold")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summarize(a.summary)
        return
    if not (a.src and a.weights and a.out):
        ap.error("--src, --weights and --out are needed to play")
    from common import eval_seeds
    seeds = eval_seeds(a.games)
    run = {"weights": os.path.abspath(a.weights), "policy": a.policy, "temp": a.temp, "back": a.back,
           "stake": a.stake, "games": a.games, "init_seed": a.init_seed if a.policy == "untrained" else None}
    done = set()
    if os.path.exists(a.out):
        lines = [json.loads(l) for l in open(a.out) if l.strip()]
        if lines and "run" in lines[0] and lines[0]["run"] != {**run, "games": lines[0]["run"]["games"]}:
            raise SystemExit(f"{a.out} holds another run: {lines[0]['run']}")
        done = {r["seed"] for r in lines if "seed" in r and not r.get("error")}
    else:
        with open(a.out, "w") as f:
            f.write(json.dumps({"run": run}) + "\n")
    todo = [s for s in seeds if s not in done]
    print(f"{a.weights} ({a.policy}), {a.back} stake {a.stake}, temperature {a.temp}: {len(seeds)} eval seeds "
          f"({len(done)} done), {a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=init, initargs=(a,)) as pool, open(a.out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(one_seed, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec.get("error"):
                print(f"seed {rec['seed']}: error: {rec['error'].strip().splitlines()[-1]}", flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M:%S')}  {i} of {len(todo)}  "
                      f"{i / max(time.time() - t0, 1e-9) * 3600:,.0f} games/h", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
