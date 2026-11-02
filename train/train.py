"""PPO trainer for model v1 (see docs/model_v1.md).

  python3 train.py --src ~/balatro/local/source --workers 14 --run v1 --hours 8

W worker processes each own a sim and a copy of the model. Every iteration they play games with
the current weights; the learner runs PPO and sends the new weights back. Checkpoints, config and
metrics go to runs/<run>/. Nothing here prefers one outcome over another except --reward.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import ACT, F, Net, param_count  # noqa: E402
from rewards import REWARDS, shaping  # noqa: E402

SEED_CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def rand_seed(rng):
    return "".join(rng.choice(SEED_CHARS) for _ in range(8))


def eval_seeds(n):
    rng = random.Random("balatro-eval-v1")          # fixed, so every run is scored on the same seeds
    return [rand_seed(rng) for _ in range(n)]


# ---------------- batching ----------------

def collate(obs, device="cpu"):
    B = len(obs)
    T = max(len(o["kind"]) for o in obs)
    A = max(len(o["act"]) for o in obs)
    kind = np.zeros((B, T), np.int64)
    vocab = np.zeros((B, T), np.int64)
    feat = np.zeros((B, T, F), np.float32)
    tmask = np.zeros((B, T), bool)
    act = np.full((B, A, ACT), -1, np.int64)
    act[..., 0] = 0
    act[..., 8] = 0
    amask = np.zeros((B, A), bool)
    for i, o in enumerate(obs):
        n, m = len(o["kind"]), len(o["act"])
        kind[i, :n], vocab[i, :n], feat[i, :n], tmask[i, :n] = o["kind"], o["vocab"], o["feat"], True
        act[i, :m], amask[i, :m] = o["act"], True
    t = lambda x: torch.from_numpy(x).to(device)
    return t(kind), t(vocab), t(feat), t(tmask), t(act), t(amask)


# ---------------- workers ----------------

def worker_main(wid, args, in_q, out_q):
    torch.set_num_threads(1)
    from sim import Sim
    sim = Sim(args.src, args.mods)
    model = Net(len(sim.vocab), d=args.d, layers=args.layers, heads=args.heads)
    model.eval()
    reward_fn = REWARDS[args.reward]
    rng = random.Random(f"{args.seed}-{wid}")
    out_q.put(("ready", wid, sim.vocab))
    while True:
        msg = in_q.get()
        if msg is None:
            return
        state, seeds, greedy, it = msg
        model.load_state_dict(state)
        trng = random.Random(f"{args.seed}-{wid}-{it}")      # reproducible sampling per worker+iteration
        episodes = []
        for seed in seeds:
            seed = seed or rand_seed(rng)
            o = sim.reset(seed, args.back, args.stake)
            steps, done = [], False
            while not done and sim.steps < args.max_steps:
                with torch.no_grad():
                    logits, v = model(*collate([o]))
                p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
                if greedy:
                    a = int(p.argmax())
                else:
                    a = int(np.searchsorted(np.cumsum(p), trng.random() * p.sum()))
                    a = min(a, len(p) - 1)
                logp = float(math.log(max(p[a], 1e-30)))
                prev, cur, done = sim.step(a)
                r = reward_fn(prev, cur) + args.shaping * shaping(prev, cur, args.gamma)
                steps.append((o, a, logp, float(v[0]), r, reward_fn(prev, cur)))
                if not done:
                    o = sim.observe()
            episodes.append({"seed": seed, "steps": steps if not greedy else None, "n": sim.steps,
                             "progress": sim.progress, "capped": not done,
                             "return": sum(s[5] for s in steps)})
        out_q.put(("done", wid, episodes))


# ---------------- PPO ----------------

def gae(rewards, values, gamma, lam):
    adv = np.zeros(len(rewards), np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nv = values[t + 1] if t + 1 < len(values) else 0.0   # terminal or capped: no bootstrap
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def ppo_update(model, opt, episodes, args, device):
    obs, acts, oldlp, adv, ret = [], [], [], [], []
    for ep in episodes:
        r = [s[4] for s in ep["steps"]]
        v = [s[3] for s in ep["steps"]]
        a = gae(r, v, args.gamma, args.lam)
        for s, ai, vi in zip(ep["steps"], a, v):
            obs.append(s[0]); acts.append(s[1]); oldlp.append(s[2]); adv.append(ai); ret.append(ai + vi)
    adv = np.array(adv, np.float32)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = np.array(ret, np.float32)
    n = len(obs)
    order_by_size = np.argsort([len(o["act"]) for o in obs])   # batch similar sizes: less padding
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
    nb = 0
    model.train()
    for _ in range(args.epochs):
        chunks = [order_by_size[i:i + args.minibatch] for i in range(0, n, args.minibatch)]
        random.shuffle(chunks)
        for idx in chunks:
            kind, vocab, feat, tmask, act, amask = collate([obs[i] for i in idx], device)
            logits, value = model(kind, vocab, feat, tmask, act, amask)
            logp_all = torch.log_softmax(logits, -1)
            a = torch.tensor([acts[i] for i in idx], device=device)
            lp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            olp = torch.tensor([oldlp[i] for i in idx], device=device)
            A = torch.tensor(adv_n[idx], device=device)
            R = torch.tensor(ret[idx], device=device)
            ratio = torch.exp(lp - olp)
            pi_loss = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
            v_loss = 0.5 * (value - R).pow(2).mean()
            p_all = torch.softmax(logits, -1)
            ent = -(p_all * logp_all.masked_fill(~amask, 0)).sum(-1).mean()
            loss = pi_loss + args.vf * v_loss - args.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item(); stats["entropy"] += ent.item()
                stats["kl"] += (olp - lp).mean().item()
                stats["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
            nb += 1
    model.eval()
    return {k: v / max(nb, 1) for k, v in stats.items()} | {"decisions": n}


# ---------------- bookkeeping ----------------

def find_info(src):
    d = os.path.abspath(src)
    for _ in range(4):
        p = os.path.join(d, "info.txt")
        if os.path.exists(p):
            return open(p, encoding="utf-8", errors="replace").read()
        d = os.path.dirname(d)
    return ""


def provenance(args, vocab):
    info = find_info(args.src)
    exe = next((l.split(":", 1)[1].strip() for l in info.splitlines() if "sha256" in l), "unknown")
    try:
        commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {"config": vars(args), "commit": commit, "balatro_exe_sha256": exe,
            "modded": bool(args.mods), "python": platform.python_version(), "torch": torch.__version__,
            "vocab_sha256": hashlib.sha256("\n".join(vocab).encode()).hexdigest(), "host": platform.node()}


def summarize(episodes):
    n = len(episodes)
    antes = {}
    for e in episodes:
        a = e["progress"]["ante"]
        antes[a] = antes.get(a, 0) + 1
    return {"games": n,
            "wins": sum(e["progress"]["won"] for e in episodes),
            "blinds_mean": float(np.mean([e["progress"]["blinds_beaten"] for e in episodes])),
            "bosses_mean": float(np.mean([e["progress"]["bosses_beaten"] for e in episodes])),
            "return_mean": float(np.mean([e["return"] for e in episodes])),
            "decisions_mean": float(np.mean([e["n"] for e in episodes])),
            "capped": sum(e["capped"] for e in episodes),
            "final_ante": dict(sorted(antes.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="game source (or <modded>/game with --mods)")
    ap.add_argument("--mods", help="Mods dir with Steamodded, for the modded game")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--out", help="output dir (default: train/runs/<run>)")
    ap.add_argument("--reward", choices=sorted(REWARDS), default="blinds")
    ap.add_argument("--shaping", type=float, default=1.0, help="potential-based shaping weight (0 = off)")
    ap.add_argument("--back", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--games-per-worker", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10**9)
    ap.add_argument("--hours", type=float, default=8.0, help="stop after this much wall time")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="checkpoint to continue from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.out or os.path.join(HERE, "runs", args.run)
    os.makedirs(out, exist_ok=True)

    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(args.workers)]
    out_q = ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(w, args, in_qs[w], out_q), daemon=True) for w in range(args.workers)]
    for p in procs:
        p.start()
    vocab = None
    for _ in procs:
        tag, wid, v = out_q.get()
        vocab = vocab or v
        assert v == vocab, "workers disagree on vocabulary"

    model = Net(len(vocab), d=args.d, layers=args.layers, heads=args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        assert ck["vocab"] == vocab, "checkpoint vocabulary differs from this game"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_it = ck["iteration"] + 1
    prov = provenance(args, vocab)
    json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=2)
    print(f"model {param_count(model):,} params on {device}; {args.workers} workers; reward '{args.reward}'; "
          f"{args.back} stake {args.stake}; {'modded' if args.mods else 'unmodded'}; run dir {out}", flush=True)
    metrics = open(os.path.join(out, "metrics.jsonl"), "a")
    esd = eval_seeds(args.eval_games)

    def play(seeds_per_worker, greedy, it):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        for w in range(args.workers):
            in_qs[w].put((state, seeds_per_worker[w], greedy, it))
        eps = []
        for _ in range(args.workers):
            tag, wid, e = out_q.get()
            eps.extend(e)
        return eps

    def save(it, tag):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it, "vocab": vocab,
              "provenance": prov, "arch": {"d": args.d, "layers": args.layers, "heads": args.heads}}
        torch.save(ck, os.path.join(out, f"{tag}.pt"))

    t_start = time.time()
    games_total = 0
    for it in range(start_it, args.iterations):
        if time.time() - t_start > args.hours * 3600:
            break
        if it % args.eval_every == 0:
            te = time.time()
            split = [esd[w::args.workers] for w in range(args.workers)]
            ev = summarize(play(split, True, it))
            rec = {"iteration": it, "eval": ev, "eval_seconds": round(time.time() - te, 1)}
            print(f"[eval {it}] {json.dumps(ev)}", flush=True)
            metrics.write(json.dumps(rec) + "\n"); metrics.flush()
            save(it, f"ckpt_{it:05d}")
        t0 = time.time()
        eps = play([[None] * args.games_per_worker for _ in range(args.workers)], False, it)
        t1 = time.time()
        st = ppo_update(model, opt, eps, args, device)
        t2 = time.time()
        games_total += len(eps)
        s = summarize(eps)
        rec = {"iteration": it, "train": s, "ppo": st, "play_seconds": round(t1 - t0, 1),
               "update_seconds": round(t2 - t1, 1), "games_per_hour": round(len(eps) / (t2 - t0) * 3600),
               "games_total": games_total, "hours": round((time.time() - t_start) / 3600, 3)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()
        print(f"[it {it}] blinds {s['blinds_mean']:.2f} wins {s['wins']}/{s['games']} ante {s['final_ante']} "
              f"ent {st['entropy']:.2f} kl {st['kl']:.4f} | {rec['games_per_hour']} games/h "
              f"(play {rec['play_seconds']}s, update {rec['update_seconds']}s)", flush=True)
        save(it, "latest")
    save(it, "final")
    for q in in_qs:
        q.put(None)
    print("done", flush=True)


if __name__ == "__main__":
    main()
