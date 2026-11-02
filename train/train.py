"""PPO trainer for model v1 (see docs/model_v1.md).

  python3 train.py --src ~/balatro/local/source --workers 14 --run v1 --hours 8

W worker processes each own a sim and a copy of the model. Every iteration they play games with
the current weights; the learner runs PPO and sends the new weights back. Checkpoints, config and
metrics go to runs/<run>/. Nothing here prefers one outcome over another except --reward.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import queue
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

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

def inject_at(args, kind):
    """Iterations where a fault is injected (testing the safety checks), e.g. --inject nan:3,crash:5"""
    out = set()
    for part in (args.inject or "").split(","):
        if part.startswith(kind + ":"):
            out.add(int(part.split(":")[1]))
    return out


def worker_main(wid, args, in_q, out_q):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    try:
        from sim import Sim
        sim = Sim(args.src, args.mods)
    except Exception:
        out_q.put(("error", wid, traceback.format_exc()))
        return
    model = Net(len(sim.vocab), d=args.d, layers=args.layers, heads=args.heads)
    model.eval()
    reward_fn = REWARDS[args.reward]
    rng = random.Random(f"{args.seed}-{wid}")
    out_q.put(("ready", wid, sim.vocab))
    crash_at = inject_at(args, "crash")
    while True:
        msg = in_q.get()
        if msg is None:
            return
        try:
            out_q.put(("done", wid, play_games(sim, model, reward_fn, rng, args, wid, msg, crash_at)))
        except Exception:
            out_q.put(("error", wid, traceback.format_exc()))


def play_games(sim, model, reward_fn, rng, args, wid, msg, crash_at):
        # mode: "train" (sampled, steps kept for PPO), "eval" (sampled with a fixed RNG per seed, so
        # scores are comparable between rounds), "greedy" (most likely action; can loop forever
        # on free actions like dragging jokers, so it is a diagnostic only).
        state, seeds, mode, it, gen = msg
        if mode == "replay":
            return replay_games(sim, reward_fn, args, seeds)
        greedy = mode == "greedy"
        if it in crash_at and gen == 0 and wid == 0 and mode == "train":
            raise RuntimeError(f"injected worker crash at iteration {it}")
        model.load_state_dict(state)
        trng = random.Random(f"{args.seed}-{wid}-{it}")      # reproducible sampling per worker+iteration
        episodes = []
        for seed in seeds:
            seed = seed or rand_seed(rng)
            srng = random.Random(f"eval-{seed}") if mode == "eval" else trng
            o = sim.reset(seed, args.back, args.stake)
            steps, done, tail = [], False, []
            while not done and sim.steps < args.max_steps:
                with torch.no_grad():
                    logits, v = model(*collate([o]))
                p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
                if not np.isfinite(p).all():
                    raise FloatingPointError("non-finite action probabilities")
                if greedy:
                    a = int(p.argmax())
                else:
                    a = int(np.searchsorted(np.cumsum(p), srng.random() * p.sum()))
                    a = min(a, len(p) - 1)
                logp = float(math.log(max(p[a], 1e-30)))
                if mode != "train" and sim.steps >= args.max_steps - 30:
                    tail.append((sim.H.state_name(), int(o["act"][a][0]), int(o["act"][a][1]),
                                 int(o["act"][a][2]), round(float(p[a]), 4), len(p)))
                prev, cur, done = sim.step(a)
                r = reward_fn(prev, cur) + args.shaping * shaping(prev, cur, args.gamma)
                steps.append((o, a, logp, float(v[0]), r, reward_fn(prev, cur)))
                if not done:
                    o = sim.observe()
            ep = {"seed": seed, "steps": steps if mode == "train" else None, "n": sim.steps,
                  "progress": sim.progress, "capped": not done, "tail": tail if not done else None,
                  "return": sum(s[5] for s in steps),
                  # the exact game: seed + every action (index into the legal list, and as text)
                  "acts": list(sim.route), "desc": list(sim.desc), "final": sim.final_hash(), "verified": None}
            if ep["progress"]["won"]:
                ep["verified"] = verify_route(sim, args, ep)
            episodes.append(ep)
        return episodes


def verify_route(sim, args, ep):
    """Plays a won game again from its saved seed and actions; True only if every step is the same
    move and the game ends in the same state. (The game is deterministic, so this must hold.)"""
    prog, n, final, problems = sim.replay(ep["seed"], args.back, args.stake, ep["acts"], ep["desc"])
    return not problems and n == ep["n"] and final == ep["final"] and prog == ep["progress"]


def replay_games(sim, reward_fn, args, routes):
    """Self-imitation: replays saved games into training data (observation, action, reward per step).
    Saved routes are exact, so this regenerates the very states the bot saw when it played them."""
    out = []
    for r in routes:
        o = sim.reset(r["seed"], args.back, args.stake)
        steps, done, bad = [], False, None
        for k, a in enumerate(r["acts"]):
            if done or a >= len(o["act"]) or sim.H.action_desc(a + 1) != r["desc"][k]:
                bad = f"step {k} differs from the saved game"
                break
            prev, cur, done = sim.step(a)
            steps.append((o, a, reward_fn(prev, cur) + args.shaping * shaping(prev, cur, args.gamma)))
            if not done:
                o = sim.observe()
        if bad is None and sim.final_hash() != r["final"]:
            bad = "final state differs from the saved game"
        out.append({"key": r["key"], "steps": None if bad else steps, "problem": bad})
    return out


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


def discounted(rewards, gamma):
    out, g = np.zeros(len(rewards), np.float32), 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def ppo_update(model, opt, episodes, args, device, sil=None):
    """sil: optional (obs, actions, returns) from replayed saved games. Self-imitation (Oh et al.
    2018): each step pulls the policy toward the saved action by how much the game's actual return
    beat the critic's estimate, max(R - V, 0), and the critic toward R where it underestimated.
    Steps the critic already expected contribute nothing, so this reuses only what the bot
    undervalued. The return is the same reward (+ shaping) the bot is trained on: no new goal."""
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
    adv_scale = float(adv.std() + 1e-8)        # SIL advantages on the same scale as PPO's
    if sil is not None:
        stats.update(sil_pi=0.0, sil_v=0.0, sil_pos=0.0, sil_adv=0.0)
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
            if sil is not None:
                sobs, sact, sret = sil
                sidx = np.random.randint(0, len(sobs), size=min(args.minibatch, len(sobs)))
                sk = collate([sobs[i] for i in sidx], device)
                slog, sval = model(*sk)
                slp = torch.log_softmax(slog, -1).gather(1, torch.tensor(sact[sidx], device=device).unsqueeze(1)).squeeze(1)
                sadv = (torch.tensor(sret[sidx], device=device) - sval).clamp(min=0)
                sil_pi = -(slp * (sadv / adv_scale).detach()).mean()
                sil_v = 0.5 * sadv.pow(2).mean()
                loss = loss + args.sil_weight * (sil_pi + args.sil_value * sil_v)
                with torch.no_grad():
                    stats["sil_pi"] += sil_pi.item(); stats["sil_v"] += sil_v.item()
                    stats["sil_pos"] += (sadv > 0).float().mean().item(); stats["sil_adv"] += sadv.mean().item()
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


# ---------------- safety ----------------

class Stop(Exception):
    """Ends training cleanly: weights are saved and the reason recorded."""


class WorkerFailure(Exception):
    pass


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("inf")


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30


class Pool:
    """The worker processes. Restartable: a crashed or hung worker gets the whole pool restarted."""

    def __init__(self, args):
        self.args, self.ctx, self.gen, self.procs = args, mp.get_context("spawn"), -1, []

    def start(self, timeout=600):
        self.gen += 1
        W = self.args.workers
        self.in_qs = [self.ctx.Queue() for _ in range(W)]
        self.out_q = self.ctx.Queue()
        self.procs = [self.ctx.Process(target=worker_main, args=(w, self.args, self.in_qs[w], self.out_q), daemon=True)
                      for w in range(W)]
        for p in self.procs:
            p.start()
        vocab, deadline = None, time.time() + timeout
        for _ in range(W):
            try:
                tag, wid, v = self.out_q.get(timeout=max(1, deadline - time.time()))
            except queue.Empty:
                raise WorkerFailure("workers did not start in time")
            if tag == "error":
                raise WorkerFailure(f"worker {wid} failed to start:\n{v}")
            vocab = vocab or v
            if v != vocab:
                raise WorkerFailure("workers disagree on vocabulary")
        self.vocab = vocab

    def stop(self):
        for q in getattr(self, "in_qs", []):
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.procs = []

    def play(self, state, seeds_per_worker, mode, it, timeout, should_stop=lambda: None):
        for w in range(self.args.workers):
            self.in_qs[w].put((state, seeds_per_worker[w], mode, it, self.gen))
        eps, errors, deadline = [], [], time.time() + timeout
        for _ in range(self.args.workers):
            while True:                       # wait in 1 s slices so a stop request is seen at once
                why = should_stop()
                if why:
                    raise Stop(why)
                try:
                    tag, wid, e = self.out_q.get(timeout=1)
                    break
                except queue.Empty:
                    if time.time() > deadline:
                        dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
                        raise WorkerFailure(f"round timed out after {timeout}s (dead workers: {dead or 'none, hung'})")
            if tag == "error":
                errors.append(f"worker {wid}:\n{e}")
            else:
                eps.extend(e)
        if errors:
            raise WorkerFailure("\n".join(errors))
        return eps


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
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many training games in total (0 = no limit)")
    ap.add_argument("--max-wins", type=int, default=0,
                    help="stop after this many won training games in total, i.e. runs that beat the ante 8 boss (0 = no limit)")
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
    # safety
    ap.add_argument("--max-kl", type=float, default=0.2, help="reject an update that moves the policy further than this")
    ap.add_argument("--max-strikes", type=int, default=3, help="consecutive rejected updates before stopping")
    ap.add_argument("--round-timeout", type=float, default=1200, help="seconds before a round counts as hung")
    ap.add_argument("--min-mem-gb", type=float, default=1.0, help="stop cleanly below this much free memory")
    ap.add_argument("--worker-mem-gb", type=float, default=0.45, help="memory budget per worker, for sizing")
    ap.add_argument("--min-disk-gb", type=float, default=2.0)
    ap.add_argument("--stall-hours", type=float, default=3.0,
                    help="stop if the 10-round average training score hasn't improved in this long")
    ap.add_argument("--inject", help="test the safety checks: nan:<it>,crash:<it>")
    # saved games and self-imitation
    ap.add_argument("--save-games", choices=["all", "wins", "none"], default="all",
                    help="write every game exactly (seed + actions) to <out>/games/; wins also go to wins.jsonl")
    ap.add_argument("--sil-buffer", type=int, default=64,
                    help="self-imitation: keep the N highest-return training games (0 = off)")
    ap.add_argument("--sil-games", type=int, default=8, help="saved games replayed into each update")
    ap.add_argument("--sil-weight", type=float, default=1.0)
    ap.add_argument("--sil-value", type=float, default=0.01, help="weight of the critic's self-imitation loss")
    # second drive
    ap.add_argument("--mirror-dir", help="also append every win to <dir>/wins.jsonl (e.g. an external drive)")
    ap.add_argument("--spill-dir", help="when the main drive runs low, write saved games and checkpoints here")
    ap.add_argument("--spill-at-gb", type=float, default=50.0, help="free space on the main drive that triggers spilling")
    # seed retries: the solver side. Lost seeds are kept by the latest ante they reached (1-8) and
    # replayed with fresh sampling. Each round's games are split across 9 arms (fresh seeds, and
    # retries of seeds whose best was ante 1..8) by Thompson sampling on wins per game, so effort
    # goes where it actually turns into beaten seeds.
    ap.add_argument("--retry", type=int, default=1, help="1 = split games across the arms, 0 = fresh seeds only")
    ap.add_argument("--fresh-min", type=float, default=0.5, help="share of each round always on fresh seeds")
    ap.add_argument("--arm-prior", type=float, default=100.0,
                    help="prior losses per arm, Beta(1, prior): how many games before an arm's record dominates")
    ap.add_argument("--retry-max", type=int, default=30, help="attempts per seed before it is dropped")
    ap.add_argument("--retry-pool", type=int, default=500, help="most seeds waiting per ante")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.out or os.path.join(HERE, "runs", args.run)
    os.makedirs(out, exist_ok=True)
    log_alerts = []

    def alert(msg):
        line = f"ALERT {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        log_alerts.append(line)
        del log_alerts[:-30]
        print(line, flush=True)

    # Size the pool to the memory actually free (resource allocation).
    avail = mem_available_gb()
    fit = int(max(1, (avail - args.min_mem_gb - 1.0) // args.worker_mem_gb))
    if fit < args.workers:
        alert(f"{avail:.1f} GB free fits {fit} workers, not {args.workers}; using {fit}")
        args.workers = fit
    if disk_free_gb(out) < args.min_disk_gb:
        raise SystemExit(f"only {disk_free_gb(out):.1f} GB free on disk; need {args.min_disk_gb}")

    stop_requested = []
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append("interrupted (Ctrl+C)"))
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append("terminated"))

    pool = Pool(args)
    pool.start()
    vocab = pool.vocab

    model = Net(len(vocab), d=args.d, layers=args.layers, heads=args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_it, resumed = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        assert ck["vocab"] == vocab, "checkpoint vocabulary differs from this game"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_it = ck["iteration"] + 1
        resumed = ck
    prov = provenance(args, vocab)
    json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=2)
    print(f"model {param_count(model):,} params on {device}; {args.workers} workers; reward '{args.reward}' "
          f"(shaping {args.shaping}); {args.back} stake {args.stake}; {'modded' if args.mods else 'unmodded'}; "
          f"{avail:.1f} GB free; run dir {out}", flush=True)
    metrics = open(os.path.join(out, "metrics.jsonl"), "a")
    esd = eval_seeds(args.eval_games)
    nan_at = inject_at(args, "nan")

    def weights():
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def run_round(seeds, mode, it):
        for attempt in (1, 2):
            try:
                return pool.play(weights(), seeds, mode, it, args.round_timeout,
                                 lambda: stop_requested[0] if stop_requested else None)
            except WorkerFailure as e:
                alert(f"workers failed in round {it} (attempt {attempt}): {str(e)[-1500:]}")
                pool.stop()
                if attempt == 2:
                    raise Stop("workers failed twice in a row")
                try:
                    pool.start()
                except WorkerFailure as e2:
                    raise Stop(f"workers could not restart: {e2}")

    def save(it, tag):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it, "vocab": vocab,
              "provenance": prov, "arch": {"d": args.d, "layers": args.layers, "heads": args.heads},
              "best_eval": best_eval, "games_total": games_total, "wins_total": wins_total,
              "lr": opt.param_groups[0]["lr"], "sil_buffer": sil_buffer, "retry_pools": pools, "arms": arms,
              "beaten_seeds": sorted(beaten)}
        d = out
        if tag.startswith("ckpt_") and spilling():
            d = os.path.join(args.spill_dir, "runs", args.run)
            os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{tag}.pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(d, f"{tag}.pt"))      # never leaves a half-written file

    health = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    drive_note = {}

    def usable(d):
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".write_test")
            open(t, "w").write("ok"); os.remove(t)
            return True
        except OSError:
            return False

    def spilling():
        """True when the main drive is low and the spill drive is writable."""
        if not args.spill_dir or disk_free_gb(out) >= args.spill_at_gb:
            return False
        ok = usable(args.spill_dir)
        if drive_note.get("spill") != ok:
            drive_note["spill"] = ok
            alert(f"main drive under {args.spill_at_gb} GB free; " + (f"writing games and checkpoints to {args.spill_dir}"
                  if ok else f"spill drive {args.spill_dir} not writable, staying on the main drive"))
        return ok

    def mirror_win(line):
        if not args.mirror_dir:
            return
        try:
            os.makedirs(args.mirror_dir, exist_ok=True)
            with open(os.path.join(args.mirror_dir, "wins.jsonl"), "a") as f:
                f.write(line)
            if drive_note.get("mirror") is False:
                alert(f"mirror {args.mirror_dir} writable again")
            drive_note["mirror"] = True
        except OSError as ex:
            if drive_note.get("mirror") is not False:
                alert(f"couldn't copy a win to {args.mirror_dir} ({ex}); the main copy is saved")
            drive_note["mirror"] = False

    def write_health(**kw):
        health.update(kw, alerts=log_alerts[-10:], mem_available_gb=round(mem_available_gb(), 2),
                      disk_free_gb=round(disk_free_gb(out), 1), lr=opt.param_groups[0]["lr"],
                      updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = os.path.join(out, ".health.json.tmp")
        json.dump(health, open(tmp, "w"), indent=2)
        os.replace(tmp, os.path.join(out, "health.json"))

    t_start = time.time()
    games_total, wins_total, strikes, it = 0, 0, 0, start_it
    best_eval, best_time = -1.0, time.time()
    train_hist, best_roll = [], -1.0
    if resumed:
        best_eval = resumed.get("best_eval", -1.0)
        games_total = resumed.get("games_total", 0)
        wins_total = resumed.get("wins_total", 0)
        alert(f"resumed from {args.resume} at round {start_it} (best eval {best_eval:.2f} blinds/game, "
              f"{games_total:,} games and {wins_total:,} wins so far)")
    sil_buffer = list(resumed.get("sil_buffer", [])) if resumed else []
    sil_rng = random.Random(f"sil-{args.seed}-{start_it}")
    games_dir = os.path.join(out, "games")
    if args.save_games != "none":
        os.makedirs(games_dir, exist_ok=True)
        with open(os.path.join(games_dir, "meta.jsonl"), "a") as f:   # what every saved game was played on
            f.write(json.dumps({"from_iteration": start_it, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "back": args.back, "stake": args.stake, "mode": "subsets",
                                "profile": "unlocked", "fast": True, "provenance": prov}) + "\n")
    saved = {"games": 0, "wins": 0, "wins_verified": 0}
    if args.mirror_dir:
        alert(f"wins also copied to {args.mirror_dir}" if usable(args.mirror_dir)
              else f"mirror {args.mirror_dir} not writable now; wins stay on the main drive until it is")
    # retry pools: ante -> {seed: {"attempts", "best_blinds", "first_it"}}; arms: wins and games per arm
    ARMS = ["fresh"] + [str(a) for a in range(1, 9)]
    pools = {a: dict(v) for a, v in (resumed.get("retry_pools", {}) if resumed else {}).items()}
    for a in ARMS[1:]:
        pools.setdefault(a, {})
    arms = dict(resumed.get("arms", {})) if resumed else {}
    for a in ARMS:
        arms.setdefault(a, {"games": 0, "wins": 0})
    beaten = set(resumed.get("beaten_seeds", [])) if resumed else set()
    arm_rng = random.Random(f"arms-{args.seed}-{start_it}")

    def train_seeds():
        """Splits this round's games across the arms. Fresh keeps at least --fresh-min; every other
        slot goes to the arm with the highest draw from Beta(1 + wins, prior + losses), among arms
        with seeds waiting. Within an arm, seeds with the fewest attempts go first."""
        n = args.workers * args.games_per_worker
        if not args.retry:
            return [[None] * args.games_per_worker for _ in range(args.workers)], {}
        floor = int(math.ceil(n * args.fresh_min))
        left = {a: len(pools[a]) for a in ARMS[1:]}
        alloc = {a: 0 for a in ARMS}
        alloc["fresh"] = floor
        for _ in range(n - floor):
            best, bv = "fresh", -1.0
            for a in ARMS:
                if a != "fresh" and alloc[a] >= left[a]:
                    continue
                st = arms[a]
                v = arm_rng.betavariate(1 + st["wins"], args.arm_prior + st["games"] - st["wins"])
                if v > bv:
                    best, bv = a, v
            alloc[best] += 1
        source = {}
        flat = []
        for a in ARMS[1:]:
            order = sorted(pools[a].items(), key=lambda kv: (kv[1]["attempts"], -kv[1]["best_blinds"]))
            for sd, _ in order[:alloc[a]]:
                flat.append(sd)
                source[sd] = a
        flat += [None] * (n - len(flat))
        return [flat[w::args.workers] for w in range(args.workers)], source

    def update_retries(it, eps, source):
        for e in eps:
            sd, p = e["seed"], e["progress"]
            arm = source.get(sd, "fresh")
            e["arm"] = arm
            arms[arm]["games"] += 1
            arms[arm]["wins"] += int(p["won"])
            if arm != "fresh":
                pools[arm].pop(sd, None)
            if p["won"]:
                beaten.add(sd)
                continue
            if sd in beaten:
                continue
            prev = None if arm == "fresh" else e.get("_prev_entry")
            attempts = (prev or {}).get("attempts", 0) + (0 if arm == "fresh" else 1)
            if arm != "fresh" and attempts >= args.retry_max:
                continue
            a = str(max(int(arm) if arm != "fresh" else 0, min(p["ante"], 8)))   # best ante reached so far
            pools[a][sd] = {"attempts": attempts, "best_blinds": max(p["blinds_beaten"], (prev or {}).get("best_blinds", 0)),
                            "first_it": (prev or {}).get("first_it", it)}
            if len(pools[a]) > args.retry_pool:     # full: drop the most-tried, then the oldest
                worst = max(pools[a].items(), key=lambda kv: (kv[1]["attempts"], -kv[1]["first_it"]))[0]
                del pools[a][worst]

    def record(it, mode, eps):
        """Writes each game exactly. Wins also go to wins.jsonl, flagged if they didn't replay."""
        if args.save_games == "none":
            return
        keep = [e for e in eps if args.save_games == "all" or e["progress"]["won"]]
        recs = [{"iteration": it, "mode": mode, "seed": e["seed"], "back": args.back, "stake": args.stake,
                 "n": e["n"], "capped": e["capped"], "return": e["return"], "progress": e["progress"],
                 "verified": e["verified"], "final": e["final"], "acts": e["acts"], "desc": e["desc"],
                 "retry": e.get("retry", False), "arm": e.get("arm")} for e in keep]
        if recs:
            gd = os.path.join(args.spill_dir, "runs", args.run, "games") if spilling() else games_dir
            os.makedirs(gd, exist_ok=True)
            path = os.path.join(gd, f"r{it:05d}_{mode}.jsonl.gz")
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, path)
            saved["games"] += len(recs)
        for r in recs:
            if r["progress"]["won"]:
                line = json.dumps(r | {"commit": prov["commit"][:12]}) + "\n"
                with open(os.path.join(out, "wins.jsonl"), "a") as f:
                    f.write(line)
                mirror_win(line)
                saved["wins"] += 1
                if r["verified"]:
                    saved["wins_verified"] += 1
                else:
                    alert(f"won game (seed {r['seed']}, round {it}) did NOT replay exactly; saved, flagged unverified")

    def add_to_buffer(it, eps):
        """Keeps the highest-return training games (return under the reward of record), as routes."""
        if args.sil_buffer <= 0:
            return
        for k, e in enumerate(eps):
            if e["capped"]:
                continue
            sil_buffer.append({"key": f"{it}-{k}", "seed": e["seed"], "acts": e["acts"], "desc": e["desc"],
                               "final": e["final"], "return": e["return"], "won": e["progress"]["won"], "it": it})
        sil_buffer.sort(key=lambda r: (r["return"], r["it"]), reverse=True)
        del sil_buffer[args.sil_buffer:]

    def sil_data(it):
        if args.sil_buffer <= 0 or not sil_buffer:
            return None, {}
        pick = [sil_buffer[sil_rng.randrange(len(sil_buffer))] for _ in range(args.sil_games)]
        split = [pick[w::args.workers] for w in range(args.workers)]
        got = run_round(split, "replay", it)
        obs, act, ret, bad = [], [], [], set()
        for g in got:
            if g["problem"]:
                bad.add(g["key"])
                alert(f"saved game {g['key']} no longer replays ({g['problem']}); dropped from the buffer")
                continue
            for (o, a, _), R in zip(g["steps"], discounted([x[2] for x in g["steps"]], args.gamma)):
                obs.append(o); act.append(a); ret.append(R)
        if bad:
            sil_buffer[:] = [r for r in sil_buffer if r["key"] not in bad]
        if not obs:
            return None, {}
        info = {"buffer": len(sil_buffer), "buffer_wins": sum(r["won"] for r in sil_buffer),
                "buffer_min_return": min(r["return"] for r in sil_buffer),
                "replayed_games": len(got) - len(bad), "replayed_steps": len(obs)}
        return (obs, np.array(act), np.array(ret, np.float32)), info

    rates = []                     # decisions per second of play, for the throttling flag
    reason = "finished"
    try:
        for it in range(start_it, args.iterations):
            if stop_requested:
                raise Stop(stop_requested[0])
            hours = (time.time() - t_start) / 3600
            if hours > args.hours:
                raise Stop(f"time limit ({args.hours} h)")
            if args.max_games and games_total >= args.max_games:
                raise Stop(f"reached {games_total:,} training games (limit {args.max_games:,})")
            if args.max_wins and wins_total >= args.max_wins:
                raise Stop(f"reached {wins_total:,} won games (limit {args.max_wins:,})")
            if mem_available_gb() < args.min_mem_gb:
                raise Stop(f"free memory below {args.min_mem_gb} GB ({mem_available_gb():.2f} GB)")
            if disk_free_gb(out) < args.min_disk_gb:
                raise Stop(f"free disk below {args.min_disk_gb} GB")

            if it % args.eval_every == 0:
                te = time.time()
                split = [esd[w::args.workers] for w in range(args.workers)]
                eval_eps = run_round(split, "eval", it)
                ev = summarize(eval_eps)
                record(it, "eval", eval_eps)
                beaten.update(e["seed"] for e in eval_eps if e["progress"]["won"])
                print(f"[eval {it}] {json.dumps(ev)}", flush=True)
                metrics.write(json.dumps({"iteration": it, "eval": ev, "eval_seconds": round(time.time() - te, 1)}) + "\n")
                metrics.flush()
                save(it, f"ckpt_{it:05d}")
                if ev["blinds_mean"] > best_eval:
                    best_eval = ev["blinds_mean"]
                    save(it, "best")
                capped = [e for e in eval_eps if e["capped"]][:20]
                if capped:
                    with open(os.path.join(out, "eval_capped.jsonl"), "a") as f:
                        for e in capped:
                            f.write(json.dumps({"iteration": it, "seed": e["seed"], "progress": e["progress"],
                                                "last_moves": e["tail"]}) + "\n")
                health["last_eval"] = ev
                health["best_eval_blinds"] = best_eval

            t0 = time.time()
            seeds, source = train_seeds()
            prev_entries = {sd: dict(pools[a][sd]) for sd, a in source.items()}
            eps = run_round(seeds, "train", it)
            t1 = time.time()
            for e in eps:
                e["retry"] = e["seed"] in source
                e["_prev_entry"] = prev_entries.get(e["seed"])
            update_retries(it, eps, source)
            record(it, "train", eps)
            sil, sil_info = sil_data(it)
            add_to_buffer(it, eps)       # after sampling: this round's games are already in the PPO batch

            # Update guard: keep a copy, reject the update if it is non-finite or moves too far.
            snap_m = {k: v.detach().clone() for k, v in model.state_dict().items()}
            snap_o = copy.deepcopy(opt.state_dict())
            st = ppo_update(model, opt, eps, args, device, sil)
            if it in nan_at:
                with torch.no_grad():
                    next(model.parameters()).fill_(float("nan"))
            finite = all(torch.isfinite(p).all() for p in model.parameters()) and \
                all(math.isfinite(v) for v in st.values())
            if not finite or abs(st["kl"]) > args.max_kl:
                model.load_state_dict(snap_m)
                opt.load_state_dict(snap_o)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                strikes += 1
                why = "non-finite loss or weights" if not finite else f"kl {st['kl']:.3f} > {args.max_kl}"
                alert(f"update {it} rejected ({why}); "
                      f"reverted, learning rate now {opt.param_groups[0]['lr']:.2e} (strike {strikes}/{args.max_strikes})")
                if strikes >= args.max_strikes:
                    raise Stop(f"{strikes} rejected updates in a row")
            else:
                strikes = 0
            t2 = time.time()

            games_total += len(eps)
            fresh = [e for e in eps if not e["retry"]]
            s = summarize(fresh or eps)       # the learning curve stays on fresh seeds only
            sr = summarize([e for e in eps if e["retry"]]) if source else None
            round_alloc = {a: sum(1 for e in eps if e["arm"] == a) for a in ARMS}
            wins_total += sum(e["progress"]["won"] for e in eps)
            train_hist.append(s["blinds_mean"])
            roll = float(np.mean(train_hist[-10:]))
            if len(train_hist) >= 10 and roll > best_roll + 0.05:
                best_roll, best_time = roll, time.time()
            elif len(train_hist) >= 10 and (time.time() - best_time) / 3600 > args.stall_hours:
                raise Stop(f"training score flat for {args.stall_hours} h (best 10-round average "
                           f"{best_roll:.2f} blinds/game)")
            decisions = sum(e["n"] for e in eps)
            rate = decisions / max(t1 - t0, 1e-6)
            rates.append((time.time() - t_start, rate))
            early = [r for t, r in rates if t < 1800]
            if (time.time() - t_start) > 3600 and early and rate < 0.6 * sorted(early)[len(early) // 2]:
                alert(f"play speed {rate:.0f} decisions/s is under 60% of the first half hour's "
                      f"({sorted(early)[len(early) // 2]:.0f}); likely heat throttling")
            if st["entropy"] < 0.05:
                alert(f"policy entropy {st['entropy']:.3f}: the policy has become nearly deterministic")
            rec = {"iteration": it, "train": s, "ppo": st, "wins_total": wins_total, "play_seconds": round(t1 - t0, 1),
                   "update_seconds": round(t2 - t1, 1), "games_per_hour": round(len(eps) / (t2 - t0) * 3600),
                   "decisions_per_second": round(rate, 1), "games_total": games_total,
                   "hours": round((time.time() - t_start) / 3600, 3), "sil": sil_info, "saved": dict(saved),
                   "retry": sr, "alloc": round_alloc, "arms": {a: dict(v) for a, v in arms.items()},
                   "pools": {a: len(v) for a, v in pools.items()}, "beaten_seeds": len(beaten)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(f"[it {it}] blinds {s['blinds_mean']:.2f} wins {s['wins']}/{s['games']} (total {wins_total:,}) ante {s['final_ante']} "
                  f"ent {st['entropy']:.2f} kl {st['kl']:.4f} | {rec['games_per_hour']} games/h "
                  f"{rate:.0f} dec/s (play {rec['play_seconds']}s, update {rec['update_seconds']}s)"
                  + (f" | sil buffer {sil_info['buffer']} ({sil_info['buffer_wins']} wins, min return {sil_info['buffer_min_return']:.0f}) "
                     f"pos {st.get('sil_pos', 0):.2f}" if sil_info else "")
                  + (f" | retry {sr['games']} games ante {sr['final_ante']} wins {sr['wins']}" if sr else "")
                  + " | alloc " + " ".join(f"{a[0] if a == 'fresh' else a}:{round_alloc[a]}" for a in ARMS if round_alloc[a])
                  + f" | waiting {sum(len(v) for v in pools.values())} beaten seeds {len(beaten)}"
                  + f" | saved {saved['games']:,} games, {saved['wins_verified']}/{saved['wins']} wins verified", flush=True)
            save(it, "latest")
            write_health(iteration=it, hours=rec["hours"], games_total=games_total, wins_total=wins_total, last_train=s,
                         decisions_per_second=rec["decisions_per_second"], saved=dict(saved), sil=sil_info,
                         last_retry=sr, alloc=round_alloc, arms={a: dict(v) for a, v in arms.items()},
                         pools={a: len(v) for a, v in pools.items()}, beaten_seeds=len(beaten))
    except Stop as e:
        reason = str(e)
    except Exception:
        reason = "crashed: " + traceback.format_exc()[-1500:]
    finally:
        try:
            save(it, "final")
        except Exception as e:
            reason += f" (final save failed: {e})"
        alert(f"stopped: {reason}")
        write_health(status=f"stopped: {reason}", iteration=it)
        pool.stop()
        print("done", flush=True)


if __name__ == "__main__":
    main()
