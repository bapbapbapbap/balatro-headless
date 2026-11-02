"""PPO trainer for model v1 (see docs/model_v1.md).

  python3 train.py --src ~/balatro/local/source --workers 14 --run v1 --hours 8

W worker processes each own a sim and a copy of the model. Every iteration they play games with
the current weights; the learner runs PPO and sends the new weights back. Checkpoints, config and
metrics go to runs/<run>/. Nothing here prefers one outcome over another except --reward.
"""
import argparse
import copy
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
        state, seeds, greedy, it, gen = msg
        if it in crash_at and gen == 0 and wid == 0 and not greedy:
            raise RuntimeError(f"injected worker crash at iteration {it}")
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
                if not np.isfinite(p).all():
                    raise FloatingPointError("non-finite action probabilities")
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
        return episodes


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

    def play(self, state, seeds_per_worker, greedy, it, timeout, should_stop=lambda: None):
        for w in range(self.args.workers):
            self.in_qs[w].put((state, seeds_per_worker[w], greedy, it, self.gen))
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
    ap.add_argument("--stall-hours", type=float, default=3.0, help="stop if the eval score hasn't improved in this long")
    ap.add_argument("--inject", help="test the safety checks: nan:<it>,crash:<it>")
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

    def run_round(seeds, greedy, it):
        for attempt in (1, 2):
            try:
                return pool.play(weights(), seeds, greedy, it, args.round_timeout,
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
              "best_eval": best_eval, "games_total": games_total, "lr": opt.param_groups[0]["lr"]}
        tmp = os.path.join(out, f".{tag}.pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(out, f"{tag}.pt"))      # never leaves a half-written file

    health = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def write_health(**kw):
        health.update(kw, alerts=log_alerts[-10:], mem_available_gb=round(mem_available_gb(), 2),
                      disk_free_gb=round(disk_free_gb(out), 1), lr=opt.param_groups[0]["lr"],
                      updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = os.path.join(out, ".health.json.tmp")
        json.dump(health, open(tmp, "w"), indent=2)
        os.replace(tmp, os.path.join(out, "health.json"))

    t_start = time.time()
    games_total, strikes, it = 0, 0, start_it
    best_eval, best_time = -1.0, time.time()
    if resumed:
        best_eval = resumed.get("best_eval", -1.0)
        games_total = resumed.get("games_total", 0)
        alert(f"resumed from {args.resume} at round {start_it} (best eval {best_eval:.2f} blinds/game, "
              f"{games_total} games so far)")
    rates = []                     # decisions per second of play, for the throttling flag
    reason = "finished"
    try:
        for it in range(start_it, args.iterations):
            if stop_requested:
                raise Stop(stop_requested[0])
            hours = (time.time() - t_start) / 3600
            if hours > args.hours:
                raise Stop(f"time limit ({args.hours} h)")
            if mem_available_gb() < args.min_mem_gb:
                raise Stop(f"free memory below {args.min_mem_gb} GB ({mem_available_gb():.2f} GB)")
            if disk_free_gb(out) < args.min_disk_gb:
                raise Stop(f"free disk below {args.min_disk_gb} GB")

            if it % args.eval_every == 0:
                te = time.time()
                split = [esd[w::args.workers] for w in range(args.workers)]
                ev = summarize(run_round(split, True, it))
                print(f"[eval {it}] {json.dumps(ev)}", flush=True)
                metrics.write(json.dumps({"iteration": it, "eval": ev, "eval_seconds": round(time.time() - te, 1)}) + "\n")
                metrics.flush()
                save(it, f"ckpt_{it:05d}")
                if ev["blinds_mean"] > best_eval:
                    best_eval, best_time = ev["blinds_mean"], time.time()
                    save(it, "best")
                elif (time.time() - best_time) / 3600 > args.stall_hours:
                    raise Stop(f"no eval improvement in {args.stall_hours} h (best {best_eval:.2f} blinds/game)")
                health["last_eval"] = ev
                health["best_eval_blinds"] = best_eval

            t0 = time.time()
            eps = run_round([[None] * args.games_per_worker for _ in range(args.workers)], False, it)
            t1 = time.time()

            # Update guard: keep a copy, reject the update if it is non-finite or moves too far.
            snap_m = {k: v.detach().clone() for k, v in model.state_dict().items()}
            snap_o = copy.deepcopy(opt.state_dict())
            st = ppo_update(model, opt, eps, args, device)
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
            s = summarize(eps)
            decisions = sum(e["n"] for e in eps)
            rate = decisions / max(t1 - t0, 1e-6)
            rates.append((time.time() - t_start, rate))
            early = [r for t, r in rates if t < 1800]
            if (time.time() - t_start) > 3600 and early and rate < 0.6 * sorted(early)[len(early) // 2]:
                alert(f"play speed {rate:.0f} decisions/s is under 60% of the first half hour's "
                      f"({sorted(early)[len(early) // 2]:.0f}); likely heat throttling")
            if st["entropy"] < 0.05:
                alert(f"policy entropy {st['entropy']:.3f}: the policy has become nearly deterministic")
            rec = {"iteration": it, "train": s, "ppo": st, "play_seconds": round(t1 - t0, 1),
                   "update_seconds": round(t2 - t1, 1), "games_per_hour": round(len(eps) / (t2 - t0) * 3600),
                   "decisions_per_second": round(rate, 1), "games_total": games_total,
                   "hours": round((time.time() - t_start) / 3600, 3)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(f"[it {it}] blinds {s['blinds_mean']:.2f} wins {s['wins']}/{s['games']} ante {s['final_ante']} "
                  f"ent {st['entropy']:.2f} kl {st['kl']:.4f} | {rec['games_per_hour']} games/h "
                  f"{rate:.0f} dec/s (play {rec['play_seconds']}s, update {rec['update_seconds']}s)", flush=True)
            save(it, "latest")
            write_health(iteration=it, hours=rec["hours"], games_total=games_total, last_train=s,
                         decisions_per_second=rec["decisions_per_second"])
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
