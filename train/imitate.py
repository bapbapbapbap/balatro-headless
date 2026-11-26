"""Model v6, stage 1: imitation of the teacher's recorded games (docs/hypotheses-v6.md, 2026-11-18 section).

The data is reproduce/balatro_lab/teach.py's output folder: one games/<seed>.pkl.gz per game, a list of decisions
{obs, member, a, labeled, kind, desc, progress}. obs is the student's own observation (as sim.observe gives it),
a the index of the move that was made (the teacher's when labeled, a fallback when not), progress the game's
counters after the move.

What this module does, without torch where it can (tests/v6_imitate_check.py):
  games(dir)              the finished, error-free games teach.py logged (games_w*.jsonl) that have a .pkl.gz
  held_out(seed)          the fixed 5% of games never trained on (sha256 of the seed)
  returns(decisions, a)   each decision's discounted return to the game's end under the run's reward
  agreement_set(...)      2,000 labeled held-out decisions, fixed by seed and move number
  train_imitation(...)    the passes, keeping the weights with the lowest held-out cross-entropy
  score(model, ...)       cross-entropy and top-1 agreement with the teacher on a set of decisions
"""
import collections
import glob
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rewards import REWARDS, build_shaping, shaping, step_discount  # noqa: E402

HOLDOUT_MOD = 20            # 1 game in 20 held out
AGREEMENT_N = 2000


def held_out(seed):
    return int(hashlib.sha256(f"v6-holdout:{seed}".encode()).hexdigest(), 16) % HOLDOUT_MOD == 0


def where(spec):
    """{seed: folder} over every teacher-data folder in spec: comma-separated, each a folder or a glob of folders
    (stage 2: "{work}/teach_v6,{work}/dagger/d*"). A seed in two folders is taken from the first."""
    out = {}
    for part in [p for p in spec.split(",") if p]:
        for d in sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else [part]:
            for s in games(d):
                out.setdefault(s, d)
    return out


def games(d):
    """Seeds of the games teach.py finished without an error and saved, in a fixed order."""
    seeds = set()
    for f in sorted(os.listdir(d)):
        if f.startswith("games_w") and f.endswith(".jsonl"):
            for line in open(os.path.join(d, f)):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                          # a file still being written can end mid-line
                if not r.get("error") and os.path.exists(os.path.join(d, "games", f"{r['seed']}.pkl.gz")):
                    seeds.add(r["seed"])
    return sorted(seeds)


def load(d, seed):
    """One game's decisions; d is its folder, or a where() index."""
    if isinstance(d, dict):
        d = d[seed]
    with gzip.open(os.path.join(d, "games", f"{seed}.pkl.gz"), "rb") as f:
        return pickle.load(f)


def returns(decisions, args):
    """Discounted return from each decision to the game's end, under the run's reward and shaping, with the
    run's per-step discounts (step_discount: the blind clock with --gamma-blind). The lambda = 1 form of the
    routed returns: summed over every later step, whichever network acted. Rewards are rewards.py's, from
    consecutive progress snapshots. teach.py saves the progress after each move only, so the first move's
    "before" is its after-state with blinds, bosses and phi at 0."""
    reward_fn = REWARDS[args.reward]
    first = dict(decisions[0]["progress"])
    first.update(blinds_beaten=0, bosses_beaten=0, won=False, phi=0.0, blinds_passed=0)
    prev, rs, gs = first, [], []
    for d in decisions:
        cur = d["progress"]
        g, _ = step_discount(prev, cur, args)
        rs.append(reward_fn(prev, cur) + args.shaping * shaping(prev, cur, g)
                  + getattr(args, "build_shaping", 0) * build_shaping(prev, cur, g))
        gs.append(g)
        prev = cur
    out, acc = np.zeros(len(rs), np.float32), 0.0
    for t in reversed(range(len(rs))):
        acc = rs[t] + gs[t] * acc
        out[t] = acc
    return out


def agreement_set(d, seeds, n=AGREEMENT_N):
    """n labeled decisions from the held-out games, drawn with a fixed generator: [(seed, move), ...]."""
    pool = []
    for s in seeds:
        if held_out(s):
            pool += [(s, i) for i, x in enumerate(load(d, s)) if x["labeled"]]
    pool.sort()
    rng = random.Random("v6-agreement")
    return sorted(rng.sample(pool, min(n, len(pool))))


def decisions_of(d, refs):
    """The decisions a list of (seed, move) names, with obs['member'] set (collate reads it)."""
    by, out = {}, []
    for s, i in refs:
        by.setdefault(s, []).append(i)
    for s in sorted(by):
        g = load(d, s)
        for i in by[s]:
            x = g[i]
            x["obs"]["member"] = x["member"]
            out.append(x)
    return out


def score(model, decs, device, minibatch=64):
    """Cross-entropy and top-1 agreement with the teacher's move (labeled decisions), per network and overall."""
    import torch
    from common import collate
    model.eval()
    idx = np.argsort([len(x["obs"]["act"]) for x in decs])
    tot = {"ce": 0.0, "agree": 0, "n": 0, "by": {0: [0, 0], 1: [0, 0]}}
    def logits_of(b):
        try:
            return model(*collate([x["obs"] for x in b], device))[0]
        except torch.cuda.OutOfMemoryError:
            if len(b) == 1:
                raise
        torch.cuda.empty_cache()
        h = len(b) // 2
        x, y = logits_of(b[:h]), logits_of(b[h:])
        w = max(x.shape[1], y.shape[1])                  # each half is padded to its own longest action list
        pad = lambda t: torch.nn.functional.pad(t, (0, w - t.shape[1]), value=float("-inf"))  # noqa: E731
        return torch.cat([pad(x), pad(y)])

    with torch.no_grad():
        for i in range(0, len(idx), minibatch):
            b = [decs[j] for j in idx[i:i + minibatch] if decs[j]["labeled"]]
            if not b:
                continue
            logits = logits_of(b)
            a = torch.tensor([x["a"] for x in b], device=device)
            tot["ce"] += torch.nn.functional.cross_entropy(logits, a, reduction="sum").item()
            hit = (logits.argmax(-1) == a).cpu().numpy()
            tot["agree"] += int(hit.sum())
            tot["n"] += len(b)
            for x, h in zip(b, hit):
                tot["by"][x["member"]][0] += int(h)
                tot["by"][x["member"]][1] += 1
    n = max(tot["n"], 1)
    return {"ce": tot["ce"] / n, "agree": tot["agree"] / n, "n": tot["n"],
            "agree_table": tot["by"][0][0] / max(tot["by"][0][1], 1),
            "agree_shop": tot["by"][1][0] / max(tot["by"][1][1], 1)}


def train_imitation(model, args, device, out, log, stop=lambda: None):
    """Stage 1's passes over the teacher's games. Returns a summary; leaves the kept weights in the model and
    in out/imitate.pt (weights only; train.py's save() writes the full checkpoint after)."""
    import torch
    from common import collate
    d = where(args.imitate)
    seeds = sorted(d)
    train_s = [s for s in seeds if not held_out(s)]
    hold_s = [s for s in seeds if held_out(s)]
    if args.imitate_max_games:
        train_s, hold_s = train_s[:args.imitate_max_games], hold_s[:max(2, args.imitate_max_games // 10)]
    assert train_s and hold_s, f"imitation: {len(seeds)} usable games in {args.imitate}; need training and held-out games"
    ref_path = os.path.join(out, "agreement_set.json")
    if os.path.exists(ref_path):
        refs = [tuple(r) for r in json.load(open(ref_path))]
    else:
        refs = agreement_set(d, hold_s, args.imitate_agreement)
        json.dump(refs, open(ref_path, "w"))
    agree_decs = decisions_of(d, refs)
    hold_decs = []
    for s in hold_s:
        for x in load(d, s):
            x["obs"]["member"] = x["member"]
            hold_decs.append(x)
    log(f"imitation: data {args.imitate} ({len(set(d.values()))} folders); {len(train_s):,} training games, {len(hold_s):,} held out ({len(hold_decs):,} decisions); "
        f"agreement set {len(agree_decs):,} decisions; up to {args.imitate_epochs} passes, lr {args.imitate_lr}")
    opt = torch.optim.Adam(model.parameters(), lr=args.imitate_lr)
    rng = random.Random(f"imitate-{args.seed}")
    best, since, hist, t0 = None, 0, [], time.time()
    for ep in range(args.imitate_epochs):
        order = list(train_s)
        rng.shuffle(order)
        model.train()
        st = {"ce": 0.0, "acc": 0.0, "v": 0.0, "n": 0, "decisions": 0, "oom_splits": 0, "oom_skipped": 0}
        for c in range(0, len(order), args.imitate_chunk):
            if stop():
                raise KeyboardInterrupt(stop())
            obs, act, lab, ret = [], [], [], []
            for s in order[c:c + args.imitate_chunk]:
                g = load(d, s)
                R = returns(g, args)
                for x, r in zip(g, R):
                    x["obs"]["member"] = x["member"]
                    obs.append(x["obs"]); act.append(x["a"]); lab.append(x["labeled"]); ret.append(r)
            idx = np.argsort([len(o["act"]) for o in obs])
            batches = [idx[i:i + args.minibatch] for i in range(0, len(idx), args.minibatch)]
            rng.shuffle(batches)
            todo = collections.deque(batches)
            while todo:
                b = todo.popleft()
                try:
                    logits, value = model(*collate([obs[i] for i in b], device))
                    a = torch.tensor([act[i] for i in b], device=device)
                    m = torch.tensor([lab[i] for i in b], device=device, dtype=torch.bool)
                    R = torch.tensor([ret[i] for i in b], device=device)
                    ce = (torch.nn.functional.cross_entropy(logits[m], a[m]) if m.any()
                          else torch.zeros((), device=device))
                    v_loss = 0.5 * (value - R).pow(2).mean()
                    loss = ce + args.vf * v_loss
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    with torch.no_grad():
                        st["ce"] += ce.item(); st["v"] += v_loss.item(); st["n"] += 1
                        if m.any():
                            st["acc"] += (logits[m].argmax(-1) == a[m]).float().mean().item()
                except torch.cuda.OutOfMemoryError:
                    # as ppo.py: the largest action lists (up to ~1,100 rows) don't fit 64 at a time on a 6 GB
                    # GPU; free it and do the same decisions as two half minibatches; one that still doesn't fit
                    # is skipped and counted
                    logits = value = loss = ce = v_loss = None
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if len(b) > 1:
                        todo.appendleft(b[len(b) // 2:])
                        todo.appendleft(b[:len(b) // 2])
                        st["oom_splits"] += 1
                    else:
                        st["oom_skipped"] += 1
            st["decisions"] += len(obs)
            done = min(c + args.imitate_chunk, len(order))
            if (c // args.imitate_chunk) % 10 == 0 and st["n"]:
                log(f"[imitate] pass {ep + 1} {done:,}/{len(order):,} games  move loss {st['ce'] / st['n']:.3f}  "
                    f"same move as teacher {st['acc'] / st['n']:.1%}  value loss {st['v'] / st['n']:.2f}  "
                    f"{(time.time() - t0) / 60:.1f} min", health={"imitate_pass": ep + 1, "imitate_games": done})
        h = score(model, hold_decs, device, args.minibatch)
        ag = score(model, agree_decs, device, args.minibatch)
        rec = {"pass": ep + 1, "train_ce": st["ce"] / max(st["n"], 1), "train_agree": st["acc"] / max(st["n"], 1),
               "value_loss": st["v"] / max(st["n"], 1), "heldout_ce": h["ce"], "heldout_agree": h["agree"],
               "agreement": ag["agree"], "agreement_table": ag["agree_table"], "agreement_shop": ag["agree_shop"],
               "oom_splits": st["oom_splits"], "oom_skipped": st["oom_skipped"],
               "minutes": round((time.time() - t0) / 60, 1)}
        hist.append(rec)
        kept = best is None or h["ce"] < best["heldout_ce"]
        log(f"[imitate] pass {ep + 1} done: held-out move loss {h['ce']:.3f}, agreement {ag['agree']:.1%} "
            f"(table {ag['agree_table']:.1%}, shop {ag['agree_shop']:.1%})" + ("  kept" if kept else ""),
            metric={"imitate": rec})
        if kept:
            best, since = rec, 0
            tmp = os.path.join(out, ".imitate.pt.tmp")
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
            os.replace(tmp, os.path.join(out, "imitate.pt"))
        else:
            since += 1
            if since >= args.imitate_patience:
                break
    model.load_state_dict(torch.load(os.path.join(out, "imitate.pt"), map_location=device, weights_only=True))
    model.eval()
    return {"kept": best, "passes": hist, "train_games": len(train_s), "held_out_games": len(hold_s),
            "agreement_refs": ref_path}
