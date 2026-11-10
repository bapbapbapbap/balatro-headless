"""Advantages (GAE, routed per network for v4) and the guarded PPO update."""

import collections
import math
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import collate  # noqa: E402
from credit import routed_gae  # noqa: E402
from model import PHASE_NAMES, PHASE_SHOP, PHASE_TABLE  # noqa: E402


# ---------------- PPO ----------------

def per_step(x, n):
    """A discount given as one number or one per step, as a list of n numbers."""

    return [x] * n if np.isscalar(x) else list(x)


def gae(rewards, values, gamma, lam):
    """gamma and lam: one number, or one per step (step t's discount applies to the move from t to
    t + 1; see rewards.step_discount)."""
    n = len(rewards)
    gamma, lam = per_step(gamma, n), per_step(lam, n)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nv = values[t + 1] if t + 1 < len(values) else 0.0   # terminal or capped: no bootstrap
        delta = rewards[t] + gamma[t] * nv - values[t]
        last = delta + gamma[t] * lam[t] * last
        adv[t] = last
    return adv


def discounted(rewards, gamma):
    n = len(rewards)
    gamma = per_step(gamma, n)
    out, g = np.zeros(n, np.float32), 0.0
    for t in reversed(range(n)):
        g = rewards[t] + gamma[t] * g
        out[t] = g
    return out


def taken_logp(model, obs, acts, idx, device):
    """Log-probs of the taken actions for steps idx (no grad), in halves if they don't fit the GPU."""
    try:
        lg, _ = model(*collate([obs[j] for j in idx], device))
        return torch.log_softmax(lg, -1).gather(
            1, torch.tensor([acts[j] for j in idx], device=device).unsqueeze(1)).squeeze(1).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if len(idx) == 1:
            raise
    lg = None
    torch.cuda.empty_cache()
    h = len(idx) // 2
    return np.concatenate([taken_logp(model, obs, acts, idx[:h], device), taken_logp(model, obs, acts, idx[h:], device)])


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
        d = ep.get("disc")
        G = [x[0] for x in d] if d else args.gamma
        L = [x[1] for x in d] if d else args.lam
        k = (ep.get("moe") or {}).get("handoff_step") if getattr(args, "moe_cut", 0) else None
        if getattr(args, "phase_split", False):
            # v4: each network's advantages over its own decisions, with everything earned in between
            # (train/credit.py; docs/hypotheses.md 2026-11-09, change 4)
            lh = getattr(args, "lam_hand", 1.0)
            if lh < 1.0 and d and len(d[0]) > 2:
                # v4b: the table network's lambda also decays by --lam-hand at each play or discard (a hand
                # clock); the shop network's stays on the blind clock (docs/hypotheses-v4b.md)
                L = {PHASE_TABLE: [x[1] * (lh if x[2] else 1.0) for x in d], PHASE_SHOP: [x[1] for x in d]}
            a = routed_gae(r, v, G, L, [s[0]["member"] for s in ep["steps"]])
        elif k:
            # The early head's run ends at the handoff: its returns count only up to ante 4 (chud's
            # spec), and the expert's start there.
            Gs, Ls = per_step(G, len(r)), per_step(L, len(r))
            a = np.concatenate([gae(r[:k], v[:k], Gs[:k], Ls[:k]), gae(r[k:], v[k:], Gs[k:], Ls[k:])])
        else:
            a = gae(r, v, G, L)
        for s, ai, vi in zip(ep["steps"], a, v):
            if getattr(args, "moe", False) and not args.moe_train_early and s[0].get("member") == 0:
                continue          # the early-game model is frozen in joint training (--moe-train-early 0)
            obs.append(s[0]); acts.append(s[1]); oldlp.append(s[2]); adv.append(ai); ret.append(ai + vi)
    if not obs:       # MoE with a frozen early model and no game reaching the handoff this round
        return {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 1.0, "kl": 0.0, "clipfrac": 0.0, "early_stop": 0.0,
                "kl_end": 0.0, "kl_update": 0.0, "decisions": 0, "minibatches": 0}
    adv = np.array(adv, np.float32)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    split = getattr(args, "phase_split", False)
    if split:
        # v4: each network's advantages normalized on their own (two critics, two scales)
        mem = np.array([o["member"] for o in obs])
        for m in range(len(PHASE_NAMES)):
            sel = mem == m
            if sel.any():
                adv_n[sel] = (adv[sel] - adv[sel].mean()) / (adv[sel].std() + 1e-8)
        ent_coef = torch.tensor(phase_ent(args), dtype=torch.float32, device=device)
    ret = np.array(ret, np.float32)
    n = len(obs)
    order_by_size = np.argsort([len(o["act"]) for o in obs])   # batch similar sizes: less padding
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
    if split:
        stats.update({f"entropy_{nm}": 0.0 for nm in PHASE_NAMES})
        ent_n = {nm: 0 for nm in PHASE_NAMES}
    adv_scale = float(adv.std() + 1e-8)        # SIL advantages on the same scale as PPO's
    if sil is not None:
        stats.update(sil_pi=0.0, sil_v=0.0, sil_pos=0.0, sil_adv=0.0)
    stats.update(early_stop=0.0, kl_end=0.0)
    # Self-imitation runs on only --sil-steps of the update's minibatches, spread evenly: one step
    # per minibatch (~225 a round) sharpened the policy (entropy 2.07 -> 1.48 in 3 rounds).
    planned = args.epochs * max(1, math.ceil(n / args.minibatch))
    sil_every = max(1, planned // max(1, args.sil_steps))
    # Log-probs of the taken actions under the weights this update starts from. PPO's ratio uses the
    # policy that played (one update older with --overlap); early stopping measures how far this
    # update itself has moved, so it compares against these.
    start_lp = np.zeros(n, np.float32)
    if args.target_kl:
        model.eval()
        with torch.no_grad():
            for i in range(0, n, args.minibatch):   # training's batch size: big action lists run out of GPU memory at 256
                idx = order_by_size[i:i + args.minibatch]
                start_lp[idx] = taken_logp(model, obs, acts, idx, device)
    nb, stop, kl_seen = 0, False, []
    model.train()
    for _ in range(args.epochs):
        if stop:
            break
        chunks = [order_by_size[i:i + args.minibatch] for i in range(0, n, args.minibatch)]
        random.shuffle(chunks)
        todo = collections.deque(chunks)
        while todo:
            idx = todo.popleft()
            oom = False
            try:
                batch = collate([obs[i] for i in idx], device)
                amask = batch[5]
                logits, value = model(*batch)
                logp_all = torch.log_softmax(logits, -1)
                a = torch.tensor([acts[i] for i in idx], device=device)
                lp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
                olp = torch.tensor([oldlp[i] for i in idx], device=device)
                A = torch.tensor(adv_n[idx], device=device)
                R = torch.tensor(ret[idx], device=device)
                ratio = torch.exp(lp - olp)
                # Early stop (as in common PPO implementations): once this round's updates have moved the
                # policy past 1.5x the KL target on fresh data, stop here instead of overshooting.
                with torch.no_grad():
                    r0 = torch.exp(lp - torch.tensor(start_lp[idx], device=device))
                    k3 = ((r0 - 1) - torch.log(r0)).mean().item()          # low-variance KL estimate, vs. update start
                kl_seen.append(k3)
                recent = float(np.mean(kl_seen[-4:]))      # one 64-step minibatch alone is too noisy to stop on
                stats["kl_end"] = recent
                if args.target_kl and nb >= 4 and recent > 1.5 * args.target_kl:
                    stats["early_stop"] = 1.0
                    stop = True
                    break
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                v_loss = 0.5 * (value - R).pow(2).mean()
                p_all = torch.softmax(logits, -1)
                ent_each = -(p_all * logp_all.masked_fill(~amask, 0)).sum(-1)
                ent = ent_each.mean()
                if split:
                    # per-network entropy bonus (v4: the shop's on its schedule, --shop-ent)
                    loss = pi_loss + args.vf * v_loss - (ent_coef[batch[6]] * ent_each).mean()
                    with torch.no_grad():
                        for m, nm in enumerate(PHASE_NAMES):
                            sel = batch[6] == m
                            if sel.any():
                                stats[f"entropy_{nm}"] += ent_each[sel].mean().item()
                                ent_n[nm] += 1
                else:
                    loss = pi_loss + args.vf * v_loss - args.ent * ent
                if sil is not None and nb % sil_every == 0:
                    sobs, sact, sret = sil
                    sidx = np.random.randint(0, len(sobs), size=min(args.minibatch, len(sobs)))
                    sk = collate([sobs[i] for i in sidx], device)
                    slog, sval = model(*sk)
                    slp = torch.log_softmax(slog, -1).gather(1, torch.tensor(sact[sidx], device=device).unsqueeze(1)).squeeze(1)
                    sadv = (torch.tensor(sret[sidx], device=device) - sval).clamp(min=0)
                    # capped: one high-return game can't pull a move harder than --sil-clip PPO std devs
                    w = (sadv / adv_scale).clamp(max=args.sil_clip).detach()
                    sil_pi = -(slp * w).mean()
                    sil_v = torch.nn.functional.smooth_l1_loss(sval, torch.max(torch.tensor(sret[sidx], device=device), sval).detach())
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
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                # A minibatch of the largest action lists didn't fit under the GPU cap (moe2, round 154:
                # 1.09 GiB more needed with 1.63 GiB reserved but fragmented). Free it and do the same
                # steps as two half minibatches; a single step that still doesn't fit is skipped.
                batch = logits = value = logp_all = p_all = loss = sk = slog = sval = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if len(idx) > 1:
                    h = len(idx) // 2
                    todo.appendleft(idx[h:])
                    todo.appendleft(idx[:h])
                    stats["oom_splits"] = stats.get("oom_splits", 0) + 1
                else:
                    stats["oom_skipped"] = stats.get("oom_skipped", 0) + 1
    model.eval()
    # How far this update moved the policy, on up to 2048 of its steps (what the reject-guard checks).
    kl_update = 0.0
    if args.target_kl:
        sample = order_by_size[np.linspace(0, n - 1, min(n, 2048)).astype(int)]
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(sample), args.minibatch):
                idx = sample[i:i + args.minibatch]
                lp1 = torch.tensor(taken_logp(model, obs, acts, idx, device), device=device)
                r1 = torch.exp(lp1 - torch.tensor(start_lp[idx], device=device))
                tot += ((r1 - 1) - torch.log(r1)).sum().item()
        kl_update = tot / len(sample)
    stats["kl_update"] = kl_update
    once = {"early_stop", "kl_end", "kl_update", "oom_splits", "oom_skipped"}
    if split:
        for m, nm in enumerate(PHASE_NAMES):
            stats[f"entropy_{nm}"] = stats[f"entropy_{nm}"] / max(ent_n[nm], 1)
            stats[f"decisions_{nm}"] = float((mem == m).sum())
            stats[f"ent_coef_{nm}"] = float(ent_coef[m])
            once |= {f"entropy_{nm}", f"decisions_{nm}", f"ent_coef_{nm}"}
    return {k: (v if k in once else v / max(nb, 1)) for k, v in stats.items()} | {"decisions": n, "minibatches": nb}


def parse_schedule(spec):
    """--shop-ent "0.02,0.006@1.5M,0.001@4M": a start value, then values reached at counts of training
    decisions (k/M suffixes allowed), linear in between (a repeated count gives a step), constant after the
    last. -> [(decisions, value), ...]"""
    parts = [x.strip() for x in str(spec).split(",") if x.strip()]
    knots = [(0, float(parts[0]))]
    for x in parts[1:]:
        v, g = x.split("@")
        mul = {"k": 1e3, "M": 1e6}.get(g[-1], 1)
        knots.append((int(float(g.rstrip("kM")) * mul), float(v)))
    assert all(knots[i][0] <= knots[i + 1][0] for i in range(len(knots) - 1)), f"--shop-ent counts must not decrease: {spec}"
    return knots


def schedule_at(knots, games):
    for (g0, v0), (g1, v1) in zip(knots, knots[1:]):
        if games < g1:
            return v0 + (v1 - v0) * (games - g0) / (g1 - g0) if g1 > g0 else v1
    return knots[-1][1]


def phase_ent(args):
    """Entropy coefficients (table, shop) for this update (v4). args.decisions_done (training decisions so far,
    both networks) is set by the learner."""
    shop = schedule_at(parse_schedule(args.shop_ent), getattr(args, "decisions_done", 0)) if getattr(args, "shop_ent", None) else args.ent
    return [args.ent, shop]
