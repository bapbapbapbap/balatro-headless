"""Routed credit for model v4's phase split (docs/hypotheses.md, 2026-11-09 amendment, change 4).

Two networks act in one game: member 0 (table) and member 1 (shop). Each network's advantages run over
its own decisions only. The reward between two of a network's decisions is everything earned in
between, including what the other network earned (so a blind's reward reaches the shop network that
built the board, as arXiv 2610.39049 routes it), discounted step by step by the same per-step discounts
the single-network trainer uses (the blind clock):

    R    = sum_{j=t}^{t'-1} (prod_{i=t}^{j-1} gamma_i) r_j       t' = this network's next decision
    Gam  = prod_{j=t}^{t'-1} gamma_j,   Lam = prod_{j=t}^{t'-1} lam_j
    adv_t = (R + Gam V(t') - V(t)) + Gam Lam adv_{t'}

With one network acting everywhere every gap is one step, and this is exactly train.gae.

Shaping needs nothing extra: the per-step terms gamma_j phi(s_{j+1}) - phi(s_j), summed with these
discounts between two own decisions, telescope to Gam phi(next own decision) - phi(this one).

numpy only, so it is tested without torch (tests/v4_credit_check.py).
"""
import numpy as np


def _per_step(x, n):
    return [x] * n if np.isscalar(x) else list(x)


def routed_gae(rewards, values, gamma, lam, owner):
    """rewards: per decision of the whole game. values: the acting network's value at each decision;
    one more value past the last decision bootstraps a capped game (as in train.gae, absent = 0).
    gamma, lam: one number or one per step (step t's applies to the move from t to t+1).
    owner: which network made each decision. Returns the advantage of each decision."""
    n = len(rewards)
    G, L = _per_step(gamma, n), _per_step(lam, n)
    tail = values[n] if len(values) > n else 0.0
    adv = np.zeros(n, np.float32)
    for m in sorted(set(owner)):
        idx = [t for t in range(n) if owner[t] == m]
        nxt = 0.0                                          # adv at this network's next decision
        for k in reversed(range(len(idx))):
            t = idx[k]
            end = idx[k + 1] if k + 1 < len(idx) else n
            R, gam, lam_ = 0.0, 1.0, 1.0
            for j in range(t, end):
                R += gam * rewards[j]
                gam *= G[j]
                lam_ *= L[j]
            if k + 1 < len(idx):
                nv, carry = values[end], nxt
            else:                                          # its last decision: to the game's end
                nv, carry = tail, 0.0
            nxt = (R + gam * nv - values[t]) + gam * lam_ * carry
            adv[t] = nxt
    return adv


def routed_returns(adv, values):
    """Critic targets, as train.ppo_update forms them: advantage + the value that was predicted."""
    return [a + v for a, v in zip(adv, values)]
