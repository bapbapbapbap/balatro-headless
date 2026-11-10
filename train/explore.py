"""Hand-type exploration (--explore-hands, v2b). Known fault: a move whose float32 probability is 0 is stored at
log(1e-30), so PPO gets no gradient from it (docs/hypotheses-v4.md, 2026-11-10 ~10:30). Off since v4b."""

import math

import numpy as np


def explore_sample(p, act, eps, rng, allow=None):
    """Training games only. With probability eps at a decision with plays of two or more poker hand
    types on offer, picks one of those hand types uniformly, then a play of that type in proportion
    to the policy; otherwise samples the policy.

    Why: v2 never plays a Full House when one is on offer (0 of 374 in its wins). It was cloned from
    v1, which never played one, so the policy gives that move almost no probability and RL never
    samples it to learn what it pays. The hand types are the labels the game shows on the play; no
    hand type is preferred, and the reward alone decides what the bot keeps.

    Returns the move and its log-probability under the policy itself, not under the mixture, so an
    explored move that turns out well gets a full-strength PPO update, as in epsilon-greedy. With
    the mixture's probability the update would be scaled by policy/mixture (about 1/100 for a move
    the policy almost never makes): the same expected push as never exploring. The cost is a small
    bias, since eps of the decisions are not the policy's own."""
    play = act[:, 0] == 1
    if allow is not None:
        play = play & allow                  # a hand pin (apply_pin) is never explored around
    ht = act[:, 8]
    types = np.unique(ht[play])
    if len(types) >= 2 and rng.random() < eps:
        t = types[min(int(rng.random() * len(types)), len(types) - 1)]
        idx = np.flatnonzero(play & (ht == t))
        w = p[idx] if p[idx].sum() > 0 else np.ones(len(idx))
        a = int(idx[min(int(np.searchsorted(np.cumsum(w), rng.random() * w.sum())), len(idx) - 1)])
    else:
        a = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), len(p) - 1)
    return a, float(math.log(max(p[a], 1e-30)))
