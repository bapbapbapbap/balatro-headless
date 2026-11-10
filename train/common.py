"""Shared by the trainer, its workers and the tools: seeds, batching (collate), action codes, game setup."""

import hashlib
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import F, ZERO_BY_WIDTH  # noqa: E402


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
    W = obs[0]["act"].shape[1]
    kind = np.zeros((B, T), np.int64)
    vocab = np.zeros((B, T), np.int64)
    feat = np.zeros((B, T, F), np.float32)
    tmask = np.zeros((B, T), bool)
    act = np.full((B, A, W), -1, np.int64)
    for f in ZERO_BY_WIDTH[W]:
        act[..., f] = 0
    amask = np.zeros((B, A), bool)
    for i, o in enumerate(obs):
        n, m = len(o["kind"]), len(o["act"])
        kind[i, :n], vocab[i, :n], feat[i, :n], tmask[i, :n] = o["kind"], o["vocab"], o["feat"], True
        act[i, :m], amask[i, :m] = o["act"], o["allow"] if "allow" in o else True
    t = lambda x: torch.from_numpy(x).to(device)
    out = (t(kind), t(vocab), t(feat), t(tmask), t(act), t(amask))
    if "member" in obs[0]:        # mixture of experts: which member made each decision (model.MoE)
        out += (t(np.array([o["member"] for o in obs], np.int64)),)
    return out


# ---------------- action codes and game setup ----------------

PLAY, DISCARD = 1, 2          # action type codes (harness/encode.lua ACTION)
HAND_COL = 8                  # action row column: vocab id of "hand:<type>" for plays


def game_config(seed, args):
    """(deck, stake, challenge) for a game, derived from its seed, so a retried or replayed seed
    and every eval seed always get the same one. With one deck, one stake and no challenges this is
    (--back, --stake, None), as before."""

    backs, stakes = args.backs or [args.back], args.stakes or [args.stake]
    if not args.challenges and len(backs) == 1 and len(stakes) == 1:
        return backs[0], stakes[0], None
    h = hashlib.sha256(f"config-{seed}".encode()).digest()
    if args.challenges and h[0] / 256 < args.challenge_share:
        return "b_challenge", 1, args.challenges[int.from_bytes(h[1:5], "big") % len(args.challenges)]
    return (backs[int.from_bytes(h[5:9], "big") % len(backs)],
            stakes[int.from_bytes(h[9:13], "big") % len(stakes)], None)


def same_config(r, args):
    return (r["back"], r["stake"], r.get("challenge")) == game_config(r["seed"], args)
