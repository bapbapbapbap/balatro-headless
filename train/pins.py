"""Hand pins (strategy experts, docs/hypotheses-moe.md section I) and their bookkeeping, including the MoE's pins."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DISCARD, HAND_COL, PLAY  # noqa: E402


# ---------------- hand pin (strategy experts, docs/hypotheses.md section I) ----------------
# An expert is pinned to one poker hand type. At a decision where a play of that type is on offer,
# the pin binds with probability --pin-rate: the bot may then make only plays of that type.
# Discards and every non-play move stay allowed, and when no play of the type is on offer nothing
# is restricted. The pin goes into the observation as o["allow"]; collate() turns it into the
# action mask, so the policy that plays and the policy PPO updates are the same masked
# distribution. The game's legal list is untouched, so saved routes replay exactly as before.

def pin_vocab_id(vocab, hand):
    if not hand:
        return None
    key = "hand:" + hand
    assert key in vocab, f"--pin-hand {hand!r}: no such poker hand in this game's vocabulary"
    return vocab.index(key)


def new_pin_stats():
    # offered: decisions with a play of the pinned type on offer; bound: of those, where the pin
    # applied; plays / plays_pinned: every play, and plays of the pinned type; discard_offered:
    # discards made while a play of the type was on offer (dodging the hand, see section I review).
    return {"offered": 0, "bound": 0, "plays": 0, "plays_pinned": 0, "discard_offered": 0}


def _pinned(hand_col, pin_id):
    """pin_id is one hand's vocab id, or a tuple of them (the early pin: a hand or any higher one)."""

    return np.isin(hand_col, pin_id) if isinstance(pin_id, tuple) else hand_col == pin_id


def apply_pin(o, pin_id, rate, rng, st):
    act = o["act"]
    play = act[:, 0] == PLAY
    target = play & _pinned(act[:, HAND_COL], pin_id)
    if not target.any():
        return
    st["offered"] += 1
    o["offered"] = True
    if rate >= 1.0 or rng.random() < rate:
        o["allow"] = ~play | target
        st["bound"] += 1


# The game's poker hands, lowest to highest (its own G.handlist, reversed).
HAND_RANK = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House",
             "Four of a Kind", "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"]


def early_pin(vocab, args):
    """--moe-early-pin "Hand=rate": (vocab ids of that hand and every higher one, rate), or ((), 0).
    At an early-model decision where a play of one of them is on offer, the pin binds with that rate
    (times the fade): only those plays are allowed; discards and other moves stay allowed."""
    if not getattr(args, "moe_early_pin", None):
        return (), 0.0
    hand, rate = args.moe_early_pin.rsplit("=", 1)
    assert hand in HAND_RANK, f"--moe-early-pin {hand!r}: no such poker hand"
    return tuple(pin_vocab_id(vocab, h) for h in HAND_RANK[HAND_RANK.index(hand):]), float(rate)


def moe_min_plays(hand, args):
    """Plays of a hand in antes 1-3 needed before the router may pick its expert (--moe-min-plays)."""
    for item in getattr(args, "moe_min_plays", []) or []:
        h, n = item.rsplit("=", 1)
        if h == hand:
            return int(n)
    return 0


def moe_pin_rate(hand, args):
    """An expert's pin rate before annealing: --moe-pin "Hand=rate" entries, else 1.0."""
    for item in args.moe_pin:
        h, r = item.rsplit("=", 1)
        if h == hand:
            return float(r)
    return 1.0


def count_pin_move(o, a, pin_id, st):
    row = o["act"][a]
    if row[0] == PLAY:
        st["plays"] += 1
        st["plays_pinned"] += int(bool(_pinned(row[HAND_COL], pin_id)))
    elif row[0] == DISCARD and o.get("offered"):
        st["discard_offered"] += 1


def pin_summary(episodes):
    pins = [e["pin"] for e in episodes if e.get("pin")]
    out = {"pin": {k: sum(p[k] for p in pins) for k in pins[0]}} if pins else {}
    moe = [e["moe"] for e in episodes if e.get("moe")]
    if moe:
        by = {}
        for e in episodes:
            m = e.get("moe")
            if m and m["expert"]:
                b = by.setdefault(m["expert"], {"games": 0, "rule": 0, "random": 0, "wins": 0, "near_wins": 0, "blinds": 0, **new_pin_stats()})
                b["games"] += 1; b[m["routed"]] += 1; b["wins"] += int(e["progress"]["won"])
                b["near_wins"] += int(bool(e["progress"].get("near_win")))
                b["blinds"] += e["progress"]["blinds_beaten"]
                for k, v in (e.get("pin") or {}).items():
                    b[k] += v
        out["moe"] = {"handoffs": sum(1 for m in moe if m["expert"]), "by_expert": by}
        ep_pins = [m["early_pin"] for m in moe if m.get("early_pin")]
        if ep_pins:
            out["moe"]["early_pin"] = {k: sum(p[k] for p in ep_pins) for k in ep_pins[0]}
    return out
