"""Model v4's backward curriculum: training games that start at the first shop of ante 4, 5 or 6 of a saved game."""

import json
import os
import random
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import eval_seeds, rand_seed  # noqa: E402


# ---------------- curriculum (model v4; docs/hypotheses.md, 2026-11-09 amendment, change 1) ----------------
# A share of training games start at the first shop of ante 4, 5 or 6 of a saved game, reached by
# replaying that game's moves from its seed (Sim.fast_forward). Starts come from published wins (whole
# routes, replayed until that shop) and from this run's own games (the exact moves up to it).

CURRICULUM_ANTES = (4, 5, 6)


def curriculum_entry(src, rec, ante, acts=None, desc=None):
    """A start: a published win (src 'published', replayed until the shop) or one of this run's games
    (src 'own', its moves up to that shop)."""

    e = {"from": src, "seed": rec["seed"], "back": rec.get("back", "b_red"), "stake": rec.get("stake", 1),
         "challenge": rec.get("challenge"), "ante": ante,
         "acts": list(acts if acts is not None else rec["acts"]), "desc": list(desc if desc is not None else rec["desc"])}
    if src == "published":
        e["until_ante"] = ante
    return e


def published_starts(args, n, banned):
    """n published wins (wins/routes.jsonl) on this run's action list, Red Deck White Stake, no challenge,
    none on a held-out seed, chosen by a fixed RNG: one start per ante each."""
    path = os.path.join(HERE, "..", "wins", "routes.jsonl")
    rs = []
    for line in open(path):
        r = json.loads(line)
        if (r.get("actions") == args.actions and r.get("back", "b_red") == "b_red" and r.get("stake", 1) == 1
                and not r.get("challenge") and r["seed"] not in banned):
            rs.append(r)
    rs.sort(key=lambda r: r["seed"])
    rs = random.Random("curriculum-published").sample(rs, min(n, len(rs)))
    return {a: [curriculum_entry("published", r, a) for r in rs] for a in CURRICULUM_ANTES}


def held_out_seeds():
    """Seeds no curriculum start may come from: the eval 600 and every registered paired-test tag."""
    sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
    from compare_models import TAGS, seed_list
    banned = set(eval_seeds(600))
    for t in TAGS:
        banned |= set(seed_list(t, 2000, rand_seed, set(eval_seeds(600))))
    return banned
