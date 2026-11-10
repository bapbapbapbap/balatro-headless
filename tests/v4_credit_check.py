"""Checks train/credit.py (model v4's routed credit) without torch or the game.

    python3 tests/v4_credit_check.py        -> prints ALL PASS or the failures

1. One network acting everywhere: identical to ppo.gae (copied here from ppo.py's source, so this
   needs no torch).
2. lam = 1, no bootstrap: each decision's advantage = the discounted return from it, counting every
   reward until the game ends (the other network's included), minus its own value.
3. Routing: a blind's reward reaches the shop decision that preceded it.
4. Shaping telescopes per network: with r = gamma*phi' - phi only and values 0, a network's advantage at
   a decision with lam = 0 is Gam*phi(next own decision) - phi(this one).
5. Blind-clock discounts (1 inside a blind, gamma between blinds) and a capped game's bootstrap.
6. Per-network lambda (model v4b's hand clock): the same lambda for both networks as a dict equals the plain
   list; with the table network's lambda at 0.8 per hand, a death charge reaches a table move k hands before
   it scaled by 0.8^k, a drag shares the credit of the hand after it, and the shop's advantage is unchanged.
"""
import ast
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from credit import routed_gae  # noqa: E402

src = open(os.path.join(HERE, "..", "train", "ppo.py")).read()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("per_step", "gae"):
        exec(compile(ast.Module([node], []), "train.py", "exec"), ns)
gae = ns["gae"]

FAILS = []


def check(name, ok, info=""):
    if not ok:
        FAILS.append(f"{name}: {info}")


rng = random.Random(7)
for trial in range(200):
    n = rng.randint(1, 40)
    r = [rng.uniform(-1, 2) for _ in range(n)]
    v = [rng.uniform(-1, 1) for _ in range(n + rng.randint(0, 1))]
    g = [rng.choice([1.0, 0.95, 0.9]) for _ in range(n)] if trial % 2 else 0.97
    l = [rng.choice([1.0, 0.95]) for _ in range(n)] if trial % 3 else 0.95
    a1, a2 = gae(r, v, g, l), routed_gae(r, v, g, l, [0] * n)
    check("1 one network == gae", np.allclose(a1, a2, atol=1e-5), f"trial {trial}")

    owner = [rng.randint(0, 1) for _ in range(n)]
    v0 = [rng.uniform(-1, 1) for _ in range(n)]
    gg = g if not np.isscalar(g) else [g] * n
    adv = routed_gae(r, v0, gg, 1.0, owner)
    for t in range(n):
        R, d = 0.0, 1.0
        for j in range(t, n):
            R += d * r[j]
            d *= gg[j]
        check("2 lam=1 return", abs(adv[t] - (R - v0[t])) < 1e-4, f"trial {trial} t {t}")

    phi = [rng.uniform(0, 3) for _ in range(n + 1)]
    phi[n] = 0.0                                                  # terminal potential
    rs = [gg[j] * phi[j + 1] - phi[j] for j in range(n)]
    adv = routed_gae(rs, [0.0] * n, gg, 0.0, owner)
    for t in range(n):
        nxt = next((u for u in range(t + 1, n) if owner[u] == owner[t]), n)
        gam = float(np.prod(gg[t:nxt]))
        check("4 shaping telescopes", abs(adv[t] - (gam * phi[nxt] - phi[t])) < 1e-4, f"trial {trial} t {t}")

# 3. A shop decision, then a blind of table decisions whose last one beats the blind (+1).
owner = [1, 1, 0, 0, 0, 1]
r = [0, 0, 0, 0, 1.0, 0]
adv = routed_gae(r, [0.0] * 6, 1.0, 1.0, owner)
check("3 blind reward reaches the shop", adv[1] == 1.0 and adv[0] == 1.0, str(adv))
check("3 not the shop decision after it", adv[5] == 0.0, str(adv))

# 5. Blind clock: gamma 0.95 only on the step that ends a blind; a capped game bootstraps.
owner = [1, 0, 0, 1, 0, 0]
g = [1.0, 1.0, 0.95, 1.0, 1.0, 1.0]
r = [0, 0, 1.0, 0, 0, 0]
v = [0.0] * 6 + [2.0]                                             # capped: value 2 past the end
adv = routed_gae(r, v, g, 1.0, owner)
check("5 shop sees the blind undiscounted", abs(adv[0] - (1.0 + 0.95 * 2.0)) < 1e-6, str(adv))
check("5 capped bootstrap", abs(adv[5] - 2.0) < 1e-6 and abs(adv[3] - 2.0) < 1e-6, str(adv))

# 6. Per-network lambda.
for trial in range(200):
    n = rng.randint(1, 40)
    r = [rng.uniform(-1, 2) for _ in range(n)]
    v = [rng.uniform(-1, 1) for _ in range(n + rng.randint(0, 1))]
    g = [rng.choice([1.0, 0.95]) for _ in range(n)]
    l = [rng.choice([1.0, 0.95, 0.8]) for _ in range(n)]
    owner = [rng.randint(0, 1) for _ in range(n)]
    check("6 dict == list", np.allclose(routed_gae(r, v, g, l, owner), routed_gae(r, v, g, {0: l, 1: l}, owner),
                                        atol=1e-6), f"trial {trial}")
    l0 = [x * rng.choice([1.0, 0.8]) for x in l]
    a_split = routed_gae(r, v, g, {0: l0, 1: l}, owner)
    a_plain = routed_gae(r, v, g, l, owner)
    shop = [t for t in range(n) if owner[t] == 1]
    check("6 shop unchanged", np.allclose(a_split[shop], a_plain[shop], atol=1e-6), f"trial {trial}")
# a final blind: shop, then table moves play, drag, drag, discard, play (dies: -2 on the last step)
owner = [1, 0, 0, 0, 0, 0]
hand = [False, True, False, False, True, True]
r = [0, 0, 0, 0, 0, -2.0]
lam_table = [0.8 if h else 1.0 for h in hand]
adv = routed_gae(r, [0.0] * 6, 1.0, {0: lam_table, 1: [1.0] * 6}, owner)
check("6 last move takes it all", abs(adv[5] + 2.0) < 1e-6, str(adv))
check("6 one hand back: 0.8", abs(adv[4] + 1.6) < 1e-6, str(adv))
check("6 drags share the next hand's credit", abs(adv[3] + 1.6) < 1e-6 and abs(adv[2] + 1.6) < 1e-6, str(adv))
check("6 first play: two hands back, 0.8^2", abs(adv[1] + 1.28) < 1e-6, str(adv))
check("6 shop on the blind clock", abs(adv[0] + 2.0) < 1e-6, str(adv))

print("\n".join(FAILS) if FAILS else "ALL PASS")
sys.exit(1 if FAILS else 0)
