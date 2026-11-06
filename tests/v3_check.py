"""Pre-flight for model v3 (docs/hypotheses.md, 2026-11-05 ~21:19 CST): the game-time discount.
No torch needed. Replays published wins (wins/routes.jsonl) on the real sim and checks, at every step:

  1. blinds_passed rises by exactly one when a blind becomes Defeated or Skipped, and never otherwise;
     at the end it equals blinds beaten + blinds skipped (24 for a win with no skips).
  2. step_discount: per action (no --gamma-blind) every step is (gamma, lam), as before v3; with
     --gamma-blind every step is (1, 1) except the steps where a blind is passed.
  3. The shaping term still telescopes: over a whole game, sum(gamma_t * phi' - phi) = -phi(start) = 0
     under the blind clock, so it adds nothing to a game's return (Ng et al. 1999).
  4. gae / discounted with per-step discounts: equal to the scalar versions when every step is the
     same, and equal to a hand computation on a small example.

  python3 v3_check.py <game source dir> [--routes N]
"""
import argparse
import ast
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from rewards import shaping, step_discount  # noqa: E402
from sim import Sim  # noqa: E402


def trainer_functions():
    """gae, discounted and per_step from train.py, without importing torch."""
    src = open(os.path.join(HERE, "..", "train", "train.py")).read()
    ns = {"np": np}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("per_step", "gae", "discounted"):
            exec(compile(ast.Module([node], []), "train.py", "exec"), ns)
    return ns["gae"], ns["discounted"]


def check_returns():
    gae, discounted = trainer_functions()
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=50).tolist(), rng.normal(size=50).tolist()
    assert np.allclose(gae(r, v, 0.995, 0.95), gae(r, v, [0.995] * 50, [0.95] * 50))
    assert np.allclose(discounted(r, 0.995), discounted(r, [0.995] * 50))
    # hand example: 3 steps, the blind passed on step 1 (gamma 0.9, lam 0.5 there), 1 elsewhere
    r, v, g, lam = [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [1.0, 0.9, 1.0], [1.0, 0.5, 1.0]
    d2 = 3.0 + 0 - 1.5                       # last step: no bootstrap
    d1 = 2.0 + 0.9 * 1.5 - 1.0
    d0 = 1.0 + 1.0 * 1.0 - 0.5
    a2, a1 = d2, d1 + 0.9 * 0.5 * d2
    a0 = d0 + 1.0 * 1.0 * a1
    assert np.allclose(gae(r, v, g, lam), [a0, a1, a2]), (gae(r, v, g, lam), [a0, a1, a2])
    assert np.allclose(discounted(r, g), [1 + 2 + 0.9 * 3, 2 + 0.9 * 3, 3])
    print("returns with per-step discounts: PASS")


def check_skips(src, clock):
    """Two skipped blinds in a row tick the clock twice (published wins rarely skip)."""
    sim = Sim(src, mode="ordered-capped", enc=2)
    o = sim.reset("SKIPTEST", "b_red", 1, None)
    for n in (1, 2):
        sk = [i for i in range(len(o["act"])) if sim.H.action_desc(i + 1).startswith("skip")]
        assert sk, "no skip offered"
        prev, cur, done = sim.step(sk[0])
        assert cur["blinds_passed"] == n and cur["blinds_beaten"] == 0, (n, cur)
        assert np.allclose(step_discount(prev, cur, clock), (0.95, 0.9))
        o = sim.observe()
    print("skipped blinds tick the clock: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--routes", type=int, default=12)
    a = ap.parse_args()
    check_returns()
    per_action = SimpleNamespace(gamma=0.995, lam=0.95)
    clock = SimpleNamespace(gamma=0.995, lam=0.95, gamma_blind=0.95, lam_blind=0.9)
    check_skips(a.src, clock)
    routes = [json.loads(l) for l in open(os.path.join(HERE, "..", "wins", "routes.jsonl")) if l.strip()]
    routes = [r for r in routes if not r.get("challenge")][-a.routes:]
    sims, fails, skips_seen = {}, 0, 0
    for r in routes:
        key = (r.get("actions", "subsets"), r.get("enc", 1))
        if key not in sims:
            sims[key] = Sim(a.src, mode=key[0], enc=key[1])
        sim = sims[key]
        o = sim.reset(r["seed"], r["back"], r["stake"], r.get("challenge"))
        marks = dict(zip(("Small", "Big", "Boss"), sim.H.blind_marks()))
        passed = skipped = 0
        tel = 0.0
        prob = None
        for k, act in enumerate(r["acts"]):
            prev, cur, done = sim.step(act)
            now = dict(zip(("Small", "Big", "Boss"), sim.H.blind_marks()))
            new = sum(1 for b, m in now.items() if m in ("Defeated", "Skipped") and marks.get(b) not in ("Defeated", "Skipped"))
            skipped += sum(1 for b, m in now.items() if m == "Skipped" and marks.get(b) != "Skipped")
            marks = now
            passed += new
            if cur["blinds_passed"] - prev["blinds_passed"] != new:
                prob = f"step {k}: blinds_passed +{cur['blinds_passed'] - prev['blinds_passed']}, marks say +{new}"
            if step_discount(prev, cur, per_action) != (0.995, 0.95):
                prob = f"step {k}: per-action discount changed"
            g, l = step_discount(prev, cur, clock)
            want = (0.95 ** new, 0.9 ** new)
            if not np.allclose((g, l), want):
                prob = f"step {k}: blind clock gave {(g, l)}, want {want}"
            tel += shaping(prev, cur, g)
            if prob:
                break
            if not done:
                sim.observe()
        if not prob and cur["blinds_passed"] != cur["blinds_beaten"] + skipped:
            prob = f"blinds_passed {cur['blinds_passed']} != beaten {cur['blinds_beaten']} + skipped {skipped}"
        if not prob and abs(tel) > 1e-9:
            prob = f"shaping under the blind clock sums to {tel:.3g}, not 0"
        skips_seen += skipped
        print(f"{r['seed']}: {len(r['acts'])} steps, won {cur['won']}, beaten {cur['blinds_beaten']}, "
              f"skipped {skipped}, passed {cur['blinds_passed']}, shaping sum {tel:+.1e}  "
              + ("ok" if not prob else "FAIL: " + prob))
        fails += bool(prob)
    if skips_seen == 0:
        print("note: none of these routes skipped a blind (the skip path is checked separately above)")
    print("v3 checks: " + ("ALL PASS" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
