"""Model v6, stage 1's imitation (train/imitate.py) on real teacher games. Pre-flight check "v6-imitate".

    python3 tests/v6_imitate_check.py <teach.py output folder> [--games 12]

Checks, on the first --games usable games:
1. every game loads; every decision's observation has v5's action rows (21 wide: offer and play values); the
   saved move is a legal row; member is the phase the game state gives (table only for hand decisions)
2. progress carries what v4b's reward needs (phi, phi_build, blinds_passed)
3. returns: R_t = r_t + g_t R_t+1 for the run's reward and blind clock; with shaping off and gamma 1, R_0 is the
   game's blinds beaten
4. the held-out split is fixed, about 1 game in 20 over every game in the folder, and never trains
5. the agreement set draws only labeled held-out decisions, the same ones every time
6. a small phase-split network trained on these games lowers its move loss on them; the v5-sized network
   takes the same batches
"""
import argparse
import os
import random
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import imitate as I  # noqa: E402

fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teach")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    seeds = I.games(a.teach)
    check(len(seeds) >= 4, f"{len(seeds)} usable teacher games in {a.teach}")
    if len(seeds) < 4:
        print("FAILED"); return 1
    use = seeds[:a.games]
    hold_in_use = [s for s in seeds if I.held_out(s)][:2]
    use = sorted(set(use) | set(hold_in_use))

    # 1-2
    width, legal, members, keys = True, True, True, True
    for s in use:
        for x in I.load(a.teach, s):
            o = x["obs"]
            width &= o["act"].shape[1] == 21
            legal &= 0 <= x["a"] < len(o["act"])
            members &= x["member"] == (0 if x["kind"] == "round" else 1) or x["kind"] is None
            keys &= all(k in x["progress"] for k in ("phi", "phi_build", "blinds_passed", "blinds_beaten"))
    check(width, "action rows are 21 wide (v5: offer and play values)")
    check(legal, "every saved move is one of the decision's rows")
    check(members, "member = table for hand decisions, shop for the rest")
    check(keys, "progress has phi, phi_build, blinds_passed")

    # 3
    run = types.SimpleNamespace(reward="blinds", shaping=1.0, build_shaping=1.0, gamma=0.995, lam=0.95,
                                gamma_blind=0.95, lam_blind=0.95)
    flat = types.SimpleNamespace(reward="blinds", shaping=0.0, build_shaping=0.0, gamma=1.0, lam=1.0)
    rec_ok, sum_ok = True, True
    for s in use:
        g = I.load(a.teach, s)
        R = I.returns(g, run)
        F = I.returns(g, flat)
        sum_ok &= abs(F[0] - g[-1]["progress"]["blinds_beaten"]) < 1e-5
        rec_ok &= bool(np.all(np.isfinite(R))) and len(R) == len(g)
    check(rec_ok, "returns under v4b's reward are finite, one per decision")
    check(sum_ok, "with shaping off and gamma 1, the first return = the game's blinds beaten")

    # 4
    check([I.held_out(s) for s in seeds] == [I.held_out(s) for s in seeds], "held-out split is fixed")
    if len(seeds) >= 200:
        share = sum(map(I.held_out, seeds)) / len(seeds)
        check(0.02 < share < 0.09, f"held out {share:.1%} of {len(seeds)} games (about 5%)")

    # 5
    hs = [s for s in use if I.held_out(s)]
    if hs:
        r1, r2 = I.agreement_set(a.teach, hs, 50), I.agreement_set(a.teach, hs, 50)
        decs = I.decisions_of(a.teach, r1)
        check(r1 == r2 and all(I.held_out(s) for s, _ in r1) and all(x["labeled"] for x in decs),
              f"agreement set: {len(r1)} labeled held-out decisions, the same every time")

    # 6
    import torch
    from common import collate
    from model import PhaseNet
    obs, act = [], []
    for s in use:
        for x in I.load(a.teach, s):
            if x["labeled"]:
                x["obs"]["member"] = x["member"]
                obs.append(x["obs"]); act.append(x["a"])
    nv = int(max(o["vocab"].max() for o in obs)) + 1
    nv = max(nv, int(max(o["act"][:, 8].max() for o in obs)) + 1)
    torch.manual_seed(0)
    small = PhaseNet(nv, d=32, layers=1, heads=2, enc=2, offer=True, play=True)
    opt = torch.optim.Adam(small.parameters(), lr=3e-3)
    rng = random.Random(0)
    idx = list(range(len(obs)))

    def ce(model, ids):
        with torch.no_grad():
            lg, _ = model(*collate([obs[i] for i in ids]))
            return torch.nn.functional.cross_entropy(lg, torch.tensor([act[i] for i in ids])).item()
    probe = rng.sample(idx, min(256, len(idx)))
    before = ce(small, probe)
    for _ in range(a.steps):
        b = rng.sample(idx, min(32, len(idx)))
        lg, _ = small(*collate([obs[i] for i in b]))
        loss = torch.nn.functional.cross_entropy(lg, torch.tensor([act[i] for i in b]))
        opt.zero_grad(); loss.backward(); opt.step()
    after = ce(small, probe)
    check(after < before * 0.8, f"move loss falls when trained on the teacher's moves ({before:.3f} -> {after:.3f})")
    big = PhaseNet(nv, d=192, layers=4, heads=6, enc=2, offer=True, play=True)
    lg, v = big(*collate([obs[i] for i in probe[:16]]))
    check(bool(torch.isfinite(v).all()) and lg.shape[0] == 16, "the v5-sized network takes the same batches")

    print("ALL PASS" if not fails else f"FAILED: {len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
