"""Checks that two commits' trainers do exactly the same thing: same games, same weights after training.

    (from inside the clone) python3 tests/compare_trainers.py --src ~/balatro/local/source --ref 8a186ae --new HEAD [--rounds 3]

Used when the trainer's code is reorganized without meaning to change what it does (2026-11-10: train.py split
into modules). Each commit is checked out into its own temporary worktree and trains for a few rounds on the
CPU with one worker and a fixed seed, with model v4b's settings (phase split, phi_build, offer values,
curriculum, hand clock, routed credit, both potentials). The reference runs twice first, to show the setup is
deterministic; then the new commit runs once. Every saved game (seed, moves, final state hash, outcome) and
every weight in final.pt must be identical. Prints ALL PASS or what differs. Writes only under /tmp.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

# the clone this is run from (the script itself may be a copy outside it)
REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()

V4B = ("--seed 3 --lam-hand 0.8 --gamma-blind 0.95 --lam-blind 0.95 --phase-split --build-shaping 1 "
       "--offer-values --curriculum 0.3 --curriculum-published 300 --curriculum-keep 600 "
       "--shop-ent 0.02,0.006@1.5M,0.001@4M --enc 2 --actions ordered-capped --d 192 --layers 4 --heads 6 "
       "--lr 1e-4 --reward blinds --back b_red --stake 1").split()


def worktree(commit, where):
    subprocess.run(["git", "-C", REPO, "worktree", "add", "-q", "--detach", where, commit], check=True)
    vendor = os.path.join(REPO, "harness", "vendor")          # the LuaJIT 2.0.5 runtime is built, not committed
    shutil.copytree(vendor, os.path.join(where, "harness", "vendor"))


def train(tree, out, src, rounds):
    cmd = [sys.executable, os.path.join(tree, "train", "train.py"), "--src", src, "--run", "eq", "--out", out,
           "--device", "cpu", "--workers", "1", "--games-per-worker", "4", "--iterations", str(rounds),
           "--eval-every", "2", "--eval-games", "6", "--hours", "1", "--save-games", "all"] + V4B
    r = subprocess.run(cmd, cwd=os.path.join(tree, "train"), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(out, "final.pt")):
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"training failed in {tree}")


def games(out):
    d, rows = os.path.join(out, "games"), {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl.gz"):
            rows[f] = [{k: v for k, v in json.loads(line).items() if "utc" not in k and "time" not in k}
                       for line in gzip.open(os.path.join(d, f), "rt")]
    return rows


def weights(out):
    return torch.load(os.path.join(out, "final.pt"), map_location="cpu", weights_only=False)["model"]


def diff(a, b, name):
    problems = []
    ga, gb = games(a), games(b)
    if sorted(ga) != sorted(gb):
        problems.append(f"{name}: different game files {sorted(set(ga) ^ set(gb))}")
    for f in sorted(set(ga) & set(gb)):
        if ga[f] != gb[f]:
            k = next(i for i, (x, y) in enumerate(zip(ga[f] + [None], gb[f] + [None])) if x != y)
            problems.append(f"{name}: {f} differs from game {k}")
    wa, wb = weights(a), weights(b)
    if sorted(wa) != sorted(wb):
        problems.append(f"{name}: different parameters")
    else:
        bad = [k for k in wa if not torch.equal(wa[k], wb[k])]
        if bad:
            problems.append(f"{name}: {len(bad)} of {len(wa)} weight tensors differ, e.g. {bad[0]}")
    n = sum(len(v) for v in ga.values())
    print(f"{name}: {n} saved games, {len(wa)} weight tensors compared: {'PASS' if not problems else 'FAIL'}", flush=True)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True, help="commit whose trainer is the reference")
    ap.add_argument("--new", default="HEAD")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="compare_trainers_")
    trees = {}
    try:
        for name, commit in (("ref", a.ref), ("new", a.new)):
            trees[name] = os.path.join(tmp, name)
            worktree(commit, trees[name])
        outs = {k: os.path.join(tmp, "out_" + k) for k in ("ref1", "ref2", "new")}
        for k, tree in (("ref1", trees["ref"]), ("ref2", trees["ref"]), ("new", trees["new"])):
            print(f"training {k} ({a.ref if k != 'new' else a.new}), {a.rounds} rounds ...", flush=True)
            train(tree, outs[k], os.path.expanduser(a.src), a.rounds)
        problems = diff(outs["ref1"], outs["ref2"], "reference twice (determinism)")
        if problems:
            print("\n".join(problems))
            raise SystemExit("the reference isn't deterministic here, so the comparison means nothing")
        problems = diff(outs["ref1"], outs["new"], "reference vs new")
        print("\n".join(problems) if problems else "ALL PASS")
    finally:
        for t in trees.values():
            subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force", t], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
