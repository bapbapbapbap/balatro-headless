"""Checks every Gold win credited to v6g best.pt before the seeds and the policy are published.

Route replays (ladder_to_routes, verify_new.sh) prove a move list wins a seed. They don't prove the frozen
policy chose those moves, or that "1 win in 9,000" counted every game. This checks both:

  1. Weights   best.pt's sha256 starts with the registered r660 hash.
  2. Records   every eval_ladder file a win came from: the header says best.pt, policy model, its temperature,
               the deck, stake 8, no test filters; its games are exactly the first N eval seeds (no gaps, no
               duplicates, no extra seeds), so N is an honest denominator; its wins and the routes are the same
               seeds with the same moves.
  3. Policy    best.pt plays each win seed again in a fresh process at the logged temperature; it must choose the
               route's moves exactly and win.
  4. Replay    every route replayed on the untouched game (train/verify_games.py --full).
  5. Coverage  a deck counts only if a route passes 1-4. Red's training-time wins (mode "train") don't count:
               they come from training, not from the frozen policy.

  python3 verify_gold.py [--static-only] [--workers 12]
Exit 0 only if every check passes and all 15 decks are covered.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from argparse import Namespace

HOME = os.path.expanduser("~")
DECKS = ["red", "blue", "yellow", "green", "black", "magic", "nebula", "ghost", "abandoned", "checkered",
         "zodiac", "painted", "anaglyph", "plasma", "erratic"]

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=f"{HOME}/balatro/balatro-headless")
ap.add_argument("--src", default=f"{HOME}/balatro/local/source")
ap.add_argument("--gold", default=os.path.join(os.environ.get("BALATRO_WORK", "."), "gold"))
ap.add_argument("--runs", default=os.path.join(os.environ.get("BALATRO_WORK", "."), "runs", "v6g"))
ap.add_argument("--weights", default=None, help="default <runs>/best.pt")
ap.add_argument("--sha", default="4462f80e4b41fb9e", help="registered sha256 prefix of v6g best.pt (r660)")
ap.add_argument("--published-ref", default="origin/publish-v6g-gold-green-abandoned",
                help="git ref whose wins/routes.jsonl holds the Gold routes already published")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--static-only", action="store_true", help="checks 1, 2 and 5 only (no game needed)")
a = ap.parse_args()
a.weights = a.weights or os.path.join(a.runs, "best.pt")
sys.path[:0] = [os.path.join(a.repo, "train"), os.path.join(a.repo, "tools")]

fails = []


def fail(msg):
    fails.append(msg)
    print(f"  FAIL {msg}", flush=True)


def temp_of(r):
    m = re.search(r"temperature ([0-9.]+)", r.get("converted_from") or "")
    if m:
        return float(m.group(1))
    return 1.0 if r.get("mode") == "eval" else None      # a training run's own eval samples at temperature 1


# ---------------- 1. weights ----------------
print("1. Weights")
h = hashlib.sha256(open(a.weights, "rb").read()).hexdigest()
print(f"  {a.weights}: sha256 {h}")
if not h.startswith(a.sha):
    fail(f"best.pt sha256 {h[:16]} is not the registered {a.sha}")

# ---------------- gather routes ----------------
routes = {}       # (seed, back) -> route
for f in sorted(glob.glob(os.path.join(a.runs, "*_gold_t*_routes.jsonl"))):
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            routes[(r["seed"], r["back"])] = dict(r, _file=os.path.basename(f))
pub = subprocess.run(["git", "-C", a.repo, "show", f"{a.published_ref}:wins/routes.jsonl"],
                     capture_output=True, text=True)
if pub.returncode:
    fail(f"can't read wins/routes.jsonl at {a.published_ref}: {pub.stderr.strip()}")
train_only = collections.Counter()
for l in pub.stdout.splitlines():
    r = json.loads(l) if l.strip() else None
    if not r or r["stake"] != 8:
        continue
    from_v6g = "v6g_t" in (r.get("converted_from") or "") or \
        (r.get("mode") == "eval" and r.get("iteration") == 660 and r.get("source", "").startswith("v6g/"))
    if not from_v6g:
        train_only[r["back"]] += 1
        continue
    routes.setdefault((r["seed"], r["back"]), dict(r, _file=f"{a.published_ref}:wins/routes.jsonl"))
for r in routes.values():
    r["_temp"] = temp_of(r)
    m = re.match(r"\S*?(v6g_t\d_s8(?:_\w+)?\.jsonl)", r.get("converted_from") or "")
    r["_ladder"] = os.path.join(a.gold, m.group(1)) if m else None
print(f"\n{len(routes)} Gold routes credited to best.pt; training-time Gold routes not counted: "
      f"{dict(train_only) or 'none'}")

# ---------------- 2. records ----------------
print("\n2. Records (eval_ladder files)")
from common import eval_seeds  # noqa: E402
by_ladder = collections.defaultdict(list)
for r in routes.values():
    if r["_ladder"]:
        by_ladder[r["_ladder"]].append(r)
    elif r.get("mode") == "eval" and r.get("iteration") == 660:
        print(f"  {r['back']} {r['seed']}: from the training run's own eval at r660 (no eval_ladder file; "
              f"its denominator isn't checked here)")
    else:
        fail(f"{r['back']} {r['seed']}: no eval_ladder file named in converted_from")
denoms = {}
for path, rs in sorted(by_ladder.items()):
    name = os.path.basename(path)
    if not os.path.exists(path):
        fail(f"{name}: missing")
        continue
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head = lines[0].get("run", {}) if lines else {}
    back, temp = rs[0]["back"], rs[0]["_temp"]
    want = {"policy": "model", "temp": temp, "back": back, "stake": 8, "survive": 0, "shun_stickers": 0, "no_skip": 0}
    filters = ("survive", "shun_stickers", "no_skip")          # older headers predate some filters: absent = off
    bad = {k: head.get(k) for k in want if head.get(k, 0 if k in filters else None) != want[k]}
    if bad:
        fail(f"{name}: header {bad}, expected {want}")
    if os.path.basename(head.get("weights", "")) != "best.pt" or "v6g" not in head.get("weights", ""):
        fail(f"{name}: played by {head.get('weights')}, not v6g best.pt")
    games = [g for g in lines[1:] if "seed" in g and not g.get("error")]
    seeds = [g["seed"] for g in games]
    dup = [s for s, c in collections.Counter(seeds).items() if c > 1]
    expect = eval_seeds(len(set(seeds)))
    missing, extra = set(expect) - set(seeds), set(seeds) - set(expect)
    if dup or missing or extra:
        fail(f"{name}: {len(dup)} duplicate seeds, {len(missing)} of the first {len(expect)} eval seeds "
             f"missing, {len(extra)} seeds outside them")
    won = {g["seed"]: g for g in games if g["won"]}
    got = {r["seed"]: r for r in rs}
    if set(won) != set(got):
        fail(f"{name}: wins {sorted(won)} but routes {sorted(got)}")
    for s in set(won) & set(got):
        if list(won[s]["acts"]) != list(got[s]["acts"]):
            fail(f"{name} {s}: route moves differ from the logged game's")
        if got[s]["back"] != back or got[s]["stake"] != 8 or not got[s]["progress"]["won"]:
            fail(f"{name} {s}: route is {got[s]['back']} stake {got[s]['stake']} won {got[s]['progress']['won']}")
    denoms[name] = (len(set(seeds)), len(won))
    print(f"  {name}: {len(set(seeds))} games = the first {len(expect)} eval seeds, {len(won)} wins, "
          f"{len(got)} routes match")

# ---------------- 3. policy rerun ----------------
policy_ok = {}
if not a.static_only:
    print("\n3. Policy: best.pt replays each win seed in fresh processes")
    import multiprocessing as mp
    import eval_ladder as EL
    groups = collections.defaultdict(list)
    for r in routes.values():
        groups[(r["back"], r["_temp"])].append(r)
    for (back, temp), rs in sorted(groups.items()):
        ns = Namespace(src=a.src, weights=a.weights, temp=temp, survive=0, shun_stickers=0, no_skip=0, back=back,
                       stake=8, policy="model", init_seed=0)
        with mp.Pool(min(a.workers, len(rs)), initializer=EL.init, initargs=(ns,)) as pool:
            for r, g in zip(rs, pool.map(EL.one_seed, [r["seed"] for r in rs])):
                ok = not g.get("error") and g["won"] and list(g["acts"]) == list(r["acts"])
                policy_ok[(r["seed"], back)] = ok
                why = "" if ok else (g.get("error", "").strip().splitlines()[-1:] or
                                     [f"won {g['won']}, moves {'same' if list(g['acts']) == list(r['acts']) else 'differ'}"])[0]
                print(f"  {back} t{temp:g} {r['seed']}: {'ok' if ok else 'FAIL ' + why}", flush=True)
                if not ok:
                    fails.append(f"policy rerun {back} {r['seed']}: {why}")

# ---------------- 4. full replay ----------------
replay_ok = {}
if not a.static_only:
    print("\n4. Replay on the untouched game")
    tmp = tempfile.mkdtemp(prefix="verify_gold_")
    rs = list(routes.values())
    procs = []
    for i in range(min(a.workers, len(rs))):
        p = os.path.join(tmp, f"shard{i:02d}.jsonl")
        with open(p, "w") as f:
            for r in rs[i::a.workers]:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
        procs.append((p, rs[i::a.workers], subprocess.Popen(
            [sys.executable, os.path.join(a.repo, "train", "verify_games.py"), "--src", a.src, "--wins-only",
             "--full", p], stdout=open(p + ".log", "w"), stderr=subprocess.STDOUT, cwd=os.path.join(a.repo, "train"))))
    for p, shard, proc in procs:
        proc.wait()
        log = open(p + ".log").read()
        bad = set(re.findall(r"MISMATCH .*? seed (\S+):", log))
        for r in shard:
            replay_ok[(r["seed"], r["back"])] = proc.returncode == 0 and r["seed"] not in bad
        lines = log.strip().splitlines() or ["no output"]
        print(f"  {os.path.basename(p)}: {lines[-2] if len(lines) > 1 else lines[-1]}")
        if proc.returncode:
            fails.append(f"full replay: {p}.log")
    print(f"  logs: {tmp}")

# ---------------- 5. coverage ----------------
print("\n5. Coverage (Gold, routes from best.pt that pass every check)")
shared = collections.defaultdict(list)
for (s, b) in routes:
    shared[s].append(b[2:])
for d in DECKS:
    keys = [k for k in routes if k[1] == f"b_{d}"]
    good = [k for k in keys if a.static_only or (policy_ok.get(k) and replay_ok.get(k))]
    rate = "; ".join(f"{w}/{n} {name.split('_')[1]}" for name, (n, w) in denoms.items() if name.endswith(f"_{d}.jsonl"))
    print(f"  {d:10s} {len(good)}/{len(keys)} routes ok  {rate}")
covered = sum(any(k[1] == f"b_{d}" and (a.static_only or (policy_ok.get(k) and replay_ok.get(k))) for k in routes)
              for d in DECKS)
if covered < 15:
    fail(f"{covered} of 15 decks covered")
multi = {s: b for s, b in shared.items() if len(b) > 1}
if multi:
    print("\n  win seeds shared across decks: " + "; ".join(f"{s} ({', '.join(sorted(b))})" for s, b in sorted(multi.items())))

print(f"\n{covered}/15 decks, {len(fails)} problems -> {'PASS' if not fails else 'FAIL'}"
      + (" (static checks only)" if a.static_only else ""))
for f in fails:
    print(f"  - {f}")
sys.exit(0 if not fails else 1)
