"""Turns the wins in a tools/eval_ladder.py output into route records (the format of wins/routes.jsonl), each
replayed on the fast sim and again on the untouched game, and kept only if both replays win with the same
final state. Records are marked verified (win_check 2), so train/publish_wins.py publishes them as they are.

  python3 tools/ladder_to_routes.py gold/v6dt_t0_s6.jsonl --src ~/balatro/local/source --out purple_routes.jsonl
  python3 train/publish_wins.py --src ~/balatro/local/source --out wins/ <folder named for the model>/purple_routes.jsonl

The folder the routes file sits in names the model in the tally (wins/README.md), so put it in one named for the model.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repo", default=os.path.expanduser("~/balatro/balatro-headless"))
ap.add_argument("--actions", default="ordered-capped")
ap.add_argument("--enc", type=int, default=2)
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.repo, "train"))
from sim import Sim  # noqa: E402

recs = [json.loads(l) for l in open(a.file) if l.strip()]
head = recs[0].get("run", {}) if recs and "run" in recs[0] else {}
back, stake = head.get("back", "b_red"), head.get("stake", 1)
wins = [r for r in recs if r.get("won")]
commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
src_name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(a.file))), os.path.basename(a.file))
print(f"{a.file}: {len(wins)} wins, {back}, stake {stake}")

fast = Sim(a.src, fast=True, mode=a.actions, enc=a.enc)
full = Sim(a.src, fast=False, mode=a.actions, enc=a.enc)
out, bad = [], 0
for r in wins:
    p1, n1, f1, e1 = fast.replay(r["seed"], back, stake, r["acts"])
    desc = list(fast.desc)
    p2, n2, f2, e2 = full.replay(r["seed"], back, stake, r["acts"], desc)
    ok = p1["won"] and p2["won"] and not e1 and not e2 and f1 == f2 and n1 == n2 == len(r["acts"])
    print(f"{r['seed']}: {'ok' if ok else 'MISMATCH'}  fast {p1['won']} {f1}, full {p2['won']} {f2}, {n1} moves {(e1 or e2)[:1]}")
    if not ok:
        bad += 1
        continue
    out.append({"win_check": 2, "verified": True, "seed": r["seed"], "back": back, "stake": stake,
                "found_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "actions": a.actions, "enc": a.enc,
                "acts": list(r["acts"]), "desc": desc, "final": f1, "n": n1, "progress": p1, "mode": "eval",
                "iteration": None, "commit": commit,
                "converted_from": f"{src_name} (tools/eval_ladder.py; {head.get('weights')}, temperature {head.get('temp')})"})
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"{len(out)} routes written to {a.out}; {bad} mismatches")
print("PASS" if out and not bad else "FAIL")
sys.exit(0 if out and not bad else 1)
