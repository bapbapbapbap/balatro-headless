#!/usr/bin/env bash
# Replays on the untouched game (--full) only the routes in wins/routes.jsonl that were added after
# a given commit, split across parallel processes. For checking a publish without replaying every
# route again.
#   bash tools/verify_new.sh <commit before the publish> [jobs, default 12]
#   e.g. bash tools/verify_new.sh c22251e 12      (the 578 seeds published 2026-11-03)
# Env: SRC (game source, default ~/balatro/local/source), VENV (default ~/balatro/venv).
# Each process is one sim (~0.45 GB); don't run it alongside training.
set -uo pipefail
cd "$(dirname "$0")/.."
SINCE="${1:?usage: verify_new.sh <commit before the publish> [jobs]}"
JOBS="${2:-12}"
SRC="${SRC:-$HOME/balatro/local/source}"
VENV="${VENV:-$HOME/balatro/venv}"
TMP=$(mktemp -d)
git show "$SINCE:wins/routes.jsonl" > "$TMP/old.jsonl" || { echo "no wins/routes.jsonl at $SINCE"; exit 1; }
python3 - "$TMP" "$JOBS" <<'EOF' || exit 1
import json, sys
tmp, jobs = sys.argv[1], int(sys.argv[2])
key = lambda r: (r["seed"], r["back"], r["stake"])
old = {key(json.loads(l)) for l in open(f"{tmp}/old.jsonl") if l.strip()}
new = [l for l in open("wins/routes.jsonl") if l.strip() and key(json.loads(l)) not in old]
if not new:
    sys.exit("no routes added since that commit")
jobs = min(jobs, len(new))
outs = [open(f"{tmp}/shard{i:02d}.jsonl", "w") for i in range(jobs)]
for i, l in enumerate(new):
    outs[i % jobs].write(l)
print(f"{len(new)} new routes in {jobs} parallel replays on the untouched game")
EOF
. "$VENV/bin/activate"
cd train
ls "$TMP"/shard*.jsonl | xargs -P "$JOBS" -I{} sh -c \
  'python3 verify_games.py --src "$0" --wins-only --full "$1" > "$1.log" 2>&1; tail -2 "$1.log" | head -1 | sed "s|^|$(basename "$1" .jsonl): |"' "$SRC" {}
echo
grep -h MISMATCH "$TMP"/shard*.log || true
python3 - "$TMP" <<'EOF'
import glob, re, sys
n = bad = 0
for f in glob.glob(f"{sys.argv[1]}/shard*.log"):
    m = re.search(r"(\d+) games replayed \(\d+ wins\), (\d+) mismatches", open(f).read())
    if not m:
        print(f"{f}: no summary line (crashed?); see the log"); bad += 1; continue
    n += int(m.group(1)); bad += int(m.group(2))
print(f"total: {n} routes replayed on the untouched game, {bad} problems -> {'PASS' if n and not bad else 'FAIL'}")
print(f"logs: {sys.argv[1]}")
EOF
