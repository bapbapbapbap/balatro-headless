#!/usr/bin/env bash
# Trains model v1 overnight on this machine. Run after tools/setup_linux.sh.
#   bash run_v1.sh <shared folder for results> [hours, default 168] [workers, default 14]
# Extra train.py flags for one launch: TRAIN_ARGS="--reset-lr" bash run_v1.sh ...
# Stops at MAX_WINS won games (default 1,000,000: runs that beat the ante 8 boss), MAX_GAMES total
# games (default 0 = no limit), the hours cap, or any safety stop, whichever comes first.
# Weights (ckpt_*.pt, best.pt, latest.pt, final.pt), metrics.jsonl, health.json, provenance.json
# and train.log go to <shared folder>/runs/v1. Keep this window open and the machine awake.
#
# Safety: pre-flight accuracy checks below must pass or nothing starts. During the run, train.py
# rejects bad updates, restarts crashed or hung workers, sizes the pool to free memory, stops
# cleanly on low memory or disk or if learning stalls, flags heat throttling, keeps best.pt, and
# stops within seconds on Ctrl+C with weights saved. See health.json for its state at any time.
set -uo pipefail
OUT="${1:?usage: run_v1.sh <shared folder> [hours] [workers]}/runs/v1"
HOURS="${2:-168}"
WORKERS="${3:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
mkdir -p "$OUT"
# Update first, then start over from the updated script: bash reads a script as it runs, so a pull
# that changes this file mid-run would make it skip or garble lines.
if [ -z "${RUN_V1_UPDATED:-}" ]; then
  git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
  RUN_V1_UPDATED=1 exec bash "$R/train/run_v1.sh" "$@"
fi
exec > >(tee -a "$OUT/train.log") 2>&1
echo "== run_v1 $(date)"
. "$W/venv/bin/activate"
git -C "$R" log --oneline -1
python3 -c "import torch" 2>/dev/null || pip install -q torch numpy
python3 -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"

echo "== pre-flight checks"
python3 "$R/harness/check_env.py" "$R/tests/fixtures/1793565485_2730e932.json.gz" --src "$W/local/source" \
  --fast --mode subsets | tail -1 | grep -q PASS \
  && echo "env vs logged run: PASS" || { echo "PRE-FLIGHT FAILED: env vs logged run"; exit 1; }
( cd "$R/harness" && python3 diff_check.py --src "$W/local/source" --runs 5 --encode | tail -1 | grep -q PASS ) \
  && echo "differential check with encoder: PASS" || { echo "PRE-FLIGHT FAILED: differential check"; exit 1; }
( cd "$R/train" && python3 verify_games.py --src "$W/local/source" "$R/tests/fixtures/win_v1_r549_1JA35NNR.jsonl" --wins-only --full --fresh-each | tail -1 | grep -q PASS ) \
  && echo "saved win replays exactly (untouched game): PASS" || { echo "PRE-FLIGHT FAILED: saved-game replay"; exit 1; }
free -g | head -2
df -h "$OUT" | tail -1

echo "== training"
cd "$R/train"
RESUME=()
if [ -f "$OUT/latest.pt" ]; then RESUME=(--resume "$OUT/latest.pt"); echo "resuming from $OUT/latest.pt"; fi
python3 train.py --src "$W/local/source" --run v1 --out "$OUT" --hours "$HOURS" --workers "$WORKERS" \
  --reward blinds --back b_red --stake 1 --max-games "${MAX_GAMES:-0}" --max-wins "${MAX_WINS:-1000000}" \
  --mirror-dir "${MIRROR_DIR:-/mnt/f/balatro data}" --spill-dir "${SPILL_DIR:-/mnt/f/balatro data}" "${RESUME[@]}" ${TRAIN_ARGS:-}
echo "== exited $(date)"
