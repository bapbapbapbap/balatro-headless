#!/usr/bin/env bash
# Grows several early-game models one after another (run_early.sh), one per seed.
#   bash run_earlies.sh <shared folder> [hours each, default 8] [workers, default 14]
# SEEDS="1 2 3 4 5" by default; give each machine its own list to split them, e.g. SEEDS="4 5".
# Compare them afterwards by best eval (share reaching ante 4) in runs/early-s*/metrics.jsonl.
set -uo pipefail
SHARED="${1:?usage: run_earlies.sh <shared folder> [hours each] [workers]}"
HOURS="${2:-8}"
WORKERS="${3:-14}"
D="$(cd "$(dirname "$0")" && pwd)"
for SEED in ${SEEDS:-1 2 3 4 5}; do
  echo "==== early-game model, seed $SEED ($(date))"
  bash "$D/run_early.sh" "$SHARED" "$SEED" "$HOURS" "$WORKERS" || echo "seed $SEED exited with an error; going on"
done
