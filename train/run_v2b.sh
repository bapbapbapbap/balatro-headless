#!/usr/bin/env bash
# v2b: v2's model and weights, continued from runs/v2/latest.pt with three changes (docs/hypotheses.md,
# section H):
#   - joker-drag cap: --actions ordered-capped (env.lua: at most jokers-1 drags in a row)
#   - hand-type exploration: --explore-hands (train.py explore_sample), default 0.05
#   - learning rate 1e-5, as v2 had at the end
# Results in <shared folder>/runs/v2b; runs/v2 is left as it is.
#   bash run_v2b.sh <shared folder> [hours, default 168] [workers, default 14]
set -uo pipefail
SHARED="${1:?usage: run_v2b.sh <shared folder> [hours] [workers]}"
export RUN=v2b ACTIONS=ordered-capped RESUME_FROM="$SHARED/runs/v2/latest.pt" LR="${LR:-1e-5}"
export TRAIN_ARGS="--explore-hands ${EXPLORE:-0.05} ${TRAIN_ARGS:-}"
W="${BALATRO_HOME:-$HOME/balatro}"
exec bash "$W/balatro-headless/train/run_v2.sh" "$@"
