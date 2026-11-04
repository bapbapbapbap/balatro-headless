#!/usr/bin/env bash
# Trains one early-game model (docs/hypotheses.md section I; chud's design, 17:43): from scratch on
# 1.0's architecture, no pin, and its run ends once ante 4 is reached (the ante-3 boss beaten), so
# the expert for the build takes over from the cash-out and shop going into ante 4.
#   bash run_early.sh <shared folder> <seed> [hours, default 8] [workers, default 14]
# Several seeds are grown and the best kept: best.pt is picked by the share of the 200 fixed eval
# seeds that reach ante 4. Reward `blinds` (at most 9 before the stop), shaping on, seed retries
# off (fresh seeds only), hand-type exploration 0.05, lr 1e-4. Results in runs/early-s<seed>.
# STAKE=1..8 / BACK=<deck> change the stake / deck (default White Stake, Red Deck).
set -uo pipefail
SHARED="${1:?usage: run_early.sh <shared folder> <seed> [hours] [workers]}"
SEED="${2:?usage: run_early.sh <shared folder> <seed> [hours] [workers]}"
HOURS="${3:-8}"
WORKERS="${4:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
SUFFIX=""; [ "${STAKE:-1}" != 1 ] && SUFFIX="$SUFFIX-stake${STAKE}"; [ "${BACK:-b_red}" != b_red ] && SUFFIX="$SUFFIX-${BACK}"
export ENC="${ENC:-3}" RUN="early-s$SEED$SUFFIX" ACTIONS=ordered-capped LR="${LR:-1e-4}" FROM_SCRATCH=1 RUN_V2_UPDATED=1
export TRAIN_ARGS="--seed $SEED --stop-ante ${STOP_ANTE:-4} --best-by reached --retry 0 --explore-hands ${EXPLORE:-0.05} ${TRAIN_ARGS:-}"
exec bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS"
