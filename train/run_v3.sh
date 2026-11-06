#!/usr/bin/env bash
# Model v3 (docs/hypotheses.md, 2026-11-05 ~21:19 CST, and its stopping rule ~21:24): two arms from
# random weights, side by side, identical except the discount clock.
#   v3          --gamma-blind 0.95 --lam-blind 0.95   (discount once per blind passed)
#   v3-control  --gamma 0.995 --lam 0.95               (per action, as every run before)
# Everything else is 1.0's: 1.6M parameters (d 192, 4 layers, 6 heads), encoding 2, ordered-capped
# actions, reward blinds with shaping, hand-type exploration 0.05, seed retries, lr 1e-4, Red Deck,
# White Stake, training seed 3 for both (identical starting weights), eval on the 600 fixed seeds every
# 20 rounds (best.pt by eval blinds). No behavior cloning.
#
#   bash run_v3.sh <shared folder> [hours, default 168] [workers per arm, default 6]
#
# Results in <shared>/runs/v3 and <shared>/runs/v3-control, each with its own train.log. ARMS="v3"
# runs one arm only (after the control is stopped, a relaunch resumes v3 from its latest.pt).
# Stop one arm: tools\balatro.ps1 stop -Run v3-control (it saves latest.pt).
# The A/B checks (analysis/compare_models.py) are run by hand; the rule is in docs/hypotheses.md.
set -uo pipefail
SHARED="${1:?usage: run_v3.sh <shared folder> [hours] [workers per arm]}"
HOURS="${2:-168}"
WORKERS="${3:-6}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
. "$W/venv/bin/activate"

echo "== v3 pre-flight: the game-time discount on the real game"
python3 "$R/tests/v3_check.py" "$W/local/source" --routes 12 | tail -1 | grep -q "ALL PASS" \
  && echo "blind clock, skips, shaping and returns: PASS" || { echo "PRE-FLIGHT FAILED: tests/v3_check.py (run it to see why)"; exit 1; }

COMMON="--seed 3 --explore-hands 0.05 --eval-games 600 --eval-every 20"
export ENC=2 ACTIONS=ordered-capped LR="${LR:-1e-4}" FROM_SCRATCH=1 RUN_V2_UPDATED=1
pids=()
for ARM in ${ARMS:-v3 control}; do
  case "$ARM" in
    v3)      RUN=v3;         CLOCK="--gamma-blind 0.95 --lam-blind 0.95" ;;
    control) RUN=v3-control; CLOCK="--gamma 0.995 --lam 0.95" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  echo "== starting $RUN ($CLOCK), $WORKERS workers, log: $SHARED/runs/$RUN/train.log"
  RUN="$RUN" TRAIN_ARGS="$COMMON $CLOCK ${TRAIN_ARGS:-}" bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS" > /dev/null 2>&1 &
  pids+=($!)
  sleep 20          # stagger: the two pre-flights and worker pools don't start at once
done
echo "== running (pids ${pids[*]}); watch with: tail -f $SHARED/runs/v3/train.log"
wait
echo "== both arms exited $(date)"
