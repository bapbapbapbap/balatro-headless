#!/usr/bin/env bash
# Trains model v2 (encoding 2, ordered actions, 1.6M parameters) on this machine.
#   bash run_v2.sh <shared folder> [hours, default 168] [workers, default 14]
# First launch: converts v1's most recent training games (and every converted win) to v2's action
# list, imitates them (behavior cloning), then continues with RL. Later launches resume from
# runs/v2/latest.pt and skip straight to RL. Results in <shared folder>/runs/v2.
# Env: RUN (run name, default v2), ACTIONS (default ordered), RESUME_FROM (checkpoint to continue
# from when the run has no latest.pt yet, e.g. runs/v2/latest.pt for v2b; see run_v2b.sh), INIT_WEIGHTS
# (a released weights file to start from instead, fresh optimizer; see run_expert.sh), PIN_HAND and
# PIN_RATE (strategy-expert hand pin; see run_expert.sh), FROM_SCRATCH=1 (random weights, no cloning;
# see run_early.sh), MOE_ON, MOE_INIT, MOE_EARLY and MOE_EXPERT_ARGS (joint training; see run_moe.sh), STAKE (1-8, default 1 = White) and BACK (deck, default b_red).
set -uo pipefail
SHARED="${1:?usage: run_v2.sh <shared folder> [hours] [workers]}"
RUN="${RUN:-v2}"
OUT="$SHARED/runs/$RUN"
HOURS="${2:-168}"
WORKERS="${3:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
mkdir -p "$OUT"
if [ -z "${RUN_V2_UPDATED:-}" ]; then
  git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
  RUN_V2_UPDATED=1 exec bash "$R/train/run_v2.sh" "$@"
fi
exec > >(tee -a "$OUT/train.log") 2>&1
echo "== run_v2 $(date)"
. "$W/venv/bin/activate"
git -C "$R" log --oneline -1
python3 -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"

echo "== pre-flight checks"
( cd "$R/train" && python3 verify_games.py --src "$W/local/source" "$R/tests/fixtures/win_v1_r549_1JA35NNR.jsonl" --wins-only --full --fresh-each | tail -1 | grep -q PASS ) \
  && echo "saved win replays exactly (untouched game): PASS" || { echo "PRE-FLIGHT FAILED: saved-game replay"; exit 1; }
python3 "$R/tests/ordered_semantics.py" "$W/local/source" | tail -1 | grep -q "ALL PASS" \
  && echo "ordered actions do what they say (lead, Death, boss reroll): PASS" || { echo "PRE-FLIGHT FAILED: ordered actions"; exit 1; }
( cd "$R/harness" && python3 diff_check.py --src "$W/local/source" --runs 5 --mode ordered --encode --enc 2 | tail -1 | grep -q PASS ) \
  && echo "fast vs untouched game, ordered + encoder v2: PASS" || { echo "PRE-FLIGHT FAILED: differential check"; exit 1; }

RESUME=(); BC=()
if [ -f "$OUT/latest.pt" ]; then
  RESUME=(--resume "$OUT/latest.pt"); echo "resuming from $OUT/latest.pt"
elif [ -n "${RESUME_FROM:-}" ]; then
  RESUME=(--resume "$RESUME_FROM"); echo "starting $RUN from $RESUME_FROM"
elif [ -n "${MOE_ON:-}" ]; then
  echo "starting $RUN as a mixture of experts (members loaded by train.py)"
elif [ -n "${FROM_SCRATCH:-}" ]; then
  echo "starting $RUN from scratch (random weights, no behavior cloning)"
elif [ -n "${INIT_WEIGHTS:-}" ]; then
  RESUME=(--init-weights "$INIT_WEIGHTS"); echo "starting $RUN from released weights $INIT_WEIGHTS"
else
  BCG="$SHARED/v2/bc_games.jsonl"
  if [ ! -s "$BCG" ]; then
    echo "== converting v1's recent training games to v2 (${BC_GAMES:-3000} games, $WORKERS processes)"
    mkdir -p "$SHARED/v2"
    ( cd "$R/train" && python3 convert_routes.py --src "$W/local/source" "$SHARED/runs/v1/games" --out "$BCG" \
        --train-only --max-games "${BC_GAMES:-3000}" --jobs "$WORKERS" | tail -2 )
  fi
  BC=(--bc "$BCG")
  [ -s "$SHARED/v2/wins_v2.jsonl" ] && BC+=("$SHARED/v2/wins_v2.jsonl")
  echo "behavior cloning from: ${BC[*]:1}"
fi

MOE=()
if [ -n "${MOE_ON:-}" ]; then
  MOE=(--moe)
  [ -n "${MOE_INIT:-}" ] && MOE+=(--moe-init "$MOE_INIT")
  [ -n "${MOE_EARLY:-}" ] && MOE+=(--moe-early "$MOE_EARLY")
  eval "MOE+=(${MOE_EXPERT_ARGS:-})"
fi
MIX=()
[ -n "${BACKS:-}" ] && MIX+=(--backs $BACKS)
[ -n "${STAKES:-}" ] && MIX+=(--stakes $STAKES)
[ -n "${CHALLENGES:-}" ] && MIX+=(--challenges $CHALLENGES --challenge-share "${CHALLENGE_SHARE:-0.25}")
[ ${#MIX[@]} -gt 0 ] && echo "mixing per game: ${MIX[*]}"
PIN=()
[ -n "${PIN_HAND:-}" ] && PIN=(--pin-hand "$PIN_HAND" --pin-rate "${PIN_RATE:-1.0}") && echo "hand pin: $PIN_HAND at ${PIN_RATE:-1.0}"

echo "== training"
cd "$R/train"
# No PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: under WSL it crashed moe2 (2026-11-04 19:50,
# "CUDA driver error: device not ready / Failed to create GPU mapping"). Out-of-memory minibatches are
# halved in ppo_update instead.
python3 train.py --src "$W/local/source" --run "$RUN" --out "$OUT" --hours "$HOURS" --workers "$WORKERS" \
  --enc "${ENC:-2}" --actions "${ACTIONS:-ordered}" --d 192 --layers 4 --heads 6 --lr "${LR:-1e-4}" \
  --reward blinds --back "${BACK:-b_red}" --stake "${STAKE:-1}" --max-wins "${MAX_WINS:-1000000}" \
  --mirror-dir "${MIRROR_DIR:-/mnt/f/balatro data/$RUN}" --spill-dir "${SPILL_DIR:-/mnt/f/balatro data}" \
  "${BC[@]}" --bc-epochs "${BC_EPOCHS:-2}" "${RESUME[@]}" "${PIN[@]}" "${MOE[@]}" "${MIX[@]}" ${TRAIN_ARGS:-}
echo "== exited $(date)"
