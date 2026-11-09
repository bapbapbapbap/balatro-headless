#!/usr/bin/env bash
# Model v4 (docs/hypotheses.md: registered 2026-11-05 ~23:01 CST, amended 2026-11-09 ~12:00, definitions fixed
# before training): the strongest bot buildable from what this project and others found.
#   phase split      --phase-split: table and shop networks, each 1.0's size with its own critic; credit
#                    routed per network (train/credit.py; arXiv 2610.39049's routing)
#   build potential  --build-shaping 1: phi_build (harness/build_phi.lua), scaling jokers projected 6 rounds
#   offer values     --offer-values: each joker/planet/voucher offer's row carries phi_build after taking it
#   curriculum       --curriculum 0.3: 30% of training games start at the first shop of ante 4, 5 or 6 of a
#                    saved game, replayed from its seed (300 published wins + this run's own games)
#   shop entropy     0.02 -> 0.006 at 1.5M training decisions -> 0.001 at 4M (arXiv 2610.39049); table 0.01
#   clock            the blind clock: --gamma-blind 0.95 --lam-blind 0.95
#   stall stop       12 h (not 3): a week-long run from random weights has flat stretches
# Held from 1.0: d 192, 4 layers, 6 heads (per network), encoding 2, ordered-capped actions, reward blinds,
# hand-type exploration 0.05, seed retries, lr 1e-4, from random weights, no behavior cloning, Red Deck,
# White Stake, eval on the 600 fixed seeds every 20 rounds (best.pt by eval blinds).
#
#   bash run_v4.sh <shared folder> [hours, default 168] [workers, default 12]
#   SMOKE=1 bash run_v4.sh <shared folder>       pre-flight, then 3 rounds into runs/v4-smoke, then stop
#
# Results in <shared>/runs/v4 (train.log, metrics.jsonl with play_seconds and update_seconds for W4b).
# Curriculum games train but are in no stat; their wins go to curriculum_wins.jsonl, never wins.jsonl.
# Stop with Ctrl+C (saves latest.pt); a relaunch resumes from it.
set -uo pipefail
SHARED="${1:?usage: run_v4.sh <shared folder> [hours] [workers]}"
HOURS="${2:-168}"
WORKERS="${3:-12}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
. "$W/venv/bin/activate"
RUN=v4
[ -n "${SMOKE:-}" ] && RUN=v4-smoke
mkdir -p "$SHARED/runs/$RUN"
LOG="$SHARED/runs/$RUN/preflight.log"

if [ -z "${SKIP_V4_CHECK:-}" ]; then
  echo "== v4 pre-flight (log: $LOG)"
  : > "$LOG"
  check() {     # name, command...
    local name="$1"; shift
    if "$@" >> "$LOG" 2>&1 && tail -1 "$LOG" | grep -q "ALL PASS"; then echo "$name: PASS"
    else echo "PRE-FLIGHT FAILED: $name (see $LOG)"; exit 1; fi
  }
  check "routed credit" python3 "$R/tests/v4_credit_check.py"
  check "phase split: routing, replay, one network per decision, PPO" \
        python3 "$R/tests/v4_phase_check.py" "$W/local/source" --games 4 --routes 3
  check "phi_build and offer values: game unchanged, cache = fresh, offers = change" \
        python3 "$R/tests/v4_phi_check.py" "$W/local/source" --routes "${CHECK_ROUTES:-3}"
  check "curriculum: starts at the first shop of ante 4-6, own starts exact" \
        python3 "$R/tests/v4_curriculum_check.py" "$W/local/source" --routes "${CHECK_ROUTES:-3}"
  grep "sim cost per move\|replay cost per start" "$LOG"
fi

export ENC=2 ACTIONS=ordered-capped LR="${LR:-1e-4}" FROM_SCRATCH=1 RUN_V2_UPDATED=1 RUN
V4="--seed 3 --explore-hands 0.05 --eval-games 600 --eval-every 20 --gamma-blind 0.95 --lam-blind 0.95
    --phase-split --build-shaping 1 --offer-values --curriculum 0.3 --curriculum-published 300
    --curriculum-keep 600 --shop-ent 0.02,0.006@1.5M,0.001@4M --stall-hours 12"
if [ -n "${SMOKE:-}" ]; then
  # 3 rounds with v4's settings, a small eval, from scratch each time (no resume)
  rm -f "$SHARED/runs/$RUN/latest.pt"
  V4="$V4 --iterations 3 --eval-games 12 --eval-every 2 --games-per-worker 2"
  HOURS=1
fi
echo "== starting $RUN, $WORKERS workers, log: $SHARED/runs/$RUN/train.log"
TRAIN_ARGS="$(echo $V4) ${TRAIN_ARGS:-}" bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS"
if [ -n "${SMOKE:-}" ]; then
  echo "== smoke run done; last rounds:"
  grep "^\[it\|^\[eval" "$SHARED/runs/$RUN/train.log" | tail -4
fi
