#!/usr/bin/env bash
# Model v4b (docs/hypotheses-v4b.md, registered 2026-11-10 ~10:50 CST before any of its code): v4 from random
# weights with two changes, after v4's diagnosis (docs/hypotheses-v4.md, 2026-11-10 ~10:45):
#   hand clock       --lam-hand 0.8: the table network's GAE lambda also x0.8 at every play or discard (drags,
#                    sells, consumables and every shop-side move leave it unchanged); the shop's stays v4's
#   no exploration   --explore-hands off (its stored log-probs fell to the float32 floor in v4)
#   sells stop       --stop-sells 1.5: stops at the first eval averaging more than 1.5 joker sells per game
# Everything else is run_v4.sh's (see there): phase split, phi_build, offer values, curriculum, shop entropy,
# blind clock, --stall-evals 25, the same four pre-flight checks.
#
#   bash run_v4b.sh <shared folder> [hours, default 168] [workers, default 12]
#   SMOKE=1 bash run_v4b.sh <shared folder>      pre-flight, then 3 rounds into runs/v4b-smoke, then stop
set -uo pipefail
SHARED="${1:?usage: run_v4b.sh <shared folder> [hours] [workers]}"
HOURS="${2:-168}"
WORKERS="${3:-12}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
if [ -z "${RUN_V4_UPDATED:-}" ]; then
  # pull, then run the script as pulled (bash had already read this one before the pull)
  git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
  RUN_V4_UPDATED=1 exec bash "$R/train/run_v4b.sh" "$@"
fi
. "$W/venv/bin/activate"
RUN=v4b
[ -n "${SMOKE:-}" ] && RUN=v4b-smoke
mkdir -p "$SHARED/runs/$RUN"
LOG="$SHARED/runs/$RUN/preflight.log"

if [ -z "${SKIP_V4_CHECK:-}" ]; then
  echo "== v4b pre-flight (log: $LOG)"
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
V4="--seed 3 --lam-hand 0.8 --stop-sells 1.5 --eval-games 600 --eval-every 20 --gamma-blind 0.95 --lam-blind 0.95
    --phase-split --build-shaping 1 --offer-values --curriculum 0.3 --curriculum-published 300
    --curriculum-keep 600 --shop-ent 0.02,0.006@1.5M,0.001@4M --stall-evals 25"
if [ -n "${SMOKE:-}" ]; then
  # 3 rounds with v4b's settings, a small eval, from scratch each time (no resume)
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
