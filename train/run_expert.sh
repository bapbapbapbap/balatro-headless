#!/usr/bin/env bash
# Trains one strategy expert (docs/hypotheses.md section I, chud's design): model 0.2's weights and
# architecture, pinned to one poker hand type (train.py apply_pin), reward `blinds` as before.
#   bash run_expert.sh <shared folder> "<hand>" [hours, default 12] [workers, default 14]
#   hand: Pair, "Two Pair", "Three of a Kind", Straight, Flush, "Full House", "Four of a Kind",
#         "Straight Flush" (the game's names; no High Card, no secret hands)
# Pin rate: Pair 0.5 (chud: "half encouraged"), every other hand 1.0. PIN_RATE overrides.
# Same settings as v2b otherwise: ordered-capped actions, hand-type exploration 0.05 (never around
# the pin), lr 1e-5, Red Deck, White Stake, unmodded. Results in <shared folder>/runs/expert-<hand>.
# A relaunch resumes from that run's latest.pt. STAKE=1..8 / BACK=<deck> change the stake / deck
# (default White Stake, Red Deck; higher stakes are not covered by the differential check yet).
set -uo pipefail
SHARED="${1:?usage: run_expert.sh <shared folder> <hand> [hours] [workers]}"
HAND="${2:?usage: run_expert.sh <shared folder> <hand> [hours] [workers]}"
HOURS="${3:-12}"
WORKERS="${4:-14}"
case "$HAND" in
  "Pair") RATE_DEFAULT=0.5 ;;
  "Two Pair"|"Three of a Kind"|"Straight"|"Flush"|"Full House"|"Four of a Kind"|"Straight Flush") RATE_DEFAULT=1.0 ;;
  *) echo "unknown or excluded hand: '$HAND'"; exit 1 ;;
esac
RATE="${PIN_RATE:-$RATE_DEFAULT}"
SLUG="$(echo "$HAND" | tr 'A-Z ' 'a-z-')"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
. "$W/venv/bin/activate"
echo "== pre-flight: hand pin (Pair at 1.0 binds often; then $HAND at $RATE)"
python3 "$R/tests/pin_check.py" "$R/weights/balatro-bot-0.2.pt" --src "$W/local/source" --stake "${STAKE:-1}" --back "${BACK:-b_red}" --hand Pair --rate 1.0 | tail -1 | grep -q PASS \
  && python3 "$R/tests/pin_check.py" "$R/weights/balatro-bot-0.2.pt" --src "$W/local/source" --stake "${STAKE:-1}" --back "${BACK:-b_red}" --hand "$HAND" --rate "$RATE" --games 2 --allow-unbound | tail -1 | grep -q PASS \
  && echo "hand pin: PASS" || { echo "PRE-FLIGHT FAILED: hand pin (run tests/pin_check.py by hand to see why)"; exit 1; }
SUFFIX=""; [ "${STAKE:-1}" != 1 ] && SUFFIX="$SUFFIX-stake${STAKE}"; [ "${BACK:-b_red}" != b_red ] && SUFFIX="$SUFFIX-${BACK}"
export ENC="${ENC:-3}" RUN="expert-$SLUG$SUFFIX" ACTIONS=ordered-capped LR="${LR:-1e-5}" INIT_WEIGHTS="$R/weights/balatro-bot-0.2.pt" RUN_V2_UPDATED=1
export PIN_HAND="$HAND" PIN_RATE="$RATE" TRAIN_ARGS="--explore-hands ${EXPLORE:-0.05} ${TRAIN_ARGS:-}"
exec bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS"
