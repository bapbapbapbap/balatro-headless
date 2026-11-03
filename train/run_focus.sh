#!/usr/bin/env bash
# Trains a hand-focus bot: model 1.0 continued with the hand-focus rule on for one poker hand
# (env.lua, "hand focus": when a play of that hand is available, plays of other hands aren't
# offered). Same reward as 1.0 (blinds beaten), same model and moves otherwise, so the bot works
# out on its own how to get the most out of that hand (discards, planets, jokers, shop).
#
#   bash run_focus.sh <shared folder> "<hand>" [hours, default 12] [workers, default 14]
#   bash run_focus.sh <shared folder> all [hours per hand] [workers]    # the 9 hands, one after another
#
# <hand> as the game names it: "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
# "Flush", "Full House", "Four of a Kind", "Straight Flush" (also "Five of a Kind", "Flush House",
# "Flush Five", which "all" skips: Red Deck can't make them without changing the deck, so those
# bots would be ~1.0 unrestricted).
# Results in <shared folder>/runs/focus-<hand>. Launching again resumes from its latest.pt; the
# first launch starts from weights/balatro-bot-1.0.pt (model only, fresh optimizer).
# Env: LR (default 1e-5, as 0.3 ended), EXPLORE (default 0.05, as 0.3), TRAIN_ARGS (extra args).
set -uo pipefail
SHARED="${1:?usage: run_focus.sh <shared folder> \"<hand>\"|all [hours] [workers]}"
HAND="${2:?usage: run_focus.sh <shared folder> \"<hand>\"|all [hours] [workers]}"
HOURS="${3:-12}"
WORKERS="${4:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"

if [ "$HAND" = all ]; then
  for h in "High Card" "Pair" "Two Pair" "Three of a Kind" "Straight" "Flush" "Full House" \
           "Four of a Kind" "Straight Flush"; do
    bash "$R/train/run_focus.sh" "$SHARED" "$h" "$HOURS" "$WORKERS"
  done
  exit 0
fi

SLUG="$(echo "$HAND" | tr 'A-Z ' 'a-z-')"
RUN="focus-$SLUG"
OUT="$SHARED/runs/$RUN"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/train.log") 2>&1
echo "== run_focus $HAND $(date)"
. "$W/venv/bin/activate"
git -C "$R" log --oneline -1

echo "== pre-flight: hand-focus rule ($HAND)"
python3 "$R/tests/focus_check.py" "$W/local/source" --modes ordered --seeds 2 --hands "$HAND" | tail -1 | grep -q "ALL PASS" \
  && echo "hand-focus rule does what it says: PASS" || { echo "PRE-FLIGHT FAILED: hand-focus rule"; exit 1; }

if [ -f "$OUT/latest.pt" ]; then
  START=(--resume "$OUT/latest.pt"); echo "resuming from $OUT/latest.pt"
else
  START=(--init "$R/weights/balatro-bot-1.0.pt"); echo "starting $RUN from model 1.0"
fi

echo "== training"
cd "$R/train"
python3 train.py --src "$W/local/source" --run "$RUN" --out "$OUT" --hours "$HOURS" --workers "$WORKERS" \
  --enc 2 --actions ordered-capped --d 192 --layers 4 --heads 6 --lr "${LR:-1e-5}" \
  --reward blinds --back b_red --stake 1 --focus "$HAND" --explore-hands "${EXPLORE:-0.05}" \
  --mirror-dir "${MIRROR_DIR:-/mnt/f/balatro data/$RUN}" --spill-dir "${SPILL_DIR:-/mnt/f/balatro data}" \
  "${START[@]}" ${TRAIN_ARGS:-}
echo "== exited $(date)"
