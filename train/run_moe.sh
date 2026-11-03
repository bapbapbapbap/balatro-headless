#!/usr/bin/env bash
# Joint training (mixture of experts; docs/hypotheses.md section I, chud's design): the early-game
# model plays antes 1-3, the expert for the hand the run leaned on takes over at ante 4, and each
# expert learns from the runs routed to it. Run after the early-game models and the 8 experts.
#   bash run_moe.sh <shared folder> [hours, default 24] [workers, default 14]
# Starts from: the early-game run with the best eval (share reaching ante 4) among runs/early-s*,
# and each runs/expert-<hand>/best.pt (same stake and deck). EARLY=<file> overrides the early model. Router: the hand played most in antes 1-3 relative to the early model's own average;
# 30% of training handoffs go to a random expert (load balancing); pins (Pair 0.5, others 1.0) fall
# to 0 over 100,000 games; the early model is frozen (MOE_TRAIN_EARLY=1 trains it too). lr 1e-5.
set -uo pipefail
SHARED="${1:?usage: run_moe.sh <shared folder> [hours] [workers]}"
HOURS="${2:-24}"
WORKERS="${3:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
. "$W/venv/bin/activate"
SUFFIX=""; [ "${STAKE:-1}" != 1 ] && SUFFIX="$SUFFIX-stake${STAKE}"; [ "${BACK:-b_red}" != b_red ] && SUFFIX="$SUFFIX-${BACK}"
EARLY="${EARLY:-$(python3 - "$SHARED" "$SUFFIX" <<'PY'
import glob, os, sys, torch
shared, suffix = sys.argv[1], sys.argv[2]
best = None
for p in glob.glob(os.path.join(shared, "runs", "early-s*", "best.pt")):
    run = os.path.basename(os.path.dirname(p))
    rest = run.split("-", 2)[2] if run.count("-") >= 2 else ""      # early-s3-stake5 -> "stake5"
    if ("-" + rest if rest else "") != suffix:
        continue
    e = torch.load(p, map_location="cpu", weights_only=False).get("best_eval", -1)
    if best is None or e > best[0]:
        best = (e, p)
print(best[1] if best else "")
PY
)}"
[ -n "$EARLY" ] && [ -f "$EARLY" ] || { echo "no early-game model found (runs/early-s*/best.pt); run run_earlies.sh first"; exit 1; }
echo "early-game model: $EARLY"
EXPERTS=()
for HAND in "Pair" "Two Pair" "Three of a Kind" "Straight" "Flush" "Full House" "Four of a Kind" "Straight Flush"; do
  F="$SHARED/runs/expert-$(echo "$HAND" | tr 'A-Z ' 'a-z-')$SUFFIX/best.pt"
  [ -f "$F" ] || { echo "missing expert: $F (run run_experts.sh first)"; exit 1; }
  EXPERTS+=("$HAND=$F")
done
echo "== pre-flight: MoE routing"
python3 "$R/tests/moe_check.py" --src "$W/local/source" --early "$EARLY" --experts "${EXPERTS[@]}" \
    --stake "${STAKE:-1}" --back "${BACK:-b_red}" | tail -1 | grep -q PASS \
  && echo "MoE routing: PASS" || { echo "PRE-FLIGHT FAILED: MoE (run tests/moe_check.py by hand to see why)"; exit 1; }
export RUN="moe$SUFFIX" ACTIONS=ordered-capped LR="${LR:-1e-5}" RUN_V2_UPDATED=1
export MOE_EARLY="$EARLY" MOE_EXPERT_ARGS="$(printf '%q ' --moe-experts "${EXPERTS[@]}")"
export TRAIN_ARGS="--moe-train-early ${MOE_TRAIN_EARLY:-0} --explore-hands ${EXPLORE:-0.05} ${TRAIN_ARGS:-}"
exec bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS"
