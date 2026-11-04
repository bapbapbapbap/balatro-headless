#!/usr/bin/env bash
# The mixture of experts (docs/hypotheses.md section I, chud's design), trained jointly from the
# start: one shared trunk and 7 heads (the early-game model, then one expert for each of 6 hands:
# Pair, Two Pair, Three of a Kind, Straight, Flush, Full House; no Four of a Kind, chud 2026-11-04). The early
# head plays antes 1-3; at ante 4 the expert for the hand the run leaned on takes over. Every game
# trains the trunk; each head learns only from its own decisions. The early head's returns end at
# the handoff, so it is judged on reaching ante 4.
#   bash run_moe.sh <shared folder> [hours, default 24] [workers, default 14]
# Starts from model 0.2's weights (trunk and every head), unless solo runs exist: the best
# runs/early-s*/best.pt (share reaching ante 4) gives the trunk and early head, and each
# runs/expert-<hand>/best.pt its expert head. Same stake and deck as the run (STAKE, BACK).
# Router: the hand played most in antes 1-3 relative to the early head's own average; 30% of
# training handoffs go to a random expert. Pins (Pair 0.5, others 1.0) fall to 0 over 100,000
# games, and not before each expert's own 5,000 (MOE_MIN_GAMES). The early head has its own pin,
# MOE_EARLY_PIN (default "Full House=1.0": a Full House or higher when one is on offer), fading over
# the same games; MOE_EARLY_PIN=none turns it off. The router needs MOE_MIN_PLAYS plays of a hand in
# antes 1-3 to pick its expert (default "Full House=2"). The stall stop waits until every pin is 0.
# MOE_RUN=<name> names the run folder (default moe; a fresh start needs a new name). Encoding 3. Evals on 600 seeds every 20 rounds. MOE_SEED=<n> grows another one
# (runs/moe-s<n>); MOE_TRAIN_EARLY=0 leaves the early head as loaded; MOE_CUT=0 lets it learn
# from the whole run.
set -uo pipefail
SHARED="${1:?usage: run_moe.sh <shared folder> [hours] [workers]}"
HOURS="${2:-24}"
WORKERS="${3:-14}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
. "$W/venv/bin/activate"
SUFFIX=""; [ "${STAKE:-1}" != 1 ] && SUFFIX="$SUFFIX-stake${STAKE}"; [ "${BACK:-b_red}" != b_red ] && SUFFIX="$SUFFIX-${BACK}"
INIT="${MOE_INIT:-$R/weights/balatro-bot-0.2.pt}"
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
echo "trunk and early head from: ${EARLY:-$INIT}"
EXPERTS=()
for HAND in "Pair" "Two Pair" "Three of a Kind" "Straight" "Flush" "Full House"; do
  F="$SHARED/runs/expert-$(echo "$HAND" | tr 'A-Z ' 'a-z-')$SUFFIX/best.pt"
  [ -f "$F" ] && EXPERTS+=("$HAND=$F")
done
echo "expert heads from solo runs: ${#EXPERTS[@]} of 6 (the rest from $(basename "$INIT"))"
EARLY_PIN="${MOE_EARLY_PIN:-Full House=1.0}"; [ "$EARLY_PIN" = none ] && EARLY_PIN=""
MIN_PLAYS="${MOE_MIN_PLAYS:-Full House=2}"
MP=(); [ -n "$MIN_PLAYS" ] && IFS='|' read -r -a MP <<< "$MIN_PLAYS"      # several: "Full House=2|Flush=1"
echo "early pin: ${EARLY_PIN:-off}; router minimum plays: ${MIN_PLAYS:-none}"
CHK=(--init "$INIT" --early-pin "$EARLY_PIN" --min-plays "${MP[@]}"); [ -n "$EARLY" ] && CHK+=(--early "$EARLY"); [ ${#EXPERTS[@]} -gt 0 ] && CHK+=(--experts "${EXPERTS[@]}")
echo "== pre-flight: MoE routing"
python3 "$R/tests/moe_check.py" --src "$W/local/source" "${CHK[@]}" --enc "${ENC:-3}" \
    --stake "${STAKE:-1}" --back "${BACK:-b_red}" | tail -1 | grep -q PASS \
  && echo "MoE routing: PASS" || { echo "PRE-FLIGHT FAILED: MoE (run tests/moe_check.py by hand to see why)"; exit 1; }
export RUN="${MOE_RUN:-moe}${MOE_SEED:+-s$MOE_SEED}$SUFFIX" ACTIONS=ordered-capped LR="${LR:-1e-5}" ENC="${ENC:-3}" RUN_V2_UPDATED=1
export MOE_ON=1 MOE_INIT="$INIT" MOE_EARLY="$EARLY" MOE_EXPERT_ARGS=""
[ ${#EXPERTS[@]} -gt 0 ] && export MOE_EXPERT_ARGS="$(printf '%q ' --moe-experts "${EXPERTS[@]}")"
[ -n "$EARLY_PIN" ] && MOE_EXPERT_ARGS+=" $(printf '%q ' --moe-early-pin "$EARLY_PIN")"
[ -n "$MIN_PLAYS" ] && MOE_EXPERT_ARGS+=" $(printf '%q ' --moe-min-plays "${MP[@]}")"
export MOE_EXPERT_ARGS
export TRAIN_ARGS="--seed ${MOE_SEED:-0} --moe-train-early ${MOE_TRAIN_EARLY:-1} --moe-cut ${MOE_CUT:-1} --moe-min-games ${MOE_MIN_GAMES:-5000} --eval-games ${EVAL_GAMES:-600} --eval-every ${EVAL_EVERY:-20} --explore-hands ${EXPLORE:-0.05} ${TRAIN_ARGS:-}"
exec bash "$R/train/run_v2.sh" "$SHARED" "$HOURS" "$WORKERS"
