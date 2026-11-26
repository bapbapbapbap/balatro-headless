#!/usr/bin/env bash
# Model v6, step 1 (docs/hypotheses-v6.md, 2026-11-20 after chud's 12:38 CST message), one queue on the laptop:
#   bash train/step1.sh            from chud's clone (~/balatro/balatro-headless), branch v6
# 1. M1: stage 1's best.pt on the 600 eval seeds at eval temperature 1 (must give stage 1's 78/600 again: checks
#    tools/eval_ladder.py), 0 and 0.5 -> balatro-work/step1/m1_t*.jsonl  (~8 min each at 12 workers)
# 2. M2: configs/v6m2.toml (imitation continued from stage 1)              -> runs/v6m2   (~35 min)
# 3. Scaling curve: configs/v6c900.toml, configs/v6c450.toml (from scratch) -> runs/v6c900, runs/v6c450 (~45 min)
# Each run folder gets stage 1's agreement set first, so every agreement number is on the same 2,000 decisions.
# Re-running resumes: finished parts are skipped. Options (environment): WORK, SRC, WORKERS, BALATRO_HOME.
set -uo pipefail
WORK="${WORK:-/mnt/c/Users/Jacobi/balatro-work}"
SRC="${SRC:-$HOME/balatro/local/source}"
WORKERS="${WORKERS:-12}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R"
git pull -q --ff-only || echo "git pull failed; running the code already here"
. "${BALATRO_HOME:-$HOME/balatro}/venv/bin/activate"
LOG="$WORK/step1.txt"
mkdir -p "$WORK/step1"
STAGE1="$WORK/runs/v6"
[ -f "$STAGE1/best.pt" ] && [ -f "$STAGE1/agreement_set.json" ] || { echo "no stage 1 at $STAGE1"; exit 1; }
echo "== step 1 $(date '+%F %T')  $(git log -1 --format='%h %s' | cut -c1-80)" | tee -a "$LOG"

for t in 1 0 0.5; do
  out="$WORK/step1/m1_t$t.jsonl"
  echo "== M1: stage 1 at eval temperature $t -> $out" | tee -a "$LOG"
  python3 tools/eval_ladder.py --src "$SRC" --weights "$STAGE1/best.pt" --temp "$t" --games 600 \
      --workers "$WORKERS" --out "$out" 2>&1 | tail -n 4 | tee -a "$LOG"
done

for run in v6m2 v6c900 v6c450; do
  if grep -q '^\[eval' "$WORK/runs/$run/train.log" 2>/dev/null; then
    echo "== $run: done already" | tee -a "$LOG"; continue
  fi
  echo "== $run $(date '+%T')" | tee -a "$LOG"
  mkdir -p "$WORK/runs/$run"
  cp -n "$STAGE1/agreement_set.json" "$WORK/runs/$run/"
  bash train/launch.sh "$run" --work "$WORK" --src "$SRC" --workers "$WORKERS"
  grep -h '^\[imitate\] pass .* done\|^\[eval' "$WORK/runs/$run/train.log" | tail -n 3 | tee -a "$LOG"
done

echo "== step 1 summary $(date '+%F %T')" | tee -a "$LOG"
for t in 1 0 0.5; do python3 tools/eval_ladder.py --summary "$WORK/step1/m1_t$t.jsonl" | tee -a "$LOG"; done
for run in v6 v6m2 v6c900 v6c450; do
  echo "$run: $(grep -h '^\[eval' "$WORK/runs/$run/train.log" | tail -n 1 | cut -c1-400)" | tee -a "$LOG"
done
