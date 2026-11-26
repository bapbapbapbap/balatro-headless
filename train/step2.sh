#!/usr/bin/env bash
# Model v6, step 2 (docs/hypotheses-v6.md, 2026-11-21 09:38 section): the student on all 3,599 teacher games.
#   v6e   stage 1's network from random weights, up to 16 passes (patience 2)
#   v6eb  the same with d 256, 6 layers, 8 heads (bigger-network arm)
# Each: the launcher's 600-game eval (temperature 1, the registered score), then tools/eval_ladder.py at
# temperature 0 (the delivery score) -> balatro-work/step2/<run>_t0.jsonl. Log: balatro-work/step2.txt.
#
#   wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step2.sh"
#
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
LOG="$WORK/step2.txt"
mkdir -p "$WORK/step2"
STAGE1="$WORK/runs/v6"
[ -f "$STAGE1/agreement_set.json" ] || { echo "no stage 1 agreement set at $STAGE1"; exit 1; }
n=$(cat "$WORK"/teach_v6/games_w*.jsonl | grep -vc '"error"')
echo "== step 2 $(date '+%F %T')  $(git log -1 --format='%h %s' | cut -c1-80)  teacher games: $n" | tee -a "$LOG"

for run in v6e v6eb; do
  if grep -q '^\[eval' "$WORK/runs/$run/train.log" 2>/dev/null; then
    echo "== $run: trained already" | tee -a "$LOG"
  else
    echo "== $run $(date '+%T')" | tee -a "$LOG"
    mkdir -p "$WORK/runs/$run"
    cp -n "$STAGE1/agreement_set.json" "$WORK/runs/$run/"
    bash train/launch.sh "$run" --work "$WORK" --src "$SRC" --workers "$WORKERS"
    grep -h '^\[imitate\] pass .* done\|^\[eval' "$WORK/runs/$run/train.log" | tail -n 3 | tee -a "$LOG"
  fi
  out="$WORK/step2/${run}_t0.jsonl"
  if [ -f "$WORK/runs/$run/best.pt" ] && [ ! -s "$out" ]; then
    echo "== $run at temperature 0 -> $out" | tee -a "$LOG"
    python3 tools/eval_ladder.py --src "$SRC" --weights "$WORK/runs/$run/best.pt" --temp 0 --games 600 \
        --workers "$WORKERS" --out "$out" 2>&1 | tail -n 4 | tee -a "$LOG"
  fi
done

echo "== step 2 summary $(date '+%F %T')" | tee -a "$LOG"
for run in v6e v6eb; do
  echo "$run t1: $(grep -h '^\[eval' "$WORK/runs/$run/train.log" 2>/dev/null | tail -n 1 | cut -c1-400)" | tee -a "$LOG"
  [ -s "$WORK/step2/${run}_t0.jsonl" ] && python3 tools/eval_ladder.py --summary "$WORK/step2/${run}_t0.jsonl" | tee -a "$LOG"
done
