#!/usr/bin/env bash
# Model v6, step 3 (docs/hypotheses-v6.md, 2026-11-21 23:42 section): DAgger round 1, then its attribution twin.
#   1. train/dagger_round.sh 1   v6e plays 600 fresh seeds, the teacher labels; training from v6e on teach_v6 +
#                                dagger/d1; the 600-game eval; temperature 0 -> dagger/r1_t0.jsonl
#   2. v6dt                      the same training without dagger/d1; temperature 0 -> dagger/twin_t0.jsonl
#
#   wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step3.sh"
#
# Re-running resumes: finished parts are skipped. Options (environment): WORK, LAB, SRC, WORKERS, BALATRO_HOME.
set -uo pipefail
WORK="${WORK:-/mnt/c/Users/Jacobi/balatro-work}"
SRC="${SRC:-$HOME/balatro/local/source}"
WORKERS="${WORKERS:-12}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R"
git pull -q --ff-only || echo "git pull failed; running the code already here"
. "${BALATRO_HOME:-$HOME/balatro}/venv/bin/activate"
LOG="$WORK/step3.txt"
mkdir -p "$WORK/dagger"
[ -f "$WORK/runs/v6e/best.pt" ] || { echo "no v6e at $WORK/runs/v6e/best.pt"; exit 1; }
echo "== step 3 $(date '+%F %T')  $(git log -1 --format='%h %s' | cut -c1-80)" | tee -a "$LOG"

bash train/dagger_round.sh 1 || { echo "== DAgger round 1 not finished; re-run to resume" | tee -a "$LOG"; exit 1; }

if grep -q '^\[eval' "$WORK/runs/v6dt/train.log" 2>/dev/null; then
  echo "== v6dt: trained already" | tee -a "$LOG"
else
  echo "== v6dt $(date '+%T')" | tee -a "$LOG"
  mkdir -p "$WORK/runs/v6dt"
  cp -n "$WORK/runs/v6/agreement_set.json" "$WORK/runs/v6dt/"
  bash train/launch.sh v6dt --work "$WORK" --src "$SRC" --workers "$WORKERS"
fi
out="$WORK/dagger/twin_t0.jsonl"
if [ -f "$WORK/runs/v6dt/best.pt" ] && [ ! -s "$out" ]; then
  python3 tools/eval_ladder.py --src "$SRC" --weights "$WORK/runs/v6dt/best.pt" --temp 0 --games 600 \
      --workers "$WORKERS" --out "$out" 2>&1 | tail -n 4 | tee -a "$LOG"
fi

echo "== step 3 summary $(date '+%F %T')" | tee -a "$LOG"
echo "round 1 t1: $(grep -h '^\[eval' "$WORK/runs/v6d-r1/train.log" 2>/dev/null | tail -n 1 | cut -c1-300)" | tee -a "$LOG"
echo "v6dt    t1: $(grep -h '^\[eval' "$WORK/runs/v6dt/train.log" 2>/dev/null | tail -n 1 | cut -c1-300)" | tee -a "$LOG"
for f in r1_t0 twin_t0; do
  [ -s "$WORK/dagger/$f.jsonl" ] && python3 tools/eval_ladder.py --summary "$WORK/dagger/$f.jsonl" | tee -a "$LOG"
done
