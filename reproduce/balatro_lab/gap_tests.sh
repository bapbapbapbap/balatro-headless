#!/usr/bin/env bash
# The teacher's gap, both laptop measurements one after the other (docs/hypotheses-v6.md, 2026-11-19):
#   1. F1c: the 244 teacher games that hit Cerulean Bell, rerun with fixes F1 + F1b   (~1 h at 12 workers)
#   2. Test P: Balatro Lab's bot in its own simulator on teacher seeds 0-599          (~2.6 h at 12 workers)
# then a paired summary against the bridge's first data run on the same seeds.
#
# Run from chud's reproduce worktree, on branch gap-check, after stage 1 has finished:
#   wsl bash -lc "cd ~/balatro/reproduce && git fetch origin && git checkout gap-check && git pull && bash reproduce/balatro_lab/gap_tests.sh"
# Re-running resumes both: games already in their output files are skipped.
# Options (environment): WORK (balatro-work), LAB (Balatro Lab clone), SRC (game source), WORKERS.
set -uo pipefail
WORK="${WORK:-/mnt/c/Users/Jacobi/balatro-work}"
LAB="${LAB:-$HOME/balatro/balatro-lab}"
SRC="${SRC:-$HOME/balatro/local/source}"
WORKERS="${WORKERS:-12}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
. "$HOME/balatro/venv/bin/activate"
LOG="$WORK/gap_tests.txt"
echo "== gap tests $(date '+%F %T')  $(git log -1 --format='%h %s' | cut -c1-80)" | tee -a "$LOG"

echo "== 1/2 F1c: 244 Bell games with F1 + F1b -> $WORK/bell_f1c" | tee -a "$LOG"
python3 teach.py --lab "$LAB" --src "$SRC" --seed-file results/2026-11-19/bell_seeds.txt --games 244 \
    --workers "$WORKERS" --out "$WORK/bell_f1c" 2>&1 | tee -a "$LOG"
echo "== F1c exit ${PIPESTATUS[0]}" | tee -a "$LOG"

echo "== 2/2 test P: Balatro Lab in its own simulator, teacher seeds 0-599 -> $WORK/lab_on_teach600.jsonl" | tee -a "$LOG"
python3 eval_on_our_seeds.py --lab "$LAB" --seed-set teach --games 600 --workers "$WORKERS" \
    --out "$WORK/lab_on_teach600.jsonl" 2>&1 | tee -a "$LOG"
echo "== test P exit ${PIPESTATUS[0]}" | tee -a "$LOG"

echo "== summary" | tee -a "$LOG"
python3 gap_summary.py "$WORK" 2>&1 | tee -a "$LOG"
echo "== done $(date '+%F %T')" | tee -a "$LOG"
