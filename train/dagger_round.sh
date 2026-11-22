#!/usr/bin/env bash
# Model v6, stage 2: one DAgger round (docs/hypotheses-v6.md, 2026-11-19; docs/runs/v6d.md).
#   bash train/dagger_round.sh <k>          from chud's clone (~/balatro/balatro-headless), after stage 1
# 1. The student (round 1: runs/v6e/best.pt; round k: runs/v6d-r<k-1>/best.pt) plays GAMES (600) fresh teacher
#    seeds, indices 100,000 + GAMES(k-1) on, clear of the teacher runs (re-registered 2026-11-21, 23:42 section);
#    Balatro Lab's bot, through the bridge, records its move for every state.
#    -> balatro-work/dagger/d<k>/   (~2-3 h at 12 workers; re-running resumes)
# 2. Imitation on every teacher game so far, from the student's weights; one 600-game eval with agreement on stage
#    1's agreement set; stop (configs/v6d.toml, through train/launch.sh: pre-flight checks first).
# 3. runs/v6d -> runs/v6d-r<k>, then a temperature-0 eval -> balatro-work/dagger/r<k>_t0.jsonl.
# Re-running the same round resumes where it stopped. Options (environment): WORK, LAB, SRC, WORKERS, GAMES,
# BALATRO_HOME (as launch.sh), LAUNCH_ARGS (extra launch.py options, e.g. --skip-checks for a relaunch).
set -uo pipefail
K="${1:?round number}"
WORK="${WORK:-/mnt/c/Users/Jacobi/balatro-work}"
LAB="${LAB:-$HOME/balatro/balatro-lab}"
SRC="${SRC:-$HOME/balatro/local/source}"
WORKERS="${WORKERS:-12}"
GAMES="${GAMES:-600}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R"
git pull -q --ff-only || echo "git pull failed; running the code already here"
. "${BALATRO_HOME:-$HOME/balatro}/venv/bin/activate"
LOG="$WORK/dagger/round_$K.txt"
mkdir -p "$WORK/dagger"

if [ -e "$WORK/runs/v6d-r$K" ]; then echo "round $K is done already (runs/v6d-r$K)"; exit 0; fi
if [ "$K" -eq 1 ]; then PREV="$WORK/runs/v6e/best.pt"; else PREV="$WORK/runs/v6d-r$((K - 1))/best.pt"; fi
[ -f "$PREV" ] || { echo "no student at $PREV"; exit 1; }
[ -f "$WORK/runs/v6/agreement_set.json" ] || { echo "no stage 1 agreement set at runs/v6/agreement_set.json"; exit 1; }
if [ ! -e "$WORK/runs/v6d" ]; then
  cp "$PREV" "$WORK/dagger/student.pt"
  mkdir -p "$WORK/runs/v6d"
  cp "$WORK/runs/v6/agreement_set.json" "$WORK/runs/v6d/"
fi
echo "== round $K $(date '+%F %T')  student $PREV  $(git log -1 --format='%h %s' | cut -c1-70)" | tee -a "$LOG"

echo "== 1/2 the student plays $GAMES fresh seeds, the teacher labels -> $WORK/dagger/d$K" | tee -a "$LOG"
python3 reproduce/balatro_lab/teach.py --lab "$LAB" --src "$SRC" --student "$WORK/dagger/student.pt" \
    --first $((100000 + GAMES * (K - 1))) --games "$GAMES" --workers "$WORKERS" --out "$WORK/dagger/d$K" 2>&1 \
    | tee -a "$LOG"

echo "== 2/2 imitation on all teacher data from the student's weights, eval -> runs/v6d" | tee -a "$LOG"
# shellcheck disable=SC2086
bash train/launch.sh v6d --work "$WORK" --src "$SRC" --workers "$WORKERS" ${LAUNCH_ARGS:-}
ev="$(grep -h '^\[eval' "$WORK/runs/v6d/train.log" 2>/dev/null | tail -n 1)"
if [ -z "$ev" ] || [ ! -f "$WORK/runs/v6d/best.pt" ]; then
  echo "== round $K: no eval yet; re-run the same command to resume" | tee -a "$LOG"; exit 1
fi
mv "$WORK/runs/v6d" "$WORK/runs/v6d-r$K"
echo "== round $K done $(date '+%F %T'): $ev" | tee -a "$LOG"
out="$WORK/dagger/r${K}_t0.jsonl"
if [ ! -s "$out" ]; then
  echo "== round $K at temperature 0 -> $out" | tee -a "$LOG"
  python3 tools/eval_ladder.py --src "$SRC" --weights "$WORK/runs/v6d-r$K/best.pt" --temp 0 --games 600 \
      --workers "$WORKERS" --out "$out" 2>&1 | tail -n 4 | tee -a "$LOG"
fi
