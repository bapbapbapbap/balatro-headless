#!/usr/bin/env bash
# Trains model v1 overnight on this machine. Run after tools/setup_linux.sh.
#   bash run_v1.sh <shared folder for results> [hours, default 10] [workers, default 12]
# Weights (ckpt_*.pt, latest.pt, final.pt), metrics.jsonl, provenance.json and the log go to
# <shared folder>/runs/v1. Keep this window open and the machine awake until it finishes.
set -uo pipefail
OUT="${1:?usage: run_v1.sh <shared folder> [hours] [workers]}/runs/v1"
HOURS="${2:-10}"
WORKERS="${3:-12}"
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
mkdir -p "$OUT"
. "$W/venv/bin/activate"
git -C "$R" pull -q --ff-only
python3 -c "import torch" 2>/dev/null || pip install -q torch numpy
python3 -c "import numpy" 2>/dev/null || pip install -q numpy
python3 -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"
cd "$R/train"
python3 train.py --src "$W/local/source" --run v1 --out "$OUT" --hours "$HOURS" --workers "$WORKERS" \
  --reward blinds --back b_red --stake 1 2>&1 | tee -a "$OUT/train.log"
