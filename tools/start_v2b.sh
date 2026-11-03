#!/usr/bin/env bash
# Stops v2 cleanly (it saves latest.pt) and starts v2b from that checkpoint: joker-drag cap,
# hand-type exploration, learning rate 1e-5 (train/run_v2b.sh). runs/v2 stays as it is.
#   bash start_v2b.sh
# Env: EXPLORE (default 0.05), WORKERS (default 14)
set -uo pipefail
SHARED=/mnt/c/Users/Jacobi/balatro-work
if pgrep -f "train.py .*--run v2 " >/dev/null; then
  echo "stopping v2 cleanly (it saves latest.pt)..."; pkill -INT -f "train.py .*--run v2 "
  for i in $(seq 1 120); do pgrep -f "train.py .*--run v2 " >/dev/null || break; sleep 1; done
  pgrep -f "train.py .*--run v2 " >/dev/null && { echo "v2 didn't stop (paused console window?). Unpause or close it, then rerun."; exit 1; }
  echo "v2 stopped."
fi
sleep 3
cd ~/balatro/balatro-headless && git pull -q --ff-only || { echo "git pull failed"; exit 1; }
bash train/run_v2b.sh "$SHARED" 168 "${WORKERS:-14}"
