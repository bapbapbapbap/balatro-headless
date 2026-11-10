#!/usr/bin/env bash
# Starts or resumes a run from configs/<run>.toml (see train/launch.py for everything it does).
#   bash train/launch.sh <run> --work /mnt/c/Users/<you>/balatro-work [--smoke] [--hours H] [--workers N]
# Pulls the branch this clone is on first, then runs the launcher as pulled, in the venv.
set -uo pipefail
W="${BALATRO_HOME:-$HOME/balatro}"
R="$W/balatro-headless"
if [ -z "${LAUNCH_UPDATED:-}" ]; then
  git -C "$R" pull -q --ff-only || echo "git pull failed; running the code already here"
  LAUNCH_UPDATED=1 exec bash "$R/train/launch.sh" "$@"
fi
. "$W/venv/bin/activate"
exec python3 "$R/train/launch.py" "$@"
