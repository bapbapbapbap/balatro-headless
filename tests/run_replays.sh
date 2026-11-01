#!/usr/bin/env bash
# Replays every fixture against a modded game tree built by harness/build_modded_tree.sh.
# usage: tests/run_replays.sh <modded tree dir>   (contains game/ and Mods/)
set -uo pipefail
TREE="${1:?usage: tests/run_replays.sh <modded tree dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
fail=0
for log in "$HERE"/fixtures/*.json.gz; do
  out=$(python3 "$HERE/../harness/replay.py" "$log" --src "$TREE/game" --mods "$TREE/Mods" --fast --quiet)
  echo "$out" | tail -6
  echo "$out" | grep -q "mismatching nodes: 0/" && ! echo "$out" | grep -q "stopped at node" || fail=1
  echo
done
[ $fail -eq 0 ] && echo "ALL FIXTURES MATCH" || { echo "MISMATCH"; exit 1; }
