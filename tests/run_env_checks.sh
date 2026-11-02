#!/usr/bin/env bash
# Checks env.lua against every fixture (both action modes), then fuzzes both modes.
# usage: tests/run_env_checks.sh <game source dir> [--mods DIR]
#   unmodded: pass the vanilla source; modded: pass <tree>/game and --mods <tree>/Mods
set -uo pipefail
SRC="${1:?usage: tests/run_env_checks.sh <game source dir> [--mods DIR]}"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
H="$HERE/../harness"
fail=0
for log in "$HERE"/fixtures/*.json.gz; do
  for mode in controls subsets; do
    out=$(python3 "$H/check_env.py" "$log" --src "$SRC" "$@" --fast --mode "$mode")
    echo "$(basename "$log") $mode: $(echo "$out" | grep -E '^(controls|subsets) mode:' | cut -d: -f2-) [$(echo "$out" | tail -1)]"
    echo "$out" | tail -1 | grep -q PASS || fail=1
  done
done
for mode in controls subsets; do
  out=$(cd "$H" && python3 fuzz.py --src "$SRC" --runs 30 --mode "$mode")
  echo "fuzz $mode: $(echo "$out" | grep -E '^failures:') $(echo "$out" | grep -E '^no-op actions:')"
  echo "$out" | grep -q "^failures: none" && echo "$out" | grep -q "^no-op actions: none" || fail=1
done
[ $fail -eq 0 ] && echo "ALL ENV CHECKS PASS" || { echo "ENV CHECKS FAILED"; exit 1; }
