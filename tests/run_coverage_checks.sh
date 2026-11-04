#!/usr/bin/env bash
# Accuracy across every deck, stake and challenge (the plan in docs/hypotheses.md, roadmap):
# the fast sim against the untouched game, with the encoder (v3) running every step, on
#   - every deck at White Stake (1) and Gold Stake (8): RUNS games each (default 2)
#   - every challenge: RUNS games each
# Also starts each deck and challenge once in a plain env and checks the run reports the right
# deck and challenge. Prints one line per check and PASS/FAIL at the end.
#   bash tests/run_coverage_checks.sh ~/balatro/local/source [runs]
set -uo pipefail
SRC="${1:?usage: run_coverage_checks.sh <game source> [runs]}"
RUNS="${2:-2}"
D="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
python3 "$D/tests/start_check.py" --src "$SRC" | tail -1 | grep -q PASS && echo "every deck and challenge starts as itself: PASS" \
  || { echo "decks/challenges start check: FAIL (run tests/start_check.py to see)"; fail=1; }
for DECK in $(python3 "$D/tests/list_game.py" --src "$SRC" --decks); do
  for STAKE in 1 8; do
    if ( cd "$D/harness" && python3 diff_check.py --src "$SRC" --runs "$RUNS" --mode ordered --encode --enc 3 \
         --back "$DECK" --stake "$STAKE" | tail -1 | grep -q PASS ); then echo "$DECK stake $STAKE: PASS"
    else echo "$DECK stake $STAKE: FAIL"; fail=1; fi
  done
done
for CH in $(python3 "$D/tests/list_game.py" --src "$SRC" --challenges); do
  if ( cd "$D/harness" && python3 diff_check.py --src "$SRC" --runs "$RUNS" --mode ordered --encode --enc 3 \
       --challenge "$CH" | tail -1 | grep -q PASS ); then echo "challenge $CH: PASS"
  else echo "challenge $CH: FAIL"; fail=1; fi
done
[ "$fail" = 0 ] && echo PASS || echo FAIL
