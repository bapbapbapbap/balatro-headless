#!/usr/bin/env bash
# Trains the eight strategy experts one after another (docs/hypotheses.md section I).
#   bash run_experts.sh <shared folder> [hours each, default 12] [workers, default 14]
# Order: Full House first. An expert that stops early (stall stop, safety stop) is followed by the
# next; relaunching skips nothing, and each run resumes from its own latest.pt.
set -uo pipefail
SHARED="${1:?usage: run_experts.sh <shared folder> [hours each] [workers]}"
HOURS="${2:-12}"
WORKERS="${3:-14}"
D="$(cd "$(dirname "$0")" && pwd)"
for HAND in "Full House" "Flush" "Straight" "Three of a Kind" "Two Pair" "Pair" "Four of a Kind" "Straight Flush"; do
  echo "==== expert: $HAND ($(date))"
  bash "$D/run_expert.sh" "$SHARED" "$HAND" "$HOURS" "$WORKERS" || echo "expert $HAND exited with an error; going on"
done
