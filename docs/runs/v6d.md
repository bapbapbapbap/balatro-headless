# v6d: model v6, stage 2 (DAgger rounds)

**What it is.** Stage 2 of model v6 ([registration](../hypotheses-v6.md), sections of 2026-11-16 and 2026-11-19):
the student from stage 1 plays fresh seeds while Balatro Lab's bot, through the bridge, records its own move for
every state the student reaches; then the student is trained again on every teacher game so far, from its own
weights, and scored on the 600 eval seeds with agreement. Rounds stop when eval wins fail to rise two rounds in a
row.

**Status:** built 2026-11-19 (branch `gap-check`), not run. Runs only if stage 1's reading calls for it (the rule
in the 2026-11-19 roadmap section).

**A round** (from chud's clone, after stage 1):

    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/dagger_round.sh 1"

Output: `balatro-work/dagger/d<k>/` (the round's games), `runs/v6d-r<k>/` (its training, eval and checkpoints).
