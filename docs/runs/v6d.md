# v6d: model v6, stage 2 (DAgger rounds)

**What it is.** Stage 2 of model v6 ([registration](../hypotheses-v6.md), sections of 2026-11-16 and 2026-11-19):
the student from stage 1 plays fresh seeds while Balatro Lab's bot, through the bridge, records its own move for
every state the student reaches; then the student is trained again on every teacher game so far, from its own
weights, and scored on the 600 eval seeds with agreement. Rounds stop when eval wins fail to rise two rounds in a
row.

**Status:** round 1 ran 2026-11-22 (from v6e, 600 games, by `train/step3.sh`); no round 2.

**Result (2026-11-22):** round 1 kept pass 6 of 8; 183/600 at temperature 1, **244/600 (40.7%) at temperature 0**;
agreement 76.3% (hand 58.6%, shop 82.9%). Its twin [v6dt](v6dt.md) won 273 at temperature 0 (paired z = -2.23
against round 1): D5, D6 and D7 missed. Scored in docs/hypotheses-v6.md (2026-11-22, after chud's 18:03 CST message).

**A round** (from chud's clone, after stage 1):

    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/dagger_round.sh 1"

Output: `balatro-work/dagger/d<k>/` (the round's games), `runs/v6d-r<k>/` (its training, eval and checkpoints).
