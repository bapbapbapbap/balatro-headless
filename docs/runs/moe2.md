# Run moe2 (moe2-pins0)

**What it was.** The second mixture of experts: shared trunk, early head plus 6 hand experts (no Four of a
Kind), from model 0.2's weights. An early-game Full House pin, a router minimum of 2 Full House plays, all
pins fading to 0 over 100,000 games. Changes from [moe](moe.md):
[2026-11-04 ~13:40](../hypotheses.md#added-2026-11-04-1340-cst-the-first-moe-run-stopped-changes-for-the-second-section-i). Started 2026-11-04 ~13:30;
stopped itself 2026-11-05 13:42 at pins zero (round 1042, 100,032 games).

**Registered and scored:**
- I2, Full House among the strongest experts: [registered](../hypotheses.md#added-2026-11-04-1335-cst-i2-full-house-is-one-of-the-strongest-experts-section-i); closed unscored
  ([2026-11-09 ~18:17](../hypotheses.md#added-2026-11-09-1817-cst-the-full-house-venture-closed-and-the-bar-for-going-up-the-decks-and-stakes)).
- Full House completion criterion: [set](../hypotheses.md#added-2026-11-04-1340-cst-completion-criterion-the-moe-has-learned-full-house), [not met](../hypotheses.md#added-2026-11-05-1540-cst-moe2-at-pins-zero-results);
  closed ([2026-11-09 ~18:17](../hypotheses.md#added-2026-11-09-1817-cst-the-full-house-venture-closed-and-the-bar-for-going-up-the-decks-and-stakes)). Review of the pin: [2026-11-04 ~13:50](../hypotheses.md#added-2026-11-04-1350-cst-review-of-the-early-full-house-pin-for-a-critical-reader).
- Pins-zero results: [2026-11-05 ~15:40](../hypotheses.md#added-2026-11-05-1540-cst-moe2-at-pins-zero-results); 9 of 600 eval wins, 12.94 blinds per game.
- Paired experts from the same positions (C1–C5, 2,000 seeds): [predictions](../hypotheses.md#added-2026-11-05-1545-cst-claudes-predictions-for-the-first-paired-run-section-j),
  [size](../hypotheses.md#added-2026-11-05-1545-cst-the-first-paired-runs-size-2000-seeds-and-its-pilot), [scored](../hypotheses.md#added-2026-11-05-2110-cst-the-first-paired-run-scored-section-j). Data: `paired_moe2.jsonl`
  (chud's work folder), scored by `analysis/score_paired.py`.
- What it says about aim 1: [2026-11-05 ~21:15](../hypotheses.md#added-2026-11-05-2115-cst-what-moe2-says-about-aim-1-chuds-reading-with-limits).

**Left behind:** `weights/balatro-bot-moe2-pins0.pt` ([system card](../system_card_moe2.md),
[evaluation](../moe2_evaluation.md)), the baseline [v4](v4.md) is tested against; 790 published wins
(732 training, 58 eval), 712 near misses.
