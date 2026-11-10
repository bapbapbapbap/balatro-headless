# Run v2b (models 0.3, 0.3-beta, 1.0)

**What it was.** v2 continued from round 504 with the joker-drag cap (`ordered-capped`) and hand-type
exploration 0.05, lr 1e-5. From the evening of 2026-11-02 to 2026-11-03 (round 810's `best.pt` became 1.0).

**Registered and scored:**
- H1, solved rate vs v2 (chud's): [registered](../hypotheses.md#added-2026-11-02-2055-cst-h-v2b-vs-v2-solved-rate-registered-before-the-window-opens), [scored (pass)](../hypotheses.md#added-2026-11-03-1245-cst-h-result-v2b-vs-v2-solved-rate);
  void as scored, re-scoring registered: [2026-11-03 ~20:50](../hypotheses.md#added-2026-11-03-2050-cst-correction-final-boss-losses-were-counted-as-wins).
- R, 1.0 re-solving the published seeds: [registered](../hypotheses.md#added-2026-11-03-1615-cst-r-re-solve-model-10-on-the-published-won-seeds-registered-after-the-run-started-before-any-of-its-output-was-read), [scored](../hypotheses.md#added-2026-11-03-2015-cst-r-result-re-solve-by-10);
  void as scored, same correction. Data: `analysis/results/resolve_1.0/`.

**Left behind:** `weights/balatro-bot-1.0.pt` (round 810) and `balatro-bot-0.3-beta.pt` (round 550)
([model card](../models.md)); 232 published wins, 362 near misses (counted under 0.3).
