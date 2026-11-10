# Run v4b (current)

**What it is.** v4 from random weights with two changes: a hand clock on the table network's GAE lambda
(`--lam-hand 0.8`: x0.8 per play or discard), and `--explore-hands` off. Stops itself at the first eval with more
than 1.5 joker sells per game (`--stop-sells 1.5`), or on v4's stall rule.

**Registered:** [design and predictions S1-S3](../hypotheses-v4b.md#added-2026-11-10-1030-cst-model-v4b-registered-before-any-of-its-code-section-j),
after [v4's diagnosis](../hypotheses-v4.md#added-2026-11-10-1030-cst-why-v4-stalled-diagnosis-after-the-run-section-j).
Code on branch `v4b`; output in `runs/v4b/`.
