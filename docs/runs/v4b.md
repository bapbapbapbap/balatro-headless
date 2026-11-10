# Run v4b (current)

**What it is.** v4 from random weights with two changes: a hand clock on the table network's GAE lambda
(`--lam-hand 0.8`: x0.8 per play or discard), and `--explore-hands` off. Stops itself at the first eval with more
than 1.5 joker sells per game (`--stop-sells 1.5`), or on v4's stall rule.

**Registered:** [design and predictions S1-S3](../hypotheses-v4b.md#added-2026-11-10-1030-cst-model-v4b-registered-before-any-of-its-code-section-j),
after [v4's diagnosis](../hypotheses-v4.md#added-2026-11-10-1030-cst-why-v4-stalled-diagnosis-after-the-run-section-j).
Code on branch `v4b`; output in `runs/v4b/`.

**The run.** Smoke test passed 2026-11-10 10:39; training started 10:40:47 CST on chud's laptop from commit 8a186ae
(`run_v4b.sh` from 64dcf04 plus the watcher's `-Run v4b`), 12 workers, 168-hour cap.

**Paused and moved to the new code.** The laptop rebooted at about 11:19 on 2026-11-10 with v4b at round 70 (eval
4.08 blinds per game at round 60, 0.49 joker sells per game; `latest.pt` saved). Before it resumed, the trainer was
reorganized: `train.py` split by capability and every per-run script replaced by `train/launch.sh` + `configs/v4b.toml`
(chud's call, 11:46: resume on the new code). Evidence that nothing it does changed:
`tests/compare_trainers.py` on chud's laptop trained v4b's launch commit (8a186ae) twice and the split trainer
(3296286) once, 3 rounds each on the CPU with v4b's settings: all 24 saved games and all 154 weight tensors identical;
and `launch.py v4b --check-provenance runs/v4b/provenance.json` gives SAME SETTINGS. It resumes from round 70 on main.
