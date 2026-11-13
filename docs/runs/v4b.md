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
and `launch.py v4b --check-provenance runs/v4b/provenance.json` gives SAME SETTINGS. It resumed from round 70 on main at 11:51.

**Stopped at round 280.** At 15:58 on 2026-11-10 the sell stop fired: eval 280 averaged 2.56 joker sells per game
(7.20 blinds/game). Best eval 8.31 blinds/game at round 240 (`best.pt`); 5 wins in 26,880 training games.
[S1 failed, S2 and S3 met](../hypotheses-v4b.md#added-2026-11-11-0925-cst-v4b-stopped-at-round-280-on-the-sell-stop-s1-s3-scored). The sells were the shop network's, not the table network's
mid-blind selling the stop was written for: [diagnosis](../hypotheses-v4b.md#added-2026-11-11-0925-cst-why-v4bs-shop-network-sold-and-what-it-cost-diagnosis-after-the-stop).

**Continuing with the sell stop off** (chud's call, 2026-11-11; [registered with T1-T3](../hypotheses-v4b.md#added-2026-11-11-0925-cst-v4b-continues-from-round-280-with-the-sell-stop-off-registered-before-its-config-changes)).
`stop_sells = 0` in `configs/v4b.toml`, nothing else changed. The same launch command resumes it from
`runs/v4b/latest.pt` (round 279; round 280 and its eval run again). It stops on the stall rule: 25 evals with no
new best above 8.31.

**Still training (2026-11-13).** At round 1890: 48.4 h and 181,536 games since the first launch. Best eval 13.40
blinds/game at round 1840 (`best.pt`); evals r1700-1880 average 13.09. T1 not met, T2 met, T3 open:
[scored](../hypotheses-v4b.md#added-2026-11-13-1130-cst-t1-t3-scored). Where its games end, economy or scaling:
[diagnosis while it runs](../hypotheses-v4b.md#added-2026-11-13-1130-cst-where-v4bs-games-end-economy-or-scaling-diagnosis-while-it-runs)
(`analysis/walls.py`, `analysis/walls_report.py`).
