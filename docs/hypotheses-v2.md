# Hypotheses: Run v2 (model 0.2)

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-02 ~10:35 CST: v2 vs v1 (registered before v2's RL phase has produced any result)

Requested by chud: v2 replaces v1 only if it beats v1 on **wins for the compute spent**. v2 was
launched at ~10:20 (`train/run_v2.sh`: conversion of v1's recent games, behavior cloning, then RL).
At registration it had not finished cloning; no v2 RL round, eval or win exists yet.

### What is compared
Same laptop (Core Ultra 9 285H, RTX 40 learner, WSL), 14 workers, Red Deck, White Stake, unmodded
1.0.1o, reward `blinds` + shaping, seed retries on (same settings), self-imitation off. v2 differs
in encoding (v2), action list (ordered), model size (1.6M vs 250k parameters) and its start
(behavior cloning from v1's games instead of v1's weights).

### F1 (primary): new beaten seeds per hour
- **Measure:** seeds beaten by v2 that no run had beaten before (not in v1's beaten set, the A/B
  test runs, or `wins/routes.jsonl`), per hour of RL wall-clock time. Behavior cloning and
  conversion time are not counted (one-time cost); the first hour of RL is warm-up and not counted.
- **Window:** the next 6 hours of v2 RL after warm-up. One analysis at the end of the window, no
  interim stopping on results. If v2 is stopped early, the result is reported as underpowered and
  no claim is made either way.
- **v1 baseline:** 80 new beaten seeds in 1.89 h of RL = **42.3 per hour** (v1 rounds 388–555,
  the segments with seed retries on; from `runs/v1/metrics.jsonl`).
- **Test:** one-sided exact Poisson rate comparison (conditional binomial), v2 rate > v1 rate,
  alpha 0.05.
- **Decision:** v2 passes → v1 is retired as the producer of beaten seeds and v2 continues. v2
  fails → v1 resumes from `runs/v1/latest.pt`; v2's weights and games are kept.
- **Prediction:** v2 passes. Direction only; no effect size predicted.

Note: v2 starts with empty retry pools (v1's near-miss seeds stay with v1), which works against v2
early; that's part of what is compared.

### Reported alongside (not decisive)
- Wins per hour (v1: 52.4/h, same window) and per 1,000 games (v1: 4.73 fresh-seed wins per
  1,000, rounds 494–555).
- Eval blinds/game on the fixed 200 eval seeds: mean of v2's evals in the window vs v1's last six
  evals (rounds 500–550: 11.29, 10.42, 10.89, 10.05, 11.17, 11.40; mean 10.87).
- Games per hour, and model vs sim milliseconds per decision (whether the GPU would help v2).
- v2's level straight after behavior cloning (`[eval 0]`).

---

## Added 2026-11-02 ~11:05 CST: authorship, and v3 (registered before v3 exists)

### Authorship
- **Sections A–F, and every item in the dated sections above:** chud's hypotheses and decisions,
  reviewed by Claude (Claude did the write-up).
- **Section G below:** Claude's hypotheses, reviewed by chud.

The sections above are not edited; this note applies to them as they stand.

### G. v3: the goal as the only ought
**Design.** v3 is v2 with one change: reward `win` instead of `blinds`. Same encoding, model,
action list, sim (unmodded 1.0.1o), deck and stake, seed retries and shaping (potential-based,
so it leaves the optimal policy for either reward unchanged). Both arms start from the same v2
checkpoint (`best.pt` when the F1 window closes) and get the same compute:
- **v3 arm:** `--reward win`, 6 hours of RL.
- **Control arm:** `--reward blinds`, 6 hours of RL.

Run one after the other on the laptop, 14 workers, no other training running. Not started until
v2's F1 analysis is done, so F1's compute conditions are untouched. Starting both arms from one
checkpoint removes the before/after confound A6 had.

**Evaluation.** 2,000 held-out seeds never used in training or eval before, each played once per
arm with seeded sampling (same RNG per seed in both arms). "Behind" positions: decisions where
the frozen starting checkpoint's value estimate is in the bottom 20% of decisions at the same
ante, scored with that one checkpoint in both arms' games.

- **G1. Win rate rises.** The v3 arm wins more of the 2,000 seeds than the control arm.
  Test: one-sided exact McNemar on per-seed outcomes, alpha 0.05. *Falsified* if not significant.
  This is the first direct data on C2 (C2 predicts the opposite direction or a tie).
- **G2. Blinds per game holds.** v3's mean blinds per game is at least the control's minus 1.0.
  *Falsified* if lower by more than 1.0.
- **G3. Hail-mary play when behind (headline).** In behind positions, the v3 arm rerolls more per
  shop visit, skips more blinds, and leaves shops holding less money than the control arm.
  Reason: under `blinds`, a losing run still earns credit for one more blind; under `win`, a run
  that can't win is worth zero whatever it does, so only high-variance lines carry value.
  Test: each statistic paired by seed, one-sided permutation test, Holm correction across the
  three, alpha 0.05. *Falsified* if none of the three moves in the predicted direction
  significantly.
- **G4. Money is spent down late.** At ante 8 shop exits (all positions, not only behind), v3 holds
  less money than the control. Reason: money has no value after the final boss, while `blinds`
  still rewards the interest buffer that buys future blinds. *Falsified* if not lower
  (same test as G3, alone).

Reported alongside (not decisive): blinds per game over the first hours of each arm. A sharp
early drop in the v3 arm would mean the win signal is too sparse at v2's win rate, not that the
reward is wrong; if it happens, it's reported as such and G1–G4 are still scored at 6 hours.

### Clarification (2026-11-02 ~11:05 CST, requested by chud)
The idea behind this project is chud's: exhibiting Hume's is-ought gap with a Balatro bot, by
holding the facts fixed (the pinned game, same seeds, same sim) and varying only the ought (the
reward). The framing of oughts as things to strategize over, like jokers, is also his. Section C
and section G both apply that idea. Section G's specific predictions (G1–G4) are Claude's; the
question they test is chud's.

---

## Added 2026-11-02 ~18:25 CST: F1 result (v2 vs v1)

Scored once, at the end of the registered window, as registered. Nothing above is edited.

### Window
v2 RL began at iteration 0 (~11:17 CST, after behavior cloning). Warm-up: RL hours 0-1. Window:
RL hours 1.0-7.0 (~12:17-18:17 CST), iterations 71-438. Settings were unchanged for the whole
window (lr 5e-5 throughout, after the one automatic halving at iteration 6, which is in warm-up).
Hours are the trainer's own clock (`hours` in `runs/v2/metrics.jsonl`, minus its value at
iteration 0).

### F1 (primary): new beaten seeds per hour — **PASS**
- **New beaten seeds in the window:** 545. Training wins only (evaluation seeds excluded); a seed
  counts once, at its first v2 win; "new" excludes every seed in v1's `wins.jsonl`, the A/B runs
  (`ab/`, `ab2/`), `test-retry/` and `wins/routes.jsonl` (142 seeds in all, eval-won included).
- **Rate:** 545 / 6.0 h = **90.8 per hour**, vs v1's 42.3 per hour (80 in 1.89 h).
  Rate ratio 2.15 (approx. 95% CI 1.70-2.71).
- **Test:** one-sided exact Poisson rate comparison (conditional binomial: n = 625,
  p0 = 6.0 / 7.89 = 0.760), **p = 2 x 10^-12** < 0.05.
- **Decision (as registered):** v1 is retired as the producer of beaten seeds; v2 continues.
  v1's weights and games are kept.
- **Prediction:** v2 passes. Confirmed.

### Reported alongside (not decisive)
- **Wins per hour:** 91.7 (550 in the window, fresh and retry arms) vs v1 52.4.
- **Fresh-seed wins per 1,000 games:** 8.78 (181 / 20,608) vs v1 4.73.
- **Eval blinds/game** (fixed 200 seeds, sampled): mean of the 36 evals in the window **10.66**
  (range 9.69-11.51; best 11.505 at iteration 300) vs v1's last six, mean 10.87. v2 doubles the
  win rate without beating v1 on average depth. Not tested; worth a look (more of v2's gain may be
  in closing out deep runs, which ordered actions make expressible, than in getting further on
  average).
- **Throughput:** median 8,054 games per hour; per decision, model 6.2 ms vs sim 29.4 ms
  (medians). The sim dominates, so a faster GPU would not speed v2 up; more sim processes would.
- **Level straight after behavior cloning** (`[eval 0]`): 9.905 blinds/game, 1 win in 200.

### What happens next (changes after the window)
- **Section G start checkpoint fixed now.** G says both arms start from v2's `best.pt` when the F1
  window closes. That file (iteration 300 eval, 11.505) is copied to
  `runs/v2/g_start_best_at_f1_close.pt`, SHA-256
  `70a49a7c2c563fb56a2eca4a238867706a279fb62f2f57358962584e0ea021de`, before v2 trains further
  and can overwrite `best.pt`.
- **v2 learning rate lowered after the window** (5e-5 -> 1e-5, resumed from `latest.pt` with
  `--reset-lr`; `restart_v2_lr.sh`). Reason: during the window nearly every PPO update hit the KL
  early stop, most after 15-90 of ~580 planned minibatches, so a typical update used under ~15% of
  the games it had just played. Not a pre-registered change, and made only after F1 was scored; any v2 result
  from here on is on the new learning rate and is reported that way.

---
