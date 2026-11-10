# Hypotheses: Run v2b (models 0.3 and 1.0)

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-02 ~20:55 CST: H. v2b vs v2, solved rate (registered before the window opens)

### Authorship
The hypothesis is chud's: "I bet I will improve the solved rate." The changes are the ones he
asked for, v2b: the joker-drag cap and hand-type exploration (PR #23). The metrics and the test
below were picked by Claude at his request.

### What was already seen
While picking the baseline, Claude counted fresh-seed wins in v2b's first rounds (505–535):
39 wins in 1,736 fresh games (22.5 per 1,000). Those rounds are **excluded**. The window starts
after this registration, and nothing in it has been looked at.

### What is compared
- **v2b:** the run in `runs/v2b` as it is now: ordered-capped actions, exploration 0.05,
  lr 1e-5, continued from v2's round-504 `latest.pt`.
- **v2 (baseline):** v2's last stretch on the same learning rate, rounds 445–504. Fresh-seed
  training games: 60 wins in 3,360 (17.9 per 1,000). This is the latest v2 policy and the closest
  comparison available. The F1-window rate (8.78 per 1,000) is from an older policy and isn't used.
- **No concurrent control** (one machine). This compares v2b's changes as a bundle; it can't say
  which change did it. Exploration plays 5% of play decisions off-policy in training games. That
  cost counts against v2b here, because these are the games that solve seeds.

### Window
v2b rounds **540 onward**, until **20,000 fresh-seed training games** or **8 hours** (training
clock), whichever comes first. One look, at the end; no early stopping. A crash and resume
inside the window are allowed and reported; a change to v2b's settings ends the window early,
and the result is reported as such.

### H1 (primary): solved rate, fresh-seed wins per 1,000 training games
- **Solved rate:** the share of fresh-seed training games (arm `fresh`; retry-arm games and eval
  games excluded) that win (beat the ante 8 boss), with a saved route that replays.
- **Test:** one-sided exact rate comparison, conditional binomial (as F1): given the total wins
  in both, v2b's count is Binomial(total, n_v2b / (n_v2b + n_v2)) under equal rates; p < 0.05 passes.
- **Power:** v2's 60 baseline wins set a floor. Even with unlimited v2b games, a ratio below
  about 1.25 can't reach p < 0.05.
- **Prediction (chud):** v2b's solved rate is higher than v2's.

### Reported alongside (not decisive)
- **New beaten seeds per hour** (the project's production rate), defined as in F1.
  - Confound: v2 ran slowly at the end (updates 160–194 s from about round 488, cause unknown).
    The v2b restart cleared that, so per-hour numbers mix speed with play.
  - Reported against both v2's F1 rate (90.8 per hour) and v2's rounds 445–504.
- **Full House:** the share of plays where it was the best hand type on offer and was played
  (v2: 0 of 2,274 in its wins). Measured by replaying v2b's training games in the window.
- **Joker drags:** share of moves, and moves per game (v2: 22.0%, 124).
- **Eval blinds/game:** mean of the evals in the window vs v2's evals in rounds 445–504.

---

## Added 2026-11-03 ~12:45 CST: H result (v2b vs v2, solved rate)

Scored once, at the end of the registered window.

### Window
- **Rounds:** v2b 540–882. The 20,000-fresh-game limit came first, during round 882: rounds are
  scored whole, so the window holds 20,045 games. The 8-hour limit would have fallen at round 920.
- **Training clock:** 7.36 h.
- **Stops inside the window:** two relaunches by chud (22:17, 22:39), a crash at about 02:17 (the
  PC went down; the log had heat-throttling warnings from 21:55), and a resume at 09:07. Every
  relaunch used the same `run_v2b.sh` settings. Crash and resume were allowed as registered; no
  setting was changed.
- **Interim look:** at about 09:05, while checking the crash, Claude's progress count of window
  games also tallied wins (335 in 12,670 fresh games). That number wasn't shared, and nothing was
  decided on it: the window continued to its registered end.

### H1 (primary): solved rate, fresh-seed wins per 1,000 training games — **PASS**
- **v2b:** 489 wins in 20,045 fresh-seed training games = **24.4 per 1,000**.
- **v2 (baseline, rounds 445–504):** 60 in 3,360 = 17.9 per 1,000.
- **Rate ratio:** 1.37 (approx. 95% CI 1.04–1.79).
- **Test:** one-sided exact conditional binomial (n = 549, p0 = 20,045 / 23,405 = 0.856),
  **p = 0.011** < 0.05.
- **Prediction (chud): v2b's solved rate is higher than v2's. Confirmed.**
- **What it does and doesn't show:**
  - This is v2b's changes as a bundle: the joker-drag cap, hand-type exploration, and 260+ more
    rounds of training.
  - There is no concurrent control, so further training alone could account for some or all of
    the gain. The comparison is to v2's last stretch, not to v2 continued for the same time.

### Reported alongside (not decisive)
- **New beaten seeds per hour:** 568 new seeds in 7.36 h = 77.1 per hour.
  - Against v2 rounds 445–504: 62.2 per hour. Against the F1 rate: 90.8 per hour.
  - The pool of unbeaten seeds that wins can come from shrinks as more get beaten, and v2's
    speed varied a lot, so this mixes speed, play and novelty.
- **Full House:** in a sample of 262 window training games (the first part of rounds 580, 680,
  780 and 870, replayed exactly), Full House was the best hand on offer 497 times and was played
  5 times (1.0%). v2 played it 0 of 2,274 times in its wins. The exploration alone (5% of
  decisions, spread over the hand types on offer) would produce about that rate, so this is no
  evidence yet that the policy itself plays Full House.
- **Joker drags:** 25.9% of moves, 140.3 moves per game, over all 38,416 window training games.
  v2 rounds 493–504: 22.0% and 124. The cap holds, but drags didn't fall.
- **Eval:** mean of the 36 evals in the window is **12.03** blinds per game (range 11.36–12.79),
  against v2's 6 evals in rounds 445–504, mean 11.10.

---

## Added 2026-11-03 ~16:15 CST: R. Re-solve: model 1.0 on the published won seeds (registered after the run started, before any of its output was read)

### Setup
`analysis/resolve.py` (PR #48, commit `27d1c3b`), started by chud on his machine at ~16:14 CST.
Model 1.0 plays each of the 1,549 published won seeds 10 times (sample seeds 0–9, sampled as in
`train/play.py`), and 1,549 control seeds 10 times: fresh seeds from the fixed RNG
`resolve-control-v1`, excluding every published seed and the 200 eval seeds. Red Deck, White
Stake, unmodded 1.0.1o, move cap 1,500. Games already on disk when this was written had not been
opened by chud or Claude. Scored with `resolve.py summary` on the complete run (k = 10 for every
seed); if the run stops early, scored on the games played, with the count stated.

### R0 (chud's, reviewed by Claude): proven seeds clear more often
- **Claim:** 1.0's clear rate on the seeds training proved beatable is higher than its average
  clear rate, 2.5% (5 of 200 eval seeds).
- **Test:** published minus control, mean per-seed win rate, one-sided; seed-level bootstrap.
  **Pass** if the lower end of the 95% interval is above 0 and the published rate is above 2.5%.
  *Falsified* otherwise.

### Claude's, reviewed by chud
Reasoning: if each seed has its own win chance p under 1.0, a seed found by winning it once is
selected in proportion to p, so its expected rate is E[p²]/E[p] = mean × (1 + CV²). Claude
expects large spread between seeds (early shops and tags decide much of a run).
- **R1. Control:** 2.0–3.5% of games won. *Falsified* outside that range.
- **R2. Published:** about 11% of games won, predicted 7–16%. *Falsified* below 5% or above 25%.
- **R3. Seeds matter without selection:** on the control seeds, dispersion index ≥ 1.5 with
  p < 0.01 (the summary's test). *Falsified* if the index is below 1.2.
- **R4. Retry-found seeds re-solve less** than fresh-found ones: retry ≤ 0.7 × fresh (rates per
  game). *Falsified* if retry ≥ fresh.
- **R5. Seeds first won by 0.1 re-solve most:** 0.1 > 0.2 ≥ 0.3. *Falsified* if 0.1 is lowest.
- **R6. No memorization:** the seeds 1.0 never trained on (0.3 wins after round 810 on a fresh
  seed, 97 seeds) re-solve at a rate within the interval of the rest. *Falsified* if the rest is
  significantly higher (non-overlapping 95% intervals). Weak test: 97 seeds.
- **R7.** 35–55% of published seeds are won at least once in 10.
- **R8.** Published seeds average at least 2 more blinds per game than control.

---

## Added 2026-11-03 ~20:15 CST: R result (re-solve by 1.0)

Scored once, on the complete run: 30,980 games, k = 10 for all 3,098 seeds, 0 hit the move cap.
The run finished at ~19:58 CST; `summary.txt` was the first output either of us read. Data:
`analysis/results/resolve_1.0/` (`summary.txt`, `ranked.csv`, every game in `resolve_1.0.jsonl.gz`).

| | Games won | 95% CI | Blinds/game | Seeds won ≥ 1 of 10 |
|---|---|---|---|---|
| Control (fresh seeds) | 455 / 15,490 = **2.94%** | 2.68–3.22 | 12.40 | 186 (12.0%) |
| Published won seeds | 3,254 / 15,490 = **21.01%** | 20.37–21.66 | 17.15 | 958 (61.8%) |

Published minus control, mean per-seed win rate: **+18.07 points** (seed bootstrap 95% +16.69 to +19.36).

### R0 (chud's): proven seeds clear more often — **PASS**
21.01% > 2.5%, and the interval's lower end (+16.69) is above 0. 1.0 clears a proven seed about
7 times as often as a fresh one. A win is largely a property of the seed, not only of the moves
sampled.

### Claude's
- **R1. Control 2.0–3.5%: pass** (2.94%).
- **R2. Published ~11%, 7–16%: missed, not falsified.** 21.01% is above the predicted range and
  below the 25% falsification line. Claude underestimated the spread between seeds.
- **R3. Dispersion ≥ 1.5 on control: pass** (4.01, p ≈ 0). 3 to 10 wins in 10 on 62 control seeds,
  where seeds that didn't differ would give about 4.
- **R4. Retry ≤ 0.7 × fresh: pass, narrowly** (17.87% vs 25.70%, ratio 0.695).
- **R5. 0.1 > 0.2 ≥ 0.3: not supported, not falsified.** 0.1 22.69%, 0.2 24.83%, 0.3 17.24%.
  0.1 is not the lowest, but it isn't the highest either.
- **R6. No memorization: not falsified.** Unseen seeds 24.12% (21.54–26.91, 97 seeds) vs the rest
  20.80% (20.15–21.47). The rest isn't higher; the unseen seeds are, so the registered "within the
  rest's interval" also misses, in the direction that rules memorization out. Confound: the unseen
  seeds are all fresh-found, and fresh-found seeds overall re-solve at 25.70%.
- **R7. 35–55% won at least once: missed** (61.8%).
- **R8. ≥ +2 blinds per game: pass** (+4.75).

Tally: R0 pass. Of R1–R8, 4 pass (R1, R3, R4, R8), 3 missed without being falsified (R2, R5, R6),
1 missed (R7). R2 and R7 both missed high: seeds differ more than Claude predicted.

### Reported alongside (post hoc, not registered)
- **Self-selection bound.** From the control's own per-seed counts, seeds picked by 1.0's own wins
  would re-solve at E[p²]/E[p] ≈ 35% (unbiased estimate, w(w−1)/k(k−1) per seed). The published
  seeds re-solve at 21%: they were picked by earlier weights (0.1, 0.2, 0.3 at other rounds) and
  by retries, so the finders' p only partly matches 1.0's.
- **New beaten seeds.** 1.0 won 186 control seeds (455 games). Their routes are in chud's
  `resolve_1.0_wins.jsonl`; none are published or replayed on the untouched game yet.
- **Ranking.** `ranked.csv` orders all 3,098 seeds by 1.0's wins; 16 published seeds were won
  10 of 10 and 2 control seeds were too.

---
