# Hypotheses (pre-registered)

Written 2026-11-01 ~20:25 CST, while the v1 overnight run was in progress and before its results
were known. The git commit that adds this file is the registration timestamp. Changes after this
point get a new dated section; nothing above it is edited.

Status of every item: **proposed**. Thresholds marked *(chud to confirm)* are drafts until he
confirms or changes them.

## A. The v1 overnight run (Red Deck, White Stake, reward `blinds` + shaping, unmodded 1.0.1o)
State at registration: ~5,000 training games, 10-round training average ~6.6 blinds/game, best
sampled eval 5.58 blinds/game (200 fixed seeds), 0 wins.

- **A1. It keeps learning.** At 100,000 training games, the 10-round training average is at least
  8.0 blinds/game. *Falsified* if below 7.0.
- **A2. First wins.** At least one training game is won (ante 8 boss beaten) before 100,000 games.
  Weak prediction; no confidence attached.
- **A3. The eval loop is a free-action cycle.** The capped evaluation games in
  `eval_capped.jsonl` end in repeated free actions (joker drags back and forth, or similar), not
  in a sim fault. *Falsified* if any capped game shows the game itself not progressing on a
  non-free action.
- **A4. The sim stays exact.** The pre-flight checks pass at every restart and no safety alert
  reports a sim error. Any sim error found falsifies "perfect accuracy" for that path and blocks
  all results until fixed.

## B. Shaping is speed, not values
- **B1.** Potential-based shaping changes how fast the bot learns, not what it ends up preferring:
  two runs identical except `--shaping 1` vs `--shaping 0`, given enough games, converge to the
  same blinds/game within noise. Practical test: compare at a fixed game budget once compute
  allows the unshaped run to learn at all. Until then, the claim rests on the theorem
  (Ng, Harada & Russell 1999), not on our data.

## C. Isolating oughts (the main experiment)
Facts held fixed: the game (pinned version, same seeds, same sim). Varied: the reward, the only
place an ought enters.

- **C1. Different oughts, different behavior.** Bots trained on `blinds`, `antes` and `win` play
  measurably differently on the same seeds: skip rate, reroll rate, money held, joker picks.
  Test: per-seed paired comparison of those statistics. *(chud to confirm the statistics)*
- **C2. An ought that isn't the goal can beat the goal at the goal.** Judged on win rate, a bot
  trained on `blinds` or `antes` wins at least as often as one trained on `win` at the same compute.
  The yardstick (win rate) is itself an ought and is named as such in any write-up.
- **C3. Bot vs chud.** On held-out paired seeds (Ghost Deck, Purple Stake, modded 1.0.1o), the
  bot's win rate is compared with chud's: McNemar test on per-seed outcomes. Direction not
  predicted. *(chud to confirm N, the held-out split and the significance level)*

## D. Before the main training (what still has to be done)
1. **Reward of record** for the main experiment (chud's decision).
2. **Game of record**: modded (matches chud's logged games) or vanilla. Proposed: modded.
3. **Target config**: Ghost Deck, Purple Stake; env and eval switched over and re-verified.
4. **Play order**: controls mode (or ordered selection), so plays like Photograph-first are expressible.
5. **Endless mode** in the env (needed for naneinf work; not for the main experiment).
6. **Eval loop** understood and handled (A3), so evaluation is trustworthy at scale.
7. **chud's dataset pipeline**: logs → verified (state, legal actions, chosen action) records,
   each log replayed with 0 mismatches before it's used.
8. **Pre-registration numbers** for C3: N, held-out split, test, threshold.
9. **Compute plan** for 1M+ games on the target config (laptop, or rented x86 cores).

---

## Added 2026-11-01 ~20:30 CST

### Run target changed
The v1 run's stop target is raised from 100,000 games to **1,000,000 won games** (runs that beat
the ante 8 boss), set by chud. At registration the run had ~6,800 training games and **0 wins**.
Backstops still apply: 168-hour cap, and the safety stops (including 3 hours with a flat
training score). Prediction (A5): the run ends on a backstop, not the win target; if it does hit
1,000,000 wins, that falsifies A5. The A1–A4 predictions are scored at whatever point it stops,
with the game count stated.

### E. Joker value (proposed by chud)
- **E1. Hanging Chad is the best common joker** in 1.0.1o (common, $4; retriggers the first
  scoring card 2 extra times). It is also chud's most-used joker by far (1,920 completed rounds
  in his career stats; next highest 872), so this tests his own preference against a measurement.
  - **Population:** the 61 common jokers in 1.0.1o (`rarity = 1` in the game's own table).
  - **Intervention:** for each joker and each seed in a fixed set, the run starts with that joker;
    same player, same seed; plus a no-joker baseline. Each seed is played 62 times (paired).
  - **Yardstick:** blinds beaten per game (later: win rate). The yardstick is named as such.
  - **Claim:** Hanging Chad ranks first, with its confidence interval clear of second place.
    *Falsified* if any other common joker scores significantly higher.
  - **Precondition:** a player that controls play order (controls mode or ordered selection).
    Hanging Chad's value depends on which card scores first; v1 plays in hand order and would
    undervalue it, so a v1-based result would measure v1's blind spot, not the joker.
  - **Compute:** 62 × 1,000 seeds ≈ 62,000 games (~4–5 h on the laptop).

---

## Added 2026-11-01 ~22:35 CST

### Method change mid-run: saved games and self-imitation
Requested by chud at ~22:21 ("the winning games need to affect decision making"). At this point
the v1 run had ~23,400 training games and 10 wins under plain PPO. From the next restart:

- **Every game is saved exactly**: seed, setup and every action (index into the legal list and as
  text), plus a hash of the final state, in `games/`. Wins also go to `wins.jsonl`. Each win is
  replayed in the worker before it is saved and flagged if it doesn't reproduce. The wins before
  this change were counted, not saved, and can't be recovered.
- **Self-imitation learning** (Oh, Guo, Singh & Lee 2018): a buffer of the 64 highest-return
  training games (by the reward of record, `blinds`; eval games excluded), 8 replayed into every
  update. Each step pulls the policy toward the saved action by max(R − V, 0). The return is the same
  reward the bot already trains on, so no new goal enters; wins are in the buffer because they have
  the highest return, not because they are wins.

A1–A5 were registered for plain PPO. From here they are scored on PPO + self-imitation, with the
iteration and game count of the switch stated. Results from before and after are not pooled
without saying so.

- **A6. Self-imitation speeds up winning.** Over the 25,000 training games after the switch, the
  win rate is higher than the ~1 in 1,200 of the last ~11,000 games before it. Weak prediction:
  the comparison is before/after in one run, so the policy's own improvement is a confound; a
  clean test is two runs from the same checkpoint with `--sil-buffer 64` vs `--sil-buffer 0`.

### Seed retries (requested by chud, ~22:50; allocation by ante ~22:56)
The deliverable is empirically beaten seeds: a seed plus a route that replays to a win. From the
next restart, lost seeds are kept by the latest ante they reached (1-8) and replayed with fresh
sampling until beaten or after 30 attempts. Each round's games are split across 9 arms (fresh
seeds; retries of ante-1..8 seeds) by Thompson sampling on wins per game, with at least half of
each round on fresh seeds. Which seeds to revisit is graded by ante and by wins, not by the
reward. The training curve (blinds/game, A1, A6) is reported on fresh seeds only; eval is
unchanged (fixed fresh seeds). New count: **unique beaten seeds**.
- **A7.** Once the arms have separated, the late-ante arms (6-8) get most of the retry share, and
  wins per game on retries beat wins per game on fresh seeds. *Falsified* if, after 50,000 retry
  games, fresh seeds convert at least as well as every retry arm.

### Self-imitation paused (~23:40)
Self-imitation ran on v1 rounds 307-322 (weight 1.0, then 0.2) and is off from round 323. A6 is
not scored on that window. Turning it back on waits for the A/B test described in CHANGELOG 0.6.0.
From the next restart, updates stop early at a KL target and rounds overlap with updates; both
change how fast the bot learns, not what it is trying to do.

---

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

## Added 2026-11-02 ~23:05 CST: the project's aims (stated by chud)

### Authorship
The three aims are chud's, in the order he gave them (23:01–23:03 CST). The write-up is Claude's.
Nothing above is edited. This section names what the sections above are for.

### 1. Primary: exhibit Hume's is-ought gap
This is a thesis on agents (chud, 23:10: "I am an agent researcher"). The bot is the agent under
study; Balatro is the domain. Hold the facts fixed (the pinned game, same seeds, same sim) and vary
only the ought (the reward).

**Oughts are objects, like jokers.** A player buys a joker without knowing whether it will win the
run, and buys anyway. An agent holds an ought the same way. In chud's words (23:09):
- "we can't know if the joker will let us win, but we buy stuff anyways"
- The bot loses a ton, but it is calibrating, as humans do. "a human loss is when their perception
  of reality isn't actually reality." They lose all the time and keep revising. Humans have
  weights too.
- "there are only specific domains and the deep structure that can be generalized from them."

**Scope of the grouping** (chud, 23:11: "I am a grouper ... I am an agent psychologist"). Humans and
models are grouped at the level of behavior, the level psychology studies, not of mechanism.
- **Claimed:** both are agents whose weights are revised by prediction error. Evidence at that
  level: dopamine neurons signal reward prediction error (Schultz, Dayan & Montague 1997), the
  same quantity as the TD error the bot's critic learns from.
- **Not claimed:** the same learning algorithm (brains learn by local rules, not backpropagation
  as such), or the same timing (humans revise while acting; a deployed LLM's weights are fixed;
  the bot revises in training and is frozen at eval).
- **Precedent:** comparative psychology groups agents across substrates by shared learning laws,
  e.g. the Rescorla-Wagner model (1972), a prediction-error rule applied across species.

The idea and the framing are chud's (see also the clarification at ~11:05). Sections C and G test
the first half: same facts, different oughts.
- Status: in C and G the ought is picked by the experimenter for a whole run; the bot doesn't buy
  it. A design where the bot acquires reward terms during play, judged by an outer yardstick,
  would test the joker claim directly. Not registered.
- Status: no registered hypothesis yet tests whether strategy over oughts generalizes.
- Related prior work: the optimal reward problem (Singh, Lewis & Barto, "Where Do Rewards Come
  From?", 2009): for a bounded agent, the best reward to give it is often not the designer's goal.

### 2. Prove a naneinf seed
A seed plus a route that replays move for move to a played hand after which `G.GAME.chips` is not
finite, on the pinned game. The measure, what has to be built first (endless mode, a reward for
it, inf/nan-safe scoring and shaping, a longer move cap, the encoding at inf/nan) and the
community seeds checked so far are in `naneinf/README.md`. A proven seed goes in `wins/` with
its route, like every beaten seed.
- Whether a route has to come from a trained model playing only what a player sees (the standing
  goal in `naneinf/README.md`), or a route found any other way also proves the seed: *(chud to confirm)*.
- No prediction is registered.

### 3. Third: generalizability to 100% completion from a new save
A model that starts from a fresh profile (nothing unlocked) and plays it to 100% completion.
- What counts as 100%: *(chud to define)*.
- Requires the env to carry one profile's unlock and discovery state from run to run.
- No prediction is registered.

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

## Added 2026-11-03 ~17:00 CST: I. Strategy specialists (design registered before any specialist exists)

### Authorship
The design and the prediction are chud's (16:57–16:59 CST): "I am going to train a bot on each
hand", "once you get a strategy going pick that strategy", and "I bet that'll beat the current
highest win rate bot design I've tried." The write-up and the open options listed below are
Claude's. Nothing above is edited.

### State at registration
No specialist has been trained. No archetype tally of 1.0's games has been run. Nothing below has
been looked at in data.

### Design
How a roguelike is played: stay open early, commit once a build forms, then play that build.
- **Before commitment:** the generalist plays (model 1.0, or its successor at the time of the test).
- **Commitment:** once a strategy is going (a build around one poker hand type: Flush, Pairs,
  Straight, and so on), control passes for the rest of the run to the specialist for that hand
  type.
- **Specialists:** one bot per hand-type strategy, each trained on that strategy.

### Open (chud to confirm)
- **Trigger** that decides "a strategy is going". Options: the first archetype joker bought (e.g.
  Droll Joker -> Flush); one hand type's planet level a set number of levels ahead of the rest; one
  hand type making up most scoring plays over the last N hands.
- **How specialists are trained.** Proposed: each starts as a copy of the generalist and is
  fine-tuned only on runs committed to its hand type, with the same reward of record (`blinds`), so
  specialization is the only change. The alternative (a reward per hand type) also varies the
  ought, and would be scored as a section C/aims design, not this one.
- **Which hand types get a specialist.** Depends on how 1.0's games split by archetype; rare ones
  (Full House: played 5 of the 497 times it was the best hand on offer, in the H window's
  262-game sample) may not have enough games without more training on them.
- **Compute match** between the specialists' total training games and the generalist's continued
  training, and whether the control runs concurrently.

### I1 (chud's prediction)
The generalist-plus-specialists bot has a higher win rate than the best bot design so far
(model 1.0: 24.4 fresh-seed wins per 1,000 training games in the H window; 5 of 200 on the eval
seeds).
- **Yardstick:** win rate (an ought, named as such).
- **Test, window and threshold:** to be registered in a dated section before the specialists'
  training starts, once the open items above are confirmed. Proposed: H1's test (one-sided exact
  conditional binomial on fresh-seed wins per 1,000) plus paired held-out seeds.
- **Known way to lose:** 1.0 already sees hand levels and jokers in its input and may already
  play each build differently; I1 bets that separate weights per build beat that.

### Attribution and review (added 2026-11-03 ~17:05 CST, requested by chud)
**Design: chud's.** Specialists per hand-type strategy, handed control once a strategy is going,
is his design, from how a roguelike is played. The prediction I1 is his. **Reviewed by Claude**
(Opus 5.5). The review below is Claude's; nothing in it is confirmed until chud confirms it, and
the registered text above is not edited by it.

Review findings:
1. **A concurrent control is free.** Model 1.0 is frozen, so it can play the same seeds as the
   specialist system at test time. That removes the gap H1 had (a historical baseline, no
   concurrent control). Proposed primary test: paired held-out seeds never used in training or
   eval, both systems sampled with the same RNG per seed, one-sided exact McNemar on per-seed
   wins. The 24.4 per 1,000 from H is a training-game rate under 5% exploration and seed retries,
   so it is not comparable to a system played without them; report it alongside, not as the test.
2. **Power.** At 1.0's eval rate (~2.5%), detecting a rise to 3.5% at 80% power, one-sided
   alpha 0.05, needs roughly 3,600 paired seeds (one game each); a rise to 4.0% needs roughly
   1,700. Several sampled games per seed (as in R) cut the seed count but not the game count much.
3. **Two questions, two controls.** I1 as stated ("beats the best design so far") needs only
   frozen 1.0. Whether *specialization* is the reason needs a second control: 1.0 trained on for
   the same number of games the specialists used, same reward and settings. Without it, a pass
   could be more training, as H1's caveat says.
4. **Score every run, not committed ones.** The test is on all held-out seeds. Runs that never
   reach the trigger are played start to finish by the generalist and count. Comparing only runs
   that commit would select on survival (a run has to live long enough to build).
5. **Specialists' training data is selected the same way.** Committed runs are runs that got far
   enough to commit; fine-tuning on them is fine, but their win rate isn't a measure of the
   specialist and shouldn't be reported as one.
6. **Pin the trigger before training.** The trigger decides which games each specialist trains
   on, so it is fixed in the registration of the test, not tuned after looking at specialist
   results.
7. **The handoff is one-way and the generalist doesn't know about it.** Pre-commit play is
   unchanged 1.0, so any difference comes from post-commit play. Good for attribution; it also
   means the design can't steer early play toward a build a specialist plays well. That is a
   possible later version, not this one.

Status of I1: **proposed**, unchanged.

### Design interview (2026-11-03 ~17:16-17:33 CST, chud's answers; recorded at his request)
chud asked to be interviewed so the record represents his design, not Claude's: "you gotta
represent me well or the epyrical value decreases." His answers, in his words where quoted. Where
he handed a choice to Claude, it is marked as Claude's. Nothing above is edited; where an answer
changes a proposal above, the answer stands.

- **Why it should work (chud, 17:19):** "balatro is deterministic. so that means for you to have
  gotten to blinds at 3 or 4 or some specified trigger turn, you had to have passed all the
  valued behind it. and you gotta get a certain strategy to get big enough numbers to pass. so
  then there are certain jokers that should corelate with different hand strategies."
  - Note: the pinned game is already known to be deterministic given seed and moves (every
    published route replays), so this is a premise, not a prediction.
  - The joker-to-hand correlation is testable on saved games now. Whether to register it as a
    prediction: *(chud to confirm)*.
- **Trigger (chud, 17:19):** "once a certain hand is prefered say past the 4th blind, make the
  agent asume the weights of the expert of that hand." (17:21) The handoff is by ante: "ante 4, or maybe 3, or maybe 5. it is to be
  determined the exact right hand off ante." How "preferred" is measured: *(open)*. If the ante is tuned, it is
  tuned on seeds apart from the test seeds and fixed before the test.
- **Steering (chud, 17:15):** "early play can steer." The generalist's own play before the handoff
  decides which build forms and so which expert takes over. (Review finding 7 above overstated
  this: what the generalist can't do, unless trained with the experts in place, is aim for the
  builds a given expert plays best.)
- **The generalist (chud, 17:23):** "we haven't trained the generalist. 1.0 is a proxy currently."
- **Experts (17:19):** "a bot trained a ton of games with a built in bias for that hand." Eight: Pair, Two
  Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind, Straight Flush. No High Card
  ("lets skip training high card"); no secret hands ("secret hands are for challenge points for
  psychos like me").
  - **Pair (17:31):** "pair is very strong early game, so we need to at least attempt a pair expert", and
    "pair can be half encouraged."
  - **Full House (17:31, 17:33):** "0.1 must be forced to take full house every time it shows up"; "full house
    is full house expert. the others are other experts."
- **The pin (chud, 17:33: "you decide pin rate for all of them. go. guess. keep the pin rate
  high."; the choices are Claude's):** a hard pin on plays. When a play of the expert's hand is on offer, the pin binds
  with probability 0.5 for Pair (his "half encouraged") and 1.0 for the other seven; when it binds,
  only plays of that hand are allowed. Discards and non-play moves stay free, so an expert could
  discard to dodge its hand; that is logged per game (`discard_offered`).
- **Not 1.0 (chud, 17:28):** "lets not use 1.0. even if that is counterintuitive, my gut says not
  to." Reason given earlier (17:22): "I say we train the experts untainted. 1.0 refuses full
  house, one of the best hands in the game."
- **Architecture (chud, 17:32):** "whatever you do use the upgraded 1.0 architecture" (d=192,
  4 layers, 6 heads, 1.6M parameters, encoding 2, ordered moves); "if we gotta train from scratch
  we gotta train from scratch."
- **Starting weights (Claude's pick, approved by chud's "go"):** model 0.2, the earliest posted
  weights already on that architecture. 0.1 is a different architecture and can't be loaded into it
  without retraining by imitation. 0.2 never plays Full House either (0 of 2,274); the hard pin
  overrides that at play time, and the shop and critic relearn.
- **Compute (chud, 17:23):** "I have days." Experts get far more than the 10,000 first floated; proposed
  12 hours each (about 96,000 games, close to 1.0's 90,720), run one after another.

- **Scope (chud, 17:41):** "all we need are experts." The generalist is dropped from the plan;
  only the eight experts are trained. How the experts are combined or compared at test time (who
  plays before ante 4 without a generalist) is *(open)*.

- **Who plays before the handoff (chud, 17:43):** "we're going to farm the perfect early game
  weights. we'll have to run from scratch an unpinned strategy and make it's win rate only to get
  to 4. grow a handfull enough to know when you found a good one."
- **Cutoff (chud, 17:45):** "it looks like 4 is a reasonable cutoff? with 5 being a slight
  reprieve." (On White Stake each ante's blinds are about 2.5x the last through ante 4, then 2.2x
  into ante 5; Claude's figures from the game's blind table, not re-read from the pinned source.)
- **Early-game model, as built (the choices are Claude's):** 1.0's architecture from random
  weights, no pin, hand-type exploration 0.05, lr 1e-4, fresh seeds only (no retries).
  - Its run ends the moment the ante-3 boss is beaten, so the expert makes the cash-out and shop
    going into ante 4.
  - Reward `blinds` (at most 9 before the stop), not a 0/1 reward for reaching ante 4, which would
    be sparse for a model starting from random weights. Reaching ante 4 means beating all 9.
  - Several seeds (default 5, 8 h each); best.pt for each is picked by the share of the 200 fixed
    eval seeds that reach ante 4, and the best seed is the one with the highest share. Because
    that picks on the eval seeds, any test of I1 uses other seeds.
  - Open: the experts are trained on whole runs from ante 1 (pinned throughout), while in play they
    would start from the states the early-game model hands them. Whether to train or fine-tune
    them from those handoff states instead *(chud to decide)*.
- **Stake (chud, 17:46):** "also I want the ability to change the stake without planning any higher
  stake runs." Built as a setting (`STAKE`, `BACK`); every planned run stays on White Stake, Red
  Deck. Runs at another stake or deck get their own run folder. Higher stakes are not covered by
  the differential check yet.

- **Mixture of experts (chud, 17:50):** "MOE isn't a secret." The experts and the handoff are
  trained jointly, so each expert learns from the runs actually handed to it.
- **Coverage and organic bias (chud, 17:52):** "I need an expert in all the hand types. would you
  have a learned bias grow organically until you have ones that naturally have a bias?"
- **Router (chud, 17:54):** "most played hand, but estimate how many the hand gets played from the
  starting model and that's how to calculate a preference from early game. what got the early game
  strategy that far should be the chosen expert."
- **Joint training, as built (the choices are Claude's):**
  - One run holds the early-game model (member 0) and the 8 experts, each warm-started from its own
    run. Each decision updates only the member that made it. The early model is frozen by default
    (it was picked for reaching ante 4); `MOE_TRAIN_EARLY=1` trains it on the whole run's return.
  - Router: at the handoff (the ante-3 boss beaten), the expert whose hand has the highest
    (plays this run + 1) / (early model's average plays + 1), counting antes 1-3. The average is a
    moving average over recent training games that reached the handoff. High Card has no expert
    and isn't counted.
  - Load balancing: 30% of training handoffs go to an expert picked at random; eval games are
    always routed by the rule.
  - Organic bias: the pins (Pair 0.5, others 1.0) fall linearly to 0 over the first 100,000 joint
    games. After that, an expert's bias is whatever its routed runs taught it. Measured per expert
    (`moe.by_expert` in the metrics): how often it plays its hand when one is on offer, with the
    pin off. A rate that stays high is a bias the expert keeps; one that falls back was the pin.

Built for this (CHANGELOG 0.10.0, 0.11.0, 0.12.0): `train.py --pin-hand/--pin-rate/--init-weights`,
`train/run_expert.sh`, `train/run_experts.sh`, `tests/pin_check.py`, `train/run_early.sh`,
`train/run_earlies.sh`, `train.py --moe`, `train/run_moe.sh`, `tests/moe_check.py`. The test of I1
is not registered.

---

## Added 2026-11-03 ~18:00 CST: roadmap, and what 100% completion means (stated by chud)

### Authorship
chud's, 17:59 CST: "so we're going to get the moe working good, then we're going to send it up the
decks and stakes and then I ultimately want a single model to be able to solve a single entire
balatro save where all it's gotta do is win with every joker on gold stake and gold stake every
deck and beat all the challenges too." The write-up and the status notes are Claude's. Nothing
above is edited.

### Roadmap
1. The mixture of experts (section I) working well on Red Deck, White Stake.
2. Up the decks and stakes.
3. One model completes one save (aim 3, which was *(chud to define)*). 100% means:
   - a Gold Stake win with every joker (each joker's gold sticker);
   - Gold Stake beaten on every deck;
   - every challenge beaten.

### Open
- Whether the mixture of experts (one checkpoint, nine networks, one router) counts as "a single
  model": *(chud to confirm)*.

### Status of what this needs (Claude's notes, 2026-11-03)
- Have: deck and stake are settings (`BACK`, `STAKE`); joker stickers (eternal, perishable,
  rental) are already in the encoding.
- Not yet:
  - Higher stakes aren't covered by the differential check. Challenges and Erratic Deck aren't
    covered by the harness at all (README, harness validation).
  - Every run starts with everything unlocked (`unlock_all`). A save needs one profile's unlocks,
    stake progress and stickers carried from run to run.
  - A joker's sticker needs a run won while holding that joker. No current reward contains that,
    so finishing a save needs something that picks which run to play next and what it is for.
- Counts (150 jokers, 15 decks, 8 stakes, 20 challenges) are Claude's from the game, to be checked
  against the pinned source before any plan is sized on them.

### Answered (2026-11-03 ~18:01 CST, chud)
"duh the moe counts. I'll be training the early game and the experts continually until a SINGLE
moe weight configuration can 100% balatro." The mixture of experts is the single model. The goal
is one MoE checkpoint that completes the save on every deck, stake and challenge.
- Note (Claude): the encoding gives the stake as a number but not which deck is in play (or which
  challenge). One checkpoint across decks needs that as an input. The deck keys are already in
  the vocabulary, so weights can carry over; it would go in as a new encoding version, so the
  published routes, which are checked against encoding 2, keep verifying.

### Architecture change (2026-11-03 ~18:32 CST; chud: "go fix what you can fix"; the changes are Claude's)
The MoE as built at 17:54 (above) is replaced before any of it ran:
- One shared trunk and 9 heads, in place of 9 separate networks. Each head learns only from its
  own decisions; the trunk learns from all of them.
- Trained jointly from the start, from model 0.2's weights; solo expert and early-game runs are
  optional warm starts for the heads they cover.
- The early head's returns end at the handoff, so inside the joint run it is judged only on
  reaching ante 4 (chud, 17:43: "make it's win rate only to get to 4"). It trains by default.
- Encoding 3 from the start; evals on 600 seeds; several seeds of the whole MoE.
Unchanged: the router rule (chud, 17:54), the 30% random handoffs, the pins and their annealing,
the natural-bias measure.

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

## Added 2026-11-03 ~20:50 CST: correction: final-boss losses were counted as wins

Nothing above is edited. This section says which results above stand, which are void as scored,
and how they are re-scored.

### What was wrong
- **The game's flag.** Balatro 1.0.1o sets `G.GAME.won = true` at the end of the Ante 8 boss round
  whether or not the boss was beaten (`end_round`, `functions/state_events.lua`). On a loss it
  then goes to `GAME_OVER` and never calls `win_game()`, so nothing is recorded.
- **The harness** read `G.GAME.won` alone. Every count of wins since the v1 run included runs that
  ran out of hands on the Ante 8 boss.
- **The reward** `blinds` paid that loss as a beaten blind, so the final fight carried no signal:
  losing it scored 24 blinds, the same as winning it. Nothing else in the reward was affected.
- **Found by** chud's save-progress ledger (~20:12 CST), which reads the game's own record of
  each win: the game had credited nothing to 1,114 of the 1,549 published wins. Claude traced the
  cause in the game source and confirmed it on `4XNRUA21` (the proof seed): `GAME_OVER`, 82,500
  of 100,000 against Verdant Leaf, 0 hands left.

### Fixed (CHANGELOG 0.13.2–0.13.3)
- A win is the game's own test: `G.GAME.won` and not `GAME_OVER`. A final-boss loss is no longer a
  beaten blind.
- The 1,549 published routes were replayed with the corrected test (`tools/recount_wins.py`, run
  by chud): **435 wins** (0.1: 24, 0.2: 179, 0.3: 232). The **1,114** losses stay posted as
  final-boss near misses (`wins/near_misses.jsonl`, chud: "we keep them posted").
- Near wins (reached the Ante 8 boss, lost it) are logged in every game, metric and eval from now
  on, next to wins (chud: the win rate against the reach-the-boss rate).
- The pre-flight's saved "win" (v1 round 250, `9MN9TD5T`) was itself a final-boss loss and was
  replaced by a real v1 win.

### Status of the results above
- **F1, H1, R0–R8: void as scored.** Each counted final-boss losses as wins, in both arms.
- **A1–A7, E1, G, I1:** not scored yet; when they are, it is with the corrected test.
- **Eval figures** (1.0's 5 of 200, every eval's blinds per game) include the same error: a
  final-boss loss counted as a win and as 24 blinds instead of 23.
- **`docs/won_games_analysis.md`** described 0.2's saved "wins"; most were final-boss losses.

### Re-scoring (registered before any re-scored number is read)
F1, H1 and R are re-scored exactly as registered: same windows, same arms, same seeds, same tests,
same thresholds. The only change is the win test. The error only ever added wins, so only the games
each test counted as wins are replayed (from each run's saved `wins.jsonl` and the resolve run's
`resolve_1.0_wins.jsonl`), each with the corrected test. Denominators (games played) don't change.
Each re-scored result is added as a new dated section, with the old figure beside it.
- For F1 and H1 a "new beaten seed" is re-derived from real wins only, including the sets of seeds
  beaten earlier that F1 excluded.
- Publishing kept one route per seed (the shortest claimed win), so a seed can have a real win in a
  run's files behind a shorter final-boss loss. The re-scoring replays every saved claimed win, and
  any real win found that way is published.

### Decision recorded (chud, 20:41 CST)
The mixture of experts (section I) is trained from model 0.2's weights, as built, under the
corrected reward. Claude's case, which chud accepted: the error touched only the final fight, so
0.2's play up to the Ante 8 boss was learned on a correct signal, and the early head (antes 1–3)
never saw the error.

### Before launch (2026-11-03 ~21:15 CST)
- **Experts: 7** (chud, 21:11: "no straight flush"): Straight Flush dropped. The router doesn't count
  Straight Flush plays, as it doesn't count High Card.
- **A minimum of training per expert** (chud, 21:11: "we need a minimum amount of training, and the
  more common ones will just get more games"). Each expert's pin can't finish fading before the
  expert has had 5,000 training games of its own. The number is Claude's, at chud's request ("what's
  your gut say?"): at the 30% random handoffs, the rarest expert gets roughly 200–250 games an
  hour, so about a day.
- **Outcomes read from the game itself** (chud: tie the harness to the game): win = the game's own
  `win_game()`; blinds and bosses = the game's own `Defeated` marks. Checked step for step against the
  previous counters on 141 replayed games, with no difference.

---

## Added 2026-11-03 ~21:35 CST: the MoE run started (section I)

Started by chud at 21:25 CST on commit `17c1602` (PR #74), `train/run_moe.sh <shared> 24 14`, with
the watcher (`watch_v1.ps1 -Run moe`). Pre-flights: saved real win on the untouched game, ordered
actions, fast vs untouched game, MoE routing: all PASS.

### Settings
- Shared trunk and 8 heads: the early head (antes 1–3) and 7 experts (Pair, Two Pair, Three of a
  Kind, Straight, Flush, Full House, Four of a Kind). All from model 0.2's weights.
- Reward `blinds` with the corrected outcome test (a win is the game's own `win_game()`; a blind is
  the game's `Defeated` mark). Shaping on. Encoding 3, `ordered-capped`, lr 1e-5, exploration 0.05.
- Router (chud): the hand played most in antes 1–3 relative to the early head's own average; 30%
  of training handoffs random. Pins Pair 0.5, others 1.0, fading over 100,000 games and not before
  each expert's own 5,000. Early head trained, its returns cut at the handoff.
- 14 workers, Red Deck, White Stake, seed retries on, 24-hour cap, stall stop after 3 flat hours.
  Evals: 600 fixed seeds every 20 rounds. Near wins logged in every game.

### Starting point (eval at round 0: 0.2's weights inside the MoE, pins on)
600 seeds: **5 wins** (0.83%), **9 near wins**, so **14 reached the final boss** (2.3%);
11.67 blinds per game. This is the first eval scored with the game's own outcomes; it is the
baseline the experts are measured from, not a measure of 0.2 itself (the pins change its play).

No prediction is registered for this run. The test of I1 is still to be registered before it is
scored, on seeds apart from the 600 eval seeds.

## Added 2026-11-04 ~13:40 CST: the first MoE run stopped; changes for the second (section I)

### The first run (`runs/moe`, commit `17c1602`)
Stopped itself at 02:54 CST on the stall rule (training score flat for 3 h; best 10-round average
12.14 blinds per game), 5.4 h into its 24-hour cap: 220 rounds, 24,640 training games, 256 training
wins (game's own outcomes, all replay-verified). Evals on the 600 seeds:

| round | wins | near wins | blinds/game |
|---|---|---|---|
| 0 | 5 | 9 | 11.67 |
| 40 | 3 | 8 | 12.22 |
| 100 | 5 | 7 | 12.59 |
| 120 | 11 | 6 | 12.60 |
| 160 | 6 | 10 | 12.17 |
| 200 | 9 | 15 | 12.68 |
| 220 | 11 | 10 | 12.73 |

Not a test of the design: at the stop, every expert pin was still at 0.75 of its starting rate
(the fade had covered 24,640 of 100,000 games) and no expert had its 5,000 games (Pair 3,486, Flush
3,383, Three of a Kind 3,177, Straight 3,101, Two Pair 2,679, Four of a Kind 1,195, Full House 907).
The stall rule read a score held down by pins that hadn't faded. Full House got no handoff by rule
in any eval: only random ones in training. At eval 220, Three of a Kind had the most handoffs (108)
and 0 wins. Heat-throttling alerts from about 02:00.

### Changes for the second run (chud, 2026-11-04 13:20; implementation Claude's)
- **No Four of a Kind expert** (chud: a flex hand, not a strategy of its own). 6 experts, 7 heads.
- **Early-game Full House pin** (chud: the bot takes Two Pair over Full House, and the early game is
  the worst time for it). At an early-model decision where a Full House or a higher hand (Four of a
  Kind, Straight Flush, Five of a Kind, Flush House, Flush Five) is on offer, only those plays are
  allowed, at rate 1.0 times the fade; discards and other moves stay allowed. It fades over the same
  100,000 games as the expert pins. Why the early game: a Full House contains Two Pair, so every
  Two Pair joker still fires on it, and at level 1 it scores 40 x 4 against 20 x 2; Two Pair
  only overtakes it after two Uranus (level 3: 60 x 4). *(Corrected 2026-11-04 ~13:50: this line
  said "about three"; see the review below.)* The reward (`blinds`) pays nothing past a blind's
  target, so it doesn't separate the two while Two Pair clears the blind. This pin is an ought
  from chud, put into the early game, not into the reward.
- **Router minimum**: Full House's expert is picked by rule only after 2 Full House plays in antes
  1–3 (a single play against a near-zero base rate would otherwise win almost every such game).
- **The stall clock starts once every pin (experts' and the early one) is 0.** The 24-hour, now
  36-hour, cap is the backstop.
- Fresh start in `runs/moe2` (a checkpoint with 8 heads doesn't load into 7).

No prediction is registered for this run either. I1's test is still to be registered before scoring.


## Added 2026-11-04 ~13:35 CST: I2, Full House is one of the strongest experts (section I)

Recorded at 13:31 CST, minutes after chud started the second MoE run (`runs/moe2`), before any of its
evals existed.

### I2 (chud's prediction)
> "It might take a longer time for the weights to learn Full House, but I hypothesise it is one of
> the most effective experts at winning."

Two parts: the Full House expert learns more slowly than the others, and once it has, it is among
the experts with the highest win rate.

### Test (proposed by Claude; chud to confirm before any moe2 eval is read)
- **Which state is compared.** The router hands each expert different games: Full House gets runs
  that played Full Houses in antes 1–3. Comparing each expert's win rate over the games it was routed
  would mix the expert with the games it was given. Evals are deterministic per seed up to the
  handoff (fixed sampling RNG per seed), so the comparison can be paired: on every eval seed
  that reaches the handoff, replay the early model's moves to the handoff, then play the game out
  once with each of the 6 experts, from the same state. Every expert is scored on the same
  positions. Run on seeds outside the 600 eval seeds, with the final weights (all pins 0).
- **"One of the most effective"**: Full House's win rate ranks in the top 2 of the 6 experts.
  Reported overall and on the seeds the router itself sends to Full House.
- **"Takes longer to learn"**: the number of its own training games before the Full House expert's
  win rate in the per-expert eval stats first reaches the median expert's, compared with the other
  experts. Full House should need more.
- **Size**: win rates are around 1–2%; the number of seeds is set before the test is run so that the
  top-2 ranking means something (to be worked out, then recorded here).

Status: **proposed**. The test design needs chud's confirmation; nothing scored.

## Added 2026-11-04 ~13:50 CST: review of the early Full House pin (for a critical reader)

The pin is chud's decision. Its settings (rate 1.0, "Full House or higher", the router's minimum of 2
plays) are Claude's. This is the case for it and what it costs, written before any moe2 result.

**1. "Is Two Pair over Full House actually a mistake?"** In the hands where it mattered, yes.
- A Full House contains Two Pair (and Three of a Kind, and Pair), so every joker that fires on
  "contains Two Pair" (Clever, Mad, Spare Trousers) fires on the Full House as well. Same jokers,
  more base score, and one more scoring card.
- Base scores at level 1, from the game's hand table: Full House 40 x 4 = 160, Two Pair 20 x 2 = 40.
  Two Pair at level L is 20L x (L + 1): 120 at level 2, 240 at level 3. So Two Pair overtakes an
  unleveled Full House only after two Uranus.
- The evidence that the bot gets this wrong already accounts for levels. `docs/won_games_analysis.md`
  (model 0.2's 709 wins, the weights this MoE starts from) counts decisions where Full House had the
  highest chips x mult **at the hand levels of that moment**: 2,274. It played Full House 0 times and
  split it into Two Pair 2,075 times. A leveled Two Pair doesn't explain those; in each of them Two
  Pair was the lower hand by the game's own table.
- The first MoE run agrees: across 443 eval handoffs the router never picked Full House by rule.
  With Full House's base rate near zero, one Full House play in antes 1–3 would almost always have
  won the vote, so the early head played essentially none.
- Where Two Pair can be right: Square Joker (exactly 4 cards), effects of cards held in hand (a
  fifth card kept back), and Two Pair leveled twice or more. The pin covers the early game only,
  fades, and leaves discards open, so these cases still have a way out.

**2. "Why not leave it to exploration or to RL?"** Both were tried. Hand-type exploration
(`--explore-hands 0.05`, CHANGELOG 0.8.0) was added for exactly this and was on through 0.3 and the
first MoE run; the habit stayed. RL alone can't fix it: the reward is blinds beaten, and score past a
blind's target pays nothing. In antes 1–3 a Two Pair usually clears the blind, so both plays get the
same return and the gradient between them is zero. The habit came in by cloning v1 (CHANGELOG 0.7.1)
and nothing in the reward removes it.

**3. "Why not change the reward instead?"** The reward is the variable under study. The main
experiment (sections C and G) varies the ought and holds everything else fixed; a bonus for bigger
hands would put a second ought into every run, unrecorded in the reward of record. The pin leaves the
reward untouched, acts only on which plays are allowed, is logged per decision (`moe.early_pin`:
offered, bound, plays, discards while on offer), and goes to 0.

**4. "Why the early game?"** It's where the choice is cleanest and where it decides the most.
Levels haven't diverged yet, so the Full House is the higher hand almost every time. And the early
game feeds a loop: Two Pair played early sends the run to the Two Pair expert, which levels Two Pair,
after which Two Pair over Full House becomes correct. The Full House expert can only be reached
through early Full House plays. Pinning the experts alone (the first run's design) never reached it.

**5. "Rate 1.0 is heavy-handed."** It is, at the start. It fades linearly to 0 over 100,000 games,
the same schedule as the expert pins, and the stall stop waits for it. "Or higher" keeps it from
forcing a Full House over a Four of a Kind, Straight Flush or the secret hands. Discards stay legal,
so the bot can dodge the pin; `discard_offered` counts how often it does.

**6. "The router minimum of 2 is arbitrary."** It's a judgment call. With a near-zero base rate, a
single Full House play would outvote a run built on a dozen Pairs. One play is weak evidence that a
deck suits Full House; two is the smallest step up. The routing counts in every eval show whether
it's too strict (Full House starved) or too loose (Full House flooded).

**7. What the pin costs (stated, not defended).**
- **Path dependence.** After the fade, the weights still carry the pin's history. If moe2 plays Full
  House with the pin at 0, that does not show the reward led there; it may be a habit the pin
  installed and nothing removed, the mirror image of the Two Pair habit cloning installed. moe2 alone
  can't separate the two.
- **For the Hume aim**, the pin is an ought from chud entering through the action mask rather than
  the reward. It's recorded and it ends; it isn't nothing.
- **The control that would separate them:** the same code and settings with `MOE_EARLY_PIN=none`
  (a run `moe2-nopin`). Not registered or run yet. Without it, results from moe2 are about the MoE
  with an early Full House pin, not about the MoE.


## Added 2026-11-04 ~13:40 CST: completion criterion: the MoE has learned Full House

chud, 13:35 CST: "I am willing to make this entire MoE have to learn Full House before I say it's
complete." Recorded before any moe2 eval.

This is a criterion for calling the MoE finished, not a prediction, and it is about behavior: it is
met whether or not the early pin helped get there. Whether the reward alone leads there is a separate
question, for the no-pin control (review above, item 7).

### Measured (thresholds proposed by Claude; chud to confirm)
All with every pin at 0 (expert pins and the early pin), on the fixed eval seeds, with
`analysis/replay_events.py` + `analyze.py` on the saved eval games. "Full House on offer as the best
hand" uses the analysis's existing test: highest chips x mult at that moment's hand levels.
1. **The early head plays it.** In antes 1–3, when Full House is the best hand on offer, it plays a
   Full House (or a higher hand) at least **75%** of the time. Reference points from model 0.2's
   wins: Full House 0%, Flush 76%, Straight 92%. So the bar is "treats Full House the way it already
   treats Flush".
2. **The experts play it too.** Same test after the handoff, each expert separately: no expert below
   **50%** (Two Pair's expert included). Leveling can make Two Pair the better hand later in a run;
   the analysis's test already counts levels, so that case doesn't count against it.
3. **The Full House expert is in use.** Routed by rule in at least **5%** of eval handoffs (the first
   run: 0%).
4. **It holds.** 1–3 are met at two evals in a row, at least 20 rounds apart, after the pins reach 0.

I2 (Full House among the strongest experts) is separate and can fail while this criterion passes.

If moe2 ends without meeting it, the MoE isn't complete: next steps are decided then, with the
measurements in hand.


## Added 2026-11-05 ~10:40 CST: Full House not learned so far (moe2, through round 900)

An observation, not a scored test. The completion criterion above (2026-11-04 ~13:40) is measured
only once every pin is at 0, at about 100,000 training games (round ~1,041); at round 900 the pins
are at 0.13. Recorded now, before that eval exists, because chud asked to note that Full House
training has failed so far. The early head is **trained** in moe2 (`--moe-train-early 1`), not frozen.

### What the eval logs show (600 fixed seeds, every 20 rounds)
- **Neither the Full House expert nor the early head plays a Full House unless the pin forces it.**
  From round 540 to 900 (19 evals, pins 0.45 → 0.13): the Full House expert had a Full House on
  offer at 993 decisions; the pin bound 330 of them, and it played 326 Full Houses. The early head
  (pin: Full House or higher) had one on offer at 17,284 decisions; the pin bound 5,190, and it
  played 5,072. In every one of those evals, plays never exceeded the decisions the pin bound. That
  is what a policy that never chooses Full House on its own would give. These counters are per eval, not per
  decision, so they can't prove that no unforced Full House was ever played; the per-decision
  replay (`analysis/`) is the exact test, at pins 0.
- **The Full House expert is starved.** Games routed to it: 81 of the round-120 eval's handoffs, 87 at
  round 160, 60 at 400, 35 at 600, 18 at 800, **5 of 408 (1.2%) at 900**. The router needs two Full
  House plays in antes 1–3, and as the early pin fades the early head stops making them.
- Its eval record across all 46 evals: 2,207 routed games, 34 wins, 34 near wins. Not comparable with
  the other experts' records (each gets different games; I2's test is the paired comparison).

### Against the completion criterion, projected (not scored)
1. Early head plays it at ≥75% when it's the best hand: on course to fail (no unforced plays seen).
2. Experts play it at ≥50%: not measured (needs the per-decision replay), likely to fail.
3. Full House expert routed in ≥5% of eval handoffs: 1.2% at round 900 and falling; on course to fail.

### Not yet known
Whether, with the pins at 0, the reward alone leads any head to Full House. The first pin-free eval
(PR #95: `pins0.pt`, eval at the first round with every pin at 0) and the checkpoint at 100,200
games are the measurement points.

### An idea for later (chud, 2026-11-05)
Train each expert only on the games that expert won. Not designed or registered.


## Added 2026-11-05 ~11:00 CST: J, the next mixture of experts (design draft; nothing registered)

### What chud asked for (10:37–10:39 CST)
> "begin the formulation of a superior MOE based on the results of our currently running pin method,
> and we should also comb the winning strategies on the winning seeds and link it back to each
> expert. if we find out that a single expert is clearly better, (hopefully we can train a full
> house expert) we should probably have the early game model steer into the strategy that wins the
> most"

> "if the handoff is at the wrong blind, that'd be good to know. if the handoff doesn't work and
> doesn't modify strategy, that'd be good to know. if one strategy wins lots, that'd be good to know."

### Measurement first: `analysis/paired_experts.py`
Each expert's own record mixes the expert with the games the router gives it (Full House only gets
runs that already made Full Houses), so the logs can't link winning strategies to experts. The
tool plays every expert from the **same** handoff state: the early head plays a seed to the
handoff exactly as in an eval, then each of the 6 experts plays the rest. A seventh player is the
control: no handoff, so the early head plays on. Seeds are outside the eval 600. Repeated with the handoff at
other antes. What it gives, against chud's three questions:
1. **One strategy wins the most?** Each expert's win rate on the same positions, the positions on which each expert
   is best, the router's own choice, and the ceiling of a perfect router.
2. **Does the handoff change play?** The control's win rate against the experts', and each expert's
   plays after the handoff by hand type, from the same positions. Experts that play alike, or that don't beat
   the control, mean the handoff adds nothing.
3. **Is the handoff at the right blind?** All of it at handoff antes 3, 4 and 5 (moe2 trained at 4;
   at other antes the experts face positions they weren't trained from, so read it as a probe).
It also serves I2's paired test and the Full House completion criterion. First run: on `pins0.pt`
(moe2, every pin at 0).

### Design options, by what the measurement shows
- **One expert best from most positions:** a single expert may beat the mixture, and steering the
  early game toward its strategy is the right move.
- **The best expert depends on the position:** a learned router, which predicts each expert's outcome from the
  handoff state and is trained on the paired play-outs, instead of the hand-count rule. This also ends
  Full House's starvation, which comes from the rule (it needs Full House plays in antes 1–3).
- **Experts don't beat the control, or play alike:** the handoff isn't working; the experts need
  another design before anything is built on them.
- **Another ante is clearly better:** move the handoff there.

### How to steer the early game (Claude's recommendation)
- **Not with pins or a hand-specific bonus.** moe2's pins forced Full House for ~100,000 games, and
  the policy took none of it up: through round 900, Full House plays never exceeded what the pins forced
  (section above). A pin changes behavior while it binds; it doesn't change what the policy prefers.
  Steering by hand also writes a "how" into the ought, which aim 1 (behavior from the reward alone)
  can't afford.
- **Extend the early head's return through the handoff** (`--moe-cut 0`). Today its returns stop at
  ante 4 (chud's spec for moe1/moe2), so it is rewarded for reaching ante 4, not for reaching it
  in a position the experts can win from. With the return running to the end of the game, the
  reward alone favors whatever setups the experts convert. If a Full House expert wins most, the
  early head would learn to build toward it without being told to.

### Later
- chud's idea: train each expert only on the games it won. Close to self-imitation, which was turned off after it
  collapsed entropy (0.6.0). At ~1% wins the data is thin. Worth revisiting with the paired
  play-outs, which give more wins from good positions.

### Before moe3
Predictions about the paired results (chud's, if any) get registered here **before** the tool's first run
on `pins0.pt`. The scoring rules stay as written whatever the outcome, including if Full House doesn't win.

Status: **draft**.


## Added 2026-11-05 ~15:40 CST: moe2 at pins zero: results

moe2 stopped itself at 13:42 CST on 2026-11-05, as built (`--stop-after-pins0`, PR #98). The last eval
was at round 1042, the first round with every pin at 0, after 100,032 training games. Its weights are
released as `weights/balatro-bot-moe2-pins0.pt` ([system card](system_card_moe2.md)). Full numbers and
tables are in [moe2_evaluation.md](moe2_evaluation.md). This section scores what was registered and
records the findings. The analysis is Claude's, done on chud's request ("update the hypothesis and
write up the evaluation").

### Pins-zero eval (600 fixed seeds)
12.94 blinds per game, 9 wins, 13 near wins, 22 reached the final boss. Round 0 was 11.59 with 2 wins.
The run's best eval was 13.08 at round 1020, and its most wins 14 at round 1040.

### Completion criterion (2026-11-04 ~13:40): **not met**
1. **Early head plays Full House when it's the best hand, at least 75%: fail.** The early pin's counters
   at pins zero: a Full House or higher was on offer at 1,005 early-head decisions, and it played one
   once. The criterion's exact measure, "best hand at that moment's levels", uses a subset of those
   decisions, so its numerator is at most 1. The per-decision replay wasn't run; it can only confirm.
2. **Every expert at least 50%: not measured.** The Full House expert got no games. The others need the per-decision
   replay.
3. **Full House expert routed in at least 5% of eval handoffs: fail.** 0 of 442.
4. **Holds at two evals: not applicable,** since items 1 and 3 fail.

As the criterion says, the MoE isn't complete, and next steps are decided with these measurements
in hand. Full House's routing fell 81 → 5 → 0 (rounds 120, 900, 1042), as projected on 2026-11-05
~10:40.

### I2 (Full House among the strongest experts): **still unscored**
The evals can't test it: the router sent the Full House expert nothing. The paired test can, because
`paired_experts.py` forces every expert from the same handoff state, Full House's included. Its
design still needs chud's confirmation (2026-11-04 ~13:35).

### I1 (MoE beats model 1.0): **still unregistered**
No comparison exists. moe2 used 600 eval seeds and the earlier models 200, under a win test since
corrected. 1.0 is still being re-measured.

### Findings (details and tables in the evaluation)
- **The pins didn't teach the heads their hands.** With the pins at 0, the early head played Full House
  1 time in 1,005 chances. Experts play their own hand when it's on offer at these rates: Two Pair
  99.4%, Straight 78%, Flush 42%, Three of a Kind 37%, Pair 7%. Two Pair's near-total rate can't be
  credited to the pins without a baseline: model 0.2, where every head started, already leaned on
  Two Pair. The paired tool's no-handoff control gives that baseline.
- **The mid-run dip was the early game, and it recovered without the pins.** Training games ending
  by ante 3 went 33% → 42.5% (rounds 200–599) as the pins faded through 0.8–0.5. They were 30.6% by
  rounds 1000–1043, below the start. The learning rate halved at round 611 (update guard), close to
  the turn, so that cause isn't separated.
- **Blinds per game rose to the run's best by the end; wins didn't clearly.** Last six evals: 12.93 blinds and 8.3 wins on
  average, against 12.49 and 8.2 at rounds 100–200. Fresh-seed training wins: 1.12% (rounds
  100–199), 1.13% (1000–1043). Eval trend since round 600: +0.51 blinds per 100 rounds.
- **Most games are lost in antes 4–6** (352 of 600 at pins zero), and 56 by ante 1. Across all moe2
  games, each expert converts 15.5–20.7% of its final-boss fights.
- **Unexplained:** `ppo.kl` (policy that played vs policy being updated) rose 0.01 → 0.19 between
  rounds 200 and 600, while each update's own movement stayed about 0.025. One update of lag accounts for
  about 0.025. The suspects are hand-type exploration and float32 flooring; neither is tested.

### For aim 1 (Claude's reading, for chud to accept or reject)
The pins were an ought entered through the action mask, not the reward, and recorded as such
(review of 2026-11-04 ~13:50, item 7). With them at 0, the forced behavior is mostly gone. The bot
lost ground as they faded, then rebuilt on the reward and passed its starting point in blinds per game,
toward play the pins didn't ask for. The review's caveat stands: the weights carry the pins'
history, so this is a statement about this run, not about what the reward alone does from 0.2. The
no-pin control (`moe2-nopin`) is still unrun.

### Next (section J; nothing registered)
1. `analysis/paired_experts.py` on `pins0.pt`. chud's predictions, if any, are registered in J first.
2. Split `ppo.kl` by explored and sampled moves, and a low-temperature eval of `pins0.pt`.
3. Then `--moe-cut 0` from `pins0.pt`, as a MoE or a single network depending on 1.


## Added 2026-11-05 ~15:45 CST: Claude's predictions for the first paired run (section J)

Registered before `analysis/paired_experts.py` has run on anything. These are Claude's predictions,
made at chud's invitation ("would you like to make predictions?"); chud registered none. The run:
`pins0.pt`, 200 seeds (`--seed-tag paired-v1`, outside the eval 600), one rollout, handoff antes 3, 4 and 5.

### What this run can and can't resolve
About 74% of seeds reach the ante-4 handoff (442 of 600 in the pins-zero eval), so each player gets
about 150 games at ante 4. At the ~2% win rate after the handoff, that is about 3 wins each.
**Win rates can't be ranked at this size**, so none of the predictions below uses them; they use
blinds per game and the play-share table. Ranking win rates would take roughly ten times the seeds
(an overnight run). The tool's rows for different handoff antes are also not on the same seeds:
ante 5 only counts seeds that reach ante 5, which are stronger runs. So comparing antes in its
summary is biased, and no prediction is made about the best handoff ante until the summary compares
antes on the seeds common to all three.

### Predictions (all at handoff ante 4; scored from the tool's summary)
- **C1. The experts are not redundant (70%).** In the play-share table, for at least 3 of the 6
  hands, the expert named for that hand has a higher share of it than every other player (the 5
  other experts and "no handoff"). Expected among them: Two Pair, Straight, Flush. Not expected:
  Pair. *Falsified* if fewer than 3.
- **C2. The handoff beats playing on with the early head (60%).** "No handoff" averages at least
  0.5 fewer blinds per game than the mean of the 6 experts. Reason: the early head's return has
  ended at ante 4 for 100,000 games, so nothing has trained its play past there. *Falsified* if the
  gap is under 0.5 or the other way round.
- **C3. The experts are close (65%).** Between the best and worst expert, blinds per game differ by
  less than 1.0. Reason: across all of moe2's games, final-boss conversion spans only 15.5–20.7% by
  expert. *Falsified* at 1.0 or more.
- **C4. The router adds nothing over picking an expert at random (65%).** "Router (its own choice)"
  is within 0.3 blinds per game of the mean of the 6 experts. *Falsified* if it is 0.3 or more above.
- **C5. Full House is not among the top 2 experts by blinds per game (65%).** This bets against
  I2's second half on this measure (I2 is about win rate, which this run can't rank).

Any prediction whose condition can't be read from the summary (e.g. a player with no games) is
scored as void, not as passed.
