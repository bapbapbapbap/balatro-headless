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
1 missed (R7). Every miss went the same way: seeds matter more than Claude predicted.

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
