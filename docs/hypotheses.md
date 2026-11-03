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
