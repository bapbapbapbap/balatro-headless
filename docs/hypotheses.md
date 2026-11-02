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
