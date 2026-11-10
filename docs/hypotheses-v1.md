# Hypotheses: Run v1 (model 0.1)

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

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
