# moe2: evaluation at pins zero

Written 2026-11-05 (~15:40 CST), after the run stopped itself at pins zero. Design and settings:
`hypotheses.md` sections I and J and the 2026-11-04 ~13:40 entry. Released weights and how to use
them: the [system card](system_card_moe2.md). This document is the record of what the run showed.
Hypotheses and the completion criterion are scored in `hypotheses.md` (2026-11-05 ~15:40).

**Short version.** The mixture of experts played its best evals at the end of the run, after every
pin had faded, but on win rate it hasn't clearly passed its own early peak. The pins didn't teach the
experts their hands. The Full House expert went unused (moe2 has no Four of a Kind expert). Two Pair's expert plays Two Pair at
almost every chance it gets, but the logs can't say whether the pins put that there. Most games are
lost in antes 4–6. Once a game reaches the final boss, each expert wins it at about the same rate.
One training statistic changed sharply in mid-run, in the same window as the dip, and isn't
explained yet.

## What moe2 is
A shared transformer trunk (d=192, 4 layers, 6 heads) with 7 heads: an early-game head for antes
1–3, and one expert for each of Pair, Two Pair, Three of a Kind, Straight, Flush and Full House.
2,281,742 parameters, all started from model 0.2. At the handoff (ante 4, the ante-3 boss beaten)
a rule-based router picks the expert whose hand the run played most in antes 1–3, relative to how
often the early head usually plays it. Full House needs at least 2 plays to be picked. In
training, 30% of handoffs go to a random expert; in evals, all go by the rule.

"Pins" restricted each head's plays toward its hand for part of training: experts toward their
own hand (Pair at half rate), the early head toward Full House or higher. They faded linearly to 0
over 100,000 training games. Reward: blinds beaten (`blinds`) with potential-based shaping, as for
every model since 0.1. Red Deck, White Stake, unmodded Balatro 1.0.1o-FULL.

Run: launched 2026-11-04 13:33 CST, stopped 2026-11-05 13:42 CST at pins zero (`--stop-after-pins0`).
That is 1,043 rounds and 100,224 training games across 11 launches, all resumed from `latest.pt`,
on code from PR #77 to PR #98. 947 training wins. One update was rejected, at round 611 (KL 0.224 > 0.2),
and the learning rate halved to 5e-6.

## Evaluation protocol
600 fixed seeds (Red Deck, White Stake), every 20 rounds, plus one extra eval at the first round
with every pin at 0 (round 1042). Moves are sampled from the policy with a fixed random stream per
seed, so an eval is reproducible from its weights. **Near win**: reached the Ante 8 boss and lost.
**Final boss** = wins + near wins. Outcomes are the game's own (`win_game()`, the `Defeated` mark),
as since CHANGELOG 0.13.2.

Noise: at about 1.5% wins, binomial chance alone moves an eval's win count by about ±3. Single
evals are weak evidence; compare windows of several evals.

## The pins-zero eval (round 1042, `pins0.pt`)

| | |
|---|---|
| Blinds beaten per game | **12.94** |
| Wins / near wins / reached the final boss | **9 / 13 / 22** of 600 |
| Games ending by ante 3 / in antes 4–6 / 7–8 / 9 | 159 / 352 / 80 / 9 (56 by ante 1) |
| Handoffs | 442 of 600 |

Final ante counts, all 600: ante 0: 1, ante 1: 55, ante 2: 58, ante 3: 45, ante 4: 86, ante 5: 157,
ante 6: 109, ante 7: 66, ante 8: 14, ante 9: 9.

## Evals across the run

| Round | Pin scale | Blinds/game | Wins | Near wins | Final boss | Ended by ante 3 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 11.59 | 2 | 13 | 15 | 183 |
| 100 | 0.90 | 12.37 | 11 | 11 | 22 | 158 |
| 120 | 0.88 | 12.57 | 11 | 14 | 25 | 158 |
| 160 | 0.85 | 12.65 | 7 | 8 | 15 | 161 |
| 200 | 0.81 | 12.34 | 6 | 15 | 21 | 176 |
| 300 | 0.71 | 12.25 | 3 | 9 | 12 | 179 |
| 400 | 0.61 | 11.21 | 5 | 9 | 14 | 227 |
| 500 | 0.52 | 10.85 | 3 | 6 | 9 | 240 |
| 600 | 0.42 | 10.32 | 2 | 4 | 6 | 260 |
| 700 | 0.33 | 11.73 | 6 | 9 | 15 | 206 |
| 800 | 0.23 | 11.56 | 4 | 6 | 10 | 209 |
| 900 | 0.14 | 12.18 | 6 | 12 | 18 | 193 |
| 1000 | 0.04 | 13.01 | 3 | 19 | 22 | 172 |
| 1020 | 0.02 | **13.08** | 10 | 15 | 25 | 164 |
| 1040 | 0.001 | 12.82 | **14** | 15 | 29 | 181 |
| 1042 | 0 (pins zero) | 12.94 | 9 | 13 | 22 | 159 |

All 54 evals are in `runs/moe2/train.log` (`[eval N]` lines) and `metrics.jsonl`. Summary:
- **Best blinds per game: 13.08 (round 1020).** Most wins: 14 (round 1040).
- **Last six evals (rounds 960–1042): 12.93 blinds, 8.3 wins on average.** Rounds 100–200 (6 evals): 12.49 and
  8.2. Blinds per game rose; wins per eval did not clearly rise.
- **Trend since round 600: +0.51 blinds per game per 100 rounds** (least squares over the 24 evals from round 600 on). It
  had not flattened when the run stopped.

## Over training (100-round windows, training games)

| Rounds | Pin scale (start) | Blinds/game | Fresh-seed wins | Ended by ante 3 | KL vs playing policy | KL of the update |
|---|---|---|---|---|---|---|
| 0–99 | 1.00 | 11.41 | 29 / 4,951 = 0.59% | 35.4% | 0.008 | 0.014 |
| 100–199 | 0.90 | 11.64 | 54 / 4,832 = 1.12% | 33.8% | 0.009 | 0.019 |
| 200–299 | 0.81 | 11.78 | 39 / 4,911 = 0.79% | 33.0% | 0.015 | 0.026 |
| 300–399 | 0.71 | 11.19 | 44 / 4,821 = 0.91% | 38.4% | 0.046 | 0.029 |
| 400–499 | 0.61 | 10.67 | 22 / 4,804 = 0.46% | 42.2% | 0.100 | 0.032 |
| 500–599 | 0.52 | 10.31 | 19 / 4,840 = 0.39% | 42.5% | 0.160 | 0.028 |
| 600–699 | 0.42 | 10.69 | 21 / 4,947 = 0.42% | 40.5% | 0.188 | 0.022 |
| 700–799 | 0.33 | 11.46 | 24 / 4,835 = 0.50% | 36.9% | 0.176 | 0.024 |
| 800–899 | 0.23 | 11.50 | 45 / 4,813 = 0.93% | 37.4% | 0.189 | 0.023 |
| 900–999 | 0.14 | 11.98 | 56 / 4,797 = 1.17% | 34.5% | 0.194 | 0.022 |
| 1000–1043 | 0.04 | **12.39** | 24 / 2,119 = 1.13% | **30.6%** | 0.178 | 0.025 |

Blinds/game and "ended by ante 3" are over all training games, including retried seeds. Fresh-seed
wins are from the fresh arm only, with retried seeds excluded. Over the whole run the fresh arm had
377 wins in 50,670 games (0.74%).

### 1. The mid-run dip was the early game
Between rounds 300 and 660 the evals fell from about 12.3 to 10.3 blinds per game. The losses came
early. Training games ending in antes 1–3 rose from 33% to 42.5% while the early pin faded from 0.71 to 0.42.
By rounds 1000–1043 that share was down to 30.6%, lower than at the start. The early head lost
ground as its pin went away and then recovered without it.

Two things changed near the turn and can't be told apart from these logs. The pins kept fading,
and at round 611 the update guard halved the learning rate. Recovery shows from about round 680.

### 2. A training statistic changed sharply (unexplained)
"KL vs playing policy" is `ppo.kl`: the mean of (log-prob when the move was played − log-prob under
the weights being updated), over the round's moves. "KL of the update" is `ppo.kl_update`: how far
this update itself moved the policy. Between rounds 200 and 600 the first rose from 0.01 to 0.19
and stayed there, while the second stayed near 0.025 the whole run. With `--overlap` the weights that played are one update older than the weights being
updated, so one update's movement (~0.025) is the expected gap. The rest is not explained.

Candidates, neither tested:
- **Hand-type exploration** (`--explore-hands 0.05`). An explored move is stored with the policy's
  own log-prob (by design, CHANGELOG 0.8.0), but it wasn't sampled from the policy, so it adds to
  this mean without being a KL. When its advantage is negative and the ratio is large, PPO's clip
  doesn't bound the step. As the pins faded, more decisions became open to exploration.
- **Precision.** Workers compute probabilities in float32 and take `log(max(p, 1e-30))`, so very
  small probabilities are floored.

The rise falls in the same window as the dip. That timing is all that links them.

## The experts at pins zero (round 1042)

| Expert | Games routed | Wins | Near wins | Blinds/game | Own hand on offer | Played it | Rate |
|---|---|---|---|---|---|---|---|
| Straight | 119 | 2 | 1 | 15.75 | 403 | 316 | 78% |
| Three of a Kind | 107 | 2 | 5 | 15.64 | 379 | 141 | 37% |
| Two Pair | 100 | 3 | 2 | 15.81 | 1,264 | 1,257 | **99.4%** |
| Flush | 85 | 0 | 2 | 15.86 | 252 | 107 | 42% |
| Pair | 31 | 2 | 3 | 17.97 | 1,064 | 73 | 7% |
| Full House | **0** | – | – | – | – | – | – |

"Own hand on offer" counts decisions where a play of the expert's hand type was legal. "Played it"
counts plays of that type. Both come from the pin counters, which count even at pin rate 0. Pair
discarded at 410 of its 1,064.

- **Full House got no games.** Its routing fell from 81 of the round-120 eval's handoffs to 5 at
  round 900 and 0 here. The router needs two Full House plays in antes 1–3, and the early head made
  none (next section).
- **Two Pair's expert plays Two Pair almost every time it can.** The data can't say whether the
  pins put that there. Model 0.2, which every head started from, already leaned on Two Pair.
  Without a baseline (what the other heads play from the same positions), this can't be read as
  specialization the pins taught. The paired tool's no-handoff control is that baseline.
- **Blinds/game by expert is not a comparison of experts.** The router gives each one different
  games. `analysis/paired_experts.py` compares them from the same positions: see "The paired
  comparison" below.

### Full House in the early game
The early pin's counters at pins zero: a Full House or higher hand was on offer at 1,005 of the early head's
decisions, and it played one **once**, out of 9,149 plays.

## Where it wins and loses (all moe2 games)
From `wins.jsonl` (1,242 games: 950 training, 292 eval; 939 distinct seeds) and `near_wins.jsonl`
(5,648 games). Both include retried seeds.

**Final-boss conversion by expert**, wins / (wins + near wins):

| Expert | Wins | Near wins | Conversion |
|---|---|---|---|
| Two Pair | 225 | 864 | 20.7% |
| Straight | 235 | 964 | 19.6% |
| Three of a Kind | 232 | 993 | 18.9% |
| Full House | 124 | 591 | 17.3% |
| Flush | 206 | 1,034 | 16.6% |
| Pair | 220 | 1,202 | 15.5% |

These span 15.5% to 20.7%. Most games end before this point. At pins zero, 352 of 600 eval games
ended in antes 4–6, against 22 that reached the final boss.

**Jokers.** Wins and near wins hold the same core jokers, each present at the end in 35–50% of
both: Banner (42.2% of wins / 49.6% of near wins), Spare Trousers (40.7 / 33.2), Green Joker
(37.4 / 43.1), Blue Joker (36.4 / 41.4), Bootstraps (36.0 / 27.3). Cavendish is the clearest
difference: 14.9% of wins, 6.0% of near wins. 4.92 jokers held at the end of a win, 4.75 at the end
of a near win.

## What this does and doesn't show

**For a better bot:**
- **More training raises blinds per game.** This is the one finding with direct evidence behind it: the
  trend since round 600 and the end-of-run evals. Whether it also raises win rate is not shown.
  Fresh-seed wins at the end (1.13–1.17%) equal rounds 100–199 (1.12%).
- **Untested ideas:** extending the early head's return through the handoff (`--moe-cut 0`, section
  J), aimed at the antes 4–6 losses; and whatever the KL rise turns out to be.
- **What not to spend on:** more work on the expert structure. Final-boss conversion barely differs
  by expert, the Full House expert went unused, and from the same positions the experts play the
  same hands and don't differ in win rate (the paired comparison below).

**For aim 1 (the is-ought gap):** the ought never changed. It was `blinds` from start to finish. The
pins were a constraint on the action mask, not an ought: they told the bot what it may play, never
what it wants. When they were lifted they left nothing behind. The Full House pin left 1 Full House
play in 1,005 chances, and in the paired comparison all six experts play the same hand mix as each other
and as the early head, which was never pinned. Under one reward, a constraint that isn't the reward
doesn't persist. That is the negative control for the real experiment (rewards varied, the facts held
fixed; `docs/hypotheses.md`, 2026-11-05 ~21:15 CST).

**Not shown:** whether the MoE beats a single network. I1 (MoE vs model 1.0, paired held-out seeds)
is still unregistered. moe2's evals use 600 seeds and the earlier models' used 200, under a win test
since corrected, so the numbers don't compare directly.

## The paired comparison (2,000 seeds, 2026-11-05)
`analysis/paired_experts.py` on `pins0.pt`: 2,000 seeds outside the eval 600 (tag `paired-v1`), the
handoff at antes 3, 4 and 5, every expert and a no-handoff control from the same state. 0 prefix
mismatches. Scored with `analysis/score_paired.py`; predictions C1–C5 and chud's three questions are
scored in `docs/hypotheses.md` (2026-11-05 ~21:10 CST).

**Handoff at ante 4** (1,368 seeds reached it):

| Player after the handoff | Wins | Blinds/game | Pair | Two Pair | Three | Straight | Flush | Full House |
|---|---|---|---|---|---|---|---|---|
| no handoff (early head) | 11 | 15.02 | 4.9% | 65.0% | 6.9% | 14.3% | 7.0% | 0.0% |
| Pair | 28 | 15.97 | 5.7% | 64.4% | 6.9% | 13.7% | 6.9% | 0.0% |
| Two Pair | 27 | 16.14 | 5.4% | 64.7% | 6.7% | 14.1% | 7.1% | 0.0% |
| Three of a Kind | 34 | 16.11 | 5.2% | 64.8% | 6.9% | 13.6% | 7.1% | 0.0% |
| Straight | 38 | 16.21 | 5.0% | 64.8% | 6.9% | 14.2% | 7.1% | 0.0% |
| Flush | 30 | 16.08 | 5.3% | 64.4% | 6.9% | 13.9% | 7.2% | 0.0% |
| Full House | 25 | 16.08 | 5.0% | 65.6% | 6.8% | 13.0% | 7.5% | 0.0% |
| router's own choice | 31 | 16.12 | | | | | | |

The hand columns are each player's share of its plays after the handoff.

- **The experts play the same hands.** From the same positions, every expert and the early head play
  the same mix within about a point. On this measure, at pins zero the six experts are one policy.
- **They don't differ in win rate** (Cochran's Q p = 0.24 at ante 4) and barely in blinds (0.24
  between best and worst, no pair past Bonferroni).
- **Any expert beats the early head after the handoff**: +1.08 blinds per game, about 2.7× the wins.
  The early head was never trained past ante 4.
- **Earlier handoff is better.** As a policy over all 2,000 seeds: ante 3 12.65 blinds per seed,
  ante 4 12.49, ante 5 12.11, never 11.75. Antes 1–2 untested.
- **The router adds nothing** over a random expert (+0.02 blinds).

## Next
1. Decide the next run (section J) from the paired result: a single network (`--moe-cut 0` without
   the expert split), and/or a paired run with `--handoff-antes 1,2` (should an expert play the
   whole game?).
2. Split `ppo.kl` by explored and sampled moves; a low-temperature eval of `pins0.pt` (how much of the
   ante-1 loss is sampling).

## Data
`C:\Users\Jacobi\balatro-work\runs\moe2\` on chud's machine: `train.log`, `metrics.jsonl`, `wins.jsonl`,
`near_wins.jsonl`, `pins0.pt`, `games_100200.pt`, `provenance.json`. The tables above were computed
from these files on 2026-11-05. Wins are published in `wins/` (790 new seeds from moe2).
