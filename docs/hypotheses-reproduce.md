# Reproducing other bots

The bots that beat ours: does their result hold on the game's own code, and where exactly does their play beat
ours? Work, code and results so far: [`reproduce/balatro_lab/README.md`](../reproduce/balatro_lab/README.md).
chud's two questions: (1) did the bots that beat us simulate the game faithfully enough for their results to
count? (2) where does their bot beat ours, and which of those differences can ours learn by itself?

## Added 2026-11-14 ~15:55 CST: Balatro Lab, what is already measured, and two tests registered before they run

Claude's, reviewed by chud. Balatro Lab is KvashaIhor's (github.com/KvashaIhor/balatro-lab, MIT, pinned at commit
3e09d47e); none of its code is in this repository. It reports 70.2% wins (Red Deck, White Stake, beat the ante 8
boss) on its seed set 90210, the 480 seeds its bot was hand-tuned and evolved on, and 72.5% on a second set.

**Already measured (before this section; data in `reproduce/balatro_lab/results/2026-11-14/`):**
- Its own simulator, its own bot, its 480 seeds, on chud's laptop: 340/480 = 70.8% (mean progress 22.43).
- Its simulator against our 1.0.1o code: run openings 100/100 seeds; hand scoring 4,143/4,143 chance-free hands;
  a lockstep twin over 60 seeds followed 44 whole runs exactly (3,018 decisions); every other stop was a chance
  effect it rolls with its own randomness, or a small gap that does not raise wins (README lists them).
- Answer to question 1 for Balatro Lab: yes. The 70% is not a simulator artifact.
- Disagreement mining, step 1 (`mine.py`, 11 seeds, 774 decision points, cloud, stock LuaJIT): v4b best.pt
  (round 1840, 13.40 blinds/game) and Balatro Lab's bot make the same choice at 33% of decision points (blind 89%,
  hand play ~10%, shop ~37%, pack ~57%). Who is right is not measured yet.

**Test T1: no cherry-picking or seed tuning.** Balatro Lab's bot (`--bot base`), in its own simulator, on our 600
evaluation seeds (`train/common.py eval_seeds(600)`, fixed RNG "balatro-eval-v1", chosen long before we knew of
Balatro Lab). Script `reproduce/balatro_lab/eval_on_our_seeds.py`. Outcome: wins and blinds beaten per game. (A 4-game smoke test of the script ran
before this section: 3 of 4 won; those 4 seeds are the first 4 and count in the result.)
- T1a: win rate >= 60%. Claude 85%.
- T1b: win rate within 5 points of its 70.8% on set 90210. Claude 70%.
- T1c: blinds beaten per game (skipped blinds not counted, as `blinds_mean` in our evals) > v4b's 13.40 by at
  least 5. Claude 95%.
chud's predictions: to be added in a section of his own.

**Test T2: who is right where the bots disagree (disagreement mining, step 2).** Design, before its code:
- Points: every decision point where the two bots disagree, from `mine.py` on seeds 1-100 of Balatro Lab's set
  90210 (`--rng 90210`), run on chud's laptop (LuaJIT 2.0.5). 600 of them drawn uniformly at random (seed 0); all
  of them if there are fewer.
- Settling one point: rebuild the twin at that point (replay the route), copy it twice, apply our bot's choice to
  one copy and Balatro Lab's to the other, then let Balatro Lab's bot play both copies on to the end of the run
  (ante 8), with the seed's own future (seeded draws). Outcome per copy: blinds beaten, ante 8 boss beaten =
  24. The point's score: theirs minus ours.
- Known bias, stated now: the continuation is their bot in both copies, which favors choices their bot knows how to
  follow up. Our bot can't play inside their simulator, so this is the available control. Read results as "which
  choice is better for a strong continuation", not "which our bot should make".
- Reported per category (blind skip/take; hand: they discard / we play, we discard / they play, both play with
  different scoring cards, kicker-only; shop: the pairs in `mine_report.py`; pack): mean score, 95% interval by
  bootstrap over seeds. A category is scored only with at least 30 points.
- T2a: pooled over all points, mean score > 0 (theirs better) with the interval above 0. Claude 85%.
- T2b: "they discard / we play": theirs better, interval above 0. Claude 65%.
- T2c: kicker-only differences: interval contains 0. Claude 70%.
- T2d: shop "they buy a joker / we open a pack": theirs better, interval above 0. Claude 60%.

## Added 2026-11-14 ~16:15 CST: T1 withdrawn; T2 replaced by ranking test R; how v5 is chosen

Claude's, reviewed by chud, except where marked chud's.

**T1 withdrawn before it ran** (chud, 16:00: why spend the laptop on it). Balatro Lab's held-out second set
(72.5%) and the fidelity results above already speak against seed tuning; the hours go to v5 instead.

**T2 replaced, before any of its code, by R.** T2 judged a choice by letting Balatro Lab's bot play on from it,
which favors its own choices. R measures directly, on our game code, what v4b would gain if one kind of decision
were made the way Balatro Lab's bot makes it.

**Correction to the 11:00 plan:** "power as the shaping potential" was listed as untried; v4/v4b already train
with it (`harness/build_phi.lua`: phi_build, mean best score over 16 deals, scaling jokers projected 6 rounds),
and v4c added offer values. What v4b lacks is choosing by computed values: no score per candidate play, no search.

**R, the ranking test.**
- Seeds: our eval seeds 1-300 (`train/common.py eval_seeds(600)[:300]`), Red Deck, White Stake, 1.0.1o, on the
  laptop (LuaJIT 2.0.5).
- Arms (same seeds, paired): A0 v4b best.pt alone (round 1840), playing as in its evals; A1 hand play (play,
  discard) handed to Balatro Lab's bot; A2 shop (buy, buy and use, redeem, open pack, reroll, leave, and selling
  or using in the shop); A3 pack picks; A4 blind select or skip; A5 all of them (Balatro Lab's bot plays the real
  game: its win rate on our code).
- How a decision is handed over: Balatro Lab's seeded run follows the game in lockstep (`twin.py`); at a handed-over
  decision its bot chooses on a copy of the twin's state (as `mine.py`), and that choice is played in the game as
  the matching harness move. v4b makes every other move. Random draws the twin rolls differently are copied from the
  game (as in `mine.py`); where the twin can't follow at all, v4b plays the rest of that game and the game is
  counted, with the share of such games reported per arm.
- Check before any arm is scored: A0 must reproduce v4b's own eval games (same blinds beaten, every seed).
- Outcome: blinds beaten per game (skips not counted, as `blinds_mean`) and wins, each arm minus A0 paired by
  seed, 95% interval by bootstrap over seeds.
- Claude's predictions: R1 A1 gains the most blinds of A1-A4, 55%. R2 A1 gains at least 1.0 blind/game, interval
  above 0, 70%. R3 A2 gains, interval above 0, 60%. R4 A4 within 0.3 blinds/game of A0, 70%. R5 A5 wins at least
  60% of games, 75%.
- chud's prediction (stated 16:05, his words): "given the reproduced 340/480 = .7083 win rate, I believe aligning
  our bot's behavior with its behavior more will increase our win rate, which [will] give the best win rate we've
  gotten so far." Scored on v5: v5's best eval win rate above every earlier model's best eval win rate (the number
  to beat is looked up from the record and written into v5's registration before v5's code).

**How v5 is chosen (rule, fixed now).** v5 = v4b's design, warm-started from v4b best.pt, plus the computation
behind the arm with the largest blind gain in R (ties: the larger win gain): for hand play, each legal play's and
discard's exact value from the game's own scoring as an action input; for the shop, each offer's computed value
(v4c's offer values, extended to what R shows); never Balatro Lab's hand-set values or its weights, and no
imitation of its moves (those would make the oughts its, not ours). If no arm gains with its interval above 0, v5
is not built from R and the next step is reconsidered with chud. v5's design and predictions are registered after
R and before v5's code.
