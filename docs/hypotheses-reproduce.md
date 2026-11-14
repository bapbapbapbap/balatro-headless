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
