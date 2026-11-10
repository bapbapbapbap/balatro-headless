# Hypotheses: Model v4b (current)

New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-10 ~10:30 CST: model v4b, registered before any of its code (section J)

v4 with two changes, from random weights, after [v4's diagnosis](hypotheses-v4.md#added-2026-11-10-1030-cst-why-v4-stalled-diagnosis-after-the-run-section-j).
chud (10:22): "a solution category of reduce the damage to learning ... I'm totally ok restarting from scratch."
Everything else is v4 as registered and defined (phase split, phi_build, offer values, curriculum, shop entropy
schedule, gamma's blind clock, reward `blinds`, lr 1e-4, Red Deck, White Stake, the 600-seed eval every 20 rounds,
`--stall-evals 25`).

### What changes
1. **A hand clock for the table network's lambda (`--lam-hand 0.8`).** GAE lambda on the table network's credit
   is multiplied by 0.8 at every play or discard, on top of 0.95 per blind passed; every other move (joker drags,
   sells, consumables, every shop-side move) leaves it unchanged. The shop network's lambda stays v4's.
   - Why: with lambda = 1 inside a blind, the cost of dying lands at full strength on every move of the final
     blind (v4: -0.9 standard deviations on 29% of table decisions), and that is what moved probability onto
     selling. With 0.8 per hand, a move 7 hands before the death carries 0.8^7 = 0.21 of it.
   - Why plays and discards, not every move (chud, 10:27): joker order sets scoring order (Blueprint, Brainstorm,
     +mult before x-mult), so a drag pays off at the next scoring hand; it shares that hand's credit undecayed.
   - Why not the shop network: its moves pay off over blinds, not hands; its credit stays on the blind clock.
   - Lambda changes how credit is spread over moves, not the return: the reward, the shaping and the optimal policy
     are unchanged, so the ought stays `blinds`. Cost: advantages lean more on the critic (bias that shrinks as
     the critic improves).
2. **`--explore-hands` off.** Its stored log-probabilities fell to the floor (v4: 13% of table plays by round 300,
   no gradient from them). It was for the Full House question, closed 2026-11-09.

### Added: a measured stop
Eval and training metrics carry `sells_mean` (jokers sold per game). **`--stop-sells 1.5`: the run stops itself
at the first eval averaging more than 1.5 joker sells per game** (v4: 0.50 at its best eval, 1.91 at round 140).
That eval is the failure of change 1.

### Predictions (Claude's, 2026-11-10)
- **S1. The fix holds:** no eval averages more than 1.5 sells per game before the run ends. **70%.**
- **S2. v4b beats v4:** its best 600-seed eval is above 4.68 blinds/game. **65%.**
- **S3. v4b reaches v1's pace:** its eval reaches 5.58 blinds/game (v1 at round 50) by round 200. **35%.** If it
  holds S1 but misses S3, the architecture costs more than the credit fault explains.
- W1-W4 apply to v4b's `best.pt` as registered for v4 (same tests, seeds and thresholds; W4b's 16.71 h on v4b's
  own play + update time). Claude's odds after v4: W1 25%, W2 8%, W3 25%, W4a met 10%, W4b met 5%, aim 1's
  precondition 5%.
