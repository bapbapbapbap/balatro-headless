# Model v5

v4b plus what Balatro Lab's bot computes when it plays a hand: the score of every play it could make. Chosen by the
rule registered before ranking test R (docs/hypotheses-reproduce.md, 2026-11-14 16:05) from R's result
(2026-11-15 section there): v4b with Balatro Lab's bot playing its hands beat 3.34 more blinds per game
[+2.56, +4.14] on 300 eval seeds; no other part of Balatro Lab's play helped v4b.

## Added 2026-11-15, after chud's 09:56 CST message: v5 registered before any of its code

Claude's design, reviewed by chud; chud's prediction marked as his.

**What it is.** v4b exactly (configs/v4b.toml: two networks, encoding 2, ordered-capped actions, offer values,
blinds reward, build shaping, v4b's credit and sampling), warm-started from v4b's best.pt (round 1840, 13.40
blinds/game on the 600 eval seeds), plus two numbers on every move in the action list, the play values:
- For a play: S = what the play scores now, from the game's own scoring functions (build_phi.lua's score(), the
  steps of G.FUNCS.evaluate_play), with the played cards in the order the move leaves them (its lead), the rest of
  the hand held, and the blind being played applied (its debuffs as the game sets them, its debuff_hand rule, its
  modify_hand rule). Chance at its midpoint (as phi_build: rarer than 1 in 2 never fires). Left out, as in
  phi_build: what the play itself changes before scoring (jokers that grow when a hand is played), destruction,
  dollars. N = the blind's target minus the chips already scored this round.
  First number: clamp(log2((S + 1) / (N + 1)), -12, 3) / 4. Second: 1 if S >= N, else 0.
- Every other move (discards included): 0, 0.
- Computing them changes nothing in the game.
- Discards get no value in v5: what a discard is worth needs the unseen cards' draws (a search), the costlier
  piece; left for later, by R's next-ranked question if v5 doesn't close the gap.
- The warm start loads every weight of best.pt as it is; the weights reading the two new numbers start at zero, so
  at round 0 v5 plays exactly as best.pt (checked, below). Fresh optimizer, round 0, its own best eval and stall
  clock. The shop entropy is pinned at v4b's floor (0.001), as v4c did and for the same reason: a warm start counts
  decisions from 0, which would put v4b's entropy schedule back at its start.
- Not imitation and none of Balatro Lab's values or weights: the network still chooses every move, by the same
  reward as v4b. What changes is that it can see what each play scores, which Balatro Lab's bot computes and v4b
  never could.

**Checks before the smoke test** (tests/v5_play_check.py, in the launcher's pre-flight with v4b's checks):
1. The game's state is identical after computing play values, along 3 saved routes (as v4_phi_check).
2. S equals the chips the game scores, for every play along 3 routes that has no `before` effect, no chance
   effect and no blind effect left out: at least 98% exact. Every miss is listed.
3. v5 at round 0 (warm start) gives the same action probabilities as best.pt on the same observations (largest
   difference under 1e-5).
4. Sim time per decision, v5 over v4b on the same routes: reported; over 2x stops the launch for chud to decide.

**Run.** chud's laptop, 12 workers, 168 h cap, v4b's stall rule (25 evals without a better best), evals of 600
games every 20 rounds, no sell stop.

**Predictions (Claude's):**
- V1: best eval above 13.40 blinds/game (v4b's best) before the stall stop. 70%.
- V2: best eval at least 15.0 blinds/game. 35%.
- V3: best eval wins above 22 of 600 (v4b's best eval win count, round 1780). 45%.

**chud's prediction** (2026-11-14 16:05, his words): "given the reproduced 340/480 = .7083 win rate, I believe
aligning our bot's behavior with its behavior more will increase our win rate, which [will] give the best win rate
we've gotten so far." Scored on v5: its best eval win rate above every earlier model's. The record's best: v4b,
22 of 600 (3.67%) at round 1780 (runs/v4b/metrics.jsonl); before that 0.3-beta's 7 of 200 (3.5%, docs/models.md).
