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

## 2026-11-15: implementation, checks and smoke (before launch)

Code on branch `v5` (this commit). Claude's, reviewed by chud.

- `harness/build_phi.lua`: `H.play_scores()` and `H.play_values()` as registered. Two details of "what the play
  scores now", both the game's order of events and not a change of design: inside the sandbox the hand counts as
  played before it is scored, as `play_cards_from_highlighted` and `G.FUNCS.evaluate_play` have it (hands left
  minus 1, which Acrobat reads; the hand type's `played` and `played_this_round` plus 1, which Card Sharp and
  Supernova read). The blind is applied with `debuff_hand(..., check = true)` (no state change), The Arm's level
  loss by hand, then `modify_hand`.
- `train/sim.py` (`play_values`, two int16 columns after the offer value; off during curriculum replay as offer
  values are), `train/model.py` (`play`: the `num` layer reads 5 numbers, action rows 21 wide), `train/options.py`
  (`--play-values`, `--warm-start`), `train/train.py` and `train/launch.py` (the warm start and `warm:` start,
  ported from model v4c's, branch `v4c` commit 96703f9; v5's `warm_start_state` pads the `num` weights with zero
  columns instead of adding a swap row). Pre-flight checks may now name `{work}` (check 3 needs best.pt).
- `configs/v5.toml`: v4b's settings plus `play_values`, `start = "warm:{work}/runs/v4b/best.pt"`, `shop_ent = "0.001"`,
  and the `v5-play` check.

**Checks** (tests/v5_play_check.py, cloud, stock LuaJIT, v4b's exported best.pt rebuilt as a checkpoint):
1. Game state and v4b's whole observation identical at every move: PASS (3 routes; also 10 routes).
2. S = the game's chips: 60 of 60 plays exact on 3 routes; 192 of 192 on 10 routes. Plays outside the check
   (a left-out joker or The Hook) on the 10 routes: Green Joker 209, Misprint 92, Spare Trousers 60, The Hook 10,
   Obelisk 6, Space Joker 6. For those plays S leaves out what the registration says it leaves out; not measured.
3. Warm-started v5 vs best.pt: largest probability difference 0.00e+00 over every move of the 3 routes: PASS.
4. Sim time per move: v4b 35.7 ms, v5 48.5 ms (1.36x): under the 2x gate.

**Smoke** (cloud, 1 worker, `launch.py v5 --smoke`): warm start loaded (best.pt's round 1840, 13.40), 3 rounds ran
and stopped clean; eval 0 at 12.25 blinds/game on 12 games (noise-level sample of best.pt's play). chud's launch
runs every check again in pre-flight on his machine, with LuaJIT 2.0.5.

## 2026-11-15: correction to the section above

Two of check 2's out-of-check tallies on 10 routes were summed wrong: Green Joker 161 (not 209), The Hook 9 (not
10). Recounted from the check's per-route lines; the other tallies and every result stand.

## Added 2026-11-16, after chud's 11:19 CST message: v5 stopped by chud; V1-V3 and chud's prediction scored

Claude's record, reviewed by chud; the stop and its reason are chud's.

**Stopped** by chud (Ctrl+C, 2026-11-16 11:18:40 CST) at round 278, 11.96 h, not to be restarted. His words
(11:19): "running v5 will not get us there. I am pausing v5 without the intent to restart it." At the stop the
stall clock stood at 13 of 25 evals; the best eval was still round 0's, which is v4b's best.pt unchanged.

Evals (600 games: blinds/game, wins, deaths at ante <= 3), from runs/v5/train.log:
r0 13.40/16/163; r20 12.92/17/156; r40 11.89/6/193; r60 11.88/5/188; r80 11.13/7/219; r100 11.23/5/207;
r120 10.86/8/207; r140 10.99/9/230; r160 11.09/5/208; r180 11.21/6/214; r200 11.48/8/197; r220 12.31/11/187;
r240 12.47/12/157; r260 12.23/11/168. One update rejected (round 121, KL 24.5 > 0.2): reverted, learning rate
halved to 5e-5 (strike 1/3). 459 wins saved, all replayed exactly.

**Scored:**
- V1 (best eval above 13.40; Claude 70%): no. No eval after round 0 reached it.
- V2 (best eval >= 15.0; 35%): no.
- V3 (best eval wins above 22/600; 45%): no. Best 17, round 20.
- chud's (2026-11-14 16:05; scored on v5 as its best eval win rate above every earlier model's, 22/600): not met
  by v5 (17/600). v5 aligned only the inputs of hand play, not the behavior; the prediction's own claim, aligning
  our bot's behavior with Balatro Lab's, is what model v6 tests (docs/hypotheses-v6.md).

**What it shows (Claude's reading, not tested):** like v4c, v5 fell from its warm start in the first ~120 rounds,
through more early deaths, and was climbing back when stopped. Both started from v4b's weights with a fresh
optimizer at learning rate 1e-4; v5's recovery began after the halving. Whether the fall came from the restart or
from what each run added was not separated (no control run of v4b restarted the same way).
