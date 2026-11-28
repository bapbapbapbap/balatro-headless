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

## Added 2026-11-14 ~16:50 CST: R's code, decided before R runs (two details beyond the 16:05 section); a time correction

Claude's, reviewed by chud.
- Time correction: the section above headed "~16:15 CST" was committed at 16:05 (commit 62eb5d7).
- `reproduce/balatro_lab/rank.py`. Checked before this section: A0 reproduces v4b's own eval games (blinds beaten
  and number of moves identical on 3 seeds). Smoke runs in the cloud (2-3 seeds, stock LuaJIT) are discarded; R's
  result is the laptop run.
- Detail 1, chance outcomes: where the twin rolled a chance effect its own way (Misprint's mult, a Wheel of
  Fortune edition, Business Card money, The Hook's cards, a pack's contents...), the game's outcome is copied
  into the twin and counted per field (`mirrored`, `resyncs`); the game is the truth. A twin that has already moved
  past the game (it ended a round the game did not) can't be repaired and is lost.
- Detail 2, after the twin is lost: in A1 and A5, hand play stays with Balatro Lab's bot, deciding on a Run built
  from the game state (`state.py`, whose scoring matched the game on 4,143/4,143 hands), counted as
  `theirs_no_twin`. Shop, pack and blind decisions after a loss go to v4b, counted by `lost_twin`.
- Reported per arm with the result: moves decided by Balatro Lab's bot, unmapped ones, games that lost the twin.

## Added 2026-11-14 16:24 CST: time correction

Claude's. The section above headed "~16:50 CST" was committed at 16:24 (commit 8fca40f). Section times from
here on are read from the clock when written.

## Added 2026-11-14, after chud's 16:48 CST message (the commit's time is the record): R's pre-flight, and jokers mirrored whole

Claude's, reviewed by chud. Before R's laptop run:
- `rank.py` runs a pre-flight on a fresh run (eval seeds 599-600, outside R's 300) and starts R only if it passes:
  the game's LuaJIT 2.0.5; A0 replays v4b's own eval games exactly; every arm finishes a game; Balatro Lab's bot
  decides in the arms that hand it decisions and nearly all its moves map to legal moves; the output folder is
  writable; the twin follows past the first half of the game in most arms.
- The pre-flight's first cloud run caught the twin lost within ~25 moves in every arm: Gros Michel's 1-in-6
  extinction rolled by Balatro Lab's own randomness. Detail 1 of the 16:24 section now covers whole jokers: the twin's
  jokers are rebuilt from the game's (its own objects kept where they match). After that the twin followed every arm
  but two past the first half (A1 to move 103, A5 to move 63); pre-flight passed in the cloud.

## Added 2026-11-15, after chud's 09:56 CST message: R's result, scored

Claude's, reviewed by chud. Run on chud's laptop overnight (LuaJIT 2.0.5; pre-flight passed), 300 eval seeds x 6
arms, 0 errors. Data: `reproduce/balatro_lab/results/2026-11-15/`.

| arm | wins | blinds/game | vs A0, paired (95% bootstrap) | wins vs A0 | their moves/game | games the twin was lost |
|---|---|---|---|---|---|---|
| A0 v4b alone | 7 | 13.33 | | | 0 | 0 |
| A1 hand play theirs | 13 | 16.67 | **+3.34 [+2.56, +4.14]** | +6 | 35.0 | 199 |
| A2 shop theirs | 9 | 12.44 | -0.89 [-1.71, -0.10] | +2 | 19.0 | 206 |
| A3 packs theirs | 4 | 12.73 | -0.60 [-1.29, +0.07] | -3 | 8.3 | 124 |
| A4 blinds theirs | 11 | 12.11 | -1.21 [-2.03, -0.36] | +4 | 6.9 | 148 |
| A5 all theirs | 23 | 14.70 | +1.38 [+0.63, +2.15] | +16 | 71.8 | 252 |

- A0 agrees with v4b's eval (13.33 on these 300 seeds; 13.40 on all 600).
- Read with care: after the twin is lost, shop, pack and blind decisions go back to v4b (registered); in A2-A5 that
  happened in 124-252 of 300 games. A1 doesn't have this problem (hand play stays theirs on the game state). A5 is
  therefore not Balatro Lab's bot on our code, and R5 can't be read as a fidelity result.
- Scored: R1 (A1 gains most; Claude 55%) yes. R2 (A1 >= 1.0 blind/game, interval above 0; 70%) yes. R3 (A2
  gains, interval above 0; 60%) no: A2 loses, interval below 0. R4 (A4 within 0.3; 70%) no: A4 loses 1.21. R5
  (A5 wins >= 60%; 75%) no: 23/300, with the caveat above.
- By the rule fixed at 16:05: the arm with the largest blind gain is A1, interval above 0. v5 gets the computation
  behind hand play (docs/hypotheses-v5.md).

## Added 2026-11-27, after chud's 20:57 CST message (the commit's time is the record): AnteZero's simulator, what is already measured, and tests Z1-Z2 registered before their code

Claude's, reviewed by chud. AnteZero is namecallfilter's (github.com/namecallfilter/antezero, MIT, pinned at commit
`4fe07f771104a811ca8452efd4831b0dbf097e6e`, 2026-11-27); none of its code is in this repository. Its README calls
its Rust simulator "a bit-exact Rust reimplementation of Balatro v1.0.1o, verified in lockstep against the real
game", and reports win rates measured in that simulator (94% White, 12% Gold, all decks, no search). chud's aim:
test that claim as we tested Balatro Lab's, then use AnteZero's released bot as a teacher.

**How AnteZero checks itself (read from its code before this section).** `sim/oracle` runs the game's scripts
headless in the game's `lua51.dll` (Windows only) through its own harness (`sim/oracle/lua/harness.lua`), which
performs actions by calling `G.FUNCS` directly. `difftest` plays random games (with an injection rate that adds
jokers, cards, money, vouchers) in both engines and compares a canonical state (every RNG stream included) and the
order of RNG calls, draws and removals after every step. Two things that harness changes, as its own docs say: it
sets `G.TIMERS.TOTAL` to 0 at the start of every action, and it overrides `Card.unique_val` (which the game derives
from a UI node counter and uses as a tie-break in `Card:get_nominal`) to the simulator's model, "so this oracle
cannot check that tie-break on its own". The bridge (`bridge/`) also compares with the real game through a
Steamodded mod; that needs the Windows game running and is not repeated here.

**Already measured (before this section, setup):**
- Their oracle ported to Linux (`dlopen` in place of `LoadLibraryExW`, by our script
  `reproduce/antezero/port_oracle_linux.py`, which edits only the pinned clone), on LuaJIT 2.0.5 built from its
  release commit 0bf80b07, our 1.0.1o source (exe SHA-256 `0D75FE16…3F47`, the same pin as the harness). The
  bundle lacks `resources/`; empty font files were placed so the game's font loader finds them (no game logic
  reads them). Their `difftest 3 1 400 b_red 1 0.15`: 3/3 games matched, 227 steps, 230/230 frame-exact
  timelines. Their own check reproduces off Windows on a small sample.

**Test Z0 (their method, at scale).** Their `difftest` on Linux as above, 20 games per deck (15 decks) at stakes 1
and 8, 400 steps max, inject rate 0.15, first seed 1. Outcome: games matched / games. Z0: >= 99%. Claude 85%.

**Test Z1 (openings, our reader).** For 200 seeds per deck x stake (15 x 8, seeds from `random.Random("az-z1")`),
start the run in their simulator (`antezero_sim.Game`) and in our harness (`harness/env.py`, the game's code, no
timer rebase, no `unique_val` override); compare our view (below) of the opening state. Z1: every one matches.
Claude 90%.

**Test Z2 (independent lockstep).** Same seed, deck and stake in both. At every decision a weighted random policy
picks one of the simulator's legal actions (`legal_codes()`); it is translated to our env's subsets-mode action and
applied in both; then our view of each is compared. No injections (scenarios come from play). 10 games per deck x
stake (1,200 games), 400 decisions max, seeds from `random.Random("az-z2")`. Our view, built by our own code on each
side (the harness side reads `G` directly, the simulator side reads its `state_json`): phase, ante, round, money,
hands and discards left, chips, blind and its chips, hand (in order), draw pile (in order), discard pile, jokers
(key, edition, stickers, sell value), consumables, shop jokers / vouchers / boosters (key, cost), pack cards, hand
levels, tags, used vouchers, and every RNG stream in `G.GAME.pseudorandom`. Numbers compared exactly.
- Z2a: >= 95% of games match at every decision to the end. Claude 70%.
- Z2b: every divergence, if any, traces to one of: the timer rebase, the `unique_val` tie-break, or an action our
  env performs differently from their harness (blind select through the button, the boss-alert wait). Claude 60%.
- Z2c: every simulator-legal action (no-op sorts and joker/hand drags counted apart) is in our env's legal set.
  Claude 75%.

A divergence ends that game's comparison; its history is saved and classified by hand. Z3, their released bot
driving the lockstep and their win rates on the game's code (a win counts only if its route replays move for move
to the game's own win), is registered in its own section before it runs.

## Added 2026-11-27, after chud's 20:57 CST message (the commit's time is the record): Z3, AnteZero's bot on the game's code, registered before its code

Claude's, reviewed by chud. AnteZero's released model `antezero-v0.2.0.pt` (release v0.2.0, sha256
`ea28506dc6a7016b73e0f921c5fe2f1b25caab34b9e13e49d9aced9d89ffaef8`). Z1 and Z2 are running; no Z3 code exists.

**Z3.** Their bot plays in their simulator with their own eval protocol (`antezero_agent.eval.evaluate`'s loop:
`VecEnv` built by `runtime.make_env` from the checkpoint's config, seed `EVAL_SEED` 7,777,777 + batch, greedy,
no search, endless off so a game ends at the ante-8 win). For every step we record the action as the simulator's
text code (`Game.factored_to_code` on `env.game(i)` before the step). Then each game is replayed in our harness
(the game's own Lua) through `lockstep.py`'s translation, with our view compared after every action as in Z2.
Cells: White (stake 1) and Gold (stake 8), every deck, 20 games per deck x stake (600 games) in the cloud; the
laptop can extend it with the same script.
- Z3a: their bot's win rate in their simulator on these games is within their table's interval: White all decks
  >= 88%, Gold all decks within 12% +- 4 points. Claude 75%.
- Z3b: >= 95% of games replay move for move, our view matching after every action. Claude 65%.
- Z3c: wins on the game's code (the route replays move for move to the game's own win) are within 1 point of the
  simulator's wins on the same games, per stake. Claude 75%.

**Z0 moved to the laptop** (2 cloud cores run their oracle at ~0.2 s a step; 600 games is ~13 h here). Cloud
partial, before it was stopped: Red, stake 1, 19/19 games matched (the 20th was running).

## Added 2026-11-27, after chud's 20:57 CST message (the commit's time is the record): Z1-Z3 results, scored; Z0 still to run

Claude's, reviewed by chud. All in the cloud (2 cores, LuaJIT 2.0.5 via lupa, our 1.0.1o source). Data:
`reproduce/antezero/results/2026-11-27/`.

| test | what | result |
|---|---|---|
| Z1 | openings, 200 seeds x 15 decks x 8 stakes, our view (deck order, blinds, boss, tags, voucher, every RNG stream, ...) | **24,000 / 24,000** match |
| Z2 | random-policy lockstep, 10 games x 120 cells, our view after every action | **1,200 / 1,200** games match to the end; 26,823 actions; 0 simulator actions outside our env's legal list |
| Z3 | their bot (v0.2.0, greedy, no search), their eval loop, White and Gold, 20 games x 15 decks each; replayed in our harness | **600 / 600** games move for move (110,201 actions); wins in their sim = wins on the game's code: White 295/300 (98.3%), Gold 36/300 (12.0%) |

Scored:
- Z1 (every opening matches; Claude 90%): yes.
- Z2a (>= 95% of games match; Claude 70%): yes, 100%. Weak coverage: the policy is crude, 834 of 1,200 games end in
  ante 1 and none past ante 6; Z3 is the deep test.
- Z2b (every divergence traces to the timer rebase, the `unique_val` tie-break or our env's action path; Claude 60%):
  not tested, there were no divergences. Neither of the two things their oracle can't check (their harness's timer
  rebase, its `unique_val` override) showed up in 137,024 compared actions on our harness, which has neither.
- Z2c (every simulator-legal action is in our legal set; Claude 75%): yes in Z2 (0 of 26,823). Z2 had no sorts or
  drags (the simulator's default `Game` offers none). In Z3 the bot's 16,382 sorts and 11 hand drags aren't in our
  subsets-mode list (that mode leaves them out by design) and were applied with the game's own sort functions; every
  other Z3 action (1,908 joker drags included) was in our list.
- Z3a (sim win rate within their table: White >= 88%, Gold 12% +- 4; Claude 75%): yes, 98.3% and 12.0%.
- Z3b (>= 95% replay move for move; Claude 65%): yes, 100%, with one handling decided after the first test games
  showed it (code commit `ecfd909` and the README's notes, before the 600 ran): at a win the game pauses on its win
  overlay with the boss still in place (debuffs, face-down jokers, the blind's chips), and the simulator's terminal
  state is the one after the overlay is dismissed, as their harness dismisses it. Without that step all 331 wins
  differ at their last action only; after it all match. The 269 losses match at every action as they are.
- Z3c (wins on the game's code within 1 point of the sim's; Claude 75%): yes, identical (331 = 331).

Caveats:
- Their eval loop seeds a batch from `EVAL_SEED` + batch index; with one batch per cell, all 15 decks of a stake
  used the same 20 seed strings. 300 different games per stake, but 20 seeds. A laptop run with more games per cell
  (README) widens it.
- Card `ability` tables are checked on the keys their dump has; game-only keys aren't checked. Booster centers that
  differ only in art and come from unseeded `math.random` counted as equal (380 of the Z3 games had one).
- One code change after Z2 started: the drag matcher now also checks the destination slot (it had picked the first
  drag from the same slot). Z2 had no drags, so its result doesn't depend on it; Z3 ran after it.
- Z0 (their own difftest at scale) was stopped in the cloud after 19/19 games of one cell matched; it runs on the
  laptop (`reproduce/antezero/z0.sh`).

Reading: on everything compared, AnteZero's simulator follows the game's own code exactly, deep into Gold runs, and
its bot's 12% Gold is real on the game's code. For scale: our v6g r660 wins 0.01-0.32% a deck at Gold and 174/600
(29%) at Red White (docs/system_card_v6g.md). Answer to question 1 for AnteZero: yes.

## Added 2026-11-27, after chud's 22:17 CST message: correction to the Z3 seed caveat

Claude's, reviewed by chud. The 20:57 results section says "all 15 decks of a stake used the same 20 seed strings".
It is wider than that: the same 20 seeds ran in all 30 cells (15 decks x 2 stakes); each seed appears 30 times in
`z3_games.jsonl.gz`. The cause is how `z3.py` calls AnteZero's `_envs`: once per cell, 20 games, so every call is
one batch seeded `EVAL_SEED` + 0. Their own eval makes one call over all decks and stakes, with batch k seeded
`EVAL_SEED + k`, and doesn't repeat seeds this way. The chat summary that called it their loop's reuse was wrong.
Fidelity results stand (600 different games, 110,201 actions); the win rates are from 20 seeds, correlated across
cells, so their uncertainty is wider than 300 independent games per stake would give.

## Added 2026-11-27, after chud's 22:32 CST message (the commit's time is the record): AnteZero as teacher, stage 0 (the data), registered before its code

Claude's, reviewed by chud. chud asked to start a teacher from AnteZero the way v6 started from Balatro Lab. This
section covers only the teacher's data and its checks. The student (size, encoding, which decks and stakes it trains
on, imitation then PPO) is a later registration, before any of its code; chud decides the size after a write-up.

**Why no bridge.** v6 needed `bridge.py` to rebuild Balatro Lab's state from our game. Z1-Z3 found AnteZero's
simulator follows the game action for action, so the teacher plays in its own simulator and every game is replayed
in our harness, where each decision is written as our student sees it.

**Seeds.** chud's standard: never a fixed eval set. Every teacher game gets its own seed from
`random.Random("az-teach")` (8 characters from the game's alphabet), skipping our 600 eval seeds and the 20 Z3
seeds; no seed is used twice, across decks and stakes included.

**Record** (`reproduce/antezero/teach_az.py record`). AnteZero v0.2.0 (sha256 `ea28506d…ffaef8`), greedy, no
search, through its own `VecEnv` (built by its `runtime.make_env` from the checkpoint's config, endless off); each
slot is loaded with `Game(seed, back, stake)` through `VecEnv.set_game`, with the slot's own options (skip, sorts,
rearranges). Every action is saved as the simulator's text code, with the deck and stake. Self-check: the codes
replayed in a fresh `Game` reach the same end.

**Label** (`teach_az.py label`). Each recorded game replayed in `train/sim.Sim` with v6g's settings (ordered-capped
actions, encoding 2, offer and play values, phi_build), and compared after every action with the simulator, as Z3
(our view, `view.lua`). The teacher's code becomes one of our legal moves:
- Hand sorts and hand drags are not decisions in our action list: applied to the game as they are, counted
  (`reorders`), not labeled.
- Play, discard, use, pick: the legal move with the same kind, index and cards; where our list offers a lead card
  (plays) or a copy direction (Death), the one that changes nothing (the leftmost scoring card as lead, the rightmost
  selected card as Death's source). That plays the teacher's exact scoring order, since the hand is already in the
  teacher's order.
- Anything else by kind and index. A teacher move with no legal match (e.g. a joker drag past our drag cap) is
  applied as it is and that decision is unlabeled (`unmapped`).
- A game is kept only if the comparison matches after every action (the win overlay handled as in Z3); otherwise it
  is logged with an error and not saved. Output: `teach.py`'s format (`games_w*.jsonl`, `games/<seed>.pkl.gz` with
  `{obs, member, a, labeled, kind, desc, played, progress}`), so `train/imitate.py` reads it unchanged.

**Checks, cloud size** (2 cores; the laptop's data run gives the real numbers). 5 games per deck at stakes 1 and 8
(150 games).
- M1: >= 99% of games label to the end in lockstep. Claude 75%. M2: >= 98% of decisions labeled. Claude 70%.
- W1: their bot's win rate on fresh seeds on the game's code: White >= 90% (Claude 75%); Gold between 6% and 18%
  (Claude 55%). Reported with Wilson intervals; at this size they are wide.
- L1 (leak probe, as v6's B2): at every round decision of 30 of these games, the teacher decides again on a copy whose
  hidden state is redrawn (`Game.resample_hidden`); an honest teacher decides the same. >= 99.5% the same. Claude 80%.
- S1 (does sorting matter): for every labeled play that a sort came before since the last decision, whether the first
  scoring card differs from the first scoring card in the hand order before the sorts. Descriptive; Claude predicts
  it differs in >= 20% of those plays (50%).

## Added 2026-11-27, after chud's 22:32 CST message (the commit's time is the record): AnteZero teacher, stage 0 checks, scored

Claude's, reviewed by chud. Cloud, 150 games (5 per deck at stakes 1 and 8), 150 distinct fresh seeds, each used
once. Data: `reproduce/antezero/results/2026-11-27/teach/` (the labeled pickles stay out of git, 15 MB; the laptop
run makes its own).

| check | result |
|---|---|
| M1 games labeled to the end in lockstep | **150 / 150** |
| M2 decisions labeled | **26,585 / 26,587** (99.99%); the 2 unlabeled are joker drags past our drag cap |
| W1 win rate on fresh seeds, on the game's code (= its sim's on every game) | White **70/75 = 93.3%** [85.3, 97.1]; Gold **9/75 = 12.0%** [6.4, 21.3] |
| L1 leak probe, 30 games | **1,281 / 1,281** round decisions unchanged with the hidden state redrawn |
| S1 sorts and scoring order | the first scoring card changed in **0 of 76** plays after a sort where order matters; 927 more plays after a sort where it doesn't |

Scored: M1 yes (Claude 75%). M2 yes (70%). W1 White yes (75%; the interval reaches below 90); W1 Gold yes (55%;
in 6-18). L1 yes (80%). S1 no (predicted >= 20% changed, 50%): the teacher's sorts don't set which card scores
first. AnteZero's own README says a trained policy "spent a sixth of its steps re-sorting a sorted hand" under its
default `null_sorts`, which fits.

One labeling detail decided after the registration and before these 150 ran (commit "teach_az.py: ..."): in 134
plays our list offered no lead that changes nothing (a kicker sat left of the first scoring card). The label is
the lead move with the same scoring order (the leftmost scoring card leads), and the game gets the teacher's exact
move, so the replay stays the teacher's game. Without it the first such game diverged in deck order (cards return
to the deck in played order; the next shuffle orders them by `sort_id` again).

Reading: the pipeline is ready for the laptop data run. Gold on fresh seeds is in line with 12%, with a wide
interval at 75 games; the laptop run's size narrows it. The student's size: `docs/proposals/az-student-size.md`
(facts only; it corrects a chat figure: v6 is 3.38M parameters, the teacher 6.32M).
