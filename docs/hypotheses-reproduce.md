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
