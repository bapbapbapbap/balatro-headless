# Hypotheses: Run moe (first mixture of experts)

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-03 ~17:00 CST: I. Strategy specialists (design registered before any specialist exists)

### Authorship
The design and the prediction are chud's (16:57–16:59 CST): "I am going to train a bot on each
hand", "once you get a strategy going pick that strategy", and "I bet that'll beat the current
highest win rate bot design I've tried." The write-up and the open options listed below are
Claude's. Nothing above is edited.

### State at registration
No specialist has been trained. No archetype tally of 1.0's games has been run. Nothing below has
been looked at in data.

### Design
How a roguelike is played: stay open early, commit once a build forms, then play that build.
- **Before commitment:** the generalist plays (model 1.0, or its successor at the time of the test).
- **Commitment:** once a strategy is going (a build around one poker hand type: Flush, Pairs,
  Straight, and so on), control passes for the rest of the run to the specialist for that hand
  type.
- **Specialists:** one bot per hand-type strategy, each trained on that strategy.

### Open (chud to confirm)
- **Trigger** that decides "a strategy is going". Options: the first archetype joker bought (e.g.
  Droll Joker -> Flush); one hand type's planet level a set number of levels ahead of the rest; one
  hand type making up most scoring plays over the last N hands.
- **How specialists are trained.** Proposed: each starts as a copy of the generalist and is
  fine-tuned only on runs committed to its hand type, with the same reward of record (`blinds`), so
  specialization is the only change. The alternative (a reward per hand type) also varies the
  ought, and would be scored as a section C/aims design, not this one.
- **Which hand types get a specialist.** Depends on how 1.0's games split by archetype; rare ones
  (Full House: played 5 of the 497 times it was the best hand on offer, in the H window's
  262-game sample) may not have enough games without more training on them.
- **Compute match** between the specialists' total training games and the generalist's continued
  training, and whether the control runs concurrently.

### I1 (chud's prediction)
The generalist-plus-specialists bot has a higher win rate than the best bot design so far
(model 1.0: 24.4 fresh-seed wins per 1,000 training games in the H window; 5 of 200 on the eval
seeds).
- **Yardstick:** win rate (an ought, named as such).
- **Test, window and threshold:** to be registered in a dated section before the specialists'
  training starts, once the open items above are confirmed. Proposed: H1's test (one-sided exact
  conditional binomial on fresh-seed wins per 1,000) plus paired held-out seeds.
- **Known way to lose:** 1.0 already sees hand levels and jokers in its input and may already
  play each build differently; I1 bets that separate weights per build beat that.

### Attribution and review (added 2026-11-03 ~17:05 CST, requested by chud)
**Design: chud's.** Specialists per hand-type strategy, handed control once a strategy is going,
is his design, from how a roguelike is played. The prediction I1 is his. **Reviewed by Claude**
(Opus 5.5). The review below is Claude's; nothing in it is confirmed until chud confirms it, and
the registered text above is not edited by it.

Review findings:
1. **A concurrent control is free.** Model 1.0 is frozen, so it can play the same seeds as the
   specialist system at test time. That removes the gap H1 had (a historical baseline, no
   concurrent control). Proposed primary test: paired held-out seeds never used in training or
   eval, both systems sampled with the same RNG per seed, one-sided exact McNemar on per-seed
   wins. The 24.4 per 1,000 from H is a training-game rate under 5% exploration and seed retries,
   so it is not comparable to a system played without them; report it alongside, not as the test.
2. **Power.** At 1.0's eval rate (~2.5%), detecting a rise to 3.5% at 80% power, one-sided
   alpha 0.05, needs roughly 3,600 paired seeds (one game each); a rise to 4.0% needs roughly
   1,700. Several sampled games per seed (as in R) cut the seed count but not the game count much.
3. **Two questions, two controls.** I1 as stated ("beats the best design so far") needs only
   frozen 1.0. Whether *specialization* is the reason needs a second control: 1.0 trained on for
   the same number of games the specialists used, same reward and settings. Without it, a pass
   could be more training, as H1's caveat says.
4. **Score every run, not committed ones.** The test is on all held-out seeds. Runs that never
   reach the trigger are played start to finish by the generalist and count. Comparing only runs
   that commit would select on survival (a run has to live long enough to build).
5. **Specialists' training data is selected the same way.** Committed runs are runs that got far
   enough to commit; fine-tuning on them is fine, but their win rate isn't a measure of the
   specialist and shouldn't be reported as one.
6. **Pin the trigger before training.** The trigger decides which games each specialist trains
   on, so it is fixed in the registration of the test, not tuned after looking at specialist
   results.
7. **The handoff is one-way and the generalist doesn't know about it.** Pre-commit play is
   unchanged 1.0, so any difference comes from post-commit play. Good for attribution; it also
   means the design can't steer early play toward a build a specialist plays well. That is a
   possible later version, not this one.

Status of I1: **proposed**, unchanged.

### Design interview (2026-11-03 ~17:16-17:33 CST, chud's answers; recorded at his request)
chud asked to be interviewed so the record represents his design, not Claude's: "you gotta
represent me well or the epyrical value decreases." His answers, in his words where quoted. Where
he handed a choice to Claude, it is marked as Claude's. Nothing above is edited; where an answer
changes a proposal above, the answer stands.

- **Why it should work (chud, 17:19):** "balatro is deterministic. so that means for you to have
  gotten to blinds at 3 or 4 or some specified trigger turn, you had to have passed all the
  valued behind it. and you gotta get a certain strategy to get big enough numbers to pass. so
  then there are certain jokers that should corelate with different hand strategies."
  - Note: the pinned game is already known to be deterministic given seed and moves (every
    published route replays), so this is a premise, not a prediction.
  - The joker-to-hand correlation is testable on saved games now. Whether to register it as a
    prediction: *(chud to confirm)*.
- **Trigger (chud, 17:19):** "once a certain hand is prefered say past the 4th blind, make the
  agent asume the weights of the expert of that hand." (17:21) The handoff is by ante: "ante 4, or maybe 3, or maybe 5. it is to be
  determined the exact right hand off ante." How "preferred" is measured: *(open)*. If the ante is tuned, it is
  tuned on seeds apart from the test seeds and fixed before the test.
- **Steering (chud, 17:15):** "early play can steer." The generalist's own play before the handoff
  decides which build forms and so which expert takes over. (Review finding 7 above overstated
  this: what the generalist can't do, unless trained with the experts in place, is aim for the
  builds a given expert plays best.)
- **The generalist (chud, 17:23):** "we haven't trained the generalist. 1.0 is a proxy currently."
- **Experts (17:19):** "a bot trained a ton of games with a built in bias for that hand." Eight: Pair, Two
  Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind, Straight Flush. No High Card
  ("lets skip training high card"); no secret hands ("secret hands are for challenge points for
  psychos like me").
  - **Pair (17:31):** "pair is very strong early game, so we need to at least attempt a pair expert", and
    "pair can be half encouraged."
  - **Full House (17:31, 17:33):** "0.1 must be forced to take full house every time it shows up"; "full house
    is full house expert. the others are other experts."
- **The pin (chud, 17:33: "you decide pin rate for all of them. go. guess. keep the pin rate
  high."; the choices are Claude's):** a hard pin on plays. When a play of the expert's hand is on offer, the pin binds
  with probability 0.5 for Pair (his "half encouraged") and 1.0 for the other seven; when it binds,
  only plays of that hand are allowed. Discards and non-play moves stay free, so an expert could
  discard to dodge its hand; that is logged per game (`discard_offered`).
- **Not 1.0 (chud, 17:28):** "lets not use 1.0. even if that is counterintuitive, my gut says not
  to." Reason given earlier (17:22): "I say we train the experts untainted. 1.0 refuses full
  house, one of the best hands in the game."
- **Architecture (chud, 17:32):** "whatever you do use the upgraded 1.0 architecture" (d=192,
  4 layers, 6 heads, 1.6M parameters, encoding 2, ordered moves); "if we gotta train from scratch
  we gotta train from scratch."
- **Starting weights (Claude's pick, approved by chud's "go"):** model 0.2, the earliest posted
  weights already on that architecture. 0.1 is a different architecture and can't be loaded into it
  without retraining by imitation. 0.2 never plays Full House either (0 of 2,274); the hard pin
  overrides that at play time, and the shop and critic relearn.
- **Compute (chud, 17:23):** "I have days." Experts get far more than the 10,000 first floated; proposed
  12 hours each (about 96,000 games, close to 1.0's 90,720), run one after another.

- **Scope (chud, 17:41):** "all we need are experts." The generalist is dropped from the plan;
  only the eight experts are trained. How the experts are combined or compared at test time (who
  plays before ante 4 without a generalist) is *(open)*.

- **Who plays before the handoff (chud, 17:43):** "we're going to farm the perfect early game
  weights. we'll have to run from scratch an unpinned strategy and make it's win rate only to get
  to 4. grow a handfull enough to know when you found a good one."
- **Cutoff (chud, 17:45):** "it looks like 4 is a reasonable cutoff? with 5 being a slight
  reprieve." (On White Stake each ante's blinds are about 2.5x the last through ante 4, then 2.2x
  into ante 5; Claude's figures from the game's blind table, not re-read from the pinned source.)
- **Early-game model, as built (the choices are Claude's):** 1.0's architecture from random
  weights, no pin, hand-type exploration 0.05, lr 1e-4, fresh seeds only (no retries).
  - Its run ends the moment the ante-3 boss is beaten, so the expert makes the cash-out and shop
    going into ante 4.
  - Reward `blinds` (at most 9 before the stop), not a 0/1 reward for reaching ante 4, which would
    be sparse for a model starting from random weights. Reaching ante 4 means beating all 9.
  - Several seeds (default 5, 8 h each); best.pt for each is picked by the share of the 200 fixed
    eval seeds that reach ante 4, and the best seed is the one with the highest share. Because
    that picks on the eval seeds, any test of I1 uses other seeds.
  - Open: the experts are trained on whole runs from ante 1 (pinned throughout), while in play they
    would start from the states the early-game model hands them. Whether to train or fine-tune
    them from those handoff states instead *(chud to decide)*.
- **Stake (chud, 17:46):** "also I want the ability to change the stake without planning any higher
  stake runs." Built as a setting (`STAKE`, `BACK`); every planned run stays on White Stake, Red
  Deck. Runs at another stake or deck get their own run folder. Higher stakes are not covered by
  the differential check yet.

- **Mixture of experts (chud, 17:50):** "MOE isn't a secret." The experts and the handoff are
  trained jointly, so each expert learns from the runs actually handed to it.
- **Coverage and organic bias (chud, 17:52):** "I need an expert in all the hand types. would you
  have a learned bias grow organically until you have ones that naturally have a bias?"
- **Router (chud, 17:54):** "most played hand, but estimate how many the hand gets played from the
  starting model and that's how to calculate a preference from early game. what got the early game
  strategy that far should be the chosen expert."
- **Joint training, as built (the choices are Claude's):**
  - One run holds the early-game model (member 0) and the 8 experts, each warm-started from its own
    run. Each decision updates only the member that made it. The early model is frozen by default
    (it was picked for reaching ante 4); `MOE_TRAIN_EARLY=1` trains it on the whole run's return.
  - Router: at the handoff (the ante-3 boss beaten), the expert whose hand has the highest
    (plays this run + 1) / (early model's average plays + 1), counting antes 1-3. The average is a
    moving average over recent training games that reached the handoff. High Card has no expert
    and isn't counted.
  - Load balancing: 30% of training handoffs go to an expert picked at random; eval games are
    always routed by the rule.
  - Organic bias: the pins (Pair 0.5, others 1.0) fall linearly to 0 over the first 100,000 joint
    games. After that, an expert's bias is whatever its routed runs taught it. Measured per expert
    (`moe.by_expert` in the metrics): how often it plays its hand when one is on offer, with the
    pin off. A rate that stays high is a bias the expert keeps; one that falls back was the pin.

Built for this (CHANGELOG 0.10.0, 0.11.0, 0.12.0): `train.py --pin-hand/--pin-rate/--init-weights`,
`train/run_expert.sh`, `train/run_experts.sh`, `tests/pin_check.py`, `train/run_early.sh`,
`train/run_earlies.sh`, `train.py --moe`, `train/run_moe.sh`, `tests/moe_check.py`. The test of I1
is not registered.

---

## Added 2026-11-03 ~21:35 CST: the MoE run started (section I)

Started by chud at 21:25 CST on commit `17c1602` (PR #74), `train/run_moe.sh <shared> 24 14`, with
the watcher (`watch_v1.ps1 -Run moe`). Pre-flights: saved real win on the untouched game, ordered
actions, fast vs untouched game, MoE routing: all PASS.

### Settings
- Shared trunk and 8 heads: the early head (antes 1–3) and 7 experts (Pair, Two Pair, Three of a
  Kind, Straight, Flush, Full House, Four of a Kind). All from model 0.2's weights.
- Reward `blinds` with the corrected outcome test (a win is the game's own `win_game()`; a blind is
  the game's `Defeated` mark). Shaping on. Encoding 3, `ordered-capped`, lr 1e-5, exploration 0.05.
- Router (chud): the hand played most in antes 1–3 relative to the early head's own average; 30%
  of training handoffs random. Pins Pair 0.5, others 1.0, fading over 100,000 games and not before
  each expert's own 5,000. Early head trained, its returns cut at the handoff.
- 14 workers, Red Deck, White Stake, seed retries on, 24-hour cap, stall stop after 3 flat hours.
  Evals: 600 fixed seeds every 20 rounds. Near wins logged in every game.

### Starting point (eval at round 0: 0.2's weights inside the MoE, pins on)
600 seeds: **5 wins** (0.83%), **9 near wins**, so **14 reached the final boss** (2.3%);
11.67 blinds per game. This is the first eval scored with the game's own outcomes; it is the
baseline the experts are measured from, not a measure of 0.2 itself (the pins change its play).

No prediction is registered for this run. The test of I1 is still to be registered before it is
scored, on seeds apart from the 600 eval seeds.

## Added 2026-11-04 ~13:40 CST: the first MoE run stopped; changes for the second (section I)

### The first run (`runs/moe`, commit `17c1602`)
Stopped itself at 02:54 CST on the stall rule (training score flat for 3 h; best 10-round average
12.14 blinds per game), 5.4 h into its 24-hour cap: 220 rounds, 24,640 training games, 256 training
wins (game's own outcomes, all replay-verified). Evals on the 600 seeds:

| round | wins | near wins | blinds/game |
|---|---|---|---|
| 0 | 5 | 9 | 11.67 |
| 40 | 3 | 8 | 12.22 |
| 100 | 5 | 7 | 12.59 |
| 120 | 11 | 6 | 12.60 |
| 160 | 6 | 10 | 12.17 |
| 200 | 9 | 15 | 12.68 |
| 220 | 11 | 10 | 12.73 |

Not a test of the design: at the stop, every expert pin was still at 0.75 of its starting rate
(the fade had covered 24,640 of 100,000 games) and no expert had its 5,000 games (Pair 3,486, Flush
3,383, Three of a Kind 3,177, Straight 3,101, Two Pair 2,679, Four of a Kind 1,195, Full House 907).
The stall rule read a score held down by pins that hadn't faded. Full House got no handoff by rule
in any eval: only random ones in training. At eval 220, Three of a Kind had the most handoffs (108)
and 0 wins. Heat-throttling alerts from about 02:00.

### Changes for the second run (chud, 2026-11-04 13:20; implementation Claude's)
- **No Four of a Kind expert** (chud: a flex hand, not a strategy of its own). 6 experts, 7 heads.
- **Early-game Full House pin** (chud: the bot takes Two Pair over Full House, and the early game is
  the worst time for it). At an early-model decision where a Full House or a higher hand (Four of a
  Kind, Straight Flush, Five of a Kind, Flush House, Flush Five) is on offer, only those plays are
  allowed, at rate 1.0 times the fade; discards and other moves stay allowed. It fades over the same
  100,000 games as the expert pins. Why the early game: a Full House contains Two Pair, so every
  Two Pair joker still fires on it, and at level 1 it scores 40 x 4 against 20 x 2; Two Pair
  only overtakes it after two Uranus (level 3: 60 x 4). *(Corrected 2026-11-04 ~13:50: this line
  said "about three"; see the review below.)* The reward (`blinds`) pays nothing past a blind's
  target, so it doesn't separate the two while Two Pair clears the blind. This pin is an ought
  from chud, put into the early game, not into the reward.
- **Router minimum**: Full House's expert is picked by rule only after 2 Full House plays in antes
  1–3 (a single play against a near-zero base rate would otherwise win almost every such game).
- **The stall clock starts once every pin (experts' and the early one) is 0.** The 24-hour, now
  36-hour, cap is the backstop.
- Fresh start in `runs/moe2` (a checkpoint with 8 heads doesn't load into 7).

No prediction is registered for this run either. I1's test is still to be registered before scoring.
