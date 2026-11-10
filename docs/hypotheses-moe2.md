# Hypotheses: Run moe2 (moe2-pins0)

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-04 ~13:35 CST: I2, Full House is one of the strongest experts (section I)

Recorded at 13:31 CST, minutes after chud started the second MoE run (`runs/moe2`), before any of its
evals existed.

### I2 (chud's prediction)
> "It might take a longer time for the weights to learn Full House, but I hypothesise it is one of
> the most effective experts at winning."

Two parts: the Full House expert learns more slowly than the others, and once it has, it is among
the experts with the highest win rate.

### Test (proposed by Claude; chud to confirm before any moe2 eval is read)
- **Which state is compared.** The router hands each expert different games: Full House gets runs
  that played Full Houses in antes 1–3. Comparing each expert's win rate over the games it was routed
  would mix the expert with the games it was given. Evals are deterministic per seed up to the
  handoff (fixed sampling RNG per seed), so the comparison can be paired: on every eval seed
  that reaches the handoff, replay the early model's moves to the handoff, then play the game out
  once with each of the 6 experts, from the same state. Every expert is scored on the same
  positions. Run on seeds outside the 600 eval seeds, with the final weights (all pins 0).
- **"One of the most effective"**: Full House's win rate ranks in the top 2 of the 6 experts.
  Reported overall and on the seeds the router itself sends to Full House.
- **"Takes longer to learn"**: the number of its own training games before the Full House expert's
  win rate in the per-expert eval stats first reaches the median expert's, compared with the other
  experts. Full House should need more.
- **Size**: win rates are around 1–2%; the number of seeds is set before the test is run so that the
  top-2 ranking means something (to be worked out, then recorded here).

Status: **proposed**. The test design needs chud's confirmation; nothing scored.

## Added 2026-11-04 ~13:50 CST: review of the early Full House pin (for a critical reader)

The pin is chud's decision. Its settings (rate 1.0, "Full House or higher", the router's minimum of 2
plays) are Claude's. This is the case for it and what it costs, written before any moe2 result.

**1. "Is Two Pair over Full House actually a mistake?"** In the hands where it mattered, yes.
- A Full House contains Two Pair (and Three of a Kind, and Pair), so every joker that fires on
  "contains Two Pair" (Clever, Mad, Spare Trousers) fires on the Full House as well. Same jokers,
  more base score, and one more scoring card.
- Base scores at level 1, from the game's hand table: Full House 40 x 4 = 160, Two Pair 20 x 2 = 40.
  Two Pair at level L is 20L x (L + 1): 120 at level 2, 240 at level 3. So Two Pair overtakes an
  unleveled Full House only after two Uranus.
- The evidence that the bot gets this wrong already accounts for levels. `docs/won_games_analysis.md`
  (model 0.2's 709 wins, the weights this MoE starts from) counts decisions where Full House had the
  highest chips x mult **at the hand levels of that moment**: 2,274. It played Full House 0 times and
  split it into Two Pair 2,075 times. A leveled Two Pair doesn't explain those; in each of them Two
  Pair was the lower hand by the game's own table.
- The first MoE run agrees: across 443 eval handoffs the router never picked Full House by rule.
  With Full House's base rate near zero, one Full House play in antes 1–3 would almost always have
  won the vote, so the early head played essentially none.
- Where Two Pair can be right: Square Joker (exactly 4 cards), effects of cards held in hand (a
  fifth card kept back), and Two Pair leveled twice or more. The pin covers the early game only,
  fades, and leaves discards open, so these cases still have a way out.

**2. "Why not leave it to exploration or to RL?"** Both were tried. Hand-type exploration
(`--explore-hands 0.05`, CHANGELOG 0.8.0) was added for exactly this and was on through 0.3 and the
first MoE run; the habit stayed. RL alone can't fix it: the reward is blinds beaten, and score past a
blind's target pays nothing. In antes 1–3 a Two Pair usually clears the blind, so both plays get the
same return and the gradient between them is zero. The habit came in by cloning v1 (CHANGELOG 0.7.1)
and nothing in the reward removes it.

**3. "Why not change the reward instead?"** The reward is the variable under study. The main
experiment (sections C and G) varies the ought and holds everything else fixed; a bonus for bigger
hands would put a second ought into every run, unrecorded in the reward of record. The pin leaves the
reward untouched, acts only on which plays are allowed, is logged per decision (`moe.early_pin`:
offered, bound, plays, discards while on offer), and goes to 0.

**4. "Why the early game?"** It's where the choice is cleanest and where it decides the most.
Levels haven't diverged yet, so the Full House is the higher hand almost every time. And the early
game feeds a loop: Two Pair played early sends the run to the Two Pair expert, which levels Two Pair,
after which Two Pair over Full House becomes correct. The Full House expert can only be reached
through early Full House plays. Pinning the experts alone (the first run's design) never reached it.

**5. "Rate 1.0 is heavy-handed."** It is, at the start. It fades linearly to 0 over 100,000 games,
the same schedule as the expert pins, and the stall stop waits for it. "Or higher" keeps it from
forcing a Full House over a Four of a Kind, Straight Flush or the secret hands. Discards stay legal,
so the bot can dodge the pin; `discard_offered` counts how often it does.

**6. "The router minimum of 2 is arbitrary."** It's a judgment call. With a near-zero base rate, a
single Full House play would outvote a run built on a dozen Pairs. One play is weak evidence that a
deck suits Full House; two is the smallest step up. The routing counts in every eval show whether
it's too strict (Full House starved) or too loose (Full House flooded).

**7. What the pin costs (stated, not defended).**
- **Path dependence.** After the fade, the weights still carry the pin's history. If moe2 plays Full
  House with the pin at 0, that does not show the reward led there; it may be a habit the pin
  installed and nothing removed, the mirror image of the Two Pair habit cloning installed. moe2 alone
  can't separate the two.
- **For the Hume aim**, the pin is an ought from chud entering through the action mask rather than
  the reward. It's recorded and it ends; it isn't nothing.
- **The control that would separate them:** the same code and settings with `MOE_EARLY_PIN=none`
  (a run `moe2-nopin`). Not registered or run yet. Without it, results from moe2 are about the MoE
  with an early Full House pin, not about the MoE.


## Added 2026-11-04 ~13:40 CST: completion criterion: the MoE has learned Full House

chud, 13:35 CST: "I am willing to make this entire MoE have to learn Full House before I say it's
complete." Recorded before any moe2 eval.

This is a criterion for calling the MoE finished, not a prediction, and it is about behavior: it is
met whether or not the early pin helped get there. Whether the reward alone leads there is a separate
question, for the no-pin control (review above, item 7).

### Measured (thresholds proposed by Claude; chud to confirm)
All with every pin at 0 (expert pins and the early pin), on the fixed eval seeds, with
`analysis/replay_events.py` + `analyze.py` on the saved eval games. "Full House on offer as the best
hand" uses the analysis's existing test: highest chips x mult at that moment's hand levels.
1. **The early head plays it.** In antes 1–3, when Full House is the best hand on offer, it plays a
   Full House (or a higher hand) at least **75%** of the time. Reference points from model 0.2's
   wins: Full House 0%, Flush 76%, Straight 92%. So the bar is "treats Full House the way it already
   treats Flush".
2. **The experts play it too.** Same test after the handoff, each expert separately: no expert below
   **50%** (Two Pair's expert included). Leveling can make Two Pair the better hand later in a run;
   the analysis's test already counts levels, so that case doesn't count against it.
3. **The Full House expert is in use.** Routed by rule in at least **5%** of eval handoffs (the first
   run: 0%).
4. **It holds.** 1–3 are met at two evals in a row, at least 20 rounds apart, after the pins reach 0.

I2 (Full House among the strongest experts) is separate and can fail while this criterion passes.

If moe2 ends without meeting it, the MoE isn't complete: next steps are decided then, with the
measurements in hand.


## Added 2026-11-05 ~10:40 CST: Full House not learned so far (moe2, through round 900)

An observation, not a scored test. The completion criterion above (2026-11-04 ~13:40) is measured
only once every pin is at 0, at about 100,000 training games (round ~1,041); at round 900 the pins
are at 0.13. Recorded now, before that eval exists, because chud asked to note that Full House
training has failed so far. The early head is **trained** in moe2 (`--moe-train-early 1`), not frozen.

### What the eval logs show (600 fixed seeds, every 20 rounds)
- **Neither the Full House expert nor the early head plays a Full House unless the pin forces it.**
  From round 540 to 900 (19 evals, pins 0.45 → 0.13): the Full House expert had a Full House on
  offer at 993 decisions; the pin bound 330 of them, and it played 326 Full Houses. The early head
  (pin: Full House or higher) had one on offer at 17,284 decisions; the pin bound 5,190, and it
  played 5,072. In every one of those evals, plays never exceeded the decisions the pin bound. That
  is what a policy that never chooses Full House on its own would give. These counters are per eval, not per
  decision, so they can't prove that no unforced Full House was ever played; the per-decision
  replay (`analysis/`) is the exact test, at pins 0.
- **The Full House expert is starved.** Games routed to it: 81 of the round-120 eval's handoffs, 87 at
  round 160, 60 at 400, 35 at 600, 18 at 800, **5 of 408 (1.2%) at 900**. The router needs two Full
  House plays in antes 1–3, and as the early pin fades the early head stops making them.
- Its eval record across all 46 evals: 2,207 routed games, 34 wins, 34 near wins. Not comparable with
  the other experts' records (each gets different games; I2's test is the paired comparison).

### Against the completion criterion, projected (not scored)
1. Early head plays it at ≥75% when it's the best hand: on course to fail (no unforced plays seen).
2. Experts play it at ≥50%: not measured (needs the per-decision replay), likely to fail.
3. Full House expert routed in ≥5% of eval handoffs: 1.2% at round 900 and falling; on course to fail.

### Not yet known
Whether, with the pins at 0, the reward alone leads any head to Full House. The first pin-free eval
(PR #95: `pins0.pt`, eval at the first round with every pin at 0) and the checkpoint at 100,200
games are the measurement points.

### An idea for later (chud, 2026-11-05)
Train each expert only on the games that expert won. Not designed or registered.


## Added 2026-11-05 ~11:00 CST: J, the next mixture of experts (design draft; nothing registered)

### What chud asked for (10:37–10:39 CST)
> "begin the formulation of a superior MOE based on the results of our currently running pin method,
> and we should also comb the winning strategies on the winning seeds and link it back to each
> expert. if we find out that a single expert is clearly better, (hopefully we can train a full
> house expert) we should probably have the early game model steer into the strategy that wins the
> most"

> "if the handoff is at the wrong blind, that'd be good to know. if the handoff doesn't work and
> doesn't modify strategy, that'd be good to know. if one strategy wins lots, that'd be good to know."

### Measurement first: `analysis/paired_experts.py`
Each expert's own record mixes the expert with the games the router gives it (Full House only gets
runs that already made Full Houses), so the logs can't link winning strategies to experts. The
tool plays every expert from the **same** handoff state: the early head plays a seed to the
handoff exactly as in an eval, then each of the 6 experts plays the rest. A seventh player is the
control: no handoff, so the early head plays on. Seeds are outside the eval 600. Repeated with the handoff at
other antes. What it gives, against chud's three questions:
1. **One strategy wins the most?** Each expert's win rate on the same positions, the positions on which each expert
   is best, the router's own choice, and the ceiling of a perfect router.
2. **Does the handoff change play?** The control's win rate against the experts', and each expert's
   plays after the handoff by hand type, from the same positions. Experts that play alike, or that don't beat
   the control, mean the handoff adds nothing.
3. **Is the handoff at the right blind?** All of it at handoff antes 3, 4 and 5 (moe2 trained at 4;
   at other antes the experts face positions they weren't trained from, so read it as a probe).
It also serves I2's paired test and the Full House completion criterion. First run: on `pins0.pt`
(moe2, every pin at 0).

### Design options, by what the measurement shows
- **One expert best from most positions:** a single expert may beat the mixture, and steering the
  early game toward its strategy is the right move.
- **The best expert depends on the position:** a learned router, which predicts each expert's outcome from the
  handoff state and is trained on the paired play-outs, instead of the hand-count rule. This also ends
  Full House's starvation, which comes from the rule (it needs Full House plays in antes 1–3).
- **Experts don't beat the control, or play alike:** the handoff isn't working; the experts need
  another design before anything is built on them.
- **Another ante is clearly better:** move the handoff there.

### How to steer the early game (Claude's recommendation)
- **Not with pins or a hand-specific bonus.** moe2's pins forced Full House for ~100,000 games, and
  the policy took none of it up: through round 900, Full House plays never exceeded what the pins forced
  (section above). A pin changes behavior while it binds; it doesn't change what the policy prefers.
  Steering by hand also writes a "how" into the ought, which aim 1 (behavior from the reward alone)
  can't afford.
- **Extend the early head's return through the handoff** (`--moe-cut 0`). Today its returns stop at
  ante 4 (chud's spec for moe1/moe2), so it is rewarded for reaching ante 4, not for reaching it
  in a position the experts can win from. With the return running to the end of the game, the
  reward alone favors whatever setups the experts convert. If a Full House expert wins most, the
  early head would learn to build toward it without being told to.

### Later
- chud's idea: train each expert only on the games it won. Close to self-imitation, which was turned off after it
  collapsed entropy (0.6.0). At ~1% wins the data is thin. Worth revisiting with the paired
  play-outs, which give more wins from good positions.

### Before moe3
Predictions about the paired results (chud's, if any) get registered here **before** the tool's first run
on `pins0.pt`. The scoring rules stay as written whatever the outcome, including if Full House doesn't win.

Status: **draft**.


## Added 2026-11-05 ~15:40 CST: moe2 at pins zero: results

moe2 stopped itself at 13:42 CST on 2026-11-05, as built (`--stop-after-pins0`, PR #98). The last eval
was at round 1042, the first round with every pin at 0, after 100,032 training games. Its weights are
released as `weights/balatro-bot-moe2-pins0.pt` ([system card](system_card_moe2.md)). Full numbers and
tables are in [moe2_evaluation.md](moe2_evaluation.md). This section scores what was registered and
records the findings. The analysis is Claude's, done on chud's request ("update the hypothesis and
write up the evaluation").

### Pins-zero eval (600 fixed seeds)
12.94 blinds per game, 9 wins, 13 near wins, 22 reached the final boss. Round 0 was 11.59 with 2 wins.
The run's best eval was 13.08 at round 1020, and its most wins 14 at round 1040.

### Completion criterion (2026-11-04 ~13:40): **not met**
1. **Early head plays Full House when it's the best hand, at least 75%: fail.** The early pin's counters
   at pins zero: a Full House or higher was on offer at 1,005 early-head decisions, and it played one
   once. The criterion's exact measure, "best hand at that moment's levels", uses a subset of those
   decisions, so its numerator is at most 1. The per-decision replay wasn't run; it can only confirm.
2. **Every expert at least 50%: not measured.** The Full House expert got no games. The others need the per-decision
   replay.
3. **Full House expert routed in at least 5% of eval handoffs: fail.** 0 of 442.
4. **Holds at two evals: not applicable,** since items 1 and 3 fail.

As the criterion says, the MoE isn't complete, and next steps are decided with these measurements
in hand. Full House's routing fell 81 → 5 → 0 (rounds 120, 900, 1042), as projected on 2026-11-05
~10:40.

### I2 (Full House among the strongest experts): **still unscored**
The evals can't test it: the router sent the Full House expert nothing. The paired test can, because
`paired_experts.py` forces every expert from the same handoff state, Full House's included. Its
design still needs chud's confirmation (2026-11-04 ~13:35).

### I1 (MoE beats model 1.0): **still unregistered**
No comparison exists. moe2 used 600 eval seeds and the earlier models 200, under a win test since
corrected. 1.0 is still being re-measured.

### Findings (details and tables in the evaluation)
- **The pins didn't teach the heads their hands.** With the pins at 0, the early head played Full House
  1 time in 1,005 chances. Experts play their own hand when it's on offer at these rates: Two Pair
  99.4%, Straight 78%, Flush 42%, Three of a Kind 37%, Pair 7%. Two Pair's near-total rate can't be
  credited to the pins without a baseline: model 0.2, where every head started, already leaned on
  Two Pair. The paired tool's no-handoff control gives that baseline.
- **The mid-run dip was the early game, and it recovered without the pins.** Training games ending
  by ante 3 went 33% → 42.5% (rounds 200–599) as the pins faded through 0.8–0.5. They were 30.6% by
  rounds 1000–1043, below the start. The learning rate halved at round 611 (update guard), close to
  the turn, so that cause isn't separated.
- **Blinds per game rose to the run's best by the end; wins didn't clearly.** Last six evals: 12.93 blinds and 8.3 wins on
  average, against 12.49 and 8.2 at rounds 100–200. Fresh-seed training wins: 1.12% (rounds
  100–199), 1.13% (1000–1043). Eval trend since round 600: +0.51 blinds per 100 rounds.
- **Most games are lost in antes 4–6** (352 of 600 at pins zero), and 56 by ante 1. Across all moe2
  games, each expert converts 15.5–20.7% of its final-boss fights.
- **Unexplained:** `ppo.kl` (policy that played vs policy being updated) rose 0.01 → 0.19 between
  rounds 200 and 600, while each update's own movement stayed about 0.025. One update of lag accounts for
  about 0.025. The suspects are hand-type exploration and float32 flooring; neither is tested.

### For aim 1 (Claude's reading, for chud to accept or reject)
The pins were an ought entered through the action mask, not the reward, and recorded as such
(review of 2026-11-04 ~13:50, item 7). With them at 0, the forced behavior is mostly gone. The bot
lost ground as they faded, then rebuilt on the reward and passed its starting point in blinds per game,
toward play the pins didn't ask for. The review's caveat stands: the weights carry the pins'
history, so this is a statement about this run, not about what the reward alone does from 0.2. The
no-pin control (`moe2-nopin`) is still unrun.

### Next (section J; nothing registered)
1. `analysis/paired_experts.py` on `pins0.pt`. chud's predictions, if any, are registered in J first.
2. Split `ppo.kl` by explored and sampled moves, and a low-temperature eval of `pins0.pt`.
3. Then `--moe-cut 0` from `pins0.pt`, as a MoE or a single network depending on 1.


## Added 2026-11-05 ~15:45 CST: Claude's predictions for the first paired run (section J)

Registered before `analysis/paired_experts.py` has run on anything. These are Claude's predictions,
made at chud's invitation ("would you like to make predictions?"); chud registered none. The run:
`pins0.pt`, 200 seeds (`--seed-tag paired-v1`, outside the eval 600), one rollout, handoff antes 3, 4 and 5.

### What this run can and can't resolve
About 74% of seeds reach the ante-4 handoff (442 of 600 in the pins-zero eval), so each player gets
about 150 games at ante 4. At the ~2% win rate after the handoff, that is about 3 wins each.
**Win rates can't be ranked at this size**, so none of the predictions below uses them; they use
blinds per game and the play-share table. Ranking win rates would take roughly ten times the seeds
(an overnight run). The tool's rows for different handoff antes are also not on the same seeds:
ante 5 only counts seeds that reach ante 5, which are stronger runs. So comparing antes in its
summary is biased, and no prediction is made about the best handoff ante until the summary compares
antes on the seeds common to all three.

### Predictions (all at handoff ante 4; scored from the tool's summary)
- **C1. The experts are not redundant (70%).** In the play-share table, for at least 3 of the 6
  hands, the expert named for that hand has a higher share of it than every other player (the 5
  other experts and "no handoff"). Expected among them: Two Pair, Straight, Flush. Not expected:
  Pair. *Falsified* if fewer than 3.
- **C2. The handoff beats playing on with the early head (60%).** "No handoff" averages at least
  0.5 fewer blinds per game than the mean of the 6 experts. Reason: the early head's return has
  ended at ante 4 for 100,000 games, so nothing has trained its play past there. *Falsified* if the
  gap is under 0.5 or the other way round.
- **C3. The experts are close (65%).** Between the best and worst expert, blinds per game differ by
  less than 1.0. Reason: across all of moe2's games, final-boss conversion spans only 15.5–20.7% by
  expert. *Falsified* at 1.0 or more.
- **C4. The router adds nothing over picking an expert at random (65%).** "Router (its own choice)"
  is within 0.3 blinds per game of the mean of the 6 experts. *Falsified* if it is 0.3 or more above.
- **C5. Full House is not among the top 2 experts by blinds per game (65%).** This bets against
  I2's second half on this measure (I2 is about win rate, which this run can't rank).

Any prediction whose condition can't be read from the summary (e.g. a player with no games) is
scored as void, not as passed.


## Added 2026-11-05 ~15:45 CST: the first paired run's size (2,000 seeds), and its pilot

**Size changed by chud, from 200 to 2,000 seeds** (15:37 CST: "why don't I try and do 2000 seeds?").
Same weights (`pins0.pt`), seed tag (`paired-v1`), handoff antes (3, 4, 5) and one rollout. The seed
list is fixed by the tag, so the first 200 of the 2,000 are the 200 that C1–C5 were registered
against. **C1–C5 are scored on all 2,000**, and also reported on the first 200.

### The pilot (32 seeds), and what was read from it
chud started the 200-seed run at ~15:35 and stopped it with Ctrl+C after 32 seeds, to size the run.
The 32 records are kept, and the 2,000-seed run resumes from them. Claude read the pilot **after**
C1–C5 were registered, and only to size the run: the time per seed, whether the moves before the
handoff matched, how often each ante was reached, the per-player counts, and the spread of paired
blind differences. No prediction was added or changed after reading it, and the pilot scores nothing.

- **The tool works on the game.** The moves before the handoff were identical for every player on every
  seed (0 mismatches at all three antes). The pins read 0 in the weights.
- **Speed.** 52.9 s per seed per worker, about 750–800 seeds an hour on 12 workers. 2,000 seeds is
  about 2.5–3 hours.
- **Spread.** The SD of the paired difference in blinds per game between two players on the same seed
  at ante 4 was 2.75 (range 2.0–3.6 across the 21 pairs).
- **To watch, too small to read:** 1 win in 378 games across all three antes, and the ante-4
  handoff reached on 18 of 32 seeds (56%), against 74% in the pins-zero eval.

### What 2,000 seeds can resolve (Claude's power estimates, two-sided α = 0.05, power 0.8)
Assuming ~74% of seeds reach the ante-4 handoff (fewer if the pilot's 56% holds):

| Difference between two players at ante 4 | Seeds needed |
|---|---|
| 0.5 blinds per game, any of the 21 pairs (Bonferroni) | ~800 |
| 0.3 blinds per game, one pair | ~1,200 |
| Win rate 2% vs 4% (paired) | ~2,000 |
| Win rate 2% vs 3% (paired) | ~6,000 |

So 2,000 seeds settles the blinds-per-game questions (C1–C5 and chud's three questions on that
measure), and shows whether any expert wins about twice as often as another. Smaller win-rate
differences, which I2's ranking may need, would take ~6,000 seeds. The run can be extended to that
later without changing anything, because the file resumes and the seed list is fixed. If the win
rate after the handoff is nearer 1% than 2%, as the pilot hints, every win-rate row needs more seeds.


## Added 2026-11-05 ~21:10 CST: the first paired run, scored (section J)

`balatro-work/paired_moe2.jsonl`: `pins0.pt`, seed tag `paired-v1`, **2,000 of 2,000 seeds, 0 errors**,
handoff antes 3, 4, 5, one rollout, finished ~19:49 CST. 0 prefix mismatches at every ante; the pins
read 0. Scored with `analysis/score_paired.py` (reproduces every number below from the file).

### C1–C5 (handoff at ante 4; all 2,000 seeds, 1,368 reached it; first 200, 129 reached it)
| | All 2,000 | First 200 (registered) | Verdict |
|---|---|---|---|
| **C1** experts not redundant (≥3 of 6) | 2 of 6 | 1 of 6 | **Falsified** |
| **C2** handoff − no handoff ≥ 0.5 blinds | +1.08 (95% CI 0.94–1.23) | +1.40 (±0.46) | **Pass** |
| **C3** best − worst expert < 1.0 | 0.24 (Straight 16.21 − Pair 15.97; ±0.17) | 0.65 (±0.53) | **Pass** |
| **C4** router − expert mean < 0.3 | +0.02 (±0.11) | −0.06 (±0.40) | **Pass** |
| **C5** Full House not top 2 by blinds | 5th of 6 | 6th of 6 | **Pass** |

- **C1.** On all 2,000, the named expert leads its own hand only for Pair (5.75% vs 5.38%) and Three of
  a Kind (6.938% vs 6.936%, a tie at this precision). The three expected to lead, Two Pair, Straight
  and Flush, all failed. Full House is 0.0% for every player. Claude's 70% was wrong.
- C2–C5 held. C2's and C3's reasons were stated at registration; C3's held more tightly than the
  1.0 margin needed.

### chud's three questions
1. **Does one strategy win the most?** No. Wins at ante 4 (of 1,368): Straight 38, Three of a Kind 34,
   Flush 30, Pair 28, Two Pair 27, Full House 25. Cochran's Q p = 0.24 (ante 3: 0.50; ante 5: 0.83).
   Straight vs Full House, 24:11 discordant, p = 0.041 uncorrected, is the most-vs-fewest pair picked
   after the fact: not evidence. Blinds: no pair of experts differs past Bonferroni at ante 4 or 5; at
   ante 3, Flush is ahead of Two Pair (+0.29) and Straight (+0.39).
2. **Does the handoff change play?** It changes the outcome (C2: +1.08 blinds per game; wins 11 →
   25–38 of 1,368) but not the hand mix. All seven players, the early head playing on included, play
   the same mix within about a point: Two Pair ~65%, Straight ~14%, Flush ~7%, Three of a Kind ~7%,
   Pair ~5%, Full House 0.0%. At pins zero the six experts behave as one policy on this measure. Open:
   whether the mix is set by the build the early head hands over (jokers, planets, deck), or whether the
   experts converged; this run can't separate the two.
3. **Is the handoff at the right ante?** **Earlier is better** among the antes tested. The tool's
   per-ante rows can't answer this (different seeds), and neither can the 1,003 seeds common to all
   three antes: those are the seeds on which the early head itself reached ante 5, which is the
   no-handoff game, so early handoffs are compared only where the early head is known to survive
   (that comparison favors ante 5, wrongly). Scored instead as a policy over all 2,000 seeds, where a
   seed that never reaches the handoff keeps the early head's game:

   | Handoff | Blinds per seed, expert mean | Router's choice | Wins per 2,000 (expert mean / router) |
   |---|---|---|---|
   | never | 11.75 | – | 11 |
   | ante 3 | **12.65** | 12.60 | 34.7 / 29 |
   | ante 4 | 12.49 | 12.51 | 30.3 / 31 |
   | ante 5 | 12.11 | 12.14 | 29.3 / 33 |

   Expert mean: ante 4 − ante 3 = −0.16 ± 0.12; ante 5 − ante 4 = −0.38 ± 0.09. Win rates (~1.5–1.7%
   per seed) don't separate. Antes 1–2 were not tested, so this doesn't show that ante 3 is best, only
   that later is worse; the trend says an expert beats the early head from wherever it takes over.

### I2 (Full House among the strongest experts): still unscored
Its test design was never confirmed and its size never set, so this run doesn't score it. For the
record: Full House's wins rank 6th of 6 at ante 4 (25), tied 5th at ante 3 (31), 2nd at ante 5 (31),
and no win-rate difference between experts is detectable at any ante. The router sent it no seeds at
any ante.

### Reach rate
68.4% of seeds reached the ante-4 handoff, against 74% (442/600) in the pins-zero eval. The eval
seeds chose the checkpoint, so some of the gap is expected.

### What it means for the next run (section J; nothing registered)
The experts are not separate strategies, and the handoff's gain is that any expert plays past ante 3
better than the early head, which was never trained past ante 4. Candidates: a single network
(`--moe-cut 0` without the expert split), or a paired run with `--handoff-antes 1,2` to see whether an
expert should play the whole game.


## Added 2026-11-05 ~21:15 CST: what moe2 says about aim 1 (chud's reading, with limits)

### The ought was never modified
chud (21:08): "we haven't modified the ought even once. ought to favor a hand didn't affect play
patterns once the favor ran out. that is good data. the one ought was always blinds."

The reward of record was `blinds` for every head and every round of moe2. The hand pins went through
the action mask: they restricted what the bot could play, never what it was rewarded for. Earlier
write-ups (including Claude's, and `docs/moe2_evaluation.md` before this date) called the pins an
"ought". That was a mislabel, now corrected there.

### What it shows
**A constraint that isn't the ought leaves no trace once it's lifted.** With the pins at 0, the
early head played a Full House once in 1,005 chances, and in the paired comparison (above) all
six experts play the same hand mix as each other and as the early head, which was never pinned. This
is a negative control for aim 1: changing what the bot may do, with the ought held fixed, didn't move
where play settled.

### chud's claim, and what this data supports of it
chud: "there is one optimal strategy towards single rewards."

What the data supports is narrower: **under `blinds`, which hand to play is fixed by the reward and the
build.** It doesn't show a unique optimal strategy, for two reasons:
1. **Same hand mix is not same strategy.** The early head and the experts play the same mix, yet
   from the same positions the experts score 1.08 more blinds per game. Something else differs
   (shop, discards, which Two Pair is played). The hand mix is too coarse to see it.
2. **The experts aren't independent samples.** All six started from model 0.2's weights. The early
   head, trained from random weights, is the independent check, and it agrees on the mix.

Whether an optimal policy is unique is beside the point for aim 1, and games like this usually have
ties. The question aim 1 needs answered is whether the ought selects the strategy.

### The experiment this points to
Section C's first prediction, **"Different oughts, different behavior"** (2026-11-01; not the C1 of the
paired run): bots trained on `blinds`, `antes` and `win`, from the same starting weights, on the same
game, compared per seed. moe2 shows that play converges under one ought. If different oughts
converge to measurably different play from identical facts, the facts didn't fix the play: the ought
did. That is the is-ought gap shown in behavior. Its statistics are still *(chud to confirm)*, and
nothing is registered.
