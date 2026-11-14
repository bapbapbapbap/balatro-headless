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

## Added 2026-11-11 ~09:25 CST: v4b stopped at round 280 on the sell stop; S1-S3 scored

v4b resumed from round 70 on the reorganized trainer on 2026-11-10 at 11:51 CST and stopped itself at eval 280
at 15:58 CST, 4.05 hours later: "eval 280 sold 2.56 jokers per game (--stop-sells 1.5)". 26,880 training
games in all, 5 wins in them, 1 win in 600 at eval 260, 0 at eval 280. `best.pt` is round 240.

| eval round | blinds/game | joker sells/game |
|---|---|---|
| 60 | 4.08 | 0.49 |
| 80 | 4.90 | 0.12 |
| 100 | 5.32 | 0.32 |
| 120 | 6.01 | 0.10 |
| 140 | 5.74 | 0.06 |
| 160 | 6.62 | 0.09 |
| 180 | 6.69 | 0.07 |
| 200 | 6.74 | 0.34 |
| 220 | 7.63 | 0.73 |
| 240 | 8.31 | 1.36 |
| 260 | 8.29 | 1.03 |
| 280 | 7.20 | 2.56 |

- **S1 (no eval above 1.5 sells/game): failed** at eval 280.
- **S2 (best eval above v4's 4.68): met.** 8.31 at round 240.
- **S3 (5.58 by round 200): met.** 6.01 at round 120.
- W1-W4 and aim 1's precondition: not run.

**The registered reading of S1 does not hold.** The registration says an eval above 1.5 "is the failure of
change 1" (the hand clock on the table network's lambda). The sells that tripped the stop are the shop
network's. On its own training games the table network's probability of selling was about 0.0001 on blinds it
survived and 0.0005 on the blind it died in at round 260 (`analysis/v4_sells.py`; 3 table sells in 67 games at
round 160, none at 260), against v4's mid-blind selling. Change 1 did what it was for; the stop caught a
different fault. The next section is about that fault.

## Added 2026-11-11 ~09:25 CST: why v4b's shop network sold, and what it cost (diagnosis after the stop)

Claude's, reviewed by chud. Scripts: `analysis/v4b_drop.py` (the 600 eval games of rounds 240, 260 and 280
replayed, 0 mismatched) and `analysis/v4b_shop_credit.py` (training games of rounds it-4..it replayed under
checkpoint it, the shop network's credit rebuilt as training forms it). Their output is in chud's work folder
(`v4b_drop.jsonl`, `v4b_shop_credit*.jsonl` and `.txt`), not in the repo.

**What the selling cost.** A free-slot sell is a joker sold with an empty joker slot: nothing is gained but its
sell value (about $2).

| eval round | sells per blind played | free-slot share | free-slot sells/game | jokers held entering ante 3 | die in ante 2 | die in ante 3 |
|---|---|---|---|---|---|---|
| 240 | 0.146 | 0.59 | 0.80 | 4.08 | 0.20 | 0.29 |
| 260 | 0.111 | 0.53 | 0.55 | 4.07 | 0.20 | 0.26 |
| 280 | 0.312 | 0.75 | 1.91 | 3.40 | 0.28 | 0.43 |

("die in ante N" is the share of games reaching ante N that end there.) At round 280, games with a free-slot
sell before ante 3 entered it with 3.04 jokers and died there 49% of the time; games without one, 3.83 jokers
and 36%. Free-slot selling accounts for part of the fall from 8.29 to 7.20, not all of it: the games with no
free-slot sell also held fewer jokers and died more than at round 260 (ante 3: 36% against 24%). Sells did not
hurt on the way up: from round 180 to 240 sells rose from 0.07 to 1.36 per game while blinds rose from 6.69 to
8.31, and at round 240 41% of sells were with every slot full (selling to make room).

**Another event in the same window.** Round 268's update was the largest of the stretch (KL 0.068). Training
blinds per game fell to 4.79, 5.37 and 3.70 in rounds 269-271 and were 7.0-8.5 again in rounds 276-279. Eval 280
came 12 rounds after it. No checkpoint between 260 and 280 separates this from the selling.

**The shop network's credit on its sells** (training games, shop decisions where a sell was legal; adv_n is the
normalized advantage the policy is trained on; td is reward + the critic's next value - its value):

| checkpoint | shop sells (games) | P(sell) when taken | td on sells | adv_n, free slot | adv_n, slots full | critic's value per joker held |
|---|---|---|---|---|---|---|
| 160 | 5 (335) | 0.001 | | | | |
| 200 | 163 (335) | 0.014 | -0.17 | -0.67 | +0.01 | +0.03 |
| 220 | 156 (335) | 0.028 | +0.26 | -0.11 | 0.00 | -0.03 |
| 240 | 397 (335) | 0.067 | +0.17 | 0.00 | +0.15 | -0.07 |
| 260 | 264 (335) | 0.054 | +0.18 | +0.06 | 0.00 | 0.00 |
| 280 | 479 (268) | 0.090 | -0.15 | -0.39 | -0.03 | +0.45 |

(The last column is the coefficient on jokers held in a regression of the shop critic's value at `SHOP` states,
antes 1-4, on money, jokers held, a free slot and the ante.)

1. **Credit never favored free-slot sells** at any sampled checkpoint. It was against them at round 200, neutral
   at 220-260 and against them again at 280. What changed between 200 and 220 is that the charge went away.
2. **The charge went away because the critic's value rose after a sell,** by +0.52 at round 220, +0.35 at 240
   and +0.36 at 260 on free-slot sells, and by -0.10 at 200 and -0.01 at 280.
3. **It is not money.** The critic's value moves by -0.01 to +0.03 per dollar at every checkpoint; the rise does
   not follow the sell value (correlation -0.22 to +0.08); at round 220 it is the same with under $4, $4-10 or
   more; selling a consumable moves the value by +0.05.
4. **It is the critic's valuation of jokers.** The rise is there while the critic puts no value on a held joker
   (220-260) and gone when it does (280: +0.45 per joker). Over rounds 220-260 (817 sells) 81% of joker types
   with 8 or more sells have a positive mean td on being sold; the td is +0.98 when the sold joker was the only
   one held, +0.48 with two, +0.19 with three or four. Sells with every slot full rise by 0.15-0.20, which is
   what the critic gives a free slot.
5. **At round 280 the credit is against the selling** (37% of sells with a positive advantage; Green Joker
   -1.99, Raised Fist -1.39, Supernova -0.90), while the policy still sells more than at any earlier round.

**Not known.** Why the critic rated the post-sell state higher at rounds 220-260 (the files record the sold
joker, not the jokers kept); what raised the probability of selling under neutral credit (the shop entropy
bonus is a candidate, untested); how much of eval 280 is round 268's update. Rounds 221-235 and 261-275, where
selling rose, were not sampled.

## Added 2026-11-11 ~09:25 CST: v4b continues from round 280 with the sell stop off, registered before its config changes

chud's call (2026-11-11 09:12: "then lets keep growing!"; 09:11: "you just said it isn't money. why do we care
about sells?"). Design and predictions Claude's, reviewed by chud.

**What changes.** `stop_sells` goes from 1.5 to 0 (off) in `configs/v4b.toml`. Nothing else: same run, same
folder, same weights, same reward, same stall rule (`--stall-evals 25`: it stops after 25 evals with no new best
above 8.31). `sells_mean` stays in every eval as a measurement.

**Why.** The stop was written for v4's fault (the table network selling in the blind it dies in), which v4b
does not have. The sell count it watches rose alongside the best evals of the run. The fault it did catch, the
shop network's free-slot selling, was already being charged by the credit at round 280, with the critic's error
gone. Continuing tests whether the policy unlearns it.

**How it resumes.** From `runs/v4b/latest.pt`, saved after round 279 (the stop came inside round 280, before
that round's save). Round 280 and its eval run again, so `metrics.jsonl` gets a second eval-280 row; the one
after this section's time is the continuation's. Best eval (8.31) and the stall count carry over in the
checkpoint. This is not a rollback: nothing learned through round 279 is undone.

**Predictions (Claude's, 2026-11-11).**
- **T1. The selling was a transient:** every eval from 340 to 400 has under 1.5 joker sells per game. **50%.**
- **T2. It keeps growing:** an eval from 300 to 400 is above 8.31 blinds/game. **45%.**
- **T3. The two go together:** the first eval above 8.31 has fewer free-slot sells per game than eval 280's
  1.91 (`analysis/v4b_drop.py` on that eval's games). **80%, given T2.**

**What each outcome says.** T1 and T2 met: the selling came from a critic that had not yet learned what jokers
are worth, and v4b's design stands. Sells stay above 1.5 through eval 400 while the credit is against them:
something other than credit holds them up, and the shop entropy bonus is the first thing to test in a v4c.
T2 met with sells still high: the sell count was the wrong thing to watch. No eval above 8.31 by round 400 with
sells down: round 268's update, or something not yet looked at, did lasting damage; compare `ckpt_00260.pt`.
The run is not stopped at round 400; the stall rule decides that.

## Added 2026-11-11 ~09:30 CST: whose the 09:25 sections are

The diagnosis and the predictions T1-T3 in the three sections above are Claude's own (model: Claude Fable 5.2,
`claude-fable-5-2`), reviewed by chud (09:18: "your hypotheses are your own (note your model) and note reviewed by
me"). The decision to continue is chud's.

## Added 2026-11-13 ~11:30 CST: T1-T3 scored

From `runs/v4b/metrics.jsonl` (chud's laptop, read 2026-11-13). The continuation's eval 280 repeated the first
one exactly (7.20 blinds/game, 2.56 sells). Evals after it, blinds / joker sells per game:
r300 8.62/1.62, r320 8.40/1.59, r340 8.00/1.72, r360 9.59/0.78, r380 8.89/2.19, r400 8.32/1.30, r420 9.08/0.92.

- **T1 not met.** Evals 340 and 380 sold more than 1.5 jokers per game (1.72, 2.19).
- **T2 met.** Eval 300 was 8.62 > 8.31, the first new best after the stop.
- **T3 open.** It needs `analysis/v4b_drop.py` on eval 300's games (free-slot sells per game vs eval 280's 1.91);
  not run yet.

By this section's own "what each outcome says": T2 met with sells still high, so the sell count was the wrong
thing to watch. The best evals since then sell about 2 jokers per game (r1840: 13.40 blinds, 2.30 sells). The run
has not met its stall rule; at round 1890 it had trained 48.4 h and 181,536 games.

## Added 2026-11-13 ~11:30 CST: where v4b's games end, economy or scaling (diagnosis while it runs)

The question is chud's (09:45: "economy or scaling basically"), and so is the hoarding guess (09:47: "I bet its
also saving more than its interest cap and winning with big amounts saved"); reviewed by Claude. The tools, the
four decision rules below and the reading are Claude's, reviewed by chud. **The rules were written down before
the data existed (the Project's status note, 09:55) but were not committed to git first, so by this repo's rule
they are not registered predictions;** they are scored here only as a record of what was expected.

**Eval record.** Evals are 600 games at eval_temp 1.0, about ±0.2. Best 13.40 blinds/game at round 1840
(`best.pt`); evals r1700-1880 average 13.09, r1600-1680 12.81. moe2's best was 13.08 (r1020). Caveat below
(Hieroglyph) before calling it a record over moe2.

**What the bot can see.** Money, the interest cap and the interest it would earn are already in the observation
(`harness/encode.lua`, GLOBAL features 4, 11 and 23-25). Money is not in the reward and not in phi_build except
where a joker reads it (Bull, Bootstraps). The shop entropy coefficient has been at its floor 0.001 since 4M
decisions (`shop_ent` in `configs/v4b.toml`); the run is past 25M.

**Where games end** (saved move text of evals r1800-1880, 3,000 games, 75 wins, no replay): boss blinds 65.8% of
deaths, big 24.3%, small 9.9%, bosses leading at every ante. Share of all games ending in ante 1: 10.4%, 2: 9.4,
3: 7.9, 4: 10.5, 5: 20.5, 6: 18.4, 7: 12.4, 8: 6.0. Rerolls: 0.016 per shop visit.

**Replay** (`analysis/walls.py` on evals r1840/1860/1880: 1,800 games, 0 mismatches, 51 wins; report
`analysis/walls_report.py`, output `analysis/results/walls_v4b/walls_report.txt`):
- Money. Median at death by ante: $4, $8, $16, $22, $36, $47, $44, $67 (antes 1-8); at a win $78. Shop exits
  over the cap: 0.1% (ante 1), 20% (3), 60% (5), 77% (8). 97.2% of over-cap exits made no reroll; in 67% the
  money over the cap alone paid for 3+ rerolls; mean excess $30. Seed Money in 11 games, Money Tree in none.
- At the same blind, games that passed held as much money as games that died, or more (ante 7 boss: $53 died,
  $67 passed; ante 8 boss: $72, $85). Late over-cap exits seldom had a free slot (ante 5: 12% left an affordable
  joker with a free slot; ante 6: 4%).
- Build, died vs passed at the same blind. Antes 1-3: 1-2 fewer jokers, empty slots (ante 2 big 1.35 vs 2.84,
  ante 3 big 2.13 vs 3.93). Antes 5-8: same joker count (~4.9, slots full), same and few xMult jokers (~0.3, by an
  approximate key list), main-hand level about one lower (ante 5 boss 4.83 vs 5.41, ante 6 big 5.26 vs 6.19,
  ante 7 big 6.58 vs 7.48, ante 7 boss 7.39 vs 8.17).
- Bosses. Death rate when faced, antes 1-3: The Mouth 22%, The Needle 18%, The Eye 16%, against The Goad 4%,
  The Pillar 4%, The Psychic 5%. Antes 4-8: The Eye 55%, The Needle 56%, The Wall 52%, The Mouth 50%, against
  The Pillar 25%, The Psychic 26%. Violet Vessel 95% (36 of 38).
- Margin at the death blind (score / target): median 0.76 (antes 1-3), 0.78 (4-8); at least 0.8 in 44% / 47%,
  under 0.5 in 14% / 15%.
- Hieroglyph / Petroglyph (ante back by one) bought in 344 of 1,800 games; replayed antes add 0.58 blinds per
  game to the eval average. Under the "blinds" reward an ante replayed pays 3 more. moe2's rate is not known, so
  13.09 vs moe2's 12.93 is not yet like for like.

**The rules, scored.**
- H1, hoarding costs games (ante 4-8 deaths over the cap at least 30%; losers over the cap more than survivors
  at the same ante; most over-cap exits leave an affordable joker in a free slot, or reroll under 0.1 per
  visit): **not met.** 59% and the rerolls hold; losers do not hold more than survivors. Hoarding is real and
  policy-wide (chud's guess is right about that), so this data cannot say whether spending would help: nothing in
  it spends.
- H2, build (losers under the cap, with fewer jokers / xMult jokers / lower main-hand levels): **partly.** Early,
  fewer jokers with slots free; late, lower main-hand level with the same jokers.
- H3, early boss effects (top 4 bosses at least 50% of ante 1-3 boss deaths, scoring under half the target):
  **not met** (43.6%; mean ratios 0.55-0.75). Bosses that restrict which hands count (The Mouth, The Needle, The
  Eye) still kill 3-5x as often as others.
- H4, near misses (most deaths at least 0.8 of target): **not met**, nor its opposite.

**Reading (Claude's).** Early games end poor and short of jokers; late games end rich with full slots and a
weaker main hand. A late game that leaves $30 idle could have bought planets at $3 or rerolled for them: that
link is an inference, not tested here. The cheapest test leaves the reward alone: raise the shop entropy floor
(0.001 to about 0.005) in a registered v4c, and see whether late rerolls, planet buys and main-hand levels rise.
A money term in the shaping would also work on it, but it writes a human judgment ("money over the cap is
wasted") into what the bot learns from; that matters for aim 1. Nothing here is registered; chud decides.

## Added 2026-11-13 ~11:30 CST: whose the 11:30 sections are

The scoring of T1-T3, the replay tools, the four rules and the reading in the two sections above are Claude's
(model: Claude Opus 5.5, `claude-opus-5-5`), reviewed by chud. The questions and the hoarding guess are chud's.

## Added 2026-11-13 ~11:40 CST: why v4b hoards, four explanations and the test between them (registered before data)

chud paused v4b at ~11:25 so the laptop can run this. The premise is chud's: from his own play, spending money
late correlates with more blinds. The question is his too: what keeps v4b from spending (the walls section above:
late shop exits over the interest cap 60-77%, 97% of them with no reroll, 0.016 rerolls per shop visit).

**A correction first (Claude's).** In chat Claude said the build shaping "can't see" spending, because
`H.offer_values()` gives 0 to rerolls, packs and the vouchers phi_build doesn't count. That overstated it. v4b
discounts on the blind clock (`gamma_blind`, `lam_blind`), so inside one shop gamma = lambda = 1, and with
`build_shaping = 1` a reroll followed by a buy in the same shop gets the buy's phi_build gain back through GAE.
The 0 only means the bot can't see a reroll's worth before taking it.

**Four explanations, all open.**
- A. Exploration collapse: the shop entropy floor (0.001 since 4M decisions) took hold before rich late shops were
  common, so P(reroll) is near 0 and the few rerolls sampled can't move the policy.
- B. The signal says no: a reroll the bot does take doesn't pay under its training return, because its buying
  after one is poor, or phi_build misprices what rerolls find (option value, xMult jokers its 16 deals miss).
- C. Spending doesn't pay in this setting (red deck, white stake, the `blinds` reward), even spent well.
- D. Mechanical: reroll not offered when it should be.

**The test** (`analysis/economy_branch.py`, this commit; no data yet). From the eval games of rounds 1840, 1860,
1880, one shop exit per game: an `end_shop` at ante >= 5 with money over the interest cap, picked with a fixed RNG
per seed. At it, `best.pt`'s shop network gives P(reroll) and the other moves' probabilities (temperature 1) and
its value V. Three branches from that state: `end` (the bot's own move), `reroll` (one reroll, then the bot plays
on), `spend` (a scripted spender: keeps the interest cap, spends the rest on positive-offer-value buys, joker swaps
phi_build gains at least 0.02 on, vouchers with non-negative offer value, Celestial and Buffoon packs, then
rerolls; then the bot plays on). Each branch is played out 8 times by `best.pt` at temperature 1, play-out p
sharing its sampling RNG across branches. Recorded: blinds beaten after the branch point, the trainer's discounted
return from there (the shaping telescopes to the same -phi_build at the shared point, so it cancels), wins, and
each branch's one-step view, delta = phi_build' - phi_build + V' - V. Differences are paired by state, with 95%
bootstrap intervals over states. The number of games is set from the smoke's timing before any branch result is
read; the smoke's output is not part of the result.

**Decision rules** (written before the data):
- D: reroll is offered at fewer than 99% of the qualifying exits where it is affordable -> D holds; fix the harness
  first and read nothing else.
- A: median P(reroll) at the branch points under 0.01, and reroll minus end discounted return above 0 (interval
  above 0).
- B: reroll minus end return not above 0 (interval reaches 0 or below), and spend minus end blinds above 0
  (interval above 0). If spend's mean delta is at most end's while its play-outs gain, the learner's one-step view
  (critic and phi_build) misprices spending.
- C: neither reroll nor spend above end, in blinds or return (each interval reaches 0 or below).
- Size: spending is worth a v4c only if spend minus end is at least +0.3 blinds per state.
- A and a positive spend can both hold: then exploration is at least part of it, and v4c starts there.

## Added 2026-11-13 ~11:40 CST: whose the 11:40 sections are

The premise (spending correlates with blinds, from his play), the question and the call to pause v4b are chud's,
reviewed by Claude. The four explanations, the correction, the test, the scripted spender and the decision rules
are Claude's (model: Claude Opus 5.5, `claude-opus-5-5`), reviewed by chud.

## Added 2026-11-13 ~18:20 CST: why v4b hoards, the test's result

Run on chud's laptop (finished 16:41 CST), as registered except play-outs: 4 per branch, not 8, set from the
smoke's timing before any branch result was read. Rounds 1840, 1860, 1880: 1,800 eval games, 0 errors, 722 with a
qualifying exit, all three branches played at all 722. Output: `analysis/results/econ_v4b/summary.txt`.

| branch | blinds after | discounted return | wins | one-step delta |
|---|---|---|---|---|
| end (the bot's move) | 2.868 | 2.576 | 6.75% | -0.082 |
| reroll (one, then the bot) | 2.996 | 2.684 | 7.55% | +0.011 |
| spend (scripted, then the bot) | 3.162 | 2.824 | 7.96% | +0.165 |

Paired minus end, 95% bootstrap over states: reroll blinds +0.128 [+0.038, +0.219], return +0.108 [+0.035,
+0.181]; spend blinds +0.293 [+0.164, +0.426], return +0.248 [+0.142, +0.356]. At the branch points P(reroll)
median 0.0152 (under 0.01 in 35% of states), P(end_shop) median 0.690, P(buy) median 0.

**Scored by the registered rules:**
- D: missed. Reroll offered at 100% of 4,126 qualifying exits, all affordable.
- A: missed, on its threshold. Reroll minus end return is above 0, but median P(reroll) is 0.0152, not under 0.01.
- B: missed. Its first condition needs reroll minus end return not above 0; the interval is above 0. Its
  sub-clause (spend's delta at most end's) also does not hold: the one-step view ranks spend > reroll > end.
- C: missed. Both reroll and spend beat end in blinds and return.
- Size: missed by 0.007. Spend minus end is +0.293 blinds per state against a bar of +0.3; the interval contains
  0.3. By the rule as written, spending is not shown worth a v4c.

None of the four explanations is supported by its rule. chud's premise holds in this setting: late spending pays
(+0.29 blinds and +1.2 points of win rate from one exit), and a single reroll pays (+0.13). The bot almost never
takes either, though its own one-step view (critic plus phi_build) rates ending the shop lowest of the three.
Why the policy doesn't follow that is not answered by this test.

**Not registered, read after the data** (per-state blinds, spend minus end / reroll minus end): ante 5 +0.31 /
+0.20 (n 427), ante 6 +0.29 / -0.01 (187), ante 7 +0.39 / +0.13 (74), ante 8 -0.08 / +0.01 (34); money $70+
+0.46 (196), under $40 +0.20 (196); spend beat end in 40% of states, lost in 30%. The spender's gain came mostly
from rerolls (2.76 per state) and swaps (1.07), plain buys 0.46. The test branched one exit per game; these games
had 5.7 qualifying exits each, so what spending at every one would add is untested.

## Added 2026-11-13 ~18:20 CST: whose the 18:20 sections are

chud ran the test. The scoring and the unregistered breakdowns are Claude's (model: Claude Opus 5.5,
`claude-opus-5-5`), reviewed by chud.
