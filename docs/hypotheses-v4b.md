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
