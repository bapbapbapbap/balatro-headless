# Won-game analysis: v2, 709 training wins

Run 2026-11-02 (~20:45 CST). Every win in `runs/v2/wins.jsonl` up to round 492 (eval games
excluded) was replayed move by move in the sim, on the same action list it was played on, with an
event logger. All 709 replayed, with 0 mismatches. For contrast, 560 recent training games
(rounds 488–492; 16 won) were replayed the same way.
Scripts: `replay_events.py` (replay + event log) and `analyze.py` (the tests below).
Every rate is compared with a baseline from the bot's own data, or from the options it had.

## Reading it

**Learned (beats its baseline):**
- **Planets for its main hand.** Offered planets for two or more hands, it took its main hand's
  planet 91% of the time (207/227). Picking at random would give 25%.
- **Most-leveled hand = most-played hand,** slightly above chance: 46% vs 42% (p = 0.004).
- **Clever Joker raises Two Pair plays** from 47% to 57.5% while held.
- **Two Pair-scaling jokers co-occur with its Two Pair play.** Spare Trousers is in 36% of wins
  and 13% of losses; Clever + Trousers is the strongest pair in winning lineups (lift 1.59).
- **Builds to the interest cap** ($25+ leaving the shop) by ante 3–4.
- **Plays a little around The Psychic and The Eye.** More 5-card plays under Psychic (80% vs 70%);
  fewer repeated hands under Eye (19% vs 29%).

**Not learned:**
- **Full House: 0 of 2,274** times it was the best hand on offer. 2,075 of those were broken
  down into Two Pair.
- **No level-then-play.** After using a planet, plays of that hand go from 27.8% to 28.3%
  (+0.4 ± 0.6 points).
- **About 20% of plays score nothing** under The Psychic, The Mouth and The Eye.
- **Under The Ox it plays its most-played hand more,** not less (54% vs 44%), and loses all its
  money each time.
- **Hoards money:** median $97 at ante 6 and $122 at ante 8 leaving the shop. Interest stops at
  $25. Rerolls stay under one per shop; it skipped 5 of 11,804 Small/Big blinds.
- **Joker drags are 42% of the moves in wins** (130 per game).

**Caveats.**
- Wins vs losses in "held at any point" is biased: wins last longer and see more shops.
- The debuffed-cards-per-play numbers under bosses have no proper baseline.

## Full output

```
# Patterns in 709 won games (0 failed to replay); contrast: 560 recent games, 16 won

## 0. Hand mix
most-played hand per game: Two Pair 555, Pair 149, Flush 2, High Card 2, Straight 1
share of all plays: Two Pair 47.5%, Pair 25.3%, Straight 10.2%, Flush 6.6%, Three of a Kind 5.6%, High Card 4.8%, Four of a Kind 0.1%, Straight Flush 0.0%
  Straight was the best hand on offer 4670 times; played it 4304 (92.2%)
  Flush was the best hand on offer 3595 times; played it 2739 (76.2%)
  Full House was the best hand on offer 2274 times; played it 0 (0.0%)
  Straight Flush was the best hand on offer 79 times; played it 2 (2.5%)
  Four of a Kind was the best hand on offer 177 times; played it 27 (15.3%)

## 1. Hand focus: is its most-leveled hand the hand it plays most?
most-leveled == most-played: 299/651 = 45.9% (shuffled across games: 42.3%; permutation p = 0.0040)

## 2. Planet picks: offered >=2 planets, does it take its main hand's planet?
choices: 227; took its main hand's planet 207 = 91.2%; uniform pick gives 25.4%; one-sided binomial p = 2.9e-98

## 3. Level, then play: does using a planet shift play toward that hand?
planets used: 2164; plays of that hand, 20 before 27.8%, 20 after 28.3% (+0.4 pts, 95% +/-0.6)

## 4. Does it just play the highest-HUD hand on offer?
plays: 45722; played the best hand type on offer: 87.0%
most common departures (best -> played): Full House -> Two Pair 2075; Pair -> Two Pair 815; Flush -> Two Pair 693; High Card -> Pair 452; Straight -> Two Pair 310; Pair -> High Card 217

## 5. Joker-hand synergy: holding a hand-type joker, does it play that hand more?
  sly (Pair): before buying 78.1% -> while held 80.9% (n held 5690)
  jolly (Pair): before buying 78.3% -> while held 80.1% (n held 3964)
  clever (Two Pair): before buying 47.0% -> while held 57.5% (n held 2404)
  mad (Two Pair): before buying 47.5% -> while held 47.2% (n held 1838)
  duo (Pair): before buying 78.4% -> while held 80.3% (n held 707)
  devious (Straight): before buying 10.2% -> while held 11.2% (n held 402)
  wily (Three of a Kind): before buying 5.6% -> while held 5.9% (n held 353)
  crafty (Flush): before buying 6.6% -> while held 4.4% (n held 338)
  droll (Flush): before buying 6.6% -> while held 8.0% (n held 264)
  zany (Three of a Kind): before buying 5.6% -> while held 6.3% (n held 190)
  order (Straight): before buying 10.2% -> while held 11.0% (n held 82)
  crazy (Straight): before buying 10.2% -> while held 9.3% (n held 75)
  trio (Three of a Kind): before buying 5.6% -> while held 2.7% (n held 37)
hand-type jokers bought after 10+ plays: 164; avg share of its earlier plays containing that hand: 61.6%

## 6. Jokers: final lineups, wins vs recent losses, pairs above chance
most common in winning final lineups: green_joker 55.9%, banner 40.3%, blue_joker 37.2%, trousers 35.5%, abstract 29.5%, supernova 27.9%, stuntman 23.7%, raised_fist 21.6%, misprint 21.2%, burglar 18.5%, odd_todd 16.1%, sly 15.1%, bootstraps 12.7%, jolly 10.4%, cavendish 7.3%
held at any point, wins (n=709) vs recent losses reaching ante 4+ (n=391):
  over-represented in wins: runner 6% vs 1%; cavendish 7% vs 2%; stuntman 25% vs 8%; trousers 36% vs 13%; burglar 19% vs 8%; green_joker 57% vs 27%; mr_bones 13% vs 9%; blue_joker 38% vs 26%; banner 41% vs 31%; gros_michel 27% vs 21%
  under-represented in wins: stencil 3% vs 6%; walkie_talkie 8% vs 16%; jolly 11% vs 23%; lusty_joker 3% vs 7%; riff_raff 5% vs 13%; photograph 3% vs 8%; even_steven 9% vs 22%; smiley 3% vs 9%
pairs in winning lineups with the highest lift (>=15 games):
  clever + trousers: 26 games, lift 1.59
  cavendish + raised_fist: 17 games, lift 1.51
  jolly + stuntman: 25 games, lift 1.43
  sly + trousers: 53 games, lift 1.39
  raised_fist + stuntman: 49 games, lift 1.35
  odd_todd + trousers: 53 games, lift 1.31
  misprint + stuntman: 46 games, lift 1.29
  banner + swashbuckler: 16 games, lift 1.28
  burglar + sly: 24 games, lift 1.21
  abstract + stuntman: 58 games, lift 1.17

## 7. Boss blinds: does it play around them?
  psychic: plays with 5 cards (fewer scores nothing): 79.5% under the boss vs 70.1% on Small/Big (708 plays)
  eye: plays repeating a hand type this round (scores nothing): 19.0% under the boss vs 28.7% on Small/Big (315 plays)
  ox: plays of the run's most-played hand (money set to $0): 53.7% under the boss vs 43.8% on Small/Big (391 plays)
  mouth: later plays of a different type than the first (score nothing): 37.4% (273)
  zero-scoring plays: Small/Big 0.0%; by boss:
    psychic: 20.5% (708 plays)
    mouth: 20.0% (414 plays)
    eye: 19.0% (315 plays)
    window: 0.0% (791 plays)
    wheel: 0.0% (635 plays)
    water: 0.0% (562 plays)
    wall: 0.0% (565 plays)
    tooth: 0.0% (619 plays)
    serpent: 0.0% (448 plays)
    plant: 0.0% (659 plays)
    pillar: 0.0% (679 plays)
    ox: 0.0% (391 plays)
  club: debuffed cards per play 1.16 vs Small/Big 0.00 (708 plays)
  goad: debuffed cards per play 1.12 vs Small/Big 0.00 (749 plays)
  window: debuffed cards per play 1.15 vs Small/Big 0.00 (791 plays)
  head: debuffed cards per play 1.19 vs Small/Big 0.00 (848 plays)
  plant: debuffed cards per play 1.02 vs Small/Big 0.00 (659 plays)
  pillar: debuffed cards per play 1.75 vs Small/Big 0.00 (679 plays)

## 8. Economy
  ante 0: money leaving shop median $2, >=$25 0.0%, rerolls/shop 0.00
  ante 1: money leaving shop median $3, >=$25 0.1%, rerolls/shop 0.06
  ante 2: money leaving shop median $11, >=$25 9.8%, rerolls/shop 0.15
  ante 3: money leaving shop median $28, >=$25 58.2%, rerolls/shop 0.19
  ante 4: money leaving shop median $52, >=$25 87.5%, rerolls/shop 0.29
  ante 5: money leaving shop median $76, >=$25 95.2%, rerolls/shop 0.34
  ante 6: money leaving shop median $97, >=$25 96.0%, rerolls/shop 0.47
  ante 7: money leaving shop median $113, >=$25 89.6%, rerolls/shop 0.57
  ante 8: money leaving shop median $122, >=$25 83.8%, rerolls/shop 0.66
  skipped Small/Big blinds: 5 of 11804
  vouchers in wins: grabber 57.3%, paint_brush 35.4%, hieroglyph 25.7%, overstock_norm 12.0%, petroglyph 6.8%, clearance_sale 4.8%, telescope 4.1%, seed_money 3.9%

## 9. Free actions
  joker drags: 42.0% of all moves (130 per game)
```
