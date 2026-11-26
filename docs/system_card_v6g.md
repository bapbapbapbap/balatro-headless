# System card: v6g best.pt (round 660)

Written 2026-11-25. Internal: no public release of the weights yet. Every number points to where it was registered
and scored; registrations are in [`hypotheses-v6.md`](hypotheses-v6.md) (the student) and
[`hypotheses-reproduce.md`](hypotheses-reproduce.md) (the teacher). Lines marked *(chat)* were measured in working
sessions and recorded in the project's status notes, not yet as a committed section.

## The policy

| | |
|---|---|
| File | `runs/v6g/best.pt` on chud's machine, sha256 `4462f80e4b41fb9e85ec9f69a48dc3bf1ac35099d3c3a6f3213b8535aa909106` |
| Network | the v6 transformer (~250k parameters) scoring legal moves from player-visible features; no search |
| Game | Balatro 1.0.1o-FULL, unmodded, all unlocked, run headless from the game's own Lua ([README](../README.md#the-harness)) |
| Observation | cards face up, the deck as a set, the HUD; never draw order or the random state |
| Lineage | Balatro Lab's bot (teacher) → v6e (imitation from random weights, 3,599 teacher games) → v6dt (imitation continued) → v6g (PPO at Gold from v6dt, our reward alone) |
| Training decks and stakes | Red Deck only; White Stake (imitation), then Gold Stake (PPO) |
| Reward | blinds beaten, with v4b's shaping and credit settings ([`runs/v6g.md`](runs/v6g.md)) |
| Checkpoint choice | best in-run eval (temperature 1, 600 seeds): round 660, 9.705 blinds/game; run stopped by hand at round 731, flat since round 600 |

## Results

**Gold Stake, 15 decks, temperature 0** ([`gold/results/gold_rates.md`](../gold/results/gold_rates.md)): wins on
every deck; 106 wins in 144,000 games (9,000 seeds per deck, Black 18,000); per deck 0.011% (Painted, Plasma,
Black) to 0.32% (Green). Plus 7 wins at temperature 1 (Blue 5, Yellow 1, Ghost 1) and Red's training-eval win.
Only Red was trained on.

**Verification** ([`gold/verify_gold.py`](../gold/verify_gold.py), 2026-11-25): weights hash; every eval file holds
exactly the first N eval seeds (honest denominators); the frozen policy, in a fresh process, re-chooses every
winning move; every route replays on the untouched game. 114 routes, 0 mismatches, 15/15 decks.

**White Stake, Red Deck, 600 seeds, temperature 0:** 174 wins (v6dt before Gold training: 273).

## What was learned

**The simulator**
1. Balatro's random draws for Orbital Tag and To Do List follow LuaJIT 2.0.5's table order; sims on other runtimes
   silently diverge on those seeds. The harness refuses other LuaJIT builds unless told otherwise.
2. Against Balatro Lab's simulator: 100/100 opening states and 4,143/4,143 chance-free hand scores match
   ([`reproduce/balatro_lab/README.md`](../reproduce/balatro_lab/README.md)).
3. Determinism holds at scale: Red's first 600 Gold games replayed move for move (600/600) *(chat)*; 114/114 Gold
   routes replay.

**The teacher (Balatro Lab)**
4. Reproduced at 340/480 = 70.8% in its own simulator; 66.5% (1,197/1,800) through our bridge, about 68.7% after
   the Cerulean Bell bridge fix (20 of 23 lost games then won).
5. Test K: removing its hand-set value weights costs far more than removing its computed heuristics (power
   sampling, look-ahead); the weights carry most of its win rate.
6. Test R: handing it only our hand-play decisions was worth +3.3 blinds a game; our shop, pack or blind choices hurt it.
7. Where student and teacher disagree: blind choice agrees ~89%, shop ~37%, round play ~10% (first mining, v4b student).
8. Giving our network computed chip values as an input (v5) did not move its best eval in 12 hours.

**The student**
9. Imitation from random weights: 78/600 (13%), agreement on hand decisions 40.7%; on 3,599 games (v6e) 254/600 t0.
10. v6dt 273/600 (45.5%) t0. DAgger round 1 lost to its twin trained without the DAgger games (244 vs 273, paired
    z −2.23): its labels hurt. More teacher data gained about 19 wins (z 1.47).
11. A single-card failure traced to causal confusion with Ride the Bus (~9 games); most losses are in antes 5-8.

**Gold**
12. v6dt's stake ladder, Red Deck, 600 seeds: 273, 200, 46, 30, 18, 4, 0, 0 wins from White to Gold; the cliff is
    at Green Stake.
13. Gold training cost White: 273 → 174 (paired z −6.98). chud's prediction that Gold training would raise White:
    not met. Skipping explains 14% of the drop (skip test); the rest is mid-game, antes 3-6.
14. PPO on "blinds beaten" removed blind skipping (392 → 1 games with a skip): the reward pays per blind, a skip
    trades a blind for a tag.
15. Training on Red only won Gold on all 15 decks zero-shot. Red ranks 9th of 15 by mean ante; deck order tracks
    economy and hand count, not distance from Red.
16. Temperature 0 beat temperature 1 on mean ante and ante-8 arrivals on every deck tried.
17. Training-time wins: 9 of 67 came on a seed's first sight, out of 35,166 fresh games (~1 in 3,900) *(chat)*.
18. Wins share seeds ([`gold/results/seed_pool.md`](../gold/results/seed_pool.md)): 15 seeds won by 2+ decks vs 0.52
    expected if decks won independently (55 pairs vs 0.52); HTBDF155 by 7 decks. A rough band f ≈ 0.95% of seeds
    (95% 0.6-2.2%) holds the wins; inside it Green wins ~34%, Black ~1%. Claude's prediction that ≥10 of the 63
    other deck/seed cells for the first 5 shared seeds would win: **not met** (8), made in chat before those games
    ran and recorded in status notes, not committed.
19. BalatroBench conditions (v6dt, Red, White, their 15 seeds, off-board): mean final round 22.4 ± 3.7 with skips
    removed (10/15 wins), 15.8 as trained; their top LLM entry 5.7 ([`analysis/README.md`](../analysis/README.md)).

## Limits

- **Low Gold rates.** 0.01-0.32% per deck. AnteZero reports about 12% at Gold over all decks with a policy trained
  on every deck and stake, before this result (README, Related work). This policy is not competitive on rate.
- **Simulator, not a save.** The wins are the game's own code run headless and replayed; they are not the Steam
  achievement, which needs the real-save path (not built).
- **Seed band.** Wins concentrate on ~1% of seeds; what makes those seeds winnable (the seed or this policy's fit to
  it) is untested. A different trained policy on the shared seeds would separate the two.
- **White regression.** Gold training made it worse at White (273 → 174).
- **Unwritten:** predictions G1-G3 and G6 from v6g's logs (they looked met on the in-run evals; agreement 75% → 53%).

## Intended use

Research inside this project: the starting point for training at Gold across decks, and the source of seeds and
routes for the ladder and a 100% save. Not a product; no claim of priority.
