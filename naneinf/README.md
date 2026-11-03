# Standing goal: a model that reaches naneinf

Set by chud, 2026-11-02. This goal stays open until a trained model, playing only what a player
sees, reaches naneinf on the pinned game (Balatro 1.0.1o-FULL, unmodded) with a route that replays.

## What counts

**naneinf** is what the game shows when a score stops being a finite number. Scores are Lua
doubles. Past about 1.8e308 a hand's chips × mult becomes `inf`, and `inf` combined with
`inf` or 0 becomes `nan`. The game prints these as "naneinf".

What the source does with it (`functions/state_events.lua:96`, `:1139`, `blind.lua:400`, `game.lua:3196`): every "is the blind
beaten?" check is `G.GAME.chips - G.GAME.blind.chips >= 0`.
- With `inf` chips, that check is true until the blind's own target also overflows.
- With `nan` it is always false: the blind can't be beaten and the run ends when the hands run out.

So naneinf is a score to reach, not a way to win.

**Proven seeds** go in `proven.jsonl`, one route per line, only after the route has been replayed
move for move to the non-finite hand. The count at the top of the main README (`train/tally.py`)
is the number of lines in that file. It is 0.

**The measure:** a played hand after which `G.GAME.chips` is not finite (inf or nan), in a
saved game whose route replays move for move to that hand. It is reported with the seed,
deck, stake, ante, and the hand's move number.

## What the bot can't do yet

1. **Endless mode.** The env stops at the win (ante 8). Every known naneinf route plays well
   past it, roughly ante 12 to 20+. Continuing after the win has to be offered as the game
   offers it, and checked like every other action.
2. **A reward for it.** Every model so far was trained on `blinds`. Which reward to use is
   chud's decision (the ought of record), not something to slip into the trainer. Candidates:
   antes reached in endless mode; +1 for a non-finite hand; or both, reported separately.
3. **Non-finite numbers in the trainer.** Shaping uses chips / target. With `nan` chips that
   ratio is `nan`, and Python's `min(nan, 1.0)` returns `nan`. One such step would poison an
   update, which the safety checks would then reject. Scores and shaping need explicit
   inf/nan handling first.
4. **Longer games.** The move cap is 1,500. A naneinf run is long, so the cap has to rise.
   The joker-drag cap in v2b stops long drag chains, but it hasn't cut the share of moves
   spent on drags (see `docs/hypotheses.md`, the H result).
5. **Its encoding.** Score features are `slog(x)`. Their behavior at `inf`/`nan` needs a
   check (and a clamp) before the model ever sees one.

## Community seeds on this version

Community naneinf seeds were collected from published lists and checked here on 1.0.1o.
`probe.py` plays two fixed opening lines on each seed and records what a player would see:
- skip the Small and Big Blinds, take what the tags give, and take a Soul card if a pack offers one;
- play the Small Blind, then look at the first shop, each of its packs, and its rerolls.

The full record is in `community_seeds_probe.json`.

| Seed | Deck | Name (as published) | Published claim | On 1.0.1o here |
|---|---|---|---|---|
| `E4EI3UQK` | Ghost | Quick Start | Skip the first two blinds: Perkeo from an Arcana pack. Naneinf around ante 13–14. | **Matches.** Small skip: Charm Tag. Big skip: Charm Tag, whose Mega Arcana pack has The Soul, which gives **Perkeo**. |
| `5MG2J3UY` | Ghost | Triple Trouble | Skip the first blind for Perkeo. | **Matches.** Small skip: Charm Tag, The Soul, **Perkeo**. |
| `4BTVF4X1` | Ghost | Instant Combo | Instant Perkeo; Baron among the pieces. | **Matches.** Small skip: Charm Tag, The Soul, **Perkeo**. The first shop sells **Baron** and the **Blank** voucher. |
| `QJXK7HE2` | Ghost | Slow Climb | Free Perkeo and Ankh. | **Perkeo matches.** Big skip: Charm Tag, The Soul, **Perkeo**. The first shop has **Overstock**. Ankh not reached by these lines. |
| `2UULTGL9` | Plasma | Not So Legendary | No legendaries; win the first hand, then an early Blank voucher. | **Matches so far.** The first shop has the **Blank** voucher. |
| `SGRB5RPK` | Ghost | Overstocked | Arcana pack in round 1 gives Perkeo. | **Not checked.** The probe's simple player lost the first blind before the shop. |
| `26CJHF` | Ghost | Speedy Run (max-score TAS seed) | Perkeo, Cryptids, Red Seal Steel Kings, Baron, Mime. | **Not confirmed.** Neither line meets a legendary; its route starts differently. The code may also be mis-transcribed (6 characters). |
| `2K9H9HN` | Plasma (any deck) | Particle Accelerator | Early negative Perkeo and Triboulet. | **Not confirmed.** Neither line meets them. The code may be mis-transcribed (7 characters). |

"Matches" means the published opening appears here as described. It doesn't mean the rest of
the route has been checked. Only a full replay to a non-finite score would show that.

Sources (both name the same eight seeds; neither gives a game version or original author):
- [The Filibuster Blog, "Balatro: Top 8 Seeds for Achieving Infinite Scores"](https://thefilibusterblog.com/balatro-top-8-seeds-for-achieving-infinite-scores/)
- [DualShockers, "8 Best Seeds To Score Infinity in Balatro"](https://www.dualshockers.com/balatro-best-seeds-score-infinity/)

Seeds belong to nobody, but the routes and write-ups are the community's work.
Credit them when using these.

## How these seeds fit the project

They are known-good starting points for the endless-mode work, not training data from people.
The bot still only sees what a player sees and has to find its own route. A seed list is a
yardstick: did the model reach naneinf on a seed where people know it can be done? Seeds the
bot reaches naneinf on go in `wins/` with their routes, like every other proven seed.

```bash
cd naneinf && BALATRO_SRC=~/balatro/local/source python3 probe.py community_seeds.json out.json
```
