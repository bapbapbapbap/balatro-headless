# balatro-headless

[![seeds proven beatable](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge.json)](wins/README.md)
[![highest stake beaten](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge_stake.json)](wins/README.md)

**Balatro's own game code, run headless; a learned policy trained on it; and a record of every seed it has beaten,
each with a route that replays move for move to the game's own win.**

The current policy, **v6g best.pt (round 660)**, was trained on the Red Deck only. With no further training it has
won at Gold Stake on all 15 decks (the condition of the Completionist+ achievement), in the simulator, with every
win replayed on the untouched game. Its Gold win rates are low (0.01% to 0.3% per deck); see [Results](#results).

> ### Balatro is not ours.
> **Balatro** was created by **LocalThunk** and is published by **Playstack**. The game, its code, art, audio,
> card text and name belong to them. This project isn't affiliated with or endorsed by either of them.
>
> **This repository contains none of the game.** It works only with **your own purchased copy**, and reads the
> game from your install. Keep the files it extracts private. Our [MIT license](LICENSE) covers our tooling only.

<!-- tally:start -->
**Seeds proven beatable: 1,402** (24 by model 0.1, 179 by model 0.2, 232 by model 0.3, 790 by model moe2, 98 by model v6dt, 79 by model v6g; 1 on Abandoned Deck, Gold Stake, 6 on Blue Deck, Gold Stake, 1 on Ghost Deck, Gold Stake, 2 on Green Deck, Gold Stake, 30 on Red Deck, Black Stake, 18 on Red Deck, Blue Stake, 68 on Red Deck, Gold Stake, 46 on Red Deck, Green Stake, 4 on Red Deck, Purple Stake, 1,225 on Red Deck, White Stake, 1 on Yellow Deck, Gold Stake). Each one is a route in [`wins/`](wins/) that was replayed move for move to the same win before it was listed. This is the project's production count, updated each time wins are published.

**naneinf seeds proven: 0.** A seed counts once its route in [`naneinf/proven.jsonl`](naneinf/proven.jsonl) replays move for move to a played hand with non-finite chips (see `naneinf/README.md`).

**Highest stake beaten: 8 of 8 (Gold Stake).**

**Final-boss near misses: 1,601.** Seeds where the bot reached the Ante 8 boss and lost it, kept with their routes in [`wins/near_misses.jsonl`](wins/near_misses.jsonl): median 62% of the boss's target scored, 232 at 90% or more. Until 2026-11-03 these were counted as wins (see the CHANGELOG at the tag archive-2026-11-25, 0.13.2).
<!-- tally:end -->

## Results

Unmodded Balatro 1.0.1o-FULL. One game per seed; the seeds are the fixed eval list (`train/common.py`), not chosen.
t0 = temperature 0 (the policy always plays its most likely move). Every number links to its data or registration.

**Gold Stake, all 15 decks** (v6g best.pt, sha256 `4462f80e4b41fb9e85ec9f69a48dc3bf1ac35099d3c3a6f3213b8535aa909106`;
full table with intervals: [`gold/results/gold_rates.md`](gold/results/gold_rates.md)):

| Deck | t0 wins / games | Mean ante (first 600) | | Deck | t0 wins / games | Mean ante (first 600) |
|---|---|---|---|---|---|---|
| Green | 29 / 9,000 | 4.36 | | Erratic | 5 / 9,000 | 3.86 |
| Blue | 17 / 9,000 | 4.34 | | Abandoned | 3 / 9,000 | 3.80 |
| Yellow | 15 / 9,000 | 4.30 | | Checkered | 3 / 9,000 | 4.11 |
| Ghost | 10 / 9,000 | 4.07 | | Red (training deck) | 2 / 9,000 | 3.90 |
| Zodiac | 6 / 9,000 | 3.76 | | Anaglyph | 2 / 9,000 | 3.66 |
| Magic | 5 / 9,000 | 3.83 | | Black | 2 / 18,000 | 2.87 |
| Nebula | 5 / 9,000 | 3.66 | | Painted, Plasma | 1 / 9,000 each | 4.03, 3.97 |

- **Verified** ([`gold/verify_gold.py`](gold/verify_gold.py), 2026-11-25): the weights hash; every eval file's
  denominator (exactly the first N eval seeds); the frozen policy re-choosing every winning move in a fresh process;
  each route replayed on the untouched game. 114 routes, 0 mismatches, 15/15 decks.
- **Wins share seeds** ([`gold/results/seed_pool.md`](gold/results/seed_pool.md)): 15 seeds were won by two or more
  decks against 0.5 expected if decks won independently; HTBDF155 was won by 7. A rough band of ~1% of seeds
  (95% 0.6-2.2%) holds the wins, and deck strength sets the rate inside it (Green ~34%, Black ~1%).
- **White Stake, Red Deck, 600 seeds:** v6g 174 (29%); v6dt, before Gold training, 273 (45.5%). Gold training cost
  White wins ([registration and scoring](docs/hypotheses-v6.md)).
- **Not claimed:** the Steam achievement (these are simulator runs of the game's own code, not a real save), and
  priority. Other learned policies win at Gold on every deck at far higher rates; see [Related work](#related-work).

## How the policy was made

1. **Teacher data.** Balatro Lab's hand-built bot ([KvashaIhor/balatro-lab](https://github.com/KvashaIhor/balatro-lab),
   MIT) played on our harness through a bridge ([`reproduce/balatro_lab/`](reproduce/balatro_lab/README.md)); we
   reproduced its own result first (340/480 = 70.8% in its simulator). Its code stays in its repo; ours imports a pinned clone.
2. **v6e:** a ~250k-parameter transformer taught by imitation on 3,599 teacher games, from random weights.
3. **v6dt:** v6e's imitation continued; 273/600 at White, temperature 0.
4. **v6g:** PPO on our reward alone (blinds beaten), Red Deck, Gold Stake, from v6dt; 10 critic-only rounds first;
   no teacher. Stopped by hand at round 731; best.pt = round 660.

Pages: [`docs/runs/`](docs/runs/README.md). Every design and prediction was committed before its code or data
([`docs/hypotheses-v6.md`](docs/hypotheses-v6.md)). What we learned along the way:
[`docs/system_card_v6g.md`](docs/system_card_v6g.md).

## The harness

Balatro's own Lua (pinned 1.0.1o, read from your install) runs in LuaJIT 2.0.5 via lupa with LÖVE stubbed out:
no window, sound or Steam, so runs can be stepped, replayed and searched from code. Nothing is reimplemented.
It was checked decision by decision against runs logged in the real game (a 322-decision Red Deck run and a Ghost
Deck Purple Stake run: 0 mismatches). The game's LuaJIT 2.0.5 fixes table iteration order, which a few random
draws depend on (Orbital Tag, To Do List); runtimes other than 2.0.5 can diverge on those seeds, so the scripts
refuse any other LuaJIT unless told otherwise.

The bot sees what a player sees: cards face up, the deck as a set, never the draw order or the game's random state.

Setup from a Steam copy: [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Pins: Balatro 1.0.1o-FULL (`Balatro.exe`
SHA-256 `0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47`), LuaJIT 2.0.5, lupa 2.8 rebuilt
against it (`harness/build_lupa_luajit.sh`), Python 3.12, PyTorch (`requirements-lock.txt`).

## Related work

- **Balatro Lab** ([KvashaIhor/balatro-lab](https://github.com/KvashaIhor/balatro-lab), MIT): the hand-built bot
  our policy was taught by. Its strategy is the foundation of this one.
- **AnteZero** ([namecallfilter/antezero](https://github.com/namecallfilter/antezero), MIT): a PPO transformer on
  a bit-exact Rust simulator, trained on all 15 decks and 8 stakes; its v0.2.0 (released 2026-11-20) reports about
  12% at Gold over all decks (3-24% per deck). Our 15-deck Gold result came after it, at rates about 100x lower.
- **Atol8/balatro-ai** ([Attol8/balatro-ai](https://github.com/Attol8/balatro-ai)): an LLM with exact scoring
  tools; two Black Deck Gold Stake wins in the real game.
- Others surveyed and what we reproduced: [`reproduce/balatro_lab/README.md`](reproduce/balatro_lab/README.md).

## Repository map

| Folder | What |
|---|---|
| `harness/` | the game runtime (boot, legal actions, observations, replay) |
| `train/` | trainer, launcher, imitation, publishing and checks of wins |
| `configs/` | the current lineage's runs: `v6e`, `v6dt`, `v6g` |
| `reproduce/balatro_lab/` | the teacher: bridge, teacher-data runs, the tests on it |
| `gold/` | the 15-deck Gold evaluation: verification, rates, seed sharing |
| `tools/` | eval at any deck and stake (`eval_ladder.py`), route checks, Windows controller |
| `wins/` | every beaten seed with its route (the record) |
| `analysis/` | BalatroBench comparison, model-vs-model on seeds |
| `docs/` | registrations, run pages, the system card, setup |

Earlier runs (v1 to v6's other arms), their docs and released weights: [`ARCHIVE.md`](ARCHIVE.md).

## License and contact

[MIT](LICENSE), for **our tooling only**. Balatro itself remains LocalThunk's and Playstack's.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic), which wrote the code.
chud is the author and answers for any errors. Questions and collaboration: jacobhadji05@gmail.com
