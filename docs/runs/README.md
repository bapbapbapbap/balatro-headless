# Training runs: the current lineage

One page per run on main: what it was, what was registered and scored for it, and what it left behind. The record
itself stays where it is (`docs/hypotheses-v6.md`, `wins/`), unedited. Every earlier run's page is at its tag in
[`ARCHIVE.md`](../../ARCHIVE.md).

| Run | Dates (CST) | What it was | Result |
|---|---|---|---|
| [v6e](v6e.md) | 2026-11-21 | imitation of Balatro Lab's bot from random weights, all 3,599 teacher games, up to 16 passes | 254/600 at White, t0; agreement 74.1% |
| [v6dt](v6dt.md) | 2026-11-22 | v6e's imitation continued (DAgger round 1's twin, without the DAgger games) | 273/600 at White, t0 (best student); agreement 75.4% |
| [v6g](v6g.md) | 2026-11-22 → 11-23 | PPO on our reward alone at Gold Stake, Red Deck, from v6dt; 10 critic-only rounds first; no teacher | best.pt = round 660; Gold wins on all 15 decks zero-shot ([`gold/`](../../gold/README.md)); 174/600 at White |
| [v6gd](v6gd.md) | 2026-11-25 → | v6g with one change: every game's deck drawn uniformly from all 15 | registered; scored against v6g at matched rounds, 15 decks × 600 seeds |

All four: unmodded 1.0.1o. Red Deck for training, except v6gd (all 15). Run folders on chud's machine: `balatro-work/runs/<run>/`.

## Getting one run's routes

`wins/routes.jsonl` stays one file (the trainer, the checks and the tally read it). To take out one run's routes:

```
python3 -c "import json,sys; [print(l,end='') for l in open('wins/routes.jsonl') if json.loads(l)['source'].split('/')[0]==sys.argv[1]]" v6g > v6g_routes.jsonl
```
