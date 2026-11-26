# balatro-headless

[![seeds proven beatable](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge.json)](wins/README.md)
[![highest stake beaten](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge_stake.json)](wins/README.md)

Balatro's own game code, run headless; a policy trained on it; and every seed it has beaten, saved as a route that
replays move for move to the game's own win.

**A policy trained on the Red Deck alone has won Gold Stake on all 15 decks.** 114 wins, each re-chosen by the
frozen policy and replayed on the untouched game, 0 mismatches ([`gold/`](gold/README.md)). The rates are low and
learned policies elsewhere do far better ([limits](#limits), [related work](#related-work)).

Not affiliated with LocalThunk or Playstack. This repository contains none of the game; it reads the game from your
own purchased copy. Our [MIT license](LICENSE) covers our tooling only.

## Replay a win

Anyone with Balatro 1.0.1o on Steam can check any win here without our weights: the game is deterministic, so a seed
and its moves are the whole run. HTBDF155 at Gold Stake, won on 7 decks:

```bash
# one-time setup from your Steam copy (Windows + WSL): docs/REPRODUCE.md
python3 -c "import json; [print(l, end='') for l in open('wins/routes.jsonl') if json.loads(l)['seed'] == 'HTBDF155']" > htbdf155.jsonl
python3 train/verify_games.py --src ~/balatro/local/source htbdf155.jsonl --full
```

`--full` runs the game with nothing stripped and checks every move, the move count, the final state and the win.

## Results

v6g best.pt (round 660), Gold Stake, temperature 0, one game per seed on the fixed eval list:

| Deck | Wins / games | Deck | Wins / games | Deck | Wins / games |
|---|---|---|---|---|---|
| Green | 29 / 9,000 | Zodiac | 6 / 9,000 | Red (trained on) | 2 / 9,000 |
| Blue | 17 / 9,000 | Magic | 5 / 9,000 | Anaglyph | 2 / 9,000 |
| Yellow | 15 / 9,000 | Nebula | 5 / 9,000 | Black | 2 / 18,000 |
| Ghost | 10 / 9,000 | Erratic | 5 / 9,000 | Painted | 1 / 9,000 |
| Abandoned | 3 / 9,000 | Checkered | 3 / 9,000 | Plasma | 1 / 9,000 |

Intervals, mean ante and ante-8 arrivals: [`gold/results/gold_rates.md`](gold/results/gold_rates.md). The wins
cluster on a band of about 1% of seeds shared across decks: [`gold/results/seed_pool.md`](gold/results/seed_pool.md).
How the policy was made (imitation of Balatro Lab's bot, then PPO at Gold) and what we learned:
[`docs/system_card_v6g.md`](docs/system_card_v6g.md).

**BalatroBench conditions** (Red Deck, White Stake, their 15 games: seeds AAAAAAA-EEEEEEE, 3 runs each, ranked by
mean final round, 24 = win), with v6dt, the White Stake model before Gold training. Run here, not submitted: their
community board takes LLM prompt strategies, not trained policies.

| | Wins | Mean final round |
|---|---|---|
| v6dt, blind skips removed | 10/15 | 22.4 ± 3.7 |
| v6dt as trained (a skipped blind isn't a round) | 7/15 | 15.8 ± 5.5 |
| BalatroBench community #1 ("Default", gpt-oss-20b, their page 2026-11-24) | – | 5.7 ± 3.4 |

Games and method: [`analysis/README.md`](analysis/README.md#balatrobench-comparison-off-board-2026-11-24).

## Limits

- 0.01-0.32% per deck at Gold. Not competitive on rate.
- Simulator runs of the game's own code, replayed; not the Steam achievement (no real save).
- Gold training cost White Stake wins on Red: 273 → 174 of 600.

## Related work

- **Balatro Lab** ([KvashaIhor/balatro-lab](https://github.com/KvashaIhor/balatro-lab)): the hand-built bot ours
  was taught by.
- **AnteZero** ([namecallfilter/antezero](https://github.com/namecallfilter/antezero)): a PPO policy on a Rust
  reimplementation. Its v0.1.0 (2026-11-12), trained on Red Deck White Stake only, already reported 1-2% at Gold on
  six decks; its v0.2.0 (2026-11-20), trained on every deck and stake, reports about 12% at Gold over all 15. Both
  came before our 15-deck result (2026-11-24).
- **BalatroBench** ([balatrobench.com](https://balatrobench.com/community.html)): a benchmark of LLM agents playing
  real Balatro through BalatroBot, ranked by mean final round; our run under its conditions is under Results.
- **Atol8/balatro-ai** ([Attol8/balatro-ai](https://github.com/Attol8/balatro-ai)): an LLM with exact scoring
  tools; two Black Deck Gold Stake wins in the real game.

## More

- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): from a Steam copy to a running harness; pinned versions.
- [`wins/`](wins/README.md): every beaten seed (all models, all stakes) and the route format.
- [`docs/hypotheses-v6.md`](docs/hypotheses-v6.md): every design and prediction, committed before its data.
- [`reproduce/balatro_lab/`](reproduce/balatro_lab/README.md): the teacher, reproduced and bridged.
- [`ARCHIVE.md`](ARCHIVE.md): earlier runs and weights, at git tags.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic), which wrote the code; chud is
the author and answers for errors. Contact: jacobhadji05@gmail.com
