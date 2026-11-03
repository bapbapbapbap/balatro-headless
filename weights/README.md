# Model weights

**Beta.** Trained weights of the Balatro bot, snapshot of 2026-11-02 at about 21:13 CST. Training is still running.

| File | Model | Eval: blinds per game (wins in 200 fixed seeds) |
|---|---|---|
| `balatro-bot-0.3-beta.pt` | 0.3-beta: 1.6M parameters, round 550 | **12.035 (7)** |
| `balatro-bot-0.2.pt` | 0.2 (earlier run) | 11.62 (6) |
| `balatro-bot-0.1.pt` | 0.1 (first run) | 11.395 (1) |

All three were trained on Red Deck, White Stake, with the reward "blinds beaten", on unmodded Balatro 1.0.1o-FULL. The bot sees only what a player sees.

It was trained on one laptop: Core Ultra 9 285H (16 cores), RTX 40-series laptop GPU, about 23 GB RAM, under WSL2.

The model card covers what it has learned, what it hasn't (it never plays Full House, for one), file hashes, and how to watch it play a seed: [docs/models.md](https://github.com/bapbapbapbap/balatro-headless/blob/main/docs/models.md).

You need your own copy of Balatro to run it. The files contain no game code.

To check a download: `sha256sum weights/*.pt` and compare with the table in docs/models.md.
