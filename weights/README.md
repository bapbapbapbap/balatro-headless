# Model weights

Trained weights of the Balatro bot. **1.0** is the final weights of the 0.3 run, which ended on 2026-11-03, and the end of the training line that won the 435 published seeds (1,549 were published before 2026-11-03; 1,114 of them were losses on the Ante 8 boss, see the CHANGELOG, 0.13.2). 0.3-beta is the same run's snapshot from 2026-11-02, about 21:13 CST.

**1.0's current performance: being re-measured.** The 2.5% (5 of 200 fixed evaluation seeds) and 12.79 blinds per game were counted with a win test that also counted losses on the Ante 8 boss.

| File | Model | Eval: blinds per game (wins in 200 fixed seeds) |
|---|---|---|
| `balatro-bot-1.0.pt` | 1.0: 1.6M parameters, round 810 of the 0.3 run | **12.79** (5) |
| `balatro-bot-0.3-beta.pt` | 0.3-beta: same run, round 550 | 12.035 (7) |
| `balatro-bot-0.2.pt` | 0.2 (earlier run) | 11.62 (6) |
| `balatro-bot-0.1.pt` | 0.1 (first run) | 11.395 (1) |

**`balatro-bot-moe2-pins0.pt`**: a mixture of experts (an early-game head, six hand experts and a routing rule; 2.3M parameters), from the second MoE run at its first round with every pin at 0 (round 1042, 100,032 games). Eval: 12.94 blinds per game, 9 wins and 13 near wins in **600** fixed seeds, scored with the game's own outcomes, so it doesn't line up with the 200-seed numbers above. A research snapshot: it hasn't learned Full House and hasn't been compared with 1.0. [System card](../docs/system_card_moe2.md).

All five were trained on Red Deck, White Stake, with the reward "blinds beaten", on unmodded Balatro 1.0.1o-FULL. The bot sees only what a player sees.

It was trained on one laptop: Core Ultra 9 285H (16 cores), RTX 40-series laptop GPU, about 23 GB RAM, under WSL2.

The model card (and, for the mixture of experts, the system card) covers what it has learned, what it hasn't (it never plays Full House, for one), file hashes, and how to watch it play a seed: [docs/models.md](../docs/models.md).

You need your own copy of Balatro to run it. The files contain no game code.

To check a download: `sha256sum weights/*.pt` and compare with the table in [docs/models.md](../docs/models.md#files).
