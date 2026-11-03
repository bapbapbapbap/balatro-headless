# Models

Released weights are attached to the GitHub releases. Each file holds only the network and what
is needed to use it: vocabulary, architecture, action set, encoding, the reward, the repo commit
and `Balatro.exe` hash it was trained against, and its evaluation. There is no game code and no
training state. Load with `torch.load(path, weights_only=True)`.

Version names: **0.1**, **0.2** and **0.3** are the models called v1, v2 and v2b in the logs,
the CHANGELOG and `docs/hypotheses.md`.

All three are files in the release `model-0.3-beta`.

| Model | Params | Actions / encoding | Snapshot (`best.pt`) | Eval: blinds/game (wins in 200) |
|---|---|---|---|---|
| **0.3-beta** | 1.61M | ordered-capped / 2 | round 550, 61,600 games, 2026-11-02 ~21:13 CST | **12.035 (7)** |
| 0.2 | 1.61M | ordered / 2 | round 500, 56,000 games | 11.62 (6) |
| 0.1 | 249k | subsets / 1 | round 550, 55,840 games | 11.395 (1) |

**Eval** means 200 fixed seeds (Red Deck, White Stake), with moves sampled from the policy using
a fixed random stream per seed. Blinds/game is the average number of blinds beaten (24 = a win).
Each model's row is its best eval so far, which is how `best.pt` is chosen. That makes it an
optimistic number: across rounds, 0.2's evals averaged 10.66 in the F1 window.

## What 0.3-beta is
- **Network:** a transformer, d=192, 4 layers, 6 heads, over what a player sees. It sees the cards
  face up, the deck as a set (never the draw order), joker values, the HUD, shop and packs, and
  never the game's random state. It scores each legal move.
- **Reward:** blinds beaten (`blinds`), plus progress inside a blind. That progress bonus is
  potential-based, so it cancels out over a game and doesn't change which play is best. Nothing in
  the code tells it a strategy.
- **History:**
  - 0.1 was trained by RL.
  - 0.2 started by imitating 0.1's games, then trained by RL with ordered moves: it can choose
    which card scores first, which way Death copies, and boss rerolls.
  - 0.3 continues 0.2 from round 504 with two changes: a cap on chains of joker drags, and 5%
    hand-type exploration in training games.
- **Game:** Balatro 1.0.1o-FULL, unmodded, all unlocked. `Balatro.exe` SHA-256 `0D75FE16…DE93F47`.
- **Beta:** training is still running. Whether 0.3's changes raised the solved rate is under a
  pre-registered test (hypothesis H in `docs/hypotheses.md`), not yet scored.

## What it does and doesn't do (from 0.2's 709 wins, `docs/won_games_analysis.md`)
- **Does:** picks planets for the hand it plays (91% vs 25% chance). Its play leans on Two Pair,
  and it picks jokers that fit (Spare Trousers, Clever Joker). It builds money to the interest cap.
- **Doesn't:** play Full House (0 of 2,274 times it was the best hand on offer); shift play
  toward a hand it just leveled; play around The Psychic, The Mouth or The Eye (about 20% of plays
  there score nothing); spend money beyond the interest cap.

## Trained on
One laptop:
- Intel Core Ultra 9 285H (16 cores)
- NVIDIA RTX 40-series laptop GPU
- about 23 GB RAM
- Windows 11 with WSL2 Ubuntu 24.04, Python 3.12.3, PyTorch 2.15.0+cu132

It ran 14 sim workers at about 6,000–8,000 games per hour. The sim is the bottleneck: about 35 ms
per move, against about 8 ms for the model.

## Use it
Set up the harness from your own copy of the game ([REPRODUCE.md](REPRODUCE.md)), then:
```bash
cd ~/balatro/balatro-headless/train
python3 play.py balatro-bot-0.3-beta.pt --src ~/balatro/local/source --seed ABCD1234 --save route.jsonl
python3 verify_games.py --src ~/balatro/local/source route.jsonl     # replays the game it just played
```
The same weights, seed and `--sample-seed` always produce the same game.

## Files

| File | SHA-256 |
|---|---|
| `balatro-bot-0.3-beta.pt` | `e4ba9779505ce4d5e763c5344b1b620a5f3e71ab3d128c38b58d235c00eefd94` |
| `balatro-bot-0.2.pt` | `4ea94f54f16cf114f8357689ca4b6a005f1031199406ded126a6c79efa29c4bd` |
| `balatro-bot-0.1.pt` | `e649df00aaf70d04e3c11bc3edac5aac4a607f360947bd41b85f717987bd4035` |

The weights are released under the repository's MIT license. Balatro itself is LocalThunk's
and Playstack's, and you need your own copy to run anything here.
