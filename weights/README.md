# Model weights

Released as files on the [Releases page](https://github.com/bapbapbapbap/balatro-headless/releases), not in git.

| File | What | Checked |
|---|---|---|
| `balatro-bot-v6g-r660.pt` | **current best.** v6g best.pt, round 660: Red Deck, PPO at Gold Stake from v6dt. Gold wins on all 15 decks ([system card](../docs/system_card_v6g.md)) | replays the published Gold routes move for move at temperature 0 |
| `balatro-bot-v6dt.pt` | v6dt best.pt: imitation of Balatro Lab's bot. 273/600 at Red Deck White Stake, temperature 0; the BalatroBench run on the front page | replays its own eval games move for move |

Each file holds the network and what is needed to use it (vocabulary, architecture, action set, encoding, reward,
training settings without local paths, repo commit, `Balatro.exe` hash, round and eval); no optimizer state and no
game code. Checksums: `SHA256SUMS` on the release.

```bash
python3 train/play.py balatro-bot-v6g-r660.pt --src <game source> --seed 1Q6W3IJ6 --back b_green --stake 8 --temp 0
python3 tools/eval_ladder.py --src <game source> --weights balatro-bot-v6g-r660.pt --back b_green --stake 8 --temp 0 --games 600 --out green.jsonl
```

You need your own copy of Balatro 1.0.1o ([`docs/REPRODUCE.md`](../docs/REPRODUCE.md)). Earlier weights (0.1-1.0,
moe2-pins0, White Stake) are at the tag `archive-2026-11-25` ([`ARCHIVE.md`](../ARCHIVE.md#releases)).
