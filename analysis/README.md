# Analysis

`compare_models.py`: two checkpoints on the same seeds, paired seed by seed (see its docstring).
Earlier analyses (won-game events, re-solve of 1.0, paired experts of moe2, v4/v4b diagnostics) are at the tag
`archive-2026-11-25` (`ARCHIVE.md`).

## BalatroBench comparison (off-board, 2026-11-24)
[BalatroBench](https://balatrobench.com/community.html)'s community board ranks LLM strategies (prompt templates
run by gpt-oss-20b) on Red Deck, White Stake, seeds AAAAAAA..EEEEEEE, 3 runs each, temperature 1.0, by mean final
round (24 = win). A trained policy isn't a valid entry there, so this is the same test run here, not a submission.
`balatrobench.py` plays a checkpoint through `tools/eval_ladder.py`'s own code on those 15 games; run 1 per seed is
the normal eval sampling, runs 2-3 rename that stream so they differ (the game's RNG is untouched). Final round =
blinds beaten for a win, +1 for a loss.
```bash
python3 analysis/balatrobench.py --weights /mnt/c/Users/<you>/balatro-work/runs/v6dt/best.pt --out v6dt_t1.jsonl [--no-skip 1]
```
| v6dt `best.pt` (sha256 2dbe6314…), t1 | Wins | Mean final round (sample SD) |
|---|---|---|
| as trained (skips blinds; a skipped blind isn't a round, so wins read 15-23) | 7/15 | 15.8 ± 5.5 |
| `--no-skip 1` | 10/15 | **22.4 ± 3.7** |
| BalatroBench community #1 ("Default", gpt-oss-20b), from their page on 2026-11-24 | – | 5.7 ± 3.4 |

Not registered in advance: a measurement, not a test. 7 vs 10 of 15 is within noise. Their metric scores an
ante-8 boss loss as 24, same as a win (2 such losses in the no-skip run). Games: `results/balatrobench/`.
