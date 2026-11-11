# Archive

Main holds what the next run uses. Everything a past run used is kept, unchanged, in the commit it ran from; this
file names those commits. Nothing here needs reading to work on the current run.

**Fetch one file:** `git show <tag>:<path>` (e.g. `git show run-v2b:train/run_v2b.sh`).
**Re-run an old run exactly:** `git worktree add /tmp/old <tag>`, then launch from there as that run's page says
(each checkpoint also records its commit, the game exe's hash and its full settings in `provenance.json`).

## Runs (tag = the commit the run launched from, from its `runs/<run>/provenance.json`)
| Tag | Commit | Run | Dates (CST) | Launcher there | Result | Page |
|---|---|---|---|---|---|---|
| `run-v1` | fac476b | v1, model 0.1 | 2026-11-01 → 11-02 | `train/run_v1.sh` | first PPO bot; 24 published wins | [docs/runs/v1.md](docs/runs/v1.md) |
| `run-v2` | 95480b9 | v2, model 0.2 | 2026-11-02 | `train/run_v2.sh` (cloning v1, then RL) | 179 published wins | [v2](docs/runs/v2.md) |
| `run-v2b` | 34bb1f4 | v2b, models 0.3 and 1.0 | 2026-11-02 → 11-03 | `train/run_v2b.sh`, `tools/start_v2b.sh` | 232 published wins; 1.0 released | [v2b](docs/runs/v2b.md) |
| `run-moe` | 17c1602 | moe | 2026-11-03 → 11-04 | `train/run_moe.sh` (+ `run_expert(s).sh`, `run_early.sh`, `run_earlies.sh`) | stopped on the stall rule | [moe](docs/runs/moe.md) |
| `run-moe2` | 3d5b12f | moe2, moe2-pins0 | 2026-11-04 → 11-05 | `train/run_moe.sh` | 790 published wins; pins0 released | [moe2](docs/runs/moe2.md) |
| `run-v4` | fd02b0d | v4 | 2026-11-09 → 11-10 | `train/run_v4.sh` | best eval 4.68; learned to sell its jokers | [v4](docs/runs/v4.md) |
| `run-v4b` | 8a186ae | v4b | 2026-11-10 → | `train/run_v4b.sh` (same settings: `configs/v4b.toml`) | stopped at round 280 on the sell stop (best eval 8.31); continues on main with the stop off | [v4b](docs/runs/v4b.md) |

v3 never trained: its code (`train/run_v3.sh`, the blind clock it introduced is current) is at `pre-launcher`.

## Code retired from main
| Tag | Commit | What | Why it left main |
|---|---|---|---|
| `pre-launcher` | 26061d7 | main just before the launcher: every per-run script (`train/run_*.sh`, `tools/start_v2b.sh`), `tools/watch_v1.ps1`, `tools/recount_wins.py` | per-run launchers replaced by `train/launch.sh <run>` + `configs/<run>.toml`; `recount_wins.py` was the one-off fix of 2026-11-03 (final-boss losses counted as wins; `docs/hypotheses-general.md`) |
| `archive-hand-focus-bots` | 4a401f6 | hand-focus bots: one per poker hand (branch `archive/hand-focus-bots`) | never merged; became the hand experts of moe |
| `archive-known-winnable-seeds` | 23f1105 | training on known-winnable seeds, a "known" arm (branch `archive/known-winnable-seeds`) | never merged |

Capabilities stay on main even when no current run uses them (mixture of experts, hand pins, self-imitation,
behavior cloning, hand-type exploration, seed retries, the curriculum, both discount clocks): see `train/train.py`'s
docstring for where each lives. What left main is what only reproduced one run's launch.

## Releases
`0.03` (7552ae6): the 0.3-beta weights commit. Released weights themselves stay on main in `weights/`.
