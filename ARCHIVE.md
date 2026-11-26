# Archive

Main holds the current lineage only: the harness, the trainer, the teacher bridge, runs v6e -> v6dt -> v6g, the
Gold evaluation, and the record of beaten seeds. Everything else is kept, unchanged, at the commits below.

**Fetch one file:** `git show <tag>:<path>` (e.g. `git show archive-2026-11-25:docs/system_card_moe2.md`).
**Re-run an old run exactly:** `git worktree add /tmp/old <tag>`, then launch as that run's page (at the tag) says.
Each checkpoint also records its commit, the game exe's hash and its full settings in `provenance.json`.

## The cleanup of 2026-11-25
`archive-2026-11-25` is main with every open branch merged (v6-win-attempts, PR #130's Gold wins, PR #131's
BalatroBench comparison), just before the cleanup removed what the current lineage doesn't use: per-run docs and
registrations of v1-v5 and moe/moe2 (`docs/hypotheses-*.md`, `docs/runs/*.md`, `docs/closed_hypotheses.md`, the
moe2 system card, model and evaluation pages), the configs of v4b, v5, v6, v6c450, v6c900, v6d, v6eb, v6m2, the
v4/v4b analysis scripts, the stage scripts (`train/step1-3.sh`, `dagger_round.sh`), the Immolate seed check
(`tools/immolate/`), released weights 0.1-1.0 and moe2-pins0, and the
CHANGELOG to 2026-11-25. The pre-2026-11-09 hypotheses file in one piece is at commit e3778d0.

## Runs (tag = the commit the run launched from, from its `runs/<run>/provenance.json`)
| Tag | Commit | Run | Dates (CST) | Result | Page (at the tag) |
|---|---|---|---|---|---|
| `run-v1` | fac476b | v1, model 0.1 | 2026-11-01 → 11-02 | first PPO bot; 24 published wins | `docs/runs/v1.md` |
| `run-v2` | 95480b9 | v2, model 0.2 | 2026-11-02 | 179 published wins | `docs/runs/v2.md` |
| `run-v2b` | 34bb1f4 | v2b, models 0.3 and 1.0 | 2026-11-02 → 11-03 | 232 published wins; 1.0 released | `docs/runs/v2b.md` |
| `run-moe` | 17c1602 | moe | 2026-11-03 → 11-04 | stopped on the stall rule | `docs/runs/moe.md` |
| `run-moe2` | 3d5b12f | moe2, moe2-pins0 | 2026-11-04 → 11-05 | 790 published wins; pins0 released | `docs/runs/moe2.md` |
| `run-v4` | fd02b0d | v4 | 2026-11-09 → 11-10 | best eval 4.68; learned to sell its jokers | `docs/runs/v4.md` |
| `run-v4b` | 8a186ae | v4b | 2026-11-10 → | best eval 8.31 (round 240); continued on 809e40c with the sell stop off | `docs/runs/v4b.md` |
| `run-v4c` | e16a63d | v4c | 2026-11-14 → | swap move, offer values for swaps and packs; best 12/600 wins | `docs/runs/v4c.md` (branch `v4c`) |
| `run-v5` | e1504ad | v5 | 2026-11-15 → 11-16 | stopped by chud at round 278; best eval stayed round 0's | `docs/runs/v5.md` |
| `run-v6` | 8a11d21 | v6 stage 1 | 2026-11-19 → 11-20 | 78/600; agreement 68.0% | `docs/runs/v6.md` |
| `run-v6c` | fee0e8b | v6c450, v6c900, v6m2 | 2026-11-20 | step 1 scaling and continuation arms | `docs/runs/v6c450.md` |
| `run-v6e` | ed75c1d | v6e, v6eb | 2026-11-21 | v6e 254/600 t0 (current lineage); v6eb 245 | `docs/runs/v6e.md` (on main) |
| `run-v6d` | e0ff1f5 | v6d round 1, v6dt | 2026-11-22 | DAgger 244 t0; its twin v6dt 273 (current lineage) | `docs/runs/v6dt.md` (on main) |
| `run-v6g` | 74eeab7 | v6g | 2026-11-22 → 11-23 | best.pt r660; Gold on 15/15 decks zero-shot | `docs/runs/v6g.md` (on main) |

v3 never trained: its code is at `pre-launcher`.

## Code retired from main earlier
| Tag | Commit | What |
|---|---|---|
| `pre-launcher` | 26061d7 | every per-run script before the launcher (`train/run_*.sh`, `tools/start_v2b.sh`), `tools/watch_v1.ps1`, `tools/recount_wins.py` |
| `archive-hand-focus-bots` | 4a401f6 | one bot per poker hand; became the hand experts of moe |
| `archive-known-winnable-seeds` | 23f1105 | training on known-winnable seeds |

## Releases
`0.03` (7552ae6): the 0.3-beta weights commit. Weights 0.1, 0.2, 0.3-beta, 1.0 and moe2-pins0 are in `weights/` at
`archive-2026-11-25` (`git show archive-2026-11-25:weights/balatro-bot-1.0.pt > balatro-bot-1.0.pt`).

## Branches
`tools/archive_branches.sh` (run from chud's clone, once this cleanup is merged) creates the tags above, tags any
branch whose head is not already inside `archive-2026-11-25` as `archive-branch-<name>`, then deletes every remote
branch but `main`. Nothing is lost: each deleted branch's head is reachable from a tag listed by the script.
