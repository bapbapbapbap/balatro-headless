# balatro-headless: map for agents

Read this first; open other files only when the task needs them. Public overview for readers: `llms.txt`.

## What this is
Balatro's own Lua (pinned 1.0.1o, from the owner's install; none of the game is in the repo) run headless in
LuaJIT 2.0.5 via lupa with LÖVE stubbed out, a PPO bot trained on it that sees only what a player sees, and a
public record of beaten seeds whose routes replay exactly. Research aims (chud's): 1. exhibit Hume's is-ought
gap (facts fixed, only the reward varied); 2. prove a naneinf seed; 3. 100% a fresh save.

## Now
- **Training: model v4b** (run `v4b`, branch `v4b`, launched 2026-11-10 10:40 CST from commit 8a186ae,
  `train/run_v4b.sh`). Design and predictions: `docs/hypotheses-v4b.md`. Why v4 failed: last section of
  `docs/hypotheses-v4.md`. Run page: `docs/runs/v4b.md`.
- **v4/v4b's code is on branch `v4b` only** (phase split, `credit.py`, `build_phi.lua`, `tests/v4_*`,
  `run_v4b.sh`, `analysis/v4_sells.py`); it merges into main when the run ends. Read it there:
  `git fetch origin v4b && git show origin/v4b:<path>`.
- **Don't push to the branch a run trains from** (`v4b` now): a watcher restart runs `git pull` and would launch
  the new code mid-run. Work on main or a new branch.

## Where things are
| Need | Open |
|---|---|
| The trainer | `train/train.py` (PPO, workers, checks), `sim.py` (Python side of the game), `model.py`, `credit.py` (routed GAE), `rewards.py` (reward, shaping, discount clocks) |
| The game runtime | `harness/`: `boot.lua`, `env.lua` (legal actions, observations), `encode.lua`, `build_phi.lua`, `replay.py`, `diff_check.py` (fast vs untouched game) |
| A run's launcher | `train/run_<run>.sh` → `train/run_v2.sh` (shared launcher, pre-flight, publish) |
| Pre-flight checks | `tests/v4_*_check.py` (credit, phase split, phi_build, curriculum) |
| Hypotheses / results | `docs/hypotheses.md` is the index; one file per run `docs/hypotheses-<run>.md`; outcomes in `docs/closed_hypotheses.md` |
| One page per run | `docs/runs/README.md` |
| Released models | `docs/models.md`, `weights/README.md` |
| Beaten seeds | `wins/README.md` (`wins/routes.jsonl` is 7.5 MB: query it, don't read it) |
| Diagnostics | `analysis/compare_models.py` (model vs model on tagged seeds), `analysis/v4_sells.py` (per-move rewards and advantages) |

Don't read unless asked: `CHANGELOG.md` (77 KB), `wins/*.jsonl`, `weights/*.pt`, `analysis/results/`.

## How chud runs things
- Windows laptop, training in WSL (Ubuntu-24.04). Repo `~/balatro/balatro-headless`, venv `~/balatro/venv`, game
  source `~/balatro/local/source`. Work folder `C:\Users\Jacobi\balatro-work` = `/mnt/c/Users/Jacobi/balatro-work`
  (`runs/<run>/`: `metrics.jsonl`, `train.log`, `health.json`, checkpoints, saved games).
- He pastes commands into **PowerShell**: give them as `wsl bash -lc '...'` with **no `$VARIABLES`** (wsl's shell
  expands them to empty before bash sees them). For output you need to read, add `| tee /mnt/c/Users/Jacobi/balatro-work/<file>`.
- Watcher: `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Jacobi\balatro-work\watch_v1.ps1 -Run v4b`.

## Rules of the record
- A hypothesis or design change is registered (committed) **before** its code or data exists. New sections go at
  the end of the file they concern, dated with the commit's time, plus a row in `docs/hypotheses.md`. Sections
  already committed are never edited; results and corrections are new sections.
- Authorship: chud's ideas are labeled "chud's, reviewed by Claude"; Claude's "Claude's, reviewed by chud". The
  is-ought design is chud's and is credited to him.
- Don't move or reorder `wins/routes.jsonl` (the curriculum picks its starts from it at every launch).
- A seed counts as beaten only if its route replays move for move to the game's own win.
- Pushing from a cloud session: branch pushes and PRs via `gh api` work; tags and branch deletion must be done
  from chud's clone.
