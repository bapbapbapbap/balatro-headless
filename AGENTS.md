# balatro-headless: map for agents

Read this first; open other files only when the task needs them. Public overview for readers: `llms.txt`.
Past runs' code: `ARCHIVE.md` (tags); don't open tagged history unless asked.

## What this is
Balatro's own Lua (pinned 1.0.1o, from the owner's install; none of the game is in the repo) run headless in
LuaJIT 2.0.5 via lupa with LÖVE stubbed out, a PPO bot trained on it that sees only what a player sees, and a
public record of beaten seeds whose routes replay exactly. Research aims (chud's): 1. exhibit Hume's is-ought
gap (facts fixed, only the reward varied); 2. prove a naneinf seed; 3. 100% a fresh save.

## Now
- **Current run: model v4b** (`configs/v4b.toml`; design and predictions `docs/hypotheses-v4b.md`; why v4 failed:
  last section of `docs/hypotheses-v4.md`; run page `docs/runs/v4b.md`, which says whether it is training).
- **Don't push to the branch a run trains from** while it trains: a watcher restart runs `git pull` and would
  launch the new code mid-run. Work on another branch and merge after.

## Where things are
| Need | Open |
|---|---|
| Launch or resume a run | `configs/<run>.toml` (its settings), `train/launch.py` (pre-flight, train, publish; checks by name in `CHECKS`) |
| The trainer | `train/train.py` = main loop only; its docstring maps the parts: `options.py` (every setting), `rollout.py` (workers), `ppo.py` (GAE, PPO), `credit.py` (routed credit), `rewards.py` (reward, shaping, clocks), `model.py`, `sim.py`, `curriculum.py`, `pins.py`, `moe.py`, `explore.py`, `pool.py`, `common.py` |
| The game runtime | `harness/`: `boot.lua`, `env.lua` (legal actions, observations), `encode.lua`, `build_phi.lua`, `replay.py`, `diff_check.py` (fast vs untouched game) |
| Checks | `tests/` (pre-flight ones are named in `launch.py`'s `CHECKS`); `tests/compare_trainers.py` proves a refactor changed nothing |
| Hypotheses / results | `docs/hypotheses.md` is the index; one file per run `docs/hypotheses-<run>.md`; outcomes in `docs/closed_hypotheses.md` |
| One page per run | `docs/runs/README.md` |
| Released models | `docs/models.md`, `weights/README.md` |
| Beaten seeds | `wins/README.md` (`wins/routes.jsonl` is 7.5 MB: query it, don't read it) |
| Diagnostics | `analysis/compare_models.py` (model vs model on tagged seeds), `analysis/v4_sells.py` (per-move rewards and advantages) |
| Windows side | `tools/balatro.ps1` (setup, train, status, watch, publish), `tools/watch.ps1` |

Don't read unless asked: `CHANGELOG.md` (77 KB), `wins/*.jsonl`, `weights/*.pt`, `analysis/results/`.

## A new run
1. Register it first: a dated section in a new `docs/hypotheses-<run>.md` + a row in `docs/hypotheses.md`,
   committed before any of its code.
2. `configs/<run>.toml`: copy the last run's, change what the registration says. Settings are `train/options.py`'s.
3. `docs/runs/<run>.md` + a row in `docs/runs/README.md`; the "Current model" line in `llms.txt`.
4. `python3 tools/check_repo.py` must print ALL PASS (every launch runs it in pre-flight too).
5. When a run stops: tag its launch commit `run-<run>` (from chud's clone) and add its row to `ARCHIVE.md`.

## How chud runs things
- Windows laptop, training in WSL (Ubuntu-24.04). Repo `~/balatro/balatro-headless`, venv `~/balatro/venv`, game
  source `~/balatro/local/source`. Work folder `C:\Users\Jacobi\balatro-work` = `/mnt/c/Users/Jacobi/balatro-work`
  (`runs/<run>/`: `metrics.jsonl`, `train.log`, `health.json`, `provenance.json`, checkpoints, saved games).
- Commands for him: `wsl bash -lc "..."` in **double quotes** (works in PowerShell and Command Prompt) with **no
  `$`** inside. For output you need to read, add `| tee /mnt/c/Users/Jacobi/balatro-work/<file>`.
- Launch: `wsl bash -lc "cd ~/balatro/balatro-headless && bash train/launch.sh <run> --work /mnt/c/Users/Jacobi/balatro-work"`
  (`--smoke` first for a new run). Watch: `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Jacobi\balatro-work\watch.ps1 -Run <run>`.

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
