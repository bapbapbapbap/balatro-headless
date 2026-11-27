# balatro-headless: map for agents

Read this first; open other files only when the task needs them. Public overview: `llms.txt`.
Everything earlier than the current lineage is at tags in `ARCHIVE.md`; don't open tagged history unless asked.

## What this is
Balatro's own Lua (pinned 1.0.1o, from the owner's install; none of the game is in the repo) run headless in
LuaJIT 2.0.5 via lupa with LÖVE stubbed out; a policy trained on it that sees only what a player sees; and a record
of beaten seeds whose routes replay exactly. chud's goals, in order: the strongest bot; one win per deck per stake;
100% of a fresh save in few seeds; the bot becomes his (removing his parts costs more than removing Balatro Lab's).

## Now
- **Current policy: v6g best.pt, round 660** (sha256 `4462f80e4b41fb9e...`). Trained on Red only (Balatro Lab
  teacher -> v6e -> v6dt by imitation -> v6g by PPO at Gold). Gold wins on all 15 decks zero-shot, verified
  (`gold/README.md`). Weights released (GitHub Release `v6g-r660`, with v6dt); game logs not released; no priority claim.
- **v6gd** (v6g with all 15 decks mixed) stopped 11-27: at round 660 it is below Red-only v6g on 14 of 15 decks at
  Gold (`docs/runs/v6gd.md`). Nothing training. Open issue #138 (phi_build cache gap at Gold).
- Prior art to cite wherever results are stated: Balatro Lab (the teacher), AnteZero (learned policy, all decks
  and stakes, ~12% Gold), Atol8/balatro-ai (LLM, Black Gold wins). See README "Related work".
- **Don't push to the branch a run trains from** while it trains: a watcher restart runs `git pull`.

## Where things are
| Need | Open |
|---|---|
| Launch or resume a run | `configs/<run>.toml`, `train/launch.py` (pre-flight, train, publish; checks in `CHECKS`) |
| The trainer | `train/train.py` = main loop; its docstring maps the parts (`options.py` = every setting, `rollout.py`, `ppo.py`, `credit.py`, `rewards.py`, `model.py`, `sim.py`, `imitate.py`, ...) |
| The game runtime | `harness/`: `boot.lua`, `env.lua` (legal actions, observations), `encode.lua`, `replay.py`, `diff_check.py` |
| Eval any deck and stake | `tools/eval_ladder.py` (resumes; larger `--games` extends the eval seed list); wins to routes: `tools/ladder_to_routes.py` |
| Gold results | `gold/README.md` (`verify_gold.py`, `gold_table.py`, `seed_pool.py`, `results/`) |
| The teacher | `reproduce/balatro_lab/README.md`; registrations `docs/hypotheses-reproduce.md` |
| Hypotheses / results | `docs/hypotheses.md` (index); current file `docs/hypotheses-v6.md` |
| Findings so far | `docs/system_card_v6g.md` |
| Run pages | `docs/runs/README.md` |
| Beaten seeds | `wins/README.md` (`wins/routes.jsonl` is large: query it, don't read it) |
| Checks | `tests/` (pre-flight ones named in `launch.py`'s `CHECKS`); `python3 tools/check_repo.py` must print ALL PASS |
| Windows side | `tools/balatro.ps1`, `tools/watch.ps1` |

Don't read unless asked: `wins/*.jsonl`, `analysis/results/`, `reproduce/balatro_lab/results/`.

## A new run
1. Register it first: a dated section at the end of `docs/hypotheses-<run>.md` (+ a row in `docs/hypotheses.md`),
   committed before any of its code.
2. `configs/<run>.toml`: copy the last run's, change what the registration says.
3. `docs/runs/<run>.md` + a row in `docs/runs/README.md`; the "Current" line in `llms.txt`.
4. `python3 tools/check_repo.py` ALL PASS.
5. When a run stops: tag its launch commit `run-<run>` (from chud's clone) and add its row to `ARCHIVE.md`.

## How chud runs things
- Windows laptop, training in WSL (Ubuntu-24.04). Repo `~/balatro/balatro-headless`, venv `~/balatro/venv`, game
  source `~/balatro/local/source`. Work folder `C:\Users\Jacobi\balatro-work` = `/mnt/c/Users/Jacobi/balatro-work`
  (`runs/<run>/`, `gold/`). Gold scripts take the work folder from `BALATRO_WORK`.
- Commands for him: `wsl bash -lc "..."` in **double quotes** with **no `$`** inside; for output you need to read,
  put the chain in a `.sh` in `balatro-work/gold/` that tees its own log.
- Times come from chud's message timestamps; never run a command to check the time.

## Git hygiene (chud, 2026-11-25)
- `main` is the only long-lived branch. One branch per task, named for it; open its PR in the same session.
- Before starting a branch, list open PRs and unmerged branches (`gh api repos/bapbapbapbap/balatro-headless/pulls`).
  If one of yours is open, finish or close it first, or tell chud it is still open and why.
- Stack nothing on an unmerged branch unless chud asks; never leave work only on a branch at the end of a session.
- End every session that touched the repo by telling chud which PRs await his merge. STATUS lists open PRs at the top.
- After a merge the branch is deleted (GitHub setting "Automatically delete head branches"); a run's code is kept by
  its `run-<run>` tag, not by a branch.

## Rules of the record
- A hypothesis or design change is committed **before** its code or data exists. Committed sections are never
  edited; results and corrections are new sections. A prediction made only in chat is reported as such.
- Authorship: chud's ideas are "chud's, reviewed by Claude"; Claude's are "Claude's, reviewed by chud".
- Others' code stays in their repos (imported from a pinned clone) and is attributed in every file that uses it.
- Don't move or reorder `wins/routes.jsonl`. A seed counts as beaten only if its route replays move for move to
  the game's own win.
- Cloud sessions push branches and open PRs; tags and branch deletion are done from chud's clone.
