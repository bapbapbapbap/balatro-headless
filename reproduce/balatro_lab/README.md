# reproduce/balatro_lab: Balatro Lab, measured on the game's own code

**Balatro Lab is not ours.** It is KvashaIhor's Balatro simulator and value bot,
https://github.com/KvashaIhor/balatro-lab, MIT license, pinned here to commit
`3e09d47ec3fac9c2b65a7f7003348cebd50949d2` (2026-11-01). None of its code is in this repository. These scripts
import it from a separate clone you point them at (`--lab <dir>`).

**What is ours** (Claude's, reviewed by chud): everything in this folder. It connects our headless harness (the
game's own Lua) to Balatro Lab's simulator and bot, to answer chud's two questions: (1) is Balatro Lab's 70% real
on the actual game, and (2) where exactly does its bot beat ours, and which of that can ours learn by itself?
Registrations and predictions: [`docs/hypotheses-reproduce.md`](../../docs/hypotheses-reproduce.md).

## Files
| File | What it does | Balatro Lab code it calls |
|---|---|---|
| `state.py` | harness state -> a Balatro Lab `Run` (deck, hand, draw pile, jokers and counters, hand levels, money, blind, boss) | `balatro.run.Run`, `balatro.cards.Card`, `balatro.jokers.CATALOG`, `balatro.items.make` / `BOSSES` |
| `validate_scoring.py` | plays seeds in the harness; before every hand asks Balatro Lab what it will score; compares with the game | `balatro.bots.estimate`, `balatro.bots.chancy`, `Run.play_hand` |
| `twin.py` | lockstep twin: same seed and actions in the harness and in Balatro Lab's seeded run, compared at every decision | `Run` (seeded), `balatro.agent.Agent`, `balatro.bots.estimate` |
| `mine.py` | our bot (a weights file) plays; the twin follows; at every decision point Balatro Lab's bot says what it would do | `Run` (seeded), `balatro.strategy.ValueBot` |
| `mine_report.py` | mine.py's output summarized by decision type | none |
| `eval_on_our_seeds.py` | Balatro Lab's bot on our 600 evaluation seeds (test T1: no seed tuning) | `Run`, `ValueBot` |
| `results/2026-11-14/` | what has run: its own 480-seed eval, scoring check, twin, first mining | |

Results so far are in the registration's "Already measured" list.

## Setup (laptop, WSL)
Balatro Lab's clone, pinned (done once; `~/balatro/balatro-lab` already exists from the 480-seed run):

    wsl bash -lc "cd ~/balatro && git clone https://github.com/KvashaIhor/balatro-lab && cd balatro-lab && git checkout 3e09d47"

This branch in its own folder (a git worktree), so the training clone's checked-out branch never changes:

    wsl bash -lc "cd ~/balatro/balatro-headless && git fetch origin && git worktree add ~/balatro/reproduce origin/reproduce-balatro-lab"

The built LuaJIT 2.0.5 runtime is not in git (`harness/vendor/`); point the worktree at the training clone's:

    wsl bash -lc "ln -s ~/balatro/balatro-headless/harness/vendor ~/balatro/reproduce/harness/vendor"

Our bot as a weights file (v4b best.pt, round 1840, 13.40 blinds/game):

    wsl bash -lc "cd ~/balatro/reproduce && source ~/balatro/venv/bin/activate && python3 train/export_weights.py /mnt/c/Users/Jacobi/balatro-work/runs/v4b/best.pt /mnt/c/Users/Jacobi/balatro-work/v4b_best_weights.pt --version v4b-best-r1840"

## Runs
T1, Balatro Lab's bot on our 600 eval seeds (~190 s a game; 8 workers about 4 h):

    wsl bash -lc "cd ~/balatro/reproduce/reproduce/balatro_lab && source ~/balatro/venv/bin/activate && python3 eval_on_our_seeds.py --lab ~/balatro/balatro-lab --games 600 --workers 8 --out /mnt/c/Users/Jacobi/balatro-work/lab_on_our_seeds.jsonl | tee /mnt/c/Users/Jacobi/balatro-work/lab_on_our_seeds.txt"

Re-running the same command resumes: seeds already in the output file are skipped.

Mining step 1 (T2's points), seeds 1-100 of set 90210 (a few minutes a seed per worker):

    wsl bash -lc "cd ~/balatro/reproduce/reproduce/balatro_lab && source ~/balatro/venv/bin/activate && python3 mine.py --weights /mnt/c/Users/Jacobi/balatro-work/v4b_best_weights.pt --lab ~/balatro/balatro-lab --src ~/balatro/local/source --seeds 100 --workers 8 --out /mnt/c/Users/Jacobi/balatro-work/mine_100.jsonl"

Summary: `python3 mine_report.py /mnt/c/Users/Jacobi/balatro-work/mine_100.jsonl`.

Notes:
- The cloud runs in `results/2026-11-14/` used pip lupa 2.8's own LuaJIT ("LuaJIT 2.0.1774896119"), which draws
  To Do List's and the Orbital Tag's hand in a different order from the game's 2.0.5 (harness README, finding 1).
  The scripts now refuse any LuaJIT but 2.0.5 unless `ALLOW_STOCK_LUAJIT=1`; the laptop's `vendor/lupa` is 2.0.5.
- Memory: each process holds a game runtime, torch and Balatro Lab; keep workers at 8 or fewer next to anything
  else running.
