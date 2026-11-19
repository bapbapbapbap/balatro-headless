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
| `eval_on_our_seeds.py` | Balatro Lab's bot on our 600 evaluation seeds (T1, withdrawn before it ran) | `Run`, `ValueBot` |
| `rank.py` | ranking test R: v4b plays our eval seeds with one kind of decision (hand play, shop, packs, blinds, or all) made by Balatro Lab's bot | `Run` (seeded), `ValueBot` |
| `bridge.py` | model v6's bridge: Balatro Lab's state built from the game at every decision (round, shop, packs, blind choice, boss, cash out), and its bot's move on it | `Run`, `Shop`, `Offer`, `OpenPack`, `PACKS`, item registries, `ValueBot` (as the caller passes it) |
| `teach.py` | model v6's teacher data: Balatro Lab's bot plays our game through the bridge on fresh seeds; every decision saved as the student sees it, with the teacher's move. `--check` compares with the seeded twin while in sync; `--leak` is B2 | `ValueBot`, and with `--check` the seeded `Run` (twin.py) |
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

## Model v6's teacher data (docs/hypotheses-v6.md)
Balatro Lab's bot plays our game through the bridge on fresh seeds (never the 600 eval seeds); its win rate there is
the bridge's check (within 5 points of 70.8%), and the saved games are stage 1's training data. Re-running resumes.
From the training clone on branch `v6` (no run trains now):

    wsl bash -lc "cd ~/balatro/balatro-headless && git fetch && git checkout v6 && git pull && source ~/balatro/venv/bin/activate && cd reproduce/balatro_lab && python3 teach.py --lab ~/balatro/balatro-lab --src ~/balatro/local/source --games 600 --workers 12 --out /mnt/c/Users/Jacobi/balatro-work/teach_v6 | tee -a /mnt/c/Users/Jacobi/balatro-work/teach_v6.txt"

Report at any point: `python3 teach.py --report /mnt/c/Users/Jacobi/balatro-work/teach_v6`. Output: `games_w*.jsonl`
(one line per game: won, blinds, moves, unmapped, joker drags), `games/<seed>.pkl.gz` (the decisions).

## Runs
**R** (the ranking test; 300 eval seeds x 6 arms, roughly 4 h at 8 workers; re-running resumes):

    wsl bash -lc "cd ~/balatro/reproduce/reproduce/balatro_lab && source ~/balatro/venv/bin/activate && python3 rank.py --weights /mnt/c/Users/Jacobi/balatro-work/v4b_best_weights.pt --lab ~/balatro/balatro-lab --src ~/balatro/local/source --seeds 300 --workers 8 --out /mnt/c/Users/Jacobi/balatro-work/rank_R.jsonl | tee /mnt/c/Users/Jacobi/balatro-work/rank_R.txt"

Report at any point: `python3 rank.py --report /mnt/c/Users/Jacobi/balatro-work/rank_R.jsonl`.

T1 (withdrawn), Balatro Lab's bot on our 600 eval seeds (~190 s a game; 8 workers about 4 h):

    wsl bash -lc "cd ~/balatro/reproduce/reproduce/balatro_lab && source ~/balatro/venv/bin/activate && python3 eval_on_our_seeds.py --lab ~/balatro/balatro-lab --games 600 --workers 8 --out /mnt/c/Users/Jacobi/balatro-work/lab_on_our_seeds.jsonl | tee /mnt/c/Users/Jacobi/balatro-work/lab_on_our_seeds.txt"

Re-running the same command resumes: seeds already in the output file are skipped.

Both measurements below, one after the other, with a paired summary at the end (`gap_tests.sh`, `gap_summary.py`):

    wsl bash -lc "cd ~/balatro/reproduce && git fetch origin && git checkout gap-check && git pull && bash reproduce/balatro_lab/gap_tests.sh"

F1c (docs/hypotheses-v6.md, 2026-11-19), the 244 teacher games with unmapped moves (Cerulean Bell) rerun with
fixes F1 and F1b, into their own folder (not training data):

    wsl bash -lc "cd ~/balatro/reproduce && git fetch && git checkout gap-check && git pull && source ~/balatro/venv/bin/activate && cd reproduce/balatro_lab && python3 teach.py --lab ~/balatro/balatro-lab --src ~/balatro/local/source --seed-file results/2026-11-19/bell_seeds.txt --games 244 --workers 12 --out /mnt/c/Users/Jacobi/balatro-work/bell_f1c | tee -a /mnt/c/Users/Jacobi/balatro-work/bell_f1c.txt"

Test P (docs/hypotheses-v6.md, 2026-11-19), Balatro Lab's bot in its own simulator on model v6's teacher seeds 0-599,
the seeds the bridge already played (paired, seed by seed):

    wsl bash -lc "cd ~/balatro/reproduce && git fetch && git checkout gap-check && git pull && source ~/balatro/venv/bin/activate && cd reproduce/balatro_lab && python3 eval_on_our_seeds.py --lab ~/balatro/balatro-lab --seed-set teach --games 600 --workers 12 --out /mnt/c/Users/Jacobi/balatro-work/lab_on_teach600.jsonl | tee -a /mnt/c/Users/Jacobi/balatro-work/lab_on_teach600.txt"

Mining step 1 (T2's points), seeds 1-100 of set 90210 (a few minutes a seed per worker):

    wsl bash -lc "cd ~/balatro/reproduce/reproduce/balatro_lab && source ~/balatro/venv/bin/activate && python3 mine.py --weights /mnt/c/Users/Jacobi/balatro-work/v4b_best_weights.pt --lab ~/balatro/balatro-lab --src ~/balatro/local/source --seeds 100 --workers 8 --out /mnt/c/Users/Jacobi/balatro-work/mine_100.jsonl"

Summary: `python3 mine_report.py /mnt/c/Users/Jacobi/balatro-work/mine_100.jsonl`.

Notes:
- The cloud runs in `results/2026-11-14/` used pip lupa 2.8's own LuaJIT ("LuaJIT 2.0.1774896119"), which draws
  To Do List's and the Orbital Tag's hand in a different order from the game's 2.0.5 (harness README, finding 1).
  The scripts now refuse any LuaJIT but 2.0.5 unless `ALLOW_STOCK_LUAJIT=1`; the laptop's `vendor/lupa` is 2.0.5.
- Memory: each process holds a game runtime, torch and Balatro Lab; keep workers at 8 or fewer next to anything
  else running.
