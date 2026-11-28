# reproduce/antezero: AnteZero's simulator and bot, measured on the game's own code

**AnteZero is not ours.** It is namecallfilter's Rust simulator, PPO-trained bot and real-game bridge,
https://github.com/namecallfilter/antezero, MIT license, pinned here to commit
`4fe07f771104a811ca8452efd4831b0dbf097e6e` (2026-11-27); model `antezero-v0.2.0.pt` from its release v0.2.0
(sha256 `ea28506d…ffaef8`). None of its code is in this repository. These scripts import it from a separate clone
you point them at.

**What is ours** (Claude's, reviewed by chud): everything in this folder. It answers chud's question for AnteZero:
is "a bit-exact Rust reimplementation of Balatro v1.0.1o" true on the game's own code, and do its win rates hold
there? Then its bot can teach ours. Registrations and predictions: [`docs/hypotheses-reproduce.md`](../../docs/hypotheses-reproduce.md)
(the 2026-11-27 sections, Z0-Z3).

## Files
| File | What it does | AnteZero code it calls |
|---|---|---|
| `port_oracle_linux.py` | edits a pinned clone so their oracle (`sim/oracle`, which loads the game's `lua51.dll` on Windows) loads a LuaJIT 2.0.5 `.so` with `dlopen` (Z0) | their oracle and `difftest`, unchanged otherwise |
| `view.lua` | our view of the game's state, read straight from `G` (loaded into our harness) | none |
| `lockstep.py` | Z1 (openings) and Z2 (random-policy lockstep): their simulator and our harness, same seed and actions, our view compared after every action; also the action translation and comparison used by `z3.py` | `antezero_sim.Game`: `legal_codes`, `step`, `state_json`, `clone`, `chips`, `phase`, `is_terminal` |
| `z3.py` | Z3: their bot plays in their simulator with their eval loop; every game's actions recorded as text codes, then replayed in our harness move for move | `antezero_agent.runtime.load_model`, `antezero_agent.eval._envs` / `EVAL_SEED`, `antezero_agent.model.obs_from_numpy`, `Net.act`, `VecEnv`, `Game.factored_to_code` |
| `teach_az.py` | the teacher's data: its bot plays fresh seeds (one per game) in its simulator (`record`), each game is replayed in `train/sim.Sim` with v6g's settings in lockstep and written as the student sees it (`label`, `teach.py`'s format for `train/imitate.py`), checks (`report`: lockstep, labels, win rate, leak probe, sorts) | `runtime.load_model`, `runtime.make_env`, `model.obs_from_numpy`, `Net.act`, `VecEnv.set_game`, `Game.factored_to_code`, `Game.resample_hidden` |
| `results/2026-11-27/` | what has run | |

## Setup (laptop, WSL)
Their clone, pinned, with the Linux oracle port and their Python bindings (needs Rust and maturin):

    wsl bash -lc "cd ~/balatro && git clone https://github.com/namecallfilter/antezero && cd antezero && git checkout 4fe07f7 && python3 ~/balatro/balatro-headless/reproduce/antezero/port_oracle_linux.py . && cargo build --release -p oracle && source ~/balatro/venv/bin/activate && pip install maturin && cd sim/py && maturin build --release -o ~/balatro/az_wheels && pip install ~/balatro/az_wheels/antezero_sim-*.whl"

LuaJIT 2.0.5 as a shared library for their oracle (the release commit our `harness/build_lupa_luajit.sh` uses):

    wsl bash -lc "cd ~/balatro && git clone https://github.com/LuaJIT/LuaJIT.git luajit205 && cd luajit205 && git checkout 0bf80b07 && make -j8"

Their oracle reads the game's `resources/fonts/*.ttf` by name; the bundle has no `resources/`. A copy of the source
with empty font files (no game logic reads them):

    wsl bash -lc "cp -r ~/balatro/local/source ~/balatro/az_source && cd ~/balatro/az_source && mkdir -p resources/fonts && cd resources/fonts && touch GoNotoCJKCore.ttf GoNotoCurrent-Bold.ttf NotoSans-Bold.ttf NotoSansJP-Bold.ttf NotoSansKR-Bold.ttf NotoSansSC-Bold.ttf NotoSansTC-Bold.ttf m6x11plus.ttf && touch ../gamecontrollerdb.txt"

The model: download `antezero-v0.2.0.pt` from https://github.com/namecallfilter/antezero/releases/tag/v0.2.0 into
`balatro-work/antezero/` (sha256 above).

## Runs
Z0, their own difftest, 20 games per deck at stakes 1 and 8, 12 cells at a time (`reproduce/antezero/z0.sh`
writes `balatro-work/antezero/z0/<deck>_<stake>.txt`):

    wsl bash -lc "bash ~/balatro/balatro-headless/reproduce/antezero/z0.sh"

Z3 at a larger size on the laptop (record uses the GPU-free CPU path; replay uses the harness):

    wsl bash -lc "cd ~/balatro/balatro-headless/reproduce/antezero && source ~/balatro/venv/bin/activate && python3 z3.py record --az ~/balatro/antezero --ckpt /mnt/c/Users/Jacobi/balatro-work/antezero/antezero-v0.2.0.pt --stakes 1,8 --per-cell 100 --threads 4 --out /mnt/c/Users/Jacobi/balatro-work/antezero/z3_games.jsonl && python3 z3.py replay --src ~/balatro/local/source --games /mnt/c/Users/Jacobi/balatro-work/antezero/z3_games.jsonl --workers 12 --out /mnt/c/Users/Jacobi/balatro-work/antezero/z3_replay.jsonl --hist /mnt/c/Users/Jacobi/balatro-work/antezero/z3_hist && python3 z3.py report --games /mnt/c/Users/Jacobi/balatro-work/antezero/z3_games.jsonl --replay /mnt/c/Users/Jacobi/balatro-work/antezero/z3_replay.jsonl"

All three resume when re-run.

## Notes on the comparison
- Our view (`view.lua`) is built by our code on both sides; for card `ability` tables, every key the simulator
  dumps must equal the game's value (keys only the game has are not checked; a `false` flag the game leaves `nil`
  counts as equal).
- Card `sort_id` is compared after one offset fixed at the opening (the game's counter starts where the previous
  run left it in a reused runtime).
- Booster centers that differ only in art and are chosen by unseeded `math.random` in the game (first-shop Buffoon
  pack; Charm and Meteor tags' Mega packs) count as equal; each case is counted in the record (`art_n`).
- At a win the game pauses on its win overlay with the boss still in place; the simulator's terminal state is the
  one after the overlay is dismissed (AnteZero's harness dismisses it too). `z3.py` dismisses it, settles and
  compares again; both results are kept (`match_raw`, `overlay_dismissed`).
- A simulator action our env's legal list doesn't hold is applied directly and counted (`unlisted`).

## As a teacher (stage 0: the data)
Registered and checked in the cloud (docs/hypotheses-reproduce.md, 2026-11-27 sections): 150/150 games labeled in
lockstep, 99.99% of decisions labeled, leak probe 1,281/1,281. The data run on the laptop; how many games and which
decks and stakes are chud's call (this example: every deck and stake, 25 games each = 3,000 games, fresh seeds):

    wsl bash -lc "cd ~/balatro/balatro-headless/reproduce/antezero && source ~/balatro/venv/bin/activate && python3 teach_az.py record --az ~/balatro/antezero --ckpt /mnt/c/Users/Jacobi/balatro-work/antezero/antezero-v0.2.0.pt --per-cell 25 --stakes 1,2,3,4,5,6,7,8 --leak 100 --threads 8 --out /mnt/c/Users/Jacobi/balatro-work/teach_az && python3 teach_az.py label --src ~/balatro/local/source --dir /mnt/c/Users/Jacobi/balatro-work/teach_az --workers 12 && python3 teach_az.py report --dir /mnt/c/Users/Jacobi/balatro-work/teach_az | tee /mnt/c/Users/Jacobi/balatro-work/teach_az/report.txt"

Labeling costs ~16 s a game per worker (about 1 h for 3,000 games at 12 workers). Re-running resumes. The folder is
what `train/imitate.py` reads (`--imitate`). The student itself is a later registration; its size:
`docs/proposals/az-student-size.md`.
