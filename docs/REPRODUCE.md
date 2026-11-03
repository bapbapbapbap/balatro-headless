# Reproducing this setup, from a Steam copy of Balatro

This guide goes from owning Balatro on Steam to a running bot that produces the project's
output: **seeds proven beatable**. Each beaten seed comes with a route (every move from the
seed to the win) that anyone with the same game version can replay and check
([`wins/`](../wins/README.md)).

It covers what chud's machine actually ran. Where a version wasn't pinned, it says so.

## The short version: one controller, ten commands

Get this repository onto Windows (GitHub's **Code → Download ZIP**, or `git clone`), open
PowerShell in its folder, and run the steps in order. `tools\balatro.ps1` is the controller;
each command is explained in the sections below.

```powershell
Set-ExecutionPolicy -Scope Process Bypass   # this window only: lets the scripts run
.\tools\balatro.ps1 check     # what's there and what's missing: game hash, mods, WSL, bundle, work folder
.\tools\balatro.ps1 lock      # stop Steam updating Balatro (keeps 1.0.1o)
.\tools\balatro.ps1 bundle    # game source + mods + logs -> balatro_bundle.zip in your work folder
.\tools\balatro.ps1 setup     # WSL: packages, repo, LuaJIT 2.0.5 + lupa, numpy + torch, every accuracy check
.\tools\balatro.ps1 train -Run v1   # first run from scratch (v2 starts from v1's games); later -Run v2, -Run v2b
.\tools\balatro.ps1 status    # one-screen status of a run
.\tools\balatro.ps1 watch     # keep the PC awake; restart training only if it crashes
.\tools\balatro.ps1 stop      # stop a run cleanly (it saves latest.pt)
.\tools\balatro.ps1 verify    # replay every route in wins\routes.jsonl on the untouched game
.\tools\balatro.ps1 help
```
Options: `-Work C:\Users\<you>\balatro-work` (default), `-Run v1|v2|v2b` (default v2b),
`-Distro Ubuntu-24.04` (default), `-Workers 14`, `-Hours 168`. Before the first `setup`, install
WSL once from an admin PowerShell: `wsl --install -d Ubuntu-24.04`, then reboot and create your
Linux user.

## What you need

| | Used here | Needed |
|---|---|---|
| Game | Balatro on Steam (app 2379780), Windows | The Windows Steam build. Other platforms ship a different runtime (see [LuaJIT](#why-luajit-205)). |
| OS | Windows 11 + WSL2 | Windows to extract the game; Linux or WSL2 for the harness |
| CPU | Intel Core Ultra 9 285H, 16 cores | Any x86_64. Training speed scales with cores (one sim per worker). |
| Memory | 16 GB+ | About 0.45 GB per training worker plus about 1 GB of headroom. Training stops itself below 1 GB free. |
| GPU | NVIDIA RTX 40-series laptop GPU | Optional. Model updates run on it if present. |
| Disk | — | A few GB for saved games (every training game is kept). |

## 1. The game (Windows)

1. **Own Balatro on Steam and install it.** This project is pinned to **1.0.1o-FULL**:
   `Balatro.exe` must have SHA-256
   `0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47`. Check it in PowerShell:
   ```powershell
   Get-FileHash "C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
   ```
   If Steam has already moved you to a newer version, the routes in `wins/` won't replay
   on it. Everything else still works, but the results are for your version, not this one.
2. **Stop Steam from updating it.** Run [`tools/lock_balatro.ps1`](../tools/lock_balatro.ps1). It
   sets Balatro to update only on launch, makes the app manifest read-only, adds a Desktop
   shortcut that starts `Balatro.exe` directly, backs up the game and Mods folders once, and
   checks the exe against the hash above. `-Unlock` undoes it.
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File tools\lock_balatro.ps1
   ```

### Mods (only for logging your own games and replaying them)

The bot trains and wins on the **unmodded** game, and `wins/` is all unmodded. Install the
three mods below only if you want to record your own runs (a dataset of human play, or more
replay checks). They are other people's work; none of them is part of this repository.

| Mod | Version used | What it is | Where |
|---|---|---|---|
| Lovely | 0.10.0 | Lua injector; mods load through it | [ethangreen-dev/lovely-injector releases](https://github.com/ethangreen-dev/lovely-injector/releases) (Windows: `lovely-x86_64-pc-windows-msvc.zip`) |
| Steamodded | 26.1002.0 (its `version.lua`) | Mod framework | [Steamodded/smods](https://github.com/Steamodded/smods) |
| Antelytics | 1.3.0~alpha (by Chrono) | Logs every decision of a run to a gzipped JSON file | [ChronoFinale/BalatroAntelytics](https://github.com/ChronoFinale/BalatroAntelytics) |

1. **Lovely:** extract `version.dll` from the release zip into the game folder, next to
   `Balatro.exe`.
2. **Mods folder:** create `%AppData%\Balatro\Mods` if it doesn't exist.
3. **Steamodded:** put its folder in `Mods` so the files sit at `Mods\smods\<files>`, not
   `Mods\smods\smods\<files>`.
4. **Antelytics:** same, in its own folder: `Mods\Antelytics\<files>`.
5. **Launch the game once** (the pinned Desktop shortcut works). Lovely then writes its dump
   of the patched game to `Mods\lovely\dump`; the harness needs it to run the modded game.
6. **Don't update any of them** while you collect runs. Steamodded changes the game's random
   streams (README, "What we found"), and Lovely's dump is specific to the Steamodded version.

Runs you play on the modded game are logged to `Mods\Antelytics\log\`.

### The bundle

Run [`tools/bundle_balatro.ps1`](../tools/bundle_balatro.ps1). It puts `balatro_bundle.zip` on your
Desktop: the game's Lua source pulled out of your `Balatro.exe`, your Mods folder's code and
Lovely's dump (if present), your newest Antelytics logs, and `info.txt` with the version and
hash. It reads only and changes nothing.
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\bundle_balatro.ps1
```
**The zip holds the game's code from your copy. Keep it on your machine; never upload, share
or commit it.**

Put the zip in a work folder, for example `C:\Users\<you>\balatro-work`.

## 2. The harness (WSL)

1. **WSL2 with Ubuntu 24.04** (`wsl --install -d Ubuntu-24.04` in an admin PowerShell, then
   reboot). This setup ran Python 3.12.3 there. `balatro.ps1 setup` does steps 2 and 3 for you.
2. **System packages:**
   ```bash
   sudo apt update && sudo apt install -y build-essential git curl python3-dev python3-venv python3-pip
   ```
   `build-essential` (gcc, make) and `git`/`curl` are for building LuaJIT 2.0.5 and lupa.
3. **One-shot setup:**
   ```bash
   curl -sSLO https://raw.githubusercontent.com/bapbapbapbap/balatro-headless/main/tools/setup_linux.sh
   bash setup_linux.sh /mnt/c/Users/<you>/balatro-work
   ```
   It works in `~/balatro` on the Linux filesystem (much faster than `/mnt/c` under WSL) and:
   - clones this repo and makes a venv in `~/balatro/venv`;
   - installs `numpy` and `torch` into it;
   - builds **LuaJIT 2.0.5** (git commit `0bf80b07`) and **lupa 2.8** against it
     (`harness/build_lupa_luajit.sh`);
   - unpacks the bundle into `~/balatro/local` and builds the modded game tree;
   - runs every accuracy check: logged-run replays on the modded game (if you have logs and
     mods), the environment checks and fuzzing on the unmodded game, and the differential
     check (fast sim vs the untouched game).

   The report goes to `setup_report.txt` next to the bundle. Every check should say PASS.

4. **GPU (optional):** a current NVIDIA Windows driver exposes CUDA inside WSL2; nothing else
   is needed. `python3 -c "import torch; print(torch.cuda.is_available())"` in the venv
   should print `True`. This setup ran `torch 2.15.0+cu132`.

### Why LuaJIT 2.0.5

It is the runtime inside Windows Steam Balatro (LÖVE 11.5). Newer LuaJIT randomizes string
hashing on every launch, which changes `pairs()` order, and with it the Orbital Tag's hand and
To Do List's picks. The same seed then plays differently from the real game. Stock lupa ships
a newer LuaJIT, so it is rebuilt here.

## 3. Training the bot

```bash
cd ~/balatro/balatro-headless
bash train/run_v2.sh /mnt/c/Users/<you>/balatro-work      # first launch: clones v1's play, then RL
```
- The launcher runs pre-flight checks (a saved win replays exactly on the untouched game,
  ordered actions do what they say, fast sim matches the untouched game) and refuses to
  train if any fails.
- Results go to `balatro-work/runs/<run>/`: `train.log`, `health.json`, `metrics.jsonl`,
  checkpoints, every game in `games/`, and every win in `wins.jsonl`, each replayed before
  it is saved and flagged if it doesn't reproduce.
- Later launches resume from `latest.pt`. Ctrl+C stops cleanly within seconds.
- `tools/watch_v1.ps1 -Run v2` (or `-Run v2b`) keeps the PC awake, prints status every
  minute, and restarts training only after a crash, never after a deliberate stop.
- v2 needs `runs/v1` to exist for its first launch (it imitates v1's games). From scratch,
  start with `train/run_v1.sh`.

## 4. Checking the output

The output is **seeds proven beatable**: `wins/beaten_seeds.csv` lists them and
`wins/routes.jsonl` has the routes. To check every route on the untouched game:
```bash
cd ~/balatro/balatro-headless/train
python3 verify_games.py --src ~/balatro/local/source ../wins/routes.jsonl --wins-only --full
```
Each route is replayed move by move. The move text, the number of moves, the final state's
hash and the win must all match. A seed counts only if its route replays to a win.

The seeds were won with every joker, tag and voucher unlocked. The harness unlocks
everything itself. To replay a route by hand in the real game, use a profile with everything
unlocked, because what can appear depends on unlocks.

To add a run's wins to `wins/`, replay and publish them with `train/publish_wins.py`.

## Versions

| Component | Version | Pin |
|---|---|---|
| Balatro | 1.0.1o-FULL, Windows Steam | `Balatro.exe` SHA-256 `0D75FE16…DE93F47` |
| LÖVE (the game's runtime) | 11.5 | |
| LuaJIT | 2.0.5 | git `0bf80b07`, built by `harness/build_lupa_luajit.sh` |
| lupa | 2.8, rebuilt against LuaJIT 2.0.5 | |
| Python | 3.12.3 (WSL, training) and 3.13.18 (development checks) | |
| PyTorch | 2.15.0+cu132 | not pinned in the repo |
| numpy | not recorded | not pinned in the repo |
| Lovely / Steamodded / Antelytics | 0.10.0 / 26.1002.0 / 1.3.0~alpha | only for logging and the modded game |

To record the exact Python packages of your own run: `~/balatro/venv/bin/pip freeze > balatro-work/pip_freeze.txt`.

## Known gaps

- PyTorch and numpy versions aren't pinned; a `requirements.txt` from a working machine would fix that.
- Only the Windows Steam build is supported. Mac, Linux-native and mobile builds ship other runtimes.
- The environment hasn't been checked on the modded game; training uses the unmodded game.
