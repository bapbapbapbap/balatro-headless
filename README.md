# balatro-headless

> ## Balatro is not ours.
> **Balatro** was created by **LocalThunk** and is published by **Playstack**. The game, its
> code, art, audio, card text and name belong to them. This project isn't affiliated with or
> endorsed by either of them.
>
> **This repository contains none of the game.** No code, no art, no audio, no text. Everything
> here is our own tooling. It works only with **your own purchased copy** of Balatro, and it
> reads the game from your install on your machine. If you use it, keep the files it extracts
> private and never commit or share them. Our MIT license (LICENSE) covers our tooling only; it
> grants nothing about Balatro.

---

Runs Balatro's real game logic with no window, sound or Steam, so runs can be stepped,
replayed and searched from code. We didn't rewrite the rules. The harness loads the game's
own Lua from your install and runs it, so it behaves exactly like the game, including
Steamodded games as Lovely patches them. A validator replays runs you logged in the real
game and checks every decision matches.

**What it produces: seeds proven beatable.** A bot trained on this harness plays Balatro seeds
seeing only what a player sees. Every seed it beats is saved with its route, every move from
the seed to the win, and replayed before it counts. The list and the routes are in
[`wins/`](wins/README.md); anyone with the same game version can replay a route and check it.

**To set it up from your own Steam copy**, see [docs/REPRODUCE.md](docs/REPRODUCE.md): the full
dependency list, the mods (only for logging your own games), and `tools\balatro.ps1`, one
PowerShell controller for every step.

## Why
We're building a Balatro bot. A bot needs a simulator, and every simulator we found is a
reimplementation that only *claims* to match the real game. Even one that's 99% right
silently gives a bot wrong futures. So we asked: what if the simulator is the game itself,
and we prove it against real logged runs?

## What we did, and how

**1. Get the game's code out of your own copy.** Balatro is built on the LÖVE engine.
`Balatro.exe` is the LÖVE runtime with a zip of the game's Lua source appended to the end.
`tools/bundle_balatro.ps1` finds that zip inside your exe and copies the source out
(see [the bundle](#the-bundle-balatro_bundlezip)). That source lives only on your machine.

**2. Run the game without the engine.** `harness/love_stub.lua` stands in for LÖVE.
Graphics, audio, threads and input become no-ops, and files are read from your extracted
source. `harness/boot.lua` loads the game in its own `main.lua` order and steps the real
game loop one frame at a time. `settle()` runs frames until the game is waiting on the
player. Actions are made by calling the game's own button handlers (play, discard, buy,
open pack and so on), the same functions the UI calls.

**3. Match the exact runtime.** The game runs on LuaJIT, and the version turned out to
matter (see [findings](#what-we-found)). We read it from the install: Windows Steam Balatro
ships LÖVE 11.5 with **LuaJIT 2.0.5**. `harness/build_lupa_luajit.sh` builds the Python
bridge (lupa) against exactly that version.

**4. Run the modded game too.** Many players run Steamodded, which changes how the game
draws random numbers, so a run logged with it can only be reproduced by a sim running it
too. Lovely, the mod loader, writes a copy of every game file it patched to
`Mods/lovely/dump`. `harness/build_modded_tree.sh` lays that patched copy over the base
source and adds Steamodded. `harness/modded_boot.lua` then starts it up the same way Lovely
does: Lovely's virtual modules, Steamodded's preflight, and a plain-file stand-in for
Steamodded's native file-system library.

**5. Prove it.** [Antelytics](https://github.com/ChronoFinale/BalatroAntelytics) logs every
decision of a real run, with the game state at each one. `harness/replay.py` replays the
logged decisions in the sim. At every step it compares the RNG streams (to 13 decimal
places), deck order, hand, score, money, jokers, consumables, tags and poker-hand levels.
Getting to zero mismatches surfaced real subtleties:
- identical card copies (Cryptid, DNA) have to be paired by creation order;
- Death converts the *left* card into the *right* one, so the left-to-right order of selected cards matters;
- a booster's cards are placed by a background event, so an empty event queue doesn't mean the pack is open;
- Antelytics writes some snapshots after the action instead of before.

**6. Make it faster.** `--fast` removes work that only produces visuals: animation easing,
sprite and text movement, particles, button styling. The game hides logic inside some of
its visual code (the blind-select screen draws the Orbital Tag's hand, and a pack's cards
wait on its layout), so every cut has to leave both replays at zero mismatches. This went
from about 165 ms to about 60 ms per action.

## What we found
1. **The LuaJIT version is part of the game.** LuaJIT 2.0.5 hashes strings
   deterministically. Newer LuaJIT randomizes string hashing every time it starts, which
   changes the order `pairs()` walks a table. The game builds one list with
   `pairs(G.GAME.hands)` and then picks from it by index, in three places: the **Orbital
   Tag's hand** and **To Do List** (its starting hand and its end-of-round reroll). On a
   modern LuaJIT those picks stop following the seed. Three launches gave Pair, Four of a
   Kind and Four of a Kind; the real game gave Flush. Every other random pick sorts its
   candidates first, so nothing else depends on table order. If you run the game's Lua
   anywhere, pin LuaJIT 2.0.5 (Windows Steam; other platforms ship other runtimes).
2. **Steamodded changes the RNG.** It draws its own streams (`soul_smods_<set><ante>`,
   `small`, `big`) and rolls Soul/Black Hole chances on more card types than the base game.
   The same seed plays out differently modded and unmodded.
3. **Unseeded randomness that doesn't affect play:** a Stone Card's rank id is a fresh
   random number each time it's checked. The first-shop Buffoon pack and the Charm/Meteor
   Tag Mega packs pick between two art variants.

## The bundle (`balatro_bundle.zip`)
`tools/bundle_balatro.ps1` is a Windows PowerShell script that gathers everything the
harness needs from your machine into one zip on your Desktop:

| in the zip | what it is | where it comes from |
|---|---|---|
| `source/` | the game's Lua source | pulled out of your `Balatro.exe` |
| `mods/` | Steamodded, your other mods, Lovely's patched-source dump | `%AppData%\Balatro\Mods` |
| `logs/` | your newest Antelytics run logs | Antelytics' log folder |
| `info.txt` | game version, SHA-256 of `Balatro.exe`, Antelytics version, Lovely check | computed |

Art, sound, fonts, shaders, DLLs and executables are skipped.

**The zip contains the game's code, from your copy. It's for your own use on your own
machine. Don't upload it, share it or commit it anywhere.** This repo's `.gitignore`
excludes it and everything extracted from it.

The script is read-only: it reads `Balatro.exe`, one Steam registry entry (to find your
install) and your Balatro data folder. It writes only the zip and a temporary folder it
deletes. It changes nothing in the game, your saves or your mods, needs no admin rights
and makes no network connections. It's plain text; read it before you run it.
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bundle_balatro.ps1
```

## Status: research preview
- **Validated on 2 logged runs**, Windows Steam 1.0.1o with Lovely 0.10.0, Steamodded
  26.1002.0 and Antelytics 1.3.0-alpha, with **0 mismatches**:
  - Tutorial run, Red Deck, White Stake: all 322 decisions. Covers packs, Death, Hanged Man, Cryptid copies, DNA, Sixth Sense, Buy & Use, vouchers, rerolls and selling. The sim wins at ante 9, same as the real run.
  - Ghost Deck, Purple Stake: 10 decisions, including Steamodded's own RNG streams.
- Not yet covered: challenges, Erratic Deck, seeded runs, quit-and-continue runs, Mac/mobile.
- Runs on Linux (or WSL).
- Speed: ~60 ms per action with `--fast`. Fine for replays and search, not for millions of
  RL episodes. A fast rewrite of the rules could use this harness as its reference.
- **Playable from code** (see [Playing it from code](#playing-it-from-code)): legal actions and
  player-visible observations at every decision. Both fixtures can be driven entirely through
  it with 0 mismatches.

More logged runs and bug reports are the most useful contribution. A run that doesn't
replay cleanly is a finding.

## Versions
Everything here was built and checked against exactly these. Anything else may differ.

| Component | Version | Pin |
|---|---|---|
| Balatro | 1.0.1o-FULL, Windows Steam (app 2379780) | `Balatro.exe` SHA-256 `0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47` |
| LÖVE (the game's runtime) | 11.5 "Mysterious Mysteries" | `love.dll` FileVersion 11.5 r2 |
| LuaJIT (the game's runtime) | 2.0.5 | `lua51.dll` SHA-256 `d0039528…ae49fb2`; built here from LuaJIT git `0bf80b07` ("RELEASE LuaJIT-2.0.5") |
| lupa (Python bridge) | 2.8, rebuilt against LuaJIT 2.0.5 | `harness/build_lupa_luajit.sh` |
| Lovely | 0.10.0 | |
| Steamodded | 26.1002.0 | Lovely's dump of the patched game is version-specific |
| Antelytics (run logger) | 1.3.0-alpha | fixtures recorded with it |
| Python | 3.12.3 (WSL, training) and 3.13.18 (development checks) | |
| PyTorch | 2.15.0+cu132 (training) | not pinned in the repo yet |
| OS for the harness | Linux x86_64 (or WSL) | |
| Claude (wrote the code with chud) | Claude Opus 5.5, high effort | model ID `claude-opus-5-5`; each commit names the session |

What each check ran on:
- `replay.py`, `tests/run_replays.sh`: the modded game (Lovely + Steamodded above), both fixtures, 0 mismatches.
- `check_env.py`, `fuzz.py`, `tests/run_env_checks.sh`: the unmodded 1.0.1o game, both fixtures (every vanilla value matching; Steamodded-only RNG streams excluded) and 100+ fuzz runs per action mode. The env hasn't been checked on the modded game yet.

To keep your own install on these versions, run `tools/lock_balatro.ps1` (stops Steam
updating the game) and don't update the mods while you're collecting runs.

## Usage
```bash
# once: Python bridge built against the game's LuaJIT
bash harness/build_lupa_luajit.sh

# unzip your balatro_bundle.zip into ./local/ (gitignored), then build the modded game
bash harness/build_modded_tree.sh ./local/source ./local/mods ./local/modded

# replay a logged run
python3 harness/replay.py ./local/logs/<run>.json.gz \
    --src ./local/modded/game --mods ./local/modded/Mods --fast --quiet

# or check the included fixtures
bash tests/run_replays.sh ./local/modded
```
The fixtures in `tests/fixtures/` are our own logged runs: gameplay data only, no game code.
They were recorded on the setup above. Lovely's dump changes with Steamodded's version, so
re-record fixtures against your own setup.

## Playing it from code
`harness/env.py` exposes legal actions and observations at every decision, so an agent can play.
```python
from env import BalatroEnv
env = BalatroEnv("./local/source")            # mode="controls" (default) or "subsets"
env.reset(seed="ABC123", back="b_red", stake=1)
while not env.done:
    acts = env.legal_actions()                # [(type, i, j, cards), ...]
    env.step(acts[0])
```
**Two action modes, same game:**
- `controls` (default): the game's own controls. `toggle` a hand card, `move_hand` / `move_joker` (drag from slot i to j), `sort_rank` / `sort_suit`, then `play`, `discard`, `use_consumable` or `pick` with whatever is selected. Played cards score left to right in hand order, as in the game.
- `subsets`: the selection is part of the action (`play` with cards (1,3,5)). Fewer steps, ~200–2000 actions per hand.
- Both: `select_blind`, `skip_blind`, `end_shop`, `reroll`, `buy`, `buy_and_use`, `redeem`, `open_pack`, `pick`, `skip_pack`, `sell_joker`, `sell_consumable`, `cash_out`, `move_joker`.
- Legality is the game's own: the checks its buttons use (`can_play`, `can_discard`, `can_buy`, `can_redeem`, `can_open`, `can_reroll`, `can_skip_booster`, `can_select_card`, `Card:can_use_consumeable`, `Card:can_sell_card`, `check_for_buy_space`, Cerulean Bell's forced selection, sort buttons only when they'd change something). Actions go through the same `G.FUNCS` calls `replay.py` validates.
- Not offered yet: boss reroll (Director's Cut / Retcon), endless mode after a win.

**Observation = what a player can see.** Face-up cards; face-down cards (The Fish, House, Wheel, Mark; jokers under Amber Acorn) masked; the deck view as sets (remaining and full deck, never draw order); joker and card values shown on hover; HUD, blind and its effect, shop and pack contents, tags, vouchers, hand levels, the round's Idol / Mail-In / Ancient / Castle targets. Never RNG state.

**Checks** (`tests/run_env_checks.sh <source dir>`):
- `check_env.py --mode controls`: drives each fixture entirely through env actions (joker drags as `move_joker`, hand order as `move_hand`, selections as `toggle`); every action must be legal when taken and state must match the log. Both fixtures: 256 decisions through 902 env actions, 0 mismatches.
- `check_env.py --mode subsets`: every logged decision is in the legal set. 256/256.
- `fuzz.py --mode controls|subsets`: plays many runs; fails on any offered action that crashes, never settles or changes nothing, and on any observation that leaks seed/RNG state or a face-down card. No failures.
- Speed (unmodded, visuals stripped): ~12 ms per step in controls mode, ~33 ms in subsets mode; a runtime boots in ~0.1 s and `reset()` reuses it.

The unmodded game also replays both fixtures with every vanilla value matching (money, hand, deck order, score, jokers, consumables, tags, hand levels, vanilla RNG streams). The only differences are Steamodded's own streams, which an unmodded game doesn't have.

## License and contact
MIT, for **our tooling only**. Use it, fork it, build on it. Balatro itself remains
LocalThunk's and Playstack's.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic).
Questions and collaboration: jacobhadji05@gmail.com
