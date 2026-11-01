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

More logged runs and bug reports are the most useful contribution. A run that doesn't
replay cleanly is a finding.

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

## License and contact
MIT, for **our tooling only**. Use it, fork it, build on it. Balatro itself remains
LocalThunk's and Playstack's.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic).
Questions and collaboration: jacobhadji05@gmail.com
