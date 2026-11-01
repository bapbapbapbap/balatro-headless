# balatro-headless

Runs **Balatro's own game code** with no window, sound or Steam, so runs can be stepped,
replayed and searched from code. Nothing is reimplemented: the sim *is* the game, loaded
from your own install. It also runs Steamodded games exactly as Lovely patches them.

Because it's the real code, it can be checked against the real game. `replay.py` takes a
run logged by [Antelytics](https://github.com/ChronoFinale/BalatroAntelytics) on Steam and
replays every decision in the sim. At every step it compares RNG streams, deck order,
hand, score, money, jokers, consumables, tags and hand levels against the log.

> Not affiliated with or endorsed by LocalThunk or Playstack. Requires your own copy of
> Balatro. This repo contains no game code or assets; the harness reads them from your
> install at runtime. Don't commit extracted game files to forks.

## Status: research preview
- **Validated on 2 logged runs, Windows Steam 1.0.1o + Steamodded 26.1002.0, 0 mismatches.**
  - Tutorial run, Red Deck, White Stake: all 322 decisions. Covers packs, Death, Hanged Man, Cryptid copies, DNA, Sixth Sense, Buy & Use, vouchers, rerolls and selling. The sim wins at ante 9, same as the real run.
  - Ghost Deck, Purple Stake: 10 decisions, including Steamodded's own RNG streams.
- Not yet covered: challenges, Erratic Deck, seeded runs, quit-and-continue runs, other platforms.
- Linux (or WSL). The runtime is a lupa build linked against LuaJIT 2.0.5.
- Speed: ~60 ms per action with `--fast`. That's fine for replays and search, not for millions of RL episodes. A fast rewrite of the rules could use this harness as its reference.

More runs and bug reports are the most useful contribution: a log that doesn't replay
cleanly is a finding.

## What it found
1. **The LuaJIT version is part of the game.** Windows Steam Balatro runs LÖVE 11.5 with **LuaJIT 2.0.5** (`lua51.dll`, SHA-256 `d0039528…ae49fb2`). Its string hashing is deterministic. Newer LuaJIT randomizes string hashing per process, so `pairs()` order changes on every launch. The game builds one list with `pairs(G.GAME.hands)` and then picks from it by index, in three places: the **Orbital Tag's hand** and **To Do List** (its starting hand and its end-of-round reroll). On a modern LuaJIT those picks stop following the seed. In testing, three launches gave Pair, Four of a Kind and Four of a Kind; the real game gave Flush. Every other random pick sorts its candidates first (pools by `order`, the starting deck by suit+rank, `pseudorandom_element` by key or `sort_id`, shuffles by `sort_id`), so nothing else depends on table order. Other platforms ship other runtimes and may differ.
2. **Steamodded changes the RNG.** It draws its own streams (`soul_smods_<set><ante>`, `small`, `big`), and it rolls Soul/Black Hole chances on more card types than the base game does. The same seed plays out differently modded and unmodded.
3. **Unseeded randomness that doesn't affect play:** a Stone Card's rank id is a fresh random number each time it's checked. The first-shop Buffoon pack and the Charm/Meteor Tag Mega packs pick between two art variants.
4. **Antelytics snapshot timing:** some nodes are written after their action or partway through it (purchases, uses, sells, rerolls, end of round, the tag nodes after a skip). `replay.py` doesn't compare those. The next stable node checks the full state.

## How it works
- `harness/love_stub.lua`: stand-in for the LÖVE runtime. Graphics, audio, threads and input are no-ops.
- `harness/boot.lua`: loads the game in `main.lua` order. Provides `start_run(back, stake, seed)` and `settle()`, which runs frames until the game is waiting on the player (one event-queue pass per frame, as in the real game). `strip_visuals()` (`--fast`) removes visual-only work; each cut is verified by the replays.
- `harness/modded_boot.lua`: boots the way Lovely + Steamodded do: module patches, the preflight core, the `lovely` module, and a plain-io stand-in for Steamodded's nativefs.
- `harness/replay.py`: the replay validator.
- `harness/build_lupa_luajit.sh`: builds `vendor/lupa` against LuaJIT 2.0.5.
- `harness/build_modded_tree.sh`: assembles the modded game from the base source plus Lovely's dump of the patched files and Steamodded.
- `tools/bundle_balatro.ps1`: Windows script that packages the game source (pulled out of `Balatro.exe`), your Mods folder's code and your newest run logs into one zip. Art, sound and fonts are skipped.

## Usage
```bash
# once: runtime matching the game
bash harness/build_lupa_luajit.sh

# game source: extract from your Balatro.exe (tools/bundle_balatro.ps1 does this), e.g. into ./local/source
# modded game: base source + your %AppData%\Balatro\Mods (with lovely/dump and smods/)
bash harness/build_modded_tree.sh ./local/source ./local/Mods ./local/modded

python3 harness/replay.py tests/fixtures/1793565485_2730e932.json.gz \
    --src ./local/modded/game --mods ./local/modded/Mods --fast --quiet

# or both fixtures:
bash tests/run_replays.sh ./local/modded
```
The fixtures were recorded on Balatro 1.0.1o with Lovely 0.10.0, Steamodded 26.1002.0 and
Antelytics 1.3.0-alpha. Lovely's dump is rewritten on every launch, so a different
Steamodded version means different patched code. Re-record fixtures against your own setup.

## License and contact
MIT. Use it, fork it, build on it.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic). Questions and collaboration: jacobhadji05@gmail.com
