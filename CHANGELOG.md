# Changelog

All notable changes to this project. Dates are YYYY-MM-DD.

## [0.2.0] - 2026-11-01

The game can now be played from code, not only replayed from logs.

### Added
- **`harness/env.lua`, `harness/env.py`: a playable environment.** Lists the legal actions at every decision and applies one. Legality is the game's own: the same checks its buttons run (`can_play`, `can_discard`, `can_buy`, `can_redeem`, `can_open`, `can_reroll`, `can_skip_booster`, `can_select_card`, `Card:can_use_consumeable`, `Card:can_sell_card`, `check_for_buy_space`, Cerulean Bell's forced selection). Actions go through the same `G.FUNCS` calls `replay.py` validates.
  - `controls` mode (default) is the game's own controls: toggle a hand card, drag hand cards and jokers, sort by rank or suit, then play, discard, use or pick with whatever is selected. Played cards score left to right in hand order, as in the game.
  - `subsets` mode puts the card selection in the action: one step per play, ~200–2000 actions per hand.
  - Jokers can be dragged, sold and used on the cash-out screen, as in the game.
  - Not offered yet: boss reroll (Director's Cut / Retcon), endless mode after a win.
- **Observations limited to what a player can see.** Face-down cards (The Fish, The House, The Wheel, The Mark; jokers under Amber Acorn) are masked. The deck is given as sets (remaining and full deck), never in draw order. Joker and card values shown on hover, the blind and its effect, shop and pack contents, tags, vouchers, hand levels and the round's Idol / Mail-In / Ancient / Castle targets are included. RNG state never is.
- **`harness/check_env.py`: checks the env against logged runs.**
  - `--mode controls` drives a whole logged run through env actions only: joker drags as moves, hand order as moves, selections as toggles. Every action must be legal when taken and the state must match the log at every decision. Both fixtures: 256 decisions through 902 env actions, 0 mismatches.
  - `--mode subsets` checks every logged decision is in the legal set. 256/256.
- **`harness/fuzz.py`**: plays many runs and fails on any offered action that crashes, never settles or changes nothing, and on any observation that leaks seed or RNG state or a face-down card. 100+ runs per mode, no failures.
- **`harness/random_agent.py`**: uniform-random baseline.
- **`tests/run_env_checks.sh`**: both env checks on every fixture, then fuzzing in both modes.
- **`tools/lock_balatro.ps1`**: stops Steam updating Balatro so a dataset stays on one game version. Sets "update only on launch", makes the app manifest read-only, adds a Desktop shortcut that starts `Balatro.exe` directly (mods still load), makes a one-time backup of the game and Mods folders, and checks the exe against the pinned 1.0.1o hash. `-Unlock` undoes it.

### Found
- The unmodded game replays both (Steamodded-recorded) fixtures with every vanilla value matching: money, hand, deck order, score, jokers, consumables, tags, hand levels and every vanilla RNG stream. The only differences are Steamodded's own streams, which an unmodded game doesn't have.
- The build from `harness/build_lupa_luajit.sh` works from a clean clone and reports `LuaJIT 2.0.5`.

### Documented
- README **Versions** table: Balatro 1.0.1o-FULL (full exe SHA-256), LÖVE 11.5, LuaJIT 2.0.5 (DLL hash and the git commit built from), lupa 2.8, Lovely 0.10.0, Steamodded 26.1002.0, Antelytics 1.3.0-alpha, Python 3.13, Linux x86_64, Claude Opus 5.5 (`claude-opus-5-5`, high effort), and which checks ran on the modded vs. unmodded game.

### Checked on
- Env checks and fuzzing: unmodded Balatro 1.0.1o-FULL, Python 3.13.18, Linux x86_64. Not yet run on the modded game.

### Speed
- Unmodded, visuals stripped: ~12 ms per step in controls mode, ~33 ms in subsets mode. A runtime boots in ~0.1 s, and `reset()` reuses it across runs.

## [0.1.0] - 2026-11-01

### Added
- Headless harness that runs Balatro's own game code from your install (no game code in the repo), including Steamodded games as Lovely patches them.
- `replay.py`: replays Antelytics run logs decision by decision and compares RNG streams, deck order, hand, score, money, jokers, consumables, tags and hand levels. Both fixtures replay with 0 mismatches.
- `build_lupa_luajit.sh` (LuaJIT 2.0.5 runtime), `build_modded_tree.sh`, `tools/bundle_balatro.ps1`.
