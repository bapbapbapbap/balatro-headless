# Changelog

All notable changes to this project. Dates are YYYY-MM-DD.

## [0.3.4] - 2026-11-01

### Added
- **`tools/watch_v1.ps1`**: Windows watcher for unattended runs. Keeps the PC awake (`SetThreadExecutionState`, re-applied every check), prints a status line every 60 s (games, wins, 10-round training average, best eval, speed, free memory, new alerts), and restarts training only after an unexpected death (status stale 15 min with no training process, or a crashed/worker-failure stop), at most 5 times. Never restarts after a deliberate stop.

## [0.6.2] - 2026-11-02

### Added
- `--sil-seed <wins.jsonl ...>`: fills the self-imitation buffer from saved wins at start (verified training wins on the same deck and stake; eval-seed wins left out so the eval stays clean). Checked: 48 laptop-recorded wins loaded on the cloud sim, and the ones sampled replayed exactly there (cross-machine).

### Fixed
- Two runs starting at once could report the mirror drive unwritable: the write test used one shared file name. Now one per process.

### Results
- **Stage 1 A/B, learning rate** (v1 round 473, two arms side by side, 7 workers each, 27 rounds, same seeds): 3.75e-5 vs 1e-4 evals 10.79/10.88/10.15 vs 10.78/10.50/10.21 (400 seeds each), no difference beyond noise. At 1e-4 every update hit the KL target after ~100 of ~280 minibatches: the KL target, not the learning rate, now sets how far an update moves. v1 continues at 1e-4 (same learning, cheaper updates). 33 seeds beaten during the test.

## [0.6.1] - 2026-11-02

### Fixed
- **Crash on games that end at ante 0** (a Hieroglyph voucher sets the ante back by 1): the retry pools had no ante-0 bucket (`KeyError: '0'`). Such seeds now go with ante 1. Crashed v1 at 23:59, 00:07, 00:26 and 00:47; after five restarts the watcher stopped relaunching and the run was down from 00:47.
- **GPU out of memory** in the KL measurements added in 0.6.0 (start-of-update log-probs and the guard's sample): they ran 256 states at a time, and states with ~2,000 legal plays don't fit; now the training minibatch size. Crashed v1 at 00:02.
- **Early stopping fired on noise**: it compared one 64-step minibatch's KL with the threshold (67 early stops in 100 rounds while the average update moved 0.011). Now the mean of the last 4 minibatches, and never before the 4th.

## [0.6.0] - 2026-11-01

### Added
- **Overlapped rounds** (`--overlap 1`, default): the workers play the next round while the learner updates, so the CPU isn't idle during the GPU update. Those games are played with weights one update old; PPO's ratio uses the log-probs of the policy that actually played them. Requests to the workers are numbered, so evals and self-imitation replays can land in between.
- **Early stopping on KL** (`--target-kl 0.04`): an update stops once it has moved the policy past 1.5x the target, measured against the weights the update started from. Before, an update ran all epochs and was thrown out if it overshot (v1: 5 rejections between rounds 312 and 324, learning rate down to 3.75e-5).
- The reject-guard (`--max-kl`) now checks how far the update itself moved the policy (`moved` in the log, on up to 2,048 of its steps), not the drift since the games were played, which with overlap includes the previous update.

### Changed
- **Self-imitation off by default** (`--sil-buffer 0`). At weight 1.0 on v1 it went with KL past the guard and entropy 2.35 -> 1.9; a cut to 0.2 still overshot. Now dosed as `--sil-steps 8` minibatches per update with the pull capped at `--sil-clip 1` PPO standard deviations and a Huber critic loss. A 2-core test couldn't separate its effect from noise (16-game rounds: entropy fell with it on *and* off), so turning it back on waits for an A/B test at full size: two runs from the same checkpoint, `--sil-buffer 64` vs `0`.

### Checked (2-core cloud, unmodded 1.0.1o, from v1 round 325)
- Overlap: rounds pipelined, evals and replays interleaved, 32/32 saved games replay exactly. Early stop at 5-8 minibatches; injected NaN at an overlapped round caught, reverted and counted as a strike.

## [0.5.0] - 2026-11-01

### Added
- **Seed retries, allocated by ante** (`--retry 1 --fresh-min 0.5 --arm-prior 100 --retry-max 30 --retry-pool 500`): lost seeds are kept by the latest ante they reached (1-8) and replayed with fresh sampling until beaten or out of attempts. Each round's games are split across 9 arms (fresh, ante 1..8) by Thompson sampling on wins per game, Beta(1 + wins, prior + losses), with at least `--fresh-min` on fresh seeds. Training stats stay on fresh seeds; per-arm games and wins, the round's split and unique beaten seeds go to the log, `metrics.jsonl` and `health.json`; saved games carry `retry` and `arm`. Pools, arm records and beaten seeds are saved in checkpoints.

- **Second drive**: `--mirror-dir` appends every win to `<dir>/wins.jsonl` as well (a second copy of the beaten seeds); `--spill-dir` takes new saved games and checkpoints once the main drive is under `--spill-at-gb` (default 50) free. Both alert instead of failing if the drive isn't there. `run_v1.sh` points both at `/mnt/f/balatro data` (override with `MIRROR_DIR` / `SPILL_DIR`).

### Changed
- **Self-imitation weight 1.0 -> 0.2.** On v1 (rounds 307-320) weight 1.0 raised KL from ~0.035 to 0.1-0.26 (two updates rejected, learning rate halved twice to 7.5e-5), entropy fell 2.35 -> 1.9, and 6 of 200 eval games at round 320 looped on joker drags to the decision cap.
- `--reset-lr` restores `--lr` on resume; `run_v1.sh` passes `TRAIN_ARGS` through for one launch.

### Fixed
- `run_v1.sh` pulls the repo first and then restarts itself from the updated script. Before, a pull that changed the script mid-run made bash skip lines (the new pre-flight replay didn't run on the 22:48 launch).

## [0.4.0] - 2026-11-01

### Added
- **Every game saved exactly** (`--save-games all`, default): seed, setup and every action, as its index into the legal list and as text (`play 1,3,5`, `buy 2`), plus a hash of the final observation, one gzipped file per round in `<run>/games/` with `meta.jsonl` recording what they were played on. Wins also go to `<run>/wins.jsonl`. About 4 MB per hour of training at 14 workers.
- **Wins verified on the spot**: the worker replays each won game from its seed and actions before it is saved; a win that doesn't reproduce is saved, flagged `verified: false` and raises an alert.
- **`train/verify_games.py`**: replays saved games in a fresh runtime (`--fresh-each`), optionally on the untouched game (`--full`), checking every move, the step count, the final state and the outcome.
- **Self-imitation learning** (Oh et al. 2018; `--sil-buffer 64 --sil-games 8`, `--sil-buffer 0` turns it off): the highest-return training games are kept as routes, replayed by the workers into exact training data every round, and pull the policy toward their moves by max(R - V, 0), inside the same guarded update as PPO. Selected by return under the reward of record, so no new goal enters. The buffer is saved in checkpoints.

### Checked
- Recording: 12 random-play games replayed with 0 mismatches in both the fast and the untouched sim; 48 games from a short training run (resumed from v1 round 260) replayed with 0 mismatches, a fresh runtime per game.

## [0.3.3] - 2026-11-01

### Added
- **`--max-wins`**: stop after a total number of won training games (beat the ante 8 boss); carried across resumes. `run_v1.sh` now defaults to 1,000,000 wins, no games limit, and a 168-hour cap.
- **Hypotheses**: run target change registered (prediction A5: a backstop ends the run first), and E1 "Hanging Chad is the best common joker" (proposed by chud) with its test design and precondition.

## [0.3.2] - 2026-11-01

### Fixed
- **Boss counting.** `Blind.boss` is already cleared when the cash-out screen shows, so bosses beaten read 0. Now uses the game's own test from `end_round` (the beaten blind is neither Small nor Big). Checked with the laptop's round-19 weights: bosses beaten = ante - 1 in every game.
- **Evaluation by seeded sampling, not argmax.** The argmax evaluation looped on free actions (dragging a joker back and forth): 73-82% of evaluation games hit the 1,500-decision cap while training improved from 0 to ~7 blinds per game. Evaluation now samples like training, with a fixed RNG per evaluation seed.
- **Stall stop reads the training score** (10-round average), which has never looped; evaluation still picks `best.pt`. Sampled evaluation on the laptop still capped 116/200 games, not reproducible on another machine; capped evaluation games now log their last 30 states and moves to `eval_capped.jsonl`.

### Added
- **Auto-resume**: `run_v1.sh` continues from `latest.pt`; checkpoints carry the best eval and games so far.
- **`--max-games`**: stop after a total number of training games; `run_v1.sh` defaults to 100,000 and 14 workers.
- **`docs/hypotheses.md`**: pre-registered hypotheses for the v1 run, shaping, and the main experiment, plus what has to be done before the main training. Committed while the v1 run was in progress, before its results.

### Results so far (laptop, Core Ultra 9 285H, RTX 40 learner)
- v1, Red Deck / White Stake, reward `blinds` + shaping: training average 0 -> ~7 blinds per game in ~5,000 games; sampled eval 5.58 blinds per game at round 50; games reaching ante 6-7; 0 wins yet. ~13-15k games per hour on 12 workers.

## [0.3.1] - 2026-11-01

### Added
- **Automatic safety checks for unattended training.** Pre-flight accuracy checks in `train/run_v1.sh` (nothing starts unless they pass). During training: bad updates (non-finite, or KL above `--max-kl`) reverted with the learning rate halved, stop after 3 in a row; crashed or hung workers restarted with the round retried, stop after 2 failures; worker pool sized to free memory; clean stop on low memory, low disk, 3 hours without eval improvement, or Ctrl+C (within seconds); heat-throttling flag; `best.pt` kept separately; atomic checkpoint writes; `health.json` every round. Each verified by fault injection (`--inject`), forced low memory, forced stall and an interrupt.

## [0.3.0] - 2026-11-01

### Added
- **Model v1 and its trainer** (`train/`, design in `docs/model_v1.md`). A ~250k-parameter transformer over what the player sees, scoring each legal action; PPO with parallel workers, one sim per worker. Nothing in it encodes strategy; what the bot wants is `--reward` (`blinds`, `antes`, `win`), recorded in every checkpoint with the repo commit, the `Balatro.exe` hash, the vocabulary hash and the full config.
- **Potential-based shaping** (Ng, Harada & Russell 1999), on by default: a signal for progress inside a blind (chips scored / target) that is taken back in full when the blind ends, so it sums to ~0 over a game and provably leaves the optimal policy for the chosen reward unchanged. Without it, near-random early play beat a blind once in 640 games and nothing was learned; with it, blinds beaten per game rose from 0 to ~2 within ~1,100 games (Red Deck, White Stake, 2 workers). Reported scores are always the unshaped reward.
- **`harness/encode.lua`**: numeric encoding for learning. Flattens `H.observe()`, so it has exactly the same visibility rules (face-down masked, deck as sets, no RNG state); builds the action list itself and plays by index into it, so a choice can never be mapped to the wrong action. Each candidate play carries its poker hand type, which the game shows on selection.
- **`diff_check.py --encode`**: runs the encoder before every comparison on the candidate side only. 40 runs, 574 decisions: encoding never changes the game.
- **`train/run_v1.sh`**: overnight launcher for Linux/WSL after `tools/setup_linux.sh`.

## [0.2.1] - 2026-11-01

### Fixed
- **Listing legal actions no longer changes the game.** The shop's room check (`G.FUNCS.check_for_buy_space`) also shows the game's "No space!" alert, which locks the controller and queues events, as if the player had clicked Buy. `env.lua` called it for every shop joker while listing actions, so with full joker slots, merely asking for legal actions locked selling and using for a moment. The check now runs with only the alert switched off. Found by the new differential check.

### Added
- **`harness/diff_check.py`: differential accuracy check.** A reference sim (nothing stripped) and a candidate sim (`--fast`, plus any optimization under test) play the same seeds with the same actions. After every action the complete logic state must be identical: `G.STATE`, all of `G.GAME` (every RNG stream, money, round state, hand levels, pools), every card in every area with its full ability table, edition, seal, debuff, facing and selection, the blind, the tags, and the legal actions offered. Visual-only fields are excluded, each one justified by reading every use in the source (vanilla 1.0.1o): drawn positions and animation state, the debug outline flag, and the HUD's display strings (`chips_text`, `chip_text`, `mult_text`, `chip_total_text`, `handname_text`). An audit of all 57 UI callbacks found no other game state they write during a run. `--fast` vs the untouched game: 60 runs, 931 decisions across every state type, identical.
- **`tools/setup_linux.sh`**: one-shot setup on Linux or WSL from a `balatro_bundle.zip`: clones the repo, builds the runtime, unpacks the bundle on the Linux filesystem, builds the modded tree, runs every accuracy check and writes a report next to the bundle.

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
