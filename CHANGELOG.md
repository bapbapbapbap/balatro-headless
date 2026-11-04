# Changelog

All notable changes to this project. Dates are YYYY-MM-DD.

## [0.13.1] - 2026-11-03

### Added
- **The 100% proof as a list of runs** (chud): `tools/play_save.py` plays one save from a new profile, run after run, through the trainer's own game loop, and writes every run in order (wins and losses) in the `routes.jsonl` format. `verify_games.py --save` replays that list in order on a new profile. It checks that each run was selectable when it was played (`H.save_options`: unlocked decks, the next stake per deck, unlocked challenges), and reads the game's own record at the end: `SAVE 100% PROVEN` only at 100%. `docs/proof.md` says so.
- `play_games` skips the replay check when asked (`verify_wins`), so a save's wins aren't counted twice.
- The save ledger reads the game's own progress record (`H.clear_progress`, `H.progress_record`; PR #61).

## [0.13.0] - 2026-11-03

### Added
- **`docs/proof.md`: Balatro can be beaten, a proof by construction** (requested by chud). The first published win, seed `4XNRUA21` (model 0.1, round 329; Red Deck, White Stake, 715 moves), replays to the game's own win on the untouched game. What it shows and what it doesn't, and how to check it. Its second half is generated: the first seed for each part of a full save.
- **Save-progress ledger** (`tools/save_progress.py`, chud's 100%: a Gold Stake win with every joker, Gold Stake on every deck, every challenge). For each deck x stake, each challenge, and each joker x stake, the first published seed that achieved it. Old routes don't say which jokers they held, so they are replayed once in parallel, and the result is cached in `wins/won_with.jsonl`. Writes `wins/save_progress.json` and the tables in `docs/proof.md`.
- Wins now record `jokers_held` (the jokers held at the win) and `found_utc`; `publish_wins.py` keeps both.
- **Decks, stakes and challenges mixed in one run**: `train.py --backs/--stakes/--challenges/--challenge-share` (`BACKS`, `STAKES`, `CHALLENGES` in `run_v2.sh`). Each game's setup comes from its seed, so retries, replays and evals always get the same one. With one deck, one stake and no challenges, nothing changes.
- **Challenges in the env**: `reset(seed, back, stake, challenge)` starts a challenge the way the game's menu does (stake 1, the challenge's deck and rules). Routes, replays, `verify_games.py` and `publish_wins.py` carry `challenge`.
- **Encoding 3** (`--enc 3`, `ENC=3`): encoding 2 plus a token for the deck in play and one for the challenge (both also in the observation). Challenge names are appended to the vocabulary, so encoding-2 weights load into it (`load_weights_into` grows the vocabulary embedding). Encoding-2 games and their final hashes are unchanged.
- **Save mode** (`BalatroEnv(save_mode=True)`, `Sim(save_mode=True)`): a new profile, without `unlock_all`. The game's "seeded run" flag is cleared after the run starts, so runs count for unlocks and stickers; the seed still drives every draw.
- Checks to run on the game: `tests/start_check.py` (every deck and challenge starts as itself and shows in encoding 3; lists every place the source reads `GAME.seeded`), `tests/run_coverage_checks.sh` (the fast sim against the untouched game on every deck at White and Gold Stake, and on every challenge), `tests/list_game.py` (deck keys, challenge ids). `diff_check.py` takes `--back/--stake/--challenge` and `--enc 3`.

### Not checked here
- None of this has run: there's no game or torch in this environment. The checks above are the first test, on chud's machine.

## [0.12.0] - 2026-11-03

### Added
- **Joint training as a mixture of experts** (`train.py --moe`; `docs/hypotheses.md` section I, chud's design). `model.MoE` holds the early-game model (member 0) and the 8 hand experts as separate networks. Each observation names the member that decides it, so one optimizer and every existing safety check (KL guard, reverts, checkpoints) work unchanged, and a member only learns from its own decisions. In each game the early model plays until the ante-3 boss is beaten. A router then picks the expert for the hand played most in antes 1-3, relative to the early model's own average (`base_rate`, a moving average carried with the weights); 30% of training handoffs go to a random expert (`--moe-random`). Expert pins fall to 0 over `--moe-anneal-games` (100,000). The early model is frozen unless `--moe-train-early 1`. Metrics and saved games record the routing, and per expert: games, wins, blinds, and how often it plays its own hand.
- `train/run_moe.sh`: picks the early-game run with the best eval and the 8 experts' `best.pt`, runs `tests/moe_check.py` (routing, pins after the handoff, exact replay, through the trainer's own game loop), then trains.

### Changed
- `collate` adds a 7th tensor (which member decided) only when observations carry one; plain runs are unchanged. `ppo_update` returns zero stats instead of failing when a round has no steps to learn from.

## [0.11.0] - 2026-11-03

### Added
- **Early-game models** (`docs/hypotheses.md` section I, chud's design): `train.py --stop-ante N` ends a game once ante N is reached (the boss before it beaten). It counts as `reached`, never as a win, so nothing reaches `wins/`. `--best-by reached` picks best.pt by the share of eval games that reach it. `train/run_early.sh <shared> <seed>` trains one from random weights on 1.0's architecture (stop at ante 4, reward `blinds`, no pin, no seed retries). `train/run_earlies.sh` grows several seeds (`SEEDS`, default 1-5). `run_v2.sh` takes `FROM_SCRATCH=1`.
- **Stake and deck as settings** (chud): `STAKE` (1-8) and `BACK` in `run_v2.sh`, `run_expert.sh`, `run_early.sh` and `tests/pin_check.py`. Default White Stake, Red Deck; nothing planned changes. A run at another stake or deck gets its own folder (`-stake5`, `-b_blue`). The differential check doesn't cover higher stakes yet.
- `run_experts.sh` takes `HANDS` (a `|`-separated list) to split the experts across machines.

## [0.10.0] - 2026-11-03

### Added
- **Hand pin for strategy experts** (`docs/hypotheses.md` section I, chud's design). `train.py --pin-hand "<hand>" --pin-rate R`: at a decision where a play of that poker hand type is on offer, with probability R the bot may play only that type. Discards and every non-play move stay allowed; with no such play on offer nothing is restricted. The pin enters the action mask (`collate`), so the policy that plays and the policy PPO updates are the same; hand-type exploration never goes around it. The game's legal list is untouched, so saved routes replay as before. Games and metrics log, per game: decisions with the hand on offer, how many the pin bound, plays, plays of the pinned type, and discards made while it was on offer. Off by default; every existing run is unchanged.
- `train.py --init-weights FILE`: start from a released weights file (fresh optimizer, round 0); vocabulary and architecture are checked.
- `train/run_expert.sh <shared> "<hand>" [hours] [workers]`: one expert from model 0.2's weights and architecture, pin rate 0.5 for Pair and 1.0 for the rest, otherwise v2b's settings (ordered-capped, exploration 0.05, lr 1e-5, reward `blinds`). `train/run_experts.sh` runs all eight in turn, Full House first. `run_v2.sh` takes `INIT_WEIGHTS`, `PIN_HAND`, `PIN_RATE`.
- `tests/pin_check.py`: pre-flight on the real sim (the pin's moves, the model's mask, exact replay); `run_expert.sh` runs it before training.

## [0.9.10] - 2026-11-03

### Documented
- README, `weights/README.md`, `docs/models.md`: 1.0's current performance is stated on its own line, **2.5% of seeds won** (5 of 200 fixed evaluation seeds, 12.79 blinds per game), and 1.0 is named as the end of the training line (0.1 → 0.2 → 0.3) that won all 1,549 published seeds.

## [0.9.9] - 2026-11-03

### Checked
- **The 578 seeds published on 2026-11-03 replay on the untouched game: 578 of 578** (`tools/verify_new.sh c22251e 12`, run by chud on his machine). All 1,549 published seeds have now had the untouched-game replay. README and `wins/README.md` say so.

## [0.9.8] - 2026-11-03

### Added
- **Released weights: model 1.0** (`weights/balatro-bot-1.0.pt`, SHA-256 `3e75d85a…3beb64cf`): the final weights of the 0.3 (v2b) run, its `best.pt` when the run ended. Round 810, 90,720 games, eval 12.79 blinds per game and 5 wins on the 200 fixed seeds (0.3-beta: 12.035, 7 wins; `best.pt` is chosen by blinds). Exported by chud with `train/export_weights.py`. Named 1.0 by chud; the seeds the run beat stay counted under 0.3.
- README, `docs/models.md`, `weights/README.md`: 1.0 is the released model; its hash is in the model card.

## [0.9.7] - 2026-11-03

### Results
- **Hypothesis H1 scored: pass** (chud's prediction; `docs/hypotheses.md`, new dated section). Model 0.3 (v2b) solved 24.4 fresh seeds per 1,000 training games against 0.2's 17.9 (ratio 1.37, approx. 95% CI 1.04–1.79), one-sided exact test p = 0.011, over the registered window (rounds 540–882, 20,045 games). No concurrent control, so 0.3's changes and its extra training can't be separated.
- **All 850 seeds published on 2026-11-02 replayed on the untouched game**: 850 of 850 reproduced the same win (`wins/README.md`).
- **Published 578 more beaten seeds, all model 0.3's: 1,549 in total.** Each was replayed exactly in the training sim before it was saved; the untouched-game replay of these 578 is still to do.
- The 0.3 run stopped itself when its training score had been flat for 3 hours. Best evaluation: 12.79 blinds per game. Its final weights aren't exported yet; the released file is still the 0.3-beta snapshot.

### Changed
- **README rewritten for a first-time reader:** a summary at the top, results as of 2026-11-03 (seeds, a table of the pre-registered tests F1 and H1, models, what the bot has learned, what's open), the research question, and a map of the repository. The technical sections that follow are unchanged apart from headings and links.
- `docs/models.md`, `weights/README.md`: training is no longer described as running; H1's result added; the model card link is relative.
- `docs/REPRODUCE.md`: `publish` in the command list and in "Checking the output", how v2b is launched, and the 3-hour stall stop.
- `docs/model_v1.md`: the stall stop reads the training score, not the eval (changed in 0.3.2; the design doc still said eval). The reward section names chud as the one who picks the ought.
- `naneinf/README.md`: no longer says the drag cap cut wasted moves (H's report shows it didn't).
- `wins/README.md`: says which published seeds have had the untouched-game replay.

### Added
- `tools/verify_new.sh <commit> [jobs]`: replays on the untouched game only the routes added to `wins/routes.jsonl` after a commit, split across parallel processes, and prints one PASS/FAIL total. Checks a publish without replaying every route again.

## [0.9.6] - 2026-11-02

### Added
- Two more counters next to seeds proven beatable, requested by chud: **naneinf seeds proven** (0) and **highest stake beaten** (1 of 8, White Stake). Both are badges at the top of README.md and lines in the tally block. `train/tally.py` computes them: the naneinf count is the number of routes in `naneinf/proven.jsonl` (new, empty), the stake is the highest `stake` in `wins/routes.jsonl`.

## [0.9.5] - 2026-11-02

### Added
- `docs/hypotheses.md`: a dated section recording the project's three aims, as chud stated them: exhibit Hume's is-ought gap (primary), prove a naneinf seed, and generalizability to 100% completion from a new save. Open definitions are marked *(chud to confirm)* / *(chud to define)*; no predictions are added.

## [0.9.4] - 2026-11-02

### Changed
- `publish_wins.py` no longer replays wins that training already replayed (`"verified": true`, same check in `train.py` `verify_route`), so publishing is fast. Wins without that mark are still replayed; `--replay` replays every one. The output line says how many were verified by training and how many were replayed.
- The tally line and README say what the check was: each route was replayed exactly before listing, and `verify_games.py --full` replays them all on the untouched game.

## [0.9.3] - 2026-11-02

### Added
- **Running tally of seeds proven beatable**, the project's production count: a badge at the top of README.md (`wins/badge.json`, a shields.io endpoint) and a counts line in README.md and `wins/README.md` (total, by model, by deck and stake). `train/tally.py` writes all three from `wins/routes.jsonl`, so the number shown is always the number anyone can replay.
- `tools/balatro.ps1 publish`: replays every new win from every run under the work folder (`runs/*/wins.jsonl`), adds the ones that replay exactly to `wins/`, updates the tally, commits and pushes.

### Changed
- `publish_wins.py` skips a line that training is still writing instead of failing on it.
- README: the results list and F1 point to the tally instead of fixed counts.

## [0.9.2] - 2026-11-02

### Changed
- README: F1 names the models both ways (0.2 = v2, 0.1 = v1, as in `hypotheses.md`) and says 0.2's 545 new seeds were replayed but their routes aren't in `wins/` yet. The results list gives the count of published seeds (121, all 0.1's).
- README: "Two action modes" is now "Action modes" (there are four) and "Both" is "All modes". The 165 → 60 ms figure in "What we did" is marked as the first `replay.py` measurement on the modded game. The duplicate "Runs on Linux (or WSL)" line is removed (the Versions table has it).
- `wins/README.md`: the stale "No model weights are published" is replaced by a pointer to `weights/`, and it says the list holds 0.1's seeds only so far.

## [0.9.1] - 2026-11-02

### Changed
- README: a results section near the top covers seeds proven beatable, the best model, the F1 test, what the won games show, open questions, and the larger aim (Hume's is-ought gap, chud's idea). Stale lines fixed: the speed figures, "research preview", and boss reroll, which is offered in `ordered`. The `ordered` and `ordered-capped` action sets are now documented.
- The watcher defaults to `%USERPROFILE%\balatro-work`. `start_v2b.sh` requires the shared folder as an argument. Neither has a personal path anymore.
- The CHANGELOG entry 0.3.4 is back in version order.
- `docs/models.md`: model versions (0.1–0.3) are separate from tooling versions.

## [0.9.0] - 2026-11-02

### Added
- **Released weights: model 0.3-beta** (`weights/`), plus 0.1 and 0.2. Version names: 0.1, 0.2 and 0.3 are the runs called v1, v2 and v2b. Model card: `docs/models.md`.
- `train/export_weights.py`: checkpoint to a model-only weights file (float32 weights, vocabulary, architecture, reward, commit, game hash, evaluation). Drops training state and anything identifying the machine (host name, local paths), and refuses to write if any remain. Loads with `weights_only=True`.
- `train/play.py`: plays a seed with released weights, prints the moves, and can save the game as a route that `verify_games.py` replays.

### Checked (cloud)
- All three exports load with `weights_only=True`, and their metadata contains no host name or local paths.
- 0.3-beta and 0.1 each played a test seed through `play.py`, and the saved 0.3-beta route replayed exactly (`verify_games.py` PASS).

## [0.8.3] - 2026-11-02

### Added
- **`analysis/`** (`replay_events.py`, `analyze.py`) and **`docs/won_games_analysis.md`**: v2's 709 training wins replayed (0 mismatches) and tested against baselines. Learned: planets for its main hand (91% vs 25% chance), Clever Joker raises Two Pair plays (47% to 57.5%), Two Pair-scaling jokers fit its play. Not learned: Full House (0 of 2,274 times it was the best hand on offer), playing a hand after leveling it (+0.4 ± 0.6 points), boss play (about 20% of plays score nothing under Psychic, Mouth, Eye), money far past the interest cap ($122 median at ante 8).

### Found
- **The joker-drag cap doesn't cut drags.** It holds (no chain over jokers - 1), but in ordinary training games drags were mostly short already: 22.0% of moves in v2 (rounds 493-504) vs 23.6% in v2b (rounds 518-529), 124 vs 127 moves per game. The 42-43% figure is for won games only.
- **v2's slowdown on 2026-11-02 wasn't the learning rate.** Updates stayed at 25-45 s for an hour after the lr change (18:28), then jumped to 160-194 s around round 488 (about 19:25). Cause unknown; the v2b restart cleared it (40-70 s).

## [0.8.2] - 2026-11-02

### Added
- **Standing goal: a model that reaches naneinf** (`naneinf/README.md`), set by chud: what counts (a hand after which `G.GAME.chips` is not finite, in a route that replays), what the game's source does with inf and nan, and what the bot still needs (endless mode, a reward of record, inf/nan-safe scores and shaping, a longer move cap, encoding checks).
- **Community naneinf seeds checked on 1.0.1o** (`naneinf/community_seeds.json`, `probe.py`, `community_seeds_probe.json`): eight published seeds; the published openings of five match on this version (Perkeo from a Charm Tag's Soul on four Ghost seeds; Baron and Blank in a first shop; Blank on the Plasma seed); three not confirmed.

## [0.8.1] - 2026-11-02

### Added
- **`docs/REPRODUCE.md`**: from a Steam copy of Balatro to a training bot and checked routes. Covers hardware, the game pin and lock, the three mods (Lovely 0.10.0, Steamodded 26.1002.0, Antelytics 1.3.0~alpha; only for logging), the bundle, WSL packages, the harness build, training, verifying `wins/`, a versions table, and known gaps.
- **`tools/balatro.ps1`**: one PowerShell controller: `check`, `lock`/`unlock`, `bundle`, `setup`, `train`, `stop`, `status`, `watch`, `verify`, `help`. Parses under PowerShell 7.4; `help` and `status` dry-run on a sample run folder.
- README: what the project produces (seeds proven beatable, with routes) and a pointer to the guide.

### Changed
- `tools/setup_linux.sh` installs numpy and torch into the venv when missing and prints their versions.
- `tools/start_v2b.sh` takes the shared folder, hours and workers as arguments.
- README versions table: Python 3.12.3 (WSL training) alongside 3.13.18; PyTorch 2.15.0+cu132.
- `wins/README.md` points to the guide (the link it had was dead).

## [0.8.0] - 2026-11-02

### Added: v2b (opt-in; v2 unchanged)
- **`--actions ordered-capped`**: `ordered` plus a cap on joker drags. After k drags in a row, no further drag is offered until another action is taken, with k = jokers held - 1. Every joker order stays reachable (n jokers need at most n - 1 drags). Why: in v2's won games 43% of all moves were joker drags (about 136 per game, up to 248 in a row). They are free and change nothing, but each costs ~30 ms of sim time and counts toward the 1,500-move cap. Saved games record `actions: "ordered-capped"` and replay under it; older games replay as before.
- **`--explore-hands EPS`** (training games only): at a decision with plays of two or more poker hand types on offer, with probability EPS one of those types is picked uniformly, then a play of that type in proportion to the policy. Why: v2 never played a Full House when one was on offer (0 of 374 in its wins), most likely inherited from cloning v1. Stored log-probabilities are the policy's own, so an explored play that pays gets a full-strength update (epsilon-greedy; small bias, stated in the docstring).
- **A v2 checkpoint can continue under `ordered-capped`** (same model and moves); logged as an alert.
- **`train/run_v2b.sh`**, **`tools/start_v2b.sh`**: v2b = v2's `latest.pt` + drag cap + exploration 0.05 + learning rate 1e-5, in `runs/v2b`. `run_v2.sh` takes `RUN`, `ACTIONS`, `RESUME_FROM`. The watcher accepts `-Run v2b`.

### Checked (cloud)
- Cap holds on a real ante-2 state with 4 jokers: 3 drags offered in a row, then none until another move, then all 12 again.
- `ordered_semantics.py` ALL PASS; `diff_check.py --mode ordered --encode --enc 2`, 5 runs, PASS; 30 saved v2 wins replay with 0 mismatches (cap off, as recorded).
- Exploration sampler on a real decision with a Full House on offer (381 plays, 5 hand types): at EPS 0.05 each type gets ~1% of draws on top of the policy; returned log-probabilities match the policy's.
- Not run here: the trainer itself (no torch in this container). The laptop's pre-flight and first round are the smoke test.

## [0.7.1] - 2026-11-02

### Added
- **Behavior cloning before RL** (`--bc games.jsonl ... --bc-epochs 2`): workers replay saved games into exact training data; the policy learns the move that was made (cross-entropy), the critic the game's actual discounted return. Runs once on a fresh model (skipped on resume), then evals and continues with RL; saves `bc.pt`. Meant for starting v2 from v1's play: v1's games converted to v2's action list.
- `convert_routes.py`: reads `games/` directories and `.jsonl.gz`, newest rounds first; `--train-only` (eval-seed games stay out of anything v2 learns from), `--max-games`, `--jobs` (parallel).
- **`train/run_v2.sh`**: pre-flight (saved win on the untouched game, ordered-action checks, fast vs untouched under encoder v2), converts v1's most recent training games (default 3,000) plus converted wins, clones, then RL at full power; resumes from `runs/v2/latest.pt` later.
- **Time per decision** in the log: model vs sim, in the workers (whether moving move choice to the GPU would pay).

### Checked (cloud)
- Behavior cloning trains (2-core run cut short: move loss 3.68, saved move matched 23.7% after the first 8 games). Converted games whose saved final hash predates a v2 encoding change are refused, as they should be.

## [0.7.0] - 2026-11-02

### Added: v2 (opt-in; v1 defaults unchanged, its encoding byte-identical)
- **`--actions ordered`**: the subsets action list plus what card order can change.
  - A play whose score can depend on order is offered once per scoring card as the lead (dragged to the front of the selection, as a player would): when Hanging Chad is held, or a per-card x-mult source (Glass, Polychrome, Photograph, The Idol, Bloodstone, Ancient Joker, Triboulet) meets a per-card +mult source (Mult, Lucky, Holographic, Smiley Face, Scholar, Walkie Talkie, Fibonacci, Even Steven, Onyx Agate, the four suit jokers); Blueprint/Brainstorm count as both. Only the lead is chosen. Stone cards and Splash's extra scoring cards aren't offered as leads.
  - **Death** is offered with each selected card as the one copied (the game copies the rightmost; before, the bot could only copy the lower card onto the higher).
  - **Boss reroll** (Director's Cut / Retcon), with the game's own condition.
  - Checked in the real game (`harness` test): the lead card scores first; each Death direction copies the named card; boss reroll changes the boss, costs $10, only with the voucher.
- **`--enc 2`**:
  - **Joker numbers in fixed slots**: current mult, x-mult, t_mult, t_chips, hand/discard size, `extra` (scalar or up to 3 named values), Canio/Invisible/Loyalty/Yorick counters, sell value. v1 kept the first 8 numeric fields alphabetically, which for all 150 jokers are mostly constant and drop `mult` and `x_mult`: v1 sees which scaling joker it holds but not how far it has scaled.
  - Hand table: per-level increments (`l_chips`, `l_mult`); Planets carry the hand they level (as plays carry theirs).
  - Every candidate play carries its scoring cards, the HUD's chips x mult for that hand and the scoring cards' chips.
  - Interest: per-$5 amount (To the Moon), the no-interest flag (Green Deck), and the projected interest from money held, as the game computes it.
- **Model v2** (`--enc 2`, default size used: `--d 192 --layers 4 --heads 6`, 1.6M parameters vs 250k): reads the new action fields.
- **`train/convert_routes.py`**: translates saved games from the v1 action list to the v2 one move by move (a play keeps its lead, Death keeps copying the rightmost card); kept only if it ends the same way. 33/33 laptop-recorded wins converted on the cloud sim.
- **`train/publish_wins.py`** and **`wins/`**: the public record of beaten seeds, one proof per seed (the shortest win), each replayed before it is kept.

### Checked (cloud, unmodded 1.0.1o)
- v1 unchanged: the committed win replays to the identical final-state hash; `check_env --mode subsets` passes on a logged run.
- Ordered mode: 10 fuzz runs (no crash, stall, no-op or leak); fast vs untouched game identical with the v2 encoder running every step (8 runs).
- v2 training runs end to end (1.6M parameters, converted wins replayed into self-imitation), and its saved games replay exactly.

### Results
- **Stage 2 A/B, self-imitation** (v1 round 494, side by side, 25 min, buffer pre-loaded with 79 saved wins): evals 10.76/11.08/10.85 off vs 10.94/10.75 on (400 seeds), no difference beyond noise; only ~3% of replayed moves had R > V; throughput 10,700 vs 4,100 games/h. Stays off.

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

## [0.3.4] - 2026-11-01

### Added
- **`tools/watch_v1.ps1`**: Windows watcher for unattended runs. Keeps the PC awake (`SetThreadExecutionState`, re-applied every check), prints a status line every 60 s (games, wins, 10-round training average, best eval, speed, free memory, new alerts), and restarts training only after an unexpected death (status stale 15 min with no training process, or a crashed/worker-failure stop), at most 5 times. Never restarts after a deliberate stop.

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
