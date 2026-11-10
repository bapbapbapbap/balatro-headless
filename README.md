# balatro-headless

[![seeds proven beatable](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge.json)](wins/README.md)
[![naneinf seeds proven](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge_naneinf.json)](naneinf/README.md)
[![highest stake beaten](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbapbapbapbap%2Fbalatro-headless%2Fmain%2Fwins%2Fbadge_stake.json)](wins/README.md)

**Balatro's real game logic, run headless; a bot trained on it that sees only what a player
sees; and a public record of every seed the bot has beaten, each with a route anyone can replay.
Built for one experiment: hold the game fixed, vary only the reward, and see what changes
(Hume's is-ought gap).**

- **The harness** runs the game's own Lua from your install, with no window, sound or Steam, so
  runs can be stepped, replayed and searched from code. Nothing is reimplemented. It is checked
  against runs logged in the real game, decision by decision, with
  [zero mismatches](#harness-validation).
- **The bot** is a transformer trained by reinforcement learning on the harness. It sees cards
  face up, the deck as a set, and never the draw order or the game's random state.
- **The output** is seeds proven beatable. Each beaten seed is saved with its route, every move
  from the seed to the win, and replayed before it counts. The game is deterministic, so the
  route is the proof.
- **The research aim** (chud's) is to exhibit Hume's is-ought gap in an agent; the facts are the
  game, the ought is the reward. [More below](#the-research-question).

> ### Balatro is not ours.
> **Balatro** was created by **LocalThunk** and is published by **Playstack**. The game, its
> code, art, audio, card text and name belong to them. This project isn't affiliated with or
> endorsed by either of them.
>
> **This repository contains none of the game.** No code, no art, no audio, no text. Everything
> here is our own tooling. It works only with **your own purchased copy** of Balatro, and it
> reads the game from your install on your machine. If you use it, keep the files it extracts
> private and never commit or share them. Our [MIT license](LICENSE) covers our tooling only; it
> grants nothing about Balatro.

---

<!-- tally:start -->
**Seeds proven beatable: 1,225** (24 by model 0.1, 179 by model 0.2, 232 by model 0.3, 790 by model moe2; 1,225 on Red Deck, White Stake). Each one is a route in [`wins/`](wins/) that was replayed move for move to the same win before it was listed. This is the project's production count, updated each time wins are published.

**naneinf seeds proven: 0.** A seed counts once its route in [`naneinf/proven.jsonl`](naneinf/proven.jsonl) replays move for move to a played hand with non-finite chips (see `naneinf/README.md`).

**Highest stake beaten: 1 of 8 (White Stake).**

**Final-boss near misses: 1,500.** Seeds where the bot reached the Ante 8 boss and lost it, kept with their routes in [`wins/near_misses.jsonl`](wins/near_misses.jsonl): median 62% of the boss's target scored, 226 at 90% or more. Until 2026-11-03 these were counted as wins (see the CHANGELOG, 0.13.2).
<!-- tally:end -->

## Results (through 2026-11-10)

All on unmodded Balatro 1.0.1o-FULL, Red Deck, White Stake, with the reward "blinds beaten".
Every claim links to its data or its pre-registration.

> **Correction, 2026-11-03.** Until this date every win count here included runs that **lost** to
> the Ante 8 boss. Balatro sets `G.GAME.won` at the end of the Ante 8 boss round whether or not the
> boss was beaten, and the harness read that flag alone. Of 1,549 seeds published as beaten, **435
> are wins**; 1,114 reached the Ante 8 boss and lost it, and stay posted as near misses. The
> reward "blinds beaten" also paid that loss as a beaten blind. F1, H1 and R were scored on the
> wrong count; they stay void as scored and were closed without re-scoring on 2026-11-09
> ([closed hypotheses](docs/closed_hypotheses.md); [correction](docs/hypotheses-general.md#added-2026-11-03-2050-cst-correction-final-boss-losses-were-counted-as-wins)).
> Found by chud's save-progress ledger, which reads the game's own record of each win.

### Seeds proven beatable
The count is at the top; the list (`beaten_seeds.csv`) and the routes (`routes.jsonl`) are in
[`wins/`](wins/README.md).
- Every route was replayed exactly before it was listed, and all of them a second time on the
  **untouched** game (nothing stripped): each reproduced the same run, move for move. Those checks
  read the win from `G.GAME.won` alone, so they passed the final-boss losses too.
- On 2026-11-03 every route was replayed again with the game's own win test
  (`tools/recount_wins.py`, at tag `pre-launcher`: `ARCHIVE.md`): 435 wins stay in `routes.jsonl`, 1,114 final-boss losses moved to
  `near_misses.jsonl`. Anyone with the game can rerun it: `train/verify_games.py --full`.

### Pre-registered tests
Each prediction, window and test was committed to [`docs/hypotheses.md`](docs/hypotheses.md)
before the data it is scored on existed. The commit is the timestamp, and earlier sections are
never edited; results are added as new dated sections.

| Test | Prediction | Result |
|---|---|---|
| **F1**: model 0.2 vs 0.1, new beaten seeds per hour | 0.2 is faster | **Void as scored; closed without re-scoring** ([ledger](docs/closed_hypotheses.md)). Counted final-boss losses as wins. As scored: 90.8 vs 42.3 per hour (ratio 2.15, 95% CI 1.70–2.71), one-sided exact test, p = 2×10⁻¹² ([result](docs/hypotheses-v2.md#added-2026-11-02-1825-cst-f1-result-v2-vs-v1)) |
| **H1** (chud's): model 0.3 vs 0.2, solved rate | 0.3 is higher | **Void as scored; closed without re-scoring** ([ledger](docs/closed_hypotheses.md)). Counted final-boss losses as wins. As scored: 24.4 vs 17.9 wins per 1,000 fresh seeds (ratio 1.37, 95% CI 1.04–1.79), p = 0.011 ([result](docs/hypotheses-v2b.md#added-2026-11-03-1245-cst-h-result-v2b-vs-v2-solved-rate)) |
| **R0** (chud's): model 1.0 on the 1,549 proven seeds vs 1,549 fresh seeds, 10 games each | proven seeds clear above 1.0's 2.5% | **Void as scored; closed without re-scoring** ([ledger](docs/closed_hypotheses.md)). Counted final-boss losses as wins, among the seeds and in the games. As scored: 21.0% vs 2.9% of games won; per-seed difference +18.1 points (seed bootstrap 95% +16.7 to +19.4). ([result](docs/hypotheses-v2b.md#added-2026-11-03-2015-cst-r-result-re-solve-by-10)) |

H1 has no concurrent control (one machine), so 0.3's two changes and its extra training can't be
told apart. The model names 0.1, 0.2 and 0.3 are the runs called v1, v2 and v2b in
`hypotheses.md`.

### Models
- **Released: model 1.0** ([`weights/`](weights/README.md), [model card](docs/models.md)), 1.6M
  parameters: the final weights of the 0.3 run (round 810, 90,720 games), the end of the training
  line (0.1 → 0.2 → 0.3) that won the 435 seeds above.
- **1.0's performance, as counted before 2026-11-03 (not re-measured):** 2.5% of seeds (5 of 200 fixed evaluation
  seeds) and 12.79 blinds per game were counted with the old win test, so both include final-boss
  losses (each such loss counted as a win and as 24 blinds instead of 23). Its reward never paid
  more for beating the last boss than for losing to it.
- **The 0.3 run has ended.** It stopped itself on 2026-11-03 when its training score had not
  improved for 3 hours, as its safety rules require. 0.3-beta (12.035) is its mid-run snapshot;
  the seeds the run beat are counted under 0.3.
- Everything ran on one laptop, at about 6,000–8,000 games an hour. The game, not the network,
  is the bottleneck: about 35 ms per move against about 8 ms for the model.

### What the bot has learned
From 709 of 0.2's saved "wins", replayed move by move and tested against baselines
([`docs/won_games_analysis.md`](docs/won_games_analysis.md)). Most of those were final-boss losses
(see the correction above), so this describes runs that reached the Ante 8 boss, not wins; it is
to be rerun on real wins:
- **Learned:** it picks planets for the hand it plays (91% of the time, against 25% by chance),
  and picks jokers that fit its Two Pair play.
- **Not learned:** Full House (played 0 times in 2,274 chances), playing around three of the
  bosses, and spending money past the interest cap.

### The mixture of experts (moe2, ended 2026-11-05)
An early-game head for antes 1–3, then one of 6 hand experts chosen by a routing rule at ante 4, all
from model 0.2 (`hypotheses.md` section I, chud's design). "Pins" forced each head's hand some of the
time and faded to 0 over 100,000 games. The run stopped itself at the first eval with every pin at 0
(round 1042): **12.94 blinds per game, 9 wins and 13 near wins in 600 seeds** (round 0: 11.59, 2
wins). Its best was 13.08 blinds (round 1020) and 14 wins (round 1040).
- **Full House wasn't learned.** The early head played a Full House or higher 1 time in 1,005 chances, and the
  router sent the Full House expert no games. The completion criterion is not met.
- **The pins didn't stick, and the bot recovered without them.** It lost ground as they faded (10.32
  blinds at round 600) and rebuilt from the reward alone to its best evals.
- **Blinds per game rose; win rate hasn't clearly passed the run's early peak.** Most games are lost in antes 4–6.

Weights: [`weights/balatro-bot-moe2-pins0.pt`](weights/README.md), [system card](docs/system_card_moe2.md).
Full results: [docs/moe2_evaluation.md](docs/moe2_evaluation.md), scored in
[hypotheses.md](docs/hypotheses-moe2.md#added-2026-11-05-1540-cst-moe2-at-pins-zero-results). Measured
afterwards from the same positions (2,000 seeds), the six experts play the same hand mix and land within
0.24 blinds of each other: the handoff helps because any expert plays past ante 3 better than the early
head, not because the experts are different strategies
([scored](docs/hypotheses-moe2.md#added-2026-11-05-2110-cst-the-first-paired-run-scored-section-j)).

### v4 (2026-11-09 → 11-10): stopped, diagnosed
Two networks of 1.0's size, one for the table and one for the shop, each with its own critic and its own
credit; a build-strength potential that scores the board with the game's own scoring code; each shop
offer's value on its action; and 30% of games started mid-run at ante 4–6 from replayed wins
([registered](docs/hypotheses-v4.md), [page](docs/runs/v4.md)). Its best eval was 4.68 blinds per game at
round 120; then the table network learned to sell its jokers mid-blind, and the run stopped on its stall rule
at round 620. The cause: inside a blind, credit wasn't discounted, so the whole cost of dying landed on every
move of the final blind, and PPO pushed probability onto the moves it rarely took, selling among them
([diagnosis](docs/hypotheses-v4.md#added-2026-11-10-1030-cst-why-v4-stalled-diagnosis-after-the-run-section-j)).

### Now: model v4b (since 2026-11-10)
v4 with credit discounted per hand played or discarded on the table network (joker drags share the next hand's
credit), and hand-type exploration off; it stops itself if an eval averages more than 1.5 joker sells per game
([registered](docs/hypotheses-v4b.md), [page](docs/runs/v4b.md), settings
[`configs/v4b.toml`](configs/v4b.toml)). Same ought (`blinds`), Red Deck, White Stake, from random weights.
- **To score:** S1–S3 (it doesn't sell; it beats v4's 4.68; it reaches v1's pace), and W1–W4 against
  moe2-pins0 on 2,000 held-out seeds after the run.
- **The bar for going up the decks and stakes:** 25% eval wins on Red/White (chud), open to revision.

### Open
- **The main experiment** (section C, [`hypotheses-general.md`](docs/hypotheses-general.md)): same facts,
  different oughts. Registered, not yet run; it waits for a bot that reaches the late game (ante 7 in at
  least 25% of eval games), where the oughts disagree.
- **Full House: closed** (2026-11-09). No bot learned it, and no run forces a hand any more: forcing one
  muddles the variable under study, training. Full House play is still reported, no longer tested
  ([closed hypotheses](docs/closed_hypotheses.md)).
- **The joker-drag cap** in 0.3 holds, but drags didn't fall (22.0% of moves in 0.2, 25.9% in
  0.3).
- **Standing goal:** a model that reaches naneinf ([`naneinf/`](naneinf/README.md)). Proving one
  waits for a stronger base model (chud, 2026-11-04: a base mixture of experts; that line is now v4b). Until then, `naneinf/scan.py` looks for
  seeds with an early Perkeo, on the game itself; it hasn't run on the game yet.

## The research question

The idea and the framing are chud's. The bot is the agent under study; Balatro is the domain.

**Hume's is-ought gap.** Facts alone don't settle what an agent ought to do. To exhibit that,
hold the facts fixed (the pinned game, the same seeds, the same sim) and vary only the ought (the
reward). Whatever then changes in what the bot does came from the reward, not from the facts.
Section C of [`docs/hypotheses-general.md`](docs/hypotheses-general.md) registers that experiment (section G, a
first version, was closed unrun).

**Oughts are objects, like jokers.** A player buys a joker without knowing whether it will win
the run, and buys anyway. An agent holds an ought the same way, and revises by its losses. Humans
and models are grouped here at the level of behavior, as agents whose weights are revised by
prediction error, not as the same mechanism.

The project's three aims, in chud's order
([full text](docs/hypotheses-general.md#added-2026-11-02-2305-cst-the-projects-aims-stated-by-chud)):
1. Exhibit Hume's is-ought gap (primary).
2. Prove a naneinf seed: a route that replays to a hand whose score is no longer a finite number.
3. Generalize to 100% completion from a new save.

## Repository map

| Folder | What's in it |
|---|---|
| [`harness/`](harness/) | The headless game: LÖVE stand-in, boot (unmodded and modded), the replay validator, the playable environment and its checks |
| [`train/`](train/) | The model, the PPO trainer (split by capability), the launcher (`launch.sh`), publishing and verifying wins, exporting weights, playing a seed |
| [`configs/`](configs/) | One file per run: its settings, start and pre-flight checks |
| [`wins/`](wins/README.md) | Seeds proven beatable, with every route |
| [`weights/`](weights/README.md) | Released model weights |
| [`docs/`](docs/) | [Training runs, one page each](docs/runs/README.md), [closed hypotheses](docs/closed_hypotheses.md), [pre-registered hypotheses and results](docs/hypotheses.md), [model card](docs/models.md), [model design](docs/model_v1.md), [won-game analysis](docs/won_games_analysis.md), [reproduction guide](docs/REPRODUCE.md) |
| [`analysis/`](analysis/README.md) | Scripts behind the won-game analysis |
| [`naneinf/`](naneinf/README.md) | The naneinf goal, and community naneinf seeds checked on this game version |
| [`tools/`](tools/) | Windows PowerShell: one controller (`balatro.ps1`), the watcher, game bundle, update lock, WSL setup; `check_repo.py` |
| [`tests/`](tests/) | Logged real-game runs (gameplay data only) and the check scripts |
| [`CHANGELOG.md`](CHANGELOG.md) | Every change, dated |
| [`ARCHIVE.md`](ARCHIVE.md) | Past runs' code, by tag |
| [`AGENTS.md`](AGENTS.md) | The map for coding agents (`CLAUDE.md` imports it) |

**To set it up from your own Steam copy**, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md): the
full dependency list, the mods (only for logging your own games), and `tools\balatro.ps1`, one
PowerShell controller for every step.

## Why run the real game
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
wait on its layout), so every cut has to leave both replays at zero mismatches. This took
`replay.py` from about 165 ms to about 60 ms per action (the first measurement, on the modded
game; current figures are under [Harness validation](#harness-validation)).

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

## Harness validation
- **Validated on 2 logged runs**, Windows Steam 1.0.1o with Lovely 0.10.0, Steamodded
  26.1002.0 and Antelytics 1.3.0-alpha, with **0 mismatches**:
  - Tutorial run, Red Deck, White Stake: all 322 decisions. Covers packs, Death, Hanged Man, Cryptid copies, DNA, Sixth Sense, Buy & Use, vouchers, rerolls and selling. The sim wins at ante 9, same as the real run.
  - Ghost Deck, Purple Stake: 10 decisions, including Steamodded's own RNG streams.
- Not yet covered: challenges, Erratic Deck, seeded runs, quit-and-continue runs, Mac/mobile.
- Speed: ~12 ms per step (controls mode) to ~35 ms per move (ordered moves, with the encoder),
  visuals stripped. Training runs 14 sims in parallel.
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
| PyTorch | 2.15.0+cu132 (training; CUDA 13.2, cuDNN 9.26.0) | `requirements.txt` |
| numpy | 2.5.4 (training) | `requirements.txt` |
| NVIDIA driver | 595.79 (Linux) / 32.0.15.9579 (Windows) | |
| WSL | 2.6.3.0, kernel 6.6.87.2, Ubuntu 24.04.5 LTS | |
| Training machine | ASUS Vivobook Pro 15 (N6506CU): Core Ultra 9 285H, 24 GB, RTX 4050 Laptop 6 GB, Windows 11 Home 26H2 (build 26300.9711) | recorded 2026-11-04; details and dated changes in [`docs/REPRODUCE.md`](docs/REPRODUCE.md#the-training-machine) |
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
env = BalatroEnv("./local/source")            # mode="controls" (default), "subsets" or "ordered"
env.reset(seed="ABC123", back="b_red", stake=1)
while not env.done:
    acts = env.legal_actions()                # [(type, i, j, cards), ...]
    env.step(acts[0])
```
**Action modes, same game:**
- `controls` (default): the game's own controls. `toggle` a hand card, `move_hand` / `move_joker` (drag from slot i to j), `sort_rank` / `sort_suit`, then `play`, `discard`, `use_consumable` or `pick` with whatever is selected. Played cards score left to right in hand order, as in the game.
- `subsets`: the selection is part of the action (`play` with cards (1,3,5)). Fewer steps, ~200–2000 actions per hand.
- `ordered` (what models 0.2 and 0.3 play): `subsets` plus what card order can change. That covers which scoring card leads, which card Death copies, and boss rerolls (Director's Cut / Retcon). `ordered-capped` (0.3) also stops long chains of joker drags.
- All modes: `select_blind`, `skip_blind`, `end_shop`, `reroll`, `buy`, `buy_and_use`, `redeem`, `open_pack`, `pick`, `skip_pack`, `sell_joker`, `sell_consumable`, `cash_out`, `move_joker`.
- Legality is the game's own: the checks its buttons use (`can_play`, `can_discard`, `can_buy`, `can_redeem`, `can_open`, `can_reroll`, `can_skip_booster`, `can_select_card`, `Card:can_use_consumeable`, `Card:can_sell_card`, `check_for_buy_space`, Cerulean Bell's forced selection, sort buttons only when they'd change something). Actions go through the same `G.FUNCS` calls `replay.py` validates.
- Not offered yet: endless mode after a win (needed for the naneinf goal).

**Observation = what a player can see.** Face-up cards; face-down cards (The Fish, House, Wheel, Mark; jokers under Amber Acorn) masked; the deck view as sets (remaining and full deck, never draw order); joker and card values shown on hover; HUD, blind and its effect, shop and pack contents, tags, vouchers, hand levels, the round's Idol / Mail-In / Ancient / Castle targets. Never RNG state.

**Checks** (`tests/run_env_checks.sh <source dir>`):
- `check_env.py --mode controls`: drives each fixture entirely through env actions (joker drags as `move_joker`, hand order as `move_hand`, selections as `toggle`); every action must be legal when taken and state must match the log. Both fixtures: 256 decisions through 902 env actions, 0 mismatches.
- `check_env.py --mode subsets`: every logged decision is in the legal set. 256/256.
- `fuzz.py --mode controls|subsets`: plays many runs; fails on any offered action that crashes, never settles or changes nothing, and on any observation that leaks seed/RNG state or a face-down card. No failures.
- Speed (unmodded, visuals stripped): ~12 ms per step in controls mode, ~33 ms in subsets mode; a runtime boots in ~0.1 s and `reset()` reuses it.

The unmodded game also replays both fixtures with every vanilla value matching (money, hand, deck order, score, jokers, consumables, tags, hand levels, vanilla RNG streams). The only differences are Steamodded's own streams, which an unmodded game doesn't have.

## License and contact
[MIT](LICENSE), for **our tooling only**. Use it, fork it, build on it. Balatro itself remains
LocalThunk's and Playstack's.

Made by chud ([bapbapbapbap](https://github.com/bapbapbapbap)) with Claude (Anthropic).
chud is the author and answers for any errors; the code Claude wrote is credited to Claude in
each commit.

Questions and collaboration: jacobhadji05@gmail.com
