# System card: moe2-pins0 (mixture of experts)

`weights/balatro-bot-moe2-pins0.pt`, released 2026-11-05. The weights the second mixture-of-experts
run (`runs/moe2`) had at round 1042, the first round at which every pin had faded to 0. Full
results: [moe2_evaluation.md](moe2_evaluation.md). Design: `hypotheses.md` section I.

It's a system card and not a model card because what plays is several networks plus a fixed
routing rule. Which network moves depends on the game.

| | |
|---|---|
| Version | `moe2-pins0` (not part of the 0.x / 1.0 line) |
| Parameters | 2,281,742 (shared trunk + 7 heads) |
| Game | Balatro 1.0.1o-FULL, unmodded; Red Deck, White Stake |
| Reward | blinds beaten (`blinds`), potential-based shaping |
| Training | 100,032 games at this snapshot (round 1042), from model 0.2's weights |
| Eval (600 fixed seeds) | **12.94 blinds per game; 9 wins, 13 near wins** (1.5% won, 3.7% reached the final boss) |
| Status | Research snapshot. Full House not learned; not yet compared with model 1.0 |

## The system
- **Trunk:** the transformer of models 0.2–1.0 (d=192, 4 layers, 6 heads) over what a player sees:
  cards face up, the deck as a set (never its order), jokers and their values, the HUD, shop and
  packs. Never the game's random state. Encoding 3, action set `ordered-capped`. Each head scores
  every legal move and has its own value estimate.
- **Early head** (antes 1–3): plays from the start until the handoff.
- **Six experts:** Pair, Two Pair, Three of a Kind, Straight, Flush, Full House. One takes over at
  the **handoff**, the first moment the run is in ante 4 with the ante-3 boss beaten, and plays
  the rest of the game.
- **Router** (a fixed rule, not learned): for each expert's hand, (plays of it in antes 1–3 + 1) /
  (the early head's average + 1), and the highest wins. Full House needs at least 2 plays to be
  picked. The averages are stored in the file (`base_rate`).

## Training
- **Start:** every head and the trunk from model 0.2's weights.
- **RL:** PPO, trained jointly. Every move trains the trunk; a head trains only on its own moves.
  The early head's return ends at the handoff (`--moe-cut 1`). lr 1e-5, halved to 5e-6 at round
  611 by the update guard. γ 0.995, λ 0.95, clip 0.2, 3 epochs, minibatch 64, entropy 0.01,
  target KL 0.04. 5% hand-type exploration in training games. 30% of training handoffs go to a
  random expert.
- **Pins** (chud's design): for part of training, a head's plays were restricted toward one hand
  type whenever a play of that type was on offer. Each expert was pinned to its own hand (Pair at
  half rate). The early head was pinned to Full House or higher. All pins faded linearly to 0
  over 100,000 games. **This snapshot is the first with every pin at 0**, and the file holds the
  pin scales (all 0), so it plays unrestricted.
- **Seeds:** at least half of each round's games on fresh seeds. The rest retried seeds the bot
  had come close on.
- **Run:** 2026-11-04 13:33 to 2026-11-05 13:42 CST, about 24 hours across 11 launches, on repo
  code from PR #77 to PR #98 (commit at this snapshot: `3d5b12f`). 12 sim workers, about 6,000
  games an hour. One laptop: Intel Core Ultra 9 285H, RTX 40-series laptop GPU (6 GB), about
  23 GB RAM, Windows 11 / WSL2 Ubuntu 24.04, Python 3.12.3, PyTorch 2.15.0+cu132.
- `Balatro.exe` SHA-256 `0D75FE16…DE93F47`. Vocabulary SHA-256 `ac5cc301…dd43cb`.

## Evaluation
600 fixed seeds, moves sampled with a fixed random stream per seed. Outcomes are the game's own.

| | This snapshot (round 1042) | Run's best |
|---|---|---|
| Blinds beaten per game | 12.94 | 13.08 (round 1020) |
| Wins | 9 | 14 (round 1040) |
| Reached the final boss | 22 | 29 (round 1040) |

- Where games end: 159 of 600 by ante 3, 352 in antes 4–6, 22 at the final boss.
- At about 1.5% wins, an eval's win count moves by about ±3 from chance alone.
- These numbers can't be ranked against models 0.1–1.0. Those were evaluated on 200 seeds with a
  win test that counted Ante 8 boss losses as wins (corrected in CHANGELOG 0.13.2), and 1.0 is
  being re-measured.

## What it does and doesn't do (pins-zero eval)
- **Routing:** 442 of 600 games reach the handoff. Straight 119, Three of a Kind 107, Two Pair 100,
  Flush 85, Pair 31, **Full House 0**.
- **Two Pair's expert** plays Two Pair at 1,257 of the 1,264 decisions where one is on offer. Straight's
  expert plays a Straight at 78% of chances, Flush's 42%, Three of a Kind's 37%, Pair's 7%.
- **Full House:** the early head had a Full House or higher on offer at 1,005 decisions and played
  one once. With no early Full Houses, the router never picks the Full House expert. The completion
  criterion chud set for the MoE (it learns Full House) is **not met**
  ([scored](hypotheses.md#added-2026-11-05-1540-cst-moe2-at-pins-zero-results)).
- **Final boss:** across all of moe2's games, each expert converted 15.5–20.7% of its final-boss
  fights.

## Known limitations
- **Not shown to beat a single network.** I1 (MoE vs model 1.0) is not registered or run.
- **The experts don't differ.** From the same positions (`analysis/paired_experts.py`, 2,000 seeds),
  all six play the same hand mix as each other and as the early head, within about a point, and no
  expert wins detectably more. The rates above describe the games each expert was routed, not the
  expert. The handoff itself helps: any expert adds about 1 blind per game over the early head playing
  on (`docs/moe2_evaluation.md`).
- **Win rate has not clearly risen** past the run's early peak. Fresh-seed wins were 1.12% in rounds
  100–199 and 1.13% at the end. Blinds per game did rise.
- **Early losses:** 56 of 600 eval games end by ante 1, which a competent player essentially never
  does. Some of this may be sampling (evals sample from a fairly flat policy, entropy about 2); a
  low-temperature eval hasn't been run.
- **An unexplained training statistic:** from round 200 to 600 the gap between the policy that
  played and the one being updated rose from 0.01 to 0.19 and stayed there. It may be a training
  fault or a measurement artifact ([evaluation](moe2_evaluation.md#2-a-training-statistic-changed-sharply-unexplained)).
- **Path dependence:** the weights carry the pins' history. What it does at pins 0 isn't what the
  reward alone would have produced from 0.2.
- **Scope:** Red Deck, White Stake, one game version. Nothing about other decks or stakes.

## Intended use
- The is-ought research in `docs/hypotheses.md`: a recorded, reproducible snapshot of what a
  forced strategy (the pins) leaves behind once it's removed.
- Producing and replaying beaten seeds (`wins/`, `docs/proof.md`).
- Not a claim about strong play. A 1.5% win rate at White Stake is far below a good human player.

## Use it
Set up the harness from your own copy of the game ([REPRODUCE.md](REPRODUCE.md)), then:
```bash
cd ~/balatro/balatro-headless/train
python3 play.py ../weights/balatro-bot-moe2-pins0.pt --src ~/balatro/local/source --seed CZIZSE25 --save route.jsonl
python3 verify_games.py --full --src ~/balatro/local/source route.jsonl
```
`play.py` runs mixture-of-experts weights through the trainer's eval code, so a seed plays the same
game the run's evals played. `--sample-seed` has no effect for these weights, and each move is
labeled with the head that made it. Checked 2026-11-05 on chud's machine: CZIZSE25 (a win, Two Pair
expert) and 3QS7GI3C (a final-boss loss, Flush expert) reproduced the pins-zero eval's games move
for move.

Load in Python with `torch.load(path, weights_only=True)`. `arch["moe"]` is true, and the state dict
loads into `train/model.py`'s `MoE`. The run's settings are in `trained.provenance.config`, with
local paths removed. That is where `play.py` gets the router's settings.

## File

| File | Size | SHA-256 |
|---|---|---|
| `balatro-bot-moe2-pins0.pt` | 9,179,878 bytes | `1dc76af9cc45be81a74d8628f11ca30c07b46a5cc59ac185e78a35c8f00fde12` |

Exported with `train/export_weights.py` from `runs/moe2/pins0.pt`. It holds the network, its buffers
(router averages, pin scales at 0, expert game counts), vocabulary, architecture and provenance. No
optimizer, no training state and no game code. Two metadata fields need reading with care:
- `trained.best_eval_blinds` (13.08) is the run's best eval up to this round, from round 1020. It is
  not this snapshot's score, which is in `note`.
- `trained.reward` is empty. The reward is in `trained.provenance.config.reward` (`blinds`).

Released under the repository's MIT license. Balatro is LocalThunk's and Playstack's; you need your
own copy to run anything here.
