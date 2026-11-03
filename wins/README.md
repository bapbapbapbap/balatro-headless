# Beaten seeds

<!-- tally:start -->
**Seeds proven beatable: 1,549** (134 by model 0.1, 673 by model 0.2, 742 by model 0.3; 1,549 on Red Deck, White Stake). Each one is a route in [`routes.jsonl`](routes.jsonl) that was replayed move for move to the same win before it was listed. This is the project's production count, updated each time wins are published.

**naneinf seeds proven: 0.** A seed counts once its route in [`../naneinf/proven.jsonl`](../naneinf/proven.jsonl) replays move for move to a played hand with non-finite chips (see `naneinf/README.md`).

**Highest stake beaten: 1 of 8 (White Stake).**
<!-- tally:end -->

Each line of `routes.jsonl` is a run of Balatro won by the bot, with everything needed to replay it:
the seed, the deck and stake, and every move it made. `beaten_seeds.csv` lists the same seeds.

A seed listed here is proven winnable **by fair play**: the bot only ever saw what a player sees
(cards face up, the deck as a set, never the draw order or the game's random state) and chose each
move from that. The route is the proof: the game is deterministic, so the same moves from the same
seed produce the same run, ending in a win.

## Setup the seeds were won on
- Balatro **1.0.1o-FULL**, `Balatro.exe` SHA-256
  `0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47`, **unmodded**.
- A profile with **everything unlocked** (which jokers, tags and vouchers can appear depends on
  unlocks, so a fresh profile plays the same seed differently).
- The deck and stake on each line (so far: Red Deck, White Stake).

A win is the game's own: `G.GAME.won`, set when the Ante 8 Boss Blind is beaten.

## Independent check on the untouched game (2026-11-03)
The 850 seeds published on 2026-11-02 (every seed after the first 121) were published as
training-verified: each win had been replayed exactly in the training sim, which strips visuals.
They were then all replayed again on the **untouched** game (`--full`, nothing stripped), move by
move, in the cloud. **850 of 850 reproduced the same win: the same moves, the same number of moves,
and the same final state.** The first 121 were replayed on the untouched game when they were
published.

The 578 seeds published on 2026-11-03 (all model 0.3's) are training-verified so far: each win was
replayed exactly in the training sim. Their replay on the untouched game is still to do
(`tools\balatro.ps1 verify`, or the command below).

## Checking a route
With the game source extracted from your own copy (see [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md); no game code is in
this repository):

    python3 train/verify_games.py --src <game source> wins/routes.jsonl --wins-only --full

replays every route on the untouched game and checks every move, the number of moves, the final
state and the win. `--full` runs the game with nothing stripped.

## Route format
- `seed`, `back`, `stake`: the run to start.
- `actions` / `enc`: which action list the moves index (absent means `subsets` / `1`, the v1 setup) (`subsets`: one move per play/discard with
  the selected hand positions; `ordered`: the same plus the lead card, Death's direction and boss
  rerolls; `ordered-capped`: `ordered` with chains of joker drags capped) and the encoding version.
- `acts`: every move as an index into the legal moves at that point; `desc`: the same moves as text,
  e.g. `select_blind`, `play 1,3,5` (hand positions, left to right), `buy 2`, `use_consumable 1 2,4`.
  The replay checks both, so a route can't silently drift.
- `n` (moves), `final` (hash of the final state), `progress` (blinds, bosses, won).
- How it was found: `mode` (training or eval game), `iteration` (training round), `retry` (a seed
  retried after an earlier loss), `source` (`v1/wins.jsonl` = the main v1 run; `low`/`high` and
  `off`/`on` = the arms of the learning-rate and self-imitation A/B tests, which played from v1's
  weights), `commit` (the trainer's code version).

Multiple wins on one seed: the shortest is kept.

Released model weights are in [`../weights/`](../weights/README.md).
