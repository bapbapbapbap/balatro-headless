# Beaten seeds

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

## Checking a route
With the game source extracted from your own copy (see `../harness/README.md`; no game code is in
this repository):

    python3 train/verify_games.py --src <game source> wins/routes.jsonl --wins-only --full

replays every route on the untouched game and checks every move, the number of moves, the final
state and the win. `--full` runs the game with nothing stripped.

## Route format
- `seed`, `back`, `stake`: the run to start.
- `actions` / `enc`: which action list the moves index (`subsets`: one move per play/discard with
  the selected hand positions; `ordered`: the same plus the lead card, Death's direction and boss
  rerolls) and the encoding version.
- `acts`: every move as an index into the legal moves at that point; `desc`: the same moves as text,
  e.g. `select_blind`, `play 1,3,5` (hand positions, left to right), `buy 2`, `use_consumable 1 2,4`.
  The replay checks both, so a route can't silently drift.
- `n` (moves), `final` (hash of the final state), `progress` (blinds, bosses, won).
- How it was found: `mode` (training or eval game), `iteration` (training round), `retry` (a seed
  retried after an earlier loss), `source`.

Multiple wins on one seed: the shortest is kept.
