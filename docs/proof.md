# Balatro can be beaten: a proof by construction

> **Corrected 2026-11-03.** The first version of this proof used seed `4XNRUA21`. That run lost:
> it ran out of hands on the Ante 8 boss (Verdant Leaf, 82,500 of 100,000). Balatro sets
> `G.GAME.won` at the end of the Ante 8 boss round whether or not the boss is beaten, and the
> harness read that flag alone. The proof below uses the first published run that is a win by the
> game's own test. See the CHANGELOG (0.13.2) and `docs/hypotheses.md` (correction, 2026-11-03).

## Claim
**There is a run of Balatro that can be won.** Precisely: on Balatro 1.0.1o-FULL (`Balatro.exe`
SHA-256 `0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47`), unmodded, on a profile
with everything unlocked, Red Deck, White Stake, the run started from seed **`Y45RM46C`** is won by
the sequence of 235 moves recorded for that seed in [`../wins/routes.jsonl`](../wins/routes.jsonl).

## Proof
1. **The game is deterministic given the seed and the moves.** Every random draw in a run comes from
   the game's own pseudorandom streams, seeded from the run's seed. The same seed and the same moves
   give the same run. Every route the project has published replays move for move to the same final
   state, on the untouched game as well as in the training sim.
2. **The route.** Seed `Y45RM46C`, Red Deck, White Stake: 235 moves, from `select_blind`,
   `play 3,4,6,7,8`, `play 2,3,5,6`, `discard 5,7` … to `play 1,2,5,6,8`. The final state's hash
   is `57d6a5d83d721d3b`.
3. **The win is the game's own.** Replayed from the seed, the moves beat the Ante 8 Boss Blind
   (24 blinds, 8 bosses): the game sets `G.GAME.won`, stays out of `GAME_OVER`, and calls its own
   `win_game()`, which records the win on the profile. `G.GAME.won` alone is not the test: the
   game sets it on a loss to the Ante 8 boss too.
4. **It was checked on the untouched game,** with nothing stripped, on 2026-11-03 with the
   corrected test: the same moves, the same number of moves, the same final state, and a win.

So a winning run exists, and Balatro can be beaten. ∎

## Where it came from
`Y45RM46C` is the first published win by the game's own test: model 0.1, training round 351, on a
retried seed, in the v1 run on 2026-11-01. Every published route from before it (including
`4XNRUA21`, round 329) was a loss on the Ante 8 boss. Publishing kept one route per seed (the
shortest), so the training files may still hold an earlier real win on some seed; replaying every
saved win in them (pending) settles that. The 10 "wins" counted before saving every game started
weren't recorded and can't be checked. The move list was chosen by
the bot from what a player sees (cards face up, the deck as a set, never the draw order or the
game's random state), so it is a fair-play win, not a search over hidden state.

## What it does not show
- That **every** seed can be won. Some may be unwinnable. 435 seeds are proven so far, all of
  them Red Deck, White Stake. Another 1,114 reached the Ante 8 boss and lost it
  ([`../wins/near_misses.jsonl`](../wins/near_misses.jsonl)).
- Anything about other decks, stakes or challenges, or about a fresh profile. Which jokers, tags
  and vouchers can appear depends on unlocks, so the same seed plays differently on a new save.
  The ledger below tracks those, one first seed per item, as wins come in.

## Checking it yourself
With the game source extracted from your own copy (see [`REPRODUCE.md`](REPRODUCE.md); no game
code is in this repository):

    grep '"seed": "Y45RM46C"' wins/routes.jsonl > first_win.jsonl
    python3 train/verify_games.py --src <game source> first_win.jsonl --wins-only --full

## Proof that Balatro can be 100%ed: a list of runs to replay
The proof is **one save**: a list of runs, each a seed with its deck, stake or challenge and every
move, in the order played, starting from a new profile. Losses are in the list too, because they
change the profile as well. Replaying the list in order on a new profile must reproduce every
run exactly, start each run only when the game would let a player start it, and end with the
game's own record showing 100%: a Gold Stake win with every joker, Gold Stake on every deck, and
every challenge beaten.

- The list is written by `tools/play_save.py`, one line per run, in the same format as
  `wins/routes.jsonl`.
- The check is `python3 train/verify_games.py --src <game source> --save <the list>`. Its last
  line is `SAVE 100% PROVEN` only when all three hold.
- Status: no save has been played yet, so there is no list yet.

The published wins below can't be part of the list: they were played on a profile with
everything unlocked, and on a new save the same seed plays differently.

## The first seed for each part of a full save (all-unlocked profile)
A full save, as chud defines it (`hypotheses.md`, roadmap): a Gold Stake win with every joker,
Gold Stake on every deck, and every challenge. Each entry below is the first seed that achieved
it, so each is a proof by construction of its own: that win is possible. Generated by
`tools/save_progress.py`. What counts is the game's own record: each published route is replayed
once with an empty profile, and what the game credits that win with (jokers, deck, challenge, at
which stake) is read from the profile afterwards. Training runs never kept this record, because
the game records nothing for seeded runs.

<!-- proof:start -->
**Save progress: not tallied yet.** Run `python3 tools/save_progress.py --src <game source>` to
fill this in. It replays the published wins once and reads what the game credits each one.

Known before the tally: Red Deck, White Stake: first seed `Y45RM46C`.
<!-- proof:end -->
