# Proposal (not registered): giving the v6 student the teacher's hand-play computation

Claude's, 2026-11-19, for chud to decide on. Used only if the roadmap's rule 3 fires (stage 1's agreement on hand
decisions below 65%: the student can't copy hand play from moves alone). Nothing here is built.

## What the student lacks

At a hand decision, Balatro Lab's bot computes before it chooses; our network chooses in one pass. Its hand-play
machinery (balatro-lab `strategy.py`, `search.py`; test K measures each part):
- **Discards scored on sampled draws** (`DRAWS` 6, the leaders on 24 more): what each discard's next hand is
  likely to be, from the unseen cards.
- **Power** (`HANDS` 12): the build's typical best hand over sample hands from the deck, behind every valuation.
- **Round search** (`ROUND_SEARCH`, boss rounds only): the bot's pick and a few others each played to the round's
  end 6-18 times on the same redraws; the one that wins most is taken.

Test R (2026-11-15) put the teacher's edge in hand play (+3.3 blinds/game; its shop, pack and blind choices did
worse than v4b's). v5 gave our network every play's chips now (play values) and its best eval did not move: the
current hand's score is not what the student is missing; what comes after the draw is.

All three use only what a player can see: the unseen cards as a set (the deck view), never their order. So
they are facts the game's state implies, not values; the student can have them without taking the teacher's oughts.

## Options

**S1. Discard values from sampled draws (an input, like play values).** For each legal discard: the mean, over n
draws sampled from the unseen cards with a fixed RNG, of the best hand's score after the draw, computed with the
game's own scoring (`build_phi.lua` already snapshots and scores hands with the jokers for play values). Added as
action-row features; the network still decides.
- Cost: build ~1 day; sim speed maybe 1.5-3x slower on hand decisions (n=6 draws x up to ~200 discards; capped to
  the top discards by a cheap filter, as the teacher does). Training needs teacher data with the new inputs (the
  saved games hold observations, not game states): the same seeds played again, ~6 h for 1,800 games.
- Fits if test K says K3 (draws) or K2 (power) costs most.

**S2. Round search for the student (decision-time, boss rounds).** The student's top few hand moves each played to
the round's end by the student itself, on the same sampled redraws, in a snapshot of the game state; take the one
that wins most (the teacher's own scheme). Needs a full round snapshot/restore in the harness (build_phi's covers
scoring, not a whole round's events), and m x k round play-outs per searched decision.
- Cost: build 2-3 days (snapshot/restore of a round must be checked against the untouched game like diff_check);
  play much slower in boss rounds (the teacher pays 10-60 s a searched round).
- Fits if K1 (search) costs most. Search improves the student beyond what it learned, and it can later be
  distilled back into the network (the student trained on its own searched moves): an improvement on the student,
  never on the teacher.

**S3. Neither: more imitation (DAgger) only.** Cheapest; fits if agreement is limited by drift, not representation.

## Claude's recommendation

S1 first if K2/K3 dominate (cheap, an input, the reward still decides), S2 only if K1 dominates and S1 leaves a
gap. Either is registered with predictions before its code, as always. chud's calls: whether the student may search
at all (it changes what "the student's strategy" means for aim 1: a searching student's choices are partly
computed at the moment, not learned), and which option.
