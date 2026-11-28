# The student's size, for an AnteZero teacher: what is known (no decision here)

Claude's, reviewed by chud. Written 2026-11-27 night for chud's decision; the decision goes in the student's own
registration, before its code.

## The two networks
| | AnteZero v0.2.0 (teacher) | ours, v6 (v6e → v6dt → v6g) |
|---|---|---|
| parameters | **6.32M** (counted from the checkpoint) | **3.38M** (counted from `train/model.py` with v6g's settings) |
| shape | one network: width 256, 6 layers, 8 heads, feed-forward x4 | two networks (`phase_split`), one for hands, one for everything else; each width 192, 4 layers, 6 heads |
| per decision | the whole 6.3M | about half of 3.38M (one of the two networks) |
| inputs | its own tokens (`sim::obs`): card effects as token features, candidate plays, joker-order gains | encoding 2 + offer values + play values (what a play scores against what the blind needs) |
| actions | factored: an op, then a pointer over hand cards, then a destination | a list of whole moves, scored one by one (ordered-capped: plays name a lead card) |

A correction: in chat I said ours was ~250k parameters, about 25 times smaller. That figure was v4b's. v6 is 3.38M,
so the teacher is under 2x the size, and about 4x the network that decides any one move.

Same sizes for reference, our architecture with both networks: width 256 / 6 layers / 8 heads = 7.96M; width 384 /
8 layers = 22.3M.

## What our own history says
- v6e learned from Balatro Lab by imitation at v6's size: 68.0% top-1 agreement overall, **40.7% on hand
  decisions**, 13.0% wins against the teacher's 68.7% in our game. The loss was still falling when the 8 passes
  ran out. The registration's reading then: the gap was in hand play, and Balatro Lab's hand play is computation
  (power over sampled hands, draws, search), which a network can't copy by watching (test K: those computations are
  worth about 20 + 8 + 3 points).
- That limit is mostly gone with AnteZero: its teacher is itself a network with no search (no search in v0.2.0's
  greedy play), so there is a function of the state for the student to learn, not a search.
- What still limits it is inputs more than size: the student can only copy what it can see. Where AnteZero computes
  something into its tokens that our encoding lacks, the student has to learn it from data or miss it.

## What size changes, as far as known
- Imitation time grows with parameters x decisions. The teacher data costs nothing to make bigger (seconds per game
  in its sim, ~8 s per game to label).
- PPO: our rollouts are the Lua harness (12-66 ms per action); how much of a training round is the network rather
  than the game is not measured in our logs. Unknown until measured.
- Not known for our setup: whether 3.38M can hold a 6.32M teacher's play. Students half a teacher's size usually
  keep most of it in other fields, but that is no measurement here.

## A cheap way to decide with data
An imitation-only pilot on the laptop, no PPO: the same teacher games (say 3,000, from `teach_az.py`), two students,
v6's size and a teacher-sized one (width 256, 6 layers), the same passes. Read held-out agreement (hand decisions
apart) and a 600-game eval of each. If the small one's agreement is within a few points of the big one's, size is
not the limit and the cheaper network goes forward. That pilot is itself a registration.
