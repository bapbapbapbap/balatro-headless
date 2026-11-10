# Hypotheses: Model v4

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-05 ~23:01 CST: model v4, registered before any of its code (section J)

chud (22:58): "it'd be wasteful not to do the best approach when we finally looked." v4 is the
strongest bot buildable from what this project and others have learned. It replaces v3 as the
target; v3 (registered above) may still run as a baseline, and its A/B result, if it exists, decides
v4's discount clock. Nothing below is built yet.

### What others found (looked up 2026-11-05 ~22:50 CST; Red Deck, White Stake unless noted)
| Approach | Win rate | Source |
|---|---|---|
| Pure PPO, one policy (ours) | ~2% | this repo |
| Pure PPO, one policy (taggarttufte/balatro-rl V7) | 2.35% peak, "plateaus at Green+Space lock-in" | github.com/taggarttufte/balatro-rl |
| Two-policy PPO, simplified sim (arXiv 2610.39049) | no wins; mean max ante ~3 | arxiv.org/abs/2610.39049 |
| Hand-written heuristics | ~5% | balatro-gym "strategist" (100 seeds); Optilatro `heuristic_v10` (300 seeds) |
| Planner: each shop decision scored by cloning the state and measuring the board's hand-scoring power | 20% (47/240) | github.com/roshantaneja/balatro-gym |
| Separate shop and blind agents, multi-agent PPO with routed credit (GIEWEV; Reddit claim, partial game) | ~30% sim, ~25% real game | cited as [3] in arXiv 2610.39049 |
| Skilled humans | "95–99%" (forum opinion, not a measurement) | Steam discussions |

The other projects run on reimplementations, some partial; ours runs on the real game. Read
together: **every approach that scores the shop directly beats every single-policy PPO run by about
10x.** That independently supports v3's diagnosis (the build decides the run; a single long-horizon
policy can't credit it) and says v3's fix, the clock, is the weakest of the known fixes.

### What changes (v4)
1. **Build-strength potential.** A second potential-based shaping term, phi_build(s), how strong the
   current build is against the blinds ahead, computed from game state with the game's own scoring
   code (reference hands scored with the current jokers, hand levels and deck; the exact definition
   is fixed and recorded here before training). Requirements: a function of the state only (so
   Ng et al. 1999 holds and the optimal policy under `blinds` is unchanged: the ought stays
   `blinds`), bounded, using the step's own discount, and computed only from what a player can see.
   It gives the shop a direct signal for what each purchase did to the build. It encodes no rule:
   the bot still decides what to buy.
2. **Separate shop and table networks, each with its own critic.** Decisions in a blind (play,
   discard, use while selecting a hand) go to one network; shop, packs, blind select and cash-out go to
   the other. Built on the MoE machinery, routed by phase instead of by hand. Reason: one critic can't
   judge both "is this hand good" and "is this build good".
3. **Backward curriculum.** A share of training games (30%) start mid-run: a saved game's moves are
   replayed to the start of ante 4, 5 or 6 (uniform), then the bot plays on. Sources: this run's own
   games and the published routes. These games train but are never counted in training stats or
   evals. Reason: the late game gets practice while the bot still dies early (and aim 1 needs a
   late-game bot; see below).
4. **Shop entropy schedule:** the shop network's entropy bonus starts at 0.02 and decays to 0.01
   (arXiv 2610.39049 reports it prevents shop collapse into "reroll forever" or "leave at once").
5. **Discount clock:** the v3 A/B winner if it has been run; the blind clock otherwise.

Held: 1.0's size per network, encoding 2, ordered-capped actions, reward `blinds`, seed retries,
lr 1e-4, from random weights, no behavior cloning, Red Deck, White Stake.

### Build order (each part checked on the real game before the next)
Backward curriculum (the replayed start must equal the saved game's state at that step), then the
phase split (every decision routed to the right network; saved games replay exactly), then the
potential (it must telescope to 0 over every published win, never read hidden information, and
take the same value on identical states).

### The test
2,000 seeds from tag `v4-test` (disjoint from the eval 600, `paired-v1`, `v3-ab`, `v3-test`);
v4's `best.pt` vs `pins0.pt`, and vs v3's kept arm if it exists, each playing each seed once,
sampled, per-seed RNG. Win = the ante-8 boss beaten.

### Predictions (Claude)
- **W1. v4 wins ≥ 4% of `v4-test`, more than `pins0.pt` (one-sided exact McNemar, p < 0.05). (70%)**
- **W2. v4 wins ≥ 10%. (35%)** Others' 20–25% came on easier or partial simulators; 10% on the real
  game would be a large step.
- **W3. Mechanism: v4 reaches ante 7 in more games than `pins0.pt` and dies less in antes 4–6
  (paired, p < 0.05). (70%)**
No attribution to single changes is claimed; four change at once.

### Aim 1 needs a late-game bot (recorded for the Hume run)
`blinds`, `antes` and `win` reward nearly the same things until the late game. A bot that dies by
ante 5 gets almost the same signal under all three, so its arms would barely differ, and "no
difference" would mean the bot was too weak to face the tradeoffs, not that the facts fixed the ought.
**Precondition for running aim 1's experiment:** the starting weights reach ante 7 in at least 25% of
eval games (pins0.pt: 22 of 600 reached the final boss, so far below). The threshold is a proposal;
chud confirms it with the Hume run's statistics.

### Aims 2 and 3 are existence proofs: a separate search track (direction only; nothing predicted)
A naneinf seed and a 100% save are proved by routes that replay on the untouched game. Seeds are
chosen and the game is deterministic, so routes can be found by search with lookahead (branches
replayed before committing), with the policy as a prior, on CPU, independent of training. A
searched 100% save would prove that one exists, and show it, not that a player without foresight
could do it; every write-up says so (as `docs/proof.md` does). First milestone: one Gold Stake Red
Deck win found by search.


## Added 2026-11-09 ~12:00 CST: v4 amended before any of its code (section J)

The 2026-11-06 build of v4 was lost before it was pushed (the cloud session ended). Nothing of it is on
GitHub, and no v4 training game has been played. chud (11:03): rebuild with what the other projects did
that v4 left out, several changes at once ("I trust the superior results of the papers"). This amends the
v4 registration above. W1–W3 stand as registered; everything below is added before any v4 code.

### Sources re-read 2026-11-09 (what each one actually did)
- **balatro-gym planner** (github.com/roshantaneja/balatro-gym, `docs/strategy.md`, `env/planner.py`): a
  search agent, not a learner. Its board "power" is the mean best-play score over hands dealt from the run's
  own deck, on a board whose scaling jokers are projected 6 rounds ahead (a hand-set growth table). Each shop
  offer is valued by the power after buying it (or replacing the weakest joker), per dollar. Measured: about
  10% without endgame rollouts, 20% (47/240) with them; the growth-table correction was part of 10% → 17%;
  from a $100 start the same agent wins 57%.
- **arXiv 2610.39049**: blind and shop policies; the blind agent's blind-clear cash and round-won rewards are
  credited to the shop agent; shop entropy 0.02 → 0.006 (1.5M steps) → 0.001 (4M); γ 0.99. It never won a
  run (mean max ante about 3).
- **GIEWEV** (r/reinforcementlearning post, 2025, read 2026-11-09 from a saved copy chud supplied): one
  policy for blinds, one for the shop, RLlib PPO, a partial custom simulator; about 30% simulated and 25% in
  the real game (with face-down cards visible on two bosses). **No reward or credit design is described.**
  The v4 registration's "GIEWEV ... with routed credit" is corrected: routed credit is arXiv 2610.39049's.
  His main lesson was an action design that treats a card selection as a set; ours already does
  (`model.py` scores each legal play from its cards' pooled embeddings).
- **taggarttufte/balatro-rl**: a play/shop split failed by shop starvation (shop under 0.1% of steps). Not a
  risk here: in our 1,225 published wins, shop-side decisions (about 31k) outnumber plays and discards
  (about 23k).

### What changes (amending v4's list)
1. **Curriculum by full replay, not by the game's save.** As first registered: a curriculum start replays a
   saved game's moves from the seed to the start of ante 4, 5 or 6, then the bot plays on. (The lost build
   loaded the game's own save instead; five state gaps turned up in it.) A replayed start is the game itself.
2. **phi_build projects scaling jokers forward**, as the planner does: on a copy of the state, each scaling
   joker's counter is advanced by a fixed per-round rate times a fixed number of rounds before the board is
   scored. The rates are hand-set judgments (the first in phi); the table is fixed in the definitions entry
   before training. phi stays a function of the visible state only, so the shaping stays policy-invariant
   and the reward stays `blinds`.
3. **The shop network sees each offer's value.** Each buyable joker, planet and voucher action carries
   phi_build after buying it minus phi_build now (a full board: after replacing the joker whose removal costs
   least), computed on a copy of the state. An input, not a rule: the network still chooses.
4. **Credit routing, explicit.** Each network's advantages run over its own decisions only. The reward between
   two of a network's decisions is everything earned in between, including the other network's blinds (so
   blind-clear rewards reach the shop network, as arXiv 2610.39049 routes them), discounted by the blind
   clock. Shaping per network is γ^Δ·phi(next own decision) − phi(this decision), Δ = blinds in between, so
   it telescopes per network. The arXiv paper's other shop terms (interest-tier penalty, joker-count bonus)
   are not used: they are not potential-based and would change the ought.
5. **Shop entropy: the full schedule** 0.02 → 0.006 → 0.001, at the same fractions of training as
   arXiv 2610.39049 (1.5M and 4M of its steps, scaled to our games by mean decisions per game).

Held as registered: 1.0's size per network, encoding 2, ordered-capped actions, reward `blinds`, the blind
clock, seed retries, lr 1e-4, from random weights, no behavior cloning, Red Deck, White Stake.

### W4 (chud's hypothesis, 11:19): v4 at least doubles wins per compute
chud: "we need to at the very least raise wins per compute more than the added weight of all this
architecture ... lets at least double the rate of wins per compute." Two parts:
- **W4a, play cost.** On the 2,000 `v4-test` seeds, each model playing each seed once on the same machine
  and worker count, wins per sim-hour (wins / summed per-game play seconds): v4 at least 2× `pins0.pt`.
- **W4b, training cost.** Compute = summed `play_seconds` + `update_seconds` from `metrics.jsonl`, on chud's
  laptop. pins0's compute is its whole lineage (model 0.2's training plus moe2's); if part of the lineage's
  logs is missing, moe2's alone is used (which only makes the test harder for v4). v4's 600-seed eval win
  rate at the first eval where its own compute reaches pins0's must be at least 2× pins0's pins-zero eval
  (9/600).
- **Passing** (chud, 11:51: both stipulations kept, as two tiers): **W4 met** if the measured ratio is ≥ 2 and
  its one-sided 95% lower bound (paired bootstrap over seeds, 10,000 resamples; for W4b, a binomial
  bound on v4's eval wins against pins0's 9/600) is above 1; **W4 met strongly** if that lower bound is ≥ 2.
  With about 30 pins0 wins on `v4-test`, the strong tier needs v4 to measure about 2.85×.
- chud's precedence (proposed by Claude, 11:19, not objected to): if a design reaches aim 1's precondition
  but fails W4, aim 1 rules. W4 doesn't apply to the search track for aims 2–3 (existence proofs).

### Odds (Claude's, 2026-11-09; W1–W3's registered odds stand as registered)
- W1 (≥4%, beats pins0): 75%. W2 (≥10%): 40%. W3: 70%. ≥20% (10×): 15%.
- W4a met 50%, met strongly 35%. W4b met 35%, met strongly 25%.
- Why not higher: no learning-only method is known to reach 20% on the full game; the one inspectable 20%
  (the planner) doubled from 10% only with endgame rollouts, a search v4 doesn't do.


## Added 2026-11-09 ~15:25 CST: model v4's definitions, fixed before any v4 training (section J)

Written with the code (branch `v4`; `train/run_v4.sh`), before any v4 training game. W1–W4 and the odds
above stand as registered; this fixes what the amendment left to be defined, and records measured costs.

### Phase split (`--phase-split`; `model.PhaseNet`)
Two networks of 1.0's size (d 192, 4 layers, 6 heads; 3,219,844 parameters in all), each with its own
encoder, policy and critic; no weights shared. Table = every decision made in state `SELECTING_HAND`
(plays, discards, consumables used and jokers moved mid-blind); shop = every other decision (blind select,
cash-out, shop, every pack). Advantages: `train/credit.py` (routed per network, blind clock), normalized per
network. Entropy bonus: table 0.01; shop on its schedule (below).

### phi_build (`harness/build_phi.lua`; `--build-shaping 1`)
- **B, the board's power:** the mean, over 16 deals, of the best score the deal can make. A deal is hand-size
  cards (the hand size a round starts with; The Manacle's −1 is the blind's, not the build's) drawn from the
  whole deck (`G.playing_cards`) sorted by card (suit, rank, enhancement, edition, seal, ability), with a fixed
  generator, so draw order is never read and the same deck gives the same deals. For each poker hand the
  game's own `evaluate_poker_hand` finds in the deal, those cards are played and the rest held; the best
  score counts.
- **Scoring:** the steps of `G.FUNCS.evaluate_play`, calling the game's functions with the same contexts:
  hand level; each scoring card's seal and joker repetitions, its own effects and every joker's individual
  effect; held cards (Steel, Baron, Raised Fist, Shoot the Moon, Mime, red seals); every joker's and
  consumable's edition, main and joker-on-joker effects; the deck's final step. Left out: the current
  blind's debuffs and modifiers (playing cards are scored undebuffed; an expired perishable joker stays
  debuffed), the `before` context, destruction, dollars, and chance (`pseudorandom` returns its midpoint, so
  1-in-2 and rarer effects never trigger). Hands and discards left at the round's full allowance; no hand
  played this round.
- **Projection:** scaling jokers advanced 6 rounds at fixed per-round rates. From the balatro-gym planner's
  table (`env/planner.py` GROWTH, "rough community-consensus rates"), on the real game's fields: Green Joker
  and Ride the Bus +2 mult; Castle +9 and Wee Joker +12, Runner +4, Square Joker +12 chips; Spare Trousers +0.5
  mult; Constellation +0.04, Hologram +0.05, Vampire +0.04, Lucky Cat +0.05, Glass Joker +0.05, Campfire +0.1,
  Obelisk +0.05, Yorick +0.15 x-mult; Flash Card +1, Red Card +0.5 mult. Ours: the decays, Ice Cream −20 chips
  (floor 0), Popcorn −4 mult (floor 0), Ramen −0.1 x-mult (floor 1). The planner's: Burnt Joker and Space Joker
  each level the most-played hand once per round (most plays; ties to the weaker hand); that hand's play
  count +3 per round (Supernova).
- **phi_build = (clamp(log2(hands per round × B / T8), −10, 2) + 10) / 4**, in [0, 3]; T8 = the ante-8 boss's
  target (`get_blind_amount(8)` × 2 × ante scaling). One doubling = +0.25; hands × B = T8 gives 2.5. The window
  is set from published wins: projected, they reach x ≈ 0 only in antes 7–8 (median x at ante 1 −5.7, ante 4
  −2.1, ante 7 −0.15, ante 8 +0.14; never above 4), so a ceiling at 0 would flatten the late game. 0 at the
  end of a game. Hands per round (not a fixed 4) so Grabber, Troubadour and Hieroglyph change it.
- **Shaping:** reward + 1 × (γ_t·phi(s') − phi(s)) + 1 × (γ_t·phi_build(s') − phi_build(s)); routed per network
  it telescopes per network (`credit.py`). The reward stays `blinds`.
- **Accuracy of 16 deals** (69 shop states of 3 wins, against 256 deals): median |log2 B ratio| 0.02, 90th
  percentile 0.24 (about 0.06 of phi); offer values correlate 0.997.

### Offer values (`--offer-values`)
Every legal buy or pick of a joker or planet and every voucher redeem carries phi_build after taking it
minus phi_build now, as one more number on its action row (×10,000, nearest integer; 0 on every other row).
Computed on the live state and restored (nothing in the game changes). The offer is paid for at its price
(Bull, Bootstraps read money). A joker joins the end of the board with what joining does to the build (hand
size, discards, hands, Oops! All 6s, a negative's slot), and Joker Stencil and Swashbuckler are refreshed as
`Card:update` refreshes them. A planet is valued as used: its hand leveled, and Constellation grown, also when
bought into a slot to use later. A voucher changes what phi_build reads: hand size (Paint Brush, Palette),
hands (Grabber, Nacho Tong, Hieroglyph), discards (Wasteful, Recyclomancy, Petroglyph), joker slots
(Antimatter), Observatory's flag; every other voucher is valued 0. Buying is offered only with room
(`env.lua` uses the game's own space check), so the amendment's full-board case never reaches a row.

### Curriculum (`--curriculum 0.3`)
A start = the first move made in a SHOP at ante 4, 5 or 6 (the shop after the previous boss), reached by
replaying a saved game's moves from its seed, each checked against its saved text (`Sim.fast_forward`). 300
published wins (ordered-capped, Red/White, no challenge, none on the eval 600 or any registered tag's seeds;
chosen by `random.Random("curriculum-published")`), each a start at each ante; plus this run's own training
games as they reach each ante (the 600 most recent per ante). Each round, 30% of the games (in fresh-game
slots) start from a start: ante uniform among antes with starts, then a start uniform among that ante's.
Curriculum games train (both networks) but are in no training stat, arm, retry pool, beaten-seed list or
tally; their wins are replayed from the seed and written to `curriculum_wins.jsonl`, never `wins.jsonl`.

### Shop entropy schedule
0.02 → 0.006 at 1.5M training decisions → 0.001 at 4M, linear between, constant after; decisions of both
networks, curriculum games included. The amendment asked for arXiv 2610.39049's step counts in our units;
counting decisions directly is that, without a conversion through decisions per game. Reading the paper's
schedule as piecewise-linear by environment steps is ours: the paper could not be opened in this session to
check it.

### Measured cost (cloud CPU, 8 published wins, 2,155 moves; `tests/v4_phi_check.py`)
Sim time per move: 17.6 ms without; 26.6 ms with offer values only (+51%: evals and the W4 tests read only
these); 30.0 ms with phi_build too (+70%: training). The model's time per decision doesn't change (one
network runs per decision). Replaying to a curriculum start: 1.4 s (ante 4), 2.1 s (ante 5), 3.0 s (ante 6);
with random weights that is 50–64% of play time, since their own games last about 10 moves. For W4a this
means v4 needs roughly 2.5–3× pins0's wins per game to show 2× wins per sim-hour; W4 stands as registered.

### Checked before training (all on the real game, unmodded 1.0.1o)
`tests/v4_credit_check.py`, `tests/v4_phase_check.py`, `tests/v4_phi_check.py` (8 wins: game state and
observation identical with and without phi_build at every move; cache = fresh; same value in another runtime
and with the deck reversed; bounded; telescopes; every offer taken, 91, moved phi_build by exactly its value),
`tests/v4_curriculum_check.py`, and a 3-round training run of `run_v4.sh` (SMOKE=1). `run_v4.sh` runs the four
checks before every launch.


## Added 2026-11-09 ~15:48 CST: W4b's compute target fixed, and Claude's odds after the measured costs (section J)

Before any v4 training game (chud asked whether anything was left to hypothesize before the run).

### W4b's target, from the logs (`balatro-work/runs/*/metrics.jsonl`, read 2026-11-09)
Model 0.2's run (v2) logs 7.11 h of play + update seconds to round 500, but its behavior-cloning phase (from
v1's games) logs no times. Part of the lineage's logs is missing, so by W4b's rule **moe2's alone is used:
16.71 h** (play + update seconds summed over its 1,044 rounds, a re-run round counted once; 100,224 games).
The test point is v4's first 600-seed eval at which its own summed play + update seconds reach 16.71 h, on
the same laptop. Met: v4 wins ≥ 18 of 600 (2 × pins0's 9) and the one-sided 95% binomial lower bound of its
rate over 9/600 is above 1; met strongly: that bound ≥ 2.

### Claude's odds, revised after the costs measured today (the 12:00 odds above stand as registered)
- **W4a met 35%, met strongly 20%** (were 50%, 35%). Sim time per move in evals is +51%, and a stronger bot also
  plays longer games, so seconds per game may rise ~1.5–1.8×; v4 then needs ~3–3.6× pins0's wins per game.
- **W4b met 15%, met strongly 7%** (were 35%, 25%). 16.71 h is about a tenth of a 168-hour run, and v4 starts
  from random weights with dearer games (+70% sim per move, curriculum replay); pins0 reached 9/600 in the same
  compute starting from 0.2.
- W1 75%, W2 40%, W3 70% unchanged: they compare wins, not cost.
- **New: aim 1's precondition** (v4's best 600-seed eval reaches ante 7 in ≥ 25% of games, within the 168-hour
  run): **25%**.


## Added 2026-11-10 ~10:30 CST: why v4 stalled (diagnosis after the run, section J)

v4 stopped itself at round 620 (2026-11-10 00:55, `--stall-evals 25`): best eval 4.68 blinds/game at round 120,
2.83 by round 200, ~3.5 after. Read from `runs/v4/metrics.jsonl`, the saved eval and training games, and
`analysis/v4_sells.py` (branch `v4-diag`: replays saved training games under the checkpoint that played them and
rebuilds each move's training reward and routed advantage).

**What broke: the table network learned to sell its jokers mid-blind.** Eval (600 fixed seeds), jokers sold per
game: 0.50 at round 120, 1.91 at 140, 2.69 at 180, ~3 from 300 on, all but ~0.01 of them in `SELECTING_HAND`
(the table network's state); buys stayed ~3-4. The usual sequence: every discard used, then 1-4 sells in a row,
then the remaining hands played.

**Why: the death charge is copied onto every move of the blind the game is lost in.** The blind clock sets
gamma = lambda = 1 inside a blind, so the whole cost of dying (phi_build to 0, the blind's progress potential taken
back) lands undiscounted on every play, discard and drag of the final blind. Normalized advantages, table network,
own training games:

| | round 120, survived blinds | round 120, final blind | round 160, survived | round 160, final |
|---|---|---|---|---|
| play | +0.42 | -0.98 | +0.40 | -0.69 |
| discard | +0.41 | -1.17 | +0.32 | -1.06 |
| move_joker | +0.28 | -0.69 | +0.42 | -0.62 |
| sell_joker | -0.09 | -0.71 | +0.32 (38 sells) | -0.70 |

The final blind holds 29% of the table network's decisions (round 120). PPO pushes down the moves taken there and
the softmax moves that probability to moves rarely taken, selling among them: the policy's mean probability of a
sell per table decision in the final blind rose from 0.014 (round 120) to 0.065 (round 160), and in survived
blinds from 0.003 to 0.014. The sells' own advantage was negative at every checkpoint measured, in own games
(-0.55, -0.40, -0.32, -0.10 at rounds 120, 160, 180, 300) and in curriculum games (-0.43, -0.22 at 140, 160): the
rise did not come from selling being rewarded. Knock-on: the routed credit charges the table's sells to the shop
network's return, and the shop's `buy` advantage turned negative after round 120 (+0.09, -0.14, -0.11, -0.20 at
120, 160, 180, 300); buys per training game fell from 5.6 to 3.9.

**A second, separate fault: `--explore-hands` stored log-probabilities at the floor.** An explored play whose
probability is 0 in float32 is stored as log(1e-30) = -69; its true log-probability was ~-135 (median, round 300).
At round 300, 109 of 6,591 decisions (13% of table plays): PPO's ratio on them is ~e^-65, so they give no
gradient, and 109 x ~65 / 6,591 ~ 1.1 accounts for the logged `kl` of ~1.0-1.5 from round ~220 on. None at rounds
120-180, so not the cause of the collapse. (Measured with the checkpoint one update after the weights that played.)
