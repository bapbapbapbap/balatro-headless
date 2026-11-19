# Model v6

Our network taught by Balatro Lab's bot, then trained on our reward alone. The teacher is Balatro Lab's value bot
(KvashaIhor/balatro-lab, MIT, pinned 3e09d47), reproduced at 340/480 = 70.8% in its own simulator
(docs/hypotheses-reproduce.md). Its code stays in its repo; v6 only records its decisions, credited as its.

## Added 2026-11-16, after chud's 11:30 CST message: v6 registered before any of its code

The decision and the order are chud's; the design below is Claude's, reviewed by chud; chud's prediction is marked
as his.

**What changes in the record.** The 2026-11-14 plan kept imitation of Balatro Lab's bot as a control arm only, and
v5 was registered as "not imitation". chud's decision (2026-11-16, his words): "the goal is that this superior bot
remains only a teacher" (11:25); "we are going to give it a strong starting foundation, by doing d3 before d2. the
improvements will fall on our newly taught bot" (11:25); "the student can be called v6" (11:30). D3 is teaching our
network the teacher's play; D2, improving on it, comes after and is done on the student, not on the teacher.

**Stage 0, the bridge** (`reproduce/balatro_lab/`, extending `state.py`): the teacher decides on our game's own
state at every decision point, so it plays the real game with no twin to lose (R lost the twin in 124-252 of 300
games). `state.py` already rebuilds a round (cards, jokers and counters, hand levels, money, blind, boss); stage 0
adds the shop (cards, prices, vouchers, reroll cost), packs (contents, picks left), the blind choice (blind, tag,
next tags) and consumables held (not carried over today). Each teacher answer becomes one legal move of ours, as
rank.py maps them (R: 0.1 unmapped moves per game).

**Checks before any training data:**
- B1: the teacher on our game through the bridge (LuaJIT 2.0.5), on the 600 eval seeds, paired with the teacher
  in its own simulator on the same seeds (the withdrawn T1, needed now as the reference). Gate: the real-game win
  rate within 5 points of its own-sim rate; otherwise the bridge's loss is found and fixed first.
- B2, information: the teacher sees only what a player sees. On 200 states from B1, permuting the order of the
  unseen cards leaves its decision unchanged (its own rollouts seeded the same). Gate: 100%, or every change
  explained. A teacher reading the deck order would teach the student moves it cannot make from what it sees.
- B3: unmapped teacher moves under 1 per game.

**Stage 1, imitation.** Teacher games on fresh seeds from a fixed generator, checked disjoint from the 600 eval
seeds. Per decision: the student's own observation (v5's encoder, with offer and play values), the teacher's move
as an index in our action list, and the reward stream. The student is v5's network (configs/v5.toml: phase split,
d 192, 4 layers, 6 heads, encoding 2, ordered-capped actions, offer values, play values), **randomly initialized**:
chud's hypothesis is about which strategy is cemented first, and a start from v4b would cement v4b's. Loss:
cross-entropy to the teacher's move over legal moves; the value head regresses on returns under v4b's reward
(blinds, build shaping, v4b's gamma and lambdas), so the critic knows our reward before any policy update.

**Stage 2, DAgger.** The student plays fresh seeds; the teacher labels every decision the student reaches; the
data is aggregated and the student retrained. Rounds continue until the student's eval win count fails to rise
for two rounds in a row. This teaches the student in the positions its own mistakes lead to, which the teacher's
games never visit.

**Stage 3, teacher removed.** PPO with v4b's reward and settings only, from stage 2's best student (policy and
value), learning rate 5e-5, no imitation term, v4b's stall rule (25 evals without a new best). The teacher is not
consulted again except to measure agreement.

**Measured throughout:** evals of 600 games on the eval seeds, as every run; top-1 agreement with the teacher on a
fixed set of 2,000 states from B1's teacher games, at every eval of stages 2 and 3. Agreement is the instrument for
whether the taught strategy holds once only our reward trains the student (and, for aim 1, whether a reward moves
the student off the teacher's oughts: every later reward variant can start from the same taught student, so the
teaching becomes one of the fixed facts).

**Cost.** The teacher takes about 191 s a game in its simulator; through the bridge it is expected to be similar.
At 12 workers, about 225 games and roughly 45,000 labeled decisions per laptop-hour. B1 is two runs of 600 games
(about 2.75 h each).

**Predictions (Claude's):**
- W1: the teacher in its own simulator wins at least 60% of our 600 eval seeds. 85%.
- W2: B1's gate passes on the first full bridge run. 50% (shop, packs and consumables are new bridge code).
- W3: stage 2's best student wins at least 60 of 600 (10%; the record is 22). 60%.
- W4: stage 2's best student wins at least 210 of 600 (35%). 25%.
- W5: stage 3's best eval wins more than stage 2's best. 45%.
- W6: at stage 3's end, agreement with the teacher within 10 points of stage 2's best student's. 60%.

**chud's prediction** (2026-11-16 11:30, his words): "my idea is that what strategy gets cemented first matters,
and starting with a much better strategy than one derived from random chance starting weights will perform worse
than one trained on a bot with a reproduced 70% wr." Scoring, as Claude reads it (chud to confirm before stage 1
trains): v6's best eval win count, stage 2 or 3, above the best of every model whose strategy grew from random
starting weights; the record is v4b's 22 of 600 (round 1780).

## Added 2026-11-16, after chud's 12:00 CST message: v6 cut down before any of its code

chud's decision, reviewed by Claude. chud asked (11:36) why all of the 11:32 section was needed; Claude proposed
the cuts below; chud agreed (11:59 "Ok fine", 12:00 "Ok go start"). Everything not named here stands.

- **B1 replaced.** No separate run of the teacher in its own simulator. The data run itself (the teacher on fresh
  seeds, on our game, through the bridge) is the check: its win rate within 5 points of the teacher's reproduced
  70.8% (set 90210). Below that, the bridge's loss is found and fixed before any training.
- **W1 withdrawn** (it needed the own-simulator run). **W2 now reads:** the data run's win rate is within 5 points
  of 70.8% on the first full bridge run. 50%, as before.
- **B2 (no deck-order leak) and B3 (unmapped under 1 per game) stay;** both run in the cloud.
- **Stage 2 (DAgger) is not run up front.** After stage 1's student is scored on the 600 eval seeds, a new section
  decides whether DAgger is needed. W3 and W4 then score the best student before stage 3, with or without DAgger.

## 2026-11-16: stage 0 built; the bridge's checks in the cloud (before the data run)

Claude's, reviewed by chud. Code: `reproduce/balatro_lab/bridge.py`, `teach.py` (branch `v6`, commit ad33019).
Cloud runs use pip lupa's stock LuaJIT (`ALLOW_STOCK_LUAJIT=1`), so they test the bridge, not win rates.

- **Bridge vs twin** (`--check`, first 2 teacher seeds): while Balatro Lab's seeded twin stayed in sync, the
  teacher's decision on the bridge's state was compared with its decision on the twin's. The first version agreed
  on 67 of 73. The flip test (copy one twin attribute into the bridge's state, ask again) found three causes, all
  fixed: (1) the game keeps draw-pile cards facing "back", which state.lab_card read as face down, hiding the deck
  from the bot (it then played where it should discard); (2) Balatro Lab numbers cards in its standard deck order
  and keeps its deck in that order, and its bot keys its samples on both, so the bridge now numbers and orders
  cards the same way; (3) run totals the game shows (rerolls, cards played, discarded and bought, best hand, last
  hand, bosses seen, The Pillar's cards, blinds beaten, the last round's counts) are now carried over. After: 49 of
  50 agree; the one difference is an Arcana pick whose twin hand came from the twin's own randomness (the check
  now re-syncs that hand, as rank.py does). The twin is lost by move 18-51 in both games, so this covers early
  game states only; the data run's win rate checks the rest.
- **Joker order:** the teacher reorders jokers through its own free action; the bridge plays each reorder that
  changes the order as one of our drag moves (`move_joker i>j`), one per decision, and asks again after it.
- **B2 (no deck-order leak):** every hand decision asked again with the draw pile shuffled: 151 of 151 unchanged
  (2 games). Pass.
- **B3 (unmapped):** 0 unmapped teacher moves in 4 games (884 decisions). Pass.
- Teacher games so far (stock LuaJIT, not a rate): 2 wins in 4 (the same 2 seeds, before and after the fixes).

## Added 2026-11-18, after chud's 23:15 CST message: the data run's result, more teacher games, and stage 1 registered before its code

Sources marked per item. Nothing here edits a committed section.

**The data run, as measured** (chud's laptop, LuaJIT 2.0.5, 2026-11-17 ~23:00 to 11-18 morning, 12 workers;
`balatro-work/teach_v6/`, read by `teach.py --report`): 600 games on teacher seeds 0-599, 0 errors. Teacher wins
397 of 600 = 66.2% (±3.8, 95%); 19.32 blinds per game; ~215 s a game. 135,517 decisions, 135,282 labeled (99.8%).
Unmapped teacher moves 0.39 per game, all in a round: play 102, use 84, discard 49 (87 of 600 games had any; at
most 8 in one game). 4,608 joker reorders played as drags.
- **Gate (within 5 points of 70.8%): pass,** by 0.4 points. **W2: hit** (50%). **B3 (< 1 unmapped a game): pass.**
- 66.2% vs the 70.8% reproduction (480 games, Balatro Lab's own simulator, other seeds): z ≈ 1.6, not a
  significant difference, but the point estimate is 4.6 points low. If stage 1's gap needs explaining, the
  unmapped round moves are where the bridge is known to lose the teacher.

**More teacher games** (Claude's proposal, chud's go at 23:15): seeds 600-1799 of the same generator
(`teach.py --first 600 --games 1200`, same folder; `teach_seeds` still skips the 600 eval seeds), running from
2026-11-18 ~23:15. Stage 1 trains on every teacher game in `teach_v6/` when it starts, minus the held-out games.

**Stage 1's reading** (chud's, 2026-11-17 17:38, in his words' sense; reviewed by Claude): graded, not pass/fail.
Stage 1 may land below the teacher; the closer to it, the more merit this line of research has. The readout is the
gap to the teacher's win rate. Floor for chud's hypothesis: more than 22 of 600 (the best of any model grown from
random weights). Beyond-teacher work happens only on the student.
- Diagnostic with it (Claude's): top-1 agreement with the teacher on a fixed set of held-out states. High agreement
  with a big win gap = drift into states the teacher never showed (DAgger's job); low agreement = what the network
  can't represent without the teacher's computation (search, computed features).
- **The teacher's rate on the eval seeds is not measured** (B1 was cut 11-16). Until it is, the gap is read
  against 66.2% on teacher seeds, which is a different seed set. Claude proposed (11-18 09:11) running the teacher
  on the 600 eval seeds as a reference only, never training data; open, chud's call.

**Stage 1's design** (Claude's; chud to review). Replaces "2,000 states from B1's teacher games", since B1's games
don't exist:
- **Held-out games:** a teacher game is held out when sha256("v6-holdout:" + seed) mod 20 = 0 (about 5%). Held-out
  games are never trained on. **The agreement set** is 2,000 labeled decisions drawn from them with
  random.Random("v6-agreement"), fixed for every later stage (saved with the run as `agreement_set.json`, by seed
  and move number).
- **Loss:** cross-entropy to the teacher's move over the legal moves, on labeled decisions only; the value head
  regresses (weight 0.5, as `--vf`) on every decision's discounted return to the game's end under v4b's reward
  (blinds, chip shaping 1, build shaping 1, blind clock gamma 0.95): the lambda = 1 form of the routed returns
  PPO uses, so the critic starts on our reward. Each decision trains only its own network (table or shop), by the
  member the game's state gives it. The first decision's "before" progress is not saved by teach.py; it is taken
  as the first after-state with blinds and phi at 0, which can only move decision 0's return.
- **Training procedure** (not part of the ought): Adam, learning rate 3e-4, minibatch 64, gradient clip 0.5,
  games streamed in shuffled chunks of 32. Up to 8 passes over the training games; after each, held-out
  cross-entropy and agreement; the weights with the lowest held-out cross-entropy are kept (`imitate.pt`), and
  training stops after 2 passes without a new lowest.
- **Then one eval** of 600 games on the eval seeds, exactly as every run's eval (sampling at `--eval-temp` 1, as
  every run), with agreement on the agreement set; then the run stops. Network: v5's (configs/v5.toml's model
  settings), randomly initialized.

**Predictions (Claude's), scored on that eval:**
- S1: the student wins more than 22 of 600. 65%.
- S2: the student wins at least 60 of 600 (10%). 40%.
- S3: top-1 agreement on the agreement set at least 75%. 45%.
- S4: within 10 points of 66.2% (at least 337 wins). 5%.

## Added 2026-11-19, after chud's 11:28 CST message: the second teacher run's result

As measured (chud's laptop, LuaJIT 2.0.5, 12 workers; `balatro-work/teach_v6/games_w600.jsonl` to `w1700`):
teacher seeds 600-1799, 1,200 games, 0 games with errors, 0 duplicate seeds across both runs.
- Teacher wins 800 of 1,200 = 66.7% (±2.7); 19.35 blinds per game; ~216 s a game. 271,590 decisions, 271,194
  labeled. Unmapped 0.33 a game (play 212, use 125, discard 59).
- Same as the first run (66.2%; z ≈ 0.2). **Both runs: 1,197 of 1,800 = 66.5% (±2.2)**, 407,107 decisions,
  406,476 labeled (99.8%), unmapped 0.35 a game.
- Against the 70.8% reproduction in Balatro Lab's own simulator (480 games): z ≈ 1.8; 70.8% is above the
  1,800-game interval's top (68.7%). The teacher through the bridge, in the real game, most likely wins about 4
  points less than in its own simulator. Not separated yet: the bridge's losses (unmapped and fallback moves)
  vs. what their simulator gets wrong about the game. Stage 1's reading uses 66.5% on teacher seeds until the
  teacher's rate on the eval seeds is measured (open, chud's call).

## Added 2026-11-19, after chud's 11:42 CST message: where the teacher's ~4 points go (registered before the fix and the run)

chud asked whether the gap (66.5% in our game, 1,800 games, vs 70.8% in Balatro Lab's own simulator, 480 games)
is fixable on our side or comes from theirs. Claude's, reviewed by chud.

**Already measured** (from `teach_v6/games_w*.jsonl` and four saved games, before this section):
- Every one of the 244 games with an unmapped move reached ante 8, and in the four games read move by move every
  unmapped move is in the game's last round, where every hand-phase decision is unmapped. Their plays all hold
  the same card. **Cause, from the code:** Cerulean Bell (an ante 8 boss) forces one card into every play and
  discard (`blind.lua`: `ability.forced_selection`); the harness only offers plays that hold it. Balatro Lab
  models the Bell (`bosses.py` `CeruleanBell.forced`, `Run.selection` adds the forced card), but the bridge builds
  the boss without setting `forced`, so the teacher picks a hand without the card, nothing matches, and teach.py
  plays its fallback (the first legal play) for the rest of the round. The bridge's fault, not theirs.
- What it costs, roughly: games that reached ante 8 won 86.6% (998/1,152) without unmapped moves and 81.6%
  (199/244) with them; about 12 games of 1,800, **~0.7 points**. Not separated from the Bell being a hard boss.
- For stage 1's data: the Bell rounds' decisions are unlabeled, so the student never trains on the fallback
  moves; only the value head sees those states. Stage 1 is not rerun for this.

**Fix F1** (bridge.py): in a round against Cerulean Bell, set the boss's forced card to the hand card the game
marks `forced_selection`, so the teacher's plays and discards include it as in its own simulator.
- F1a: on teacher games whose last round is against the Bell (cloud, stock LuaJIT), unmapped moves in that round
  go from all of them to 0. 85%.

**Test P (paired, same seeds):** Balatro Lab's bot in its own simulator (`eval_on_our_seeds.py --seed-set teach`,
`VALUEBOT_WEIGHTS=hand`, as the bridge's teacher) on teacher seeds 0-599, the seeds of the first data run, which
its run is seed-exact with at the opening. Each seed then has two outcomes: their simulator, and our game through
the bridge (already measured, before F1). Reading:
- If their simulator wins about as often on these seeds as the bridge did (397/600), the 4 points were the seed
  set (their 70.8% is on set 90210, which their bot was tuned on), not a loss.
- If it wins clearly more, the loss is real on the same seeds; the seeds won there and lost here are where to look,
  with `teach.py --check` (the twin) on them, and each is put down to the bridge (its state or its moves) or to
  their simulator (a rule it plays differently).
- P1: their simulator wins at least 414 of 600 (69%): the gap is real on the same seeds. 55%.
- P2: seeds won in their simulator and lost here outnumber the reverse by at least 15. 50%.
- P3: of the bridge's Bell-round losses among these seeds, at least half are won in their simulator. 60%.
## Added 2026-11-19, after chud's 11:47 CST message: stage 1 code fix (GPU memory), no design change

Claude's, reviewed by chud. The first stage 1 run (11:46) stopped in pass 1 with CUDA out of memory: a minibatch
of 64 decisions with the longest action lists (up to ~1,100 rows) needs more than the 6 GB GPU has; the smoke's 40
games had none that long. Fix (train/imitate.py), as ppo.py already does for PPO: a minibatch that doesn't fit is
freed and done as two halves, each its own step; a single decision that still doesn't fit is skipped and counted
(`oom_splits`, `oom_skipped` in each pass's metrics line). Held-out scoring splits the same way (no gradients).
Nothing trained before the crash is kept: the run restarts from scratch (no checkpoint was written).

## Added 2026-11-19: fix F1's first cloud check, and F1b (registered before its code)

Claude's, reviewed by chud. F1 in the cloud (stock LuaJIT) on the two Bell seeds of teacher run 1 it was checked
on: VGV5MD36 went from 5 unmapped moves to 0 (and won, as before). DNM59TTX still has 8, all `round:use`: the
teacher wants to use The Empress on two cards that leave out the Bell's forced card. In the game the forced card
is always selected, so every legal use of a targeting consumable includes it; Balatro Lab's simulator applies the
Bell to plays and discards (`Run.selection`) but not to a consumable's targets, so its bot picks targets the real
game never allows. That part is their simulator's rule, not the bridge.
- **F1b** (teach.py's move mapping): when the Bell forces a card and the teacher's consumable targets leave it out,
  the targets become the teacher's first targets with the forced card in place of the last, the rule Balatro Lab's
  `Run.selection` applies to plays. The move is labeled as the teacher's.
- F1b-a: DNM59TTX's last round has 0 unmapped moves after F1b. 80%.

## Added 2026-11-19: F1 and F1b's cloud results so far, and measurement F1c (registered before it runs)

Claude's, reviewed by chud. **F1b-a: hit** (DNM59TTX: 8 unmapped -> 0, and it won; it had lost). The Bell games
of runs 1-2 are being rerun in the cloud with F1 + F1b (stock LuaJIT, one process a game): of the 45 that were
lost, the first 6 rerun all won, with 0 unmapped moves. Up to the Bell round the games are the same (the fixes
act only on a Cerulean Bell round, which is only ever the ante 8 boss), so a flip is the fix's doing.
- So the ~0.7 points estimated above was low: the Bell games did not lose more because the Bell is a hard boss,
  but because the teacher could not play it.

**F1c** (laptop, LuaJIT 2.0.5): all 244 games of runs 1-2 that had unmapped moves
(`reproduce/balatro_lab/results/2026-11-19/bell_seeds.txt`), rerun with F1 + F1b (`teach.py --seed-file`,
output in its own folder, never training data). Before: 199/244 won (81.6%), 0.35 unmapped moves a game over
all 1,800.
- F1c-a: unmapped moves 0 in all 244. 85%.
- F1c-b: at least 225/244 won (92%). 70%.
- The teacher's rate over the 1,800 games, recounted with the reruns in place of the originals, is the bridge's
  teacher rate after F1; it goes into test P's reading in place of 66.5%.

## Added 2026-11-19: the cloud Bell reruns, stopped at 23 (the laptop's F1c will measure all 244)

Claude's, reviewed by chud. Measured, cloud, stock LuaJIT, one fresh process a game
(`reproduce/balatro_lab/results/2026-11-19/bell_cloud_23lost.jsonl`): 23 of the 45 lost Bell games rerun with F1 +
F1b. **20 of the 23 now win**; the 3 still lost (G8CDAHJR, A38E825K, FNCQF6SM) have no unmapped moves, so they lose
on play. One unmapped move is left in the 23 (F2BUH5E2, won anyway): two identical Aces of Spades, one forced; the
mapping matches cards by what they are, takes the other Ace, and that play is illegal. Rare (a duplicate of the
forced card); not fixed, counted in F1c. The won Bell games were not rerun in the cloud (stopped for time); F1c
reruns all 244.
- Estimate, until F1c measures it: 45 x 20/23 ≈ 39 more wins, so the teacher's rate in our game would be about
  1,236/1,800 = 68.7%, against 70.8% in its own simulator on its tuning seeds: z ≈ 0.9, no longer a significant
  gap. About half of the ~4 points was the bridge's Bell bug; what is left is within noise of the seed-set
  difference. Test P (same seeds, paired) is still the clean answer.
- For v6: stage 1's data was taught by the pre-fix bridge, but the Bell rounds' decisions were unlabeled, so the
  student never copied a fallback move. Any new teacher data (DAgger, more games) uses the fixed bridge.

## Added 2026-11-19, after chud's 15:37 CST message: the roadmap after the pending laptop runs, test K, and stage 2's details (registered before their code)

chud's direction (15:35-15:37): keep running what is planned; Claude writes down what comes after and builds it
now, so it is ready when the laptop is free. Test K is chud's idea (find which of the teacher's programmed
heuristics carry its play, as competing claims), made concrete by Claude; the rest is Claude's, reviewed by chud.

**Queue on the laptop:** stage 1 (running) → `gap_tests.sh` (F1c, then test P) → test K (below). Then the rule.

**The rule after stage 1 and the gap tests** (T = the teacher's rate in our game after F1, from F1c; S = the
student's eval wins / 600; table agreement = agreement on the agreement set's hand decisions):
1. S within 5 points of T → stage 3 (PPO, our reward alone; KL anchor: chud's call before it starts).
2. Else, table agreement at least 65% → stage 2, DAgger rounds (below).
3. Else → the student can't copy hand play from moves alone. Claude brings chud a design for decision-time
   search at hand play for the student (public information only), shaped by test K; DAgger may run meanwhile.

**Test K: which heuristics carry the teacher** (`reproduce/balatro_lab/ablate.py`). Balatro Lab's bot in its own
simulator, teacher seeds 0-299, paired seed by seed with the full bot (test P's run on the same seeds), one switch
changed per arm, by subclassing its bot (its code untouched). Its switches exist for measuring (strategy.py).
Outcome per arm: wins lost against the full bot, paired (McNemar).
- K1 `ROUND_SEARCH` off (no playing a boss round out before a play or discard in doubt): costs at least 3 points. 70%.
- K2 `HANDS` 1 (power, behind every valuation, from 1 sample hand instead of 12): costs at least 5 points. 55%.
- K3 `DRAWS` 1, `RERANK` 0 (a discard scored on 1 possible draw): costs at least 3 points. 50%.
- K4 the hand-set bonuses in `Weights` at 0 (purchases valued by the build's power alone; thresholds kept): costs
  at least 10 points. 45%.
- Reading for aim 1: K1-K3 switch off computations (facts about the game the bot works out); K4 switches off the
  values someone set by hand. Whatever K4 costs is how much of the teacher's strategy is chosen values, i.e. what
  the student inherits as oughts from imitation and what stage 3's reward gets the chance to move.
- Cost: ~1,200 games in their simulator, ~5 h at 12 workers.

**Stage 2's details (DAgger, as registered 11-16, made concrete):**
- Round k: the student (round 1: stage 1's best.pt; then the last round's) plays 300 fresh teacher seeds
  (indices 1,800 + 300(k-1) onward), sampling its moves at temperature 1 as in evals, with a fixed RNG per seed.
  At every decision the fixed bridge (F1, F1b) records the teacher's move for the state the student is in; the
  student's own move is what is played (`teach.py --student`). Data in `balatro-work/dagger/d<k>/`.
- Then imitation as stage 1 (same loss, procedure, held-out rule by seed hash) on every teacher game so far
  (teach_v6 and every dagger round), starting from the student's weights (warm start, fresh optimizer); the
  agreement set stays stage 1's. Then the 600-game eval with agreement.
- Stop: when eval wins fail to rise for two rounds in a row (11-16). The best round's student goes to stage 3.
- D1: round 1 raises eval wins over stage 1's. 70%. D2: agreement on the agreement set rises in round 1. 60%.

## Added 2026-11-19: test K and stage 2 built, checked in the cloud (no results)

Claude's, reviewed by chud. Code, as registered in the roadmap section above; nothing about the design changed.
- Test K: `reproduce/balatro_lab/ablate.py` (arms by subclassing Balatro Lab's bot; settings checked per arm).
  Cloud, one seed per arm: every arm finishes a game. Part 3 of `gap_tests.sh`.
- Stage 2: `teach.py --student` (the checkpoint plays, sampling at temperature 1 with a per-seed RNG; the teacher's
  move is the label, the student's is `played`), `--imitate` takes several folders and globs, `configs/v6d.toml`,
  `train/dagger_round.sh <k>`. Cloud, end to end on a cloud-trained checkpoint (2 student games, 9 teacher games):
  the student plays, every decision is labeled, imitation warm-starts from the student on both folders with stage
  1's agreement set, the 600-game eval runs with agreement, and the round's run folder is kept as runs/v6d-r1.
- Stage 2 must run on the fixed bridge: `gap-check` is merged into `v6` before round 1.
- `docs/proposals/v6-student-search.md`: options for rule 3 (not registered; chud decides).
