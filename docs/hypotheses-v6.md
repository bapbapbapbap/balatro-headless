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

## Added 2026-11-20, after chud's 11:06 CST message: stage 1's result, the gap tests' results, and the rule

Measured on chud's laptop (LuaJIT 2.0.5); scored by Claude. Nothing here edits a committed section.

**Stage 1** (`v6` 8a11d21, 2026-11-19 15:27-16:05; `balatro-work/runs/v6/`): imitation ran all 8 passes; every
pass set a new lowest held-out cross-entropy, so the patience stop never fired and the loss was still falling at
the cap (pass 7 1.233, pass 8 1.205). Kept pass 8. Agreement on the 2,000-decision set **68.0%** (hand decisions
**40.7%**, shop 78.1%). Eval, 600 eval seeds, temperature 1: **78/600 won (13.0%)**, 10.88 blinds/game; deaths by
ante {1: 101, 2: 62, 3: 43, 4: 64, 5: 77, 6: 79, 7: 58, 8: 38}.
- **S1 hit** (78 > 22). **S2 hit** (78 >= 60). **S3 miss** (68.0% < 75%). **S4 miss** (78 < 337).
- chud's floor for his hypothesis (more than 22, the best grown from random weights): passed, 3.5x. His graded
  reading, the gap to the teacher: 55.7 points (T below).
- Where the student loses: 101 of 600 games (16.8%) die in ante 1, 163 (27%) by ante 2. The teacher in our game
  died in ante 1 in 12 of 1,800 (0.7%), by ante 2 in 32 (1.8%) (`teach_v6/`). The early antes are where the hand
  decisions are cheap and the student's 40.7% hand agreement, sampled at temperature 1, costs most.

**F1c** (`balatro-work/bell_f1c/`): 238/244 won (97.5%), was 199/244; lost -> won 40, won -> lost 1. Unmapped moves
11, one each in 11 games (play 7, discard 2, use 2), all 11 games won; every loss had none.
- **F1c-a miss** (11, not 0; the duplicate-forced-card case recorded 11-19 is one of them). **F1c-b hit.**
- **T = 1,236/1,800 = 68.7%** (+-2.1), the teacher's rate in our game after F1, equal to the cloud estimate.

**Test P** (`lab_on_teach600.jsonl`): Balatro Lab's bot in its own simulator, teacher seeds 0-599: 436/600 (72.7%).
Our game before F1 397/600: both won 333, theirs only 103, ours only 64 (McNemar z 3.02). After F1 (F1c reruns in
place) 416/600 (69.3%): theirs only 90, ours only 70 (z 1.58).
- **P1 hit** (436 >= 414). **P2 hit** (net 39 on the registered pre-F1 pairing; 20 after F1). **P3 hit** (13 of 20).
- Reading: their bot wins 72.7% on seeds it was never tuned on, vs 70.8% on its tuning set: no sign of
  cherry-picked seeds (chud's 11-14 question). After F1 the gap on the same seeds is 3.4 points, not significant.
  The 90 seeds won only there are listed in `balatro-work/gap_tests.txt` for `teach.py --check`.

**Test K** (`ablate_k.jsonl`; seeds 0-299; full bot 223/300 = 74.3%):

| Arm | Won | Cost (points) | Lost / gained | McNemar z | s/game |
|---|---|---|---|---|---|
| K1 round search off | 214 | 3.0 (+-2.5) | 12 / 3 | 2.32 | 46 |
| K2 power from 1 hand | 162 | 20.3 (+-6.6) | 82 / 21 | 6.01 | 188 |
| K3 discard on 1 draw | 200 | 7.7 (+-5.3) | 45 / 22 | 2.81 | 111 |
| K4 hand-set bonuses 0 | 29 | 64.7 (+-9.1) | 195 / 1 | 13.86 | 283 |

- **K1 hit** (exactly at 3.0). **K2 hit. K3 hit. K4 hit.** No arm is net-negative.
- Reading for aim 1: the hand-set bonuses carry 64.7 of the teacher's 74.3 points; without them it wins 9.7%,
  near the 8.5% its own ladder starts from. The computations (facts it works out) are worth 20.3 (power), 7.7
  (draws), 3.0 (search), not additive. The teacher is mostly chosen values sitting on a fact-computing machine.

**The rule** (registered 11-19 15:37): S is 13.0%, T 68.7%, not within 5 points (rule 1 no); hand agreement 40.7%,
below 65% (rule 2 no). **Rule 3 fires:** Claude brings chud the student-search design, shaped by test K
(`docs/proposals/v6-student-search.md`); DAgger may run meanwhile. Test K puts power and draws (K2, K3) well above
search (K1), which in the proposal's own terms points to S1 (discard values from sampled draws, an input) before
S2 (round search). Power also sits behind the teacher's shop valuations, so not all of K2's 20 points are hand play.
- Two things the rule's reading doesn't separate: imitation stopped at its pass cap while still improving, and the
  eval samples at temperature 1. Whether the 40.7% is the network's limit or the run's is open; any measurement of
  it is registered before it runs.

## Added 2026-11-20, after chud's 12:38 CST message: the new direction, step 0's findings, and step 1 registered before its code

The roadmap agreed with chud 11:06-12:38 (full text in the Project doc `claude/ROADMAP.md`). Decisions are chud's,
reviewed by Claude; measurements, predictions and designs are Claude's, reviewed by chud.

**chud's decisions (11-20).**
- Goals, in order: the strongest bot (working target **5% wins at Gold stake**, Red deck first); one win per deck
  per stake; 100% of a fresh save (`tools/save_progress.py`'s definition) in as few seeds and runs as possible. The
  is-ought work moves to the backlog: its questions read cleanly only on a strong agent.
- **Reverses the 11-16 rule** ("Balatro Lab's bot stays only a teacher"): the teacher may be the base that is
  improved. Every file that imports it keeps its credit (KvashaIhor/balatro-lab, MIT, 3e09d47).
- **Ownership test:** the bot is chud's when removing his parts costs more win rate than removing Balatro Lab's,
  measured as test K measured theirs.
- The student may search, with rollouts judged by its own value head (deliberation toward the reward adds no
  second ought).
- Two scores per model from here: the registered one (eval temperature 1, as every run) and a delivery one (the
  best settings found). Only the first is ever compared with earlier runs.
- The real save is touched only after the headless proof exists (roadmap step 7).
- First actionable: finish teaching the student, measuring what limits it before spending days of teacher games.

**Step 0's findings (read or measured 11-20, after 12:38 and before any of step 1's code).**
- Code paths: `--eval-temp 0` divides the logits by zero (non-finite probabilities, the game stops with an error);
  imitation from stage 1's weights works through `start = "warm:..."` (same architecture: every weight loads as
  it is; fresh optimizer); `imitate_max_games N` takes the first N training seeds in sorted order (fixed, and
  nested: 450 is inside 900) and cuts the held-out games to N/10, so a run on a subset needs stage 1's
  `agreement_set.json` copied into its folder first (as `dagger_round.sh` does) to be scored on the same set.
- **Replay (S1's data path):** 3 saved teacher games (11EVA2HX, 11MKTA84, 11WHX9KR; 723 decisions) replayed from
  their seeds in our harness alone (cloud, stock LuaJIT): every observation and every move's text equal to the
  saved ones, to the end of each game. New inputs can be computed on the 1,800 teacher games by replay, without
  the teacher.
- **Balatro Lab's code (3e09d47):** its simulator has all 15 decks, the 8 stakes' rules (`data/stakes.json`: blind
  scaling, small blind reward, discards, sticker chances) and the stickers. Its bot accounts for Eternal (can't
  sell) and a joker already perished; nothing in `strategy.py` reads Rental or a Perishable's coming debuff, and its
  survival constants were fitted at White stake.
- **A bridge gap (found by reading, like the Bell):** `bridge.py` builds the teacher's run as
  `Run(None, deck="Red Deck")`, so at any stake the teacher reasons with White stake's rules and blind scaling, and
  on any deck as if it were Red. Every teacher measurement so far was Red/White, where this is right.
- **Option scores (soft targets):** the bot values each shop and pack option and ranks discards, so scores exist
  there; its plays come mostly from rules (hand plans), not one score per legal play. Soft targets would be
  partial.

**Fix F2 (bridge.py), registered before its code:** the teacher's run is built with the game's own deck
(`G.GAME.selected_back`) and stake (`G.GAME.stake`), mapped to Balatro Lab's names.
- F2a: on 20 teacher seeds at Red/White, the teacher's moves are identical with and without F2. 90%.
- F2 runs before any teacher game above White (roadmap step 3; its Gold gate is registered when it is built).

**Step 1 (laptop, one queue), on stage 1's data and network; all evals on the 600 eval seeds:**
- **M1** (`tools/eval_ladder.py`, new): stage 1's `best.pt` at eval temperature 0 (the most likely move; a game
  over the step cap counts as a loss and is reported) and 0.5. Temperature 1 is the registered 78/600.
  - M1a: temperature 0 wins more than 78. 75%. M1b: at least 120. 35%.
  - M1c: temperature 0 loses at most 50 games in ante 1 (temperature 1: 101). 55%.
- **M2** (`configs/v6m2.toml`): imitation continued from stage 1's `best.pt` (warm start, fresh optimizer), same
  data, held-out games and agreement set, up to 8 more passes with the same stop rule; then the eval.
  - M2a: hand agreement rises at least 3 points (to 43.7% or more). 60%. M2b: it reaches 50%. 20%.
  - M2c: eval wins more than 78. 55%.
- **Scaling curve** (`configs/v6c450.toml`, `configs/v6c900.toml`): stage 1 again from random weights on the first
  450 and 900 training games (stage 1 = 1,707), same procedure (up to 8 passes, so fewer steps on less data);
  stage 1's agreement set.
  - C1: wins rise with data, W450 < W900 < 78. 60%.
  - C2: steep at the top, 78 - W900 at least 20. 40%.
  - C3: hand agreement at 900 at least 3 points below stage 1's 40.7%. 50%.
- **Reading (registered):** steep (C2 hit) → more teacher games are worth their ~9 h per 1,800. Flat (78 - W900
  under 10) and M2a missed → the student lacks inputs, not data: S1 first. Between → both. S1's data costs no
  teacher time either way (replay, above).

**S2's criteria (registered before S1 exists):** after S1, round search for the student (judged by its value head)
is built if hand agreement stays under 65% **and** the gap sits in boss rounds: hand agreement on boss-round
decisions at least ~10 points below other rounds, or more of the student's losses at bosses than the teacher's
share (83% in Balatro Lab's own report). It is not needed to close the gap if the student comes within 5 points of
the teacher, or reaches 65% hand agreement with wins following (then DAgger, for drift). As the main route past
the teacher (search plus distillation) it stays on the roadmap either way.

## Added 2026-11-20, after chud's 15:13 CST message: step 1 scored, the delivery score, the single-card finding, and more teacher games registered before they run

Scores and the finding are Claude's, reviewed by chud; the data run follows the reading registered in 788c324
(chud's direction, reviewed by Claude). All evals: the 600 eval seeds, Red deck, White stake.
Files: `balatro-work/step1.txt`, `balatro-work/step1/m1_t*.jsonl`, `m2_t0.jsonl`.

**Step 1, scored.**
- **M1** (stage 1 `best.pt`): temperature 1 = 78 (13.0%), **0 = 182 (30.3%, 95% 26.8-34.1)**, 0.5 = 163 (27.2%).
  Ante-1 losses 101 / 39 / 54. **M1a hit, M1b hit, M1c hit** (39 <= 50). Paired with temperature 1: 51 seeds won
  at both, 27 only at 1, 131 only at 0.
- **M2** (warm start, kept pass 7; pass 8 fell to 66.6%): 119 (19.8%); agreement 71.0% (hand 45.7%, shop 80.3%).
  **M2a hit** (45.7 >= 43.7), **M2b miss** (< 50), **M2c hit**.
- **Scaling curve** (temperature 1): 450 = 17 (2.8%), 900 = 55 (9.2%), 1,800 = 78. Hand agreement 30.9 / 34.4 /
  40.7. **C1 hit, C2 hit** (78 - 55 = 23), **C3 hit** (6.3 points).
- **Reading (as registered): steep → more teacher games.** The confound registered with it stands: the smaller
  runs also had fewer steps, and M2's extra passes on the same data gained more (+41 wins) than the step from 900 to
  1,800 games (+23). But M2 peaked at pass 7 and fell at pass 8, so the same data's headroom looks mostly spent.

**Delivery score (chud's two-score rule; not compared with earlier runs).** M2's `best.pt` at temperature 0:
**206/600 = 34.3% (±3.8)**. Paired with stage 1 at temperature 0: 95 seeds won only by M2, 71 only by stage 1,
111 by both (z = 1.86). Losses by ante 1:40, 2:28, 3:26, 4:32, 5:62, 6:68, 7:69, 8:69. Ante-1 losses did not move
(39 → 40); the gain is in antes 2-4 (107 → 86).

**Finding (Claude's, measured in the cloud after the evals, not a registered test): unsafe single-card plays.**
- Stage 1's temperature-0 losses replayed in the harness (50 of 418; the rest owed) and 40 of the teacher's saved
  games (`teach_v6/games`, per-round from the saved move text).
- The teacher plays one card in 21.7% of its plays (the student 16.7%) and two or more single cards in ~9.5% of its
  rounds; it lost **1 of 72** such rounds. The student, in its losing games, lost **14 of 51**.
- The pattern: with scaling jokers held (Ride the Bus, Obelisk, Supernova, Scary Face, Spare Trousers), the student
  plays four single cards with every discard left and a pair or two pair in hand: RGCEV1TD (Big Blind, 61/450,
  holding 6-6-4-4), JDJCEA1D (Big Blind, 175/450), LWPD9VF1 (The Manacle, 360/600). Before it has jokers it plays
  five-card hands.
- Reading: the student copied the teacher's farming move without its condition (the round is safe anyway).
  `play_values` gives each play's score against what is still needed, not whether the hands left can still clear
  the blind. Candidate input for S1: the round's margin (best play's score × hands left against the need).
  Search judged by the value head (S2) would cover it as well. Not built; any fix is registered before its code.
- The repeated "discard the lowest four" is the teacher's own pattern too: not the defect.

**Registered now, before it runs: teacher games 1,800 → 3,600** (step 2's steep branch).
- `reproduce/balatro_lab/teach.py --first 1800 --games 1800 --workers 12 --out balatro-work/teach_v6`: the next
  1,800 teacher seeds (same generator, never an eval seed), same code and bridge as the second run (F1 in), Red
  deck, White stake. No option scores recorded (not built; partial anyway).
- D1: the teacher wins 64-72% of the new 1,800. 85%.
- D2: unmapped moves at most 0.5% of decisions. 80%.
- The student run on 3,600 games gets its own section before its config (whether it starts from M2 or from
  scratch, and the bigger-network arm, decided then).

## Added 2026-11-21, after chud's 09:38 CST message: the third teacher run's result, and step 2 (the student on 3,600 games) registered before its code

Measurements, predictions and designs are Claude's, reviewed by chud; the direction (steep → more data) is the
reading registered in 788c324 and f7293c9.

**Teacher games 1,800 → 3,600, scored** (`balatro-work/teach_v6/games_w1800.jsonl` to `w3450`, chud's laptop,
12 workers, overnight 11-20 → 11-21).
- 1,800 seeds played, 1 error (IndexError in `parse_desc`: the teacher's move index past the legal list; one game,
  logged, no decisions saved for it). **1,249 of 1,799 won = 69.4% (±2.1). D1 hit** (64-72%).
- 411,409 decisions, 411,401 labeled; **8 unmapped = 0.002%** (round:play 4, round:use 3, round:discard 1). **D2 hit.**
- Ante-1 losses 10 of 1,799. Mean 223 s per game.
- With the second run (after F1): 2,485 of 3,599 = 69.0%.

**Step 2, registered now: the student on all 3,599 games.** Two runs, both from random weights, same data, same
held-out rule (sha256 of the seed, 1 in 20; unchanged, so stage 1's held-out games stay held out) and stage 1's
agreement set copied in, as in step 1.
- **v6e:** stage 1's network and settings, up to 16 passes (patience 2). 16 passes on 2x the data puts it at
  roughly M2's step budget (8 + 7 passes on 1,800), so v6e against M2 is mostly data, not steps.
- **v6eb (bigger network arm):** the same with d 256, 6 layers, 8 heads (v6e: 192, 4, 6), run after v6e.
- Each: the registered score (eval temperature 1, 600 seeds) and the delivery score (temperature 0, `eval_ladder`).
- E1: v6e beats M2 at temperature 1 (> 119 wins). 75%.
- E2: v6e reaches 140 wins (23.3%) at temperature 1. 50%.
- E3: v6e's hand agreement >= 47.7% (M2 + 2). 65%.
- E4: v6e at temperature 0 still loses >= 25 games in ante 1 (data alone doesn't remove the single-card deaths of
  the 15:13 finding). 70%.
- B1: v6eb beats v6e by >= 12 wins at temperature 1. 45%.
- Reading: E1 and a gain >= 20 wins → data still limits: another teacher run (3,600 → 7,200) overnight. E1 miss or
  a gain < 10 → stop adding data; S1 (the round-margin input and draw-based discard values) next. E4 hit → the
  round-margin input is built either way, registered before its code. B1 hit → later runs use v6eb's size.

## Added 2026-11-21, after chud's 16:09 CST message: step 2 scored; the round-margin input and a fourth teacher run registered before their code and data

Scores, designs and predictions are Claude's, reviewed by chud. Files: `balatro-work/step2.txt`,
`runs/v6e/train.log`, `runs/v6eb/train.log`, `step2/v6e_t0.jsonl`, `step2/v6eb_t0.jsonl`.

**Step 2, scored** (3,388 training games, 211 held out, stage 1's agreement set).
- **v6e:** kept pass 15 of 16 (held-out loss 0.918; pass 16 rose to 0.951). Temperature 1: **132/600 (22.0%)**;
  agreement 74.1% (hand 54.3%, shop 81.4%). Temperature 0: **254/600 (42.3%, 95% 38.4-46.3)**; losses by ante
  1:29, 2:27, 3:25, 4:25, 5:42, 6:52, 7:69, 8:77.
- **v6eb:** kept pass 16 of 16, held-out loss still falling (0.882). Temperature 1: 125 (20.8%); agreement 74.7%
  (hand 54.3%, shop 82.2%). Temperature 0: 245 (40.8%); ante-1 losses 29.
- **E1 hit** (132 > 119). **E2 miss** (132 < 140). **E3 hit** (hand 54.3% >= 47.7%). **E4 hit** (29 >= 25).
  **B1 miss** (v6eb 7 wins below v6e at temperature 1; at temperature 0, 99 seeds won only by v6e and 90 only by
  v6eb, z = 0.65: no difference).
- Delivery, paired on the 600 seeds: v6e at temperature 0 against M2 at temperature 0: 115 seeds won only by v6e,
  67 only by M2, **z = 3.56**. 382 of the 600 seeds are won by at least one of M2, v6e, v6eb at temperature 0.
- **Reading.** The registered reading left a gap: E1 hit with a gain of 13 at temperature 1 is neither ">= 20" (more
  data) nor "< 10" (stop adding data). Stated now, before anything runs: the delivery score moved by 48 wins with
  z = 3.56, so data still pays; and E4 hit, so the round-margin input is built. Both go ahead: the teacher runs on
  the laptop overnight while the input is built in the cloud. B1 miss: stage 1's size stays. Both runs ended at or
  near the 16-pass cap with held-out loss still falling (v6eb kept its last pass): the next student run gets more
  passes.

**Registered now, before it runs: teacher run 4, games 3,600 → 5,400.** Same command form as run 3
(`--first 3600 --games 1800`), same code, Red deck, White stake. ~9 h at run 3's rate.
- D3: the teacher wins 64-72% of the new 1,800. 85%.
- D4: unmapped moves at most 0.5% of decisions. 85%.

**Registered now, before its code: the round-margin input (S1, part 1).** The 15:13 finding (2026-11-20): the
student plays single cards to grow its jokers when the hands left can't clear the blind; `play_values` gives each
play's score against what's still needed but not whether the round stays winnable after it.
- Two more numbers per play in `H.play_values` (each play's row; new setting `play_margin`, off by default so every
  earlier run replays unchanged): with `need` the chips still needed, `s` this play's score, `best` the highest
  score among this state's plays and `h` the hands left including this one:
  (1) margin = log2((s + best × (h - 1) + 1) / (need + 1)), clipped and scaled like the existing value;
  (2) safe = 1 when s + best × (h - 1) >= need, else 0.
  This is an estimate (the next hands' best is taken to equal this hand's), not a lookahead.
- The teacher's saved observations predate it, so the inputs are rebuilt for every teacher game by replay (exact
  from the seed: step 0's finding) before training. A check: the rebuilt observations match the saved ones in
  every other number, on all games.
- The student run on it (v6f, own config and run page, built after this commit): v6e's recipe on every teacher game
  then in `teach_v6`, up to 24 passes, patience 3, `play_margin` on; scored at temperature 1 and 0 as before.
- R1: v6f at temperature 0 loses at most 18 games in ante 1 (v6e: 29). 60%.
- R2: v6f at temperature 0 wins at least 15 more than v6e (269+), counted on the same data size if run 4 is not done
  (then against a v6e-recipe twin without the input, trained alongside). 55%.
- R3: in v6f's temperature-0 losses, rounds with two or more single-card plays are lost at most half as often as
  v6e's (replay count, as in the 15:13 finding). 60%.

## Added 2026-11-21, after chud's 23:42 CST message: the score check, causal confusion, and three tests registered before their code (the game-rule constraint, the teacher on the student's death rounds, DAgger re-registered)

Readings, designs and predictions are Claude's, reviewed by chud. The constraint applies the game's own rule
(a run ends when a blind is not beaten), pointed out by chud.

**The score check (done before this section, in the cloud).** Is the single-card failure a bug between our harness
and the game? The harness's predicted score (`H.play_scores`, behind `play_values`) against the chips the game
actually added, for every play in three stage 1 temperature-0 death games (RGCEV1TD, JDJCEA1D, LWPD9VF1):
five-card plays exact; single-card plays 14/18/15/15 predicted vs 16/15/15/15 (Spare Trousers, Obelisk), 14/26/36/48
vs 28/39/48/60 (Smeared Joker, Ride the Bus), and exact under The Manacle (Supernova, Juggler, Scary Face).
**Not an interface bug:** the student was shown a single card scores ~15 against 450 needed and played it. A small
real bug found on the way: predictions miss a scaling joker's increase on the hand being played (Ride the Bus ~12
low); it errs low, so it does not cause farming. Backlog.

**The reading: causal confusion** (de Haan, Jayaraman & Levine 2019). In the teacher's games, farming with single
cards almost always comes with a safe round (1 of 72 such rounds lost), so "scaling joker held → farm" predicts the
teacher's move as well as "farm only if the round survives" does; imitation learns the cue that is easier to read.
No input says whether the round survives a play (that needs the best other play and the hands left), and the
student never sees an outcome. DAgger (Ross, Gordon & Bagnell 2011) supplies exactly the missing examples.
The 11-20 plan put DAgger after more data and new inputs; stage 1's ante-1 deaths (101 vs the teacher's 12 in
1,800) already pointed at it. Ordering error, Claude's.

**Test G, the game-rule constraint (eval only; new setting `survive`, off by default).** At a hand decision, with
`need` the chips still needed, `h` the hands left (this one included) and `best` the highest predicted score among
the plays: a play is *safe* if its score + best × (h - 1) >= need. When at least one play is safe, unsafe plays are
removed before the move is chosen; when none is, nothing is removed. Discards and other moves are never removed.
Run: v6e's `best.pt`, temperature 0, the 600 eval seeds, Red, White; paired with v6e's own temperature-0 run (254).
Each game also counts how often the constraint changed the model's move.
- G1: v6e + constraint wins more than 254. 80%.
- G2: it wins at least 274 (+20). 50%.
- G3: ante-1 losses fall from 29 to at most 15. 65%.
- G4: seeds lost with the constraint that v6e won without it: at most 25. 75%.
- Reading: the gain is how much of the 160-game gap this one failure is. G2 hit → the constraint stays on in the
  delivery score until a student no longer needs it (its move-change count near 0).

**Test T, the teacher on the student's death rounds.** v6e's temperature-0 ante-1 losses (up to 12, by seed order),
replayed move for move through `teach.py`'s game loop with a replaying student, so the teacher labels every state
the student reached (the shadow state stays in step). At each death round's first play, compare the teacher's move
with the student's.
- T1: in at least 2/3 of the death rounds where the student's first play was a single card with a pair or better in
  hand, the teacher plays a different move. 80%. (Hit = the teacher would not have farmed there: causal confusion,
  not a teacher habit.)

**DAgger, re-registered (v6d; built 11-19, not run).** Two fixes before any round runs:
- Seeds: rounds took seeds from index 1,800 + 300(k-1), which teacher runs 3 and 4 have used since; a seed in two
  folders is taken from the first (`teach_v6`), so round 1's labels would have been dropped without an error.
  Rounds now take indices from 100,000 + GAMES × (k-1) (the same generator, never an eval seed).
- Round 1's student: v6e's `best.pt` (was stage 1's).
- Round 1: 600 games (was 300), so its states are ~10% of the data after run 4. Training: from the student's weights
  on teach_v6 + dagger/d*, up to 8 passes (as built). After training, a temperature-0 eval of the round's model.
- An attribution twin (v6dt): the same training from v6e on teach_v6 alone (no DAgger folders), so a round's gain
  is not read off the extra passes or run 4's data.
- D5: round 1 at temperature 0 beats v6dt at temperature 0. 75%.
- D6: round 1's temperature-0 ante-1 losses <= 15. 60%.
- D7: round 1 at temperature 0 wins >= 290 (48.3%). 50%.

## Added 2026-11-22, after chud's 23:42 CST message (11-21): tests G and T scored

Claude's, reviewed by chud. Both ran in the cloud (2 cores) from v6e's `best.pt` (staged from the laptop); with the
constraint off, the cloud reproduced the laptop's v6e temperature-0 games move for move (4 seeds checked).

**Test G, the game-rule constraint** (v6e, temperature 0, 600 eval seeds, `--survive 1`; b819d47):
- **263/600 (43.8%, 95% 39.9-47.8)** against v6e's 254. Paired: 10 seeds won only with the constraint, 1 only
  without; z = 2.71. Losses by ante 1:22, 2:20, 3:18, 4:25, 5:41, 6:60, 7:67, 8:84 (v6e: 1:29, 2:27, 3:25, 4:25,
  5:42, 6:52, 7:69, 8:77). Blinds per game 17.33 (v6e 16.73).
- The constraint removed plays at 2,609 decisions and changed the model's move 199 times, in 102 games; 496 games
  were identical to v6e's.
- **G1 hit** (263 > 254). **G2 miss** (+9, not +20). **G3 miss** (ante-1 losses 22, not <= 15). **G4 hit** (1 seed lost).
- Reading: the single-card failure is real (z = 2.71) and **small: about 9 games of the ~160-game gap**. In most of
  the 102 changed games the student survived the round and died later. Per the registered reading (G2 miss), the
  constraint does not join the delivery score by default; it remains a setting (chud's call whether to use it).

**Test T, the teacher on the student's death rounds** (v6e's first 12 temperature-0 ante-1 losses by seed;
`analysis/test_t_teacher_on_deaths.py`):
- 6 of 12 death rounds began with a single-card play; **all 6 held Ride the Bus**. In 4 of the 6 the hand had a pair
  or better; there the teacher's move differed in 3 (it discarded first; the student played the single card at once)
  and matched in 1 (D8EIKW46: the teacher also played the single card). **T1 hit** (3 of 4 >= 2/3; n is small).
- The other 6 deaths were five-card plays the teacher mostly agreed with (agreement 4-6 of 8 moves per round):
  weak hands at a 600-chip boss (Chaos the Clown, The Trio, Astronomer) or the Big Blind, not farming.
- Reading: causal confusion is confirmed where farming occurs, and it is narrow (Ride the Bus at ante 1). Most early
  deaths, and the 252 losses in antes 5-8, are something else. DAgger stays next: it addresses every state the
  student reaches that the teacher never did, not only this one.

## Added 2026-11-22, after chud's 09:45 CST message: teacher run 4 scored

Claude's, reviewed by chud. `balatro-work/teach_v6/games_w3600.jsonl` to `w5250` (seeds 3,600-5,399, laptop,
12 workers, 11-21 → 11-22).
- 1,800 played, 0 errors. **1,265 won = 70.3% (±2.1). D3 hit** (64-72%).
- 405,492 decisions; **13 unmapped = 0.003%** (pack:skip 8, round:use 4, round:discard 1). **D4 hit.**
- Ante-1 losses 12. `teach.py --report` over the whole folder: 5,399 games, 1 error, 3,711 won = 68.7% (±1.2).
- DAgger round 1 and its twin (`train/step3.sh`, registered in the 2026-11-21 23:42 section) train on all of these.

## Added 2026-11-22, after chud's 18:03 CST message: DAgger round 1 and its twin scored

Claude's, reviewed by chud. `train/step3.sh` on the laptop (e0ff1f5), 11-22 09:48 → 17:56; log
`balatro-work/step3.txt`. Both trained from v6e's `best.pt` with a fresh optimizer, up to 8 passes, stage 1's
agreement set; the temperature-0 evals are `balatro-work/dagger/r1_t0.jsonl` and `twin_t0.jsonl` (600 eval seeds,
0 errors each).

| | trained on | kept pass | agreement (table, shop) | t1 | **t0** | t0 losses by ante 1-8 |
|---|---|---|---|---|---|---|
| v6e (step 2) | 3,599 teacher games | 15 of 16 | 74.1% (54.3, 81.4) | 132 | 254 (42.3%) | 29, 27, 25, 25, 42, 52, 69, 77 |
| round 1 (v6d-r1) | teach_v6 + dagger/d1: 5,648 (351 held out) | 6 of 8 | 76.3% (58.6, 82.9) | 183 | **244 (40.7%, 95% 36.8-44.6)** | 23, 32, 18, 24, 43, 66, 70, 80 |
| twin (v6dt) | teach_v6: 5,085 (314 held out) | 7 of 8 | 75.4% (56.0, 82.9) | 187 | **273 (45.5%, 95% 41.6-49.5)** | 15, 14, 9, 32, 56, 48, 72, 81 |

Paired at temperature 0 (seeds won by only one of the two):
- round 1 vs twin: 70 vs 99, z = -2.23.
- twin vs v6e: 93 vs 74, z = 1.47.
- round 1 vs v6e: 77 vs 87, z = -0.78.

- **D5 miss** (round 1 244 < twin 273; the twin is ahead, z = -2.23).
- **D6 miss** (round 1's ante-1 losses 23, not <= 15).
- **D7 miss** (244, not >= 290).

Reading:
- Adding DAgger round 1's labels to the same training cost about 29 temperature-0 wins against the twin. Round 1
  does not beat its twin, so the condition the plan set for a round 2 (round 1 beats its twin) is not met. Whether
  DAgger gets another design is chud's call.
- The twin is the best student so far: 273 at temperature 0 (v6e 254, z = 1.47, not significant alone). Its gain
  over v6e mixes two things the design does not separate: run 4's 1,800 teacher games and 8 more passes from v6e.
- Agreement on stage 1's agreement set did not track wins: round 1 agreed more (76.3% vs 75.4%) and won fewer.
  Round 1's kept pass was chosen on a held-out set that includes DAgger games, so the two runs' pass choices are
  not on the same footing.
- Not tested here, so not a finding: why the labels hurt. One candidate is that the teacher's moves from the
  student's own (often losing) states pull the policy away from ordinary play. Round 1's antes 2-3 losses (32, 18)
  are above the twin's (14, 9) but close to v6e's (27, 25), so the early-ante numbers do not settle it.

## Added 2026-11-22, after chud's 18:10 CST message: the student at Gold, registered before it runs

The test is chud's (measure the student at Gold before running the teacher there), reviewed by Claude; the
predictions are Claude's. Goal 1's working target is 5% at Gold stake (2026-11-20 section). The student plays our
game directly, so this needs no bridge fix (F2 is the teacher's). The student's training games were all White
stake, so its sticker features (eternal, perishable, rental; encode.lua) have only ever been 0.

**Test S, the best student at Gold.** v6dt's `best.pt`, temperature 0, Red deck, Gold stake (`--stake 8`, the
game's `stake_gold`, stake_level 8), the 600 registered eval seeds, `tools/eval_ladder.py`; output
`balatro-work/gold/v6dt_t0_s8.jsonl`.
- S1: wins between 6 and 24 (1-4%). 50%.
- S2: fewer than 30 wins (below the 5% target). 85%.
- S3: mean blinds per game below 12 (White: 17.23). 65%.

Reading:
- 30 wins or more (S2 miss): the working target is met without the teacher at Gold. The teacher's Gold rate becomes
  a reference only, not a gate.
- Fewer than 30: the next step depends on the route, which is chud's call. If imitation continues, the teacher's
  Gold rate is needed (F2, then the teacher at Gold), since a student rarely passes its teacher by much. If training
  on our own reward is next, the teacher's Gold rate matters little.

## Added 2026-11-22, after chud's 18:18 CST message: test S scored

Claude's, reviewed by chud. Laptop, 7849069, 18:11 → 18:13; `balatro-work/gold/v6dt_t0_s8.jsonl` (log `gold/s.txt`).
Before it ran, a cloud check confirmed `--stake 8` starts the game's Gold rules: 3 discards with the Red deck (4 at
White), blind scaling 3, eternal, perishable and rental stickers in the shop, no Small Blind reward.
- 600 games, 0 errors. **Won 0** (95% upper bound 0.5%). 6.98 blinds per game (White: 17.23). Near wins 0.
- Losses by ante: 1:46, 2:155, 3:233, 4:128, 5:32, 6:6. No game reached ante 7; 562 of 600 ended by ante 4
  (White: 70 of 327 losses).
- **S1 miss** (0, not 6-24). **S2 hit** (below 30). **S3 hit** (6.98 < 12).
- Reading (as registered): the 5% target is not met by the student as it stands. The next step depends on the route,
  chud's call: continuing to imitate needs the teacher's Gold rate (F2, then the teacher at Gold); training on our
  own reward does not. This test does not say which Gold rule does the damage (blind scaling, one fewer discard,
  stickers the student has never seen, the lost Small Blind money).

## Added 2026-11-22, after chud's 18:26 CST message: fix F2 built, and the teacher on the eval seeds at any stake

Claude's, reviewed by chud. Built as registered (2026-11-19, fix F2), for chud's request to run the teacher where
the student was run (test S and the stake sweep).
- `state.run_args(G)`: the game's deck (`G.GAME.selected_back.name`, the English name Balatro Lab uses) and stake
  (`G.GAME.stake`, 1-8, mapped to Balatro Lab's "White Stake" ... "Gold Stake"). Every Run the bridge builds (in a
  round, `state.lab_run`; outside one, `bridge.base_run`) is made with them. `twin.py` (--check) is unchanged.
- `teach.py --back/--stake` start the game there; `--eval-seeds` plays the 600 eval seeds (these games are for
  scoring and never go in a training folder). Each game's record carries its back and stake.
- Cloud checks: at Red/White the Run F2 builds equals the pre-F2 Run field for field, so the teacher's moves there
  are unchanged by construction. At Red/Gold it differs in exactly the stake fields, and they match the game: 3
  discards, the scaling-3 table (300, 1,000, 3,200, 9,000, 25,000, 60,000, 110,000, 200,000). Smoke: the first 2
  eval seeds at Gold, 0 errors, every decision labeled, 0 unmapped; the teacher reached ante 4 on both (10 blinds
  each; the student, test S: 7 blinds, ante 3).
- F2a as registered (moves identical on 20 seeds at Red/White) runs on the laptop: teacher seeds 3,600-3,619 (run
  4's, made with the code just before F2) again with F2, compared move for move by `f2a_check.py`.

## Added 2026-11-22, after chud's 19:15 CST message: test X, the student kept off stickered jokers (registered before its code)

The direction is chud's ("fix the Gold and Orange bug"), reviewed by Claude; the design and predictions are Claude's,
reviewed by chud.

**Already measured** (the stake sweep, `balatro-work/gold/sweep.txt`, v6dt temperature 0, Red deck, 600 eval seeds;
read here only for this test, not scored): Purple 4 won, 9.69 blinds/game, 199 games over by ante 3; Orange 0, 8.39,
293; Gold 0, 6.98, 434 (test S). Orange adds only Perishable stickers to Purple's rules and Gold only Rental, so the
student's further loss above Purple comes from those two stickers. Its sticker inputs (encode.lua, item features 8-10)
are read from the game correctly (env.lua, `card.ability`), but they were 0 in every training game (all White).
Also found by reading: `H.offer_values`' build value (build_phi.lua) prices a Perishable joker as if it never expires
until it has, and never counts Rental's $3 a round.

**Test X** (eval only; new setting `shun_stickers`, off by default, so every earlier run replays unchanged). At every
decision, buying (buy, buy_and_use) or picking (pack) a joker that carries a Perishable or Rental sticker is removed
before the move is chosen, unless that would remove every move. Eternal is not touched (it arrives at Black, below
Purple, where the student already wins). Counted per game: decisions filtered, and how often the model's own move was removed.
Run: v6dt `best.pt`, temperature 0, Red deck, the 600 eval seeds, at Orange (stake 7) and Gold (stake 8), paired
with the sweep's runs on the same seeds.
- X1: at Gold, blinds/game rises from 6.98 to at least 8.0. 70%.
- X2: at Gold, games over by ante 3 fall from 434 to at most 330. 60%.
- X3: at Orange, blinds/game rises from 8.39 to at least 9.0. 60%.
- X4: at least one win at Gold or Orange. 30%.
- X5: at Gold the model's own move is removed in at least 1 game in 3 (it buys stickered jokers when it can). 75%.

Reading: if X1-X3 hit, buying stickered jokers is most of the Orange/Gold loss; the setting joins the delivery score
at those stakes and stays until a student trained with stickers no longer needs it (its removed-move count near 0).
If they miss, the loss is in living with stickers or in the economy, not in the buying, and only training at those
stakes addresses it. Either way the lasting fix is training where the stickers appear (the from-scratch run); this
test measures how much of the gap is one avoidable choice.

## Added 2026-11-22: test X built (no results)

Claude's, reviewed by chud. As registered (2026-11-22 19:15 section): `H.sticker_flags()` (build_phi.lua, beside
`H.survive_flags`) flags buy, buy_and_use and pick of a joker whose `ability` has `perishable` or `rental`; the
filter in `rollout.py` (eval only, setting `shun_stickers`) removes flagged moves unless every move is flagged, and
counts filtered decisions and removed model moves; `tools/eval_ladder.py --shun-stickers 1` sets it, records the
counts per game and prints them in the summary. Checked in the cloud without the game (its source is only on the
laptop): the Lua parses, and on a stub shop and pack the flags come out as designed (perishable buy, rental pick and
rental buy_and_use flagged; eternal buy, end_shop and an unstickered pick not). Off by default: no earlier run changes.

## Added 2026-11-22, after chud's 19:26 CST message: test X scored

Claude's, reviewed by chud. Laptop, c352a15; `balatro-work/gold/x_s7.jsonl`, `x_s8.jsonl` (log `gold/x.txt`), each
paired seed for seed with the stake sweep's run without the setting (`gold/v6dt_t0_s7.jsonl`, `v6dt_t0_s8.jsonl`).

| | blinds/game | paired change (z) | seeds better / worse / same | over by ante 3 | wins | model's move removed |
|---|---|---|---|---|---|---|
| Orange, without | 8.39 | | | 293 | 0 | |
| Orange, test X | 7.80 | -0.59 (-5.24) | 129 / 206 / 265 | 345 | 0 | 1,162 times in 435 games |
| Gold, without (test S) | 6.98 | | | 434 | 0 | |
| Gold, test X | 6.12 | -0.86 (-6.62) | 159 / 298 / 143 | 455 | 0 | 3,152 times in 572 games |

- **X1 miss** (6.12, not >= 8.0; it fell). **X2 miss** (455, not <= 330). **X3 miss** (7.80, not >= 9.0).
  **X4 miss** (no win). **X5 hit** (the model's move was removed in 572 of 600 games at Gold).
- Reading (as registered, the miss branch): buying stickered jokers is not the Orange/Gold loss. Keeping the student
  off them costs it, significantly at both stakes: at those stakes stickered jokers are a large share of what the
  shop and packs offer, and the jokers it buys are worth more to it than they cost, sticker included. The loss is in
  living with the stickers (a Perishable joker it builds around going dead after five rounds; Rental's $3 a round)
  or in the economy, and only training at those stakes addresses it. `shun_stickers` stays off; it does not join the
  delivery score. The build-value finding of the 19:15 section (Perishable priced as if it never expires, Rental's
  cost not counted) stands as a candidate input fix for that training, registered before its code if it is made.

## Added 2026-11-22, after chud's 19:36 CST message: stage 3 at Gold (run v6g), registered before its code

The route is chud's ("train a bot on gold difficulty... skip the balatrobot teacher"), reviewed by Claude; the
design and predictions are Claude's, reviewed by chud. This is stage 3 as registered (2026-11-20: PPO from the best
student, our reward alone, learning rate 5e-5, no imitation term, v4b's stall rule), run at Gold, with one addition.

**Why no teacher at Gold.** Besides speed: the teacher ignores Rental and a Perishable's coming debuff and keeps
White's survival constants (F2 section), and test X found the student's Gold loss is in living with stickered
jokers, not in buying them. Teacher labels at Gold would be weakest exactly there. Under test S's registered fork
(own reward: the teacher's Gold rate is not needed), teach_s8 and F2a are not run.

**Dropped, with its reason: the sticker-pricing fix** (offer values, 19:15 section). Read in full,
`H.offer_values` is the change in phi_build, a scoring-power measure projected PROJECT_ROUNDS = 6 rounds ahead.
Every offered joker is fresh (`perish_tally = G.GAME.perishable_rounds = 5`, card.lua 517, game.lua 1914), so a
Perishable offer is overvalued by at most its sixth round, about 1/6 of its value. Rental's $3 a round is money,
and phi has no money term at all (dollars reach it only through Bull and Bootstraps): it is outside what the offer
value measures, not a mispricing within it. The network already sees each item's sticker flags and price; under
PPO at Gold those inputs get their first gradient. So the fix would move little and would add a second change to
this run; it is not made.

**Run v6g** (`configs/v6g.toml`): v6dt's settings without the imitation lines; `start = warm:runs/v6dt/best.pt`
(fresh optimizer, round 0); Red deck, stake 8 alone (a `--stakes` mix would also mix the eval seeds, so the eval
and best.pt would track the mix, not Gold); reward `blinds` with v4b's shaping and credit settings as in v6dt;
lr 5e-5; eval every 20 rounds on the 600 eval seeds (temperature 1, as every in-run eval); stall rule 25 evals with
no new best; 168-hour cap; no curriculum (its published starts are White wins).

**The addition: a critic warm-up**, new setting `--critic-rounds N` (default 0, so every earlier run is unchanged).
For rounds 0 to N-1 only the value heads' weights train (the policy heads and the trunks they share are frozen, so
the policy is exactly v6dt's), then normal PPO. v6g: N = 10 (~960 games). Reason (Claude's reading, untested): v4c
and v5 both fell for ~120 rounds after a fresh-optimizer warm start, and here the critic was fit to the teacher's
White returns (~17 blinds a game) while the student gets ~7 at Gold, so the first advantages would be biased.
A runtime guard: in a critic-only round the policy's measured KL over the update must be 0, or the run stops.

**Agreement without imitation**, new setting `--agreement-from <teacher data>` (default off): the run's fixed
agreement set (`agreement_set.json`, copied in from runs/v6dt by the setting `--agreement-set`) is scored at every
eval as in stages 1-2, without running imitation. v6g scores it on teach_v6.

**Predictions** (Claude's), scored when v6g stops:
- G1: the value loss of critic round 9 is at most half of critic round 0's. 70%.
- G2: no eval in rounds 20-120 falls more than 10% below round 0's eval (blinds/game), i.e. the warm-start dip of
  v4c and v5 does not recur at that size. 50%.
- G3: v6g's best in-run eval beats round 0's by at least 1.0 blinds/game. 55%.
- G4: v6g's best.pt at temperature 0 on the 600 eval seeds at Gold (eval_ladder, paired with test S's run) wins at
  least one game. 35%.
- G5: that eval reaches the working target, 30 of 600 (5%). 5%.
- G6: agreement with the teacher at v6g's best eval is at least 5 points below round 0's. 60%.
- G7: v6g's best.pt at temperature 0 on the 600 eval seeds at White wins fewer than 258 (v6dt's 273 - 15). 40%.

Reading: G3 + G4 hit → own reward at Gold works from the taught student; continue it and register the next lever
(late-ante coverage, the round-margin input). G3 misses → PPO from the student does not climb at Gold either, as at
White (v4c, v5); the next question is the update itself (learning rate, credit), not the stake.

## Added 2026-11-22, after chud's 23:12 CST message: chud's White-vs-Gold prediction, registered before its evals

chud's, reviewed by Claude. Registered while v6g trains (round ~170), before any temperature-0 eval of its best.pt.

**The prediction, in chud's words (23:05-23:10):** "the bot trained on gold stake will have a much better white
stake rate afterwards"; "a model this good at reaching x is better at reaching x-difficulty." His scope (23:12):
White and Gold only. It is the opposite of G7 (Claude's).

**Scored:** v6g's best.pt (as it stands when v6g stops), Red deck, White stake, temperature 0, the 600 eval seeds
(`tools/eval_ladder.py`), paired seed by seed with v6dt's same eval (`balatro-work/dagger/twin_t0.jsonl`, 273 wins).
**Met** if v6g wins more games and the paired z = (b - c) / sqrt(b + c) over discordant seeds is above 2. At the
discordance measured between v6dt and its DAgger round (169 seeds) that is about +26 wins (~299 of 600); more
discordance raises it. "Much" has no threshold; the gain is printed.

**Computed alongside (no verdict):** the same paired numbers at Gold against `balatro-work/gold/v6dt_t0_s8.jsonl`
(0 wins, 6.98 blinds/game), and the compression factor = White gain / Gold gain in blinds per game (wins at Gold
too few for a ratio), plus the ratio in wins when the Gold win gain is not 0.

**Pipeline, run once v6g stops (stall rule or 168 h):** the two evals G4 and G7 already need, then the scorer,
committed here with this section: `tools/score_white_gold.py --white-new <v6g White t0> --white-old
dagger/twin_t0.jsonl --gold-new <v6g Gold t0> --gold-old gold/v6dt_t0_s8.jsonl`. Outputs
`balatro-work/gold/v6g_t0_s1.jsonl`, `v6g_t0_s8.jsonl`, `v6g_white_gold.txt`. Checked on existing files: DAgger
round 1 vs its twin gives z -2.23 (70 vs 99 discordant), as recorded in step 3.

## Added 2026-11-23, after chud's 11:06 CST message: v6g stopped by hand at round 731, before its evals

chud's call, reviewed by Claude. v6g was stopped with Ctrl+C (SIGINT; health.json "stopped: interrupted (Ctrl+C)",
iteration 732, 70,272 games, 67 train wins) instead of by the registered stall rule, which would have fired at round
1160 at the earliest. Reason (chud): growth had stalled. In-run evals (temperature 1, 600 seeds) since round 600 were
9.678, 9.467, 9.697, 9.705, 9.512, 9.573, 9.547: flat within eval noise, and a lucky eval above 9.705 would only
have restarted the 25-eval stall clock. The scored checkpoint is unchanged by the stop: best.pt is round 660's
(eval 9.705, 1 win), sha256 4462f80e4b41fb9e..., written 09:21 CST; every prediction scored from best.pt (G4, G5,
G7, chud's White-vs-Gold) uses that file. No temperature-0 eval of v6g had run when this section was committed.

## Added 2026-11-23, after chud's 11:39 CST message: v6g scored at temperature 0 (G4, G5, G7, chud's prediction)

Laptop, branch v6-win-attempts (1fcc1c1), 11:08-11:20 CST. v6g best.pt (round 660, sha256 4462f80e4b41fb9e...),
Red deck, the 600 eval seeds, `tools/eval_ladder.py --temp 0`; outputs `balatro-work/gold/v6g_t0_s1.jsonl`,
`v6g_t0_s8.jsonl`; scorer output `gold/v6g_white_gold.txt`. 0 errors.

| | v6dt White | v6g White | v6dt Gold | v6g Gold |
|---|---|---|---|---|
| Wins | 273 | 174 | 0 | 0 |
| Blinds beaten/game | 17.23 | 17.89 | 6.98 | 9.92 |
| Mean ante reached | 7.29 | 6.56 | 2.94 | 3.91 |
| Games with a skipped blind (lower bound) | 392 | 1 | 5 | 0 |
| Wins with all 24 blinds beaten | 16 of 273 | 173 of 174 | - | - |

- **chud's prediction (White, paired z > 2): NOT MET.** The opposite: 174 vs 273, discordant 51 / 150, z -6.98.
- **G7 (White below 258): MET** (174). **G4 (≥1 Gold win at t0): NOT MET** (0 of 600; the test-S pairing is moot
  at 0). **G5: NOT MET.** G1-G3 and G6 are scored from the run's logs in a later section.
- **Correction to the scorer's compression line.** `score_white_gold.py` printed a blinds-per-game factor of 0.23.
  Blinds beaten per game does not compare these two models: skipped blinds count as not beaten, and v6g stopped
  skipping. Its White "gain" of +0.66 blinds is the skips it no longer makes. Measured by mean ante reached, which
  skips don't affect, White moved -0.73 and Gold +0.97: a compression factor of about **-0.75**. The registered
  definition (blinds/game) stands in the 23:12 section; this section records why it misleads here.
- **Found, not predicted:** PPO on the `blinds` reward removed skipping (v6dt skipped in 392 of 600 White games;
  257 of its 273 wins include a skip; v6g skipped once). The reward pays for each blind beaten, and a skip trades
  that blind's reward for a tag. Whether this causes the White drop is untested; the next section registers a test.

## Added 2026-11-23, after chud's 11:50 CST message: the skip test, registered before its code ran

Claude's, reviewed by chud (chud: no prediction). Question: how much of v6g's White drop (273 -> 174) comes from
no longer skipping blinds?

**Design:** v6dt best.pt (the same file as `dagger/twin_t0.jsonl`), Red deck, White, temperature 0, the 600 eval
seeds, with every `skip_blind` move removed from the legal set before the argmax (`tools/eval_ladder.py --no-skip 1`,
new; default off, so every earlier eval is unchanged; the filter is in `train/rollout.py` next to tests G and X's
and acts only in eval mode). Paired seed by seed with v6dt as it plays and with v6g. Scorer
`tools/score_skip_test.py`. Output `balatro-work/gold/v6dt_t0_s1_noskip.jsonl`, `gold/skip_test.txt`.

**Check:** the filter must change v6dt's move in some decisions (v6dt skips ≥1.9 blinds a game at White); if it
changes none, the test did not run as registered.

**Prediction (Claude's):**
- S1: v6dt without skips wins 175-257 at White: skipping explains part of the drop, not all of it. 50%.
  (Below 175, removing skips alone is as bad as all of v6g's changes: 15%. 258 or more, skipping is not the cause
  at the size of the drop: 35%.)

**Reading:** the share of the drop reproduced = (273 - no-skip wins) / (273 - 174). Near 100%: v6g's White loss is
mostly the skips its reward taught it to give up, i.e. the reward's ought (count blinds) diverging from winning at
one decision. Near 0%: the loss is elsewhere in the drift from the teacher (agreement fell 75% -> 53%).
Caveat: removing skips puts v6dt in states it rarely reached (more shops and money, no tags), so this measures
"v6dt forced not to skip", not "v6dt had it learned not to skip".

## Added 2026-11-23, after chud's 12:21 CST message: the skip test scored

Laptop, 8f8cbe4, 11:52-12:00 CST; `balatro-work/gold/v6dt_t0_s1_noskip.jsonl`, `gold/skip_test.txt`. 0 errors.
Check passed: 8,143 decisions had a skip removed; 1,441 changed v6dt's move (2.40 a game).

- v6dt without skips: **259** wins (v6dt 273, v6g 174); mean ante reached 7.24 (7.29, 6.56); vs v6dt discordant
  63 / 77, z -1.18 (not significant). Losses by ante 1:16, 2:15, 3:11, 4:22, 5:53, 6:66, 7:78, 8:80.
- Share of v6g's White drop reproduced by removing skips: **14%**.
- **S1 (175-257): NOT MET**, in the bin given 35% (258 or more: skipping is not the cause at the size of the drop).

Reading (as registered): v6g's lost skipping is real but is not what costs it at White. About 85 of the 99 lost wins
are elsewhere in its drift from v6dt. Where v6g loses more than v6dt at White: antes 3-6 (losses 23/49/90/71 vs
9/32/56/48), i.e. the mid-game, not the late antes where v6dt's own losses sit.

## Added 2026-11-23, after chud's 12:24 CST message: v6g on Blue Deck at Gold, zero-shot, registered before it runs

chud's call (Blue first), predictions Claude's. v6g best.pt (round 660, sha256 4462f80e4b41fb9e...), no training on
Blue. Blue = Red's 52 cards with +1 hand per round (Red has +1 discard instead); the network sees the deck only
through its contents and the HUD.

- **B1, transfer:** `tools/eval_ladder.py --back b_blue --stake 8 --temp 0`, the 600 eval seeds, paired seed by seed
  with Red Gold (`gold/v6g_t0_s8.jsonl`: 0 wins, mean ante 3.91). Prediction: mean ante reached higher than Red's
  (the extra hand), 70%; at least 1 win, 25%. Output `gold/v6g_t0_s8_blue.jsonl`.
- **B2, a win for the ladder:** the same at temperature 1 on 3,000 eval seeds (the registered 600 plus their
  extension), `gold/v6g_t1_s8_blue.jsonl`. Not a rate claim: a search for a Blue Gold route that replays.
  Prediction: at least 1 win, 55%.

## Added 2026-11-23, after chud's 13:02 CST message: v6g on Blue Deck at Gold scored (B1, B2)

Laptop, 54f2371, 12:24-12:55 CST; `balatro-work/gold/v6g_t0_s8_blue.jsonl`, `v6g_t1_s8_blue.jsonl`, `gold/blue.txt`.
0 errors in both.

- **B1, transfer (temperature 0, the 600 seeds):** mean ante reached **4.38** vs Red Gold's 3.91, paired +0.47,
  z 9.77: **MET** (70%). **1 win** (XX1IAKXP; Red Gold had 0): **MET** (25%). Near wins 4. Losses by ante
  1:16, 2:33, 3:86, 4:163, 5:204, 6:76, 7:16, 8:5.
- **B2, win search (temperature 1, 3,000 seeds):** **5 wins** (6YVBLGC3, G4K8XTZL, 257HLXF1, UT5JR15B, ILVQPF1D),
  0.17%; 11 near wins; 10.95 blinds/game: **MET** (55%).
- None of the six counts as beaten until its route replays on the fast sim and the untouched game to the game's own
  win (`tools/ladder_to_routes.py`, brought over from branch purple-wins, a525fbb, unchanged).

## Added 2026-11-23, after chud's 13:16 CST message: the six Blue Deck Gold routes replay

Laptop, ec0c051, `tools/ladder_to_routes.py`, 13:05-13:06 CST. Each route was replayed on the fast sim and on the
untouched game; all six won with the same final state and move count on both, 0 mismatches:
XX1IAKXP (t0, 286 moves), 6YVBLGC3 (318), G4K8XTZL (267), 257HLXF1 (338), UT5JR15B (303), ILVQPF1D (318).
Routes: `balatro-work/v6g/blue_gold_t0_routes.jsonl`, `blue_gold_t1_routes.jsonl`; log `v6g/blue_gold_verify.txt`
(the t1 file's; the t0 file passed first, as its routes file is written only on PASS and the command chained on it).
Blue Deck at Gold Stake is beaten (repo rule: a route that replays to the game's own win). Not yet published to wins/.

## Added 2026-11-25, after chud's 22:00 CST message: v6g at Gold on all 15 decks, verification, seed sharing (results)

Results only; nothing here was registered in a committed section before its data. Predictions made in chat are
marked as such. v6g best.pt (round 660, sha256 4462f80e4b41fb9e85ec9f69a48dc3bf1ac35099d3c3a6f3213b8535aa909106),
stake 8, the fixed eval seed list, one game per seed. Scripts and outputs: `gold/` (the repo) and
`balatro-work/gold/` (eval files, logs).

**How the wins were searched (2026-11-23 → 11-24).** `crack3.sh` (t1, 3,000 seeds on Yellow, Checkered, Ghost):
Yellow EAWZSSK8, Ghost NIU5VBUN, Checkered 0. `crack_t0.sh` (t0, chosen because t0 beat t1 on mean ante and ante-8
arrivals on every deck tried): each deck still at 0 wins taken to 3,000, 9,000, 18,000 seeds, stopping at its first
win. All 15 decks had a win by 05:15 on 11-24. Because that rule stops at a win, its rates lean high; `even_9000.sh`
(11-24 10:31 → 18:04) then took every deck to the first 9,000 seeds at t0 (Black 18,000), so the rates share a
denominator. Red's first 600 games replayed move for move against the 11-23 file: 600/600.

**Rates at t0** (`gold/results/gold_rates.md`, Wilson 95% intervals there): Green 29/9,000, Blue 17, Yellow 15,
Ghost 10, Zodiac 6, Erratic 5, Magic 5, Nebula 5, Abandoned 3, Checkered 3, Red 2, Anaglyph 2, Painted 1, Plasma 1,
Black 2/18,000. 106 wins; first 600 seeds pooled over 15 decks: 4/9,000.

**Verification** (`gold/verify_gold.py`, chud's laptop, 2026-11-25 21:48-21:54, repo ec0c051): weights hash
matches; every eval file holds exactly the first N eval seeds; best.pt, in fresh processes, re-chose every route's
moves and won; every route replayed on the untouched game. 114 routes (106 t0, 7 t1, Red's training-eval win
WCRKZKR1), 0 mismatches, 15/15 decks, PASS.

**Seed sharing** (`gold/seed_pool.py`, same run; `gold/results/seed_pool.md`), t0 wins only: 15 seeds won by two or
more decks against 0.521 expected if decks won independently; 55 shared deck pairs against 0.524. HTBDF155 was won
by 7 decks, B88LZ88C by 5. Under one band of seeds shared by all decks, the band is f = 0.95% of seeds (95%
0.6-2.2%, Poisson bootstrap over seeds); win rate inside it from 34% (Green) to 1.2% (Black, Painted, Plasma).
f is this policy's rough ceiling, not the game's winnable fraction.

**Claude's prediction, made in chat on 11-24 ~10:37 before even_9000 played those cells (recorded in the project's
status notes, not committed):** of the 63 deck/seed cells for the 5 seeds shared at that time that hadn't been
won, at least 10 win at t0. **NOT MET: 8 won** (B88LZ88C blue, ghost; HTBDF155 red, blue, yellow, erratic;
9QGDSETV abandoned, erratic). The direction held (independence predicts ~0.4); the size was overestimated.

**Prior art found 2026-11-25, before any claim was made.** AnteZero (github.com/namecallfilter/antezero, MIT),
released v0.2.0 on 2026-11-20, reports a PPO transformer trained on all 15 decks and 8 stakes winning about 12% at
Gold over all decks (3-24% per deck, about 1,024 games per cell). So v6g's 15-deck Gold result is not first and its
rates are about 100x lower; what differs is that v6g trained on one deck and every win is a replayed route.
chud's decision (21:49): the project stays internal for now, citing Balatro Lab and AnteZero.

Still owed for v6g: G1-G3 and G6 from the run's logs.

## Added 2026-11-25, after chud's 22:31 CST message: correction to the prior-art paragraph of the 22:00 section

Claude's, reviewed by chud. The 22:00 section said that what differs from AnteZero is that v6g trained on one deck.
That is wrong. AnteZero's history (tags in its repository, read 2026-11-25): v0.1.0, tagged 2026-11-12 (commit
ed13c5a), README: trained on Red Deck, White Stake only, about 91% there, and zero-shot at Gold Stake 1-2% on Red,
Blue, Yellow, Green, Magic and Zodiac (about 1,024 games per cell; the other nine decks round to 0%). v0.2.0, tagged
2026-11-20 (commit 426c6ba), trained on all 15 decks and 8 stakes: Gold 3-24% per deck, about 12% overall. Dates are
publication dates; when AnteZero first won at Gold is not known, only that it was no later than those dates.

v6g's first Gold win was 2026-11-22 (training) and its 15th deck 2026-11-24 05:15 CST. So AnteZero was ahead both in
achievement (no later than 11-12 for zero-shot Gold from a one-deck White policy on six decks, no later than 11-20
for all 15) and in publication. What remains distinct about v6g's record is the evidence form: every win is a route
on the game's own code, re-chosen by the frozen policy and replayed on the untouched game; AnteZero reports rates
from its Rust reimplementation, without per-win routes as far as its README shows.

## Added 2026-11-25, after chud's 23:23 CST message: v6gd, v6g with the deck mixed, registered before it runs

The question is chud's, reviewed by Claude: v6g trained on the Red Deck alone and generalized to the other 14 at
Gold zero-shot; there is no comparison yet to training across the decks. The design is Claude's, reviewed by chud.

**Run v6gd** (`configs/v6gd.toml`): v6g's run with one change. Same start (v6dt's best.pt, fresh optimizer,
round 0), same reward, credit, learning rate, critic warm-up (10 rounds), agreement scoring, Gold stake, eval
cadence and stall rule. **The change: every game's deck is drawn uniformly from the 15 decks** (`--backs`, all 15
listed once). The deck is fixed for the whole game and comes from a hash of the game's seed (`common.game_config`),
so a retried seed and every eval seed always get the same deck.

**Why equal games, not equal time or equal decisions** (chud, 23:23: "don't punish it for doing the right
things"). At r660 a Gold game takes 2.6 s on Black and 6.7 s on Painted, 58 to 126 decisions (eval files in
`balatro-work/gold/`). Weighting decks by time or decisions would give a deck fewer games as the policy learns to
survive longer on it. Uniform per game doesn't, and it needs no code. The cost: decks with longer games get
more of the training signal (by r660's lengths, 3.7% of the time for Black up to 9.7% for Painted).

**Kept as in v6g, on purpose:** encoding 2 (no deck token; the policy sees the deck only through what a player
sees, as v6g did), so the deck mix is the only change. A deck token is a later, separate test.

**What else changes as a consequence:** the in-run eval (600 eval seeds) is also mixed, about 40 games per deck,
so best.pt is picked on mixed-deck blinds and v6gd's in-run numbers are not comparable to v6g's (Red only).
`--smoke` first: `--backs` has not been used in the v6 lineage.

**The comparison** (scored when chud stops v6gd): R = the last round both runs have a checkpoint for (every 20
rounds), at most 660. v6gd's `ckpt_R` and v6g's `ckpt_R`, temperature 0, Gold, `tools/eval_ladder.py` (which plays
one deck per call and ignores the checkpoint's deck mix), the first 600 eval seeds on each of the 15 decks: 9,000
games per checkpoint, paired seed by seed and deck by deck.

**Predictions** (Claude's):
- D1: pooled over the 15 decks, v6gd's mean blinds per game beats v6g's. 70%.
- D2: on the Red Deck alone, v6g's mean blinds per game is at least v6gd's (the specialist keeps its deck). 55%.
- D3: v6gd's mean blinds per game is higher on at least 11 of the 15 decks. 50%.
- D4: pooled wins, v6gd at least v6g's. 55% (both are likely in single digits per 9,000; reported, not decisive).

Reading: D1 hit → training across decks beats specializing at equal games; the next Gold run trains mixed. D1
miss → at this scale one deck generalizes as well as the mix, and the deck set is not the lever (boss play and
xMult, from the walls report, stay first).

## Added 2026-11-26, after chud's 10:55 CST message: v6gd's pre-flight, a phi_build cache gap, v6gd unchanged

Claude's, reviewed by chud. The v6gd smoke (2026-11-25 23:30, ccfb59e) stopped in pre-flight at `v4-phi`
(`tests/v4_phi_check.py --routes 3`): on route ZZ6JHUKR (Red, Gold, found 2026-11-23 in v6g's training) at move 22
(SELECTING_HAND) the cached phi_build was 1.2129 and a fresh computation 0.8167 ("key misses something"). Nothing
trained; the real launch was not reached.

Why now: the check replays routes spread evenly over `wins/routes.jsonl`, and the last one is now a Gold route
(PR #132 added them). When v6g launched (74eeab7) the file had none. `harness/` is identical between the tag
`run-v6g` and ccfb59e and `train/` differs only in eval-only and logging lines, so **v6g trained with this same
gap**: its build shaping and its offer-values input could read a stale phi_build at some decisions. How often, and
by how much, is not measured. A likely cause (unconfirmed): `build_key` does not include a card's `debuff`
(card_sig omits it, `build_power_fresh` reads it), so a boss that debuffs cards would hit an entry cached
before the boss.

Decision for v6gd: it keeps v6g's code exactly, so the comparison changes only the deck. `v4-phi` is dropped from
`configs/v6gd.toml`'s pre-flight list. Every other check stays. The gap is tracked as its own issue. Its size will
be measured on v6g's saved Gold games, and any fix registered as its own change. It does not enter v6gd.

## Added 2026-11-26, after chud's 10:55 CST message: the seed band test (v6dt on r660's shared seeds)

Claude's, reviewed by chud. Laptop, 2026-11-25 ~23:30, `balatro-work/gold/seed_band.py` (not in the repo: it runs
`tools/eval_ladder.py` unchanged with `common.eval_seeds` swapped for the 15 seeds v6g r660 won on two or more
decks, `gold/results/seed_pool_wins.csv`). v6dt best.pt, temperature 0, Gold, all 15 decks: 225 games, 0 errors.
- **Won 0 of 225**, including 0 of the 43 cells r660 won. Mean 7.83 blinds a game (Red on these seeds 7.20; v6dt
  on Red Gold's 600 eval seeds 6.98, test S).
- **Claude's prediction, made in chat on 11-25 ~23:25 before the run (not committed): v6dt wins at most 3. Met.**
- Reading: these seeds are not won by any policy that reaches them. The wins need what Gold training added to v6dt.
  This does not fully separate seed from policy: v6dt is r660's parent and won nothing at Gold anywhere. A strong,
  unrelated policy on the same seeds would (v6gd's checkpoints are the nearest candidate).

## Added 2026-11-27, after chud's 13:52 CST message: v6gd scored against v6g: the Red specialist wins

Claude's, reviewed by chud. v6gd ran 2026-11-26 18:18 to 2026-11-27 ~10:20 (chud's Ctrl+C at iteration 766, 16.0 h;
the stall rule had not fired; in-run best 9.57 blinds at 700, mixed decks ~40 games each). Scored as registered at
R = 660 by `balatro-work/gold/score_v6gd.py` (not in the repo): v6g's `ckpt_00660` first replayed 20 Red seeds
move for move against `gold/v6g_t0_s8_red.jsonl` (20/20, so v6g's existing best.pt files are round 660). Then
v6gd's `ckpt_00660` played the first 600 eval seeds on each of the 15 decks at Gold, temperature 0, 0 errors.

| Deck | v6g blinds | v6gd blinds | diff | v6g wins | v6gd wins |
|---|---|---|---|---|---|
| Red | 9.923 | 9.493 | -0.430 | 0 | 0 |
| Blue | 11.417 | 11.265 | -0.152 | 1 | 1 |
| Yellow | 11.165 | 10.847 | -0.318 | 0 | 0 |
| Green | 11.535 | 10.985 | -0.550 | 2 | 2 |
| Black | 6.722 | 6.165 | -0.557 | 0 | 0 |
| Magic | 9.580 | 9.783 | +0.203 | 0 | 0 |
| Nebula | 9.182 | 8.975 | -0.207 | 0 | 0 |
| Ghost | 10.545 | 10.315 | -0.230 | 0 | 0 |
| Abandoned | 9.795 | 9.455 | -0.340 | 1 | 0 |
| Checkered | 10.757 | 10.587 | -0.170 | 0 | 1 |
| Zodiac | 9.677 | 9.207 | -0.470 | 0 | 0 |
| Painted | 10.493 | 10.118 | -0.375 | 0 | 0 |
| Anaglyph | 9.297 | 8.867 | -0.430 | 0 | 0 |
| Plasma | 10.228 | 10.165 | -0.063 | 0 | 0 |
| Erratic | 10.062 | 9.720 | -0.342 | 0 | 0 |
| **All 15** | **10.025** | **9.730** | **-0.295** | **4** | **4** |

Paired over the 9,000 seed × deck games: v6gd minus v6g = -0.295 blinds (sd 3.76, z -7.45).

- **D1** (pooled blinds, v6gd higher, 70%): **missed**, and the other way: v6g is higher, z -7.45.
- **D2** (Red: v6g at least v6gd, 55%): **met** (9.92 vs 9.49).
- **D3** (v6gd higher on at least 11 decks, 50%): **missed**: higher on 1 (Magic, +0.20).
- **D4** (pooled wins, v6gd at least v6g, 55%): **met**, 4 vs 4. Too few wins to read either way.

Reading: at equal games (660 rounds of 96), training only on Red beat training on all 15. It won on 14 of
15 decks, including the 14 it never trained on. The mixed run saw about 4,200 games of each deck against v6g's
63,000 Red games, and still lost on the decks it trained on. So more games of a deck did not buy skill on it at this
scale. Claude's main prediction was wrong.

What this does not settle:
1. **One run each.** z -7.45 is over games, not training runs. Run-to-run variance (same config, another
   training seed) is unmeasured, so part of -0.3 could be the luck of either run. A second v6g or v6gd would size it.
2. **Encoding 2 has no deck token.** v6gd had to learn one policy for 15 decks it can tell apart only from what it
   sees (hand size, starting cards, money). A deck token (encoding 3) is the obvious next arm. It is a different
   question from this one.
3. **One checkpoint.** R = 660 was registered. v6gd's best in-run eval was 700, on ~40 games per deck; not tested.

The registered reading for a D1 miss stands, stronger: the deck mix is not the lever at this scale. Boss play
and xMult (the walls report) stay first. Red-only stays the default for the next Gold run unless a deck-token arm
or a second seed says otherwise.

## Added 2026-11-27, after chud's 13:58 CST message: issue #138 found (Blackboard reads the hand on screen), fix registered

Claude's, reviewed by chud. Written after the cause was found with a probe in a cloud session (game source from
chud's bundle, LuaJIT 2.0.5), before the fix is committed. The guess in the 2026-11-26 10:55 section (card
`debuff` missing from the key) is wrong: `build_power_fresh` clears every playing card's debuff, so neither the
cached nor the fresh value sees a boss's debuffs.

**Cause.** On ZZ6JHUKR the board at move 22 is Gros Michel, Swashbuckler, **Blackboard**, Business Card. The key
at move 22 (SELECTING_HAND) equals the key at move 20 (blind select), but the fresh value fell from 1.2129 to
0.8167. With `G.hand.cards` emptied at move 22, fresh gives 1.2129 again. In `score()` the deal's held cards are
put in `G.hand.cards` only for the held-card step, and the real hand is put back before the joker step.
Blackboard's joker_main reads `G.hand.cards` (card.lua: X3 if every held card is a Spade or Club). So:
- outside a round (shop, blind select, packs) the hand is empty, and Blackboard gave X3 on every deal;
- in a round it read the real hand, the played cards included.
The key doesn't hold the hand, so the cache also went stale between those states. The 1.2129 was the wrong value.

**What it touched.** On any board with Blackboard: phi_build (v4+ build shaping), the offer values input (a
Blackboard on offer was valued as a free X3 in every shop), and the play values input (v5+, `score(..., live)`).
In the game, joker_main runs with the played cards already in G.play, so G.hand.cards is the held cards.

**The fix.** `score()` keeps `G.hand.cards` = the held cards through the jokers and the final step, and puts
the hand back just before it returns. Same for both callers. `tests/v4_phi_check.py` gets one more invariant
every 25th move: phi_build is the same with the hand on screen emptied (catches any other joker that reads
G.hand). It goes back into the pre-flight of the next run.

**What it doesn't change.** v6g and v6gd ran the old code (tags/launch commits keep it); their results stand as
measured under it. r660's weights were trained on the old inputs, so an r660 eval on the fixed harness differs on
Blackboard boards: evals of r660 are reported with the harness commit.

**To measure (results go in a new section).** On every Gold route in `wins/routes.jsonl`, old vs fixed code side
by side: the share of moves (and of moves with Blackboard held) where phi_build or an offer value differs, and by
how much; and that the fixed code has cache = fresh on all of them. Claude's prediction (made here, before the
sweep finishes): differences only on boards holding Blackboard; cache = fresh everywhere with the fix.

## Added 2026-11-27, after chud's 13:58 CST message: issue #138 measured, old code vs fixed

Claude's, reviewed by chud. Cloud session, game source from chud's bundle (1.0.1o), LuaJIT 2.0.5. Every Gold
route in `wins/routes.jsonl` (181, all 15 decks) replayed twice, at e69ea7f and with the fix. At every move it
records progress phi_build (what training reads), a fresh value, and the offer values. Probe scripts are not in the
repo (sweep in the PR description).
- 49,957 moves; Blackboard held at 1,563 of them, on 15 of the 181 routes.
- **phi_build differs at 1,473 moves, all with Blackboard held**; largest difference 0.79 (phi is in [0, 3]).
- **Offer values differ at 240 moves**: 187 with Blackboard held; 53 without, all in a shop or Buffoon pack (43 of
  them the same old value +0.3962 on different boards: Blackboard on offer, valued as a free X3; one checked by hand,
  134LAWJ6 Red move 39: shop Riff-raff, Blackboard; old +0.3962, fixed 0). Largest 0.396.
- **Cache vs fresh**: old code 267 stale moves, all with Blackboard held; fixed code **0**.
- `tests/v4_phi_check.py --routes 6` (with the new hand-on-screen invariant): ALL PASS, 86 of 86 offers move
  phi_build by their value.
- `tests/v5_play_check.py --routes 3`: old 101 of 102 plays exact, the miss ZZ6JHUKR move 22 (play 3,4,5,6: S 720,
  game 2160: the played cards counted as held, so no X3); fixed **102 of 102**.

Prediction (2026-11-27 13:58 section): cache = fresh everywhere with the fix: **met**. Differences only on boards
holding Blackboard: **missed in part**: also when Blackboard is on offer and not held (53 moves).

Reading: the gap was Blackboard alone on these routes, and v6g's build signal was wrong only around it: about 3% of
moves on Gold wins, 8% of the routes. That sample is wins, so it may undercount training games that bought
Blackboard on the inflated value. v6g's saved games would size that; not run. Small next to the walls (bosses,
xMult); it doesn't explain v6g's Gold rate. The fix goes into the next run's pre-flight with `v4-phi` back on.

## Added 2026-11-27, after chud's 14:56 CST message: money earned in one valued play leaked into the next (fix registered)

Claude's, reviewed by chud. Found by the per-joker audit (PR for `tests/joker_audit.py`, its 14:56 section): 3
of 150 jokers missed outside the labeled set, every one with Bull or Bootstraps on the board (j_ancient t7: Bootstraps
and Marble, $4, play A K 4 2 as High Card: S 208, game 16; j_ice_cream t11 with Bootstraps: S 8320, game 2240;
j_bull t8: S 352, game 136).

**Cause.** A card that pays during scoring (Gold Seal, and in card.lua Rough Gem, Business Card, Golden Ticket,
Reserved Parking, Lucky Card's money, and others) adds to `G.GAME.dollar_buffer` at once and clears it through an
event. The sim's valuations stub events (`G.E_MANAGER.add_event = NOOP`), so the buffer is never cleared inside one
valuation: `H.play_scores` values every legal play in turn, and each inherits the money of the plays valued before it;
`build_power_fresh` does the same over 12 deals x every hand type. Bull and Bootstraps read dollars + buffer.
Both restore `G.GAME` scalars only after the whole loop.

**What it touched.** On boards holding Bull or Bootstraps with any money-paying card in the deck or on a joker:
play values (v5+), phi_build and offer values (v4+), inflated. In the game, the buffer holds only the play being
scored, so a Gold Seal scored earlier in the same play does count for Bull later in it.

**The fix.** `score()` saves `G.GAME.dollar_buffer` on entry and puts it back before it returns: money earned
inside one play still counts for jokers later in that play, never for another play or deal.

**Prediction (Claude's).** With the fix: the audit's 3 misses are exact; the full audit (150 x 12, same seeds) has
0 unlabeled misses: 75%; Bull and Bootstraps at 60 trials each: all exact: 80%. `v4_phi_check`, `v5_play_check`
still ALL PASS.
