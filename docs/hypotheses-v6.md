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
