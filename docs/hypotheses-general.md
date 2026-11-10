# Hypotheses: General: aims, the main experiment, corrections, standing decisions

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## B. Shaping is speed, not values
- **B1.** Potential-based shaping changes how fast the bot learns, not what it ends up preferring:
  two runs identical except `--shaping 1` vs `--shaping 0`, given enough games, converge to the
  same blinds/game within noise. Practical test: compare at a fixed game budget once compute
  allows the unshaped run to learn at all. Until then, the claim rests on the theorem
  (Ng, Harada & Russell 1999), not on our data.

## C. Isolating oughts (the main experiment)
Facts held fixed: the game (pinned version, same seeds, same sim). Varied: the reward, the only
place an ought enters.

- **C1. Different oughts, different behavior.** Bots trained on `blinds`, `antes` and `win` play
  measurably differently on the same seeds: skip rate, reroll rate, money held, joker picks.
  Test: per-seed paired comparison of those statistics. *(chud to confirm the statistics)*
- **C2. An ought that isn't the goal can beat the goal at the goal.** Judged on win rate, a bot
  trained on `blinds` or `antes` wins at least as often as one trained on `win` at the same compute.
  The yardstick (win rate) is itself an ought and is named as such in any write-up.
- **C3. Bot vs chud.** On held-out paired seeds (Ghost Deck, Purple Stake, modded 1.0.1o), the
  bot's win rate is compared with chud's: McNemar test on per-seed outcomes. Direction not
  predicted. *(chud to confirm N, the held-out split and the significance level)*

## D. Before the main training (what still has to be done)
1. **Reward of record** for the main experiment (chud's decision).
2. **Game of record**: modded (matches chud's logged games) or vanilla. Proposed: modded.
3. **Target config**: Ghost Deck, Purple Stake; env and eval switched over and re-verified.
4. **Play order**: controls mode (or ordered selection), so plays like Photograph-first are expressible.
5. **Endless mode** in the env (needed for naneinf work; not for the main experiment).
6. **Eval loop** understood and handled (A3), so evaluation is trustworthy at scale.
7. **chud's dataset pipeline**: logs → verified (state, legal actions, chosen action) records,
   each log replayed with 0 mismatches before it's used.
8. **Pre-registration numbers** for C3: N, held-out split, test, threshold.
9. **Compute plan** for 1M+ games on the target config (laptop, or rented x86 cores).

---

## Added 2026-11-02 ~23:05 CST: the project's aims (stated by chud)

### Authorship
The three aims are chud's, in the order he gave them (23:01–23:03 CST). The write-up is Claude's.
Nothing above is edited. This section names what the sections above are for.

### 1. Primary: exhibit Hume's is-ought gap
This is a thesis on agents (chud, 23:10: "I am an agent researcher"). The bot is the agent under
study; Balatro is the domain. Hold the facts fixed (the pinned game, same seeds, same sim) and vary
only the ought (the reward).

**Oughts are objects, like jokers.** A player buys a joker without knowing whether it will win the
run, and buys anyway. An agent holds an ought the same way. In chud's words (23:09):
- "we can't know if the joker will let us win, but we buy stuff anyways"
- The bot loses a ton, but it is calibrating, as humans do. "a human loss is when their perception
  of reality isn't actually reality." They lose all the time and keep revising. Humans have
  weights too.
- "there are only specific domains and the deep structure that can be generalized from them."

**Scope of the grouping** (chud, 23:11: "I am a grouper ... I am an agent psychologist"). Humans and
models are grouped at the level of behavior, the level psychology studies, not of mechanism.
- **Claimed:** both are agents whose weights are revised by prediction error. Evidence at that
  level: dopamine neurons signal reward prediction error (Schultz, Dayan & Montague 1997), the
  same quantity as the TD error the bot's critic learns from.
- **Not claimed:** the same learning algorithm (brains learn by local rules, not backpropagation
  as such), or the same timing (humans revise while acting; a deployed LLM's weights are fixed;
  the bot revises in training and is frozen at eval).
- **Precedent:** comparative psychology groups agents across substrates by shared learning laws,
  e.g. the Rescorla-Wagner model (1972), a prediction-error rule applied across species.

The idea and the framing are chud's (see also the clarification at ~11:05). Sections C and G test
the first half: same facts, different oughts.
- Status: in C and G the ought is picked by the experimenter for a whole run; the bot doesn't buy
  it. A design where the bot acquires reward terms during play, judged by an outer yardstick,
  would test the joker claim directly. Not registered.
- Status: no registered hypothesis yet tests whether strategy over oughts generalizes.
- Related prior work: the optimal reward problem (Singh, Lewis & Barto, "Where Do Rewards Come
  From?", 2009): for a bounded agent, the best reward to give it is often not the designer's goal.

### 2. Prove a naneinf seed
A seed plus a route that replays move for move to a played hand after which `G.GAME.chips` is not
finite, on the pinned game. The measure, what has to be built first (endless mode, a reward for
it, inf/nan-safe scoring and shaping, a longer move cap, the encoding at inf/nan) and the
community seeds checked so far are in `naneinf/README.md`. A proven seed goes in `wins/` with
its route, like every beaten seed.
- Whether a route has to come from a trained model playing only what a player sees (the standing
  goal in `naneinf/README.md`), or a route found any other way also proves the seed: *(chud to confirm)*.
- No prediction is registered.

### 3. Third: generalizability to 100% completion from a new save
A model that starts from a fresh profile (nothing unlocked) and plays it to 100% completion.
- What counts as 100%: *(chud to define)*.
- Requires the env to carry one profile's unlock and discovery state from run to run.
- No prediction is registered.

---

## Added 2026-11-03 ~18:00 CST: roadmap, and what 100% completion means (stated by chud)

### Authorship
chud's, 17:59 CST: "so we're going to get the moe working good, then we're going to send it up the
decks and stakes and then I ultimately want a single model to be able to solve a single entire
balatro save where all it's gotta do is win with every joker on gold stake and gold stake every
deck and beat all the challenges too." The write-up and the status notes are Claude's. Nothing
above is edited.

### Roadmap
1. The mixture of experts (section I) working well on Red Deck, White Stake.
2. Up the decks and stakes.
3. One model completes one save (aim 3, which was *(chud to define)*). 100% means:
   - a Gold Stake win with every joker (each joker's gold sticker);
   - Gold Stake beaten on every deck;
   - every challenge beaten.

### Open
- Whether the mixture of experts (one checkpoint, nine networks, one router) counts as "a single
  model": *(chud to confirm)*.

### Status of what this needs (Claude's notes, 2026-11-03)
- Have: deck and stake are settings (`BACK`, `STAKE`); joker stickers (eternal, perishable,
  rental) are already in the encoding.
- Not yet:
  - Higher stakes aren't covered by the differential check. Challenges and Erratic Deck aren't
    covered by the harness at all (README, harness validation).
  - Every run starts with everything unlocked (`unlock_all`). A save needs one profile's unlocks,
    stake progress and stickers carried from run to run.
  - A joker's sticker needs a run won while holding that joker. No current reward contains that,
    so finishing a save needs something that picks which run to play next and what it is for.
- Counts (150 jokers, 15 decks, 8 stakes, 20 challenges) are Claude's from the game, to be checked
  against the pinned source before any plan is sized on them.

### Answered (2026-11-03 ~18:01 CST, chud)
"duh the moe counts. I'll be training the early game and the experts continually until a SINGLE
moe weight configuration can 100% balatro." The mixture of experts is the single model. The goal
is one MoE checkpoint that completes the save on every deck, stake and challenge.
- Note (Claude): the encoding gives the stake as a number but not which deck is in play (or which
  challenge). One checkpoint across decks needs that as an input. The deck keys are already in
  the vocabulary, so weights can carry over; it would go in as a new encoding version, so the
  published routes, which are checked against encoding 2, keep verifying.

### Architecture change (2026-11-03 ~18:32 CST; chud: "go fix what you can fix"; the changes are Claude's)
The MoE as built at 17:54 (above) is replaced before any of it ran:
- One shared trunk and 9 heads, in place of 9 separate networks. Each head learns only from its
  own decisions; the trunk learns from all of them.
- Trained jointly from the start, from model 0.2's weights; solo expert and early-game runs are
  optional warm starts for the heads they cover.
- The early head's returns end at the handoff, so inside the joint run it is judged only on
  reaching ante 4 (chud, 17:43: "make it's win rate only to get to 4"). It trains by default.
- Encoding 3 from the start; evals on 600 seeds; several seeds of the whole MoE.
Unchanged: the router rule (chud, 17:54), the 30% random handoffs, the pins and their annealing,
the natural-bias measure.

---

## Added 2026-11-03 ~20:50 CST: correction: final-boss losses were counted as wins

Nothing above is edited. This section says which results above stand, which are void as scored,
and how they are re-scored.

### What was wrong
- **The game's flag.** Balatro 1.0.1o sets `G.GAME.won = true` at the end of the Ante 8 boss round
  whether or not the boss was beaten (`end_round`, `functions/state_events.lua`). On a loss it
  then goes to `GAME_OVER` and never calls `win_game()`, so nothing is recorded.
- **The harness** read `G.GAME.won` alone. Every count of wins since the v1 run included runs that
  ran out of hands on the Ante 8 boss.
- **The reward** `blinds` paid that loss as a beaten blind, so the final fight carried no signal:
  losing it scored 24 blinds, the same as winning it. Nothing else in the reward was affected.
- **Found by** chud's save-progress ledger (~20:12 CST), which reads the game's own record of
  each win: the game had credited nothing to 1,114 of the 1,549 published wins. Claude traced the
  cause in the game source and confirmed it on `4XNRUA21` (the proof seed): `GAME_OVER`, 82,500
  of 100,000 against Verdant Leaf, 0 hands left.

### Fixed (CHANGELOG 0.13.2–0.13.3)
- A win is the game's own test: `G.GAME.won` and not `GAME_OVER`. A final-boss loss is no longer a
  beaten blind.
- The 1,549 published routes were replayed with the corrected test (`tools/recount_wins.py`, run
  by chud): **435 wins** (0.1: 24, 0.2: 179, 0.3: 232). The **1,114** losses stay posted as
  final-boss near misses (`wins/near_misses.jsonl`, chud: "we keep them posted").
- Near wins (reached the Ante 8 boss, lost it) are logged in every game, metric and eval from now
  on, next to wins (chud: the win rate against the reach-the-boss rate).
- The pre-flight's saved "win" (v1 round 250, `9MN9TD5T`) was itself a final-boss loss and was
  replaced by a real v1 win.

### Status of the results above
- **F1, H1, R0–R8: void as scored.** Each counted final-boss losses as wins, in both arms.
- **A1–A7, E1, G, I1:** not scored yet; when they are, it is with the corrected test.
- **Eval figures** (1.0's 5 of 200, every eval's blinds per game) include the same error: a
  final-boss loss counted as a win and as 24 blinds instead of 23.
- **`docs/won_games_analysis.md`** described 0.2's saved "wins"; most were final-boss losses.

### Re-scoring (registered before any re-scored number is read)
F1, H1 and R are re-scored exactly as registered: same windows, same arms, same seeds, same tests,
same thresholds. The only change is the win test. The error only ever added wins, so only the games
each test counted as wins are replayed (from each run's saved `wins.jsonl` and the resolve run's
`resolve_1.0_wins.jsonl`), each with the corrected test. Denominators (games played) don't change.
Each re-scored result is added as a new dated section, with the old figure beside it.
- For F1 and H1 a "new beaten seed" is re-derived from real wins only, including the sets of seeds
  beaten earlier that F1 excluded.
- Publishing kept one route per seed (the shortest claimed win), so a seed can have a real win in a
  run's files behind a shorter final-boss loss. The re-scoring replays every saved claimed win, and
  any real win found that way is published.

### Decision recorded (chud, 20:41 CST)
The mixture of experts (section I) is trained from model 0.2's weights, as built, under the
corrected reward. Claude's case, which chud accepted: the error touched only the final fight, so
0.2's play up to the Ante 8 boss was learned on a correct signal, and the early head (antes 1–3)
never saw the error.

### Before launch (2026-11-03 ~21:15 CST)
- **Experts: 7** (chud, 21:11: "no straight flush"): Straight Flush dropped. The router doesn't count
  Straight Flush plays, as it doesn't count High Card.
- **A minimum of training per expert** (chud, 21:11: "we need a minimum amount of training, and the
  more common ones will just get more games"). Each expert's pin can't finish fading before the
  expert has had 5,000 training games of its own. The number is Claude's, at chud's request ("what's
  your gut say?"): at the 30% random handoffs, the rarest expert gets roughly 200–250 games an
  hour, so about a day.
- **Outcomes read from the game itself** (chud: tie the harness to the game): win = the game's own
  `win_game()`; blinds and bosses = the game's own `Defeated` marks. Checked step for step against the
  previous counters on 141 replayed games, with no difference.

---

## Added 2026-11-09 ~18:17 CST: the Full House venture closed, and the bar for going up the decks and stakes

Both decisions are chud's (18:17 CST), made while v4 trains and before any v4 eval has been read. The
write-up is Claude's. Nothing above is edited.

### The Full House venture is closed
chud: "Close it because strictness of the independent variable being training. Itd muddle data to force. I
assessed itd be easy and it wasn't. I no longer care to prove full house viability."
- **Why:** the independent variable is training (the reward, with the facts held fixed). Forcing a hand
  through the action mask is a second influence on play, and it muddles that.
- **Closed, unscored:** I2 (Full House among the strongest experts, 2026-11-04 ~13:35); the completion
  criterion "the MoE has learned Full House" (2026-11-04 ~13:40), not met at pins zero (2026-11-05 ~15:40);
  the `moe2-nopin` control (never run).
- **From here no run forces a hand.** How often a bot plays Full House is still reported wherever hand mix
  is reported; it is no longer tested.
- **Results already scored stand:** the pins-zero findings and C1–C5 of the paired run.

### The bar for going up the decks and stakes
chud: "Lets say 25% eval win rate on red deck white stake. Open to revision from v4's runs data."
- **Bar:** a 25% win rate on the 600-seed eval, Red Deck, White Stake, unmodded 1.0.1o; win = the game's
  own `win_game()`. This is the number the 2026-11-03 roadmap's step 1 ("working well") never had.
- **A decision rule, not a prediction.** It may be revised once v4's data has been read; a revision goes in
  a new dated section that says what was read.
- **For reference, not part of the bar:** `pins0.pt` 9 of 600 (1.5%) at pins zero; W2 is ≥10%; Claude's
  odds for v4 reaching ≥20% are 15% (2026-11-09 ~12:00).


## Added 2026-11-09 ~18:30 CST: hypotheses that will never be scored as registered are closed

chud (18:30): "Rescoring? It tests itself as it trains. Go close all the closed hypotheses as a separate md."
The rule (chud, 18:32): "If im not training that model the hanging hypotheses of that model are unanswered and
still closed." The ledger is [`closed_hypotheses.md`](closed_hypotheses.md): each hypothesis above, where it ended (scored;
void as scored; closed unscored, with why), and which are still open. Nothing above is edited.
- **The re-scoring of F1, H1 and R** registered on 2026-11-03 ~20:50 is closed, not done. They stay void as
  scored. Every run since that correction counts wins with the game's own test.
- **Still open:** v4's (W1–W4, aim 1's precondition), the 25% bar, and those tied to no one model: section C,
  B1, E1, aims 2 and 3.

## Added 2026-11-09 ~18:40 CST: the last three open settings, fixed

chud (18:39), on the three items still marked *(chud to confirm)*: "go ahead." The choices below are Claude's,
made at his request; he can override any of them before the data they apply to exists. Nothing above is edited.

### 1. Section C's first prediction: which statistics, and how it is tested
- **Arms:** rewards `blinds`, `antes` and `win`, all from the same starting checkpoint, same game, seeds, settings
  and number of training games. **Plus a replicate:** a second `blinds` arm with a different training seed.
  Without it, a gap between two oughts could be the run-to-run spread of one ought.
- **Starting checkpoint:** the first one that meets aim 1's precondition (ante 7 in ≥ 25% of 600-seed eval
  games), since below that the three rewards pay for almost the same things.
- **Statistics, per game:** (a) blinds skipped; (b) rerolls per shop visit; (c) money held when leaving a shop,
  averaged over the game's shop exits; (d) jokers bought, as a frequency over joker keys.
- **Seeds:** 2,000 from tag `c-test`, disjoint from every tag used so far. Each arm plays each seed once,
  sampled at temperature 1 with the per-seed RNG.
- **Test:** (a)–(c) by a paired permutation test per seed (10,000 permutations); (d) by a permutation test on the
  Jensen–Shannon divergence between the two arms' purchase frequencies, permuting arm labels within seed. Three
  pairs of oughts × four statistics = 12 tests, Holm-corrected, α 0.05.
- **Passes if** at least one statistic differs between two oughts after correction, **and** the same statistic
  does not differ significantly between the two `blinds` runs under the same test (uncorrected, α 0.05, which
  makes the replicate check strict). *Falsified* if no pair of oughts differs on any statistic.

### 2. C3, bot vs chud: size, split, level
- **Seeds:** 200 from tag `c3-test` (Ghost Deck, Purple Stake, modded 1.0.1o), never used in training or eval.
- **chud** plays each seed once as a seeded run, in the listed order, no restarts, without looking any seed up
  in advance. Each run counts as it ends.
- **The bot** plays each seed once, sampled at temperature 1 with the per-seed RNG.
- **Test:** exact McNemar on per-seed outcomes (win = the ante-8 boss beaten), two-sided (no direction is
  predicted), α 0.05. The 200 is set by what chud can play, not by power; if the discordant pairs are few,
  the result is reported as underpowered rather than as no difference.
- **Precondition:** a bot trained on Ghost Deck, Purple Stake, which waits on the 25% bar (2026-11-09 ~18:17).

### 3. Aim 2: what proves a naneinf seed
- **Any route that replays proves the seed**, however it was found, as registered for aims 2 and 3 on
  2026-11-05 ~23:01 (search track). The proof is the replay.
- **Every proven seed carries how it was found**, and the two are never pooled:
  - *model*: a trained model playing only what a player sees (the standing goal in `naneinf/README.md`);
  - *search*: lookahead over replayed branches, which proves a route exists, not that a player without
    foresight could find it.
- The same split applies to aim 3: a searched 100% save proves one exists; a model's save is the stronger claim.
