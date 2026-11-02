# Model v1: design

Goal of v1: the smallest learner that can show it learns from the sim (beats random, then the
greedy test player, on Red Deck / White Stake), built so later versions (Ghost / Purple, imitation
from logged runs, search) reuse it unchanged. Not a strategy: nothing in it says what is good.
What it should want is the reward, which is a setting (see Reward).

## 1. What the bot sees (observation)
Everything comes from one Lua encoder (`harness/encode.lua`) that applies the **same visibility
rules as `H.observe()`**: face-down cards and jokers masked, the deck as counts (never draw order),
no RNG state. Output is a flat number buffer passed to Python in one string (no per-field Lua→Python
conversion, which was ~10% of step time).

Tokens, each a fixed-width feature row plus a kind id and a vocabulary id:

| kind | count | features |
|---|---|---|
| GLOBAL | 1 | state, ante, round, stake, money, hands/discards left, hand size, joker/consumable slots, reroll cost, interest cap, blind target, chips scored, chips needed ratio, pack choices left |
| DECK | 2 | remaining deck and full deck as counts: 13 ranks, 4 suits, 8 enhancements, editions, seals, total |
| HANDTYPE | 12 | level, chips, mult, times played, played this round, visible |
| BLIND | 1–4 | current blind (key, target, boss, disabled); at blind select also Small/Big/Boss choices and the two skip tags |
| CARD (hand) | ≤ hand size | rank, suit, enhancement, edition, seal, debuff, bonus chips, face-down flag (face-down: everything else zeroed) |
| JOKER | ≤ slots | key, edition, sell value, eternal/perishable(+tally)/rental, debuff, face-down, up to 8 numeric ability fields (log-scaled, in the joker's own field order) |
| CONSUMABLE | ≤ slots | key, edition, sell value |
| SHOP / VOUCHER / BOOSTER / PACK | as shown | key or playing-card fields, cost, edition |
| TAG, OWNED VOUCHER | as held | key |

Vocabulary ids come from the game itself (`G.P_CENTERS`, `G.P_TAGS`, `G.P_BLINDS`, hand types,
seals, editions), sorted, saved with every checkpoint and checked on load.

## 2. What the bot can do (actions)
`subsets` mode plus `move_joker`. An action is 9 small integers, indices into the token list:
`(type, a, b, c1..c5, hand_type)`. `a`/`b` point at the item (shop slot, joker, pack card,
destination slot), `c1..c5` at selected hand cards, `hand_type` is the poker hand the selection
makes, which the game shows the player on selection (`G.FUNCS.get_poker_hand_info`, a pure
evaluation). Actions are scored, never enumerated as fixed outputs, so any number of legal actions
works.

Known v1 limits: played cards score in hand order (the game's own default sort); deliberate hand
reordering (Photograph-style plays) needs controls mode, later. No boss reroll, no endless.

## 3. Network (~250k parameters)
- Token embedding: per-kind linear on features + vocab embedding + kind embedding, d = 96.
- 2-layer transformer encoder, 4 heads, feed-forward 192, over all tokens (masks for padding).
- Action embedding: type embedding + projections of the gathered token embeddings
  (`a`, `b`, mean of `c1..c5`) + hand-type embedding.
- Policy: MLP([action, pooled state]) → one logit per legal action; softmax over legal actions only.
- Value: MLP(pooled state) → expected return.

## 4. Training (PPO, parallel)
- W worker processes (one per core, `torch.set_num_threads(1)`), each with its own sim. Each round:
  workers play G games with the current weights, sampling actions; the learner runs PPO
  (clip 0.2, GAE λ 0.95, γ 0.999, entropy bonus, advantage normalization), then broadcasts weights.
- Storage per decision: tokens (float16) + actions (int16): ~10 KB, so a 15k-decision round is
  ~150 MB.
- Episode cap: 1,500 decisions (guards against `move_joker` loops); a capped game ends with no
  further reward and is counted.
- Seeds: training seeds random; evaluation seeds a fixed disjoint list; every result records
  its seeds.

## 5. Reward (the ought: the user's decision)
A named setting, recorded in every checkpoint and every result file:
- `blinds`: +1 per blind beaten (placeholder for the smoke test)
- `antes`: +1 per ante cleared
- `win`: +1 for beating the ante 8 boss, 0 otherwise
New ones are a few lines in `train/rewards.py`.

**Shaping (not part of the ought).** With the reward alone, near-random early play beat a blind
once in 640 games and nothing was learned. So training adds potential-based shaping (Ng, Harada &
Russell 1999): `gamma * phi(next) - phi(now)` with `phi` = the fraction of the current blind's
chips scored, 0 outside a blind and at the end. It rewards progress inside a blind and takes it
back in full when the blind ends; over a game it sums to ~0, and the theorem guarantees the optimal
policy for the chosen reward is unchanged. Scores are always reported on the unshaped reward.
`--shaping 0` turns it off. Verified: blinds beaten per game rose from 0 to ~2 within ~1,100 games.

## 6. Evaluation
Greedy policy (argmax) on the fixed evaluation seeds: final-ante distribution, win rate, blinds
beaten, all against random and the greedy test player on the same seeds.

## 7. Reproducibility
Each checkpoint stores: weights, vocabulary, config (deck, stake, reward, mode, game modded or
not), repo commit, `Balatro.exe` hash from the bundle, Python/torch versions, and the seeds used.
