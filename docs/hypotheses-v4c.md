# Hypotheses: Model v4c

New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-13 ~18:45 CST: model v4c, registered before any of its code

v4b, warm-started, with one kind of change: **the bot sees more clearly how the shop helps it win.** The reward,
the shaping, the credit and the sampling stay the same. No move is forced. Discussion and evidence: issue #124 and its
comments.

### Why (the diagnosis)
v4b's eval games end around ante 4–5 (best eval 13.40 blinds/game, r1840; window r1700–1880 13.09). Its
joker slots are full by ante 4 (walls report, PR #122). From then on, the build's makeup stops changing: xMult
jokers hold at about 0.26 per game at every ante, and games that die late differ from games that pass at the
same blind only in hand level. Blind targets grow faster than hand levels add.

The reason is in what the bot sees. With full slots, `buy` isn't a legal move, so an upgrade means `sell_joker`
then `buy`, two separate decisions. `H.offer_values()` values only buys, picks and redeems; selling, rerolling and
opening packs all show 0. So from about ante 4, nothing in the bot's view says an upgrade exists. The economy test
(last sections of `hypotheses-v4b.md`) found that spending at rich late shops pays: a scripted spender gained
+0.293 blinds [+0.164, +0.426] over the bot's own move, mostly from swaps and rerolls. The bot's own one-step
view already ranked ending the shop last, yet the policy ended the shop 69% of the time.

### What changes
1. **A `swap` move: action list `ordered-capped-swap`.** This is `ordered-capped` plus, in the shop, one move per
   (held joker i, shop joker j), offered when:
   - j is a joker, and there's no room to buy it (the game's own buy-space check);
   - i can be sold (the game's own check), isn't Negative (selling a Negative frees no slot), and isn't Invisible
     Joker (selling it adds a joker, which fills the slot);
   - after i's sale, j is affordable by the game's own rule: cost ≤ money + i's sell value − the bankrupt line.

   It is applied as the game's sell, settled, then the game's buy, settled: the same end state as the two moves
   in turn. Swap moves are listed **after** every other shop move, so every other move keeps its index.
   Published `ordered-capped` routes therefore replay unchanged on this action list, which is what lets the
   curriculum keep using them. Action type 19; the move-type embedding gets a 20th row.
2. **Offer values for what showed 0 before.** Each is phi_build after the move minus phi_build now, computed by the
   same `H.build_power`. Each uses only what a player can see, or knows from the game's public rules, and never
   the seed's hidden next shop or pack.
   - `swap i>j`: i taken off the board and j added, with i's sell value added to money.
   - `reroll`: the expected best gain over a fresh shop. 8 shops are sampled, each with the shop's slot count.
     Each slot is a joker, Tarot or Planet at the game's own rates (`G.GAME.joker_rate`, `tarot_rate`,
     `planet_rate`). A joker's rarity is drawn at 70 / 25 / 5% (common / uncommon / rare), and the joker
     uniformly from the game's own pool for that rarity (`get_current_pool`, called with the rarity given, so it
     draws no random number). A Planet is drawn uniformly from the Planet pool. A sampled joker is valued as a
     buy if a slot is free, else as a swap against the held joker whose removal costs least. A Planet is valued
     as used. A Tarot is valued at 0, as it is everywhere. A card counts only if it's affordable after paying for
     the reroll. A shop's value is its best card's gain, floored at 0. The sampler is a fixed generator seeded
     from the board, separate from the game's random numbers, so a state always gets the same value.
   - `open_pack`: Buffoon packs get the expected best joker among the pack's cards, and Celestial packs the
     expected best planet, by the same sampling (8 samples). Other packs stay 0 (phi_build doesn't count what
     they hold).

   Sampled cards are valued in their base edition.
3. **A warm start from v4b's `best.pt` (r1840)** with a fresh optimizer, at round 0. Every weight loads as is; the
   swap row of the move-type embedding starts as a copy of `buy`'s. The shop entropy coefficient is **pinned at
   0.001**, v4b's floor: the schedule counts decisions from the start of a run, and a warm start would otherwise
   reset it to 0.02. That would be a second change. Everything else is `configs/v4b.toml`: reward `blinds`,
   `build_shaping 1`, gamma/lambda clocks, lr 1e-4, curriculum, Red Deck, White Stake, the 600-seed eval every 20
   rounds, `stall_evals 25`.

### What doesn't change, and the is-ought aim
The reward, `build_shaping`, the credit and how moves are sampled all stay as they were: no exploration, no
forced moves (chud, 18:30: early shops are very important, and forcing desperation there can hurt badly). The swap
is two legal moves offered as one. The offer values are information about the shelf, the pool and the packs. One
caveat: they are measured in phi_build, the same measure `build_shaping` uses, so the bot sees its options in the
shaping's terms. That is already true of buys in v4 and v4b, and the values would stay fixed if a later
experiment varied the reward.

### Costs, and a gate before launch
- Sim time: the swap values add about 10 `build_power` calls per full-slot board. The reroll and pack values add
  about 20 more per board, each cached by board. Estimate: 1.3–1.6× v4b's sim time per decision.
  **Gate:** if the smoke shows sim time per decision above 1.6× v4b's, the reroll and pack samples are cut from 8 to
  4 before launch, and the cut is recorded in a new section. No other knob is turned.
- The warm-started network has never seen a nonzero offer value on a reroll or pack, or a swap move. It has to
  learn what they mean, starting from v4b's policy.

### Predictions
- **P1 (chud's):** v4c beats v4b's best eval, 13.40 blinds/game, on the same 600-seed eval, before its stall stop.
  Claude's odds: **55%.**
- **P1b (companion, Claude's):** the mean of some window of 5 consecutive v4c evals is above 13.40. One eval has
  noise of about ±0.2, so a v4b-strength policy can reach 13.40 once by luck. P1b can't be met that way.
  Claude's odds: **30%.**

### Measured, not predicted (v4c's evals against v4b's `best.pt` eval games)
Win rate (chud's goal metric). Swaps, rerolls per shop visit and packs opened by ante. xMult jokers held by ante.
Joker counts, sells and money at antes 1–3: early play is a baseline only, and no prediction says it should
buy more. Sim time per decision.

## Added 2026-11-13 ~18:45 CST: whose the 18:45 sections are

chud's, reviewed by Claude:
- the diagnosis that the bot is stuck at ante 4–5 because its purchases only build to there;
- clarity without touching the ought;
- no forced moves, and early shops left free;
- P1, and the goal of win rate;
- the call to build v4c.

Claude's (model: Claude Opus 5.5, `claude-opus-5-5`), reviewed by chud:
- the code reading (no buy with full slots; offer values only on buys, picks and redeems);
- the swap move, and listing it last so routes replay;
- the pool-expectation offer values for rerolls and packs, and their budget;
- the warm start and the pinned entropy;
- the gate, and P1b with both sets of odds.

## Added 2026-11-13 ~20:45 CST: smoke 1 stopped in pre-flight; two pool-card bugs fixed

Claude's (model: Claude Opus 5.5, `claude-opus-5-5`), reviewed by chud. A code fix, no design change: the
registration, the reward and the gate are unchanged.

The first smoke (launch 96703f9, 2026-11-13 19:24) stopped in pre-flight: `v4c-swap` crashed at the first
reroll value, `card.lua:981` indexing a nil `base`. This was the part flagged as least certain: pool cards are
built with `set_ability` alone. Fixed in `harness/build_phi.lua` `pool_card`:
1. **No base.** `Card:init` gives every shop joker `set_base(G.P_CARDS.empty, true)` (nominal 0) after
   `set_ability`; `get_chip_bonus` reads it. The pool card now gets the same, plus init's other plain fields.
2. **The game was changed.** With 1 fixed, the check then caught that `set_ability` sets
   `G.GAME.used_jokers[key] = true`, which would have removed every sampled joker from the seed's later shops.
   `pool_card` now lets it write to a throwaway table that reads through to the real one.

Run on the pinned 1.0.1o source (3 routes, 4 swaps each): all four game checks pass. 776 moves replay unchanged
on the swap list, 12 of 12 swaps equal a sell then a buy, 78 of 78 shelf jokers equal their pool card, and
346 of 606 reroll/pack values are nonzero (max 0.49). The warm-start check needs torch and runs in the launch's
own pre-flight.
