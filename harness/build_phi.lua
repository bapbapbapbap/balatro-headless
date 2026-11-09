-- Model v4's build-strength potential, phi_build, and each shop offer's value (docs/hypotheses.md:
-- v4, its 2026-11-09 amendment, and the definitions entry written before training).
--
-- B (board power) = the mean, over K deals of hand-size cards drawn from the whole deck, of the best
-- score any poker hand in the deal makes (each hand the deal contains, by the game's own evaluator,
-- played; the rest of the deal held), on a board whose scaling jokers are projected PROJECT_ROUNDS
-- rounds ahead. Scores come from the game's own scoring functions, called as G.FUNCS.evaluate_play
-- calls them, minus what belongs to the blind being played (its debuffs and modifiers), what a play
-- changes (the `before` context, destruction, dollars) and chance (pseudorandom returns its midpoint,
-- so chance effects at 1 in 2 or rarer never trigger).
--   phi_build = (clamp(log2(hands_per_round * B / T8), -10, 2) + 10) / 4   in [0, 3]
-- T8 = the ante-8 boss target: 0 at hands * B = T8 / 1024, 2.5 at T8, 3 at 4 x T8. One doubling = +0.25. Computed only from what a player can see:
-- the deck as a set (sorted by card, never draw order), jokers, consumables, hand levels, vouchers.
-- Computing it changes nothing in the game: every table it touches is restored (tests/v4_phi_check.py).
local H = ...

local K_DEALS = 16
local PROJECT_ROUNDS = 6
-- Per-round growth of scaling jokers (ability field -> per round). The planner's table
-- (roshantaneja/balatro-gym env/planner.py GROWTH, "rough community-consensus rates for a spam-hand
-- build"), mapped onto the real game's fields; the decays (Ice Cream, Popcorn, Ramen) are ours.
local GROWTH = {
  ["Green Joker"] = { mult = 2.0 }, ["Ride the Bus"] = { mult = 2.0 },
  ["Castle"] = { ["extra.chips"] = 9.0 }, ["Wee Joker"] = { ["extra.chips"] = 12.0 },
  ["Runner"] = { ["extra.chips"] = 4.0 }, ["Square Joker"] = { ["extra.chips"] = 12.0 },
  ["Spare Trousers"] = { mult = 0.5 }, ["Constellation"] = { x_mult = 0.04 },
  ["Hologram"] = { x_mult = 0.05 }, ["Vampire"] = { x_mult = 0.04 }, ["Lucky Cat"] = { x_mult = 0.05 },
  ["Glass Joker"] = { x_mult = 0.05 }, ["Campfire"] = { x_mult = 0.1 }, ["Obelisk"] = { x_mult = 0.05 },
  ["Yorick"] = { x_mult = 0.15 }, ["Flash Card"] = { mult = 1.0 }, ["Red Card"] = { mult = 0.5 },
  ["Ice Cream"] = { ["extra.chips"] = -20.0, floor = 0 }, ["Popcorn"] = { mult = -4.0, floor = 0 },
  ["Ramen"] = { x_mult = -0.1, floor = 1 },
}
-- Hand levels per round for the most-played hand (the planner's): Burnt Joker 1, Space Joker 1.
-- Supernova's count: that hand's plays + 3 per round.
local LEVELS_PER_ROUND = { ["Burnt Joker"] = 1, ["Space Joker"] = 1 }
local SPAM_PLAYS_PER_ROUND = 3

-- Vouchers: only what phi_build reads (hand size, hands, discards, joker slots, Observatory's flag).
local VOUCHER = {
  ["Paint Brush"] = { hs = 1 }, ["Palette"] = { hs = 1 },
  ["Grabber"] = { hands = 1 }, ["Nacho Tong"] = { hands = 1 },
  ["Wasteful"] = { discards = 1 }, ["Recyclomancy"] = { discards = 1 },
  ["Hieroglyph"] = { hands = -1 }, ["Petroglyph"] = { discards = -1 },
  ["Antimatter"] = { slots = 1 },
}

local NOOP = function() end
local STUBS = { "update_hand_text", "card_eval_status_text", "juice_card", "delay", "highlight_card",
                "play_sound", "attention_text", "check_for_unlock", "ease_dollars", "set_hand_usage",
                "discover_card" }

-- ---------------------------------------------------------------- serialization (cache keys)
-- Card fields that change during a round and that no scoring reads: left out of the keys.
local SKIP = { discarded = true, played_this_ante = true, forced_selection = true }
local function ser(v, out, depth)
  depth = depth or 0
  local t = type(v)
  if t == "table" then
    if getmetatable(v) ~= nil or depth > 6 then out[#out + 1] = "<o>"; return end
    local ks = {}
    for k in pairs(v) do if not SKIP[k] then ks[#ks + 1] = k end end
    table.sort(ks, function(a, b) return tostring(a) < tostring(b) end)
    out[#out + 1] = "{"
    for _, k in ipairs(ks) do
      out[#out + 1] = tostring(k); out[#out + 1] = "="
      ser(v[k], out, depth + 1); out[#out + 1] = ","
    end
    out[#out + 1] = "}"
  elseif t == "number" then out[#out + 1] = string.format("%.17g", v)
  else out[#out + 1] = tostring(v) end
end

local function card_sig(c)
  local o = {}
  ser({ c.base and c.base.suit, c.base and c.base.id, c.base and c.base.nominal, c.config and c.config.center
        and c.config.center.key, c.edition, c.seal, c.ability }, o)
  return table.concat(o)
end

-- the hand that counts as "most played" for the projection: most plays, ties to the weaker hand
local function spam_hand()
  local best, n = "High Card", -1
  for i = #G.handlist, 1, -1 do
    local h = G.handlist[i]
    local p = G.GAME.hands[h] and G.GAME.hands[h].played or 0
    if p > n then best, n = h, p end
  end
  return best
end

local function in_manacle()
  local b = G.GAME.blind
  return G.STATE == G.STATES.SELECTING_HAND and b and b.name == "The Manacle" and not b.disabled
end

-- opts: nil, or a change to evaluate (an offer): {joker = card} | {level = hand} | {voucher = card}
local function effective(opts)
  local e = {
    hs = (G.hand and G.hand.config.card_limit or 8) + (in_manacle() and 1 or 0),
    hands = G.GAME.round_resets.hands, discards = G.GAME.round_resets.discards,
    slots = G.jokers.config.card_limit, prob = G.GAME.probabilities.normal,
    dollars = G.GAME.dollars - (opts and opts.cost or 0),      -- an offer is paid for (Bull, Bootstraps)
  }
  if opts and opts.joker then
    local a = opts.joker.ability
    e.hs = e.hs + (a.h_size or 0)
    if (a.d_size or 0) > 0 then e.discards = e.discards + a.d_size end
    if a.name == "Turtle Bean" then e.hs = e.hs + a.extra.h_size end
    if a.name == "Troubadour" then e.hs = e.hs + a.extra.h_size; e.hands = e.hands + a.extra.h_plays end
    if a.name == "Stuntman" then e.hs = e.hs - a.extra.h_size end
    if a.name == "Oops! All 6s" then e.prob = e.prob * 2 end
    if opts.joker.edition and opts.joker.edition.negative then e.slots = e.slots + 1 end
  end
  if opts and opts.voucher then
    local v = VOUCHER[opts.voucher.ability.name] or {}
    e.hs, e.hands = e.hs + (v.hs or 0), e.hands + (v.hands or 0)
    e.discards, e.slots = e.discards + (v.discards or 0), e.slots + (v.slots or 0)
  end
  e.hs = math.max(1, e.hs)
  return e
end

-- Game counters that change every few moves enter the key only when a held joker reads them.
local COUNTER_READERS = { ["Bull"] = "dollars", ["Bootstraps"] = "dollars", ["Fortune Teller"] = "usage",
                          ["Throwback"] = "skips", ["Swashbuckler"] = "sell", ["Supernova"] = "played",
                          ["Burnt Joker"] = "played", ["Space Joker"] = "played",
                          ["Ancient Joker"] = "ancient", ["The Idol"] = "idol" }
local function counters(reads, o)
  local g = G.GAME
  ser({ reads.dollars and g.dollars, reads.skips and g.skips, reads.usage and g.consumeable_usage_total,
        reads.ancient and g.current_round.ancient_card, reads.idol and g.current_round.idol_card }, o)
end

-- the board's part of the key (everything but the offer); built once per observation
function H.build_key_base()
  local o, g = {}, G.GAME
  local sigs = {}
  for i, c in ipairs(G.playing_cards) do sigs[i] = card_sig(c) end
  table.sort(sigs)
  o[#o + 1] = table.concat(sigs, "|")
  local reads = {}
  for _, c in ipairs(G.jokers.cards) do
    local r = COUNTER_READERS[c.ability.name]; if r then reads[r] = true end
  end
  for _, c in ipairs(G.jokers.cards) do
    ser({ c.config.center.key, c.ability, c.edition, reads.sell and c.sell_cost or 0,
          (c.ability.perishable and (c.ability.perish_tally or 1) <= 0) and true or false }, o)
  end
  for _, c in ipairs(G.consumeables.cards) do ser({ c.config.center.key, c.edition }, o) end
  local hands = {}
  -- plays per hand: read by Supernova, and by the projection's most-played hand (Burnt, Space)
  for k, h in pairs(g.hands) do hands[k] = { h.level, h.chips, h.mult, reads.played and h.played or 0 } end
  ser(hands, o)
  counters(reads, o)
  ser({ g.starting_params.ante_scaling, g.starting_deck_size, g.used_vouchers, g.selected_back and g.selected_back.name,
        #G.jokers.cards, #G.consumeables.cards, in_manacle(), G.hand and G.hand.config.card_limit,
        g.round_resets.hands, g.round_resets.discards, G.jokers.config.card_limit, g.probabilities.normal }, o)
  return table.concat(o)
end

-- Everything build_power reads, as one string: the board, and the offer if one is being valued.
function H.build_key(opts, base)
  base = base or H.build_key_base()
  if not opts then return base end
  local o = { base }
  if opts.joker then
    o[#o + 1] = "+J"; o[#o + 1] = card_sig(opts.joker)
    local reads, r = {}, COUNTER_READERS[opts.joker.ability.name]
    if r then reads[r] = true; counters(reads, o) end
    if r == "played" then
      for _, h in ipairs(G.handlist) do o[#o + 1] = tostring(G.GAME.hands[h].played) end
    end
    if r == "sell" then for _, c in ipairs(G.jokers.cards) do o[#o + 1] = tostring(c.sell_cost) end end
  end
  if opts.level then o[#o + 1] = "+L" .. opts.level end
  o[#o + 1] = "$" .. tostring(opts.cost or 0)
  if opts.voucher then o[#o + 1] = "+V" .. opts.voucher.config.center.key end
  return table.concat(o, ";")
end

-- ---------------------------------------------------------------- snapshot / restore
local function copy(t, depth)
  local c = {}
  for k, v in pairs(t) do
    if type(v) == "table" and getmetatable(v) == nil and (depth or 0) < 8 then c[k] = copy(v, (depth or 0) + 1)
    else c[k] = v end
  end
  return c
end

local function restore(t, s, depth)
  for k in pairs(t) do if s[k] == nil then t[k] = nil end end
  for k, v in pairs(s) do
    local cur = t[k]
    if type(v) == "table" and getmetatable(v) == nil and type(cur) == "table" and getmetatable(cur) == nil
        and (depth or 0) < 8 then
      restore(cur, v, (depth or 0) + 1)
    else
      t[k] = v
    end
  end
end

local CARD_FIELDS = { "debuff", "lucky_trigger", "sell_cost" }
local function snap_card(c)
  local s = { ability = copy(c.ability) }
  for _, f in ipairs(CARD_FIELDS) do s[f] = c[f] end
  return s
end
local function restore_card(c, s)
  restore(c.ability, s.ability)
  for _, f in ipairs(CARD_FIELDS) do c[f] = s[f] end
end

-- ---------------------------------------------------------------- scoring (G.FUNCS.evaluate_play's steps)
local HELD_JOKERS = { ["Baron"] = true, ["Raised Fist"] = true, ["Shoot the Moon"] = true }
-- cards: the played cards; held: the rest of the deal. Returns chips * mult and the hand's name.
local function score(cards, held)
  local text, _, poker_hands, scoring_hand = G.FUNCS.get_poker_hand_info(cards)
  local sh = {}
  for i, c in ipairs(scoring_hand) do sh[i] = c end
  local splash = next(find_joker("Splash"))
  for _, c in ipairs(cards) do
    local inside = false
    for _, s in ipairs(sh) do if s == c then inside = true end end
    if not inside and (splash or c.ability.effect == "Stone Card") then sh[#sh + 1] = c end
  end
  -- the cards' own order in the deal (the game sorts by screen position)
  local pos = {}
  for i, c in ipairs(cards) do pos[c] = i end
  table.sort(sh, function(a, b) return pos[a] < pos[b] end)
  scoring_hand = sh
  local hand = G.GAME.hands[text]
  local mult, hand_chips = mod_mult(hand.mult), mod_chips(hand.chips)
  local function with(extra)
    local t = { full_hand = cards, scoring_hand = scoring_hand, scoring_name = text, poker_hands = poker_hands }
    for k, v in pairs(extra) do t[k] = v end
    return t
  end
  for i = 1, #scoring_hand do
    local card = scoring_hand[i]
    if not card.debuff then
      local reps = { 1 }
      local ev = eval_card(card, with({ repetition_only = true, cardarea = G.play, repetition = true }))
      if next(ev) then for _ = 1, ev.seals.repetitions do reps[#reps + 1] = ev end end
      for j = 1, #G.jokers.cards do
        local e2 = eval_card(G.jokers.cards[j], with({ cardarea = G.play, other_card = card, repetition = true }))
        if next(e2) and e2.jokers then for _ = 1, e2.jokers.repetitions do reps[#reps + 1] = e2 end end
      end
      for _ = 1, #reps do
        local effects = { eval_card(card, { cardarea = G.play, full_hand = cards, scoring_hand = scoring_hand, poker_hand = text }) }
        for k = 1, #G.jokers.cards do
          local e3 = G.jokers.cards[k]:calculate_joker(with({ cardarea = G.play, other_card = card, individual = true }))
          if e3 then effects[#effects + 1] = e3 end
        end
        card.lucky_trigger = nil
        for _, ef in ipairs(effects) do
          if ef.chips then hand_chips = mod_chips(hand_chips + ef.chips) end
          if ef.mult then mult = mod_mult(mult + ef.mult) end
          if ef.extra then
            if ef.extra.mult_mod then mult = mod_mult(mult + ef.extra.mult_mod) end
            if ef.extra.chip_mod then hand_chips = mod_chips(hand_chips + ef.extra.chip_mod) end
            if ef.extra.swap then mult, hand_chips = mod_mult(hand_chips), mod_chips(mult) end
          end
          if ef.x_mult then mult = mod_mult(mult * ef.x_mult) end
          if ef.edition then
            hand_chips = mod_chips(hand_chips + (ef.edition.chip_mod or 0))
            mult = mult + (ef.edition.mult_mod or 0)
            mult = mod_mult(mult * (ef.edition.x_mult_mod or 1))
          end
        end
      end
    end
  end
  -- held in hand (the rest of the deal). Only Baron, Raised Fist and Shoot the Moon (and Reserved
  -- Parking, dollars) act on held cards, and a held card scores only by its own held mult (Steel);
  -- with neither there, the game's loop changes nothing and is skipped.
  local held_acts = false
  for _, j in ipairs(G.jokers.cards) do if HELD_JOKERS[j.ability.name] and not j.debuff then held_acts = true end end
  for _, c in ipairs(held) do
    if (c.ability.h_mult or 0) > 0 or (c.ability.h_x_mult or 0) > 0 then held_acts = true end
  end
  local hand_cards = G.hand.cards
  G.hand.cards = held
  for i = 1, held_acts and #held or 0 do
    local hc = held[i]
    local reps, j = { 1 }, 1
    while j <= #reps do
      local effects = { eval_card(hc, with({ cardarea = G.hand })) }
      for k = 1, #G.jokers.cards do
        local e3 = G.jokers.cards[k]:calculate_joker(with({ cardarea = G.hand, other_card = hc, individual = true }))
        if e3 then effects[#effects + 1] = e3 end
      end
      if reps[j] == 1 then
        local ev = eval_card(hc, with({ repetition_only = true, cardarea = G.hand, repetition = true, card_effects = effects }))
        if next(ev) and (next(effects[1]) or #effects > 1) then for _ = 1, ev.seals.repetitions do reps[#reps + 1] = ev end end
        for jj = 1, #G.jokers.cards do
          local e2 = eval_card(G.jokers.cards[jj], with({ cardarea = G.hand, other_card = hc, repetition = true, card_effects = effects }))
          if next(e2) then for _ = 1, e2.jokers.repetitions do reps[#reps + 1] = e2 end end
        end
      end
      for _, ef in ipairs(effects) do
        if ef.h_mult then mult = mod_mult(mult + ef.h_mult) end
        if ef.x_mult then mult = mod_mult(mult * ef.x_mult) end
      end
      j = j + 1
    end
  end
  G.hand.cards = hand_cards
  -- jokers (and consumables' editions): edition, main effect, joker-on-joker, x-mult edition
  for i = 1, #G.jokers.cards + #G.consumeables.cards do
    local c = G.jokers.cards[i] or G.consumeables.cards[i - #G.jokers.cards]
    local ed = eval_card(c, with({ cardarea = G.jokers, edition = true }))
    if ed.jokers then
      if ed.jokers.chip_mod then hand_chips = mod_chips(hand_chips + ed.jokers.chip_mod) end
      if ed.jokers.mult_mod then mult = mod_mult(mult + ed.jokers.mult_mod) end
    end
    local ef = eval_card(c, with({ cardarea = G.jokers, joker_main = true }))
    if ef.jokers then
      if ef.jokers.mult_mod then mult = mod_mult(mult + ef.jokers.mult_mod) end
      if ef.jokers.chip_mod then hand_chips = mod_chips(hand_chips + ef.jokers.chip_mod) end
      if ef.jokers.Xmult_mod then mult = mod_mult(mult * ef.jokers.Xmult_mod) end
    end
    for _, v in ipairs(G.jokers.cards) do
      local e = v:calculate_joker(with({ other_joker = c }))
      if e then
        if e.mult_mod then mult = mod_mult(mult + e.mult_mod) end
        if e.chip_mod then hand_chips = mod_chips(hand_chips + e.chip_mod) end
        if e.Xmult_mod then mult = mod_mult(mult * e.Xmult_mod) end
      end
    end
    if ed.jokers and ed.jokers.x_mult_mod then mult = mod_mult(mult * ed.jokers.x_mult_mod) end
  end
  local nu_chip, nu_mult = G.GAME.selected_back:trigger_effect({ context = "final_scoring_step", chips = hand_chips, mult = mult })
  mult, hand_chips = mod_mult(nu_mult or mult), mod_chips(nu_chip or hand_chips)
  local s = hand_chips * mult
  if s ~= s or s < 0 then s = 0 end
  return s, text
end

-- ---------------------------------------------------------------- the potential
local function get_path(t, path)
  local a, b = path:match("^([^.]+)%.(.+)$")
  if a then return t[a] and t[a][b], function(v) t[a][b] = v end end
  return t[path], function(v) t[path] = v end
end

local function project(spam)
  for _, c in ipairs(G.jokers.cards) do
    local g = GROWTH[c.ability.name]
    if g and not (c.debuff) then
      for path, rate in pairs(g) do
        if path ~= "floor" then
          local v, set = get_path(c.ability, path)
          if type(v) == "number" then
            v = v + rate * PROJECT_ROUNDS
            if g.floor and v < g.floor then v = g.floor end
            set(v)
          end
        end
      end
    end
    local lv = LEVELS_PER_ROUND[c.ability.name]
    if lv and not c.debuff then level_up_hand(nil, spam, true, lv * PROJECT_ROUNDS) end
  end
  G.GAME.hands[spam].played = G.GAME.hands[spam].played + SPAM_PLAYS_PER_ROUND * PROJECT_ROUNDS
end

-- Values the game keeps in a joker's ability and refreshes every frame (Card:update) from the board: after
-- an offer changes the board, they are refreshed as Card:update would (the offer counted as bought).
local function refresh_board()
  for _, c in ipairs(G.jokers.cards) do
    local a = c.ability
    if a.name == "Joker Stencil" then
      a.x_mult = G.jokers.config.card_limit - #G.jokers.cards
      for _, x in ipairs(G.jokers.cards) do if x.ability.name == "Joker Stencil" then a.x_mult = a.x_mult + 1 end end
    elseif a.name == "Swashbuckler" then
      local sc = 0
      for _, x in ipairs(G.jokers.cards) do if x ~= c then sc = sc + x.sell_cost end end
      a.mult = sc
    end
  end
end

local ORDER = { "Flush Five", "Flush House", "Five of a Kind", "Straight Flush", "Four of a Kind", "Full House",
                "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair", "High Card" }

-- B and the phi value for the current state, or with one offer taken (opts). Changes nothing.
function H.build_power_fresh(opts)
  local deck = {}
  for i, c in ipairs(G.playing_cards) do deck[i] = c end
  if #deck == 0 then return 0, 0 end
  local e = effective(opts)
  -- the game's tables this touches, saved
  local cards = {}
  for _, c in ipairs(deck) do cards[#cards + 1] = c end
  for _, c in ipairs(G.jokers.cards) do cards[#cards + 1] = c end
  for _, c in ipairs(G.consumeables.cards) do cards[#cards + 1] = c end
  if opts and opts.joker then cards[#cards + 1] = opts.joker end
  local snaps = {}
  for _, c in ipairs(cards) do snaps[c] = snap_card(c) end
  local g = G.GAME
  local s_hands, s_round, s_resets = copy(g.hands), copy(g.current_round), copy(g.round_resets)
  local s_prob, s_vouchers = copy(g.probabilities), copy(g.used_vouchers)
  local s_scalars = {}
  for k, v in pairs(g) do if type(v) ~= "table" then s_scalars[k] = v end end
  local jl, cl, hl = G.jokers.cards, G.jokers.config.card_limit, G.hand.config.card_limit
  local held0, deck0 = G.hand.cards, G.deck.cards
  local saved = {}
  for _, k in ipairs(STUBS) do saved[k] = _G[k]; _G[k] = NOOP end
  local pr, ps, pe, mr = pseudorandom, pseudoseed, pseudorandom_element, math.random
  local ctr = 0
  pseudorandom = function(_, lo, hi) if lo and hi then return math.floor((lo + hi) / 2) end return 0.5 end
  pseudoseed = function() return 0.5 end
  pseudorandom_element = function(t) local ks = {}; for k in pairs(t) do ks[#ks + 1] = k end
    table.sort(ks, function(a, b) return tostring(a) < tostring(b) end); return t[ks[1]], ks[1] end
  math.random = function(lo, hi)            -- Stone cards' ids: distinct, as the game's are
    ctr = ctr + 1
    if lo and hi then return lo + (ctr % (hi - lo + 1)) end
    return 0.5
  end
  local add_event = G.E_MANAGER.add_event
  G.E_MANAGER.add_event = NOOP

  local ok, B = pcall(function()
    -- the board being valued
    G.GAME.probabilities.normal = e.prob
    G.GAME.dollars = e.dollars
    G.jokers.config.card_limit = e.slots
    G.GAME.round_resets.hands, G.GAME.round_resets.discards = e.hands, e.discards
    if opts and opts.joker then
      local j = {}
      for i, c in ipairs(G.jokers.cards) do j[i] = c end
      j[#j + 1] = opts.joker
      G.jokers.cards = j
    end
    if opts and opts.voucher then G.GAME.used_vouchers[opts.voucher.config.center.key] = true end
    if opts and opts.level then
      level_up_hand(nil, opts.level, true, 1)
      -- using a planet also grows Constellation (its using_consumeable effect)
      for _, c in ipairs(G.jokers.cards) do
        if c.ability.name == "Constellation" and not c.debuff then c.ability.x_mult = c.ability.x_mult + c.ability.extra end
      end
    end
    if opts then refresh_board() end
    local cr = G.GAME.current_round
    cr.hands_left, cr.discards_left = e.hands, e.discards
    for _, h in pairs(G.GAME.hands) do h.played_this_round = 0 end
    -- the blind's debuffs are not the build's; an expired perishable joker stays debuffed
    for _, c in ipairs(deck) do c.debuff = false end
    for _, c in ipairs(G.jokers.cards) do
      c.debuff = (c.ability.perishable and (c.ability.perish_tally or 1) <= 0) and true or false
    end
    project(spam_hand())
    -- the state every play starts from (scoring changes some jokers' counters; each play starts fresh)
    local base = {}
    for _, c in ipairs(G.jokers.cards) do base[c] = snap_card(c) end
    for _, c in ipairs(deck) do base[c] = snap_card(c) end
    local b_hands = copy(G.GAME.hands)
    -- the deals: the deck in a fixed order (by card), a fixed generator
    local sorted = {}
    for i, c in ipairs(deck) do sorted[i] = { sig = card_sig(c), c = c } end
    table.sort(sorted, function(a, b) return a.sig < b.sig end)
    local hs = math.min(e.hs, #sorted)
    local state, total = 12345, 0
    local function rnd(n) state = (state * 1103515245 + 12345) % 2147483648; return state % n + 1 end
    for _ = 1, K_DEALS do
      local idx, deal = {}, {}
      for i = 1, #sorted do idx[i] = i end
      for i = 1, hs do local jx = i - 1 + rnd(#sorted - i + 1); idx[i], idx[jx] = idx[jx], idx[i] end
      local chosen = {}
      for i = 1, hs do chosen[i] = idx[i] end
      table.sort(chosen)
      for i = 1, hs do deal[i] = sorted[chosen[i]].c end
      local rest = {}
      local in_deal = {}
      for _, c in ipairs(deal) do in_deal[c] = true end
      for _, x in ipairs(sorted) do if not in_deal[x.c] then rest[#rest + 1] = x.c end end
      G.deck.cards = rest
      local ph = evaluate_poker_hand(deal)
      local best = 0
      for _, t in ipairs(ORDER) do
        local inst = ph[t] and ph[t][1]
        if inst and #inst > 0 then
          local played, mark = {}, {}
          for _, c in ipairs(deal) do for _, x in ipairs(inst) do if x == c then mark[c] = true end end end
          for _, c in ipairs(deal) do if mark[c] then played[#played + 1] = c end end
          local held = {}
          for _, c in ipairs(deal) do if not mark[c] then held[#held + 1] = c end end
          local s = score(played, held)
          if s > best then best = s end
          -- a play touches only the jokers and the cards dealt (the whole deck is restored at the end)
          for _, c in ipairs(G.jokers.cards) do restore_card(c, base[c]) end
          for _, c in ipairs(deal) do restore_card(c, base[c]) end
          restore(G.GAME.hands, b_hands)
        end
      end
      total = total + best
    end
    return total / K_DEALS
  end)

  -- put everything back
  G.E_MANAGER.add_event = add_event
  pseudorandom, pseudoseed, pseudorandom_element, math.random = pr, ps, pe, mr
  for _, k in ipairs(STUBS) do _G[k] = saved[k] end
  G.jokers.cards, G.jokers.config.card_limit, G.hand.config.card_limit = jl, cl, hl
  G.hand.cards, G.deck.cards = held0, deck0
  for c, sn in pairs(snaps) do restore_card(c, sn) end
  restore(g.hands, s_hands); restore(g.current_round, s_round); restore(g.round_resets, s_resets)
  restore(g.probabilities, s_prob); restore(g.used_vouchers, s_vouchers)
  for k, v in pairs(s_scalars) do g[k] = v end
  for k, v in pairs(g) do if type(v) ~= "table" and s_scalars[k] == nil then g[k] = nil end end
  if not ok then error("build_power: " .. tostring(B)) end
  local t8 = get_blind_amount(8) * 2 * G.GAME.starting_params.ante_scaling
  local x = math.log(math.max(e.hands, 1) * B / t8) / math.log(2)
  if x ~= x then x = -10 end
  if x < -10 then x = -10 elseif x > 2 then x = 2 end
  return (x + 10) / 4, B
end

-- cached by everything it reads (most recent first; joker drags come back to earlier orders)
function H.build_power(opts, base)
  local key = H.build_key(opts, base)
  local C = H._phi_cache or {}
  H._phi_cache = C
  for i, en in ipairs(C) do
    if en.key == key then
      if i > 1 then table.remove(C, i); table.insert(C, 1, en) end
      return en.phi, en.B
    end
  end
  local phi, B = H.build_power_fresh(opts)
  table.insert(C, 1, { key = key, phi = phi, B = B })
  C[33] = nil
  return phi, B
end

-- The value of each encoded action (H._encoded_actions, as last encoded): phi_build after taking the
-- offer minus phi_build now, for buying or picking a joker or a planet and redeeming a voucher (a planet
-- valued as used: its hand leveled, Constellation grown, even when bought into a slot to use later); 0 for
-- every other action, the offer paid for at its price. As integers x 10000 (the action row's number columns), space-separated.
function H.offer_values()
  local acts = H._encoded_actions or {}
  local out = {}
  local now, base = nil, nil
  for k, a in ipairs(acts) do
    local opts = nil
    local c = nil
    if a.type == "buy" or a.type == "buy_and_use" then c = G.shop_jokers and G.shop_jokers.cards[a.i]
    elseif a.type == "pick" then c = G.pack_cards and G.pack_cards.cards[a.i]
    elseif a.type == "redeem" then c = G.shop_vouchers and G.shop_vouchers.cards[a.i] end
    if c then
      local set = c.ability.set
      if set == "Joker" then opts = { joker = c }
      elseif set == "Planet" and c.ability.consumeable and c.ability.consumeable.hand_type then
        opts = { level = c.ability.consumeable.hand_type }
      elseif set == "Voucher" then opts = { voucher = c } end
      if opts then opts.cost = a.type == "pick" and 0 or (c.cost or 0) end    -- a pack's cards are free
    end
    local v = 0
    if opts then
      base = base or H.build_key_base()
      now = now or H.build_power(nil, base)
      v = H.build_power(opts, base) - now
    end
    out[k] = math.floor(v * 10000 + 0.5)        -- to the nearest 1e-4
  end
  return table.concat(out, " ")
end
