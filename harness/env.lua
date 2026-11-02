-- Legal actions and observations over the headless game. Loaded after boot.lua; extends H.
--
-- Legality comes from the game's own checks (G.FUNCS.can_play, can_discard, can_buy,
-- can_redeem, can_open, can_reroll, can_skip_booster, can_select_card, Card:can_use_consumeable,
-- Card:can_sell_card, check_for_buy_space), evaluated the same way the buttons evaluate them.
-- Actions go through the same G.FUNCS calls replay.py uses, which the replays validate.
--
-- Two action modes (H.mode):
--   "controls" (default)  the game's own controls. Select or deselect one hand card at a time
--                         (toggle), drag hand cards and jokers (move_hand, move_joker), switch the
--                         hand's sort (sort_rank, sort_suit), then play / discard / use / pick with
--                         whatever is selected. Played cards score left to right in hand order,
--                         exactly as in the game (play_cards_from_highlighted sorts by position).
--   "subsets"             one action per selection: play/discard/use/pick carry the hand
--                         positions to select. Same game, fewer steps, ~2000 actions per hand.
-- Both modes offer move_joker, the shop, packs, blinds and selling.
--
-- An action is a table { type, i, j, cards }: i/j are 1-based positions in the relevant area;
-- cards are hand positions, ascending.
--
-- Not offered (yet): boss reroll (Director's Cut / Retcon), endless mode after a win.

local H = ...
H.mode = H.mode or "controls"

-- ---------- helpers ----------

-- Every subset of {1..n} with size in [lo, hi], ascending, containing every index in `must`.
local function subsets(n, lo, hi, must)
  local out, cur, need = {}, {}, {}
  for _, i in ipairs(must or {}) do need[i] = true end
  local function rec(start, depth)
    if depth >= lo and depth <= hi then
      local got = 0
      for _, j in ipairs(cur) do if need[j] then got = got + 1 end end
      if got == #(must or {}) then local c = {}; for k, v in ipairs(cur) do c[k] = v end; out[#out + 1] = c end
    end
    if depth == hi then return end
    for i = start, n do cur[depth + 1] = i; rec(i + 1, depth + 1); cur[depth + 1] = nil end
  end
  rec(1, 0)
  return out
end

local function forced_indices()
  local out = {}
  for i, c in ipairs(G.hand and G.hand.cards or {}) do
    if c.ability.forced_selection then out[#out + 1] = i end
  end
  return out
end

-- Runs a game check against a fake button element and reports whether it armed the button.
local function button_check(fn, card)
  local e = { config = { ref_table = card, button = 'x' }, UIBox = { states = {}, alignment = { offset = {} } } }
  local ok = pcall(fn, e)
  return ok and e.config.button ~= nil
end

local function with_highlight(idxs, f)
  local saved = G.hand.highlighted
  local hl = {}
  for _, i in ipairs(idxs) do hl[#hl + 1] = G.hand.cards[i] end
  G.hand.highlighted = hl
  local ok, r = pcall(f)
  G.hand.highlighted = saved
  return ok and r
end

local function card_needs_targets(card)
  local c = card.ability.consumeable
  if not c then return nil end
  if c.max_highlighted then return (c.min_highlighted or 1), c.mod_num or c.max_highlighted end
  if card.ability.name == 'Aura' then return 1, 1 end
  return nil
end

-- Subsets mode: target sets with which `card` can be used right now ({} = no targets).
local function use_targets(card)
  local out = {}
  local lo, hi = card_needs_targets(card)
  if not lo then
    if with_highlight(forced_indices(), function() return card:can_use_consumeable() end) then out[1] = {} end
    return out
  end
  local n = G.hand and #G.hand.cards or 0
  for _, s in ipairs(subsets(n, lo, math.min(hi, n), forced_indices())) do
    if with_highlight(s, function() return card:can_use_consumeable() end) then out[#out + 1] = s end
  end
  return out
end

-- G.FUNCS.check_for_buy_space is the game's own room check, but when there's no room it also
-- calls alert_no_space: the "No space!" popup, which locks the controller and queues events,
-- as if the player had clicked Buy. Listing legal actions must not change the game, so the
-- check runs with only that alert switched off.
local function has_buy_space(card)
  local alert = alert_no_space
  alert_no_space = function() end
  local ok, r = pcall(G.FUNCS.check_for_buy_space, card)
  alert_no_space = alert
  return ok and r
end

local function usable_now(card)
  local ok, r = pcall(function() return card:can_use_consumeable() end)
  return ok and r
end

-- ---------- legal actions ----------

local S = G.STATES
local PACK = {}
for _, k in ipairs({ "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK" }) do
  PACK[S[k]] = true
end
local HAND_SHOWN = { [S.SELECTING_HAND] = true, [S.TAROT_PACK] = true, [S.SPECTRAL_PACK] = true }

function H.is_terminal()
  return G.STATE == S.GAME_OVER or (G.GAME.won and true or false)
end

local function add(acts, a) acts[#acts + 1] = a end

local function add_jokers_and_consumables(acts, controls)
  local nj = #G.jokers.cards
  for i = 1, nj do
    for j = 1, nj do if i ~= j then add(acts, { type = "move_joker", i = i, j = j }) end end
  end
  for i, c in ipairs(G.jokers.cards) do
    if c:can_sell_card() then add(acts, { type = "sell_joker", i = i }) end
  end
  for i, c in ipairs(G.consumeables.cards) do
    if c:can_sell_card() then add(acts, { type = "sell_consumable", i = i }) end
    if controls then
      if usable_now(c) then add(acts, { type = "use_consumable", i = i }) end
    else
      for _, s in ipairs(use_targets(c)) do add(acts, { type = "use_consumable", i = i, cards = s }) end
    end
  end
end

local function add_hand_controls(acts)
  local n, lim = #G.hand.cards, G.hand.config.highlighted_limit
  local nh = #G.hand.highlighted
  for i, c in ipairs(G.hand.cards) do
    if c.highlighted then
      if not c.ability.forced_selection then add(acts, { type = "toggle", i = i }) end
    elseif nh < lim then
      add(acts, { type = "toggle", i = i })
    end
  end
  for i = 1, n do
    for j = 1, n do if i ~= j then add(acts, { type = "move_hand", i = i, j = j }) end end
  end
  -- A sort button counts only if pressing it changes the order or the hand's sort setting
  -- (the setting decides how newly drawn cards are placed). Checked with the game's own sort.
  for _, s in ipairs({ { "sort_rank", "desc" }, { "sort_suit", "suit desc" } }) do
    local copy = { cards = {}, config = { sort = G.hand.config.sort } }
    for k, c in ipairs(G.hand.cards) do copy.cards[k] = c end
    CardArea.sort(copy, s[2])
    local changed = G.hand.config.sort ~= s[2]
    for k, c in ipairs(G.hand.cards) do if copy.cards[k] ~= c then changed = true; break end end
    if changed then add(acts, { type = s[1] }) end
  end
end

-- Identity of what's on screen, for no-op checks: hand and joker order, selection, sort setting.
function H.fingerprint()
  local p = { H.state_name(), G.hand.config.sort or "" }
  for _, c in ipairs(G.hand.cards) do p[#p + 1] = c.sort_id .. (c.highlighted and "*" or "") end
  p[#p + 1] = "|"
  for _, c in ipairs(G.jokers.cards) do p[#p + 1] = c.sort_id end
  return table.concat(p, ",")
end

function H.legal_actions()
  local acts = {}
  if H.is_terminal() then return acts end
  local st, controls = G.STATE, H.mode == "controls"

  if st == S.SELECTING_HAND then
    local can_play = not G.GAME.blind.block_play
    local can_discard = G.GAME.current_round.discards_left > 0
    if controls then
      add_hand_controls(acts)
      local nh = #G.hand.highlighted
      if can_play and nh >= 1 and nh <= 5 then add(acts, { type = "play" }) end
      if can_discard and nh >= 1 then add(acts, { type = "discard" }) end
    else
      local n = #G.hand.cards
      for _, s in ipairs(subsets(n, 1, math.min(G.hand.config.highlighted_limit, n), forced_indices())) do
        if can_play and #s <= 5 then add(acts, { type = "play", cards = s }) end
        if can_discard then add(acts, { type = "discard", cards = s }) end
      end
    end
    add_jokers_and_consumables(acts, controls)

  elseif st == S.BLIND_SELECT then
    add(acts, { type = "select_blind" })
    local deck = G.GAME.blind_on_deck
    if (deck == "Small" or deck == "Big") and G.blind_select_opts and G.blind_select_opts[deck:lower()] then
      add(acts, { type = "skip_blind" })
    end
    add_jokers_and_consumables(acts, controls)

  elseif st == S.SHOP then
    add(acts, { type = "end_shop" })
    if button_check(G.FUNCS.can_reroll) then add(acts, { type = "reroll" }) end
    for i, c in ipairs(G.shop_jokers and G.shop_jokers.cards or {}) do
      if button_check(G.FUNCS.can_buy, c) and has_buy_space(c) then add(acts, { type = "buy", i = i }) end
      if c.ability.consumeable and button_check(G.FUNCS.can_buy_and_use, c) then add(acts, { type = "buy_and_use", i = i }) end
    end
    for i, c in ipairs(G.shop_vouchers and G.shop_vouchers.cards or {}) do
      if button_check(G.FUNCS.can_redeem, c) then add(acts, { type = "redeem", i = i }) end
    end
    for i, c in ipairs(G.shop_booster and G.shop_booster.cards or {}) do
      if button_check(G.FUNCS.can_open, c) then add(acts, { type = "open_pack", i = i }) end
    end
    add_jokers_and_consumables(acts, controls)

  elseif PACK[st] or (S.SMODS_BOOSTER_OPENED and st == S.SMODS_BOOSTER_OPENED) then
    if controls and HAND_SHOWN[st] and G.hand and G.hand.cards[1] then add_hand_controls(acts) end
    for i, c in ipairs(G.pack_cards and G.pack_cards.cards or {}) do
      if c.ability.consumeable then
        if controls then
          if usable_now(c) then add(acts, { type = "pick", i = i }) end
        else
          for _, s in ipairs(use_targets(c)) do add(acts, { type = "pick", i = i, cards = s }) end
        end
      elseif button_check(G.FUNCS.can_select_card, c) then
        add(acts, { type = "pick", i = i })
      end
    end
    if button_check(G.FUNCS.can_skip_booster) then add(acts, { type = "skip_pack" }) end
    add_jokers_and_consumables(acts, controls)

  elseif st == S.ROUND_EVAL then
    add(acts, { type = "cash_out" })
    add_jokers_and_consumables(acts, controls)   -- jokers can be dragged, sold and used here too
  end
  return acts
end

-- ---------- applying ----------

local function e_for(card, id) return { config = { ref_table = card, id = id } } end

local function set_highlight(idxs)
  G.hand:unhighlight_all()            -- keeps forced selections (Cerulean Bell)
  for _, i in ipairs(idxs or {}) do
    local c = G.hand.cards[i]
    if c and not c.highlighted then G.hand:add_to_highlighted(c, true) end
  end
end

-- A drag: take the card at i out, put it back at j. align_cards lays cards out by index and
-- keeps that order, the same end state as dropping a dragged card into that slot.
local function move(area, i, j)
  local c = table.remove(area.cards, i)
  table.insert(area.cards, j, c)
  area:set_ranks()
  area:align_cards()
end

function H.apply(a)
  local F, t = G.FUNCS, a.type
  local subs = H.mode == "subsets"
  if t == "toggle" then
    local c = G.hand.cards[a.i]
    if c.highlighted then G.hand:remove_from_highlighted(c) else G.hand:add_to_highlighted(c, true) end
  elseif t == "move_hand" then move(G.hand, a.i, a.j)
  elseif t == "move_joker" then move(G.jokers, a.i, a.j)
  elseif t == "sort_rank" then G.FUNCS.sort_hand_value(nil)
  elseif t == "sort_suit" then G.FUNCS.sort_hand_suit(nil)
  elseif t == "play" then if subs then set_highlight(a.cards) end; F.play_cards_from_highlighted(nil)
  elseif t == "discard" then if subs then set_highlight(a.cards) end; F.discard_cards_from_highlighted(nil)
  elseif t == "select_blind" then
    local box = G.blind_select_opts[G.GAME.blind_on_deck:lower()]
    F.select_blind(box:get_UIE_by_ID("select_blind_button"))
  elseif t == "skip_blind" then F.skip_blind({ UIBox = G.blind_select_opts[G.GAME.blind_on_deck:lower()] })
  elseif t == "end_shop" then F.toggle_shop(nil)
  elseif t == "reroll" then F.reroll_shop(nil)
  elseif t == "buy" then F.buy_from_shop(e_for(G.shop_jokers.cards[a.i]))
  elseif t == "buy_and_use" then F.buy_from_shop(e_for(G.shop_jokers.cards[a.i], "buy_and_use"))
  elseif t == "redeem" then F.use_card(e_for(G.shop_vouchers.cards[a.i]))
  elseif t == "open_pack" then F.use_card(e_for(G.shop_booster.cards[a.i]))
  elseif t == "pick" then if subs then set_highlight(a.cards) end; F.use_card(e_for(G.pack_cards.cards[a.i]))
  elseif t == "skip_pack" then F.skip_booster(nil)
  elseif t == "use_consumable" then if subs then set_highlight(a.cards) end; F.use_card(e_for(G.consumeables.cards[a.i]))
  elseif t == "sell_joker" then F.sell_card(e_for(G.jokers.cards[a.i]))
  elseif t == "sell_consumable" then F.sell_card(e_for(G.consumeables.cards[a.i]))
  elseif t == "cash_out" then F.cash_out({ config = {} })
  else error("unknown action " .. tostring(t)) end
end

-- settle(), then also wait out the boss-debuff alert: it blocks Play through a non-blocking
-- event, the same pause a player sits through before the Play button lights up.
function H.step(a)
  H.apply(a)
  local n = H.settle()
  if n < 0 then return n end
  local extra = 0
  while G.STATE == S.SELECTING_HAND and G.GAME.blind and G.GAME.blind.block_play and extra < 600 do
    HEADLESS_ADVANCE_CLOCK(1 / 30); G:update(1 / 30); extra = extra + 1
  end
  return n + extra
end

-- ---------- observation ----------
-- Only what the player can see: card faces that are face up, card and joker values shown on
-- hover, the deck view (remaining and full deck, as sets, not in draw order), the HUD, the
-- blind and its effect, shop and pack contents, tags, vouchers and hand levels. Never RNG state.

local function ed(c) local e = c.edition; if not e then return nil end
  for _, k in ipairs({ "foil", "holo", "polychrome", "negative" }) do if e[k] then return k end end end

local function face_down(c) return c.facing == "back" end

local function playing_card(c, area_hides)
  if area_hides and face_down(c) then return { face_down = true, highlighted = c.highlighted or nil } end
  local enh = c.config.center.key
  return { rank = c.base.value, suit = c.base.suit, id = c.base.id, chips = c.base.nominal,
           enhancement = enh ~= "c_base" and enh or nil, edition = ed(c), seal = c.seal,
           bonus_chips = (c.ability.perma_bonus or 0) ~= 0 and c.ability.perma_bonus or nil,
           debuff = c.debuff or nil, highlighted = c.highlighted or nil }
end

local function primitives(t, depth)
  local out = {}
  for k, v in pairs(t or {}) do
    local tv = type(v)
    if tv == "number" or tv == "string" or tv == "boolean" then out[k] = v
    elseif tv == "table" and depth > 0 and k ~= "consumeable" then out[k] = primitives(v, depth - 1) end
  end
  return out
end

local function item(c, area_hides)
  if area_hides and face_down(c) then return { face_down = true } end
  local a = c.ability
  return { key = c.config.center.key, set = a.set, edition = ed(c), cost = c.cost, sell = c.sell_cost,
           eternal = a.eternal or nil, perishable = a.perishable or nil, perish_tally = a.perish_tally,
           rental = a.rental or nil, debuff = c.debuff or nil, ability = primitives(a, 1) }
end

local function map(list, f, hides) local out = {}; for i, v in ipairs(list or {}) do out[i] = f(v, hides) end; return out end

local function card_sort_key(c)
  return string.format("%s|%02d|%s|%s|%s", c.suit or "", c.id or 0, c.enhancement or "", c.edition or "", c.seal or "")
end
local function as_set(cards)
  local out = map(cards, playing_card, false)
  for _, c in ipairs(out) do c.highlighted = nil end
  table.sort(out, function(a, b) return card_sort_key(a) < card_sort_key(b) end)
  return out
end

function H.observe()
  local GG, cr = G.GAME, G.GAME.current_round
  local hands = {}
  for k, v in pairs(GG.hands) do
    hands[k] = { level = v.level, chips = v.chips, mult = v.mult, played = v.played,
                 played_this_round = v.played_this_round, visible = v.visible }
  end
  local vouchers = {}
  for k, v in pairs(GG.used_vouchers or {}) do if v then vouchers[#vouchers + 1] = k end end
  table.sort(vouchers)
  local o = {
    state = H.state_name(), ante = GG.round_resets.ante, round = GG.round, stake = GG.stake,
    money = GG.dollars, chips = GG.chips, won = GG.won or false, bankrupt_at = GG.bankrupt_at,
    interest_cap = GG.interest_cap, hands_left = cr.hands_left, discards_left = cr.discards_left,
    hand_size = G.hand.config.card_limit, joker_slots = G.jokers.config.card_limit,
    consumable_slots = G.consumeables.config.card_limit, reroll_cost = cr.reroll_cost,
    free_rerolls = cr.free_rerolls, blind_on_deck = GG.blind_on_deck, hand_sort = G.hand.config.sort,
    mode = H.mode,
    hand = map(G.hand.cards, playing_card, true),
    deck = as_set(G.deck.cards),              -- the deck view's "remaining", in no particular order
    full_deck = as_set(G.playing_cards),
    jokers = map(G.jokers.cards, item, true), -- Amber Acorn turns jokers face down
    consumables = map(G.consumeables.cards, item, false),
    tags = map(GG.tags, function(t) return t.key end),
    vouchers = vouchers,
    hand_levels = hands,
    round_targets = {                          -- shown on The Idol, Mail-In Rebate, Ancient Joker, Castle
      idol = cr.idol_card and { rank = cr.idol_card.rank, suit = cr.idol_card.suit },
      mail = cr.mail_card and cr.mail_card.rank,
      ancient = cr.ancient_card and cr.ancient_card.suit,
      castle = cr.castle_card and cr.castle_card.suit,
    },
  }
  if GG.blind and GG.blind.name and GG.blind.name ~= "" then
    local b = GG.blind
    o.blind = { name = b.name, key = b.config.blind.key, target = b.chips, boss = b.boss or false,
                disabled = b.disabled or false, hands_sub = b.hands_sub, only_hand = b.only_hand }
  end
  o.blind_choices = { Small = GG.round_resets.blind_choices.Small, Big = GG.round_resets.blind_choices.Big,
                      Boss = GG.round_resets.blind_choices.Boss }
  o.blind_states = primitives(GG.round_resets.blind_states, 0)
  o.skip_tags = { Small = GG.round_resets.blind_tags and GG.round_resets.blind_tags.Small,
                  Big = GG.round_resets.blind_tags and GG.round_resets.blind_tags.Big }
  local function shop_item(c)
    local x = item(c, false)
    if c.ability.set == "Default" or c.ability.set == "Enhanced" then x.card = playing_card(c, false) end
    return x
  end
  if G.STATE == S.SHOP then
    o.shop = map(G.shop_jokers and G.shop_jokers.cards, shop_item)
    o.vouchers_for_sale = map(G.shop_vouchers and G.shop_vouchers.cards, item, false)
    o.boosters = map(G.shop_booster and G.shop_booster.cards, item, false)
  end
  if G.pack_cards and G.pack_cards.cards then
    o.pack = map(G.pack_cards.cards, shop_item)
    o.pack_choices = GG.pack_choices
  end
  return o
end

return H
