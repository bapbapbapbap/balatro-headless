-- Our view of the game's state, read straight from G, for comparison with AnteZero's simulator (lockstep.py).
-- Ours (Claude's, reviewed by chud). No AnteZero code: the field list is the registration's (docs/hypotheses-reproduce.md,
-- 2026-11-27 section); lockstep.py builds the same view from the simulator's state_json.

local function enc(v, out)
  local t = type(v)
  if t == 'nil' then out[#out + 1] = 'null'
  elseif t == 'boolean' then out[#out + 1] = v and 'true' or 'false'
  elseif t == 'number' then
    if v ~= v then out[#out + 1] = '"nan"'
    elseif v == math.huge then out[#out + 1] = '"inf"'
    elseif v == -math.huge then out[#out + 1] = '"-inf"'
    else out[#out + 1] = string.format('%.17g', v) end
  elseif t == 'string' then out[#out + 1] = string.format('%q', v):gsub('\\\n', '\\n')
  elseif t == 'table' then
    if v.__list then
      out[#out + 1] = '['
      for i = 1, #v do if i > 1 then out[#out + 1] = ',' end enc(v[i], out) end
      out[#out + 1] = ']'
    else
      local keys = {}
      for k in pairs(v) do if k ~= '__list' then keys[#keys + 1] = tostring(k) end end
      table.sort(keys)
      out[#out + 1] = '{'
      for i, k in ipairs(keys) do
        if i > 1 then out[#out + 1] = ',' end
        out[#out + 1] = string.format('%q', k) .. ':'
        local val = v[k]; if val == nil then val = v[tonumber(k)] end
        enc(val, out)
      end
      out[#out + 1] = '}'
    end
  else out[#out + 1] = '"<' .. t .. '>"' end
end
local function list(t) t = t or {}; t.__list = true; return t end

-- Plain data only (numbers, strings, booleans, nested plain tables); objects (cards, UI) are skipped.
local function plain(t, depth)
  if type(t) ~= 'table' then return t end
  if depth > 4 or getmetatable(t) then return nil end
  local out = {}
  local n = #t
  for k, v in pairs(t) do
    local tv = type(v)
    if tv == 'table' then v = plain(v, depth + 1) end
    if tv ~= 'function' and tv ~= 'userdata' and v ~= nil then out[k] = v end
  end
  if n > 0 then out.__list = true end
  return out
end

local function card(c)
  return {
    card = c.config and c.config.card_key or nil,
    key = c.config and c.config.center and c.config.center.key or nil,
    edition = c.edition and c.edition.type or nil,
    seal = c.seal,
    debuff = c.debuff and true or false,
    facing = c.facing,
    cost = c.cost,
    sell_cost = c.sell_cost,
    sort_id = c.sort_id,
    ability = plain(c.ability, 0),
  }
end
local function area(a)
  local out = list({})
  if a and a.cards then for i, c in ipairs(a.cards) do out[i] = card(c) end end
  return out
end

local state_names = {}
for k, v in pairs(G.STATES) do state_names[v] = k end

function AZ_VIEW()
  local g = G.GAME
  local cr = g.current_round or {}
  local rr = g.round_resets or {}
  local hands = {}
  for name, h in pairs(g.hands or {}) do
    hands[name] = { level = h.level, chips = h.chips, mult = h.mult, played = h.played,
                    played_this_round = h.played_this_round, visible = h.visible }
  end
  local tags = list({})
  for i, t in ipairs(g.tags or {}) do tags[i] = t.key end
  local vouchers = list({})
  for k, v in pairs(g.used_vouchers or {}) do if v then vouchers[#vouchers + 1] = k end end
  table.sort(vouchers)
  local rng = {}
  for k, v in pairs(g.pseudorandom or {}) do if type(v) == 'number' then rng[k] = v end end
  local b = g.blind
  local v = {
    state = state_names[G.STATE],
    ante = rr.ante, round = g.round, dollars = g.dollars, chips = g.chips,
    hands_left = cr.hands_left, discards_left = cr.discards_left, reroll_cost = cr.reroll_cost,
    interest_cap = g.interest_cap, skips = g.skips, probabilities = g.probabilities and g.probabilities.normal,
    won = g.won and true or false,
    joker_limit = G.jokers and G.jokers.config.card_limit,
    hand_limit = G.hand and G.hand.config.card_limit,
    consumable_limit = G.consumeables and G.consumeables.config.card_limit,
    blind = b and { name = b.name, chips = b.chips, mult = b.mult, disabled = b.disabled and true or false } or nil,
    blind_on_deck = g.blind_on_deck,
    blind_choices = plain(rr.blind_choices, 0), blind_tags = plain(rr.blind_tags, 0),
    blind_states = plain(rr.blind_states, 0),
    hand = area(G.hand), deck = area(G.deck), discard = area(G.discard),
    jokers = area(G.jokers), consumables = area(G.consumeables),
    shop_jokers = area(G.shop_jokers), shop_vouchers = area(G.shop_vouchers), shop_booster = area(G.shop_booster),
    pack_cards = area(G.pack_cards),
    hands = hands, tags = tags, used_vouchers = vouchers, rng = rng,
  }
  local out = {}
  enc(v, out)
  return table.concat(out)
end
