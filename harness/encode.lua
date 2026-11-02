-- Numeric encoding of what the player sees plus the legal actions, for learning. Loaded after
-- env.lua (subsets mode). One source of truth for visibility: everything is read from
-- H.observe(), which masks face-down cards and jokers, gives the deck as sets, and holds no RNG
-- state. This file only flattens it.
--
-- H.encode() returns one string of numbers:
--   n_tok n_act  then n_tok rows of (kind, vocab, F features)  then n_act rows of ACT ints
-- and keeps the matching action list, so H.step_index(i) plays exactly the i-th encoded action
-- (1-based). Python never rebuilds or reorders actions.

local H = ...
local F = 32          -- features per token
local ACT = 9         -- type, a, b, c1..c5, hand_type

-- ---------- vocabulary, built from the game's own tables ----------
local vocab, vocab_list = {}, { "<none>" }
local function add(name) if not vocab[name] then vocab_list[#vocab_list + 1] = name; vocab[name] = #vocab_list - 1 end end
local function sorted_keys(t) local k = {}; for name in pairs(t or {}) do k[#k + 1] = name end; table.sort(k); return k end
for _, k in ipairs(sorted_keys(G.P_CENTERS)) do add(k) end
for _, k in ipairs(sorted_keys(G.P_TAGS)) do add(k) end
for _, k in ipairs(sorted_keys(G.P_BLINDS)) do add(k) end
-- The 12 poker hands (G.GAME only exists once a run starts; checked against it on first encode).
local HANDS = { "Flush Five", "Flush House", "Five of a Kind", "Straight Flush", "Four of a Kind",
                "Full House", "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair", "High Card" }
table.sort(HANDS)
for _, k in ipairs(HANDS) do add("hand:" .. k) end
local hands_checked = false
for _, k in ipairs(sorted_keys(G.STATES)) do add("state:" .. k) end
function H.vocab() return table.concat(vocab_list, "\n") end
local function vid(name) return name and vocab[name] or 0 end

local KIND = { GLOBAL = 1, DECK = 2, HANDTYPE = 3, BLIND = 4, SKIPTAG = 5, TARGETS = 6, CARD = 7,
               JOKER = 8, CONSUMABLE = 9, SHOP = 10, VOUCHER_SALE = 11, BOOSTER = 12, PACK = 13,
               TAG = 14, VOUCHER = 15 }
local ACTION = { play = 1, discard = 2, select_blind = 3, skip_blind = 4, end_shop = 5, reroll = 6,
                 buy = 7, buy_and_use = 8, redeem = 9, open_pack = 10, pick = 11, skip_pack = 12,
                 use_consumable = 13, sell_joker = 14, sell_consumable = 15, cash_out = 16,
                 move_joker = 17 }
H.KIND, H.ACTION, H.FEATURES, H.ACT_WIDTH = KIND, ACTION, F, ACT

local SUITS = { Spades = 1, Hearts = 2, Clubs = 3, Diamonds = 4 }
local EDITIONS = { foil = 1, holo = 2, polychrome = 3, negative = 4 }
local SEALS = { Red = 1, Blue = 2, Gold = 3, Purple = 4 }
local ENH = { m_bonus = 1, m_mult = 2, m_wild = 3, m_glass = 4, m_steel = 5, m_stone = 6, m_gold = 7, m_lucky = 8 }
local SETS = { Joker = 1, Tarot = 2, Planet = 3, Spectral = 4, Default = 5, Enhanced = 5, Voucher = 6, Booster = 7 }
local RANKS = { ["2"] = 2, ["3"] = 3, ["4"] = 4, ["5"] = 5, ["6"] = 6, ["7"] = 7, ["8"] = 8, ["9"] = 9,
                ["10"] = 10, Jack = 11, Queen = 12, King = 13, Ace = 14 }
local BSTATE = { Select = 1, Upcoming = 2, Current = 3, Defeated = 4, Skipped = 5, Hide = 6 }

local function slog(x) x = tonumber(x) or 0; if x >= 0 then return math.log(1 + x) / 10 else return -math.log(1 - x) / 10 end end

-- ---------- token builder ----------
local rows, ntok
local function tok(kind, v)
  local r = { kind, v }
  for i = 1, F do r[i + 2] = 0 end
  ntok = ntok + 1; rows[ntok] = r
  return r, ntok - 1      -- row, 0-based token index
end
local function set(r, i, x) r[i + 2] = x end      -- feature i is 1-based
local function onehot(r, start, n, k) if k and k >= 1 and k <= n then r[start + k - 1 + 2] = 1 end end

local function card_features(r, c, start)       -- 29 slots from `start`
  if c.face_down then set(r, start + 27, 1); return end
  onehot(r, start, 13, RANKS[c.rank] and RANKS[c.rank] - 1)
  onehot(r, start + 13, 4, SUITS[c.suit])
  onehot(r, start + 17, 4, EDITIONS[c.edition])
  onehot(r, start + 21, 4, SEALS[c.seal])
  set(r, start + 25, c.debuff and 1 or 0)
  set(r, start + 26, slog(c.bonus_chips))
  set(r, start + 28, (c.chips or 0) / 11)
end

local function ability_numbers(ab)
  local nums = {}
  local keys = {}
  for k, v in pairs(ab or {}) do
    if type(v) == "number" and k ~= "order" and k ~= "perma_bonus" then keys[#keys + 1] = k
    elseif type(v) == "table" then
      for k2, v2 in pairs(v) do if type(v2) == "number" then keys[#keys + 1] = k .. "." .. k2 end end
    end
  end
  table.sort(keys)
  for _, k in ipairs(keys) do
    local a, b = k:match("^(.-)%.(.+)$")
    nums[#nums + 1] = a and ab[a][b] or ab[k]
    if #nums == 8 then break end
  end
  return nums
end

function H.encode()
  if not hands_checked then
    local got = sorted_keys(G.GAME.hands)
    assert(table.concat(got, "|") == table.concat(HANDS, "|"), "poker hand list differs from the game's")
    hands_checked = true
  end
  local o = H.observe()
  rows, ntok = {}, 0
  local hand_tok, joker_tok, cons_tok, shop_tok, vsale_tok, boost_tok, pack_tok = {}, {}, {}, {}, {}, {}, {}
  local skip_tok = {}

  -- GLOBAL
  local g = tok(KIND.GLOBAL, vid("state:" .. o.state))
  local target = o.blind and o.blind.target or 0
  set(g, 1, (o.ante or 0) / 8); set(g, 2, (o.round or 0) / 24); set(g, 3, (o.stake or 1) / 8)
  set(g, 4, slog(o.money)); set(g, 5, (o.hands_left or 0) / 5); set(g, 6, (o.discards_left or 0) / 5)
  set(g, 7, (o.hand_size or 0) / 10); set(g, 8, (o.joker_slots or 0) / 5); set(g, 9, (o.consumable_slots or 0) / 5)
  set(g, 10, (o.reroll_cost or 0) / 10); set(g, 11, (o.interest_cap or 0) / 25); set(g, 12, slog(target))
  set(g, 13, slog(o.chips)); set(g, 14, target > 0 and math.min((o.chips or 0) / target, 4) or 0)
  set(g, 15, (o.pack_choices or 0) / 2); set(g, 16, (o.free_rerolls or 0) / 2); set(g, 17, (o.bankrupt_at or 0) / 20)
  onehot(g, 18, 3, ({ Small = 1, Big = 2, Boss = 3 })[o.blind_on_deck or ""])
  set(g, 21, #o.deck / 52); set(g, 22, #o.full_deck / 52)

  -- DECK: remaining, then full
  for di, deck in ipairs({ o.deck, o.full_deck }) do
    local r = tok(KIND.DECK, 0)
    for _, c in ipairs(deck) do
      local rk = RANKS[c.rank]; if rk then r[rk - 2 + 1 + 2] = r[rk - 2 + 1 + 2] + 0.25 end
      local s = SUITS[c.suit]; if s then r[13 + s + 2] = r[13 + s + 2] + 1 / 13 end
      local e = ENH[c.enhancement or ""]; if e then r[17 + e + 2] = r[17 + e + 2] + 1 / 8 end
      if c.edition then r[26 + 2] = r[26 + 2] + 1 / 8 end
      if c.seal then r[27 + 2] = r[27 + 2] + 1 / 8 end
    end
    set(r, 28, #deck / 52); set(r, 29, di - 1)
  end

  -- HANDTYPE
  for _, name in ipairs(sorted_keys(o.hand_levels)) do
    local h = o.hand_levels[name]
    local r = tok(KIND.HANDTYPE, vid("hand:" .. name))
    set(r, 1, (h.level or 1) / 10); set(r, 2, slog(h.chips)); set(r, 3, slog(h.mult))
    set(r, 4, slog(h.played)); set(r, 5, (h.played_this_round or 0) / 3); set(r, 6, h.visible and 1 or 0)
  end

  -- BLIND: the current one, then the three choices
  if o.blind then
    local r = tok(KIND.BLIND, vid(o.blind.key))
    set(r, 1, slog(o.blind.target)); set(r, 2, o.blind.boss and 1 or 0); set(r, 3, o.blind.disabled and 1 or 0); set(r, 4, 1)
  end
  for i, slot in ipairs({ "Small", "Big", "Boss" }) do
    local key = o.blind_choices and o.blind_choices[slot]
    if key then
      local r = tok(KIND.BLIND, vid(key))
      onehot(r, 5, 3, i)
      onehot(r, 8, 6, BSTATE[o.blind_states and o.blind_states[slot] or ""])
    end
    local tag = o.skip_tags and o.skip_tags[slot]
    if tag and i <= 2 then local r, ti = tok(KIND.SKIPTAG, vid(tag)); onehot(r, 1, 2, i); skip_tok[slot] = ti end
  end

  -- TARGETS of The Idol, Mail-In Rebate, Ancient Joker, Castle
  do
    local t, rt = tok(KIND.TARGETS, 0), o.round_targets or {}
    if rt.idol then set(t, 1, (RANKS[rt.idol.rank] or 0) / 14); onehot(t, 2, 4, SUITS[rt.idol.suit]) end
    set(t, 6, (RANKS[rt.mail or ""] or 0) / 14)
    onehot(t, 7, 4, SUITS[rt.ancient or ""]); onehot(t, 11, 4, SUITS[rt.castle or ""])
  end

  -- CARDS in hand (order = G.hand.cards, left to right)
  for i, c in ipairs(o.hand) do
    local r, ti = tok(KIND.CARD, vid(c.enhancement)); card_features(r, c, 1); set(r, 30, (i - 1) / 10)
    hand_tok[i] = ti
  end

  local function item_features(r, it, with_card)
    if it.face_down then set(r, 32, 1); return end
    onehot(r, 1, 4, EDITIONS[it.edition]); set(r, 5, slog(it.sell)); set(r, 6, slog(it.cost))
    set(r, 7, it.eternal and 1 or 0); set(r, 8, it.perishable and 1 or 0); set(r, 9, (it.perish_tally or 0) / 5)
    set(r, 10, it.rental and 1 or 0); set(r, 11, it.debuff and 1 or 0); onehot(r, 12, 7, SETS[it.set])
    local nums = ability_numbers(it.ability)
    for k = 1, #nums do set(r, 18 + k, slog(nums[k])) end
    if with_card and it.card then
      onehot(r, 27, 4, SUITS[it.card.suit]); set(r, 31, (RANKS[it.card.rank] or 0) / 14)
    end
  end
  for i, it in ipairs(o.jokers) do
    local r, ti = tok(KIND.JOKER, it.face_down and 0 or vid(it.key)); item_features(r, it); set(r, 27, (i - 1) / 5)
    joker_tok[i] = ti
  end
  for i, it in ipairs(o.consumables) do
    local r, ti = tok(KIND.CONSUMABLE, vid(it.key)); item_features(r, it); cons_tok[i] = ti
  end
  for i, it in ipairs(o.shop or {}) do
    local r, ti = tok(KIND.SHOP, vid(it.card and (it.card.enhancement or "c_base") or it.key)); item_features(r, it, true); shop_tok[i] = ti
  end
  for i, it in ipairs(o.vouchers_for_sale or {}) do
    local r, ti = tok(KIND.VOUCHER_SALE, vid(it.key)); item_features(r, it); vsale_tok[i] = ti
  end
  for i, it in ipairs(o.boosters or {}) do
    local r, ti = tok(KIND.BOOSTER, vid(it.key)); item_features(r, it); boost_tok[i] = ti
  end
  for i, it in ipairs(o.pack or {}) do
    local r, ti = tok(KIND.PACK, vid(it.card and (it.card.enhancement or "c_base") or it.key)); item_features(r, it, true); pack_tok[i] = ti
  end
  for _, t in ipairs(o.tags or {}) do tok(KIND.TAG, vid(t)) end
  for _, v in ipairs(o.vouchers or {}) do tok(KIND.VOUCHER, vid(v)) end

  -- ACTIONS, in exactly the order H.legal_actions() returns them
  local acts = H.legal_actions()
  H._encoded_actions = acts
  local arow = {}
  for k, a in ipairs(acts) do
    local t = a.type
    local A, B = -1, -1
    if t == "buy" or t == "buy_and_use" then A = shop_tok[a.i]
    elseif t == "redeem" then A = vsale_tok[a.i]
    elseif t == "open_pack" then A = boost_tok[a.i]
    elseif t == "pick" then A = pack_tok[a.i]
    elseif t == "use_consumable" or t == "sell_consumable" then A = cons_tok[a.i]
    elseif t == "sell_joker" then A = joker_tok[a.i]
    elseif t == "move_joker" then A, B = joker_tok[a.i], joker_tok[a.j]
    elseif t == "skip_blind" then A = skip_tok[G.GAME.blind_on_deck] or -1 end
    local c = { -1, -1, -1, -1, -1 }
    for n, hi in ipairs(a.cards or {}) do if n <= 5 then c[n] = hand_tok[hi] end end
    local ht = 0
    if t == "play" then
      local cards = {}
      for _, hi in ipairs(a.cards) do cards[#cards + 1] = G.hand.cards[hi] end
      local text = G.FUNCS.get_poker_hand_info(cards)
      ht = vid("hand:" .. text)
    end
    arow[k] = table.concat({ ACTION[t] or 0, A or -1, B, c[1], c[2], c[3], c[4], c[5], ht }, " ")
  end

  local out = { ntok .. " " .. #acts }
  for i = 1, ntok do out[#out + 1] = table.concat(rows[i], " ") end
  for k = 1, #acts do out[#out + 1] = arow[k] end
  return table.concat(out, "\n")
end

function H.step_index(i)
  local a = H._encoded_actions and H._encoded_actions[i]
  assert(a, "no encoded action " .. tostring(i))
  H._encoded_actions = nil
  return H.step(a)
end

return H
