-- Boots Balatro's own game logic headless. Requires HEADLESS_SRC / HEADLESS_SAVE set
-- and love_stub.lua loaded first.

if HEADLESS_MODS then
  -- Lovely + Steamodded, as on the user's machine (see modded_boot.lua).
  dofile(HEADLESS_HARNESS .. "/modded_boot.lua")
else
  package.path = HEADLESS_SRC .. "/?.lua;" .. package.path

  -- Same load order as the game's main.lua, minus love.run / Steam.
  require "engine/object"
  require "bit"
  require "engine/string_packer"
  require "engine/controller"
  require "back"
  require "tag"
  require "engine/event"
  require "engine/node"
  require "engine/moveable"
  require "engine/sprite"
  require "engine/animatedsprite"
  require "functions/misc_functions"
  require "game"
  require "globals"
  require "engine/ui"
  require "functions/UI_definitions"
  require "functions/state_events"
  require "functions/common_events"
  require "functions/button_callbacks"
  require "functions/misc_functions"
  require "functions/test_functions"
  require "card"
  require "cardarea"
  require "blind"
  require "card_character"
  require "engine/particles"
  require "engine/text"
  require "challenges"

  dofile(HEADLESS_SRC .. "/main.lua")     -- defines love.resize etc.; love.run/love.load are never called
end

play_sound = function() end            -- audio is the one thing that would otherwise hit threads
-- The game's per-frame GC helper spins on love.timer until a time budget runs out. Our clock
-- only moves between frames, so it would always hit its 1000-step cap. Let Lua's own GC run.
nuGC = function() end

local H = {}

function H.start_up()
  Game.splash_screen = function() end  -- skip intro; we go straight to runs
  G:start_up()
  G.SETTINGS.tutorial_complete = true
  G.SETTINGS.tutorial_progress = nil
  G.SETTINGS.GAMESPEED = 1000          -- animation delays elapse in one frame; event order is unchanged
end

-- Strip work that only produces visuals. Each cut is kept only if the replays still match
-- the real game exactly (see replay.py); the game mixes logic into some visual code, so
-- this is a list of specific functions, not "turn off the UI".
--   Moveable:move   eases VT (drawn position) toward T (layout position). Snapping VT to T
--                   keeps the few logic checks that compare them (shop/round-eval/pack
--                   slide-ins) true immediately, as they'd become after the animation.
--   Sprite/DynaText/Particles move, UIElement:update, DynaText:update, Particles:update
--                   card art, text and particle positions, button styling, text animation.
-- UIBox:recalculate stays: an open pack's card area sits inside a UIBox, and its position
-- gates when the pack's cards are placed.
function H.strip_visuals()
  -- Same layout (T) as the real move(): alignment to majors, minors following their major,
  -- glued objects sharing their major's T. Only the easing of VT and juice/parallax go.
  function Moveable:move(dt)
    if self.FRAME.MOVE >= G.FRAMES.MOVE then return end
    self.FRAME.OLD_MAJOR = self.FRAME.MAJOR
    self.FRAME.MAJOR = nil
    self.FRAME.MOVE = G.FRAMES.MOVE
    if not self.created_on_pause and G.SETTINGS.paused then return end
    self:align_to_major()
    local role = self.role
    if role.role_type == 'Glued' then
      if role.major then self.T = role.major.T end
    elseif role.role_type == 'Minor' and role.major then
      if role.major.FRAME.MOVE < G.FRAMES.MOVE then role.major:move(dt) end
      local mt = role.major:get_major()
      local M = mt.major
      local ox, oy = role.offset.x + mt.offset.x, role.offset.y + mt.offset.y
      if role.r_bond ~= 'Weak' and (M.T.r > 0.0001 or M.T.r < -0.0001) then
        local c, s = math.cos(M.T.r), math.sin(M.T.r)
        local ww, hh = -self.T.w / 2 + M.T.w / 2, -self.T.h / 2 + M.T.h / 2
        local x0, y0 = ox - ww, oy - hh
        ox, oy = x0 * c - y0 * s + ww, x0 * s + y0 * c + hh
      end
      self.T.x, self.T.y = M.T.x + ox, M.T.y + oy
    end
    local T, VT = self.T, self.VT
    VT.x, VT.y, VT.w, VT.h, VT.r, VT.scale = T.x, T.y, T.w, T.h, T.r, T.scale
    if self.alignment and self.alignment.lr_clamp then self:lr_clamp() end
    self.NEW_ALIGNMENT = false
  end
  -- Sprites (card art, stickers, chips), text and particles have no layout anything reads.
  local function no_move(self) self.FRAME.MOVE = G.FRAMES.MOVE end
  Sprite.move, DynaText.move = no_move, no_move
  if Particles then Particles.move = no_move end
  UIElement.update = function() end
  -- UIBox.recalculate kept: pack areas sit inside UIBoxes and their layout gates card creation
  DynaText.update = function() end
  if Particles then Particles.update = function() end end
  H.visuals_stripped = true
end

-- Mirrors the second branch of G.FUNCS.unlock_all (the profile button).
function H.unlock_all()
  G.PROFILES[G.SETTINGS.profile].all_unlocked = true
  for _, pool in ipairs({ G.P_CENTERS, G.P_BLINDS, G.P_TAGS }) do
    for _, v in pairs(pool) do
      if not v.demo and not v.wip then v.alerted = true; v.discovered = true; v.unlocked = true end
    end
  end
end

-- States where the game waits for the player.
local DECISION = {}
for _, k in ipairs({ "SELECTING_HAND", "BLIND_SELECT", "SHOP", "ROUND_EVAL", "GAME_OVER",
                     "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK",
                     "SMODS_BOOSTER_OPENED" }) do   -- Steamodded routes every pack through this state
  if G.STATES[k] then DECISION[G.STATES[k]] = true end
end

local function queue_busy()
  for _, q in pairs(G.E_MANAGER.queues) do
    for _, ev in ipairs(q) do
      if ev.blocking or ev.blockable then return true end
    end
  end
  return false
end

-- A booster's cards are put into the pack area by a non-blocking event (card.lua, Card:open),
-- so an open pack isn't waiting on the player until its cards are actually there.
local PACK = {}
for _, k in ipairs({ "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK",
                     "SMODS_BOOSTER_OPENED" }) do
  if G.STATES[k] then PACK[G.STATES[k]] = true end
end
local function pack_ready()
  if not PACK[G.STATE] then return true end
  return G.pack_cards ~= nil and G.pack_cards.cards[1] ~= nil
end

-- Pump the game loop one frame at a time (one event-queue pass per frame, as in the real
-- game) until the game is waiting for input: a decision state, its state handler done,
-- and nothing blocking left in the queues.
function H.settle(max_steps)
  max_steps = max_steps or 5000
  local dt = 1 / 30
  local quiet = 0
  for i = 1, max_steps do
    HEADLESS_ADVANCE_CLOCK(dt)
    G:update(dt)
    if G.STATE == G.STATES.GAME_OVER or (G.GAME.won and G.OVERLAY_MENU) then
      quiet = quiet + 1                  -- game over, or the win screen: its UI events never drain
      if quiet >= 5 then return i end
    elseif DECISION[G.STATE] and G.STATE_COMPLETE and not queue_busy() and pack_ready() then
      quiet = quiet + 1
      if quiet >= 2 then return i end
    else
      quiet = 0
    end
  end
  return -max_steps   -- negative: never settled
end

-- challenge: a challenge id from G.CHALLENGES. Started the way the game's challenge menu does
-- (G.FUNCS.start_challenge_run): stake 1, the challenge passed to start_run, which applies its
-- deck, rules, starting jokers and bans.
function H.start_run(back_name, stake, seed, challenge)
  G.GAME = G.GAME or G:init_game_object()
  local ch = nil
  if challenge then
    for _, c in ipairs(G.CHALLENGES) do if c.id == challenge then ch = c end end
    assert(ch, "no such challenge: " .. tostring(challenge))
    back_name, stake = "b_challenge", 1
  end
  G.GAME.viewed_back = G.P_CENTERS[back_name] and { name = G.P_CENTERS[back_name].name } or { name = back_name }
  G:start_run({ stake = stake, seed = seed, challenge = ch })
  if H.save_mode then
    -- A save counts progress (unlocks, stickers, deck and challenge wins) only for unseeded runs.
    -- The seed still drives every random draw; only the flag the game checks before recording
    -- progress is cleared, so a run from a random seed counts as an ordinary run would.
    G.GAME.seeded = false
  end
  return H.settle()
end

function H.challenge_ids()
  local out = {}
  for _, c in ipairs(G.CHALLENGES or {}) do out[#out + 1] = c.id end
  return out
end

function H.joker_keys()
  local out = {}
  for k, v in pairs(G.P_CENTERS) do if v.set == "Joker" then out[#out + 1] = k end end
  table.sort(out)
  return out
end

-- Keys of the jokers held right now, in slot order (what earns a joker its win sticker).
function H.jokers_held()
  local out = {}
  for i, c in ipairs(G.jokers and G.jokers.cards or {}) do out[i] = c.config.center.key end
  return out
end

function H.deck_keys()
  local out = {}
  for k, v in pairs(G.P_CENTERS) do
    if v.set == "Back" and k ~= "b_challenge" then out[#out + 1] = k end
  end
  table.sort(out)
  return out
end

function H.state_name()
  for k, v in pairs(G.STATES) do if v == G.STATE then return k end end
  return tostring(G.STATE)
end

return H
