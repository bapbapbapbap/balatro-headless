-- Headless stand-in for the LOVE runtime. Game logic runs unchanged; graphics, audio,
-- threads and input become no-ops. Nothing here touches game RNG: Balatro's game
-- randomness is all keyed through pseudoseed/pseudorandom, which reseed math.random per call.

local SRC = assert(HEADLESS_SRC, "HEADLESS_SRC not set")
local SAVE = assert(HEADLESS_SAVE, "HEADLESS_SAVE not set")

-- A universal dummy: any field is itself, calling it returns itself, arithmetic yields 0.
local D
local dmt = {}
dmt.__index = function(_, k)
  if type(k) == "string" then
    if k:match("^getWidth") or k:match("^getHeight") or k == "getDPIScale" or k == "getPixelWidth" or k == "getPixelHeight" then
      return function() return 1 end
    elseif k == "getDimensions" or k == "getPixelDimensions" then
      return function() return 1, 1 end
    elseif k:match("^is") then
      return function() return false end
    end
  end
  return D
end
dmt.__newindex = function() end
dmt.__call = function() return D end
local zero = function() return 0 end
dmt.__add, dmt.__sub, dmt.__mul, dmt.__div, dmt.__mod, dmt.__pow, dmt.__unm = zero, zero, zero, zero, zero, zero, zero
dmt.__concat = function(a, b) return (type(a) == "string" and a or "") .. (type(b) == "string" and b or "") end
dmt.__len = zero
D = setmetatable({}, dmt)
HEADLESS_DUMMY = D

local function dummy_module(overrides)
  return setmetatable(overrides or {}, { __index = function() return D end })
end

local clock = 0
local function readfile(p)
  local f = io.open(p, "rb"); if not f then return nil end
  local s = f:read("*a"); f:close(); return s
end

love = {}
love.timer = dummy_module({
  getTime = function() return clock end,
  step = function() return 0 end,
  sleep = function() end,
  getFPS = function() return 60 end,
  getDelta = function() return 1/60 end,
})
HEADLESS_ADVANCE_CLOCK = function(dt) clock = clock + dt end

love.system = dummy_module({
  getOS = function() return "Linux" end,
  getProcessorCount = function() return 1 end,
  openURL = function() end,
})

love.graphics = dummy_module({
  getDimensions = function() return 1920, 1080 end,
  getWidth = function() return 1920 end,
  getHeight = function() return 1080 end,
  getDPIScale = function() return 1 end,
  isActive = function() return false end,
  getStats = function() return { drawcalls = 0, texturememory = 0 } end,
  newFont = function() return D end,
  newImage = function() return D end,
  newCanvas = function() return D end,
  newShader = function() return D end,
  newQuad = function() return D end,
})

love.window = dummy_module({
  getMode = function() return 1920, 1080, { fullscreen = false, display = 1, vsync = 1, refreshrate = 60 } end,
  getDesktopDimensions = function() return 1920, 1080 end,
  getDisplayCount = function() return 1 end,
  getDisplayName = function() return "headless" end,
  getFullscreenModes = function() return { { width = 1920, height = 1080 } } end,
  setMode = function() return true end,
  updateMode = function() return true end,
  getDPIScale = function() return 1 end,
  toPixels = function(x) return x end,
  fromPixels = function(x) return x end,
  isOpen = function() return true end,
  getPosition = function() return 0, 0, 1 end,
})

love.mouse = dummy_module({ getPosition = function() return 0, 0 end, isDown = function() return false end })
love.keyboard = dummy_module({ isDown = function() return false end })
love.touch = dummy_module({ getTouches = function() return {} end })
love.joystick = dummy_module({ getJoysticks = function() return {} end })
love.audio = dummy_module()
love.sound = dummy_module()
love.event = dummy_module({ poll = function() return function() return nil end end })
love.thread = dummy_module()
love.arg = dummy_module()
love.image = dummy_module()

love.data = dummy_module({
  -- Saves never leave the process headless, so pass data through uncompressed.
  compress = function(_, _, data) return data end,
  decompress = function(_, _, data) return data end,
})

love.filesystem = dummy_module({
  read = function(p) return readfile(SRC .. "/" .. p) or readfile(SAVE .. "/" .. p) end,
  write = function(p, data)
    local f = io.open(SAVE .. "/" .. p, "wb"); if not f then return false end
    f:write(data); f:close(); return true
  end,
  getInfo = function(p)
    if p:match("^resources/") then return { type = "file" } end  -- art/fonts/sounds aren't shipped; pretend they exist
    if readfile(SRC .. "/" .. p) or readfile(SAVE .. "/" .. p) then return { type = "file" } end
    return nil
  end,
  getDirectoryItems = function() return {} end,
  createDirectory = function() return true end,
  remove = function() return true end,
  getSaveDirectory = function() return SAVE end,
  getSourceBaseDirectory = function() return SRC end,
  getIdentity = function() return "Balatro" end,
  append = function() return true end,
})
