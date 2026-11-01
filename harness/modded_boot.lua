-- Boots the game the way Lovely + Steamodded do on the user's machine, headless.
--
-- Inputs (globals, set by the Python driver):
--   HEADLESS_SRC   game tree: vanilla source overlaid with Lovely's dump (Mods/lovely/dump),
--                  i.e. every game file exactly as Lovely patched it.
--   HEADLESS_MODS  a Mods directory containing Steamodded (smods/).
--   HEADLESS_SAVE  scratch save dir.
-- love_stub.lua must already be loaded.
--
-- What Lovely does beyond patching files, and what stands in for it here:
--   - module patches (package.preload entries)          -> preload table below
--   - SMODS.preflight.core runs before main.lua, with its lovely_path line patched
--   - the `lovely` module (version, mod_dir, reload hooks) -> minimal table
-- Steamodded's nativefs calls into LOVE's PhysFS through FFI; here it is plain io.

local SRC, MODS = HEADLESS_SRC, HEADLESS_MODS
local SMODS_DIR = MODS .. "/smods/"

local function readfile(p)
  local f = io.open(p, "rb"); if not f then return nil end
  local s = f:read("*a"); f:close(); return s
end
local function shell_quote(s) return "'" .. tostring(s):gsub("'", "'\\''") .. "'" end
local function is_dir(p) return os.execute("test -d " .. shell_quote(p)) == 0 end

-- ---------- nativefs, headless ----------
local cwd = SMODS_DIR
local function abs(p)
  p = tostring(p):gsub("\\", "/")
  if p:sub(1, 1) == "/" then return p end
  return (cwd:gsub("/?$", "/")) .. p
end
local ASSET = { png = true, ogg = true, wav = true, ttf = true, otf = true, jpg = true }
local function is_asset(p) local ext = p:match("%.(%w+)$"); return ext and ASSET[ext:lower()] end

local NFS = {}
function NFS.read(a, b)
  local name = (a == "string" or a == "data") and b or a
  local p = abs(name)
  local s = readfile(p)
  if s == nil and is_asset(p) then return "", 0 end  -- art/sound weren't copied; contents never matter headless
  if s == nil then return nil, "Could not open " .. p end
  return s, #s
end
function NFS.write(name, data) local f = io.open(abs(name), "wb"); if not f then return false end; f:write(data); f:close(); return true end
function NFS.append(name, data) local f = io.open(abs(name), "ab"); if not f then return false end; f:write(data); f:close(); return true end
function NFS.lines(name) local s = readfile(abs(name)) or ""; return s:gmatch("([^\r\n]*)\r?\n?") end
function NFS.load(name) local s = readfile(abs(name)); if not s then return nil, "not found" end; return load(s, "=" .. name) end
function NFS.getWorkingDirectory() return (cwd:gsub("/$", "")) end
function NFS.setWorkingDirectory(p) cwd = abs(p):gsub("/?$", "/"); return true end
function NFS.getDriveList() return { "/" } end
function NFS.createDirectory(p) os.execute("mkdir -p " .. shell_quote(abs(p))); return true end
function NFS.remove(p) return os.remove(abs(p)) ~= nil end
function NFS.getDirectoryItems(dir)
  local out, h = {}, io.popen("ls -1A " .. shell_quote(abs(dir)) .. " 2>/dev/null")
  for line in h:lines() do out[#out + 1] = line end
  h:close(); table.sort(out); return out
end
function NFS.getInfo(p, filtertype)
  p = abs(p)
  local t
  if is_dir(p) then t = "directory"
  else
    local f = io.open(p, "rb")
    if f then f:close(); t = "file" elseif is_asset(p) then t = "file" end
  end
  if not t or (filtertype and filtertype ~= t) then return nil end
  return { type = t, size = 0, modtime = 0 }
end
function NFS.getDirectoryItemsInfo(dir, filtertype)
  local out = {}
  for _, n in ipairs(NFS.getDirectoryItems(dir)) do
    local info = NFS.getInfo(abs(dir):gsub("/?$", "/") .. n, filtertype)
    if info then info.name = n; out[#out + 1] = info end
  end
  return out
end
function NFS.newFileData(p) return HEADLESS_DUMMY end
function NFS.newFile(p) return HEADLESS_DUMMY end
function NFS.mount() return false end
function NFS.unmount() return true end
function NFS.smodsAddRedirect() return true end
function NFS.getNormalizedPath(p) return abs(p) end

-- ---------- Lovely modules ----------
local function file_module(rel) return function() return assert(load(assert(readfile(SMODS_DIR .. rel), rel), "=[lovely " .. rel .. "]"))() end end
package.preload["SMODS.version"] = file_module("version.lua")
package.preload["SMODS.release"] = file_module("release.lua")
package.preload["json"] = file_module("libs/json/json.lua")
package.preload["nativefs"] = function() return NFS end
package.preload["SMODS.nativefs"] = function() return NFS end
package.preload["luajit-curl"] = function() return {} end
package.preload["socket"] = function()   -- Steamodded's debug console; never connects headless
  return { tcp = function() return { settimeout = function() end, connect = function() return nil end,
                                     send = function() end, close = function() end } end,
           gettime = function() return 0 end }
end
-- LOVE ships a utf8 module; only Steamodded's crash-report formatting requires it.
package.preload["utf8"] = function()
  local m = rawget(_G, "utf8") or {}
  m.charpattern = m.charpattern or "[%z\1-\127\194-\244][\128-\191]*"
  m.char = m.char or function(...) local t = {} for i, c in ipairs({...}) do t[i] = string.char(c % 256) end return table.concat(t) end
  m.len = m.len or function(s) local _, n = s:gsub(m.charpattern, "") return n end
  m.offset = m.offset or function(s, n) return n end
  m.codepoint = m.codepoint or function(s, i) return s:byte(i or 1) end
  m.codes = m.codes or function(s) local i = 0 return function() i = i + 1 if i <= #s then return i, s:byte(i) end end end
  return m
end
local quiet_print = print
print = function(...) if HEADLESS_VERBOSE then quiet_print(...) end end  -- Steamodded logs a lot
package.preload["SMODS.https"] = function()   -- no network in the sim
  return setmetatable({}, { __index = function() return function() return nil, "offline" end end })
end
for _, m in ipairs({ "sharedUtil", "logging", "loader", "sharedUI" }) do
  package.preload["SMODS.preflight." .. m] = file_module("src/preflight/" .. m .. ".lua")
end
package.preload["lovely"] = function()
  return {
    version = "0.10.0",
    mod_dir = MODS,
    reload_patches = function() return true end,
    remove_var = function() return nil end,
    set_var = function() end,
    get_var = function() return nil end,
    apply_patches = function(_, src) return src end,
  }
end

-- SMODS.preflight.core is a load_now module that runs before main.lua; Lovely patches its
-- lovely_path line to point at Steamodded's folder.
do
  local src = assert(readfile(SMODS_DIR .. "src/preflight/core.lua"))
  local n
  src, n = src:gsub("local lovely_path = false %-%- This line is patched, don't edit it",
                    "local lovely_path = [[" .. SMODS_DIR .. "]]")
  assert(n == 1, "preflight core lovely_path line not found")
  package.loaded["SMODS.preflight.core"] = assert(load(src, "=[lovely SMODS.preflight.core \"src/preflight/core.lua\"]"))() or true
end

package.path = SRC .. "/?.lua;" .. package.path
-- main.lua as Lovely patched it: crash handler prepended, requires, love callbacks,
-- and Steamodded's core appended (loads SMODS src files through NFS).
assert(load(assert(readfile(SRC .. "/main.lua")), "=main.lua"))()
