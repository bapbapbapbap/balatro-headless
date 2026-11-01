"""Python driver for the headless Balatro runtime (LuaJIT via lupa)."""
import os
import sys
import tempfile

# Balatro on Windows Steam runs LOVE 11.5 with LuaJIT 2.0.5 (read from the install's lua51.dll).
# Newer LuaJIT randomizes string hashing per process, which changes pairs() order, which changes
# some of the game's RNG draws (Orbital Tag, To Do List). vendor/lupa is lupa 2.8 rebuilt
# against LuaJIT 2.0.5 (see build_lupa_luajit.sh).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from lupa import luajit20 as luajit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_runtime(src_dir, mods_dir=None):
    """src_dir: game source (vanilla, or Lovely's patched dump overlaid on it).
    mods_dir: a Mods folder with Steamodded in it; None runs the unmodded game."""
    lua = luajit.LuaRuntime(unpack_returned_tuples=True)
    g = lua.globals()
    g.HEADLESS_HARNESS = HERE
    g.HEADLESS_SRC = os.path.abspath(src_dir)
    if mods_dir:
        g.HEADLESS_MODS = os.path.abspath(mods_dir)
    g.HEADLESS_SAVE = tempfile.mkdtemp(prefix="balatro_save_")
    lua.execute(open(os.path.join(HERE, "love_stub.lua")).read())
    H = lua.execute(open(os.path.join(HERE, "boot.lua")).read())
    return lua, H


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 run.py <game source dir> [Mods dir]")
    lua, H = make_runtime(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    H.start_up()
    H.unlock_all()
    print("booted, version", lua.globals().G.VERSION)
