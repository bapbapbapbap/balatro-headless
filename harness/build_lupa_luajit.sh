#!/usr/bin/env bash
# Rebuilds vendor/lupa: lupa 2.8 linked against LuaJIT 2.0.5, the LuaJIT in Balatro's Windows
# Steam build (LOVE 11.5; read from the install's lua51.dll: "LuaJIT 2.0.5").
# Stock lupa ships a newer LuaJIT that randomizes string hashing per process, so pairs() order
# changes run to run, and so do the Orbital Tag's and To Do List's picks. 2.0.5 hashes
# deterministically and matches the game.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
cd "$WORK"

git clone -q https://github.com/LuaJIT/LuaJIT.git luajit
REL=0bf80b07   # "RELEASE LuaJIT-2.0.5"

url=$(curl -sS https://pypi.org/pypi/lupa/2.8/json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print([u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0])")
curl -sSL -o lupa.tar.gz "$url"
tar xzf lupa.tar.gz
cd lupa-2.8
rm -rf third-party/lua5* third-party/luajit20 third-party/luajit21
git -C "$WORK/luajit" archive --format=tar --prefix=luajit20/ "$REL" | tar x -C third-party/
python3 setup.py build_ext -i

rm -rf "$HERE/vendor/lupa"
mkdir -p "$HERE/vendor"
cp -r lupa "$HERE/vendor/lupa"
find "$HERE/vendor/lupa" \( -name '*.c' -o -name '*.pyx' -o -name '*.pxi' -o -name '*.pxd' \) -delete
rm -rf "$HERE/vendor/lupa/tests"
python3 -c "import sys; sys.path.insert(0, '$HERE/vendor'); from lupa import luajit20; print(luajit20.LuaRuntime().eval('jit.version'))"
