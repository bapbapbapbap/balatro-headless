#!/usr/bin/env bash
# Assembles the game exactly as it runs on the user's machine under Lovely + Steamodded.
#
# usage: build_modded_tree.sh <vanilla source dir> <Balatro Mods dir> <out dir>
#   vanilla source: the game's Lua extracted from Balatro.exe (bundle's source/)
#   Mods dir:       %AppData%\Balatro\Mods (needs lovely/dump and smods/; the dump is
#                   rewritten each time the game launches with Lovely)
#
# Produces <out>/game (vanilla overlaid with Lovely's patched files) and <out>/Mods/smods.
# Run with:  replay.py LOG --src <out>/game --mods <out>/Mods
set -euo pipefail
SRC="$1"; MODS="$2"; OUT="$3"
rm -rf "$OUT"; mkdir -p "$OUT/game" "$OUT/Mods/lovely"
cp -r "$SRC/." "$OUT/game/"
( cd "$MODS/lovely/dump" && find . -name '*.lua' ! -path './lovely/*' ! -path './love/*' ) | while read -r f; do
  mkdir -p "$OUT/game/$(dirname "$f")"
  cp "$MODS/lovely/dump/$f" "$OUT/game/$f"
done
cp -r "$MODS/smods" "$OUT/Mods/smods"
echo "patched files: $(cd "$MODS/lovely/dump" && find . -name '*.lua' ! -path './lovely/*' ! -path './love/*' | wc -l)"
echo "Steamodded: $(cat "$OUT/Mods/smods/version.lua")"
