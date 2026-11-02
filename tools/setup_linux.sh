#!/usr/bin/env bash
# One-shot setup of the headless sim on Linux or WSL, from a balatro_bundle.zip.
#
# usage: bash setup_linux.sh <folder containing balatro_bundle.zip> [work dir, default ~/balatro]
#   e.g. in WSL: bash /mnt/c/Users/<you>/balatro-work/setup_linux.sh /mnt/c/Users/<you>/balatro-work
#
# Does: clone (or update) the repo, build the LuaJIT 2.0.5 runtime, unpack the bundle (the game
# source stays on this machine, under the work dir), build the modded game tree, then run every
# accuracy check and write the results to <bundle folder>/setup_report.txt.
#
# The work dir lives on the Linux filesystem (not /mnt/c), which is much faster under WSL.
set -uo pipefail
BUNDLE_DIR="$(cd "${1:?usage: setup_linux.sh <folder with balatro_bundle.zip> [work dir]}" && pwd)"
W="${2:-$HOME/balatro}"
REPORT="$BUNDLE_DIR/setup_report.txt"
mkdir -p "$W"
exec > >(tee "$REPORT") 2>&1
echo "setup started $(date)  host $(hostname)  cores $(nproc)  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"

step() { echo; echo "== $*"; }
fail() { echo "FAILED: $*"; exit 1; }

step "repo"
if [ -d "$W/balatro-headless/.git" ]; then
  git -C "$W/balatro-headless" pull -q --ff-only || fail "git pull"
else
  git clone -q https://github.com/bapbapbapbap/balatro-headless "$W/balatro-headless" || fail "git clone"
fi
R="$W/balatro-headless"
git -C "$R" log --oneline -1

step "python env"
python3 -m venv "$W/venv" || fail "venv (install python3-venv)"
. "$W/venv/bin/activate"
pip install -q --upgrade pip setuptools wheel cython || fail "pip"
python3 --version

step "runtime: lupa against LuaJIT 2.0.5"
if python3 -c "import sys; sys.path.insert(0, '$R/harness/vendor'); from lupa import luajit20; assert luajit20.LuaRuntime().eval('jit.version') == 'LuaJIT 2.0.5'" 2>/dev/null; then
  echo "already built"
else
  bash "$R/harness/build_lupa_luajit.sh" > "$W/build_lupa.log" 2>&1 || { tail -20 "$W/build_lupa.log"; fail "lupa build"; }
  tail -1 "$W/build_lupa.log"
fi

step "bundle"
rm -rf "$W/local" && mkdir -p "$W/local"
python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$BUNDLE_DIR/balatro_bundle.zip" "$W/local" || fail "unzip"
grep -E "game version|sha256|Antelytics|lovely dump|mod files" "$W/local/info.txt"
grep -q "0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47" "$W/local/info.txt" \
  || echo "WARNING: Balatro.exe is not the pinned 1.0.1o build; results may not match the fixtures"

step "modded game tree"
bash "$R/harness/build_modded_tree.sh" "$W/local/source" "$W/local/mods" "$W/local/modded" || fail "modded tree"

step "check 1: replay fixtures on the modded game (Lovely + Steamodded)"
bash "$R/tests/run_replays.sh" "$W/local/modded" || echo "CHECK 1 FAILED"

step "check 2: env checks + fuzz on the unmodded game"
bash "$R/tests/run_env_checks.sh" "$W/local/source" || echo "CHECK 2 FAILED"

step "check 3: differential check, fast vs untouched game, 20 runs"
( cd "$R/harness" && python3 diff_check.py --src "$W/local/source" --runs 20 | tail -3 ) || echo "CHECK 3 FAILED"

step "speed"
( cd "$R/harness" && python3 fuzz.py --src "$W/local/source" --runs 20 --mode subsets | grep -E "^mode" )

echo; echo "setup finished $(date)"
echo "work dir: $W"
