#!/usr/bin/env bash
# Z0 (docs/hypotheses-reproduce.md, 2026-11-27): AnteZero's own difftest (namecallfilter/antezero, MIT, pinned clone
# at ~/balatro/antezero after port_oracle_linux.py), 20 games per deck at stakes 1 and 8, 400 steps, inject 0.15.
# Ours (Claude's, reviewed by chud). Re-running skips cells whose output file is complete.
set -u
AZ=${AZ:-$HOME/balatro/antezero}
OUT=${OUT:-/mnt/c/Users/Jacobi/balatro-work/antezero/z0}
export BALATRO_LUA51=${BALATRO_LUA51:-$HOME/balatro/luajit205/src/libluajit.so}
export BALATRO_SRC=${BALATRO_SRC:-$HOME/balatro/az_source}
JOBS=${JOBS:-12}
mkdir -p "$OUT"
cells=()
for stake in 1 8; do for back in b_red b_blue b_yellow b_green b_black b_magic b_nebula b_ghost b_abandoned b_checkered b_zodiac b_painted b_anaglyph b_plasma b_erratic; do
  f="$OUT/${back}_${stake}.txt"
  grep -q "games matched" "$f" 2>/dev/null || cells+=("$back $stake")
done; done
echo "${#cells[@]} cells to run" | tee -a "$OUT/z0_log.txt"
printf '%s\n' "${cells[@]}" | xargs -P "$JOBS" -L 1 bash -c 'cd "$0"/sim/oracle && "$0"/target/release/difftest 20 1 400 "$1" "$2" 0.15 > "'"$OUT"'/${1}_${2}.txt" 2>&1; echo "$1 $2 $(grep "games matched" "'"$OUT"'/${1}_${2}.txt")" | tee -a "'"$OUT"'/z0_log.txt"' "$AZ"
grep -h "games matched" "$OUT"/*_?.txt | awk '{m+=$1; split($1,a,"/"); n+=a[2]} END {print "Z0 total:", m"/"n" games matched"}' | tee -a "$OUT/z0_log.txt"
