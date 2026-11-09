#!/usr/bin/env bash
# Seed check for Ghost Deck ("Hex deck"), Purple Stake, antes 1-3.
# Uses Immolate (https://github.com/SpectralPack/Immolate, CC BY-NC-SA 4.0), an
# independent OpenCL model of Balatro's RNG. No game code is involved.
#
# Usage: tools/immolate/run_seed_check.sh SEED [SEED ...]
# Linux needs an OpenCL driver; on a CPU-only box: apt-get install pocl-opencl-icd ocl-icd-opencl-dev cmake
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${IMMOLATE_DIR:-$here/.immolate}"
if [ ! -d "$work" ]; then
  git clone --depth 1 https://github.com/SpectralPack/Immolate "$work"
  # pocl (CPU OpenCL) rejects the address-space cast that only NVIDIA's guard covers.
  sed -i 's/#ifdef __NV_CL_C_VERSION/#if 1/' "$work/lib/functions.cl"
  cmake -S "$work" -B "$work/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$work/build" -j >/dev/null
fi
cp "$here/ghost_purple_soulcheck.cl" "$work/filters/"
cd "$work"
for s in "$@"; do
  echo "######## $s"
  ./build/Immolate -f ghost_purple_soulcheck -s "$s" -n 1 -g 1 2>&1 \
    | grep -v -E "Warning|Immolate Beta|Building|Starting|Done"
done
