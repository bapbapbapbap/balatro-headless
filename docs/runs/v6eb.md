# v6eb: model v6, step 2

**What it is.** The bigger-network arm: v6e with d 256, 6 layers, 8 heads; then the 600-game eval (temperature 1) and a temperature-0 eval by `tools/eval_ladder.py`. Registered in [docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-21, after chud's
09:38 CST message), with its predictions and reading.

**Status:** built 2026-11-21, not run. Started by `train/step2.sh` with the other step 2 jobs:

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step2.sh"

Output: `balatro-work/runs/v6eb/` (`train.log`, `metrics.jsonl`: the `imitate` lines per pass, then the eval
with agreement); temperature 0: `balatro-work/step2/v6eb_t0.jsonl`; log `balatro-work/step2.txt`.
