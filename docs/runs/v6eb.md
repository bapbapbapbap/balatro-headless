# v6eb: model v6, step 2

**What it is.** The bigger-network arm: v6e with d 256, 6 layers, 8 heads; then the 600-game eval (temperature 1) and a temperature-0 eval by `tools/eval_ladder.py`. Registered in [docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-21, after chud's
09:38 CST message), with its predictions and reading.

**Status:** ran 2026-11-21.

**Result (2026-11-21):** kept pass 16 of 16 (held-out loss still falling); 125/600 at temperature 1, 245/600 (40.8%) at temperature 0; agreement 74.7%. B1 miss. Scored in docs/hypotheses-v6.md (2026-11-21, after chud's 16:09 CST message).

Started by `train/step2.sh` with the other step 2 jobs:

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step2.sh"

Output: `balatro-work/runs/v6eb/` (`train.log`, `metrics.jsonl`: the `imitate` lines per pass, then the eval
with agreement); temperature 0: `balatro-work/step2/v6eb_t0.jsonl`; log `balatro-work/step2.txt`.
