# v6e: model v6, step 2

**What it is.** Stage 1 from random weights on all 3,599 teacher games, up to 16 passes (patience 2), stage 1's agreement set; then the 600-game eval (temperature 1) and a temperature-0 eval by `tools/eval_ladder.py`. Registered in [docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-21, after chud's
09:38 CST message), with its predictions and reading.

**Status:** ran 2026-11-21.

**Result (2026-11-21):** kept pass 15 of 16; 132/600 at temperature 1, **254/600 (42.3%) at temperature 0**; agreement 74.1% (hand 54.3%, shop 81.4%). Scored in docs/hypotheses-v6.md (2026-11-21, after chud's 16:09 CST message).

It was started by Started by `train/step2.sh` with the other step 2 jobs:

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step2.sh"

Output: `balatro-work/runs/v6e/` (`train.log`, `metrics.jsonl`: the `imitate` lines per pass, then the eval
with agreement); temperature 0: `balatro-work/step2/v6e_t0.jsonl`; log `balatro-work/step2.txt`.
