# v6dt: model v6, DAgger's attribution twin

**What it is.** DAgger round 1's training without the DAgger games: from v6e's best.pt, imitation on every teacher game in `teach_v6`, up to 8 passes, stage 1's agreement set; then the 600-game eval and a temperature-0 eval. A round's gain over this twin is DAgger's, not the extra passes' or run 4's data's. Registered in [docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-21, after chud's 23:42 CST message).

**Status:** built 2026-11-21, not run. Started by `train/step3.sh` after DAgger round 1:

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/step3.sh"

Output: `balatro-work/runs/v6dt/`; temperature 0: `balatro-work/dagger/twin_t0.jsonl`; log `balatro-work/step3.txt`.
