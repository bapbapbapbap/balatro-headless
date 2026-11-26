# v6g: model v6, stage 3 at Gold

**What it is.** PPO on our reward alone (blinds, v4b's shaping and credit) from v6dt's best.pt, fresh optimizer,
round 0; Red deck, Gold stake (8); no teacher. The first 10 rounds train only the value heads (`--critic-rounds 10`:
the policy stays exactly v6dt's, checked after every such update); then normal PPO at learning rate 5e-5. Agreement
with the teacher is scored at every eval on v6dt's agreement set without imitation (`--agreement-from`). Stops on
v4b's stall rule (25 evals with no new best) or the 168-hour cap. Registered with predictions G1-G7 in
[docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-22, after chud's 19:36 CST message). Settings: `configs/v6g.toml`.

**Status:** ran 2026-11-22 → 11-23 (commit 74eeab7, tag `run-v6g`); stopped by chud at round 731 (flat since
round 600). best.pt = round 660 (eval 9.705 at temperature 1), sha256 `4462f80e4b41fb9e...`.

**Result:** at temperature 0, Red Deck, 600 seeds: White 174 (v6dt 273), Gold 0 (mean ante 3.91 vs v6dt's 2.94);
G7 met, G4 and G5 not met, chud's White prediction not met; skipping explains 14% of the White drop
([docs/hypotheses-v6.md](../hypotheses-v6.md), 2026-11-23 sections). Zero-shot at Gold on the other 14 decks,
then 9,000 seeds per deck: wins on all 15 ([`gold/`](../../gold/README.md)); what was learned:
[system card](../system_card_v6g.md). G1-G3 and G6 are still to be scored from the run's logs.

Launch (smoke first):

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout main && git pull && bash train/launch.sh v6g --work /mnt/c/Users/Jacobi/balatro-work --smoke"
    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/launch.sh v6g --work /mnt/c/Users/Jacobi/balatro-work"

Output: `balatro-work/runs/v6g/`.
