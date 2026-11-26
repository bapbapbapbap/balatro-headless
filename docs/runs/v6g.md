# v6g: model v6, stage 3 at Gold

**What it is.** PPO on our reward alone (blinds, v4b's shaping and credit) from v6dt's best.pt, fresh optimizer,
round 0; Red deck, Gold stake (8); no teacher. The first 10 rounds train only the value heads (`--critic-rounds 10`:
the policy stays exactly v6dt's, checked after every such update); then normal PPO at learning rate 5e-5. Agreement
with the teacher is scored at every eval on v6dt's agreement set without imitation (`--agreement-from`). Stops on
v4b's stall rule (25 evals with no new best) or the 168-hour cap. Registered with predictions G1-G7 in
[docs/hypotheses-v6.md](../hypotheses-v6.md) (2026-11-22, after chud's 19:36 CST message). Settings: `configs/v6g.toml`.

**Status:** registered and built 2026-11-22; not launched. Smoke in the cloud (2 workers, 3 rounds, no agreement):
critic rounds 0-1 moved the policy 0.0000 and eval 2 matched eval 0 exactly; round 2's PPO update moved it 0.025.

Launch (smoke first):

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout v6 && git pull && bash train/launch.sh v6g --work /mnt/c/Users/Jacobi/balatro-work --smoke"
    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/launch.sh v6g --work /mnt/c/Users/Jacobi/balatro-work"

Output: `balatro-work/runs/v6g/`.
