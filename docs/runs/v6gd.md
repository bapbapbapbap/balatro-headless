# v6gd: v6g with the deck mixed

**What it is.** v6g's run with one change: every game's deck is drawn uniformly from all 15 decks (`--backs`), fixed
for the whole game and taken from the game's seed (`common.game_config`), so retries and eval seeds always get the
same deck. Everything else is v6g's: PPO on our reward alone from v6dt's best.pt (fresh optimizer, round 0), Gold
stake, 10 critic-only rounds, learning rate 5e-5, agreement scored at every eval, v4b's stall rule, 168-hour cap,
encoding 2 (no deck token). The question is chud's (a comparison to training across decks instead of specializing);
the design is Claude's. Registered with predictions D1-D4 in [docs/hypotheses-v6.md](../hypotheses-v6.md)
(2026-11-25, after chud's 23:23 CST message). Settings: `configs/v6gd.toml`.

**Equal games per deck, not equal time:** at r660 a Gold game takes 2.6 s (Black) to 6.7 s (Painted). Balancing
by time would give a deck fewer games as the policy survives longer on it, so it isn't done.

**Reading it while it runs:** the in-run eval is also mixed (about 40 of the 600 eval games per deck), so its
blinds per game are not comparable to v6g's Red-only evals. The start banner prints `b_red`. It is `--back`'s
default and is unused once `--backs` is set; each game's deck is in its record (`back`). The comparison with v6g
is scored afterwards by `tools/eval_ladder.py` on matched checkpoints, 15 decks × 600 seeds each.

**Status:** registered; the first smoke (2026-11-25 23:30) stopped in pre-flight at `v4-phi` (a phi_build cache gap v6g also trained with; 2026-11-26 10:55 section). That check is dropped for v6gd; not started.

Launch (smoke first; `--backs` has not been used in the v6 lineage):

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout main && git pull && bash train/launch.sh v6gd --work /mnt/c/Users/Jacobi/balatro-work --smoke"
    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/launch.sh v6gd --work /mnt/c/Users/Jacobi/balatro-work"

Output: `balatro-work/runs/v6gd/`.
