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

**Status:** stopped. Started 2026-11-26 18:18 (after `v4-phi` was dropped from its pre-flight, 2026-11-26 10:55
section); chud stopped it with Ctrl+C at iteration 766, 16.0 h, before the stall rule fired (in-run best 9.57 at
700). **Result at round 660** (2026-11-27 section): v6g, the Red specialist, beats it on 14 of 15 decks at Gold,
pooled 10.03 vs 9.73 blinds a game (paired z -7.45 over 9,000 games); wins 4 vs 4. D1 and D3 missed, D2 and D4 met.
One run each, so run-to-run variance is unmeasured.

Launch (smoke first; `--backs` has not been used in the v6 lineage):

    wsl bash -lc "cd ~/balatro/balatro-headless && git checkout main && git pull && bash train/launch.sh v6gd --work /mnt/c/Users/Jacobi/balatro-work --smoke"
    wsl bash -lc "cd ~/balatro/balatro-headless && bash train/launch.sh v6gd --work /mnt/c/Users/Jacobi/balatro-work"

Output: `balatro-work/runs/v6gd/`.
