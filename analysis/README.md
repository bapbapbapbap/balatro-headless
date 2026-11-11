# Won-game analysis

`replay_events.py` replays saved games (`wins.jsonl`, `games/*.jsonl.gz`) move by move on the
action list each was played on, checks every move against the saved text, and logs what a player
would note at each step: hand type and the best hand on offer for plays, planets offered and taken,
jokers held, boss blind, money in and out of the shop. `analyze.py` turns that into tests, each
against a baseline from the bot's own data or from the options it had.

```bash
python3 replay_events.py --src ~/balatro/local/source --out wins_events.jsonl --jobs 2 --skip-eval runs/v2/wins.jsonl
python3 replay_events.py --src ~/balatro/local/source --out late_events.jsonl --jobs 2 runs/v2/games/r0049*_train.jsonl.gz
python3 analyze.py --wins wins_events.jsonl --late late_events.jsonl
```
`--shard I/N` splits a run across machines. Results for v2's 709 wins: `docs/won_games_analysis.md`.

## Re-solve: is a win the seed or the sampled moves?
`resolve.py` plays released weights K times on every published won seed and on as many fresh
control seeds, and reports win rate per seed against the control (see the docstring).
```bash
python3 resolve.py play ../weights/balatro-bot-1.0.pt --src ~/balatro/local/source --k 10 --jobs 12 \
    --out resolve_1.0.jsonl --routes-out resolve_1.0_wins.jsonl      # rerun to resume
python3 resolve.py summary resolve_1.0.jsonl --k 10 --ranked resolve_1.0_ranked.csv
```

### Result, model 1.0 (2026-11-03)
`results/resolve_1.0/`: `summary.txt`, `ranked.csv` (all 3,098 seeds, easiest first) and every game
(`resolve_1.0.jsonl.gz`). Scored in `docs/hypotheses.md`, "R result".

## Paired experts: what the mixture of experts learned
`paired_experts.py` plays every expert from the same ante-4 handoff state on seeds outside the eval
600, plus a control where the early head plays on with no handoff. It answers three questions (chud,
2026-11-05): does one strategy win the most; does the handoff change play (plays by hand after the
handoff, per expert, from the same positions); is the handoff at the right ante (`--handoff-antes`).
It also scores the router's own choice and the ceiling a perfect router would reach. The weights must
be a trainer checkpoint (it carries the run's settings).
```bash
python3 paired_experts.py /mnt/c/Users/<you>/balatro-work/runs/moe2/pins0.pt --src ~/balatro/local/source \
    --seeds 200 --handoff-antes 3,4,5 --workers 4 --out paired_moe2.jsonl     # rerun to resume
python3 paired_experts.py --summary paired_moe2.jsonl
python3 score_paired.py paired_moe2.jsonl      # C1-C5, the three questions, handoff ante over all seeds
```
`score_paired.py` scores the output against the registered predictions and compares handoff antes as
a policy over every seed (a seed that never reaches the handoff keeps the early head's game). The
summary's per-ante rows are on different seeds and can't be compared with each other; restricting to
the seeds common to all antes is also biased (see the script's docstring). First run: 2,000 seeds on
moe2's `pins0.pt`, results in `docs/moe2_evaluation.md`.
Cost: per seed and handoff ante, 7 games per rollout (6 experts + the control). Run it when training
isn't using the CPU, or with few workers.

## v4b's stop: what the shop network's selling cost, and why its credit allowed it
Three scripts, run after v4b stopped itself at round 280 (2026-11-10). `v4b_drop.py` replays saved eval games
with no model and records what was held, bought, sold and scored at every blind. `v4b_shop_credit.py` replays
training games under a checkpoint and rebuilds the shop network's credit at every decision where a sell was
legal (as `v4_sells.py` does for the table network). `v4b_sell_followup.py` reads their output and prints the
tables the diagnosis cites; it needs neither the game nor torch.
```bash
python3 v4b_drop.py --src ~/balatro/local/source --games RUNS/v4b/games --rounds 240,260,280 --out v4b_drop.jsonl --workers 12
python3 v4b_shop_credit.py --src ~/balatro/local/source --run v4b --its 200,220 --window 5 --out v4b_shop_credit_c.jsonl
python3 v4b_sell_followup.py --drop v4b_drop.jsonl --credit v4b_shop_credit_c.jsonl v4b_shop_credit_b.jsonl
```
Results: `results/v4b_stop/` (every output, and its own README). Read in `docs/hypotheses-v4b.md`, 2026-11-11.
