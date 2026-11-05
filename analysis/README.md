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
```
Cost: per seed and handoff ante, 7 games per rollout (6 experts + the control). Run it when training
isn't using the CPU, or with few workers.
