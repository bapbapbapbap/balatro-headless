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
