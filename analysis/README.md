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
