# Gold Stake on all 15 decks (v6g best.pt, round 660)

The policy was trained on the Red Deck only. Every other deck is zero-shot. One game per seed on the fixed eval
seed list (`train/common.py`), stake 8, temperature 0 unless marked.

| File | What |
|---|---|
| [`results/gold_rates.md`](results/gold_rates.md) | win rate per deck with Wilson 95% intervals; first 600 seeds and all games; mean ante, ante-8 arrivals |
| [`results/seed_pool.md`](results/seed_pool.md), `results/seed_pool_wins.csv` | which seeds were won by which decks; sharing vs independence; band width f; rate inside the band |
| [`verify_gold.py`](verify_gold.py) | the pre-publish check: weights hash, denominators, the frozen policy re-choosing each win, untouched-game replay, coverage |
| [`gold_table.py`](gold_table.py) | builds `gold_rates.md` from the eval files |
| [`seed_pool.py`](seed_pool.py) | builds `seed_pool.md` (reads files only) |

Data (eval files, routes) lives in the work folder: `gold/v6g_t{0,1}_s8_<deck>.jsonl`, `runs/v6g/<deck>_gold_*_routes.jsonl`.
The scripts read it from `$BALATRO_WORK` (default: the current folder).

```bash
python3 gold/verify_gold.py --workers 12               # needs the game source and the weights
python3 gold/gold_table.py --md gold/results/gold_rates.md
python3 gold/seed_pool.py --md gold/results/seed_pool.md --csv gold/results/seed_pool_wins.csv
```

How the wins were found (scripts in the work folder, not the repo): `crack3.sh` (t1, 3,000 seeds, Yellow/Checkered/Ghost), `crack_t0.sh` (t0, each deck still
at 0 wins taken to 3,000-18,000 seeds), then `even_9000.sh` (every deck to 9,000 seeds at t0, so the rates share a
denominator). The last verification (2026-11-25): 114 routes, 0 mismatches, 15/15 decks.

f is a rough ceiling for this policy (one band of seeds shared by all decks), not the fraction of seeds the game
allows to be won. Only lower bounds on that are provable: a replayed win is a proof.
