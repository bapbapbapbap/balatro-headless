# v6g best.pt (round 660): win rates

Trained on the Red Deck only: imitation + RL at White stake (v6dt), then RL at Gold stake (v6g, warm-started from v6dt). Every other deck is zero-shot. One game per seed, the fixed eval seed list; t0 = temperature 0 (always the most likely move).

## Red Deck, White Stake (training deck)

- v6g best.pt, t0: 174/600 = 29% (95% 26-33%)
- v6dt best.pt (before Gold training), t0: 273/600 = 45.5% (95% 41.6-49.5%)

## Gold Stake, all 15 decks (t0 unless marked)

| Deck | Win rate, first 600 seeds | Win rate, all t0 games | t1 games | Reached ante 8 (t0) | Mean ante (t0) |
|---|---|---|---|---|---|
| Green | 0.333% (2/600; 0.091-1.2) | 0.322% (29/9,000; 0.22-0.46) |  | 1.1% | 4.36 |
| Blue | 0.167% (1/600; 0.029-0.94) | 0.189% (17/9,000; 0.12-0.3) | 0.167% (5/3,000; 0.071-0.39) | 1.1% | 4.34 |
| Yellow | 0% (0/600; 0-0.64) | 0.167% (15/9,000; 0.1-0.27) | 0.0333% (1/3,000; 0.0059-0.19) | 0.79% | 4.30 |
| Checkered | 0% (0/600; 0-0.64) | 0.0333% (3/9,000; 0.011-0.098) | 0% (0/3,000; 0-0.13) | 0.3% | 4.11 |
| Ghost | 0% (0/600; 0-0.64) | 0.111% (10/9,000; 0.06-0.2) | 0.0333% (1/3,000; 0.0059-0.19) | 0.62% | 4.07 |
| Painted | 0% (0/600; 0-0.64) | 0.0111% (1/9,000; 0.002-0.063) |  | 0.11% | 4.03 |
| Plasma | 0% (0/600; 0-0.64) | 0.0111% (1/9,000; 0.002-0.063) |  | 0.078% | 3.97 |
| Erratic | 0% (0/600; 0-0.64) | 0.0556% (5/9,000; 0.024-0.13) |  | 0.29% | 3.86 |
| Red | 0% (0/600; 0-0.64) | 0.0222% (2/9,000; 0.0061-0.081) |  | 0.33% | 3.90 |
| Abandoned | 0.167% (1/600; 0.029-0.94) | 0.0333% (3/9,000; 0.011-0.098) |  | 0.26% | 3.80 |
| Zodiac | 0% (0/600; 0-0.64) | 0.0667% (6/9,000; 0.031-0.15) |  | 0.37% | 3.76 |
| Magic | 0% (0/600; 0-0.64) | 0.0556% (5/9,000; 0.024-0.13) |  | 0.36% | 3.83 |
| Anaglyph | 0% (0/600; 0-0.64) | 0.0222% (2/9,000; 0.0061-0.081) |  | 0.23% | 3.66 |
| Nebula | 0% (0/600; 0-0.64) | 0.0556% (5/9,000; 0.024-0.13) |  | 0.28% | 3.66 |
| Black | 0% (0/600; 0-0.64) | 0.0111% (2/18,000; 0.003-0.041) |  | 0.083% | 2.87 |

First 600 seeds, all 15 decks pooled: 4/9,000 = 0.0444% (95% 0.017-0.11%).
Rows sorted by mean ante over the first 600 seeds. 'All t0 games' stopped at a deck's first win (crack_t0.sh), so those rates lean high. Red's Gold win (WCRKZKR1) came from the training run's own eval at temperature 1, not from these t0 files.
