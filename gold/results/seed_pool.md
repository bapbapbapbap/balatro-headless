# Seed pool: best.pt (v6g r660) Gold wins across decks, t0

15 decks, 144,000 games, 106 wins; common seeds played by every deck: 9,000.

## Do wins share seeds?

- Seeds won by 2+ decks: **15** observed, 0.521 expected if decks won independently.
- Shared-win deck pairs: **55** observed, 0.524 expected (E1).

| Seed | Decks that won it |
|---|---|
| HTBDF155 | red, blue, yellow, nebula, checkered, painted, erratic |
| B88LZ88C | blue, magic, nebula, ghost, anaglyph |
| 9QGDSETV | magic, abandoned, checkered, erratic |
| KCXBL885 | blue, nebula, erratic |
| RJJ79JH4 | blue, green, plasma |
| UV1H8SSM | red, blue, ghost |
| 2XYRSTYJ | yellow, magic |
| 7TSDFK5F | ghost, zodiac |
| 8NBTDYAT | blue, zodiac |
| 93PR5ZVU | yellow, green |
| A2IWFRKW | blue, yellow |
| BMFS3HJP | yellow, green |
| BV43ZAW1 | yellow, zodiac |
| T4IKVQ7T | nebula, anaglyph |
| VSM7CDD6 | blue, checkered |

By deck family (altered = Abandoned, Checkered, Erratic: starting cards differ from the standard 52):

| Pairs | Observed | Expected (independent) | Enrichment O/E1 |
|---|---|---|---|
| standard-standard | 35 | 0.405 | 86x |
| standard-altered | 16 | 0.115 | 139x |
| altered-altered | 4 | 0.004 | 923x |

## Band width f and the ceiling

**f = 0.95% of seeds (95% 0.6%-2.2%, Poisson bootstrap over seeds, 4,000 draws).** Model: one band shared by all decks, every seed in it equally winnable. f is the rough ceiling for this policy, not the game's solvable fraction (see the docstring).

| Deck | Games | Wins | Win rate (95%) | Win rate inside the band, p / f |
|---|---|---|---|---|
| Green | 9,000 | 29 | 0.322% (0.22-0.46%) | 34% (15%-54%) |
| Blue | 9,000 | 17 | 0.189% (0.12-0.3%) | 20% (8.5%-32%) |
| Yellow | 9,000 | 15 | 0.167% (0.1-0.27%) | 17% (7.5%-28%) |
| Ghost | 9,000 | 10 | 0.111% (0.06-0.2%) | 12% (5%-19%) |
| Zodiac | 9,000 | 6 | 0.0667% (0.031-0.15%) | 7% (3%-11%) |
| Magic | 9,000 | 5 | 0.0556% (0.024-0.13%) | 5.8% (2.5%-9.3%) |
| Nebula | 9,000 | 5 | 0.0556% (0.024-0.13%) | 5.8% (2.5%-9.3%) |
| Erratic | 9,000 | 5 | 0.0556% (0.024-0.13%) | 5.8% (2.5%-9.3%) |
| Abandoned | 9,000 | 3 | 0.0333% (0.011-0.098%) | 3.5% (1.5%-5.6%) |
| Checkered | 9,000 | 3 | 0.0333% (0.011-0.098%) | 3.5% (1.5%-5.6%) |
| Red | 9,000 | 2 | 0.0222% (0.0061-0.081%) | 2.3% (1%-3.7%) |
| Anaglyph | 9,000 | 2 | 0.0222% (0.0061-0.081%) | 2.3% (1%-3.7%) |
| Black | 18,000 | 2 | 0.0111% (0.003-0.041%) | 1.2% (0.5%-1.9%) |
| Painted | 9,000 | 1 | 0.0111% (0.002-0.063%) | 1.2% (0.5%-1.9%) |
| Plasma | 9,000 | 1 | 0.0111% (0.002-0.063%) | 1.2% (0.5%-1.9%) |

The interval on p / f carries f's uncertainty only, not p's.

## Registered prediction (Claude's, 11-24 ~10:37)

The 5 seeds shared before even_9000.sh, on the decks that hadn't won them (63 cells): prediction >= 10 win.

- Played 63/63, won 8: **NOT MET**. New wins: B88LZ88C blue, B88LZ88C ghost, HTBDF155 red, HTBDF155 blue, HTBDF155 yellow, HTBDF155 erratic, 9QGDSETV abandoned, 9QGDSETV erratic.
- Red at t0 on WCRKZKR1 (its win was at t1): lost.
