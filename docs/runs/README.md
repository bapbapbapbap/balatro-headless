# Training runs

One page per run: what it was, what was registered and scored for it, and what it left behind. The
pages only point into the record; the record itself stays where it is (`docs/hypotheses.md`, the
CHANGELOG, `wins/`), unedited.

| Run | Model | Dates (CST) | What changed | Status | Published wins / near misses |
|---|---|---|---|---|---|
| [v1](v1.md) (+ A/B arms) | 0.1 | 2026-11-01 → 11-02 | first PPO bot | retired | 24 / 57 |
| [v2](v2.md) | 0.2 | 2026-11-02 | cloned from v1, bigger, ordered actions | retired | 179 / 369 |
| [v2b](v2b.md) | 0.3, 0.3-beta, 1.0 | 2026-11-02 → 11-03 | joker-drag cap, hand-type exploration | ended; 1.0 released | 232 / 362 |
| [moe](moe.md) | – | 2026-11-03 → 11-04 | mixture of experts, 7 hand experts, pinned | stopped on the stall rule | 0 / 0 |
| [moe2](moe2.md) | moe2-pins0 | 2026-11-04 → 11-05 | 6 experts, early Full House pin, pins faded to 0 | stopped at pins zero; released | 790 / 712 |
| [v3](v3.md) | – | 2026-11-05 | discount per blind passed | code built; replaced by v4 | – |
| [v4](v4.md) | – | 2026-11-09 → 11-10 | shop and table networks, build potential, curriculum | stopped on the stall rule (best eval 4.68, round 120) | not yet published |
| [**v4b**](v4b.md) | – | 2026-11-10 → | v4 + hand clock on the table's lambda, hand exploration off | **registered** | – |

Counts are of `wins/routes.jsonl` and `wins/near_misses.jsonl` on 2026-11-09, by each line's `source`
(the run folder the win came from), with the corrected win test (2026-11-03). Every run so far is Red Deck,
White Stake, unmodded 1.0.1o.

## Getting one run's routes

`wins/routes.jsonl` stays one file: the trainer, the checks and the tally read it, and v4's curriculum
picks its starts from it. To take out one run's routes (here v2b's):

```
python3 -c "import json,sys; [print(l,end='') for l in open('wins/routes.jsonl') if json.loads(l)['source'].split('/')[0]==sys.argv[1]]" v2b > v2b_routes.jsonl
```

The same works on `wins/near_misses.jsonl`. Run folders in `source`: `v1`, `low`, `high`, `off`, `on`
(v1's A/B arms), `v2`, `v2b`, `moe2`.

## Names that collide
- **"v3"** is two different things. Section G (2026-11-02 ~11:05) called a `win`-reward arm "v3"; no result
  is recorded for it. Model v3 (2026-11-05 ~21:19) is the blind-clock design on [its page](v3.md).
- **"C1"** is section C's first prediction (different oughts, different behavior, 2026-11-01) and also the
  first prediction of the 2026-11-05 paired run. They are unrelated.
