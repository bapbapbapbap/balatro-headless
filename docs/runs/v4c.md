# Run v4c (built; not launched)

**What it is.** v4b, warm-started from v4b's `best.pt` (13.40 blinds/game, round 1840), with one kind of change:
clearer information about how the shop helps it win. It adds a `swap` move (sell a held joker and buy a shop
joker in one move, offered when the slots are full; action list `ordered-capped-swap`, swaps listed last). It
also adds offer values, in phi_build, for swaps, rerolls, and Buffoon and Celestial packs. Rerolls and packs are
valued as an expectation over the game's public pools, never over the seed's hidden next shop. The reward, the
shaping, the credit and the sampling are v4b's. No moves are forced. The shop entropy is pinned at v4b's floor,
0.001.

**Registered:** [design, gate and predictions P1 / P1b](../hypotheses-v4c.md#added-2026-11-13-1845-cst-model-v4c-registered-before-any-of-its-code),
after v4b's [economy test](../hypotheses-v4b.md#added-2026-11-13-1820-cst-why-v4b-hoards-the-tests-result) and the
walls report (PR #122); discussion in issue #124. Code on branch `v4c`; output goes to `runs/v4c/`.

**Before launch:** the pre-flight check `v4c-swap` (`tests/v4c_swap_check.py`), then the smoke run. Per the
registration's gate, if the smoke's sim time per decision is above 1.6x v4b's, `pool_samples` goes from 8 to 4
before launch, and that cut is recorded. v4b stays the current run until v4c launches.
