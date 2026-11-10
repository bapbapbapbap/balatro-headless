# Run moe (first mixture of experts)

**What it was.** A shared trunk with 8 heads from model 0.2's weights: an early head for antes 1–3, then
7 hand experts (Pair, Two Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind) picked by a
router at the ante-4 handoff. Pins (forcing each expert's hand) fading over 100,000 games. Encoding 3,
`ordered-capped`, reward `blinds` with the corrected win test. Design: section I
([2026-11-03 ~17:00](../hypotheses-moe.md#added-2026-11-03-1700-cst-i-strategy-specialists-design-registered-before-any-specialist-exists); changes before launch in [2026-11-03 ~20:50](../hypotheses-general.md#added-2026-11-03-2050-cst-correction-final-boss-losses-were-counted-as-wins)).

**How it went.** Started 2026-11-03 21:25 (commit `17c1602`): [2026-11-03 ~21:35](../hypotheses-moe.md#added-2026-11-03-2135-cst-the-moe-run-started-section-i). Stopped itself
2026-11-04 02:54 on the stall rule, 220 rounds, 24,640 games, with the pins still at 0.75:
[stop and changes for moe2](../hypotheses-moe.md#added-2026-11-04-1340-cst-the-first-moe-run-stopped-changes-for-the-second-section-i). No prediction was registered.

**Left behind:** nothing released and no published routes; it led to [moe2](moe2.md).
