# Seed checks: Android build, Ghost Deck ("Hex deck"), Purple Stake

Added 2026-11-03. These are seeded runs on the **Android** game, the build released on
2024-09-26. The harness in this repo can't check them: it covers only the Windows Steam
1.0.1o-FULL build, and it doesn't cover seeded runs yet. These checks use
[Immolate](https://github.com/SpectralPack/Immolate) instead, an independent OpenCL model of
the game's RNG. It is labelled 1.0.1f. Rerun them with `tools/immolate/run_seed_check.sh SEED`.

## The line being checked
On Ghost Deck you start holding **Hex**. If skipping the Small Blind in Ante 1 gives a Charm
Tag, and that Mega Arcana contains **The Soul**, it can give Perkeo. Hex then makes Perkeo
Polychrome. It also destroys any other jokers, and with Perkeo as the only joker that costs
nothing.

## Y2KWQ2B9: the line works. This is the control, checked against real screenshots.
| | Model | Android screenshots |
|---|---|---|
| Ante 1 boss | The Head | The Head |
| Ante 1 tags (Small, Big) | Charm Tag, Coupon Tag | Small skipped, Arcana pack opened |
| Charm Tag Mega Arcana | Emperor, **Soul → Perkeo**, Moon, Magician, High Priestess | Emperor, Soul, Moon, Magician, High Priestess (choose 2) |
| Result | Perkeo, made Polychrome by Hex | Polychrome Perkeo. Won Ante 8 (Round 23), lost in Endless at Ante 14 |

Every Ante 1 value checked matches, so the model agrees with the Android build on these streams.

## X4Z8KB9W: the line does not work
- **Ante 1:** the boss is The Window and the voucher is Director's Cut. The tags are
  **Economy Tag** (Small) and **Polychrome Tag** (Big). Skipping the Small Blind gives Economy,
  not Charm, so no Mega Arcana opens.
- **No Soul in Antes 1–3.** The first four 5-card Arcana draws in each ante, which is what
  any Charm Tag or Arcana pack would show, have no Soul. There's none in the first 60 shop items
  of each ante either.
- The shop packs in Ante 1 are Buffoon, Jumbo Buffoon, Arcana (Death, High Priestess, Emperor),
  Jumbo Celestial, Jumbo Standard and Jumbo Arcana.
- Ante 2: The Manacle, with Double Tag and Holographic Tag. Ante 3: The Arm, with Handy Tag and
  Standard Tag.
- The only thing here like the Y2KWQ2B9 line: skipping the **Big** Blind gives the Polychrome
  Tag. That makes the next shop joker Polychrome, and the next one in the queue is Smiley Face.

## Limits
- Immolate models 1.0.1f. The Android build isn't independently identified. It agrees with the
  model on the Y2KWQ2B9 control, which supports the X4Z8KB9W result but doesn't prove it.
- The searches go to Ante 3 only, and look at the first 4 Arcana draws and 60 shop items per ante.
- Anything decided by in-run choices isn't modelled: which vouchers you buy, rerolls, and the
  duplicate rerolls from jokers you already own.
