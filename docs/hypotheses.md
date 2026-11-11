# Hypotheses (pre-registered)

Written 2026-11-01 ~20:25 CST, while the v1 overnight run was in progress and before its results
were known. The git commit that adds this file is the registration timestamp. Changes after this
point get a new dated section; nothing above it is edited.

Status of every item: **proposed**. Thresholds marked *(chud to confirm)* are drafts until he
confirms or changes them.

## Where the sections are

On 2026-11-09 the sections were moved, verbatim and in order, into one file per training run, plus one for
what belongs to no single run. Nothing in them was edited. The file as it stood before the move, with every
section in one place and its git history, is at [commit e3778d0](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).

New sections go at the end of the file they concern, dated, with a row added here. Where each hypothesis ended
(scored, void, closed, still open) is in [`closed_hypotheses.md`](closed_hypotheses.md); one page per run is in
[`runs/`](runs/README.md).

| File | Contents |
|---|---|
| [`hypotheses-general.md`](hypotheses-general.md) | General: aims, the main experiment, corrections, standing decisions |
| [`hypotheses-v1.md`](hypotheses-v1.md) | Run v1 (model 0.1) |
| [`hypotheses-v2.md`](hypotheses-v2.md) | Run v2 (model 0.2) |
| [`hypotheses-v2b.md`](hypotheses-v2b.md) | Run v2b (models 0.3 and 1.0) |
| [`hypotheses-moe.md`](hypotheses-moe.md) | Run moe (first mixture of experts) |
| [`hypotheses-moe2.md`](hypotheses-moe2.md) | Run moe2 (moe2-pins0) |
| [`hypotheses-v3.md`](hypotheses-v3.md) | Model v3 |
| [`hypotheses-v4.md`](hypotheses-v4.md) | Model v4 |
| [`hypotheses-v4b.md`](hypotheses-v4b.md) | Model v4b (current) |

## Every section, in the order it was added

| Section | File |
|---|---|
| [A. The v1 overnight run (Red Deck, White Stake, reward `blinds` + shaping, unmodded 1.0.1o)](hypotheses-v1.md#a-the-v1-overnight-run-red-deck-white-stake-reward-blinds--shaping-unmodded-101o) | v1 |
| [B. Shaping is speed, not values](hypotheses-general.md#b-shaping-is-speed-not-values) | general |
| [C. Isolating oughts (the main experiment)](hypotheses-general.md#c-isolating-oughts-the-main-experiment) | general |
| [D. Before the main training (what still has to be done)](hypotheses-general.md#d-before-the-main-training-what-still-has-to-be-done) | general |
| [Added 2026-11-01 ~20:30 CST](hypotheses-v1.md#added-2026-11-01-2030-cst) | v1 |
| [Added 2026-11-01 ~22:35 CST](hypotheses-v1.md#added-2026-11-01-2235-cst) | v1 |
| [Added 2026-11-02 ~10:35 CST: v2 vs v1 (registered before v2's RL phase has produced any result)](hypotheses-v2.md#added-2026-11-02-1035-cst-v2-vs-v1-registered-before-v2s-rl-phase-has-produced-any-result) | v2 |
| [Added 2026-11-02 ~11:05 CST: authorship, and v3 (registered before v3 exists)](hypotheses-v2.md#added-2026-11-02-1105-cst-authorship-and-v3-registered-before-v3-exists) | v2 |
| [Added 2026-11-02 ~18:25 CST: F1 result (v2 vs v1)](hypotheses-v2.md#added-2026-11-02-1825-cst-f1-result-v2-vs-v1) | v2 |
| [Added 2026-11-02 ~20:55 CST: H. v2b vs v2, solved rate (registered before the window opens)](hypotheses-v2b.md#added-2026-11-02-2055-cst-h-v2b-vs-v2-solved-rate-registered-before-the-window-opens) | v2b |
| [Added 2026-11-02 ~23:05 CST: the project's aims (stated by chud)](hypotheses-general.md#added-2026-11-02-2305-cst-the-projects-aims-stated-by-chud) | general |
| [Added 2026-11-03 ~12:45 CST: H result (v2b vs v2, solved rate)](hypotheses-v2b.md#added-2026-11-03-1245-cst-h-result-v2b-vs-v2-solved-rate) | v2b |
| [Added 2026-11-03 ~16:15 CST: R. Re-solve: model 1.0 on the published won seeds (registered after the run started, before any of its output was read)](hypotheses-v2b.md#added-2026-11-03-1615-cst-r-re-solve-model-10-on-the-published-won-seeds-registered-after-the-run-started-before-any-of-its-output-was-read) | v2b |
| [Added 2026-11-03 ~17:00 CST: I. Strategy specialists (design registered before any specialist exists)](hypotheses-moe.md#added-2026-11-03-1700-cst-i-strategy-specialists-design-registered-before-any-specialist-exists) | moe |
| [Added 2026-11-03 ~18:00 CST: roadmap, and what 100% completion means (stated by chud)](hypotheses-general.md#added-2026-11-03-1800-cst-roadmap-and-what-100-completion-means-stated-by-chud) | general |
| [Added 2026-11-03 ~20:15 CST: R result (re-solve by 1.0)](hypotheses-v2b.md#added-2026-11-03-2015-cst-r-result-re-solve-by-10) | v2b |
| [Added 2026-11-03 ~20:50 CST: correction: final-boss losses were counted as wins](hypotheses-general.md#added-2026-11-03-2050-cst-correction-final-boss-losses-were-counted-as-wins) | general |
| [Added 2026-11-03 ~21:35 CST: the MoE run started (section I)](hypotheses-moe.md#added-2026-11-03-2135-cst-the-moe-run-started-section-i) | moe |
| [Added 2026-11-04 ~13:40 CST: the first MoE run stopped; changes for the second (section I)](hypotheses-moe.md#added-2026-11-04-1340-cst-the-first-moe-run-stopped-changes-for-the-second-section-i) | moe |
| [Added 2026-11-04 ~13:35 CST: I2, Full House is one of the strongest experts (section I)](hypotheses-moe2.md#added-2026-11-04-1335-cst-i2-full-house-is-one-of-the-strongest-experts-section-i) | moe2 |
| [Added 2026-11-04 ~13:50 CST: review of the early Full House pin (for a critical reader)](hypotheses-moe2.md#added-2026-11-04-1350-cst-review-of-the-early-full-house-pin-for-a-critical-reader) | moe2 |
| [Added 2026-11-04 ~13:40 CST: completion criterion: the MoE has learned Full House](hypotheses-moe2.md#added-2026-11-04-1340-cst-completion-criterion-the-moe-has-learned-full-house) | moe2 |
| [Added 2026-11-05 ~10:40 CST: Full House not learned so far (moe2, through round 900)](hypotheses-moe2.md#added-2026-11-05-1040-cst-full-house-not-learned-so-far-moe2-through-round-900) | moe2 |
| [Added 2026-11-05 ~11:00 CST: J, the next mixture of experts (design draft; nothing registered)](hypotheses-moe2.md#added-2026-11-05-1100-cst-j-the-next-mixture-of-experts-design-draft-nothing-registered) | moe2 |
| [Added 2026-11-05 ~15:40 CST: moe2 at pins zero: results](hypotheses-moe2.md#added-2026-11-05-1540-cst-moe2-at-pins-zero-results) | moe2 |
| [Added 2026-11-05 ~15:45 CST: Claude's predictions for the first paired run (section J)](hypotheses-moe2.md#added-2026-11-05-1545-cst-claudes-predictions-for-the-first-paired-run-section-j) | moe2 |
| [Added 2026-11-05 ~15:45 CST: the first paired run's size (2,000 seeds), and its pilot](hypotheses-moe2.md#added-2026-11-05-1545-cst-the-first-paired-runs-size-2000-seeds-and-its-pilot) | moe2 |
| [Added 2026-11-05 ~21:10 CST: the first paired run, scored (section J)](hypotheses-moe2.md#added-2026-11-05-2110-cst-the-first-paired-run-scored-section-j) | moe2 |
| [Added 2026-11-05 ~21:15 CST: what moe2 says about aim 1 (chud's reading, with limits)](hypotheses-moe2.md#added-2026-11-05-2115-cst-what-moe2-says-about-aim-1-chuds-reading-with-limits) | moe2 |
| [Added 2026-11-05 ~21:19 CST: model v3, registered before any of its code exists (section J)](hypotheses-v3.md#added-2026-11-05-2119-cst-model-v3-registered-before-any-of-its-code-exists-section-j) | v3 |
| [Added 2026-11-05 ~23:01 CST: model v4, registered before any of its code (section J)](hypotheses-v4.md#added-2026-11-05-2301-cst-model-v4-registered-before-any-of-its-code-section-j) | v4 |
| [Added 2026-11-09 ~12:00 CST: v4 amended before any of its code (section J)](hypotheses-v4.md#added-2026-11-09-1200-cst-v4-amended-before-any-of-its-code-section-j) | v4 |
| [Added 2026-11-09 ~15:25 CST: model v4's definitions, fixed before any v4 training (section J)](hypotheses-v4.md#added-2026-11-09-1525-cst-model-v4s-definitions-fixed-before-any-v4-training-section-j) | v4 |
| [Added 2026-11-09 ~15:48 CST: W4b's compute target fixed, and Claude's odds after the measured costs (section J)](hypotheses-v4.md#added-2026-11-09-1548-cst-w4bs-compute-target-fixed-and-claudes-odds-after-the-measured-costs-section-j) | v4 |
| [Added 2026-11-09 ~18:17 CST: the Full House venture closed, and the bar for going up the decks and stakes](hypotheses-general.md#added-2026-11-09-1817-cst-the-full-house-venture-closed-and-the-bar-for-going-up-the-decks-and-stakes) | general |
| [Added 2026-11-09 ~18:30 CST: hypotheses that will never be scored as registered are closed](hypotheses-general.md#added-2026-11-09-1830-cst-hypotheses-that-will-never-be-scored-as-registered-are-closed) | general |
| [Added 2026-11-09 ~18:40 CST: the last three open settings, fixed](hypotheses-general.md#added-2026-11-09-1840-cst-the-last-three-open-settings-fixed) | general |
| [Added 2026-11-10 ~10:30 CST: why v4 stalled (diagnosis after the run, section J)](hypotheses-v4.md#added-2026-11-10-1030-cst-why-v4-stalled-diagnosis-after-the-run-section-j) | v4 |
| [Added 2026-11-10 ~10:30 CST: model v4b, registered before any of its code (section J)](hypotheses-v4b.md#added-2026-11-10-1030-cst-model-v4b-registered-before-any-of-its-code-section-j) | v4b |
| [Added 2026-11-11 ~09:25 CST: v4b stopped at round 280 on the sell stop; S1-S3 scored](hypotheses-v4b.md#added-2026-11-11-0925-cst-v4b-stopped-at-round-280-on-the-sell-stop-s1-s3-scored) | v4b |
| [Added 2026-11-11 ~09:25 CST: why v4b's shop network sold, and what it cost (diagnosis after the stop)](hypotheses-v4b.md#added-2026-11-11-0925-cst-why-v4bs-shop-network-sold-and-what-it-cost-diagnosis-after-the-stop) | v4b |
| [Added 2026-11-11 ~09:25 CST: v4b continues from round 280 with the sell stop off, registered before its config changes](hypotheses-v4b.md#added-2026-11-11-0925-cst-v4b-continues-from-round-280-with-the-sell-stop-off-registered-before-its-config-changes) | v4b |
