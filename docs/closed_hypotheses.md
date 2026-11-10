# Closed hypotheses

Closed on 2026-11-09 ~18:30 CST by chud. The rule (chud, 18:32): "If im not training that model the hanging
hypotheses of that model are unanswered and still closed." A hypothesis about a model that is no longer
trained, or whose design was replaced, is closed unanswered: it will never be scored as registered. Closing edits nothing in
[`hypotheses.md`](hypotheses.md): the registered text, and any result as it was scored, stay there. This
file is the ledger of where each one ended.

**No re-scoring.** The 2026-11-03 correction ([section](hypotheses.md#added-2026-11-03-2050-cst-correction-final-boss-losses-were-counted-as-wins)) registered a re-scoring of F1,
H1 and R with the corrected win test. It is closed, not done. chud: "It tests itself as it trains." Every
run since that correction counts wins with the game's own test (`win_game()`), so the project's numbers
come from runs that are correct as they go. The earlier results stay void as scored. The decisions taken on
them at the time (v1 retired, v2 and then v2b continued) stand as history.

## Scored, and final

| ID | Claim | Registered | Result |
|---|---|---|---|
| C1 (paired run) | the experts are not redundant (≥ 3 of 6 lead their own hand) | [2026-11-05 ~15:45](hypotheses.md#added-2026-11-05-1545-cst-claudes-predictions-for-the-first-paired-run-section-j) | **falsified**, 2 of 6 ([scored](hypotheses.md#added-2026-11-05-2110-cst-the-first-paired-run-scored-section-j)) |
| C2 (paired run) | handing off beats playing on with the early head by ≥ 0.5 blinds | same | **pass**, +1.08 |
| C3 (paired run) | best and worst expert within 1.0 blinds | same | **pass**, 0.24 |
| C4 (paired run) | the router adds nothing over a random expert | same | **pass**, +0.02 |
| C5 (paired run) | Full House not in the top 2 experts by blinds | same | **pass**, 5th of 6 |

## Void as scored, closed without re-scoring

| ID | Claim | Registered | As scored (counted final-boss losses as wins) |
|---|---|---|---|
| F1 | model 0.2 beats more new seeds per hour than 0.1 | [2026-11-02 ~10:35](hypotheses.md#added-2026-11-02-1035-cst-v2-vs-v1-registered-before-v2s-rl-phase-has-produced-any-result) | pass, 90.8 vs 42.3 per hour ([scored](hypotheses.md#added-2026-11-02-1825-cst-f1-result-v2-vs-v1)) |
| H1 (chud's) | model 0.3's solved rate is higher than 0.2's | [2026-11-02 ~20:55](hypotheses.md#added-2026-11-02-2055-cst-h-v2b-vs-v2-solved-rate-registered-before-the-window-opens) | pass, 24.4 vs 17.9 per 1,000 ([scored](hypotheses.md#added-2026-11-03-1245-cst-h-result-v2b-vs-v2-solved-rate)) |
| R0 (chud's) | 1.0 clears proven seeds above its 2.5% | [2026-11-03 ~16:15](hypotheses.md#added-2026-11-03-1615-cst-r-re-solve-model-10-on-the-published-won-seeds-registered-after-the-run-started-before-any-of-its-output-was-read) | pass ([scored](hypotheses.md#added-2026-11-03-2015-cst-r-result-re-solve-by-10)) |
| R1–R8 (Claude's) | control rate, published rate, dispersion, retry vs fresh, by model, memorization, share won, blinds | same | 4 pass, 3 missed, 1 missed |
| 1.0's eval | 5 of 200 wins, 12.79 blinds per game | models.md | not re-measured; stays "as counted before 2026-11-03" |

## Closed unscored

| ID | Claim | Registered | Why closed |
|---|---|---|---|
| A1–A5 | v1 keeps learning; first wins; the eval loop is a free-action cycle; the sim stays exact; the run ends on a backstop | [section A](hypotheses.md#a-the-v1-overnight-run-red-deck-white-stake-reward-blinds--shaping-unmodded-101o), [2026-11-01 ~20:30](hypotheses.md#added-2026-11-01-2030-cst) | v1 retired 2026-11-02 before they were scored |
| A6 | self-imitation speeds up winning | [2026-11-01 ~22:35](hypotheses.md#added-2026-11-01-2235-cst) | self-imitation paused within the window; v1 retired |
| A7 | late-ante retry arms win more per game than fresh seeds | same | v1 retired |
| G1–G4 | a `win`-reward arm (then called "v3") raises win rate, holds blinds, plays hail-marys when behind, spends down late | [2026-11-02 ~11:05](hypotheses.md#added-2026-11-02-1105-cst-authorship-and-v3-registered-before-v3-exists) | never run; its start checkpoint's line was retired |
| I1 (chud's) | generalist plus specialists beats the best bot so far | [2026-11-03 ~17:00](hypotheses.md#added-2026-11-03-1700-cst-i-strategy-specialists-design-registered-before-any-specialist-exists) | its test was never registered; the MoE line was replaced by v4 |
| I2 (chud's) | Full House is one of the strongest experts | [2026-11-04 ~13:35](hypotheses.md#added-2026-11-04-1335-cst-i2-full-house-is-one-of-the-strongest-experts-section-i) | closed [2026-11-09 ~18:17](hypotheses.md#added-2026-11-09-1817-cst-the-full-house-venture-closed-and-the-bar-for-going-up-the-decks-and-stakes) (no run forces a hand) |
| Full House criterion | the MoE has learned Full House | [2026-11-04 ~13:40](hypotheses.md#added-2026-11-04-1340-cst-completion-criterion-the-moe-has-learned-full-house) | not met at pins zero; closed 2026-11-09 ~18:17 |
| `moe2-nopin` | the control that separates the pin from the reward | [2026-11-04 ~13:50](hypotheses.md#added-2026-11-04-1350-cst-review-of-the-early-full-house-pin-for-a-critical-reader) | never run; closed 2026-11-09 ~18:17 |
| Joker-to-hand correlation | jokers correlate with hand strategies (chud's interview) | [2026-11-03 ~17:00](hypotheses.md#added-2026-11-03-1700-cst-i-strategy-specialists-design-registered-before-any-specialist-exists) | never registered as a prediction |
| Section J draft | the next MoE's design options | [2026-11-05 ~11:00](hypotheses.md#added-2026-11-05-1100-cst-j-the-next-mixture-of-experts-design-draft-nothing-registered) | replaced by the v4 registration |
| V1–V3, v3 A/B | v3 wins ≥ 4% and beats pins0; more blinds; the gain comes from the build | [2026-11-05 ~21:19](hypotheses.md#added-2026-11-05-2119-cst-model-v3-registered-before-any-of-its-code-exists-section-j) | v3 replaced as the target by v4 ([2026-11-05 ~23:01](hypotheses.md#added-2026-11-05-2301-cst-model-v4-registered-before-any-of-its-code-section-j)); never run |

## Still open (not closed here)

v4's, and the general philosophical and project goals (chud, 18:33), which no one model answers and which
stay open across models.

| ID | What | Registered |
|---|---|---|
| W1–W3 | v4 wins ≥ 4% and beats pins0; ≥ 10%; reaches ante 7 more, dies less in antes 4–6 | [2026-11-05 ~23:01](hypotheses.md#added-2026-11-05-2301-cst-model-v4-registered-before-any-of-its-code-section-j) |
| W4a, W4b (chud's) | v4 at least doubles wins per compute (play cost; training cost at 16.71 h) | [2026-11-09 ~12:00](hypotheses.md#added-2026-11-09-1200-cst-v4-amended-before-any-of-its-code-section-j), [2026-11-09 ~15:48](hypotheses.md#added-2026-11-09-1548-cst-w4bs-compute-target-fixed-and-claudes-odds-after-the-measured-costs-section-j) |
| Aim 1's precondition | the best eval reaches ante 7 in ≥ 25% of games | [2026-11-05 ~23:01](hypotheses.md#added-2026-11-05-2301-cst-model-v4-registered-before-any-of-its-code-section-j), [2026-11-09 ~15:48](hypotheses.md#added-2026-11-09-1548-cst-w4bs-compute-target-fixed-and-claudes-odds-after-the-measured-costs-section-j) |
| The 25% bar | 25% eval wins, Red/White, before going up the decks and stakes (a decision rule) | [2026-11-09 ~18:17](hypotheses.md#added-2026-11-09-1817-cst-the-full-house-venture-closed-and-the-bar-for-going-up-the-decks-and-stakes) |
| C1–C3 (section C) | the main experiment: different oughts give different behavior; an ought that isn't the goal can beat it; bot vs chud | [section C](hypotheses.md#c-isolating-oughts-the-main-experiment) |
| B1 | shaping changes speed, not what the bot ends up preferring | [section B](hypotheses.md#b-shaping-is-speed-not-values) |
| E1 (chud's) | Hanging Chad is the best common joker | [2026-11-01 ~20:30](hypotheses.md#added-2026-11-01-2030-cst) |
| Aims 2 and 3 | a proven naneinf seed; a 100% save (no prediction registered) | [2026-11-02 ~23:05](hypotheses.md#added-2026-11-02-2305-cst-the-projects-aims-stated-by-chud) |

Section D (2026-11-01) is a to-do list, not hypotheses, and isn't tracked here.
