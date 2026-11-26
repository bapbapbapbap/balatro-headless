# Changelog

Dates are YYYY-MM-DD. Everything before 2026-11-25 (tooling versions 0.1.0 to 0.13.x and the unreleased v4-v6 work)
is in the CHANGELOG at the tag `archive-2026-11-25` (`git show archive-2026-11-25:CHANGELOG.md`).

## 2026-11-25 (later): front page
### Changed
- README cut to two screens: the claim, how to replay a win, one results table, limits, related work, links. Its
  layout follows the front pages of AnteZero and Atol8/balatro-ai. The harness checks moved to `docs/REPRODUCE.md`.
- The long tally now lives only in `wins/README.md`; the front page keeps the badges (`train/tally.py` writes the
  README block only if the README has the markers).
- Related work dated: AnteZero v0.1.0 (2026-11-12, Red/White only, 1-2% Gold on six decks) and v0.2.0 (2026-11-20);
  correction section in `docs/hypotheses-v6.md`; system card limits updated.

## 2026-11-25: cleanup to the current lineage
### Changed
- Main now holds only what the current policy (v6g best.pt, round 660) needs: the harness, the trainer, the teacher
  bridge (`reproduce/balatro_lab/`), runs v6e, v6dt, v6g, the Gold evaluation and the record of beaten seeds.
  Every earlier run's code, configs, docs, registrations and released weights are at tags (`ARCHIVE.md`).
- Merged: branch `v6-win-attempts` (v5-v6 code and record, eval at any deck and stake, win attempts, the skip test),
  PR #130 (Gold wins: Green, Abandoned, Yellow, Ghost), PR #131 (BalatroBench comparison).
- README, `llms.txt`, `AGENTS.md`, `ARCHIVE.md`, the run index and the hypotheses index rewritten for the current
  lineage; related work (Balatro Lab, AnteZero, Atol8/balatro-ai) cited in the README.
- `tools/balatro.ps1` defaults to run v6g.
### Added
- `gold/`: `verify_gold.py`, `gold_table.py`, `seed_pool.py` and their results (`gold/results/`).
- `docs/system_card_v6g.md`: the policy, its checks, what was learned, limits.
- `docs/hypotheses-v6.md`: the 15-deck Gold results, verification and seed sharing (2026-11-25 section).
- `tools/archive_branches.sh`: creates the archive tags and, with `--delete`, removes merged branches (chud's clone).
