# Remaining contact review

- [x] Frame: inspect sheets 102–299 visually; write contact bounds, t0 status, body part, confidence and evidence. Missing visibility remains uncertain.
- [x] Fan out: native build workers cover 102–200 and 201–299 in separate CSV files.
- [x] Aggregate: validate exact coverage and allowed values; spot-check images and report uncertainty.
- [x] Report: combine with existing 0–101, report timing distribution. No automatic label corrections from ambiguous reviews.

2026-09-09: all first 100 sheets rechecked; additional spot checks 150 and 250. Current summary: 164 uncertain, 130 intervals containing t0, 5 strictly after t0 and 1 strictly before. Independent human validation remains outstanding. See quality_v13_results_2026-09-09.md.

All nine-frame sheets include future frames for audit only. They must never enter model inputs. Workers must inspect every assigned image through an image tool; filenames or metadata are not visual evidence.
