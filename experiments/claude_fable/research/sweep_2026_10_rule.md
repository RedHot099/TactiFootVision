# Method sweep on the 172 held-out matches (2026-10-18, owner-ordered)

Written and committed before any test label of this sweep is read. Script: `scripts/sweep_test.py`.

## Status

Descriptive. Not a confirmatory test, no success claim:

- the 172 matches (135 evaluation + 37 gate) were already read by xg-vendor (G-combo), the A1 test (xg-detector 07)
  and the xg-stack test (xg-stack 04); they are spent for detector-input variants;
- the retrained detector (`models/detector_yolo11l_1280_dd2c.pt`) was trained on SoccerNet-v3 frames of these matches,
  including frames at the shots themselves, more often for goals than for other shots.

Every reading is therefore reported on the whole population and on the shots the retrained detector never saw.

## Population

The 3,819 non-penalty shots of the 172 held-out matches (`stack_heldout.testset.heldout_shots`), labels = index `goal`,
read once after every prediction is written.

## Methods (every model fitted on the 7,062 training shots of the 328 training matches only)

| Name | Inputs | Fit |
|---|---|---|
| const | none | training goal rate |
| g_combo | pixels (deployed G-combo, `results/xgd_gcombo_test`) | as deployed |
| a1 | pixels + D3 detector geometry in the head (`results/xgd_test`) | as tested |
| stack_R | G-combo logit | xg-stack final fit (`final.final_fit`, C6 spec) |
| stack_G_old, stack_GS_old, stack_ALL_old | G-combo logit + E3 features, old detector (`cache/xgd_full3/E3`) | same |
| stack_G_new, stack_GS_new, stack_Q_new, stack_ALL_new | G-combo logit + E3 features, retrained detector (`cache/xgd_full4/E3`) | same |
| pos_DA_old, pos_DA_new | detector distance (spline), angle, missing flags; no pixels | same, all coefficients penalised |
| pos_ALL_new | every E3 column of arm ALL except the G-combo logit; no pixels (approach 2, logistic) | same |

stack_R/G/GS_old must reproduce `results/xgs_test_prep/predictions.csv` exactly (else the sweep stops).

## Metrics

Log-loss (p clipped at 1e-6), AUC, Brier, calibration slope; ΔLL = LL(method) − LL(g_combo) with the paired match
bootstrap of the xg-stack test rule (5,000 draws, seed 0, percentile 95% interval). No multiplicity correction:
nothing is a test.

## Subsets

1. all 3,819 shots; 2. evaluation and gate roles; 3. **unseen**: no SoccerNet-v3 pool image of the match within 3 s of
the shot (`xgs_detector_leak.seen_shots`); 4. matches without any v3 pool image.

## StatsBomb reference (separate step, after the sweep's result is written)

On the paired primary shots of the 135 evaluation matches (`results/xgc_compare/table/shots.parquet`, population
`primary`): vendor xG, the positional comparator (freeze frame), events and distance-angle rungs, scored beside every
method on the same shots. Data: StatsBomb. Validation only; no model sees StatsBomb.
