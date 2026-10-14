# xg-stack: the frozen test rule

Status: written in ticket 04 (2026-10-14), before any test label is read. The test runs once, by the orchestrator.
Parameters: `research/xg_stack_test_rule.json` (machine-readable, read by `scripts/xgs_test.py`; `xgpoc.stack_heldout.rule.load_rule` refuses any value other than council C7's).
Hash list: `research/xg_stack_test_hashes.json`, committed alone; its sha256 is in ticket 04's Comments.
Basis: the owner's order of 2026-10-14 ("Zrób test na całym zbiorze meczy") and council decision C7 (`.scratch/xg-detector/council.md`), which is binding. C5 and C6 fixed the design.

## 1. The variant

**G is the only confirmatory variant.** It is the C6 stacker (candidate (b)+(c) of `research/xgs_stack_protocol_c6.json`, sha256 `eaaa6b3b…`) on the frozen G-combo logit:

| Arm | Inputs | Status |
|---|---|---|
| R | `z_g_combo` alone, unpenalised | reference |
| G | R + `dist@0` (natural cubic spline, 3 df) + `angle@0` (wrapped to [0, π], linear) + `shot_missing@0` + `ball_none@0`; C by inner CV, z unpenalised | confirmatory |
| GS | G + the SoccerNet flags with non-zero training variance (`sn_direct_fk`, `sn_indirect_fk_3s`, `sn_corner_20s`, `sn_prev_shot_5s`) | descriptive, no success label |

The arms' columns must equal the C6 protocol's `design.arms.<arm>.columns`; the test refuses otherwise.

Development (xg-stack ticket 02, 7,062 shots, 328 matches, out of fold): G ΔLL −0.00065 [−0.00151; +0.00020], bootstrap SD 0.00043; GS −0.00048 [−0.00138; +0.00036]. Neither was a detectable improvement. The test is the owner's order, not a recommendation of the development read-out.

## 2. The final fit (training only)

R, G and GS are fitted **once on all 7,062 training shots** (`xgpoc.stack_heldout.final`):

1. **Arms.** `fit.spec("bc", training table)`: the zero-variance list is the training rows' (it drops `sn_team_unknown`).
2. **C.** Match-grouped CV over the 5 host folds of `cache/folds_xgv_home5.csv` on the frozen grid {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}. For each host fold the design is preprocessed on the other four folds' rows only, every C is fitted there and scored on the fold; the summed held-out log-loss picks C, ties to the smaller C. An edge value of the grid is accepted and recorded. R has no penalised column and needs no CV.
3. **Refit.** Imputation means, natural-spline knots (quantiles 0, ⅓, ⅔, 1) and standardisation are fitted on all 7,062 training rows; the L2 logistic regression (`xgpoc.stack.fit.logistic_fit`) is fitted there at the chosen C, with z unpenalised.

The training outcomes are the index `goal` of the training shots, read inside the held-out label seal. The coefficients, chosen C, inner losses and every transform are written to `results/xgs_test_prep/final_fit.json` before any test label exists in the process.

## 3. Population

Every index v1.2 shot (sha256 `3968d532…`) whose match has the role `evaluation` or `gate` in the split plan (sha256 `a56054ce…`) and whose `is_penalty` is false: **3,819 shots in 172 matches** (3,005 shots in 135 evaluation matches, 814 in 37 gate matches). This is the A1 test's population (`research/xg_detector_test_rule.md` §3). There is no other exclusion; a shot that cannot be scored stops the test. The label is the index's `goal`.

## 4. Test inputs (label-free)

- **z.** The test z is the pre-Platt averaged G-combo logit, recovered from the deployed probability p of `results/xgd_gcombo_test/predictions.csv` by inverting the full-precision Platt of `results/xgv_g_combo/ensemble.json`: z = (logit(p) − b) / a, with a = 0.9803331788739801 and b = −0.051591547304639884. The prior correction is the identity there (out-of-fold rate = pooled rate) and is refused otherwise.
- **E3.** `cache/xgd_full3/E3` replayed with E3's ball-source rule; the replay must equal `features.parquet` exactly. Then the frozen sanitation, and the angle wrap min(θ, 2π − θ) with its count (`xgpoc.stack.inputs.e3_table`).
- **Unseen `ball_source` levels.** A level never seen in training enters no model, because the zero-variance list is the training rows'. It is counted and reported, not a stop. A code outside E3's levels stops the build.
- **Flags.** SoccerNet `Labels-v2.json` of the 172 matches; "Goal" is dropped and both shot types become one kind at parse (`xgpoc.stack.inputs.flags_table`).
- **Predictions.** `results/xgs_test_prep/predictions.csv`: z, the deployed G-combo p, p_R, p_G and p_GS per test shot, written before any test label is read.

## 5. Metric and bootstrap

- **Metric:** shot-weighted log-loss, p clipped to [1e−6, 1 − 1e−6]. **ΔLL = LL(G) − LL(R)**; negative means G is better.
- **Paired match bootstrap:** 5,000 draws, seed 0, over the 172 matches with replacement (`xgpoc.detector.testrule.match_weights`, A1's draws). A replicate is Σ w·d / Σ w, d the per-shot difference, w the draw count of the shot's match. Intervals are percentile intervals.

## 6. Success, worse and equivalence

- **Success** (the only confirmatory criterion): the upper bound of the two-sided interval at the **success level in force** is below 0. The level is 95% unless the synthetic-null gate (§9) moved it.
- **Worse:** the lower bound of the same interval is above 0.
- **Equivalence (descriptive, TOST at α = 0.05):** the 90% interval from the same draws lies wholly inside the open (−δ, +δ). **δ = 0.002**, the practical bar, sets the label; **δ = 0.005** (A1's) is reported beside it for comparability. Equivalence never counts as a success and cannot rescue a failed success rule.
- **Practical bar:** a success with a point ΔLL above −0.002 is below the practical bar and must be reported as such.
- **Smallest equivalence margin:** max(|lower 90% bound|, |upper 90% bound|).

## 7. Reading labels and their report sentences (fixed in advance)

The label is the first that applies: **success and equivalent**, **success**, **worse**, **equivalent within δ**, **inconclusive** (δ = 0.002). Both equivalence flags are reported in every case.

The report sentence of each label is written in advance in the JSON (`sentences`), in Polish; the test fills in the numbers. In English:

| Label | Sentence (Polish in the JSON) |
|---|---|
| success and equivalent | The test on the 172 held-out matches showed an improvement: ΔLL(G − R) = … [… ; …] (L% interval, upper bound below 0). The 90% interval lies within ±0.002, so the improvement is smaller than the equivalence margin. |
| success | The test on the 172 held-out matches showed an improvement: ΔLL(G − R) = … [… ; …] (L% interval, upper bound below 0). |
| worse | The test showed a deterioration: … (lower bound above 0). The E3 geometry in the stacker hurts on the held-out matches. |
| equivalent within δ | The test showed neither an improvement nor a deterioration: … The 90% interval lies within ±0.002, so G and R are equivalent within δ = 0.002 (TOST, descriptive). |
| inconclusive | The test showed neither an improvement nor a deterioration: … The 90% interval does not lie within ±0.002: inconclusive (smallest equivalence margin …). |

A success also gets one of the two practical clauses: "the point … is above the practical bar −0.002: the improvement is detectable but below the practical bar", or "the point … reaches the practical bar −0.002".

## 8. Label-free checks and their stop conditions (C7 Q5)

All run before the freeze and again, by replay, in `--check` and `--test`. Any failure stops the test.

| Check | Rule | Stop when |
|---|---|---|
| z inversion | max \|Platt(z) − p\| over the 3,819 shots; max \|z − member z\| where member logits are stored (the 814 gate shots' stored member predictions and the 284 unpaired evaluation shots' `unpaired_members.csv`), member z = the mean over the ten members of logit(sigmoid(logit / T)); no p at the clip. The 2,721 paired shots' p were checked against their member logits in xg-detector ticket 06 (`results/xgd_gcombo_test/checks.json`, max \|Δp\| 5.6e−17). | round trip > 1e−12, member gap > 1e−6, or any p at the clip |
| z distribution | the test z's mean and SD (ddof 1) over the 7,062 training out-of-fold z's | either ratio outside [0.8, 1.25] |
| coverage | 3,819 shots; 172 matches (135 + 37); 3,005 and 814 shots; unique shot ids; one-to-one joins of the G-combo p and flags; finite z; predictions inside (0, 1); the E3 rows' exact coverage raises in the build | any item fails |
| provenance | no test shot or match is a training one; every test shot is of an evaluation or gate match; the arms' columns equal the C6 protocol's and carry no banned name; the flags do not change when "Goal" annotations are added and shot types swapped (recorded in `checks.json`). Enforced rather than recorded: every read runs inside the label seal and with no StatsBomb or xg-compare path (`testset.no_vendor_reads`), and a refused read raises | any item fails |
| E3 replay | the replayed E3 features equal the cache exactly | any difference |
| ball source | t0 level counts in test and training, levels unseen in training | reported only |
| angle wrap | test shots per offset whose angle the wrap changed | reported only |

Before the code closure is taken, `--prepare`, `--freeze`, `--check` and `--test` also run the scoring path once on synthetic outcomes y* ~ Bernoulli(p_R) (seed 0, no label), so the labelled run loads no module the closure lacks. The replay in `--check` and `--test` rebuilds the final fit, the test inputs, the predictions and the checks, and must reproduce `results/xgs_test_prep/{final_fit.json, test_inputs.csv, predictions.csv, checks.json}` byte for byte.

## 9. Synthetic-null false-success gate (labels unread)

On the written test predictions: 5,000 outcome vectors y* ~ Bernoulli(sigmoid(logit(p_R) + u_m)), with a match random intercept u_m ~ N(0, 0.25²) drawn per replicate before the outcomes (C6's clustered-null SD), seed 301. For each y*, ΔLL(G − R) and its bootstrap with the test's own 5,000 draws. A false success at level L is an upper bound of the L interval below 0.

**Ladder.** If the false-success rate at 95% is above 0.05, the success level moves to the first of 97.5%, 99% and 99.5% whose rate is ≤ 0.05. If none is, the test does not run and the question returns to the council. The level in force is written into the JSON's `before_scoring` block before any test label is read.

## 10. Result of the checks and the gate (before scoring)

From `results/xgs_test_prep/{checks.json, gate.json, final_fit.json}`, written with no test label (training labels inside the seal only).

**Final fit.** G: C = 0.003 (not at a grid edge). GS: C = 0.001, the grid's lower edge, accepted as the rule says. R: no C (z alone, unpenalised).

**Checks** (all passed):

| Check | Result | Stop condition |
|---|---|---|
| z inversion | round trip max \|Δp\| 1.1e−16; member gap max \|Δz\| 2.7e−9 (mean 1.5e−10) on 1,098 shots (814 gate, 284 unpaired); no p at the clip | 1e−12; 1e−6; any |
| z distribution | test z mean −2.127, SD 0.875; training mean −2.167, SD 0.906; ratios 0.982 (mean) and 0.966 (SD) | outside [0.8, 1.25] |
| coverage | 3,819 shots, 172 matches (135 + 37), 3,005 and 814 shots, every join one to one | any item |
| provenance | disjoint from training, columns of R, G and GS equal the C6 protocol's, flags invariant to "Goal" and shot types | any item |
| E3 replay | identical on every row | any difference |
| ball source at t0 | test none 351, own 112, m400 2,992, m800 364; no level unseen in training | reported |
| angle wrap | 9 test shots at t0, 8 at −400 ms | reported |

Also reported: 132 test shots have no E3 shot location at t0; no test `dist@0` lies outside the training knots; the flags count `sn_direct_fk` 274, `sn_indirect_fk_3s` 25, `sn_corner_20s` 484, `sn_prev_shot_5s` 130. One test shot has `sn_team_unknown` = 1 (its flags are all 0); that column is constant in training, so it is in the training zero-variance list and enters no model.

**Gate.** On 5,000 synthetic outcome vectors (seed 301), the false-success rate of G's rule is **0.0006** at 95% (Monte Carlo SE 0.0003), and 0 at 97.5%, 99% and 99.5%. It is below 0.05, so **the success level in force is 95%**. The mean ΔLL under the null is +0.00063: with p_R as the truth, G's written predictions cost log-loss, which is why false successes are rare. The false-worse rate at 95% is 0.21 (descriptive).

## 11. Expected power (before scoring, no test label)

From `results/xgs_test_prep/power.json`. The method (normal approximation, `xgpoc.detector.testrule.expected_power`):

- **Standard errors:** the development bootstrap SD of G's ΔLL (`results/xgs_stack/real/summary.json`, 2,000 draws) × √(7,062 / 3,819) (the headline); the same SD × √(328 / 172) (match-count scaling); A1's test bootstrap SD 0.001082 (`results/xgd_test`, xg-detector ticket 07).
- **Effects:** G's development point (the headline); the practical bar −0.002 shrunk by C6's recovery (control 4: G recovered a share of a planted −0.002); the bar itself.

**Headline: at G's development point (ΔLL −0.00065) and the shot-scaled SE 0.00058, P(success) at 95% is 0.197.** A non-success is the expected outcome even if the development point is the truth.

| SE scenario | SE | True ΔLL | P(success), 95% | P(worse), 95% | P(equivalent, δ 0.002) | P(equivalent, δ 0.005) |
|---|---|---|---|---|---|---|
| development SD × √(7,062/3,819) | 0.00058 | −0.00065 (development point) | **0.197** | 0.001 | 0.748 | 1.000 |
| | | −0.00159 (−0.002 × recovery 0.793) | 0.776 | 0.000 | 0.175 | 1.000 |
| | | −0.002 | 0.929 | 0.000 | 0.050 | 1.000 |
| development SD × √(328/172) | 0.00059 | −0.00065 | 0.192 | 0.001 | 0.737 | 1.000 |
| | | −0.00159 | 0.763 | 0.000 | 0.172 | 1.000 |
| | | −0.002 | 0.922 | 0.000 | 0.050 | 1.000 |
| A1's test SE | 0.00108 | −0.00065 | 0.086 | 0.005 | 0.135 | 0.991 |
| | | −0.00159 | 0.311 | 0.000 | 0.056 | 0.935 |
| | | −0.002 | 0.456 | 0.000 | 0.030 | 0.870 |

The development SD comes from out-of-fold fits on about 5,650 shots each, while the final fit uses 7,062, and the stacker recovers about 79% of a planted effect; the A1 SE is a different comparison (A1 against G-combo) and is a pessimistic sensitivity. All of these are planning figures. P(success) at 97.5%, 99% and 99.5% is in `power.json`; the gate left the level at 95%.

## 12. Descriptive readings (C7 Q4)

None of these is a success, none rescues the primary reading, and nothing feeds the headline. Each is labelled "hypothesis formed on these matches".

- **GS against R:** ΔLL with the same draws and intervals; no success label.
- **G against the deployed G-combo:** ΔLL with the same draws. R against the deployed G-combo is added beside it for reference (beyond C7 Q4's list).
- **Scores:** log-loss, AUC, Brier, mean p, calibration slope and intercept of R, G, GS and the deployed G-combo.
- **Evaluation and gate matches:** ΔLL(G − R) restricted to the 3,005 evaluation shots and to the 814 gate shots, the same draws, each replicate the weighted mean over the role's shots.
- **E3 categories:** O, E, O/E per arm and ΔLL against R with its 95% interval from the same draws restricted to the category's shots, in the development's categories: E3 `shot_x@0` < 88.5 (outside the box), E3 `angle@0` < 15°, E3 `dist@0` < 6 m, `sn_corner_20s` = 1, `sn_direct_fk` = 1. A shot whose defining input is missing is outside the category's population.
- **StatsBomb agreement on the 135 evaluation matches** (Data: StatsBomb.), once, by `scripts/xgs_test_statsbomb.py`, which refuses unless `results/xgs_test/complete.json` exists under this hash list. Population: xg-compare's primary shots (`results/xgc_compare/table`, 2,721 paired shots), joined one to one with the test predictions.
  - Spearman's ρ of p_R and p_G with the vendor's xG; the deployed G-combo's ρ is added for reference (beyond C7 Q4's list).
  - xg-compare's gap items in StatsBomb coordinates, kept apart from the E3 < 6 m category: outside the box (x < 102 or |y − 40| > 22 yd), goal-mouth angle < 15°, distance < 6 yd. Per model, over the shots where the category is defined: shots in and out, the mean p of the model and the vendor in, the gap in xG points 100 (mean p_model − mean p_vendor) in and out, and the unadjusted contrast of the logit gap (in minus out). These use the models' own probabilities, not xg-compare's cross-fitted main scale.
  - StatsBomb is never an input, target, selection or check of any model, and is not read before the test result exists.

## 13. How the test runs

```
./run.sh --no-sync python scripts/xgs_test.py --test --expect-hashlist <sha256 from ticket 04>
```

**Refusals.** The script refuses to read a label unless all of the following hold:
- the hash list hashes to the given sha256 and is the committed, unmodified file;
- every pinned file (the rule, the C6 protocol, the preparation's files, the development records the power reads, the index, plan, folds, out-of-fold file, E3 cache, G-combo's ensemble, test predictions, member files, the 500 matches' `Labels-v2.json` and the StatsBomb descriptive code) and the loaded code closure match it;
- the preparation passed under these rule parameters and this code, and the rule's `before_scoring` block is its power and gate;
- the replay reproduces the preparation's files byte for byte;
- `results/xgs_test` does not exist yet.

`--check` runs every refusal and the replay, and reads no test label.

**Order.** Inside the label seal and with no StatsBomb or xg-compare path: the replay and every check. Then, outside the seal, the index `goal` of the 3,819 shots is read once and the rule is applied. The script writes `results/xgs_test/{result.json, categories.csv, complete.json}`.

**Reruns.** The test runs once. A rerun only after a purely technical failure, with the same inputs, written down in ticket 04.

## 14. Disclosures

- **The 172 held-out matches are now spent for detector-input variants.** This is their second labelled look for a detector variant (the first was A1, xg-detector ticket 07). No further detector-input variant may be tested on them.
- The owner lifted the one-test rule for this test only.
- E3's configuration was chosen on 135 of the 172 matches by StatsBomb position error, with no outcomes.
- The xg-compare phase looked at these matches' outcomes (exploratory). It motivated the inputs, and partly the flag windows.
- The 135 evaluation matches' labels were seen in xg-vendor; the 37 gate matches were scored by G-combo in xg-vendor's retraining gate.
- **The test z averages ten members, the training z two.** Every test shot is held out by all ten G-combo members; every training shot only by the two models of its fold. The z-distribution check (§8) bounds the resulting scale shift.
- `z_g_combo` is the averaged pre-Platt logit; each member's epoch and temperature were chosen on its own scored fold, an outcome-dependent step (C6's corrected disclosure 3).
- Stacking provenance: the training z values come from models fitted on the other training folds. The final fit uses all 7,062 training shots, so its estimation cost is below the development's.
- The flags are research-only (a product needs an action spotter); GS is descriptive.
- The match bootstrap resamples matches for fixed fitted models; it does not capture the final fit's own variability.

## 15. Scope

"The result applies to the C6 stacker over the frozen G-combo with E3 geometry as audited in pitch-keypoints, not to detector-derived structure in general."
