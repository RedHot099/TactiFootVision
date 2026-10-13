# xg-detector: the frozen test rule

Status: frozen in ticket 06 (2026-10-13), before any test prediction of A1 exists. Ticket 07 runs it once.
Parameters: `research/xg_detector_test_rule.json` (machine-readable, read by the test script).
Hash list: `research/xg_detector_test_hashes.json`; its sha256 is quoted in ticket 06's Comments and in the message of the commit that adds it.
Basis: council decision C2 (`.scratch/xg-detector/council.md`), ticket 06 and the spec (stories 5–6, 21–22).

## 1. The variant

There is one variant: **A1**. A1 is G-combo with the sanitised D3 geometry branch fused into its head. It keeps ticket 03's frozen recipe, sanitation rule and branch hyperparameters (`xgpoc/detector/approach1.py`, `sanitation.py`), with no grid.

The development selection comes from ticket 03: A1 had the lower out-of-fold log-loss of A1 and A2. Against G-combo seed 0 on the 7,062 non-penalty pool shots:

| Variant | ΔLL | 95% interval |
|---|---|---|
| A1 | −0.0007 | [−0.0032; +0.0018] |
| A2 | −0.0000 | [−0.0021; +0.0020] |
| Noise floor (G-combo seed 1 − seed 0) | −0.0006 | [−0.0031; +0.0021] |

No other variant exists. A3 was not triggered (ticket 04), and approach 2 closed at the gate (ticket 05). With one variant there is no multiplicity correction.

## 2. The two arms

Each arm is a full **10-model ensemble**: folds 0–4 × seeds 0 and 1, the plan's home-team folds on the 328 training matches.

- **A1**: `results/xgd_a1/fold<k>/seed<s>`. Seed 0 is from ticket 03; seed 1 was trained in ticket 06 with exactly the same recipe, sanitation and hyperparameters. All ten runs have completion records, and a replay of all ten passed (`results/xgd_a1/replay_ensemble.json`).
- **G-combo**: the ten retrained fold models of `results/xgv_g_combo`, unchanged.

**Aggregation, the same for both arms:**
1. Each member's goal logit is divided by its temperature. The temperature is stored in the member's bundle and was fitted on its held-out fold.
2. The ten results are averaged in logit space.

**Calibration, the same for both arms.** It is exactly the procedure that made G-combo's `ensemble.json` (`scripts/vendor_verify_g.py`, `g_bundle.fit_ensemble_calibration`):
1. Platt scaling (unregularised logistic regression on the averaged logit) is fitted on the pool's out-of-fold ensemble. For each of the 7,166 pool shots, penalties included, that is the mean of logit(stored p) over the 2 models that held the shot out.
2. Prior correction maps the out-of-fold goal rate to the pool's goal rate. The two rates are equal here, so it changes nothing.
3. Penalties are routed to the pool's penalty rate: a Beta prior of 0.76 × 40, updated with the pool's penalties, gives 0.6972. Penalties are outside the metric anyway.

The calibration files:
- A1: `results/xgd_a1_ensemble/ensemble.json`.
- G-combo: `results/xgv_g_combo/ensemble.json`. Refitting the same procedure on G-combo's stored out-of-fold files reproduces its Platt coefficients exactly (checked by `scripts/xgd_a1_ensemble.py`).

The deployed probability for both arms is `g_bundle.ensemble_probability(member logits, temperatures, calibration, is_penalty)`.

**Where the predictions come from:**
- **A1** is scored by the test script at test time. It uses the t0 frame and each bundle's training-fold standardiser applied to the sanitised D3 table, in eval mode at batch 8.
- **G-combo** comes from `results/xgd_gcombo_test/predictions.csv`, prepared in ticket 06 without touching its bundles:
  - the 2,754 paired shots come from xg-vendor's evaluation table (byte-checked);
  - the 825 gate shots come from the stored gate predictions (checked);
  - the 285 other evaluation-match shots were scored with the same ten bundles (inference only).

## 3. Population

The population is every index v1.2 shot that meets both conditions:
- its match has the role `evaluation` or `gate` in the split plan (sha256 `a56054ce…`, 135 + 37 = 172 matches);
- its `is_penalty` is false.

That gives **3,819 shots in 172 matches**:
- 3,005 evaluation shots: 2,721 paired and 284 unpaired;
- 814 gate shots.

There is no other exclusion. Every such shot has a t0 frame and a sanitised D3 row. If either arm cannot score a population shot, the test stops; it does not drop the shot. The label is the index's `goal`.

## 4. Metric

The metric is shot-weighted log-loss: the mean over the population's shots of −(y log p + (1 − y) log(1 − p)), with p clipped to [1e−6, 1 − 1e−6] for both arms. The difference is **ΔLL = LL(A1) − LL(G-combo)**. Negative means A1 is better.

## 5. Paired match bootstrap

- **5,000 draws, seed 0.**
- Each draw resamples the 172 matches with replacement (`xgpoc.probe.readability.match_draw_counts` over the sorted unique matches).
- A replicate is Σ w·d / Σ w, where d is the per-shot difference in log-loss and w is the draw count of the shot's match. The same draws weight both arms.
- Intervals are percentile intervals of the 5,000 replicates.

## 6. Success rule (unchanged)

**Success**: the upper bound of the 95% interval of ΔLL is below 0. This is the only confirmatory criterion.

**Worse**: the lower bound of the 95% interval is above 0.

## 7. Descriptive equivalence reading (council C2)

This is a TOST at α = 0.05. It holds when the **90% interval from the same draws** lies wholly inside (−0.005, +0.005); the bounds are exclusive.
- It is reported separately from the success rule.
- It can never count as a success.
- No subgroup, alternative metric or equivalence finding may rescue a failed success rule.

## 8. Reading labels (fixed in advance)

The label is the first of these that applies:

1. **success and equivalent**: success holds and equivalence holds;
2. **success**;
3. **worse**;
4. **equivalent within δ**;
5. **inconclusive**.

The equivalence flag is also reported as its own field in every case, so "worse" with an equivalent 90% interval stays visible.

The report also gives the **smallest equivalence margin**: max(|lower 90% bound|, |upper 90% bound|), the smallest δ′ whose (−δ′, +δ′) would contain the 90% interval.

## 9. Expected power (computed before scoring, development data only)

**Method** (`scripts/xgd_test_power.py`, `results/xgd_test_power/power.json`). It reads no label of a test shot: it runs inside the held-out label seal, and the test population's size comes from label-free index columns.
- The input is ticket 03's out-of-fold paired match variance: the per-shot differences A1 seed 0 − G-combo seed 0 on 7,062 non-penalty pool shots in 328 matches.
- Its bootstrap SE is 0.00127 (5,000 draws, seed 0, ticket 03's draws). The analytic match-cluster SE is 0.00124.
- Scaled to the test population by √(7,062 / 3,819) = 1.360, the expected SE of the test's ΔLL is **0.00173**. Scaling by matches, √(328 / 172), would give 0.00175.
- The probabilities use a normal approximation with the rule's levels.

| assumed true ΔLL | P(success) | P(worse) | P(equivalent within δ) | 95% half-width | expected smallest margin |
|---|---|---|---|---|---|
| −0.0007 (ticket 03's point) | **6.0%** | 0.9% | 75% | 0.0034 | 0.0035 |
| −0.002 (the A3 trigger) | **21.1%** | 0.1% | 53% | 0.0034 | 0.0048 |

**Sensitivity:** the 10-model ensembles' own out-of-fold difference. The two calibrated 10-model ensembles differ out of fold by ΔLL −0.0008 [−0.0026; +0.0011] on the same 7,062 pool shots (A1 0.3419, G-combo 0.3427; `results/xgd_a1_ensemble/comparison.json`). Its bootstrap SE is 0.00093, smaller than ticket 03's single-model 0.00127, because averaging seeds removes part of the seed noise. Scaled the same way it gives an expected test SE of 0.00127: P(success) is 7.9% at −0.0007 and 35.0% at −0.002, and P(equivalent within δ) is 96% and 76%. This is reported for orientation only; the stated power is the table above, as council C2 asked.

**Caveat:** the power of the success rule is low. At a true difference of the development size, a success would occur about 6% of the time, so a success would be read with that caveat. A non-success is the expected outcome even if A1 is truly slightly better. The equivalence reading is what makes a null result informative.

## 10. What is frozen

Before scoring, the following are frozen. Their code and data are pinned byte for byte in the hash list.
- The **A1 ensemble**: the ten bundles, the aggregation, `results/xgd_a1_ensemble/ensemble.json`.
- **G-combo's** ten bundles, its `ensemble.json` and its test predictions.
- **Shot inclusion** (section 3), the **probability clip** (1e−6) and the **shot-weighted ΔLL**.
- The **bootstrap implementation** (`xgpoc/detector/testrule.py`: `shot_log_loss`, `match_weights`, `paired_delta`, `reading`).
- The test script and its whole code closure (`scripts/xgd_test.py` and every checkout module it loads).
- Inputs: the sanitised D3 table (`cache/xgd_sanitised/D3`), the split plan and folds, the index, the overlay mask, the pretrained backbone, and the fingerprint of the population's t0 frames.

## 11. How the test runs (ticket 07)

`./run.sh python scripts/xgd_test.py --expect-hashlist <sha256 from ticket 06>`

**Refusals.** The script refuses to score unless all of the following hold:
- the hash list hashes to the given sha256 and is the committed, unmodified file;
- every pinned file and the loaded code match it;
- the t0 frames match their fingerprint;
- every variant has complete records;
- a path check passes: A1 fold 0 seed 0 rescores 16 training-pool shots within 1e−4 of its stored logits;
- `results/xgd_test` does not exist yet.

`--check` runs every refusal and scores nothing.

**Order of the run:**
1. A1 is scored, and both arms' predictions are written with a record. All of this happens inside the held-out label seal and the StatsBomb guard.
2. Only then are the test labels read.
3. The rule is applied.

**Reruns.** The test runs once. A rerun is allowed only after a purely technical failure, with the same inputs, and by the owner's decision. Any deviation is reported in ticket 07 and not fixed silently.

## 12. Scope

"The result applies to the D3 detector and homography as audited in ticket 09, not to detector-derived structure in general."

## 13. Limitations

- **The margin δ = 0.005 is a post-hoc choice.** It was adopted after development and borrowed from xg-vendor. It is loose: about 1.5% of G-combo's log-loss, and several times the development difference.
- **The 135 evaluation matches are not fresh.** Their labels were seen in xg-vendor, though no model of this study was selected on them. The 37 gate matches were scored by G-combo in xg-vendor's retraining gate.
- **Retraining variability is not captured.** The match bootstrap resamples matches for fixed models; a new set of seeds would give a different ensemble.
- **Power is low** (section 9).
- **Seed-1 code differs in its hashes, not its behaviour.** A1's seed-1 runs were trained at ticket 06's code. That code carries ticket 03's post-run refactor (9b7eea9) and the label seal. Both were shown to leave the inputs and outputs unchanged: the sanitised table is rebuilt identically, and the seed-0 replays pass at the new code.

## 14. No test label before the test

No label of a held-out shot may be read before the test. This is enforced by `xgpoc/detector/seal.py`, under which pandas' readers hand over no label of a shot outside the training pool. The training of seed 1, the replay, the A1 ensemble calibration, the power computation and G-combo's test predictions all ran inside it. The test script reads labels once, after the predictions are written, outside the seal. Tests: `tests/test_xgd_seal.py`, `tests/test_xgd_test_rule.py`.
