# Video xG Research Notes

This MVP implements deterministic baselines that make the experiment runnable in
the production `src/tactifoot_vision` package.

## Stage 1: Ball Reconstruction

Implemented:
- Linear image-space and pitch-space interpolation.
- Edge extrapolation with lower confidence.
- Optional single-frame outlier rejection using a maximum pixel speed threshold.

Deferred research candidates:
- Kalman/RTS smoothing.
- Ball-only ByteTrack association.
- TrackNet-style heatmap model.
- SAM2 ball-only propagation.
- Physics/contact constrained smoothing.

## Stage 2: Shot Detection

Implemented:
- SoccerNet metadata baseline using `actionPosition` and `actionClass`.
- Kinematic detector based on peak ball speed.

Deferred research candidates:
- Contact detector using ball-player proximity.
- Context classifier.
- SoccerNet Ball Action Spotting model adaptation.
- Candidate-generator plus ranker ensemble.

## Stage 3: xG

Implemented:
- Geometry estimator using distance, angle, centrality, ball speed, goalkeeper
  distance and player pressure when available.
- Probability metrics: Brier, log loss, ECE and aggregate MAE.

Deferred research candidates:
- StatsBomb-calibrated logistic regression.
- Random forest / gradient boosting with freeze-frame features.
- MLP with calibration.
- Set Transformer / GNN over ball-player-goal geometry.
- Distillation against `statsbomb_xg`.
