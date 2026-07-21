# Video xG MVP Report

## Decision

The production MVP integrates the simplest working baseline for each stage:

1. Ball reconstruction: linear interpolation with optional outlier rejection.
2. Shot detection: SoccerNet metadata baseline for labelled clips, kinematic
   peak detector for unlabeled clips.
3. xG: deterministic geometry model.

These choices are intentionally conservative. They create stable APIs and a
repeatable experiment runner without introducing heavyweight dependencies or
training jobs before the feedback loop is in place.

## Known Limits

- The geometry xG model is not calibrated to StatsBomb yet.
- Team direction is inferred from nearest goal unless `attacking_goal_x` is set.
- Defender pressure is neutral when pitch projection is unavailable.
- SoccerNet metadata mode should be treated as a weak-label baseline, not a
  deployable shot detector.

## Next Experiment Gate

Before replacing any baseline, run the candidate method against the same API and
compare it with the current smoke configs. Promote only if it improves the stage
metric without regressing the fast tests.
