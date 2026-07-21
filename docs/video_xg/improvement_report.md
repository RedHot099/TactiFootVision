# Video-Only xG Improvement Ablation

## Zakres

Ten raport opisuje szybki eksperyment poprawy pipeline'u video-only xG na zapisanym
pełnym runie `1 FPS`.

Runtime inferencji pozostaje video-only. Referencja StatsBomb jest ładowana dopiero
po zapisaniu kandydatów i predykcji, do strojenia wariantów oraz ewaluacji.

## Wejścia

- Baseline run: `/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_yolo11m_1fps_20260526_131429`
- Reference shots: `23`
- StatsBomb total xG: `2.5321`
- Baseline candidates: `80`
- Baseline hit@2s: `0.0435`
- Baseline best total xG: `1.0729` dla `video_kinematic_context`
- Baseline missing-center fallback: `61/80`

## Uruchomienie

```bash
uv run tactifoot video-xg ablate \
  --config configs/experiments/video_xg_improvement_fa_wsl.yaml
```

Artefakty zapisano w:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_improvement/
```

## Wyniki

Najlepszy wariant wg gated composite:

- ball: `optical_flow_template`
- shot ranking: `learned_temporal`
- projection: `line_box_heuristic`
- xG calibration: `coefficient_fit`

Metryki zwycięzcy:

- hit@2s: `0.7826`
- hit@1s: `0.3913`
- precision@2s: `0.2250`
- total predicted xG: `2.1070`
- total xG error: `-0.4251`
- composite score: `0.6351`

Najważniejsze poprawy względem baseline:

- shot detection hit@2s: `0.0435 -> 0.7826`
- missing-center fallback przy cechach zwycięskiego wariantu: `0.7625 -> 0.0000`
- total xG error najlepszego wariantu: `-1.4592 -> -0.4251`

## Caveat Metodologiczny

`learned_temporal` i `coefficient_fit` używają referencji StatsBomb po wygenerowaniu
kandydatów, więc wynik jest ablation/oracle-tuning na tym meczu, a nie niezależną
walidacją publikacyjną. Do publikacji potrzebny jest split meczowy:

- train/calibration matches do strojenia progów i kalibratorów,
- validation matches do wyboru wariantu,
- test matches trzymane wyłącznie do końcowej ewaluacji.

## Następny Krok

Zwycięskie warianty powinny zostać uruchomione na docelowym skanie `10 FPS` z
chunked detection i bez ponownego strojenia na meczu testowym.

## 10 FPS Winner Run

Uruchomienie:

```bash
uv run tactifoot video-xg end-to-end \
  --config configs/experiments/video_xg_end_to_end_fa_wsl_winners_10fps.yaml \
  --resume-from 06_shot_candidates
```

Artefakty:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_winners_10fps_20260526/
```

Wyniki detekcji strzałów:

- predicted shots: `80`
- reference shots: `23`
- hit@0.5s: `0.1739`
- hit@1s: `0.5217`
- hit@2s: `0.8261`
- temporal MAE: `1.7707s`

Jakość cech:

- missing-center fallback ratio: `0.0000`
- feature sources: `41 observed`, `38 kalman_rts_interpolated`, `1 optical_flow_template`
- projection status: `80 line_box_heuristic`

Porównanie xG:

| Run | Method | All-candidate total xG | Error vs StatsBomb total | Matched total xG | Matched total error |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline 1 FPS | video_kinematic_context | `1.0729` | `-1.4592` | `0.3063` | `-2.2259` |
| winners 10 FPS | video_geometry | `2.2554` | `-0.2767` | `0.6446` | `-1.8875` |
| winners 10 FPS | video_freeze_context | `2.9584` | `+0.4263` | `0.8644` | `-1.6677` |
| winners 10 FPS | video_kinematic_context | `3.4910` | `+0.9588` | `1.0748` | `-1.4574` |
| winners 10 FPS | coefficient_fit | `9.9206` | `+7.3884` | `2.3770` | `-0.1551` |

Interpretacja:

- Największa realna poprawa jest w detekcji czasu strzału: `hit@2s` rośnie z
  `0.0435` do `0.8261`, a temporal MAE spada z `25.86s` do `1.77s`.
- Fallback środka obrazu został usunięty z cech zwycięskiego runu.
- `coefficient_fit` bardzo dobrze dopasowuje xG dla strzałów sparowanych z GT,
  ale zawyża sumę po wszystkich 80 kandydatach, bo false positives nadal dostają
  istotne xG. Nie powinien być traktowany jako operacyjny agregat bez lepszego
  filtrowania kandydatów.
- Dla operacyjnej sumy po wszystkich wykrytych kandydatach najlepszy w tym runie
  jest `video_geometry`: `2.2554` vs StatsBomb `2.5321`, error `-0.2767`.
