# Video-Only xG End-to-End MVP

## Cel

Pipeline liczy xG dla meczu wyłącznie z wideo w runtime:

1. buduje globalną oś czasu dla `part1.mp4` i `part2.mp4`,
2. próbkuje klatki w `scan_fps`,
3. uruchamia detekcję i tracking,
4. rekonstruuje trajektorię piłki,
5. wykrywa kandydatów strzałów z kontaktu i kinematyki,
6. wyciąga cechy video-only,
7. porównuje metody xG,
8. ładuje StatsBomb dopiero w ewaluacji.

## Implementacja MVP

Główne wejście produkcyjne:

```bash
uv run tactifoot video-xg end-to-end \
  --config configs/experiments/video_xg_end_to_end_fa_wsl.yaml
```

Runner: `tactifoot_vision.video_xg.VideoOnlyXgEndToEndRunner`.

Notebook: `notebooks/video_only_xg_experiment.ipynb` tylko wywołuje runnera i czyta artefakty.

## Metody xG

Aktualnie porównywane są trzy metody:

- `video_geometry`: geometria strzału z pozycji piłki względem bramki,
- `video_freeze_context`: geometria plus dystans najbliższego gracza, bramkarka i obrońcy w stożku,
- `video_kinematic_context`: `video_freeze_context` plus prędkość piłki i kierunek ruchu do bramki.

## Artefakty

Każdy etap zapisuje checkpoint w katalogu runu:

- `00_video_timeline.json`
- `01_sampled_frames.parquet`
- `02_detections.parquet`
- `03_tracks.parquet`
- `04_homographies.parquet`
- `04_projection_quality.csv`
- `05_ball_trajectory.parquet`
- `06_shot_candidates.parquet`
- `07_refined_shots.parquet`
- `08_video_features.csv`
- `08_video_features_model.csv`
- `09_predictions.csv`
- `10_per_shot_eval.csv`
- `10_method_metrics.csv`
- `10_metrics.json`
- `final_report.md`

Obsługiwane są flagi `--stop-after`, `--resume-from` i `--force-stage`.

## Ograniczenia

Homografia ma obecnie tryb `degraded_image_normalized`, dopóki nie zostanie podłączony właściwy backend kalibracji boiska. Rekonstrukcja piłki używa interpolacji i wygładzania w wariancie MVP opisanym jako `kalman_rts`, ale bez pełnej macierzy kowariancji filtra RTS. Refinement do `30 FPS` jest przygotowany jako osobny etap artefaktów, lecz w MVP zachowuje kandydatów ze skanu.

StatsBomb/SoccerNet nie są inputem runtime. Dane referencyjne są używane dopiero po zapisaniu predykcji, w etapie ewaluacji.
