Twoim zadaniem jest wytrenowanie 4 modeli widzenia komputerowego, a następnie ich inferencja na różnych materiałach wideo 

Pamiętaj, żeby używać środowiska uv python oraz akceleracji GPU do obliczeń, które ją wspierają 

## Ustalenia (2025-12-14)
- DETR: używamy **bazowego RF-DETR (bez segmentacji)**, jak w `DetectionModelType.RFDETR`.
- YOLO: trenujemy warianty **M**: `yolov8m.pt`, `yolo11m.pt`, `yolov12m.pt`.
- Wyniki zapisujemy do `results/project/raw/{nazwa_modelu}/...`:
  - `results/project/raw/{model}/training/...`
  - `results/project/raw/{model}/inference/{match}/...`
  - `results/project/raw/{model}/configs/{match}.yaml` (generowane do batch inferencji)
- Merge ze StatsBomb360: **odkładamy na później** (na teraz robimy tylko inferencję/pipeline detection).
- Keypoints: **zawsze włączone**, używamy lokalnego checkpointu z `config/default_config.yaml` (`keypoints.checkpoint_path`).
- Trening: max **200 epok** + **early stopping** (patience z configu).
- Inferencja: mierzymy czas inferencji modelu detekcji (per-frame) i zapisujemy do pliku JSON + CSV (agregacja).

## Trening 
musisz wytrenować 4 modele widzenia komputerowego [yolo8, yolo11, yolo12, rd-detr], każdy z nich w wariancie -M (Medium) i zapisać ich rezultaty w odpowiednich directories w folderze results. 

Przykładowe parametry treningu masz w plikach run_config/train_detect_... możesz się nimi zainspirować, ale pamiętaj, że tam trenowaliśmy modele X, a tutaj chcemy wytrenować modele M. 

Do treningu modeli użyj danych, które są dostępne w folderze data/datasets/football_yolo/ 

### Configi treningowe
- `run_config/train_detect_yolov8m.yaml`
- `run_config/train_detect_yolo11m.yaml`
- `run_config/train_detect_yolo12m.yaml`
- `run_config/train_detect_rfdetr_base_earlystop.yaml`

### Uruchomienie treningu
```bash
uv run python scripts/train_detector.py --config run_config/train_detect_yolov8m.yaml
uv run python scripts/train_detector.py --config run_config/train_detect_yolo11m.yaml
uv run python scripts/train_detector.py --config run_config/train_detect_yolo12m.yaml
uv run python scripts/train_detector.py --config run_config/train_detect_rfdetr_base_earlystop.yaml
```

## Inferencja 
Gdy będziesz miał już wytrenowane modele z zapisanymi checkpointami, musisz przeprowadzić dla każdego z nich inferencję / wykonanie pipeline detection, a następnie merge_pipeline_statsbomb. 
W każdym z folderów znajdujących się w /home/kuba/projects/ball-vision/data/20232024/ znajduje się plik first_5.mp4 na którym masz przeprowadzić run_detection oraz plik frames.parquet, w którym są dane ze statsbomb360, z którymi masz te wyniki porównać. W plikach first_5.mp4 jest nagranie pierwszych 5 minut każdego meczu i tylko dla nich potrzebujemy przeprowadzić porównanie z danymi statsbomb360. 

### Batch inferencja first_5.mp4 (na teraz bez merge)
Skrypt:
- `scripts/run_first5_inference_suite.py`

Uruchomienie:
```bash
uv run python scripts/run_first5_inference_suite.py
```

Wyniki:
- per-mecz/per-model: `results/project/raw/{model}/inference/{match}/...`
- czasy inferencji (JSON per run): `results/project/raw/{model}/inference/{match}/{model}_first_5_timing.json`
- agregacja czasów:
  - `results/project/raw/{model}/inference_timings.csv`
  - `results/project/numeric/inference_timings_all_models.csv`


Na koniec, rezultaty wszystkich plików wynikowych połącz w jeden duży, na którym będę mógł przeprowadzić całościową analizę

## Wyniki eksperymentu (first_5, 18 meczów)
Poniżej zebrane są rezultaty treningu, inferencji oraz porównania do danych StatsBomb360 (freeze-frame) dla 4 modeli: `yolov8m`, `yolo11m`, `yolo12m`, `rfdetr_base`. Źródła danych to artefakty w `results/project/*` (CSV/Parquet/PNG).

### 1) Trening detektorów (walidacja)
Metryki pochodzą z logów treningowych:
- YOLO: `results/project/raw/*/training/**/results.csv` (wybieramy epokę z najlepszym `metrics/mAP50-95(B)`).
- RF-DETR: `results/project/raw/rfdetr_base/training/results.json` (`class_map.valid`, klasa `all`).

| model | mAP@50:95 (val) | mAP@50 (val) | precision (val) | recall (val) | best epoch |
|---|---:|---:|---:|---:|---:|
| `yolov8m` | 0.482 | 0.813 | 0.848 | 0.782 | 81 |
| `yolo11m` | 0.487 | 0.810 | 0.827 | 0.797 | 71 |
| `yolo12m` | 0.489 | 0.822 | 0.839 | 0.831 | 144 |
| `rfdetr_base` | 0.452 | 0.823 | 0.874 | 0.690 | — |

Wykres porównawczy: `results/project/plots/first5/training_map_comparison.png`.

### 2) Szybkość inferencji (tylko detektor)
Agregacja po 18 meczach (`results/project/numeric/inference_timings_all_models.csv`), gdzie:
- `detection_time_avg_ms` to średni czas detekcji per-frame (ms),
- `detection_fps` to FPS liczone z tego samego pomiaru.

| model | avg ms / frame | median ms / frame | mean FPS | median FPS |
|---|---:|---:|---:|---:|
| `yolo11m` | 5.163 | 5.114 | 193.82 | 195.53 |
| `yolov8m` | 5.592 | 5.533 | 178.90 | 180.75 |
| `yolo12m` | 6.290 | 6.254 | 159.04 | 159.90 |
| `rfdetr_base` | 26.775 | 26.916 | 37.36 | 37.15 |

Wykres rozkładu FPS po meczach: `results/project/plots/first5/inference_fps_boxplot.png`.

### 3) Porównanie do StatsBomb360 (freeze-frame)
Ewaluacja opiera się o `results/project/raw/statsbomb_full/*.parquet` oraz pipeline CSV z inferencji (`results/project/raw/{model}/inference/{match}/*_pipelinedata_p1.csv`) i jest liczona przez `scripts/eval_statsbomb360.py`.

**Co mierzymy**
- `distance` [m]: odległość (w metrach boiska) między predykcją a najbliższym punktem StatsBomb (dla obiektów `player/goalkeeper/referee`).
- `coverage`: udział dopasowanych punktów StatsBomb (`matched_count/sb_count`) w oknie czasowym eventu.

**Wyniki (first_5, period=1, t<300s)**
Wyniki z próbek odległości: `results/project/numeric/first5/positional_error_summary_first5.csv` (ok. 50k dopasowań / model).

| model | median dist [m] | mean dist [m] | sd [m] |
|---|---:|---:|---:|
| `rfdetr_base` | 7.201 | 15.000 | 17.580 |
| `yolov8m` | 7.278 | 15.165 | 17.770 |
| `yolo11m` | 7.346 | 15.208 | 17.613 |
| `yolo12m` | 7.373 | 15.328 | 17.948 |

Wyniki per-event (first_5): `results/project/numeric/first5/statsbomb360_first5_event_summary.csv`.
- liczba eventów w klipie: 3314 (na 18 meczach),
- średnie `coverage_mean_all`: ~97.3–97.9%,
- średnie `det_count_mean`: ~22.8–23.7 obiektów / event.

Wykres: `results/project/plots/first5/statsbomb360_mean_dist_and_coverage.png`. Dodatkowo pełny ridgeplot rozkładów: `results/project/plots/statsbomb360_eval/positional_error_ridgeplot.png`.

### 4) Najważniejsze wnioski
- **Jakość treningowa (val)**: `yolo12m` ma najwyższe `mAP@50:95`, ale różnice między YOLO M są niewielkie; `rfdetr_base` ma niższe `mAP@50:95` przy porównywalnym `mAP@50`.
- **Wydajność**: `yolo11m` jest najszybszy (~194 FPS); `rfdetr_base` jest ~5× wolniejszy (~37 FPS).
- **Zgodność ze StatsBomb360 (first_5)**: wszystkie modele są bardzo zbliżone; `rfdetr_base` ma najniższą średnią odległość, a różnice są rzędu dziesiątych części metra w medianie.

### Artefakty wynikowe
- Podsumowania CSV: `results/project/numeric/first5/training_metrics_summary.csv`, `results/project/numeric/first5/inference_timings_summary.csv`, `results/project/numeric/first5/positional_error_summary_first5.csv`, `results/project/numeric/first5/statsbomb360_first5_event_summary.csv`
- Wykresy: `results/project/plots/first5/training_map_comparison.png`, `results/project/plots/first5/inference_fps_boxplot.png`, `results/project/plots/first5/statsbomb360_mean_dist_and_coverage.png`
