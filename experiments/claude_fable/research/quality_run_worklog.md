# Worklog: runda poprawy jakości po review (start 2026-09-02)

Plan i uzasadnienie: `review_2026-09-02.md` §5. Zasada: najpierw poprawność etykiet
i przyrządu, potem nowa baza, potem architektura. Wszystkie nowe wyniki liczone na
index v1.2; stare tabele (v1.1) nieporównywalne.

## Status

| # | Krok | Stan | Artefakt |
|---|---|---|---|
| A1 | Parser: gol → najbliższy strzał, karny okno [−1 s, +2 s], testy | zrobione | `xgpoc/data/soccernet_labels.py`, `tests/test_soccernet_labels.py` |
| A2 | Index v1.2 + changelog w dataset card | zrobione (194 etykiety, 52 w teście; karnych 149) | `scripts/relabel_index.py` |
| A3 | Wspólne ścieżki (`xgpoc/paths.py`) | zrobione | |
| A4 | Trener: cykl = budżet, warm-up, selekcja po skalibrowanym LL, EMA, LR w historii, mixup, multi-task, LLRD | zrobione | `xgpoc/models/common.py` |
| A5 | Datasety: guard na klatki po strzale, losowy offset w treningu, RRC, cel pomocniczy | zrobione | `xgpoc/data/datasets.py` |
| A6 | Kalibracja: Platt + korekta priora, hierarchia karnych (w `build_ensemble.py`) | zrobione | `xgpoc/eval/calibration.py` |
| A7 | Nowa baza na v1.2 (C, D-L, G-L/4, GBM; 3 seedy) + ensemble | w toku (kolejka `run_v12.py`, log `results/v12.log`) | `results/v12/` |
| B1 | Foldy po gospodarzu, runner CV, ensemble na OOF | kod gotowy (foldy: 5 × 80 meczów, 22–23 gospodarzy); czeka na GPU | `scripts/run_cv.py`, `cache/folds_v12_home5.csv` |
| B2 | Krzywa uczenia 25/50/75/100% | kod gotowy; czeka na GPU | `scripts/run_learning_curve.py` |
| C1 | Augmentacja (RRC, mixup), augmentacja czasowa, multi-task „celny”, LLRD — screening G | — | |
| C2 | Najlepsza konfiguracja G → CV; to samo dla C | — | |
| C3 | Backbone: DINOv2-L z rejestrami / DINOv3 na zamrożonej głowie | cechy zbudowane (10 GB + 13 GB); głowy w kolejce | `scripts/extract_features.py` |
| D | Finalny ensemble, podgrupy, istotność, raport, README | — | `results_reports/v12_quality_REPORT.md` |

## Dziennik

- 2026-09-02: start. GPU wolne (RTX 4070 Ti SUPER 16 GB), 290 GB dysku, torch 2.6.0, timm 1.0.20.
- 2026-09-02: index v1.2 zbudowany (`index_v1_2.parquet`, changelog w dataset card). Sanity głowy D-L z nowym trenerem na v1.2: AUC 0.770 / LL 0.337, najlepsza epoka 13/25 (cykl 25 epok, EMA 0.999). 38 testów zielonych. Start kolejki bazowej `run_v12.py` (gbm, dl_mask, c_mask, g_l4 × 3 seedy).
- 2026-09-03: pierwszy przebieg kolejki przerwany po awarii: przy epoce 0 z EMA logity są niemal stałe, LBFGS temperatury uciekł do 0 → NaN. Poprawka: klamra T∈[0.05, 20], rozgrzewka EMA (decay rośnie od 0.1), nan_to_num w ocenie epoki; testy. Wyniki sprzed poprawki skasowane, kolejka wystartowana od nowa (gbm, dl_mask, dl3_mask, dlreg_mask, c_mask, g_l4 × 3 seedy). Wynik cząstkowy sprzed poprawki: D-L 0.766–0.768 AUC (3 seedy), C 0.758 (seed 0), DINOv3-L 0.740 (seed 0, gorzej niż DINOv2-L).
