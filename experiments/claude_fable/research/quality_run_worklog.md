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
| A7 | Nowa baza na v1.2 (C, D-L, G-L/4, GBM; 3 seedy) + ensemble | zrobione: ensemble C+G+GBM **0.7875 AUC / 0.3265 LL / BSS 15.2%** (stare modele na v1.2: 0.782 / 0.330) | `results/v12/`, `results/v12_ensemble/`, `results/v12_report/` |
| B1 | Foldy po gospodarzu, runner CV, ensemble na OOF | kod gotowy (foldy: 5 × 80 meczów, 22–23 gospodarzy); czeka na GPU | `scripts/run_cv.py`, `cache/folds_v12_home5.csv` |
| B2 | Krzywa uczenia 25/50/75/100% | D-L głowa: 0.737 / 0.748 / 0.757 / 0.769 AUC (3 seedy, std ≤0.002) — **nie wypłaszcza się**; G w kolejce | `results/v12_lc/dl_mask_learning_curve.csv` |
| C1 | Augmentacja (RRC, mixup), augmentacja czasowa, multi-task „celny”, LLRD — screening G | RRC ✓ (valid cal LL 0.3787 vs 0.3809), multi-task ✓ (0.3792; test 0.780/0.3347), mixup ? (valid ✓ 0.3785, test ✗), offset czasowy ✗, dl_mt ✗; LLRD w toku | `results/v12/g_*` |
| C2 | Najlepsza konfiguracja G → CV; to samo dla C | w toku: CV lekkich członków (gbm, dl_mask, c_mask, c_combo) + zakolejkowane CV g_combo (RRC+MT) i g_combo_mix | `results/v12_cv/` |
| C3 | Backbone: DINOv2-L z rejestrami / DINOv3 na zamrożonej głowie | zrobione, oba gorsze: DINOv3-L 0.738–0.752, DINOv2-L-reg 0.756–0.760 vs DINOv2-L 0.768–0.769 (3 seedy). Backbone zostaje. | `scripts/extract_features.py` |
| D | Finalny ensemble, podgrupy, istotność, raport, README | — | `results_reports/v12_quality_REPORT.md` |

## Dziennik

- 2026-09-02: start. GPU wolne (RTX 4070 Ti SUPER 16 GB), 290 GB dysku, torch 2.6.0, timm 1.0.20.
- 2026-09-02: index v1.2 zbudowany (`index_v1_2.parquet`, changelog w dataset card). Sanity głowy D-L z nowym trenerem na v1.2: AUC 0.770 / LL 0.337, najlepsza epoka 13/25 (cykl 25 epok, EMA 0.999). 38 testów zielonych. Start kolejki bazowej `run_v12.py` (gbm, dl_mask, c_mask, g_l4 × 3 seedy).
- 2026-09-03: pierwszy przebieg kolejki przerwany po awarii: przy epoce 0 z EMA logity są niemal stałe, LBFGS temperatury uciekł do 0 → NaN. Poprawka: klamra T∈[0.05, 20], rozgrzewka EMA (decay rośnie od 0.1), nan_to_num w ocenie epoki; testy. Wyniki sprzed poprawki skasowane, kolejka wystartowana od nowa (gbm, dl_mask, dl3_mask, dlreg_mask, c_mask, g_l4 × 3 seedy). Wynik cząstkowy sprzed poprawki: D-L 0.766–0.768 AUC (3 seedy), C 0.758 (seed 0), DINOv3-L 0.740 (seed 0, gorzej niż DINOv2-L).
- 2026-09-03 10:40: **nowa baza na v1.2** (oficjalny split, 3 seedy, trener po poprawce, Platt + korekta priora do 13.6% + routing karnych 0.681):

  | model | AUC | log-loss | BSS | uwagi |
  |---|---|---|---|---|
  | stary ensemble (v1.1 modele) przeliczony na v1.2 | 0.7823 | 0.3302 | 14.4% | punkt odniesienia |
  | D-L głowa (3 seedy) | 0.7692 / 0.7677 / 0.7685 | 0.337–0.338 | 12% | stara D-L na v1.2: 0.7563 / 0.3453 → ΔAUC +0.017 [+0.007, +0.027], P=1.0 |
  | C (3 seedy) | 0.750 / 0.760 / 0.757 | 0.342–0.345 | 10% | bez zmiany (C miało już domknięty cykl) |
  | G-L/4 (3 seedy, ens) | 0.7854 | 0.3297 | 14.9% | stare G-L/4 na v1.2: 0.7779 / 0.3363 → Δll +0.0065 [+0.0012, +0.0119], P=0.996 |
  | **ensemble C+G+GBM (wybór na valid CV)** | **0.7875 [0.761, 0.812]** | **0.3265** | **15.2%** | vs stary: ΔAUC +0.005 [−0.004, +0.014] P=0.88; Δll +0.0037 [−0.0005, +0.0077] P=0.96 |

  Średnie xG na teście 0.1331 przy częstości 0.1328 (przed korektą priora: 0.143). Karne: xG 0.68 vs 0.706 obserwowane
  (log-loss karnych 0.607 vs 0.632). Nieznane stadiony: 0.770 vs 0.791 znane (luka 2.2 pp; wcześniej 3 pp). Pas 11–16 m
  nadal najsłabszy (0.645). Historie G: najlepsza epoka 2 z 5 w każdym seedzie, train loss 0.31 w epoce 4 → regularyzacja
  to właściwy następny ruch (screening w toku: dl_mt, g_rrc, g_time, g_mt, g_mixup, g_llrd; potem krzywe uczenia).
- 2026-09-03 11:05: krzywa uczenia D-L (oficjalny split, 3 seedy): 75/150/225/300 meczów → AUC 0.737/0.748/0.757/0.769, log-loss 0.350/0.346/0.342/0.338. Przyrost na ostatnim kroku (+0.011) nie mniejszy niż na poprzednich → dane nadal są wąskim gardłem (bramka decyzyjna P5 z roadmapy: priorytet dla pozyskania danych po tej rundzie). Multi-task na głowie: bez zysku.
- 2026-09-03 12:55: screening G na oficjalnym splicie (2 seedy; baza g_l4 3 seedy: valid cal LL 0.3809, test 0.7771 / 0.3374):

  | wariant | valid cal LL | test AUC | test LL | werdykt |
  |---|---|---|---|---|
  | RRC 0.85–1.0 | 0.3787 | 0.7772 | 0.3358 | ✓ |
  | offset czasowy {−800,−400,0} | 0.3808 | 0.7705 (0.777/0.764) | 0.3362 | ✗ (brak zysku, duża wariancja) |
  | multi-task „celny” λ=0.5 | 0.3792 | 0.7797 | 0.3347 | ✓ |
  | mixup 0.2 | 0.3785 | 0.7743 | 0.3382 | ? (valid lepszy, test gorszy) → rozstrzyga CV |
  | multi-task na zamrożonej głowie D-L | 0.383 | 0.766 | 0.338 | ✗ |

  Konfiguracje do CV: `g_combo` = RRC + multi-task; `g_combo_mix` = + mixup; `c_combo` = C z RRC + multi-task.
