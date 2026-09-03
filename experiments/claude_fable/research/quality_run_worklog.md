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
| B2 | Krzywa uczenia 25/50/75/100% | D-L głowa: 0.737 / 0.748 / 0.757 / 0.769 (3 seedy); G-L/4: 0.740 / 0.766 / 0.771 / 0.780 (1 seed) — **obie nadal rosną** | `results/v12_lc/*_learning_curve.csv` |
| C1 | Augmentacja (RRC, mixup), augmentacja czasowa, multi-task „celny”, LLRD — screening G | RRC ✓ (valid cal LL 0.3787 vs 0.3809), multi-task ✓ (0.3792; test 0.780/0.3347), mixup ? (valid ✓ 0.3785, test ✗), offset czasowy ✗, dl_mt ✗; LLRD w toku | `results/v12/g_*` |
| C2 | Najlepsza konfiguracja G → CV; to samo dla C | zrobione: g_combo (10 modeli) 0.7923 / 0.3242, g_combo_mix 0.7909 / 0.3245, c_combo 0.7658 / 0.3367, dl_mask (10) 0.7740 / 0.3319 | `results/v12_cv/` |
| C3 | Backbone: DINOv2-L z rejestrami / DINOv3 na zamrożonej głowie | zrobione, oba gorsze: DINOv3-L 0.738–0.752, DINOv2-L-reg 0.756–0.760 vs DINOv2-L 0.768–0.769 (3 seedy). Backbone zostaje. | `scripts/extract_features.py` |
| D | Finalny ensemble, podgrupy, istotność, raport, README | zrobione: **0.7933 [0.768, 0.816] / 0.3232 / BSS 16.3%**; vs stary ΔAUC +0.011 (P 0.98), Δll +0.007 (P 1.00) | `results_reports/v12_quality_REPORT.md`, `results/v12_cv_ensemble/`, `results/v12_report/` |

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
- 2026-09-03 13:30: CV po gospodarzu, lekcy członkowie (5 foldów × 1 seed, kalibracja Platt na OOF 400 meczów, routing karnych), test: D-L głowa 0.7733 / 0.3323 (vs 0.768 / 0.337 na oficjalnym splicie → fold-ensembling działa), C-combo (RRC+MT) 0.7658 / 0.3367 vs C 0.7637 / 0.3377 (OOF LL 0.3659 vs 0.3672 → RRC+MT pomaga też C), GBM 0.644 / 0.369. Stack bez G: 0.7770 / 0.3308. LLRD seed 0: valid cal LL 0.3814 ≈ baza, test 0.779 / 0.339, szybsze przeuczenie — wstępnie neutralne.
- 2026-09-03 14:28: krzywa uczenia G-L/4 (1 seed): 75/150/225/300 meczów → AUC 0.740 / 0.766 / 0.771 / 0.780, log-loss 0.350 / 0.340 / 0.340 / 0.340. Ostatni krok +0.009 AUC. Wniosek bramki P5: dane nie są nasycone dla żadnej z dwóch ścieżek → po tej rundzie priorytetem jest pozyskanie meczów. CV dla g_combo/g_combo_mix wystartowało dopiero teraz (łańcuch czekał na cały poprzedni proces, w tym krzywą G).
- 2026-09-03 15:35: wstępny ensemble CV (OOF 400 meczów, Platt, routing karnych): **c_combo + g_combo + gbm → test AUC 0.7933 [0.768, 0.816], log-loss 0.3238**, wagi [0.18, 0.76, 0.32]; g_combo sam (5 foldów): 0.7919 / 0.3251. Czekają: g_combo_mix (CV), drugi seed g_combo.
- 2026-09-03 16:40: CV g_combo_mix gotowe: OOF LL 0.3496 (lepiej niż g_combo 0.3515), test 0.7909 / 0.3245 (remis). Stack po OOF: g_combo + g_combo_mix + gbm → 0.7914 / 0.3242; c_combo + g_combo + gbm → 0.7933 / 0.3238 — wszystko w szumie ±0.002. Drugi seed CV g_combo w toku (koniec ~17:45); potem finalny ensemble z max-k 4.
- 2026-09-03 17:55: **finał.** Ensemble C-combo + G-combo + G-combo-mix + GBM (wagi 0.11/0.48/0.39/0.27, OOF LL 0.3481): test AUC 0.7933 [0.768, 0.816], log-loss 0.3232, BSS 16.3%, ECE 0.016, średnie xG 0.1385. Istotność vs stary ensemble na v1.2: ΔAUC +0.0112 [+0.0004, +0.0222] P=0.98, Δll +0.0070 [+0.0024, +0.0116] P=1.00. Podgrupy: 0–6 m 0.855 (było 0.801), 11–16 m 0.664 (0.643), karne xG 0.68 vs 0.71 obs., nieznane stadiony 0.774 vs znane 0.797. Raport: `results_reports/v12_quality_REPORT.md`.
