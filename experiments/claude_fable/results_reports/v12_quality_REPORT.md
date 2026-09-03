# Runda jakości po review: index v1.2, naprawiony trener, CV po gospodarzu, regularyzacja

Data: 2026-09-03. Punkt wyjścia: `research/review_2026-09-02.md` (diagnoza) i
`research/quality_run_worklog.md` (dziennik przebiegów). Test: 2 229 strzałów / 100 meczów,
etykiety **v1.2**. Wszystkie porównania między modelami sparowane, bootstrap po meczach
(2 000 losowań); wybór składu ensemble'u wyłącznie na predykcjach out-of-fold.

## 1. Wynik w jednym akapicie

Trzy usterki z review zostały naprawione (atrybucja gola do najbliższego strzału,
harmonogram i kryterium selekcji trenera, prior walidacji w kalibracji), a na tak
naprawionym przyrządzie dołożono to, co przeszło screening: crop augmentację i głowę
pomocniczą „strzał celny" w fine-tuningu DINOv2-L, walidację krzyżową grupowaną po
drużynie gospodarzy z uśrednianiem modeli foldowych i kalibracją na 400 meczach oraz
routing karnych. **Finalny ensemble: AUC 0.793 [0.768, 0.816], log-loss 0.323 [0.303, 0.347], BSS 16.3%, ECE 0.016**
(patrz §7) wobec 0.782 / 0.330 dla poprzedniego ensemble'u przeliczonego na te same
etykiety — różnica istotna (ΔAUC +0.011 [+0.000, +0.022], P = 0.98; Δlog-loss +0.007 [+0.002, +0.012], P = 1.00).
Względem liczby raportowanej przed review (0.758 / 0.340 na etykietach v1.1) większość
przyrostu to poprawność pomiaru, nie modelu; uczciwy rozkład jest w §2. Krzywe uczenia
obu ścieżek nadal rosną, więc kolejnym krokiem po tej rundzie jest pozyskanie meczów,
nie architektura (§6).

## 2. Skąd biorą się liczby: rozkład przyrostu

| Etap | AUC | log-loss | BSS | co się zmieniło |
|---|---|---|---|---|
| Raport przedmerge'owy (etykiety v1.1) | 0.758 | 0.340 | 11.7% | — |
| Te same predykcje na etykietach v1.2 | 0.782 | 0.330 | 14.4% | **poprawność etykiet testowych** (52 odwrócone) |
| Nowy trener, oficjalny split, ensemble C+G+GBM | 0.7875 | 0.3265 | 15.2% | cykl LR, selekcja po skalibrowanym LL, EMA, Platt + prior, routing karnych |
| CV po gospodarzu, G-combo (RRC + multi-task), 5 foldów × 2 seedy | 0.792 | 0.324 | 16.1% | fold-ensembling, kalibracja na OOF, regularyzacja |
| **Finalny ensemble** (§7) | **0.793** | **0.323** | **16.3%** | stack na OOF (4 członków) |

Etykiety v1.2: gol przypisany do **najbliższego** poprzedzającego strzału tej samej
drużyny (v1.1: do najwcześniejszego w oknie 6 s, co w 100 łańcuchach dobitek odwracało
etykiety); 194 zmienione etykiety (train/valid/test 108/34/52), „niecelne z golem" 29 → 8;
flaga karnego tylko dla strzału najbliższego etykiecie `Penalty` (149 karnych, 23 zmiany
flag). Changelog w `football_data/xg_shots_full/dataset_card.md`.

## 3. Nowa baza na oficjalnym splicie (v1.2, 3 seedy)

Trener po poprawce: cykl one-cycle o długości równej budżetowi epok (głowy 25, C 8,
G 5) z rozgrzewką, selekcja epoki po log-lossie walidacyjnym **po temperaturze**, EMA
wag (z rozgrzewką), klamra na temperaturę. Kalibracja: Platt na walidacji + przesunięcie
priora do łącznej częstości train+valid (13.6%) + routing karnych (0.681 z priora
Beta(0.76, n=40) i 84 karnych treningowych).

| Model | AUC (seedy) | log-loss | stary model na v1.2 | Δ (sparowane) |
|---|---|---|---|---|
| D-L głowa (zamrożony DINOv2-L) | 0.769 / 0.768 / 0.769 | 0.337–0.338 | 0.756 / 0.345 | ΔAUC +0.017 [+0.007, +0.027], Δll +0.013, P = 1.00 |
| C ConvNeXt-T | 0.750 / 0.760 / 0.757 | 0.342–0.345 | 0.754 / 0.345 | bez zmian (C miało już domknięty cykl) |
| G fine-tune L/4 (ensemble seedów) | 0.785 | 0.330 | 0.778 / 0.336 | Δll +0.0065 [+0.0012, +0.0119], P = 0.996 |
| **Ensemble C + G + GBM** (wybór na valid CV) | **0.7875 [0.761, 0.812]** | **0.3265** | 0.782 / 0.330 | ΔAUC +0.005 [−0.004, +0.014]; Δll +0.004 [−0.001, +0.008] |

Średnie xG na teście 0.1331 przy częstości 0.1328 (przed korektą priora modele
obrazowe dawały 0.143). Karne: xG 0.68 przy 0.71 obserwowanych (log-loss karnych
0.607 vs 0.632). Historie G: najlepsza epoka 2 z 5 w każdym seedzie, train loss 0.31
w epoce 4 — zapamiętywanie nadal widoczne, stąd screening regularyzacji (§5).

## 4. Backbone: DINOv2-L zostaje

Zamrożona głowa, te same ustawienia, 3 seedy, maska nakładki:

| Backbone | tokeny | AUC (seedy) | log-loss |
|---|---|---|---|
| **DINOv2-L/14** | 1 + 576 | **0.769 / 0.768 / 0.769** | **0.337** |
| DINOv2-L/14 z rejestrami | 5 + 576 | 0.760 / 0.756 / 0.759 | 0.342 |
| DINOv3-L/16 | 5 + 448 | 0.749 / 0.752 / 0.738 | 0.345–0.348 |

Oba nowsze warianty są gorsze o 0.01–0.02 AUC przy tej rozdzielczości wejścia; nie
testowano ich fine-tuningu. Cechy pozostają w `cache/full_v1/` na wypadek dalszych prób.

## 5. Screening regularyzacji fine-tuningu G (oficjalny split, 2 seedy)

Baza `g_l4` (3 seedy): walidacyjny log-loss po temperaturze 0.3809, test 0.777 / 0.337.
Kryterium ustalone z góry: wariant zostaje, jeśli nie pogarsza walidacyjnego log-lossu.

| Wariant | valid cal LL | test AUC | test LL | werdykt |
|---|---|---|---|---|
| RandomResizedCrop 0.85–1.0 (z zachowaniem proporcji) | 0.3787 | 0.777 | 0.336 | ✓ |
| głowa pomocnicza „strzał celny" (λ = 0.5) | 0.3792 | 0.780 | 0.335 | ✓ |
| mixup α = 0.2 | 0.3785 | 0.774 | 0.338 | ✓ na walidacji, remis na teście → obie wersje do CV |
| losowa klatka z {−800, −400, 0} ms w treningu | 0.3808 | 0.771 (0.777 / 0.764) | 0.336 | ✗ |
| 8 bloków + warstwowy zanik LR 0.7 | 0.3811 | 0.779 | 0.340 | ✗ (neutralne, gorszy LL, szybsze przeuczenie) |
| „strzał celny" na zamrożonej głowie D-L (3 seedy) | 0.383 | 0.766 | 0.338 | ✗ |

Konfiguracja łączona `g_combo` = RRC + głowa „celny"; `g_combo_mix` = + mixup;
`c_combo` = C z RRC + głową „celny". Ciekawostka z historii: RRC przesuwa najlepszą
epokę z 2 na 3–4, czyli robi dokładnie to, co miał robić — opóźnia zapamiętywanie.

## 6. Krzywe uczenia: dane nie są nasycone

Zagnieżdżone podzbiory meczów treningowych, walidacja i test oficjalne.

| mecze treningowe | D-L głowa (3 seedy) AUC | G-L/4 (1 seed) AUC |
|---|---|---|
| 75 (25%) | 0.737 | 0.740 |
| 150 (50%) | 0.748 | 0.766 |
| 225 (75%) | 0.757 | 0.771 |
| 300 (100%) | 0.769 | 0.780 |

Ostatni krok (+75 meczów) daje +0.011 dla głowy i +0.009 dla fine-tuningu — nie
mniej niż poprzednie. W skali log-liniowej podwojenie liczby meczów jest warte ok.
+0.02 AUC. To jest bramka decyzyjna P5 z roadmapy i wypada jednoznacznie: **po tej
rundzie priorytetem jest pozyskanie meczów** (pozostałe sezony SoccerNet, SoccerReplay-1988),
a nie kolejne zmiany architektury.

## 7. CV po gospodarzu i finalny ensemble

5 foldów po 80 meczów z train+valid, grupowanie po drużynie gospodarzy (22–23 gospodarzy
na fold; `cache/folds_v12_home5.csv`). Każdy członek = średnia logitów modeli foldowych
(każdy uczony na 320 meczach); kalibracja Platt i wagi stacku dopasowane na 400 meczach
out-of-fold (częstość 13.6%, więc bez korekty priora); routing karnych; test dotknięty raz.

| Członek (CV) | modeli | OOF log-loss | test AUC [95% CI] | test log-loss |
|---|---|---|---|---|
| G-combo (RRC + celny) | 5 foldów × 2 seedy | 0.3493 | 0.792 [0.768, 0.817] | 0.324 |
| G-combo-mix (+ mixup) | 5 | 0.3496 | 0.791 [0.766, 0.816] | 0.3245 |
| D-L głowa | 5 × 2 | 0.3606 | 0.774 [0.747, 0.797] | 0.332 |
| C-combo | 5 | 0.3659 | 0.766 [0.743, 0.789] | 0.337 |
| C | 5 | 0.3672 | 0.764 [0.740, 0.786] | 0.338 |
| GBM pozycyjny (+ is_penalty, cechy P0) | 5 × 2 | 0.3817 | 0.647 [0.611, 0.680] | 0.369 |

Pojedynczy model foldowy G-combo ma na teście 0.775–0.791 AUC; średnia 5 foldów 0.792,
czyli fold-ensembling daje ok. +0.008 wobec pojedynczego modelu.

**Finalny ensemble** (wybór po OOF log-lossie 0.3481, do 4 członków): C-combo + G-combo +
G-combo-mix + GBM, wagi [0.11, 0.48, 0.39, 0.27] → **test AUC 0.7933 [0.768, 0.816],
log-loss 0.3232 [0.303, 0.347], BSS 16.3%, ECE 0.016**, średnie xG 0.1385 (częstość 0.1328).
Predykcje: `results/v12_cv_ensemble/best_ensemble_test_predictions.csv`. Dziesięć czołowych
składów mieści się w 0.792–0.794 AUC i 0.3232–0.3241 log-lossu — wybór między nimi jest
poniżej rozdzielczości pomiaru; wybrany skład jest po prostu najlepszy na OOF.

Testy sparowane (2 000 losowań po meczach):

| Porównanie | ΔAUC [95% CI] | P | Δlog-loss [95% CI] | P |
|---|---|---|---|---|
| finalny vs stary ensemble (oba na v1.2) | **+0.0112 [+0.0004, +0.0222]** | 0.981 | **+0.0070 [+0.0024, +0.0116]** | 1.000 |
| finalny vs ensemble z oficjalnego splitu (§3) | +0.0060 [−0.0024, +0.0144] | 0.922 | **+0.0034 [+0.0001, +0.0067]** | 0.979 |
| finalny vs sam G-combo (10 modeli) | +0.0010 [−0.0029, +0.0049] | 0.686 | +0.0010 [−0.0006, +0.0026] | 0.887 |
| G-combo vs D-L głowa (obie CV) | **+0.0183 [+0.0065, +0.0307]** | 0.999 | **+0.0078 [+0.0023, +0.0130]** | 0.999 |

Ensemble nie poprawia porządkowania względem samego G-combo (ΔAUC +0.001); jego wkład
to log-loss, czyli jakość prawdopodobieństw — tak jak w rundzie przedmerge'owej.

Podgrupy (AUC, finalny vs stary ensemble na v1.2; `results/v12_report/subgroups_final/`):

| Podgrupa | n | finalny | stary | uwagi |
|---|---|---|---|---|
| 0–6 m | 120 | **0.855** | 0.801 | największy zysk: pole bramkowe, czyli tam, gdzie leżały odwrócone etykiety |
| 6–11 m | 208 | 0.777 | 0.759 | |
| 11–16 m | 290 | 0.664 | 0.643 | nadal najsłabszy pas |
| 16–25 m | 586 | 0.793 | 0.783 | |
| 25+ m | 506 | 0.856 | 0.869 | jedyny spadek, w szumie (n goli = 41) |
| geometria niedostępna | 519 | 0.763 | 0.761 | |
| karne (n = 34) | 34 | xG 0.681 vs 0.706 obs., LL 0.607 | xG 0.560, LL 0.626 | routing zamiast zgadywania z obrazu |
| gospodarz znany z treningu | 1 880 | 0.797 / LL 0.3215 | 0.785 / 0.3289 | |
| gospodarz nieznany | 349 | 0.774 / LL 0.3321 | 0.766 / 0.3375 | luka 2.4 pp AUC; CV po gospodarzu jej nie zamyka, tylko uczciwie mierzy |
| kadr q1 (najmniej murawy) | 650 | 0.754 | 0.733 | |

## 8. Co nie zadziałało i czego nie robić

- DINOv3-L i DINOv2-L z rejestrami jako zamrożony backbone (§4).
- Losowa klatka z wcześniejszych offsetów w treningu; warstwowy zanik LR z 8 blokami;
  głowa „celny" na zamrożonych cechach (§5).
- TTA (zmierzone w rundzie przedmerge'owej, szkodzi), większy backbone, wyższa
  rozdzielczość całego kadru, dalsze przebieranie w składzie ensemble'u: czołowe składy
  różnią się o ≤0.002 log-lossu, poniżej rozdzielczości pomiaru (±0.02 AUC, ±0.02 LL
  na 100 meczach).
- Model pozycyjny GBM: po wprowadzeniu routingu karnych jego wkład w log-loss stacku
  spadł niemal do zera (dl_mask+gbm vs dl_mask: 0.3327 vs 0.3323); zostaje w składzie
  tylko jeśli stack go wybierze.

## 9. Rekomendacja i następne kroki

1. **Produkcyjnie:** 5 (lub 10) modeli foldowych G-combo (DINOv2-L, 4 odmrożone bloki,
   RRC, głowa „celny", EMA) + Platt na OOF + routing karnych; C-combo i głowa D-L jako
   tanie członki stacku. Koszt inferencji: 5–10 przebiegów ViT-L na klatkę (≈50 ms na
   RTX 4070 Ti S) — akceptowalny offline, do zredukowania destylacją, jeśli potrzebny
   będzie tryb online.
2. **Dane przed architekturą** (§6): kolejne sezony SoccerNet i SoccerReplay-1988 po NDA;
   przy obecnym nachyleniu krzywej każde 300 meczów to ok. +0.01–0.02 AUC.
3. **Nadal otwarte:** pozycje GT z SoccerTrack (odblokowują cechy bramkarza i „światła
   bramki", tor pozycyjny jako członek ~0.70 zamiast 0.64), model wideo na t ≤ 0 z
   twardym guardem (jest w `datasets.py`), dwustrumieniowe wejście z wycinkiem bramki
   (pas 11–16 m pozostaje najsłabszy).
4. **Monitorować przy wdrożeniu na nowe ligi:** luka znane/nieznane stadiony (§7,
   podgrupy).

## 10. Koszt i reprodukcja

~10 h GPU (RTX 4070 Ti SUPER) na: bazę oficjalnego splitu (3 seedy × 4 modele),
2 backbone'y na zamrożonej głowie, screening (5 wariantów × 2 seedy), 2 krzywe uczenia,
CV (6 członków × 5 foldów, G-combo × 2 seedy). Wznawialne na poziomie (etap, fold, seed).

```bash
cd experiments/claude_fable
./run.sh python scripts/relabel_index.py                       # index v1.2 z v1.1
./run.sh python scripts/run_v12.py                             # baza: gbm dl_mask c_mask g_l4
./run.sh python scripts/run_v12.py --stages g_rrc g_time g_mt g_mixup g_llrd --seeds 2   # screening
./run.sh python scripts/run_learning_curve.py --stage dl_mask --seeds 3
./run.sh python scripts/run_cv.py --stages gbm dl_mask c_mask c_combo --seeds 1
./run.sh python scripts/run_cv.py --stages g_combo g_combo_mix --seeds 1
./run.sh python scripts/build_ensemble.py --results results/v12_cv --out results/v12_cv_ensemble \
    --prior-shift --penalty-route --max-k 4
./run.sh python scripts/significance.py --preds <final.csv> <old.csv> --names NEW OLD
./run.sh python scripts/eval_subgroups.py --preds results/v12_cv_ensemble/best_ensemble_test_predictions.csv
```
