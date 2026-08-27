# Duży trening: finalne porównanie architektur xG z obrazu

Data: 2026-08-28. Dane: SoccerNet 720p, 11 035 strzałów / 1 491 goli / 500 meczów,
splity oficjalne po meczach (test: 2 229 strzałów po deduplikacji, 100 meczów).
Reżim: każdy PoC trenowany do zbieżności (early stopping na valid log-loss),
3 seedy, kalibracja temperaturą na valid, **ensemble = średnia predykcji 3 seedów**,
bootstrap CI po meczach (500 prób). Artefakty: `results/big_v1/` (poza gitem),
agregat: `results/big_v1/aggregate.csv`.

## Wyniki (ensemble po seedach; sortowanie po log-loss)

| Architektura | AUC (ens) [95% CI] | AUC seed mean±std | BSS | log-loss | ECE | koszt treningu |
|---|---|---|---|---|---|---|
| **C ConvNeXt-T fine-tune (cykl 8 epok)** | **0.732 [0.707, 0.754]** | 0.726 ± 0.008 | **7.6%** | **0.3530** | 0.013 | ~3.5 min/seed |
| C ConvNeXt-T (cykl 40, early-stop ~8) | 0.728 [0.704, 0.751] | 0.719 ± 0.005 | 7.0% | 0.3558 | 0.018 | ~3.5 min/seed |
| D zamrożony DINOv2-**B** + głowa (1 seed) | 0.722 [0.699, 0.745] | — | 6.5% | 0.3589 | 0.015 | 0.6 min + ekstrakcja |
| E hybryda @DINOv2-B (1 seed) | 0.710 [0.681, 0.737] | — | 6.6% | 0.3603 | 0.010 | 0.5 min + ekstrakcja |
| D zamrożony DINOv2-S + głowa | 0.713 [0.687, 0.737] | 0.711 ± 0.001 | 6.0% | 0.3618 | 0.021 | 0.3 min/seed |
| E hybryda @DINOv2-S | 0.707 [0.683, 0.731] | 0.702 ± 0.004 | 5.8% | 0.3637 | 0.015 | 0.4 min/seed |
| A logreg (pozycje) | 0.617 [0.581, 0.647] | 0.617 | 1.6% | 0.3826 | 0.002 | sekundy |
| B Set Transformer (pozycje) | 0.611 [0.576, 0.642] | 0.600 ± 0.012 | 1.2% | 0.3857 | 0.013 | 0.2 min/seed |
| A GBM (pozycje) | 0.605 [0.570, 0.636] | 0.605 | 1.4% | 0.3842 | 0.002 | sekundy |

Baseline stały: Brier 0.115, log-loss 0.392. Referencja literaturowa (czyste dane
eventowe, bez szumu ekstrakcji): AUC ~0.78.

## Wnioski

1. **Obrazowe >> pozycyjne, poza CI.** Wszystkie trzy podejścia obrazowe (C, D, E)
   biją wszystkie pozycyjne z marginesem ~0.09-0.13 AUC; przedziały się nie przecinają.
   Przy obecnej jakości ekstrakcji pozycji (piłka 49%, homografia 61%, ~1 s szumu
   anotacji) klasyczna droga "pozycje → model" traci większość sygnału.
2. **Zwycięzca jakościowo: C (fine-tuning ConvNeXt-Tiny)** — AUC 0.732, BSS 7.6%,
   log-loss 0.353. Dopasowanie cyklu LR do realnej długości treningu (8 epok) dało
   +0.4 pp AUC i lepszą kalibrację względem przerwanego cyklu 40-epokowego.
3. **Najlepszy stosunek jakości do kosztu: D (zamrożony DINOv2 + głowa)** — 0.6M
   trenowanych parametrów, 20 s/seed po jednorazowej ekstrakcji cech, AUC 0.722
   (DINOv2-B) czyli ~1 pp poniżej C. Różnica C vs D mieści się w CI; większy backbone
   (S→B) daje +0.9 pp, więc DINOv2-L może domknąć różnicę do C.
4. **Hybryda E nie uzasadnia swojej złożoności** — konsekwentnie minimalnie poniżej D
   na obu backbone'ach. Jawne tokeny detekcji nie dodają informacji ponad patch-e,
   przynajmniej przy obecnym detektorze.
5. **B nie przewyższa A** (0.611 vs 0.617) — spójne z tezą, że wąskim gardłem
   pozycyjnych jest jakość wejścia, nie pojemność modelu.
6. Powtarzalność: std po seedach ≤0.012 AUC; ensemble stabilnie dodaje +0.5-1 pp.
7. Drobiazg danych: 5 zduplikowanych `shot_id` w indexie (podwójne anotacje w tym
   samym ms) — deduplikowane w agregacji; do wyczyszczenia w v1.1 datasetu.

## Rekomendacja do finalnej implementacji

**Ścieżka główna: C — fine-tuning nowoczesnego CNN na klatce** (najlepsza jakość,
trening minuty). **Ścieżka produkcyjno-badawcza: D — zamrożony foundation model +
lekka głowa** (porównywalna jakość, najtańsza iteracja, mniejsze ryzyko przeuczenia
przy rozszerzaniu danych). Nie inwestować w E ani w pozycyjne A/B, dopóki ekstrakcja
pozycji nie osiągnie jakości zbliżonej do GT (test: kohorta SoccerTrack z GSR, faza 2).

## Ograniczenia

- Ranking dotyczy obecnej skali danych (11k strzałów) i rozdzielczości 448×256;
  wyższa rozdzielczość może pomóc C (piłka bywa <4 px po przeskalowaniu).
- Etykieta czasowa ma ~1 s szumu; część klatek t0 jest tuż po strzale (zaniżony
  goal rate w kubełku <8 m) — ablacja z offsetem −400 ms pozostaje do wykonania.
- SoccerTrack (pozycje GT) i klipy 8 s (kontekst ruchu) czekają jako następne ablacje.
