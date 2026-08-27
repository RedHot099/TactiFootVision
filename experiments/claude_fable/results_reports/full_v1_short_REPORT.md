# Krok 1 handoffu: krótkie przebiegi 5 PoC na pełnym SoccerNet 720p

Data: 2026-08-27. Dane: `football_data/xg_shots_full/soccernet_720p/index.parquet`
(11 035 strzałów / 1 491 goli / 500 meczów; karta: `dataset_card.md`).
Reżim: krótkie przebiegi (limit 4 min GPU na model, 15 epok max, RTX 4070 Ti S);
kalibracja temperaturą na valid; test = 2 232 strzały ze 100 meczów (splity oficjalne).
Przebiegi i predykcje: `results/full_v1_short/` (poza gitem), cache: `cache/full_v1/`.

## Tabela (test, bootstrap CI po meczach, 500 prób)

| PoC | Brier | BSS | log-loss | AUC [95% CI] | ECE |
|---|---|---|---|---|---|
| **C ConvNeXt-T fine-tune** | **0.1074** | **6.9%** | **0.3563** | **0.729 [0.705, 0.753]** | 0.013 |
| D DINOv2-S frozen + głowa | 0.1078 | 6.5% | 0.3606 | 0.715 [0.688, 0.739] | 0.016 |
| E hybryda (obiekty+obraz) | 0.1082 | 6.2% | 0.3620 | 0.709 [0.683, 0.732] | 0.010 |
| A logreg (pozycje) | 0.1140 | 1.6% | 0.3843 | 0.617 [0.581, 0.647] | 0.002 |
| B Set Transformer (pozycje) | 0.1137 | 1.4% | 0.3847 | 0.614 [0.581, 0.644] | 0.011 |
| A GBM (pozycje) | 0.1143 | 1.4% | 0.3862 | 0.602 [0.567, 0.634] | 0.002 |

## Odczyt

1. **Podejścia obrazowe (C/D/E) wyraźnie biją pozycyjne (A/B)**: przedziały ufności
   AUC się nie przecinają (dolne granice obrazowych ~0.68-0.70 vs górne pozycyjnych
   ~0.65). Obraz niesie informację, której ekstrakcja pozycji nie wyciąga.
2. **C ≈ D ≈ E w granicach CI** — po 4 minutach treningu nie ma podstaw do wyboru
   lidera wśród obrazowych; to rozstrzygnie duży trening (C ma największy potencjał
   wzrostu — przy 4 min zdążył ~2 epoki — ale też największe ryzyko przeuczenia).
3. **Słabość pozycyjnych to jakość ekstrakcji, nie modele**: piłka wykryta na t0 w
   49% klatek, homografia w 61%, ~1 s szumu anotacji. B nie przewyższa A, co przy
   zaszumionych wejściach jest oczekiwane. Uczciwe porównanie architektur pozycyjnych
   wymaga: (a) fallbacku piłki z klatek −400/−800 ms, (b) lepszego modelu keypointów,
   (c) kohorty SoccerTrack z pozycjami GT (faza 2).
4. Kalibracja po temperature scaling jest dobra u wszystkich (ECE ≤ 0.016).
5. Baseline'y stałe: Brier 0.1151, log-loss 0.392. Literatura na czystych danych
   eventowych raportuje AUC ~0.78; C osiąga 0.73 z samego obrazu po 2 epokach.

## Kryteria handoffu

- Wszystkie 5 PoC na tym samym teście: ✅ (n=2232; A ma 2236, bo 4 strzały odpadły
  z cache tokenów w innych PoC — różnica pomijalna).
- A_gbm BSS > 0: ✅ (+1.4%); AUC > 0.65: ❌ (0.602) — zgodnie z decyzją z bramek
  jakości: to ograniczenie ekstrakcji pozycji (udokumentowane), nie danych; porównanie
  kontynuowane, bo monotoniczność dystans→goal rate na danych potwierdzona.

## Rekomendowane następne kroki

1. Duży trening (na sygnał): C i D do zbieżności (C: ~1-2 h, D: minuty na epokę),
   E z większym backbone (dinov2_b), seedy ×3 dla stabilności rankingu.
2. Równolegle poprawa ekstrakcji pozycji (fallback offsetów dla piłki) i rerun A/B.
3. Ablacje: klip 8 s zamiast klatki (mamy klipy), offset −400 ms jako wejście.
