# Podsumowanie eksperymentu: wybór architektury xG z obrazu

> **Uwaga (2026-08-31):** ten dokument opisuje stan po dużym treningu. Runda
> przedmerge'owa zmieniła dwie rzeczy i obie są opisane w `premerge_REPORT.md`:
> (1) tabela w sekcji 2 nie jest rankingiem — sparowany test pokazuje, że **C i D są
> nierozróżnialne** (P = 0.64), a jedyna istotna różnica to przewaga modeli obrazowych
> nad pozycyjnymi; (2) **fine-tuning DINOv2 istotnie bije zamrożoną głowę** i finalny
> wynik to ensemble o AUC 0.758 / BSS 11.7% / log-loss 0.340, nie 0.746 / 8.4% / 0.351.

Data: 2026-08-29. Branch `claude-fable/xg-poc`. Szczegółowe raporty cząstkowe:
`big_v1_FINAL_REPORT.md` (duży trening), `full_v1_short_REPORT.md` (krok 1),
`../research/` (literatura, plan i protokół danych).

## 1. Wynik końcowy w jednym akapicie

Na 11 030 strzałach z 500 meczów SoccerNet 720p (test: 2 229 strzałów / 100 meczów,
split po meczach) **podejścia obrazowe biją pozycyjne o ~0.10-0.13 AUC, poza
przedziałami ufności**. Najlepszy pojedynczy model to **zamrożony DINOv2-L z lekką
głową (AUC 0.741)**, tuż za nim **fine-tuning ConvNeXt-Tiny (0.732)**; ich
**ensemble osiąga AUC 0.746 [0.721-0.768], BSS 8.4%, log-loss 0.351**. Klasyczna
ścieżka "obraz → pozycje → model" (A/B, AUC ~0.61) przegrywa nie przez architekturę,
lecz przez jakość ekstrakcji pozycji. Rekomendacja do finalnej implementacji:
**C (fine-tune CNN) jako ścieżka główna, D (frozen foundation model + głowa) jako
równorzędna tania alternatywa; produkcyjnie ensemble obu.**

## 2. Pełna tabela (test, ensemble po 3 seedach, bootstrap CI po meczach)

| Model | AUC [95% CI] | BSS | log-loss | koszt treningu |
|---|---|---|---|---|
| **Ensemble C + D-L** | **0.746 [0.721, 0.768]** | **8.4%** | **0.351** | suma poniższych |
| D zamrożony DINOv2-L + głowa | 0.741 [0.715, 0.767] | 8.1% | 0.353 | 3×40 s (+5 min ekstrakcji) |
| C ConvNeXt-T fine-tune 448×256 | 0.732 [0.707, 0.754] | 7.6% | 0.353 | 3×3.5 min |
| C @672×384 | 0.724 [0.698, 0.748] | 6.7% | 0.357 | 3×8 min |
| D DINOv2-B (1 seed) | 0.722 [0.699, 0.745] | 6.5% | 0.359 | 40 s |
| C na klatce −400 ms | 0.714 [0.689, 0.739] | 6.4% | 0.360 | 3×4 min |
| C @896×512 (1 seed) | 0.713 [0.688, 0.737] | 5.7% | 0.362 | 12 min |
| D DINOv2-S | 0.713 [0.687, 0.737] | 6.0% | 0.362 | 3×20 s |
| E hybryda (det. tokeny + obraz) | 0.707 [0.683, 0.731] | 5.8% | 0.364 | 3×25 s |
| F temporalny (5 klatek ≤0 ms) | 0.697 [0.668, 0.722] | 5.3% | 0.366 | 3×30 s (+40 min ekstrakcji) |
| A pozycyjny logreg | 0.617 [0.581, 0.647] | 1.6% | 0.383 | sekundy |
| A pozycyjny GBM (fallback piłki) | 0.615 [0.581, 0.647] | 1.8% | 0.383 | sekundy |
| B Set Transformer (pozycje) | 0.611 [0.576, 0.642] | 1.2% | 0.386 | 3×15 s |

Baseline stały: log-loss 0.392, Brier 0.115. Referencja literaturowa na czystych
danych eventowych (bez szumu ekstrakcji): AUC ~0.78. Kalibracja wszystkich modeli
po temperature scaling: ECE ≤ 0.02.

Wyniki ablacji (wszystkie 3-seedowe, chyba że zaznaczono):
- **skala backbone'u działa**: DINOv2 S→B→L = 0.713→0.722→0.741;
- **rozdzielczość wejścia C nie działa**: 448→672→896 = 0.732→0.724→0.713
  (mniejszy batch i szybki overfit zjadają zysk z większej piłki);
- **klatka −400 ms gorsza od t0** — szum czasowy anotacji nie dominuje;
- **kontekst ruchu (F) nie pomaga** przy zamrożonych cechach per klatka;
- **fallback piłki** podniósł wykrycie piłki 49%→76% i homografię 61%→81%,
  ale A/B poprawiły się o <1 pp — sufit pozycyjnych to jakość homografii i
  jedno-klatkowość, nie brakująca piłka.

## 3. Timeline eksperymentu

| Data | Etap |
|---|---|
| 23.08 | Research literatury i datasetów; projekt 5 PoC; szkielet (parser etykiet, ekstraktor klatek, metryki, wspólny trener); smoke-testy na atrapie. |
| 24.08 rano | Przejęcie danych po agencie codex (uratowane: 500 meczów etykiet, SoccerTrack v2, StatsBomb; odrzucony jego unieważniony benchmark). Sample v0: 434 strzały. Diagnozy: 224p za mało dla pozycji, BAS nie anotuje strzału przy golach. |
| 24.08 po poł. | Weryfikacja hasła NDA i tokena HF; protokół pełnego datasetu; produkcyjny kolektor (okna wokół strzałów, sha256, wznawialność, strażnik 5 GB); start kolekcji 720p; kompresja GSR 50 GB→0.4 GB. |
| 24–25.08 | Kolekcja 500 meczów: ~1 TB transferu, ~26 h, 0 nieodzyskanych błędów (1 stall naprawiony timeoutem+walidacją). |
| 25.08 | Bramki odbioru datasetu: **11 035 strzałów, 1 491 goli**, splity czyste, monotoniczność dystans→gol potwierdzona. |
| 27.08 | Krok 1 handoffu: cache pozycji (poprawka: detektor imgsz=1920, piłka 0%→44%), krótkie przebiegi 5 PoC → obrazowe >> pozycyjne. |
| 27.08 wieczór | Duży trening: 3 seedy × 5 PoC do zbieżności + wariant c8 + DINOv2-B — łącznie ~1.5 h GPU (C zbiega w 8 epok). Finalny ranking. |
| 27–29.08 | Autonomiczna pętla ulepszeń: index v1.1 (deduplikacja), C@672/@896, C@−400 ms, PoC F temporalny, DINOv2-L, fallback piłki, ensemble C+D-L; agregacje i wizualizacje. |

Koszt obliczeniowy całości treningów: ~4 h GPU (RTX 4070 Ti SUPER); wąskim gardłem
projektu był transfer danych (~30 h), nie trening.

## 4. Przykładowe sytuacje (test set) — porównanie modeli

Rysunki w `figures/` (klatka + predykcje wszystkich modeli; linia przerywana =
częstość bazowa 0.13):

| Plik | Sytuacja | Co pokazuje |
|---|---|---|
| `sytuacja_1.png` | Gol z bliska po kontrze (Napoli–Fiorentina) | Obrazowe 0.31–0.35 vs pozycyjne 0.13–0.16: modele obrazowe widzą otwartą bramkę i kontrę, pozycyjne tylko "5 m od bramki". |
| `sytuacja_2.png` | Gol z dystansu | Wszystkie modele nisko (słusznie — to rzadkie zdarzenie); obrazowe minimalnie wyżej. |
| `sytuacja_3.png` | Zmarnowana wielka okazja | Wysokie xG mimo braku gola — tak ma działać xG; różnica C vs A pokazuje przewagę obrazu w ocenie ustawienia obrońców. |
| `sytuacja_4.png` | Strzał z ~30 m | Zgodnie niskie predykcje wszystkich modeli. |
| `sytuacja_5.png` | Rzut karny (Arsenal–PSG) | Kluczowy przykład: obrazowe rozpoznają jedenastkę (0.42–0.46), pozycyjne widzą tylko "20 m" (0.17–0.18). Uwaga: prawdziwa konwersja karnych to ~0.76 — nawet obrazowe niedoszacowują, bo karnych jest w treningu tylko ~90. |
| `sytuacja_6.png` | Największa niezgoda między modelami | Diagnostyka przypadków, gdzie ekstrakcja pozycji i obraz "widzą" co innego. |

## 5. Co pozostaje otwarte

1. ~~**dl672**~~ — porzucone po 3 próbach; ekstraktor wymaga ~29 GB RAM przy zapisie
   (do przebudowy na zapis strumieniowy). Opisane jako known issue w `run_night.sh`.
2. **SoccerTrack v2**: czeka na ręczne pobrania (DOWNLOAD_CHECKLIST.md) → pozycje GT
   → uczciwy re-test A/B i lokalizator kontaktu (faza 2 protokołu).
3. Kierunki dalszej poprawy jakości — stan po rundzie przedmerge'owej:
   - fine-tuning DINOv2 — **zrobione, działa** (+0.019 AUC istotnie);
   - maskowanie scoreboardu — **zrobione, darmowe**, włączone domyślnie;
   - TTA — **zrobione, odrzucone** (istotnie szkodzi);
   - wciąż otwarte: klipy 8 s z modelem wideo (mamy 18 GB klipów), większy zbiór
     (SoccerReplay-1988 po NDA), pozycje GT z SoccerTrack.
