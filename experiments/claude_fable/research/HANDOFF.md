# Handoff — eksperyment xG z obrazu

Stan na 2026-09-07. Gałąź `claude-fable/xg-poc`, HEAD `7a41aad`, drzewo czyste,
41 testów przechodzi, `ruff check .` czysty. **Gałąź nie została zmergowana do `master`.**

## 1. Gdzie co jest

| | |
|---|---|
| Worktree | `/home/kuba/projects/tactifoot_vision/.claude/worktrees/claude-fable-xg-poc` |
| Katalog eksperymentu | `experiments/claude_fable` — **cała praca zostaje tutaj** |
| Uruchamianie | `./run.sh python ...` (venv projektu); nie wywołuj `python` bezpośrednio |
| Dane | `/home/kuba/projects/football_data/xg_shots_full/` (poza repo) |
| Cache cech | `cache/` — **108 GB**, drogie do odtworzenia, nie kasować bez potrzeby |
| Wolne miejsce | 263 GB. Twardy wymóg: **zawsze ≥ 5 GB wolnego** |
| Raporty | `results_reports/` — najświeższy `v12_quality_REPORT.md` |
| Dziennik przebiegów | `research/quality_run_worklog.md` (wpisy z datą i godziną) |
| Plan kierunkowy | `research/improvement_roadmap.md` |

Konwencje: raporty i dokumenty po polsku, kod i komentarze po angielsku. Commity
opisują wynik, nie czynność.

## 2. Aktualny wynik

Test: 2 229 strzałów / 100 meczów, etykiety **v1.2**.

**Finalny ensemble: AUC 0.7933 [0.768, 0.816], log-loss 0.3232 [0.303, 0.347], BSS 16.3%, ECE 0.016.**

Skład: `C-combo + G-combo + G-combo-mix + GBM`, wagi [0.11, 0.48, 0.39, 0.27], każdy
członek to średnia logitów 5 modeli foldowych. Predykcje:
`results/v12_cv_ensemble/best_ensemble_test_predictions.csv`.

Rekomendacja produkcyjna: 5–10 modeli foldowych **G-combo** (DINOv2-L, 4 odmrożone bloki,
RandomResizedCrop 0.85–1.0, głowa pomocnicza „strzał celny" λ=0.5, EMA) + Platt na OOF
+ routing karnych. Sam G-combo daje 0.792 — ensemble dokłada praktycznie tylko jakość
prawdopodobieństw (ΔAUC +0.001, Δlog-loss +0.001).

## 3. Metodologia, której trzeba się trzymać

Te reguły były w tym projekcie łamane i naprawiane — nie cofaj ich.

- **Test dotykany raz.** Skład ensemble'u i wagi wybierane na predykcjach out-of-fold
  z 400 meczów, nigdy na teście. Wcześniejsza runda wybierała skład po log-lossie
  testowym i trzeba było to przepisać.
- **Porównania sparowane**, bootstrap grupowany po meczach, 2 000 losowań
  (`scripts/significance.py`).
- **Rozdzielczość pomiaru: ±0.02 AUC** na 100 meczach testowych. Różnice mniejsze są
  nierozróżnialne od szumu — nie raportuj ich jako postępu i nie optymalizuj pod nie.
  Dziesięć czołowych składów ensemble'u mieści się w 0.792–0.794.
- **Selekcja konfiguracji tylko po walidacyjnym log-lossie po temperaturze**, uśrednionym
  po seedach. Pojedynczy seed okazał się za szumny — kiedyś wskazał wariant o najgorszej
  walidacji jako najlepszy na teście.
- **Wyciek czasowy.** `cut_clip(span_s=4.0)` tnie okno [t−4 s, t+4 s], więc druga połowa
  każdego klipu i klatki `+400`/`+800` zawierają rozstrzygnięcie. Wolno używać wyłącznie
  `t ≤ 0`; guard jest w `datasets.py` i ma tam zostać.
- **`.gitignore`**: katalog główny repo ignoruje `data/`, `models/` i `*.sh`, co kiedyś
  po cichu wykluczyło z commita cały `xgpoc/data/` i `xgpoc/models/` — gałąź była
  nieuruchamialna. Negacje w `experiments/claude_fable/.gitignore` muszą zostać.
- Sesja jest izolowana w worktree: nie `cd` do głównego repo, nigdy gołe `git stash`.

## 4. Decyzja, która już zapadła: dane przed architekturą

Krzywe uczenia (zagnieżdżone podzbiory meczów treningowych, `§6` raportu v1.2):

| mecze | D-L głowa | G-L/4 |
|---|---|---|
| 75 | 0.737 | 0.740 |
| 150 | 0.748 | 0.766 |
| 225 | 0.757 | 0.771 |
| 300 | 0.769 | 0.780 |

Ostatni krok (+75 meczów) daje +0.011 i +0.009 AUC — **nie mniej niż poprzednie**.
Krzywa nie jest nasycona dla żadnej ze ścieżek. W skali log-liniowej podwojenie liczby
meczów jest warte ok. +0.02 AUC, czyli mniej więcej tyle, ile wynosi cały dotychczasowy
zysk z architektury w tej rundzie.

**Wniosek: kolejny duży ruch to pozyskanie meczów, a nie zmiana modelu.** To była bramka
decyzyjna P5 z roadmapy i wypadła jednoznacznie.

## 5. Czego NIE próbować ponownie (zmierzone i odrzucone)

- **TTA** (odbicie poziome) — istotnie szkodzi, ΔAUC −0.0065 [−0.0121, −0.0006], P = 0.014.
- **Większy backbone / DINOv3-L / DINOv2-L z rejestrami** — 0.01–0.02 AUC gorsze przy
  zamrożonej głowie.
- **Wyższa rozdzielczość całego kadru** — 448 → 672 → 896 daje 0.732 → 0.724 → 0.713.
- **Losowa klatka z {−800, −400, 0} ms w treningu** — brak zysku, duża wariancja.
- **Warstwowy zanik LR z 8 blokami** — neutralne, gorszy log-loss, szybsze przeuczenie.
- **Głowa „strzał celny" na zamrożonych cechach** — pomaga tylko przy fine-tuningu.
- **Dalsze przebieranie w składzie ensemble'u** — poniżej rozdzielczości pomiaru.
- **dHash do wykrywania duplikatów** — nierozdzielczy w tej domenie (pary tego samego
  strzału: mediana 13 bitów; pary losowe: p1 = 21). Używamy kosinusa na CLS DINOv2-S.

## 6. Otwarte, w kolejności

1. **Więcej meczów** (§4): pozostałe sezony SoccerNet, SoccerReplay-1988 po NDA.
2. **SoccerTrack v2 — zablokowane po stronie użytkownika.** Brakuje **14 plików**
   (11 wideo + 3 GSR) do ręcznego pobrania z Google Drive; linki i ścieżki docelowe
   w `football_data/xg_shots_full/DOWNLOAD_CHECKLIST.md`, weryfikacja
   `scripts/verify_soccertrack_downloads.sh`. Stan sprawdzony 2026-09-07: lista zgadza
   się co do jednego wpisu z tym, czego faktycznie brakuje; pobrane pliki są poprawne.
   Odblokowuje pozycje GT → cechy bramkarza i „światła bramki" → tor pozycyjny jako
   członek ~0.70 zamiast 0.64.
3. **Pas 11–16 m nadal najsłabszy** (AUC 0.664 wobec 0.855 w polu bramkowym). Hipoteza:
   dwustrumieniowe wejście — pełny kadr + wycinek okolic bramki w pełnej rozdzielczości.
   To jedyna nietknięta hipoteza wyjaśniająca, dlaczego jednolite skalowanie
   rozdzielczości zawiodło.
4. **Model wideo na oknie t ≤ 0** — guard jest, model nie.
5. **Merge gałęzi do `master`** — nigdy nie był proszony ani wykonany.
6. **Luka znane/nieznane stadiony** 2.4 pp AUC (0.797 vs 0.774) — CV po gospodarzu jej
   nie zamyka, tylko uczciwie mierzy. Do monitorowania przy wdrożeniu na nowe ligi.

## 7. Reprodukcja

Pełna sekwencja w `results_reports/v12_quality_REPORT.md` §10. Wszystkie runnery są
wznawialne na poziomie (etap, fold, seed) — przerwany przebieg wznawia się bez strat.
Cała runda v1.2 to ~10 h GPU na RTX 4070 Ti SUPER.
