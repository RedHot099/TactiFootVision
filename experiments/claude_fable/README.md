# xG z obrazu: porównanie 5 architektur (PoC)

Branch `claude-fable/xg-poc`, worktree `.claude/worktrees/claude-fable-xg-poc`.
Wszystko w tym folderze; reszta repo jest używana tylko jako biblioteka (wagi
detektora i keypointów z `results/project/raw/...`, `SoccerPitchConfiguration`).

Research i uzasadnienie wyboru podejść: `research/literature_and_datasets.md`.

## Cel

Model dostaje klatkę z momentu strzału, zwraca prawdopodobieństwo gola (xG).
Kryterium główne: log-loss i Brier na wspólnym zbiorze testowym podzielonym po
meczach, z bootstrapowym CI po meczach. Raportowane też AUC, ECE, liczba parametrów
i koszt inferencji. Detekcja momentu strzału jest poza zakresem.

## Pięć podejść

| PoC | Wejście | Model | Po co w porównaniu |
|---|---|---|---|
| A `a_tabular` | klatka → detekcja + keypointy + homografia → 20 cech inżynierskich (dystans, kąt, obrońcy w stożku, GK, zagęszczenie w polu karnym) | regresja logistyczna i HistGradientBoosting | Klasyczna recepta xG. Punkt odniesienia, który wszystko inne musi pobić. |
| B `b_set_transformer` | te same pozycje jako zbiór tokenów (piłka + gracze, współrzędne kanoniczne, drużyna, GK) | Set Transformer (2 warstwy, attention pooling), geometria względem piłki i bramki liczona w modelu | Czy sieć głęboka na pozycjach wyciągnie więcej niż cechy ręczne (np. układ obrony). |
| C `c_cnn_e2e` | surowa klatka 448×256 | ConvNeXt-Tiny (ImageNet-22k) fine-tuning całości | Podejście end-to-end bez etapu pośredniego. Najbardziej "głodne danych". |
| D `d_frozen_dinov2` | surowa klatka 448×252 | zamrożony DINOv2 ViT-S/14 → siatka patchy (cache) → głowa z attention pooling (≈0.6M parametrów) | Recepta z SoccerNet challenge: foundation model + lekka głowa. Uczy się w sekundy, odporne na małe zbiory. |
| E `e_hybrid` | patch-e DINOv2 + boxy detektora | token na każdą detekcję (ROI-średnia patchy + geometria boxa + klasa) + token globalny → mały transformer | Hybryda: jawna struktura "kto gdzie" bez zaufania do homografii, plus kontekst obrazu. |

Ablacje planowane po dużym treningu (nie osobne PoC): raster boiska + mały CNN
(SoccerMap) wewnątrz B; krótki klip zamiast jednej klatki w C/D; pozycje ground-truth
z SoccerTrack v2 GSR zamiast detektora w A/B (izoluje błąd detekcji/homografii);
backbone `dinov2_b` / `siglip_b` w D/E.

## Kontrakt danych (dla agenta przygotowującego dane)

Jeden plik parquet, wiersz na strzał (`xgpoc/data/index.py`):

| kolumna | typ | opis |
|---|---|---|
| `shot_id` | str | unikalne id |
| `source` | str | `soccernet`, `soccertrack`, ... |
| `game` | str | id meczu; split zawsze po meczu |
| `split` | str | `train` / `valid` / `test` (`split_by_game` daje deterministyczny podział z hasha) |
| `goal` | bool | target |
| `frame_path` | str | absolutna ścieżka do JPEG z momentu strzału |
| `is_penalty` | bool | opcjonalnie |
| `visibility` | str | opcjonalnie (`visible` / `not shown`) |
| `gt_positions` | str | opcjonalnie, JSON z pozycjami GT (SoccerTrack GSR) |

Pomocnicze, gotowe do użycia przez tego agenta: `xgpoc/data/soccernet_labels.py`
(parser Labels-v2.json łączący `Shots on/off target` z `Goal` tej samej drużyny w oknie
6 s; gole bez strzału dostają własny event 1 s przed), `xgpoc/data/frames.py`
(ffmpeg seek, kilka offsetów, filtr `grass_ratio`).

## Jak uruchomić

```bash
cd experiments/claude_fable
./run.sh pytest -q                                   # testy jednostkowe (CPU, ~3 s)
./run.sh python scripts/make_dummy_index.py          # atrapa z klatek SoccerNet tracking, losowe etykiety
./run.sh python scripts/build_positional_cache.py cache/dummy/index.parquet cache/dummy/positional
./run.sh python scripts/run_poc.py --index cache/dummy/index.parquet --cache cache/dummy \
    --out results/dummy --poc a b c d e --epochs 5 --max-minutes 2
./run.sh python scripts/compare.py --index cache/dummy/index.parquet --results results/dummy
```

Na prawdziwych danych: podmienić `--index` na plik od agenta danych, podnieść
`--epochs` i `--max-minutes`. Cache pozycji (A, B, E) i cech DINOv2 (D, E) buduje się
raz na index.

## Stan (2026-08-28)

Duży trening zakończony: 3 seedy × 5 PoC do zbieżności + ablacje (C z domkniętym
cyklem LR, DINOv2-B). Finalny ranking (ensemble, test 2 229 strzałów / 100 meczów):
C ConvNeXt 0.732 AUC / BSS 7.6% > D DINOv2-B frozen 0.722/6.5% > E 0.710 >> A/B
pozycyjne ~0.61. Rekomendacja: C jako ścieżka główna, D jako tania alternatywa;
E i pozycyjne bez inwestycji do czasu lepszej ekstrakcji. Raport:
`results_reports/big_v1_FINAL_REPORT.md`.

## Stan poprzedni (2026-08-27)

Krok 1 handoffu wykonany na pełnym SoccerNet 720p (11 035 strzałów / 1 491 goli):
krótkie przebiegi (≤4 min/model) dały na teście (2 232 strzały, 100 meczów):
C ConvNeXt AUC 0.729 / BSS 6.9% > D DINOv2-frozen 0.715/6.5% ≈ E hybryda 0.709/6.2%
>> A/B pozycyjne ~0.61/1.5%. Obrazowe biją pozycyjne poza CI; pełny raport:
`results_reports/full_v1_short_REPORT.md`. Słabość A/B to jakość ekstrakcji pozycji
(piłka 49%, homografia 61%), nie architektura. Czeka: duży trening + poprawa
ekstrakcji + faza 2 SoccerTrack.

## Stan poprzedni (2026-08-23)

- Szkielet działa end-to-end na atrapie (240 klatek, losowe etykiety): wszystkie 5
  PoC trenuje, kalibruje temperaturę na valid, zapisuje predykcje i metryki.
- Ekstraktor pozycji: homografia z keypointów (model wymaga rozciągnięcia klatki do
  640×640, tak był trenowany), detektor YOLO11m (ball/GK/player/referee), drużyny
  przez 2-means na histogramach torsu, bramka atakowana = bliższa piłce.
- Znane słabości do zmierzenia na prawdziwych danych: piłka wykrywana w ~23%
  losowych klatek (w momencie strzału powinno być lepiej, ale to główne ryzyko A/B);
  klasa GK prawie nie występuje w wyjściu detektora.
- Duży trening: czeka na index z prawdziwymi strzałami.

## Struktura

```
xgpoc/data       index (kontrakt), parser etykiet SoccerNet, ekstrakcja klatek, datasety torch
xgpoc/features   freeze_frame (obraz → pozycje), tabular (cechy), cache
xgpoc/models     a_tabular, b_set_transformer, c_cnn_e2e, d_foundation, e_hybrid, common (trener)
xgpoc/eval       metryki (Brier, log-loss, AUC, ECE, BSS)
scripts          run_poc, compare, build_positional_cache, make_dummy_index
research         literatura i datasety
tests            parser, metryki, modele (w tym test uczenia reguły syntetycznej)
```
