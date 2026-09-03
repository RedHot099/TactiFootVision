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

## Jak odtworzyć wyniki od zera

Wymagane: SoccerNet (hasło NDA), ~600 GB miejsca na transfer i ~50 GB na cache cech,
GPU z 8 GB VRAM. Całość to ~30 h transferu i ~10 h GPU.

```bash
cd experiments/claude_fable
# 1. dane: pobranie 500 meczów 720p i wycięcie okien wokół strzałów (wznawialne,
#    pilnuje 5 GB wolnego miejsca; szczegóły i fazy w DATASET_PROTOCOL.md)
./run.sh python scripts/collect_soccernet.py --password "$SOCCERNET_PASSWORD"
./run.sh python scripts/check_index_quality.py          # bramki odbioru datasetu

# 2. cache pozycji (A, B, E) i tabela cech dodatkowych
./run.sh python scripts/build_positional_cache.py "$IDX" cache/full_v2_fallback/positional --ball-fallback
./run.sh python scripts/build_aux_table.py

# 3. wiarygodność: przeciek między splitami i lokalizacja nakładki telewizyjnej
./run.sh python scripts/audit_leakage.py --out results/audit
./run.sh python scripts/find_overlay_region.py --out configs

# 4. treningi: porównanie architektur, potem kolejka przedmerge'owa
bash scripts/run_big.sh && bash scripts/run_night.sh
./run.sh python scripts/run_premerge.py
./run.sh python scripts/build_positional_member.py

# 4b. runda jakości (2026-09): etykiety v1.2, naprawiony trener, CV po gospodarzu
./run.sh python scripts/relabel_index.py                # index v1.2
./run.sh python scripts/run_v12.py                      # baza na oficjalnym splicie
./run.sh python scripts/run_cv.py --stages gbm dl_mask c_combo g_combo g_combo_mix --seeds 2
./run.sh python scripts/build_ensemble.py --results results/v12_cv --out results/v12_cv_ensemble --prior-shift --penalty-route --max-k 4

# 5. analiza
./run.sh python scripts/aggregate_big.py --index "$IDX" --results results/premerge
./run.sh python scripts/build_ensemble.py --results results/premerge --out results/ensemble
./run.sh python scripts/eval_subgroups.py --preds "results/ensemble/*_test_predictions.csv" --out results/subgroups
./run.sh python scripts/significance.py --preds <a.csv> <b.csv> --names A B
./run.sh python scripts/make_situation_figures.py --preset finalne --out results_reports/figures_finalne
```

Wszystkie treningi są wznawialne na poziomie (etap, seed): gotowy przebieg zostawia
plik `*_metrics.json` i jest pomijany przy ponownym uruchomieniu.

## Stan (2026-09-03) — runda jakości po review

Review (`research/review_2026-09-02.md`) znalazło trzy usterki zmieniające liczby:
(1) parser przypisywał gola do najwcześniejszego strzału w oknie 6 s zamiast do
najbliższego (194 odwrócone etykiety, 52 w teście) → **index v1.2**; (2) trener ucinał
głowy w fazie rozgrzewki i wybierał epokę po niekalibrowanym log-lossie → cykl = budżet
epok, selekcja po skalibrowanym LL, EMA; (3) prior walidacji (14.9%) wdrukowany w kalibrację
→ Platt + korekta priora + routing karnych. Wszystkie nowe liczby są na etykietach v1.2 i
**nie są porównywalne** ze starymi tabelami; stare predykcje przeliczone na v1.2 dają
0.782 AUC / 0.330 log-loss.

Finalny wynik (test 2 229 strzałów / 100 meczów, etykiety v1.2, wybór składu wyłącznie na
predykcjach out-of-fold z CV grupowanego po drużynie gospodarzy):

| Model | AUC [95% CI] | log-loss | BSS |
|---|---|---|---|
| **Ensemble: C-combo + G-combo + G-combo-mix + GBM** | **0.793 [0.768, 0.816]** | **0.323** | **16.3%** |
| G-combo (fine-tune DINOv2-L/4 + RRC + głowa „celny", 5 foldów × 2 seedy) | 0.792 | 0.324 | 16.1% |
| D-L zamrożony + głowa (CV) | 0.774 | 0.332 | 14.1% |
| C-combo (ConvNeXt + RRC + głowa „celny", CV) | 0.766 | 0.337 | 12.4% |
| A pozycyjny GBM (CV) | 0.647 | 0.369 | 6.2% |
| stary ensemble przedmerge'owy na etykietach v1.2 | 0.782 | 0.330 | 14.4% |

Finalny vs stary: ΔAUC +0.011 [+0.000, +0.022] (P = 0.98), Δlog-loss +0.007 [+0.002, +0.012]
(P = 1.00). Krzywe uczenia obu ścieżek nadal rosną (+0.01 AUC na ostatnie 75 meczów), więc
następny krok to więcej meczów, nie architektura. DINOv3-L i DINOv2-L z rejestrami są gorsze
od DINOv2-L na zamrożonej głowie. Raport: `results_reports/v12_quality_REPORT.md`, dziennik:
`research/quality_run_worklog.md`.

## Stan (2026-08-31) — po rundzie przedmerge'owej

Audyt przecieku: **PASS** (0 klatek testowych z bliźniakiem w treningu). Zależność od
grafiki telewizyjnej sprawdzona i **wykluczona** (model widzący sam scoreboard: AUC
0.530, BSS −0.2%); maskowanie nakładki włączone domyślnie, bo nic nie kosztuje.

Ranking finalny (test 2 229 strzałów / 100 meczów, wszystkie porównania sparowane):

| Model | AUC [95% CI] | BSS | log-loss |
|---|---|---|---|
| **Ensemble: D-L(maska) + fine-tune L/12 + GBM** | **0.758 [0.734, 0.781]** | **11.7%** | **0.340** |
| G fine-tune DINOv2-L/4 + maska | 0.754 | 9.6% | 0.346 |
| D-L zamrożony + maska | 0.735 | 8.2% | 0.353 |
| C ConvNeXt + maska | 0.735 | 7.9% | 0.352 |
| A pozycyjny GBM (train only) | 0.622 | 5.4% | 0.372 |

**Fine-tuning DINOv2 istotnie bije zamrożoną głowę** (ΔAUC +0.019 [+0.001, +0.037]) —
ale dopiero po naprawieniu learning rate; przy pierwszym ustawieniu głębsze odmrażanie
zapadało się do stałej predykcji. **TTA odrzucone** (istotnie szkodzi C). Model
pozycyjny zostaje w składzie mimo AUC 0.62, bo jako jedyny domyka karne i górny ogon
rozkładu. Raport: `results_reports/premerge_REPORT.md`.

## Stan poprzedni (2026-08-28)

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
xgpoc/models     a_tabular, b_set_transformer, c_cnn_e2e, d_foundation (zamrożony + FinetuneViTXG),
                 e_hybrid, f_temporal, common (trener, kalibracja, TTA)
xgpoc/eval       metryki (Brier, log-loss, AUC, ECE, BSS)
configs          maski nakładki telewizyjnej wykryte z danych (overlay_mask.png, graphics_mask.png)
scripts          run_poc (PoC a-g), run_premerge (kolejka GPU), audit_leakage, find_overlay_region,
                 build_ensemble, eval_subgroups, significance, compare, build_positional_cache
research         literatura i datasety
tests            parser, metryki, modele (w tym test uczenia reguły syntetycznej)
```
