# Tracking and detection experiment
Celem tego eksperymentu jest porównanie jakości uzyskiwanych rezultatów dla różnych podejść detekcji oraz śledzenia obiektów. Chcemy wytrenować i porównać ze sobą różne konfiguracje modeli detekcji RF-DETR (Base oraz segment) razem z różnymi wariantami metod śledzenia obiektów w video ByteTrack, BoT-SORT oraz śledzenie przez segmentację z użyciem SAM2. 
Eksperyment będzie przeprowadzony na zbiorze danych SoccerNet tracking:
- trening modeli detekcji: split `train` (lokalnie: `data/soccernet/tracking/extracted/train`)
- ewaluacja eksperymentu: split `test` (pobieramy oficjalnie i przygotowujemy analogicznie do `train`)

Uwaga: potrzebujemy też wydzielić zestaw do strojenia hiperparametrów trackerów (bez używania `test`). Najprościej: wydzielić podzbiór sekwencji z `train` jako `tune` (np. 20% sekwencji), a resztę traktować jako właściwy `train` do trenowania detektorów.

## Pobranie danych (oficjalna biblioteka)
Split `test` musimy pobrać przy użyciu oficjalnej biblioteki SoccerNet (opis poniżej), a następnie wyekstrahować do formatu analogicznego jak obecny `extracted/train` (foldery `SNMOT-*/img1`, `gt/gt.txt`, `seqinfo.ini`, `gameinfo.ini`).

Używamy i raportujemy wyniki dla zadania `tracking-2023` (dla spójności `train` i pobierany `test` muszą pochodzić z tej samej “rodziny” danych).
<soccernet-lib>
How to download SoccerNet-tracking
We provide a SoccerNet pip package to easily download the data and the annotations.

To install the pip package simply run:

pip install SoccerNet

Then, to download the tracking data, enter the following commands:

from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
Data format
The ground truth and detections are stored in comma-separate csv files with 10 columns. These values correspond in order to: frame ID, track ID, top left coordinate of the bounding box, top y coordinate, width, height, confidence score for the detection (always 1. for the ground truth) and the remaining values are set to -1 as they are not used in our dataset, but are needed to comply with the MOT20 requirements.
</soccernet-lib>

## Konfiguracje eksperymentu 
W celu sprawdzenia, która konfiguracja modelu detekcji oraz metod śledzenia sprawdza się najlepiej, testujemy następujące połączenia (usuwamy warianty “without tracking”):

1. RF-DETR Base + ByteTrack
2. RF-DETR Base + BoT-SORT (z ReID)
3. RF-DETR Base + SAM2
4. RF-DETR Seg + ByteTrack
5. RF-DETR Seg + BoT-SORT (z ReID)
6. RF-DETR Seg + SAM2

W związku z tym potrzebujemy:
- wytrenować 2 modele detekcji: RF-DETR Base oraz RF-DETR Seg
- wykonać inferencję na splicie `test` dla 6 wariantów (2×3)

### Klasy obiektów
Trenujemy i wnioskujemy dla wszystkich klas obecnych w danych SoccerNet Tracking (np. `player`, `goalkeeper(s)`, `referee`, `ball`), a wyniki raportujemy rozdzielnie dla każdej klasy oraz łącznie.

W praktyce SoccerNet Tracking trzyma klasy w `gameinfo.ini` (mapowanie `trackletID -> label`), więc do ewaluacji per-klasa mapujemy `track_id` z `gt/gt.txt` do klasy na podstawie `gameinfo.ini`.

### RF-DETR Seg: pseudo-maski
Ponieważ SoccerNet Tracking dostarcza bbox-y, a nie maski, do treningu wariantu segmentacyjnego generujemy pseudo-maskę z bbox (np. wypełniony prostokąt bbox jako maska binarna).

### Statystyki czasu
Zbieramy (wystarczy na potrzeby pracy):
- `training_time` dla każdego modelu detekcji
- `inference_time` oraz `FPS` dla każdego wariantu (detektor+tracker)

Nie rozdzielamy czasu detekcji od trackingu. Sprzęt/środowisko uzupełnimy później w artykule.

## Metryki 
Na koniec eksperymentu oceniamy jakość każdego podejścia. Używamy standardowych metryk trackingu (MOT):
- `MOTA`, `IDF1`, `HOTA`, `FP`, `FN`, `ID-switch`, `Frag`, `FPS`

### Narzędzie do ewaluacji
Używamy najbardziej powszechnego i sprawdzonego narzędzia dla MOT/HOTA/IDF1: `TrackEval`. Dane wejściowe przygotowujemy w formacie MOT (predykcje i GT), per sekwencja.

### Raportowanie per-klasa + agregaty (makro/ważone)
Ponieważ trackujemy wiele klas, wyniki raportujemy:
- per-klasa (oddzielnie dla `player`, `goalkeeper(s)`, `referee`, `ball`, …)
- łącznie jako:
  - średnia makro po klasach (każda klasa ma równą wagę)
  - średnia ważona po klasach (waga = liczba obiektów/tracków w GT dla danej klasy w `test`)

### Dodatkowa metryka: “średnia liczba zmian ID na zawodnika”
Dla każdego meczu/sekwencji liczymy wskaźnik:
- `id_ratio = (liczba unikalnych ID w predykcji) / (liczba unikalnych zawodników/obiektów w GT)`

Interpretacja: im bliżej `1.0`, tym mniej “puchnie” liczba ID względem rzeczywistej liczby obiektów. (Analogicznie raportujemy per-klasa i łącznie.)

### Metryki złożone “per track”
Dodatkowo raportujemy proste statystyki jakości tracków (per sekwencja oraz agregaty po splicie):
- średnia/mediana długość tracku (liczba klatek)
- odsetek tracków krótszych niż `N` klatek (np. `N ∈ {5, 10}`)
- rozkład długości tracków (np. percentyle `p25/p50/p75/p90` lub histogram)

### Strojenie hiperparametrów (tuning)
Dodajemy tuning parametrów trackerów, ale bez zaglądania do `test`:
- stroimy na wydzielonym podzbiorze `tune` z `train`
- wybieramy najlepszą konfigurację na `tune` (np. po `HOTA` jako metryce głównej, z `IDF1` jako drugorzędną)
- finalne liczby raportujemy na `test` dla wybranego zestawu parametrów

## SAM2: dlaczego wyniki mogą być zaniżone i plan re-eksperymentu (1 sekwencja)
Poniżej spisałem „rozeznanie środowiska” (co realnie mamy w repo) oraz plan małego, **maksymalnie dopieszczonego** eksperymentu SAM2 na **jednej sekwencji wideo** (jednym `SNMOT-*`), żeby sprawdzić czy obecne wyniki nie wynikają z suboptymalnej konfiguracji/pipeline’u.

### Co mamy już w repo (istotne dla SAM2)
- Repo zawiera lokalnie SAM2 real-time: `external/segment-anything-2-real-time/` wraz z checkpointami:
  - `external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt`
  - `external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_base_plus.pt`
  - `external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt`
  - `external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt`
- W eksperymencie trackingowym SAM2 jest uruchamiany przez `tactifoot_vision/tracking/sam2_tracker.py` (wrapper na `build_sam2_camera_predictor`), który:
  - skaluje obraz do `max_side` (np. 768/1024) i przeskalowuje prompt bbox-y,
  - prog maski ustawia na stałe `mask_logits > 0.0`,
  - ma tylko prosty filtr artefaktów (`mask_filter_distance`, connected components + odległość centroidów),
  - zwraca bbox-y przez `sv.mask_to_xyxy(masks)` (czyli jakość trackingu w MOT zależy od jakości masek i ich “tightness”).
- Skrypty eksperymentów używają obecnie **różnych konfiguracji SAM2**:
  - `scripts/run_soccernet_tracking_experiment.py` używa `sam2.1_hiera_tiny.pt` + `max_side=1024`.
  - `scripts/run_soccernet_train2seq_infer1seq.py` domyślnie używa `sam2.1_hiera_tiny.pt` + `max_side=768`, ale pozwala to nadpisać flagami `--sam2-checkpoint/--sam2-config/...`.
- Implementacja referencyjna w `basketball_ai.py` (notebook) dla „najlepszej jakości” explicite używa:
  - checkpointu **large** (`sam2.1_hiera_large.pt` + `sam2.1_hiera_l.yaml`),
  - `torch.autocast("cuda", dtype=torch.bfloat16)`,
  - prostego post-processingu masek (connected components + odległość centroidów) – podobnie jak u nas.

### Co sugerują obecne liczby (dwie różne „porażki”)
Wyniki, które mamy w `results/detection_tracking/raw/`, wyglądają jak mieszanka dwóch problemów:
1) **Reżim “detector jest słaby/za ostry próg”**: wszystkie trackery mają bardzo niskie HOTA/IDF1 i ogromne FN. Wtedy SAM2 _nie ma czego_ „uratować”, bo brak detekcji = brak obiektów do promptowania/reseedingu.
2) **Reżim “detector jest OK, SAM2 robi złe bbox-y”**: np. w `soccernet_tracking_train2seq_100ep_infer1seq` ByteTrack/BoT-SORT mają sensowne wyniki dla `player`, a SAM2 ma bardzo duże FP i (dla `player`) ujemną MOTA. Z `sanity.json` widać, że SAM2 potrafi utrzymać długie tracki, ale ich **pozycja/rozmiar** jest na tyle nietrafiony, że TrackEval liczy to jako FP/FN (złe dopasowanie do GT bbox).

### Co możemy robić lepiej (żeby „wycisnąć” SAM2)
Najbardziej podejrzane miejsca, które mogą sztucznie psuć wyniki SAM2 w obecnym pipeline:
- **Checkpoint**: w tracking’u używamy domyślnie `tiny`; referencyjny notebook (`basketball_ai.py`) dla jakości używa `large`. Na soccerze różnica jakości masek może być kluczowa.
- **Prompting / reseeding**:
  - W `run_detection.py` SAM2 jest promptowany tylko na 1. klatce (brak domyślnego „dołączania” nowych obiektów, brak korekty dryfu).
  - W eksperymentach reseeding dodaje nowe obiekty, ale bazuje na bbox-ach z poprzedniego tracku (z masek). To może wzmacniać dryf: błędny bbox → błędny prompt → jeszcze gorsza maska.
  - Lepsza strategia: okresowo **re-promptować wszystkie istniejące tracki** bbox-ami z detektora (po dopasowaniu przez IoU / Hungarian), a nie bbox-ami z masek.
- **Konwersja maska → bbox**: `sv.mask_to_xyxy` bierze „tight box” wokół maski. Jeśli maska ma artefakty albo „przylepia się” do tła, bbox staje się za duży/źle położony i metryki MOT lecą w dół.
  - Potencjalne usprawnienia: filtr po **polu maski** (min/max względem prompt bbox), usuwanie małych komponentów, morfologia (open/close), ograniczanie bbox do okna wokół poprzedniego bbox (clamp), ewentualnie próg na `mask_logits` > `t` (do strojenia).
- **Mieszanie klas w jednym trackerze**: trackowanie `player/goalkeeper/referee/ball` jednym SAM2 (jedna instancja, wiele obiektów) może pogarszać stabilność promptów. Minimalnie warto przetestować:
  - SAM2 tylko dla `player` (najważniejsza klasa),
  - osobno `ball` (z innymi parametrami / częstszym reseedingiem).
- **Detekcja jako „górny limit”**: jeżeli detektor ma niską recall, to SAM2 tego nie poprawi. Dla uczciwego testu potencjału SAM2 trzeba:
  - ustawić próg detekcji tak, żeby recall nie zabijał trackingu (w `run_soccernet_train2seq_infer1seq.py` jest już auto-picking progu – warto to stosować też w innych skryptach),
  - rozważyć „SAM2 do ID, bbox z detektora” (czyli: detektor daje bbox-y co klatkę, SAM2 służy do stabilnego przypisania ID przez mask IoU / cechy) – to jest bliżej praktycznego trackingu w MOT.

### Plan małego eksperymentu “SAM2 full potential” (jedna sekwencja)
Celem jest szybkie, ale rzetelne sprawdzenie: **czy SAM2 potrafi pobić (lub przynajmniej dorównać) ByteTrack/BoT-SORT na jednej sekwencji, jeśli damy mu najlepsze warunki**.

1) Wybór sekwencji (jedno “wideo”)
- Bierzemy pojedynczą sekwencję z `train` (żeby móc stroić bez „psucia” `test`), np. `data/soccernet/tracking/extracted/train/SNMOT-116` (albo inną, ale jedną, z pełnym GT).

2) Ustalenie detektora (żeby nie mieszać problemów)
- Najpierw wybieramy jeden, możliwie mocny detektor (np. `RF-DETR Base`), i ustawiamy próg tak, żeby nie było dominującego FN.
- W praktyce najprościej użyć skryptu, który ma auto-picking progu: `scripts/run_soccernet_train2seq_infer1seq.py`.

3) Uruchomienie “baseline” + sanity
- Odpalamy `ByteTrack` i `BoT-SORT(ReID)` na tej samej sekwencji i zapisujemy:
  - `metrics_per_class.csv`, `summary.csv`,
  - `sanity.json` + podglądy klatek (czy bbox-y są sensowne).

4) SAM2 – seria konfiguracji jakościowych (priorytet: jakość, nie FPS)
Minimalny zestaw, który powinien pokazać „potencjał”:
- **Checkpoint sweep**: `tiny` → `small` → `base_plus` → `large`.
- **Rozdzielczość**: `--sam2-max-side 1024` i (jeśli VRAM pozwala) `1280`.
- **Reseeding**: skrócić `--sam2-reseed-interval` (np. 10–15), ale zamiast tylko “dodawania nowych” obiektów docelowo chcemy test strategii „re-prompt all from detector” (wymaga dopracowania logiki reseedingu w skrypcie).
- **Klasy**: co najmniej wariant `player-only` (SAM2) jako sanity-check (czy SAM2 w ogóle trzyma zawodników dobrze).

Przykładowe uruchomienie (1 sekwencja, bez treningu, SAM2 large):
```bash
python scripts/run_soccernet_train2seq_infer1seq.py \
  --skip-training \
  --detectors base \
  --infer-root data/soccernet/tracking/extracted/train \
  --infer-sequence-index 0 \
  --max-frames 750 \
  --sam2-checkpoint external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-max-side 1024 \
  --sam2-max-objects 32 \
  --sam2-reseed-interval 15
```
Uwaga: to nadal użyje `mask_logits > 0.0` i prostego filtra komponentów – to jest „baseline jakościowy” z large checkpointem.

5) Diagnostyka „dlaczego SAM2 przegrywa” (jeśli przegrywa)
Jeśli SAM2 nadal ma duże FP/FN, wchodzimy w debug, zanim uznamy model za słaby:
- Porównać bbox-y SAM2 vs bbox-y detektora na tych samych klatkach (podgląd `preview/`).
- Sprawdzić, czy problemem jest:
  - dryf (bbox-y uciekają),
  - artefakty maski (bbox-y nagle rosną),
  - zła inicjalizacja (pierwsza klatka nie zawiera wszystkich obiektów),
  - zbyt agresywny reseeding (doklejanie obiektów, które już są).

6) Docelowa poprawka pipeline’u (jeśli hipotezy się potwierdzą)
Jeśli okaże się, że problemem jest głównie sposób użycia SAM2, następny krok to wdrożenie i ponowny pomiar na tej samej sekwencji:
- Reseeding jako **re-anchoring**: dopasuj istniejące tracki do detekcji (IoU/Hungarian) i re-promptuj bbox-ami z detektora; usuń tracki bez matcha przez `K` reseedów.
- Post-processing masek: filtrowanie po polu/kształcie + strojenie progu `mask_logits`.
- (Opcjonalnie) “SAM2 do ID, bbox z detektora”: metryki MOT liczymy na bbox z detektora, a SAM2 pomaga stabilizować przypisanie ID.

Efekt końcowy tej ścieżki: albo dostajemy konfigurację SAM2, która ma sensowne metryki na 1 sekwencji (i wtedy skalujemy na więcej), albo mamy twardy dowód, że w naszych danych/ustawieniach SAM2 nie wnosi wartości.

## Wyniki eksperymentu "SAM2 Improvement v1" (SNMOT-116, 750 klatek)

Przeprowadzono pełny eksperyment na sekwencji testowej `SNMOT-116` (750 klatek) z użyciem pełnego modelu detekcji (`rfdetr_base_soccernet2023.pth`) oraz modelu SAM2 Large z poprawkami `TightBox`.

### Tabela wyników

| Wariant | Klasa | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| ByteTrack | Player | 44.59 | 55.86 | 35.87 | 55.66 | 67.75 | 99 |
| BoT-SORT | Player | 44.31 | 57.17 | 34.62 | 54.51 | 68.08 | 95 |
| **SAM2 (Large)** | **Player** | **43.40** | **53.47** | **35.44** | **51.44** | **54.60** | **54** |
| ByteTrack | Ball | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| BoT-SORT | Ball | 0.58 | 1.06 | 0.32 | 0.53 | -3.04 | 1 |
| **SAM2 (Large)** | **Ball** | **7.54** | **9.52** | **5.97** | **11.65** | **-22.79** | **3** |

### Kluczowe wnioski

1.  **Konkurencyjność SAM2 na tej sekwencji**: Wynik HOTA dla graczy (43.4%) jest zbliżony do ByteTrack (44.6%), co wskazuje, że podejście segmentacyjne może być konkurencyjne w warunkach tego testu.
2.  **Stabilność przypisanego ID**: SAM2 zanotował niemal **2x mniej zmian ID (54 vs 99)** niż ByteTrack. To istotny sygnał stabilności trajektorii, ale nie jest samodzielnym dowodem pełnej poprawności semantycznej tożsamości zawodnika.
3.  **Śledzenie piłki**: W tym przebiegu SAM2 uzyskał wyższy HOTA dla piłki (7.54 vs <1), ale przy nadal wysokim poziomie FP (ujemna MOTA), więc wynik wymaga ostrożnej interpretacji.
4.  **Precyzja lokalizacji**: Mimo `TightBox`, DetA dla SAM2 jest nieco niższa (53.5 vs 55.9), co jest spójne z hipotezą o niedokładności masek na krawędziach obiektów.

### Rekomendacje
Dla uzyskania najlepszego systemu śledzenia w piłce nożnej sugeruje się podejście hybrydowe:
*   Użycie **SAM2 do asocjacji ID** (ze względu na niskie IDSW).
*   Użycie **RF-DETR do lokalizacji** (ze względu na wyższe DetA).
*   Dla piłki: użycie SAM2 z agresywnym filtrowaniem FP.

## Eksperyment v2: Optymalizacja Hybrydowa (Snap-to-Detector)

W celu poprawy precyzji (`DetA`) i redukcji False Positives (`MOTA`), zaimplementowano mechanizm **"Snap-to-Detector"**:
1.  **Detector Blend:** Pozycja (bbox) obiektu jest średnią ważoną z maski SAM2 i najbliższej detekcji RF-DETR (Alpha = 0.6).
2.  **Unmatched Drop:** Tracki bez potwierdzenia w detektorze przez 5 klatek są usuwane.
3.  **Rzadki Reseed:** Wydłużono interwał reseedingu do 30 klatek, aby zachować ciągłość ID (wysokie AssA), polegając na "snapowaniu" do korekty pozycji.

### Wyniki (Iteracja 5, SNMOT-116, 750 klatek)

| Wariant | Klasa | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| ByteTrack | Player | 44.59 | 55.86 | 35.87 | 55.66 | 67.75 | 99 |
| BoT-SORT | Player | 44.31 | 57.17 | 34.62 | 54.51 | 68.08 | 95 |
| **SAM2 (Optimized)** | **Player** | **44.08** | **54.59** | **35.79** | **55.00** | **64.89** | **71** |

### Podsumowanie ulepszeń
Dzięki optymalizacji hybrydowej SAM2 istotnie zbliżył się do ByteTrack (różnica HOTA < 0.5 p.p.), zachowując niższe IDSW (71 vs 99) i wyraźnie poprawiając MOTA (z 54.6 do 64.9) względem wersji "czystej".

## Eksperyment v3: Post-Processing (Track Stitching)

Jako ostatni krok optymalizacji, zastosowano algorytm **Track Stitching (GSI)**, który łączy fragmenty ścieżek na podstawie predykcji ruchu liniowego i bliskości przestrzennej. Pozwoliło to na scalenie przerwanych tracków (np. po okluzji).

### Wyniki Finalne (SNMOT-116, 750 klatek)

| Wariant | Klasa | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| ByteTrack | Player | 44.59 | **55.86** | 35.87 | 55.66 | **67.75** | 99 |
| BoT-SORT | Player | 44.31 | 57.17 | 34.62 | 54.51 | 68.08 | 95 |
| **SAM2 (Final)** | **Player** | **44.57** | 54.53 | **36.60** | **56.17** | 64.90 | **70** |

### Podsumowanie
Wdrożenie pełnego pipeline'u (Hybrid Snap + Rare Reseed + Stitching) pozwoliło SAM2 na:
1.  **Uzyskanie wyniku HOTA bardzo bliskiego ByteTrack** (różnica 0.02 p.p. na tej sekwencji).
2.  **Wyższą wartość IDF1** (56.17 vs 55.66) przy jednoczesnym obniżeniu IDSW.
3.  **Redukcję ID Switches** o ~30% względem klasycznych metod w analizowanej konfiguracji.

Interpretacja tych wyników wymaga ostrożności: zastosowane mechanizmy stabilizacji (snap, rzadki reseed, stitching, a w części eksperymentów także EMA bbox) mogą poprawiać gładkość i ciągłość trajektorii, ale nie zastępują walidacji semantycznej tożsamości względem ground truth.

Dalsza poprawa wyniku (>50%) wymaga **dotrenowania modelu detekcji**, aby zwiększyć bazowy Recall, który jest obecnie głównym ograniczeniem (DetA ~55%).

## Metryki Stabilności Trajektorii (Trajectory Stability Metrics)

W odpowiedzi na sugestie recenzentów, wprowadzamy dodatkowe metryki oceniające stabilność i spójność trajektorii. Standardowe metryki MOT (MOTA, IDF1, HOTA) nie mierzą bezpośrednio jakości ruchu ani fizycznej wiarygodności ścieżek - nowe metryki uzupełniają tę lukę.

### Zakres interpretacji: metryki proxy

Poniższe metryki należy traktować jako **metryki proxy**. Mierzą one stabilność i spójność zachowania przypisanego identyfikatora (ciągłość, dynamikę ruchu, wiarygodność fizyczną), ale **nie mierzą bezpośrednio poprawności semantycznej tożsamości względem GT ID**.

Wysokie wartości metryk proxy oznaczają lepszą jakość dynamiki trajektorii, natomiast nie są autonomicznym dowodem, że ten sam identyfikator zawsze odpowiada tej samej osobie.

### TCVR vs DRR/AOR/PPS: rozszerzenie diagnostyczne

Pierwotna metryka teleportacji TCVR okazała się na tym zbiorze zbyt gruboziarnista. DRR, AOR i PPS traktujemy więc jako **czułe, lokalne rozszerzenie diagnostyczne**, a nie bezpośredni zamiennik TCVR.

Brak dużych teleportacji nie wyklucza błędów trajektorii: problemy mogą ujawniać się na poziomie lokalnej dynamiki (oscylacje kierunku, skoki przyspieszenia, jitter), które TCVR często pomija.

### Podstawowe metryki (Reviewer Request)

#### 1. Identity Stability Ratio (ISR)
Mierzy ciągłość trajektorii - stosunek najdłuższego ciągłego segmentu do całkowitej długości trajektorii.

**Definicja:**
```
ISR = max_continuous_segment_length / total_trajectory_length
```

Gdzie segment ciągły to sekwencja kolejnych klatek bez przerwy (Δframe = 1).

**Raportowane metryki:**
- `isr_mean`: średnia ISR wszystkich trajektorii
- `isr_median`: mediana ISR
- `isr_ge_0.8`: odsetek trajektorii z ISR ≥ 0.8
- `isr_ge_0.9`: odsetek trajektorii z ISR ≥ 0.9

**Interpretacja:**
- ISR = 1.0 → brak fragmentacji (idealne śledzenie)
- ISR = 0.5 → trajektoria podzielona na co najmniej 2 segmenty
- ISR < 0.3 → poważna fragmentacja

#### 2. Occlusion Recovery Consistency (ORC)
Mierzy czy tracker wraca do spójnej pozycji po przerwach w obserwacji.

**Definicja:**
Dla każdej przerwy > k klatek w trajektorii:
- Oblicz dystans centroidów przed/po przerwie
- Jeśli dystans < próg → recovery consistent

```
ORC@k = consistent_recoveries / total_recoveries_with_gap_k
```

**Raportowane metryki:**
- `orc@15`: ORC dla przerw > 15 klatek (~0.5s @30fps)
- `orc@30`: ORC dla przerw > 30 klatek (~1s)
- `orc@60`: ORC dla przerw > 60 klatek (~2s)

**Heurystyka dystansu:**
- Próg domyślny: 100px przy szerokości obrazu 1920px (~5% szerokości)
- Przy boisku 105m → ~5m tolerancji

#### 3. Direction Reversal Rate (DRR)
Wykrywa szybkie zmiany kierunku ruchu (oscylacje), które są typowym objawem jitteru i niestabilnego śledzenia.

**Definicja:**
Zliczamy przypadki, gdy kąt między kolejnymi wektorami prędkości przekracza próg (np. 90°).

```
DRR = reversals / velocity_transitions
```

**Raportowane metryki:**
- `drr`: średni udział odwróceń kierunku (niższy = lepiej)
- `drr_tracks_affected`: odsetek trajektorii z co najmniej jednym odwróceniem
- `drr_total_reversals`: liczba wszystkich odwróceń

#### 4. Acceleration Outlier Rate (AOR)
Wykrywa nietypowe przyspieszenia (skoki) w obrębie trajektorii. Metryka adaptacyjna, liczona względem rozkładu przyspieszeń danej trajektorii.

**Definicja:**
Outlier = przyspieszenie > mean + k·std (domyślnie k=2).

```
AOR = outlier_accel_events / total_accel_events
```

**Raportowane metryki:**
- `aor`: średni udział outlierów (niższy = lepiej)
- `aor_median`: mediana AOR
- `aor_total_outliers`: liczba wszystkich zdarzeń outlier

#### 5. Physical Plausibility Score (PPS)
Sprawdza, czy trajektorie mieszczą się w fizycznych ograniczeniach ruchu człowieka.

**Założenia (domyślne):**
- max speed: 12 m/s
- max acceleration: 6 m/s²
- 1920 px ≈ 105 m → ~18.3 px/m

**Definicja:**
Trajektoria jest „plausible”, jeśli wszystkie jej klatki spełniają limity prędkości i przyspieszenia.

```
PPS = plausible_tracks / total_tracks
```

**Raportowane metryki:**
- `pps`: odsetek trajektorii fizycznie poprawnych (wyższy = lepiej)
- `pps_speed_violations`, `pps_accel_violations`
- `pps_max_speed_observed`, `pps_max_accel_observed`

### Metryki dodatkowe (Novelty)

#### 6. Motion Smoothness Score (MSS)
Mierzy gładkość ruchu na podstawie przyspieszenia (stabilna prędkość = płynny ruch).

**Definicja:**
Dla każdej trajektorii oblicz średnią wartość |Δv|:
```
acceleration = |v[i] - v[i-1]|
MSS = 1 / (1 + mean_acceleration / normalization_factor)
```

**Raportowane metryki:**
- `mss_mean`: średnia MSS wszystkich trajektorii
- `mss_median`: mediana MSS

**Interpretacja:**
- MSS bliskie 1.0 → stały, płynny ruch
- MSS < 0.5 → chaotyczny ruch (potencjalne ID switches lub jitter)

#### 7. Trajectory Consistency Index (TCI)
Zagregowana metryka łącząca ISR, DRR, AOR, PPS i ORC w jeden wskaźnik.

**Definicja:**
```
TCI = w1 × ISR_mean + w2 × (1 - DRR) + w3 × (1 - AOR) + w4 × PPS + w5 × ORC@30
```

Domyślne wagi: w1=0.25, w2=0.20, w3=0.15, w4=0.25, w5=0.15

**Interpretacja:**
- TCI bliskie 1.0 → wysokiej jakości trajektorie
- Pojedyncza liczba do porównania trackerów (wskaźnik pomocniczy)

**Komentarz metodologiczny:**
- Do TCI włączono metryki opisujące różne mechanizmy błędów (ciągłość, okluzje, kierunek, przyspieszenie, plausibility), wszystkie w zakresie 0-1.
- MSS raportujemy osobno, aby nie dublować informacji o dynamice już ujętej przez DRR/AOR.
- Wagi nie są równe: większy nacisk na ISR i PPS (po 0.25) premiuje ciągłość i fizyczną wiarygodność.
- Konsekwencja: tracker z bardzo gładkim ruchem, ale słabszą ciągłością lub plausibility, nie osiągnie wysokiego TCI.

### Implementacja

**Nowy moduł:** `tactifoot_vision/metrics/trajectory_stability.py`

```python
# Publiczne API:
compute_isr(frames_by_tid) -> dict[str, float]
compute_orc(rows, gap_thresholds, max_distance_px) -> dict[str, float]
compute_drr(rows, angle_threshold_deg, min_velocity_px) -> dict[str, float]
compute_aor(rows, sigma_threshold) -> dict[str, float]
compute_pps(rows, max_speed_m_per_s, max_accel_m_per_s2, pixels_per_meter, frame_rate) -> dict[str, float]
compute_mss(rows) -> dict[str, float]
compute_tci(isr_mean, drr, aor, pps, orc_30, weights) -> float
compute_all_stability_metrics(rows, image_width, frame_rate) -> dict[str, float]
```

**Integracja:**
- Funkcja `_track_length_stats()` rozszerzona o nowe metryki
- Nowe kolumny w `per_sequence_stats.csv`
- Nowe agregaty w `track_stats_summary.csv`
- Dedykowane wykresy w `plots/`

### Oczekiwane wyniki

Nowe kolumny w raportach:
```
isr_mean, isr_median, isr_ge_0.8, isr_ge_0.9,
orc@15, orc@30, orc@60,
drr, drr_tracks_affected, drr_total_reversals,
aor, aor_median, aor_total_outliers,
pps, pps_speed_violations, pps_accel_violations, pps_max_speed_observed, pps_max_accel_observed,
mss_mean, mss_median,
tci
```

Nowe wykresy:
- `isr_distribution_by_tracker.png` - rozkład ISR per tracker
- `orc_comparison.png` - porównanie ORC@k per tracker
- `drr_by_class.png` - DRR per klasa/tracker
- `aor_by_class.png` - AOR per klasa/tracker
- `pps_by_class.png` - PPS per klasa/tracker
- `stability_radar.png` - radar chart wszystkich metryk
- `mss_vs_idsw_scatter.png` - korelacja MSS vs ID-switches

### Potencjał dla artykułu naukowego

**Contribution 1: Trajectory Stability Framework**
Systematyczny zestaw metryk wykraczających poza standardowe MOT metrics, dedykowany dla analizy sportowej.

**Contribution 2: Motion Smoothness Score**
Nowa metryka jakości ruchu nie występująca w standardowych benchmarkach MOT.

**Contribution 3: Per-Class Stability Analysis**
Porównanie stabilności śledzenia dla różnych typów obiektów (player vs ball vs referee).

**Contribution 4: SAM2 vs Traditional Trackers - Stability Perspective**
Analiza wskazująca, że SAM2 może wykazywać lepszą stabilność (niższe IDSW, wyższe ISR) mimo podobnych wyników HOTA, przy zachowaniu ostrożności interpretacyjnej dot. semantycznej poprawności ID.
