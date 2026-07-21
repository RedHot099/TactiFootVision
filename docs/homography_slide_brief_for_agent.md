# Brief Do Slajdu: Poprawa Homografii I Radar Boiskowy

## Cel Slajdu

Pokazać, że problem homografii został precyzyjnie zdiagnozowany i zmierzony:
obecny baseline `current_yolopose_7pt` nie nadaje się do projekcji na boisko,
natomiast nowa ścieżka ewaluacji image-to-pitch potwierdza, że przy poprawnych
korespondencjach można uzyskać sub-metrową jakość radaru boiskowego.

Ważne: `oracle_gsr_lines_ransac` nie jest backendem produkcyjnym. To
diagnostyczny upper-bound i walidacja metryk. Produkcyjny kolejny krok to
podpięcie realnego backendu kalibracji, np. `pnlcalib` albo `sportlight`.

## Zmiana W Metodologii

### Przed

Pipeline używał `models/yolov8n-pose.pt`, czyli ludzkiego YOLO-pose, jako źródła
punktów do homografii. Te keypointy nie mają semantyki boiska, więc macierz była
często liczona z korespondencji geometrycznie bez sensu. Efekt: niska
dostępność homografii i bardzo duże błędy projekcji.

### Po

Oddzieliliśmy ocenę homografii od błędów detekcji i trackingu. Na SoccerNet-GSR
używamy GT footpointów w obrazie i mierzymy wyłącznie jakość transformacji
image-to-pitch względem `bbox_pitch`. Dzięki temu wiemy, czy problemem jest sama
homografia, a nie detector/tracker.

Do porównania użyliśmy:

- `current_yolopose_7pt` jako baseline historyczny;
- `oracle_gsr_lines_ransac` jako kontrolę diagnostyczną, estymowaną z poprawnych
  GSR footpoint correspondences.

## Twarde Liczby

Dataset: SoccerNet-GSR `gamestate-2024` `valid`.

Zakres:

- 58 sekwencji;
- 43,500 klatek;
- 893,017 projekcji obiektów w porównaniu.

| Metryka | Baseline `current_yolopose_7pt` | Kontrola oracle | Efekt |
| --- | ---: | ---: | ---: |
| Dostępność homografii | 23.27% | 98.10% | +74.83 pp |
| Mediana błędu | 93.70 m | 0.096 m | 973x mniej |
| P90 błędu | 179.56 m | 0.300 m | 598x mniej |
| Success@2m | 0.041% | 99.35% | +99.31 pp |
| Success@5m | 0.248% | 99.69% | +99.44 pp |

## Najważniejszy Przekaz

Obecna homografia nie jest „trochę niedokładna”; ona jest metodologicznie
błędna, bo używa keypointów ludzi jako keypointów boiska. Nowa ewaluacja
pokazuje skalę możliwej poprawy i daje bezpieczną ścieżkę do wymiany backendu
kalibracji.

## Proponowany Slajd

Tytuł:

```text
Homografia: z błędnych keypointów do wiarygodnego radaru boiskowego
```

Układ:

- Lewa strona: schemat `Frame → Footpoints → Homography → Pitch radar`.
- Prawa strona: trzy duże liczby:
  - `93.70 m → 0.096 m` median error;
  - `0.041% → 99.35%` Success@2m;
  - `23.27% → 98.10%` availability.
- Stopka/caveat: `Oracle = upper-bound / sanity check, not production backend`.

## Nagranie Do Osadzenia

Użyj filmu:

```text
presentation_assets/video/homography_radar_pipeline_minimap.mp4
```

Film pokazuje:

- obraz źródłowy SoccerNet-GSR;
- pipeline'owe adnotacje zawodników na obrazie;
- przepływ `frame → tracks/footpoints → homography → minimap`;
- radar boiskowy generowany przez istniejący renderer
  `tactifoot_vision.visualization.video.SoccerPitchMinimap`;
- brak dolnego panelu metryk, żeby wideo było czystsze do osadzenia na
  slajdzie.

Opis do slajdu/video:

```text
Radar pokazuje zweryfikowaną projekcję image-to-pitch. Używamy oracle-control
do walidacji geometrii i metryk; produkcyjny backend kalibracji będzie następnym
krokiem integracji.
```
