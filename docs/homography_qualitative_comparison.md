# Qualitative Homography Backend Comparison

Date: 2026-05-25

## Cel

Ten dokument porównuje jakościowo wszystkie metody z planu eksperymentu:

- `current_yolopose_7pt`
- `tvcalib`
- `sportlight`
- `soccersegcal`
- `pnlcalib`
- `auxflow`
- `oracle_gsr_lines_ransac`

To nie jest ranking numeryczny dla metod zewnętrznych. Numery mamy tylko dla
`current_yolopose_7pt` i `oracle_gsr_lines_ransac`, bo tylko te dwie metody
zostały dotąd wykonane lokalnie na SoccerNet-GSR `valid`. Pozostałe metody są
ocenione jakościowo pod kątem architektury, ryzyka wdrożenia, spodziewanej
odporności i dopasowania do SoccerNet-GSR.

## Szybki Wniosek

Najbardziej sensowna ścieżka produkcyjna to:

1. **`pnlcalib` jako pierwszy kandydat single-frame**: najlepszy balans między
   detekcją punktów, detekcją linii i nieliniowym refinementem.
2. **`sportlight` jako mocny benchmark challenge-grade**: powinien być bardzo
   dobry jakościowo, ale jest cięższy operacyjnie i bardziej związany z
   pipeline'em zwycięskiego rozwiązania SoccerNet Calibration 2023.
3. **`auxflow` jako warstwa temporalna nad najlepszym single-frame backendem**:
   nie powinien zastępować anchor calibration; powinien poprawiać dostępność i
   stabilność między anchorami.
4. **`soccersegcal` jako dobry test klasy line-segmentation + optimizer**:
   potencjalnie dokładny, ale zależny od jakości segmentacji linii i kosztowny
   przez optymalizację.
5. **`tvcalib` jako geometrycznie interpretowalny baseline kamerowy**:
   wartościowy punkt odniesienia, ale prawdopodobnie mniej dopasowany do GSR niż
   nowsze points-and-lines podejścia.

`current_yolopose_7pt` nie powinien być dalej rozwijany produkcyjnie. Lokalnie
osiągnął `availability=23.27%`, `median_error_m=93.70` i `success@2m=0.041%` na
pełnym `valid`, co potwierdza błąd koncepcyjny: ludzkie keypointy YOLO-pose są
traktowane jak punkty boiska.

## Kategorie Metod

| Kategoria | Metody | Sens porównania |
| --- | --- | --- |
| Historyczny baseline | `current_yolopose_7pt` | Pokazuje aktualny błąd i dolny punkt odniesienia. |
| Single-frame calibration | `tvcalib`, `sportlight`, `soccersegcal`, `pnlcalib` | Bezpośrednio estymują kalibrację/homografię z pojedynczej klatki lub obrazu. |
| Temporal propagation | `auxflow` | Poprawia spójność i dostępność między anchor-frame'ami; zależy od jakości anchorów. |
| Diagnostyczny upper bound | `oracle_gsr_lines_ransac` | Waliduje parser, metryki i geometrię; nie jest kandydatem produkcyjnym. |

## Macierz Jakościowa

Skala: `low`, `medium`, `high`, `very high`. Ocena jest jakościowa i powinna być
zweryfikowana pełnym eksperymentem po wygenerowaniu artefaktów.

| Method | Expected Accuracy | Availability | Temporal Stability | Integration Risk | Runtime Risk | GSR Fit | Production Readiness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `current_yolopose_7pt` | very low | low | low | low | low | very low | reject |
| `tvcalib` | medium | medium | low-medium | medium | medium | medium | baseline candidate |
| `sportlight` | high | medium-high | medium | high | high | high | strong candidate after artifact |
| `soccersegcal` | medium-high | medium | medium | high | high | medium-high | diagnostic/candidate |
| `pnlcalib` | high | high | medium | medium-high | medium-high | very high | first production candidate |
| `auxflow` | anchor-dependent | high | high | medium-high | medium | very high | temporal layer, not standalone |
| `oracle_gsr_lines_ransac` | very high | high | medium-high | low | low | oracle-only | diagnostic only |

## Metoda Po Metodzie

### `current_yolopose_7pt`

**Charakter:** obecny baseline w repo, uruchamiany przez `PitchProjector` z
`models/yolov8n-pose.pt`.

**Mocne strony:**

- Działa lokalnie bez dodatkowych repozytoriów.
- Jest szybki do uruchomienia i dobrze pokazuje stan obecnego pipeline'u.
- Może pozostać historycznym testem regresji.

**Słabe strony:**

- Model wykrywa ludzkie keypointy, nie semantyczne punkty boiska.
- Homografia bywa liczona z korespondencji bez sensu geometrycznego.
- Nawet gdy zwraca macierz, projekcje są w praktyce bezużyteczne.

**Typowe failure modes:**

- Brak wystarczającej liczby keypointów.
- Homografia dostępna, ale z błędem dziesiątek lub setek metrów.
- Niestabilne przeskoki między klatkami.

**Decyzja:** odrzucić produkcyjnie.

### `tvcalib`

**Charakter:** metoda traktująca rejestrację boiska jako problem kalibracji
kamery, a nie tylko płaskiej homografii. Repo opisuje inference jako połączenie
segmentacji semantycznej, wyboru punktów, estymacji parametrów kamery i
wizualizacji.

**Mocne strony:**

- Bardziej interpretowalny model kamery niż czysta macierz homografii.
- Może oceniać 2D z homografii albo 3D z parametrów kamery.
- Ma mechanizmy typu self-verification/loss threshold, co może pomóc odrzucać
  niepewne klatki.

**Słabe strony:**

- Starsze podejście względem `pnlcalib` i `sportlight`.
- Adaptacja do GSR wymaga dopilnowania kierunku transformacji
  image-to-pitch.
- Jeśli widoczność linii jest słaba, pojedyncza klatka może być trudna.

**Najlepiej używać jako:**

- Solidny geometryczny baseline zewnętrzny.
- Punkt odniesienia dla pytania: czy pełna kalibracja kamery daje korzyść nad
  prostą homografią.

**Ryzyko wdrożenia:** średnie. Trzeba przekonwertować parametry kamery lub
homografię do wspólnego formatu.

### `sportlight`

**Charakter:** zwycięskie rozwiązanie SoccerNet Camera Calibration 2023. Według
repo wykorzystuje osobne modele dla keypointów i linii oraz heurystyki do
wyprowadzenia najbardziej prawdopodobnych parametrów kamery.

**Mocne strony:**

- Najsilniejszy challenge-grade kandydat single-frame.
- Łączy punkty, linie, geometrię i heurystyki.
- Dobre dopasowanie do SoccerNet Calibration i prawdopodobnie dobre przeniesienie
  na GSR, jeśli wejścia zostaną poprawnie przygotowane.

**Słabe strony:**

- Wysokie wymagania operacyjne: Docker, NVIDIA GPU, duża pamięć VRAM.
- Większy pipeline oznacza więcej miejsc, w których może pęknąć format danych.
- Heurystyki mogą być dopasowane do challenge'u calibration-2023, niekoniecznie
  idealnie do wszystkich klipów GSR.

**Najlepiej używać jako:**

- Główny mocny benchmark jakościowy.
- Kandydat do porównania z `pnlcalib` w kategorii "najlepsza single-frame
  kalibracja".

**Ryzyko wdrożenia:** wysokie, głównie przez środowisko i oczekiwany format
danych.

### `soccersegcal`

**Charakter:** pipeline dwustopniowy: segmentacja linii boiska, a potem optimizer
oparty o differentiable rendering do estymacji parametrów kamery.

**Mocne strony:**

- Bardzo czytelna struktura: najpierw maski/segmenty linii, potem optymalizacja.
- Dobrze nadaje się do analizy jakości: widać, czy problemem jest segmentacja,
  czy optimizer.
- Może być mocny w scenach, gdzie linie są dobrze widoczne, ale keypointy
  punktowe są niejednoznaczne.

**Słabe strony:**

- Optymalizacja może być kosztowna i wrażliwa na inicjalizację.
- Segmentacja linii może zawieść przy zasłonięciach, słabej jakości obrazu,
  reklamach, cieniach i nietypowym kadrowaniu.
- Repo jest mniej rozbudowane operacyjnie niż zwycięskie rozwiązania challenge.

**Najlepiej używać jako:**

- Kandydat diagnostyczny dla klasy "line segmentation + optimizer".
- Metoda, która może pomóc zrozumieć, czy linie boiska wystarczą na GSR.

**Ryzyko wdrożenia:** wysokie, głównie przez zależności typu PyTorch3D i
koszt/refinement.

### `pnlcalib`

**Charakter:** points-and-lines calibration. Repo opisuje ulepszone modele
detekcji keypointów i ekstremów linii oraz refinement wykorzystujący linie w
nieliniowej optymalizacji.

**Mocne strony:**

- Najlepsze dopasowanie do naszego celu: stabilna homografia image-to-pitch na
  podstawie punktów i linii.
- Ma osobne modele dla keypointów i linii oraz refinement geometryczny.
- Powinien być odporniejszy niż czyste keypoint-only lub line-only podejścia.
- SoccerNet GSR repo oficjalnie wspomina dodanie `pnlcalib` jako opcji
  kalibracji.

**Słabe strony:**

- Trzeba pilnować licencji i izolacji środowiska.
- Wymaga pobrania właściwych wag i dopasowania single-view vs multi-view trybu.
- Może być wrażliwy na błędne detekcje linii, choć refinement powinien to
  częściowo kompensować.

**Najlepiej używać jako:**

- Pierwszy kandydat produkcyjny do pełnego uruchomienia.
- Anchor backend dla `auxflow`, o ile smoke/full wyniki potwierdzą jakość.

**Ryzyko wdrożenia:** średnio-wysokie, ale najlepszy stosunek ryzyka do
oczekiwanej jakości.

### `auxflow`

**Charakter:** temporalna propagacja homografii z anchor frames przez optical
flow i pomocnicze punkty. Według opisu publikacji metoda identyfikuje
wysokiej-pewności anchor frames, a potem propaguje korespondencje do sąsiednich
klatek przez optical flow.

**Mocne strony:**

- Bezpośrednio adresuje problem GSR: spójność i dostępność homografii w czasie.
- Może znacznie poprawić klatki, w których single-frame backend nie widzi dość
  linii/punktów.
- Naturalny kandydat do redukcji jittera.

**Słabe strony:**

- Jakość zależy od anchorów; złe anchory propagują błąd.
- Optical flow może dryfować przy cięciach, szybkich panoramach, motion blur i
  dużych zasłonięciach.
- Nie powinien być oceniany jako czysty single-frame backend, tylko jako warstwa
  nad najlepszą metodą anchorową.

**Najlepiej używać jako:**

- Warstwa temporalna nad `pnlcalib` albo `sportlight`.
- Kandydat do poprawy `availability` i `temporal_jitter`.

**Ryzyko wdrożenia:** średnio-wysokie. Największe ryzyko to polityka anchorów i
detekcja scen, w których propagację trzeba zresetować.

### `oracle_gsr_lines_ransac`

**Charakter:** kontrola diagnostyczna liczona z GT GSR image/pitch footpoint
correspondences. Nazwa historyczna mówi "lines", ale aktualna implementacja
lokalna używa GT footpointów zawodników.

**Mocne strony:**

- Bardzo dobrze waliduje parser, metryki i kierunek transformacji.
- Ujawnia, jaki błąd wynika z samej estymacji homografii przy dobrych
  korespondencjach.
- Lokalnie osiągnął `median_error_m=0.0963` i `success@2m=99.35%`.

**Słabe strony:**

- Korzysta z GT pitch positions, więc nie może być kandydatem produkcyjnym.
- Może ukrywać problemy, których realny backend nie rozwiąże, np. brak
  widocznych linii.

**Decyzja:** zostawić jako sanity check i upper-bound, wykluczyć z rankingu
produkcyjnego.

## Oczekiwane Zachowanie Na SoccerNet-GSR

| Sytuacja | Najbardziej obiecujące metody | Największe ryzyka |
| --- | --- | --- |
| Wiele widocznych linii boiska | `pnlcalib`, `sportlight`, `soccersegcal` | Segmentacja może pomylić reklamy/cienie z liniami. |
| Mało linii, ale dobra sekwencja temporalna | `auxflow` nad `pnlcalib`/`sportlight` | Drift optical flow, reset po cięciu kamery. |
| Nietypowy kadr lub zoom | `pnlcalib`, `sportlight` | Heurystyki challenge-specific mogą źle wybrać kamerę. |
| Duże zasłonięcia graczami | `pnlcalib`, `auxflow` | Za mało pewnych korespondencji w anchor frame. |
| Chcemy szybki baseline | `tvcalib` | Niższa jakość względem nowszych points-and-lines metod. |
| Chcemy upper-bound jakości ewaluacji | `oracle_gsr_lines_ransac` | Niedopuszczalne jako metoda produkcyjna. |

## Priorytet Uruchomienia

1. **`pnlcalib`**
   - Najlepsze dopasowanie do naszej metryki i danych.
   - Powinien dać sensowny pierwszy produkcyjny kandydat.

2. **`sportlight`**
   - Najmocniejszy challenge-grade benchmark.
   - Warto go uruchomić zaraz po `pnlcalib`, żeby sprawdzić, czy bardziej
     złożone heurystyki faktycznie wygrywają na GSR.

3. **`auxflow`**
   - Uruchomić dopiero po wyborze najlepszego nie-oracle anchor backendu.
   - Oceniać szczególnie po `availability`, `temporal_jitter` i `p90_error_m`.

4. **`soccersegcal`**
   - Dobry kontrast metodologiczny: segmentation + differentiable rendering.
   - Przydatny do analizy failure cases.

5. **`tvcalib`**
   - Wartościowy baseline zewnętrzny, ale mniejsze prawdopodobieństwo, że wygra
     z nowszymi `pnlcalib`/`sportlight`.

## Jak Będziemy Porównywać Jakościowo Po Artefaktach

Po wygenerowaniu `homographies.parquet` dla każdej metody, oprócz rankingu
numerycznego trzeba obejrzeć:

- 25 najgorszych klatek per metoda według `p90_error_m`.
- Klipy side-by-side:
  - `current_yolopose_7pt` vs najlepszy kandydat.
  - najlepszy kandydat vs `oracle_gsr_lines_ransac`.
  - najlepszy single-frame backend vs `auxflow`.
- Mapę sektorową biasu `pitch_x/pitch_y`.
- Przypadki `available` z bardzo dużym błędem, bo są gorsze produkcyjnie niż
  uczciwe `unavailable`.
- Sekwencje z niską dostępnością, osobno od sekwencji z dużym błędem.

## Kryterium Rekomendacji Produkcyjnej

Metoda może być rekomendowana do integracji, jeśli:

- ma `availability` wyraźnie powyżej `current_yolopose_7pt`;
- ma `success@2m` istotnie powyżej baseline;
- nie ma dużych stabilnych biasów po sektorach boiska;
- failure cases są zrozumiałe i możliwe do wykrycia;
- runtime mieści się w docelowym pipeline albo można ją batchować offline;
- nie używa GT `bbox_pitch` ani oracle anchorów.

## Źródła

- SoccerNet-GSR task: https://www.soccer-net.org/tasks/game-state-reconstruction
- SoccerNet GSR repo and v1.3 note: https://github.com/SoccerNet/sn-gamestate
- TVCalib repo: https://github.com/MM4SPA/tvcalib
- TVCalib paper: https://arxiv.org/abs/2207.11709
- Sportlight repo: https://github.com/NikolasEnt/soccernet-calibration-sportlight
- Sportlight paper: https://arxiv.org/abs/2410.07401
- SoccerSegCal repo: https://github.com/Spiideo/soccersegcal
- PnLCalib repo: https://github.com/mguti97/PnLCalib
- PnLCalib paper: https://arxiv.org/abs/2404.08401
- AuxFlow paper page: https://www.sciencedirect.com/science/article/pii/S1077314226000299

