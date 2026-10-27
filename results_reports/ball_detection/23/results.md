# ball-detection 23: odporność filtra 45 px na rozdzielczość i skalę

Skrypt `scripts/ball_filter_scale.py` (+ `tests/test_ball_filter_scale.py`); wyniki i zrzuty:
`/mnt/storage/tactifoot_research/detector_data/ball/23/` (`read.md`, `read.json`, `dump/`). Rejestracja: komentarz w
tickecie 23 (przed odczytem). Gry 2+3 GSR valid, 39 klipów, 4 250 klatek placement; dd2c 1280 bez zmian; skale 1080p
(natywna), 720p, 540p (`cv2.resize` INTER_AREA przed detekcją, imgsz 1280 przy każdej). Filtr w pikselach 1080p
(dłuższy bok × 1080 / wysokość klatki). Przedziały: bootstrap po klipach (2 000 losowań). GPU: ~12 min łącznie
(zrzut gier 2+3 w 3 skalach: 17,4 / 16,7 / 21,5 ms na klatkę; zrzut klatek xG 720p), z blokadą `agent-state gpu`.

## Kontrole
- Skala natywna = zrzut 02: ramki top-1 ≥ 0,15 identyczne na 3 490 klatkach (różnica maks. 0,0 px), brak klatek tylko po
  jednej stronie; placement `none` = `ball/17/placement_games_2_3_top1.parquet`: 4 250 / 4 250 wierszy, 0 różnic w błędzie.
- xG: top-1 piłki w 19 461 klatkach identyczny z kandydatami z 16 (conf i ramka, różnica 0). Odstępstwo od rejestracji
  (punkt iii: „liczności kandydatów”): niższe kandydaty różnią się (16: NMS IoU 0,5, ten przebieg: 0,7 jak na GSR; 16 938
  vs 23 664 kandydatów ≥ 0,15), więc odsetki xG liczone na zamrożonych kandydatach z 16, a kontrolą jest zgodność top-1.
  Klatek xG 19 461 (nie 21 186: tyle jest w zrzucie 16).

## Gry 2+3: dobrze umieszczone top-1 (≤ 5 m, % wszystkich klatek placement)
| skala | none | abs45 | rel (k = 0,474) | abs45 − none, pp | > 5 m wśród wszystkich: abs45 − none, pp | rel − abs45, pp |
|---|---|---|---|---|---|---|
| 1080p | 51,6 [44,5; 59,3] | 52,2 [45,2; 59,8] | 52,3 [45,2; 59,8] | +0,6 [+0,1; +1,2] | −6,1 [−9,1; −3,4] | +0,02 [0,0; +0,1] |
| 720p | 51,7 [44,6; 59,4] | 52,4 [45,4; 59,9] | 52,4 [45,4; 60,0] | +0,7 [+0,1; +1,2] | −6,1 [−9,1; −3,4] | +0,02 [0,0; +0,1] |
| 540p | 52,1 [45,0; 59,7] | 52,7 [45,8; 60,1] | 52,7 [45,8; 60,1] | +0,6 [0,0; +1,1] | −6,3 [−9,4; −3,5] | +0,02 [0,0; +0,1] |

Prawdziwe piłki usunięte przez abs45: 0,0 % (górna granica 0,1 %) w każdej skali; duże fałszywe usunięte 97,9 % / 98,1 % /
97,0 % (1080p / 720p / 540p). Piłka prawdy: mediana dłuższego boku 14 px w 1080p, 9,3 px w 720p, 7 px w 540p, filtr jej
nie dotyka w żadnej skali, bo przelicza progi na 1080p.

## k względnego filtra
k0 = 0,474 (45 px / mediana wysokości ludzi 95,0 px w klatkach placement gier 2+3). Siatka zarejestrowana (0,6–2 × k0)
na skali natywnej: dobrze umieszczone 50,7 / 52,2 / 52,3 / 52,2 / 52,1 / 52,0 %; wygrywa 1,0 × k0 (k = 0,474), czyli
dokładnie odpowiednik 45 px dla mediany sceny; 0,6 k0 zaczyna usuwać prawdziwe piłki (2,2 %). Dobór w próbie (te same
klatki co porównanie natywne). Siatka daje ten sam kształt przy 720p i 540p.

## Realna różnorodność skali
- Gry 2+3, terciły 13 klipów wg mediany wysokości ludzi (kwantyle klipów 62 / 89 / 106 / 143 px):
  | grupa | abs45 − none, pp | > 5 m wśród wszystkich: abs45 − none, pp | rel − abs45, pp |
  |---|---|---|---|
  | małe osoby (szerokie ujęcie) | +0,8 [−0,4; +2,1] | −12,9 [−18,6; −7,5] | 0,0 |
  | średnie | +1,2 [+0,4; +2,0] | −5,2 [−8,4; −2,1] | 0,0 |
  | duże osoby (zbliżenie) | −0,1 [−0,2; 0,0] | 0,0 | +0,1 [0,0; +0,2] |
  Filtr 45 px pomaga tam, gdzie ludzie są mali (fałszywe duże ramki), a w grupie dużych ludzi nic nie zmienia i usuwa 0,1 %
  prawdziwych piłek (rel: 0,0 %). Wyniki te same w 540p.
- Klatki strzałów xG (19 461, bez prawdy): mediana wysokości ludzi 115 px (5 %–95 %: 86–153 px; GSR 95), czyli ujęcia
  bliższe niż gry 2+3. Odsetek kandydatów ≥ 0,15 usuniętych przez abs45 2,2 %, przez rel 1,1 %; wg tercylów wysokości
  ludzi (małe / średnie / duże): abs45 1,0 / 2,0 / 3,7 %, rel 1,0 / 1,6 / 0,6 %. Top-1 usunięty lub zmieniony: abs45 1,9 %,
  rel 0,8 % klatek; w tercylu dużych ludzi abs45 usuwa top-1 w 3,3 % klatek, rel w 0,2 %. Rozbieżność filtrów w 1,2 %
  kandydatów. Bez prawdy nie wiadomo, które usunięcia były trafne.
- Ekstraklasa dev (56 obrazów, 720p, opisowo): abs45 usuwa 0 z 53 kandydatów ≥ 0,15, rel 2 (top-1 w 1 z 33 klatek).

## Odczyt (reguła z rejestracji)
- „45 px działa przy skali s”: 1080p i 720p tak; 540p formalnie nie (dolna granica abs45 − none = 0,0 pp, nie > 0), przy
  tej samej wielkości efektu (+0,56 pp) i tym samym spadku > 5 m (−6,3 pp [−9,4; −3,5]); usunięte prawdziwe piłki bez
  różnicy względem natywnej. Nie ma oznak, że filtr przestaje działać przy niższej rozdzielczości.
- „Względny odporniejszy”: nie. rel − abs45 ≤ +0,02 pp, dolna granica 0 w każdej skali; k wybrane na skali natywnej to
  odpowiednik 45 px, więc na grach 2+3 oba filtry są praktycznie tym samym. Różnica ujawnia się tylko na klatkach xG o
  dużych ludziach (zbliżenia), gdzie 45 px odrzuca więcej kandydatów, ale bez prawdy nie ma rozstrzygnięcia.
- Wniosek: filtr 45 px zostaje; odporny na rozdzielczość (konstrukcja i pomiar). Otwarta pozostaje skala sceny większa niż w
  grach 2+3 (zbliżenia): tam brak prawdy piłki, wariant względny nieuprawniony, ale do sprawdzenia, gdy pojawi się taka
  prawda (SoccerTrack v2 / ticket 22 odpada wg 21).

## Ograniczenia
Dobór k w próbie; remisowa natura siatki (wybór = odpowiednik 45 px); jedna metoda przeskalowania (INTER_AREA) i
detektor dd2c na 1280 przy każdej skali (540p jest powiększane przez letterbox); GSR gry 2+3 wąska różnorodność scen
(klipy GSR), top-1 bez śledzenia (filtr w roli fazy 1 działa przed `track_ball`, tu mierzony na top-1); bez symulowanego zoomu.
