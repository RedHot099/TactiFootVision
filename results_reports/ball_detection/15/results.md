# ball-detection 15: trening i inferencja w 1600 px na przepisie B

2026-10-26. Odczyt tylko na grach 2+3 GSR valid (39 klipów, 4 250 klatek mianownika). Rejestracja: komentarz ticketu
15 z 18:57, zapisany przed treningiem (commit a523afd). Zgoda Jakuba: komentarze orkiestratora 15:45 i 15:50. Kod:
`scripts/ball_1600.py` (widok, odczyt), `RUNS["bd15_1600"]` w `scripts/detector_train.py`, wiersz 1600 w
`detector_bench.ROWS` i `ball_temporal.py dump --variant 1600`, testy `tests/test_ball_1600.py`. Duże wyniki leżą pod
`/mnt/storage/tactifoot_research/detector_data/ball/15/`, a wagi pod `runs/bd15_1600`.

**Przepis:** B z 14, a jedyna zmiana to rozdzielczość. Start z dd2c, **imgsz 1600** w treningu i inferencji, 1 epoka,
`close_mosaic` 0, `warmup_epochs` 0, lr0 0,0005, EMA, punkt kontrolny `last.pt`. Lista jest listą B bajt w bajt
(105 060 linii, sha256 3d7cbe201b8b…, bez gry 4). Batch 4 przy 1600 nie mieści się w 16 GB, więc trening szedł z
batchem 3. lr0 zostało bez zmian, bo Ultralytics akumuluje gradienty do batcha nominalnego 64 (21 × 3 = 63 obrazy na
krok wobec 16 × 4 = 64 w B). Została 1 epoka, żeby 1600 różniło się od B tylko rozdzielczością (uzasadnienie w
rejestracji).

**Pipeline odczytu:** taki sam jak w 14: zrzut 25 fps (wiersz 1600: natywna klatka, imgsz 1600, NMS IoU 0,7), filtr
45 px, `track_ball` z ustawieniem 04, ruch kamery 04, środek ramki. Kontrola: tory B przeliczone kodem `read` są
identyczne z torami 14, a wiersze B (−0,61 pp względem fazy 1, +0,31 pp na pojedynczej klatce) zgadzają się z 14 co
do cyfry.

## Wynik

- **Porównanie główne 1600 − B 1280 (pipeline, gry 2+3): +1,01 pp [−1,12; +3,15].** Dolna granica jest < 0, więc
  nie da się powiedzieć, że 1600 jest lepsze od B 1280.
- **Ticket nie wygrywa.** Względem kandydata fazy 1: 71,65% wobec 71,25%, **+0,40 pp [−1,60; +2,35]**. Bariera jest
  spełniona (> 5 m wśród wszystkich −0,28 pp), strażnicy też, ale dolna granica jest < 0.
- **Koszt mieści się w limicie:** 25,9 ms/klatkę w zrzucie 25 fps (B 17,6) i 28,5 ms w karcie, przy limicie 40 ms.
- **Wniosku o P2 nie ma.** Punktowo zysk na klatkę jest największy w piłkach < 8 px (+8,22 pp), ale przedział
  obejmuje 0 [−1,14; +16,13]. To tylko 146 klatek (3,4% mianownika), więc warunek z rejestracji nie jest spełniony.
- **Na pojedynczej klatce 1600 przegrywa na barierze:** +0,59 pp [−0,71; +2,09] względem top-1 B, ale odsetek
  pozycji > 5 m rośnie o +3,53 pp [+2,06; +5,19]. Pomyłki pochodzą z gry 3, gdzie sędziowie są w czerni. Tam filtr
  45 px i śledzenie odfiltrowują prawie wszystkie z nich.

| wiersz (gry 2+3) | odniesienie | wariant | zysk, pp [95%] | > 5 m wśród wszystkich, Δ pp [95%] | bariera |
|---|---|---|---|---|---|
| **1600 − B 1280, pipeline (główne)** | 70,64% | 71,65% | **+1,01 [−1,12; +3,15]** | +0,54 [−1,04; +2,04] | tak |
| **1600 − faza 1, pipeline (reguła 11)** | 71,25% | 71,65% | **+0,40 [−1,60; +2,35]** | −0,28 [−1,75; +1,43] | tak |
| B − faza 1, pipeline (wiersz 14) | 71,25% | 70,64% | −0,61 [−2,00; +0,69] | −0,82 [−2,82; +1,02] | tak |
| 1600 − top-1 B, pojedyncza klatka | 51,91% | 52,49% | +0,59 [−0,71; +2,09] | +3,53 [+2,06; +5,19] | **nie** |
| 1600 − top-1 dd2c, pojedyncza klatka | 51,60% | 52,49% | +0,89 [−0,47; +2,22] | +1,79 [+0,41; +3,29] | **nie** |
| B − top-1 dd2c (wiersz 14) | 51,60% | 51,91% | +0,31 [−0,59; +1,21] | −1,74 [−2,62; −0,95] | tak |

Wiersze pojedynczej klatki to top-1 ≥ 0,15 bez filtra. Karta `ball_scorecard.py card --rows 1600` daje te same liczby
co zrzut 25 fps (+0,89 wobec dd2c, +0,59 wobec B).

## Rozbicie po grach (opisowo)

| wiersz | gra 2 | gra 3 |
|---|---|---|
| 1600 − B 1280, pipeline | **+2,48 [+0,57; +4,72]** | −0,13 [−3,64; +3,36] (> 5 m +1,96) |
| 1600 − faza 1, pipeline | **+2,05 [+0,15; +4,35]** | −0,88 [−3,92; +2,15] |
| 1600 − top-1 B | +2,00 [+1,02; +3,05] | −0,50 [−2,57; +1,89] (> 5 m +6,09) |
| 1600 − top-1 dd2c | +2,16 [+1,31; +3,08] | −0,08 [−2,46; +2,05] (> 5 m +3,04) |

W 14 ten sam wzór wyglądał odwrotnie. Tam negatywy pomogły na grze 3, a na grze 2 nic nie zmieniły. Tutaj 1600 daje
wyraźny zysk na grze 2 (dolna granica > 0 w każdym wierszu), a na grze 3 zysku nie ma. Na grze 3 pojawia się
natomiast więcej pewnych, złych pozycji.

## Rozbicie po wielkości piłki (dłuższy bok prawdziwej piłki, px przy 1080p)

| wielkość | klatki | 1600 − B, pipeline | 1600 − faza 1, pipeline | 1600 − top-1 B | netto klatek (1600 − B, pipeline) |
|---|---|---|---|---|---|
| < 8 | 146 | +8,22 [−1,14; +16,13] | +2,74 [−11,22; +11,70] | +0,00 [−3,81; +3,11] | +12 |
| 8–14 | 2 203 | +1,32 [−2,03; +4,41] | +1,09 [−2,18; +4,33] | +1,54 [−0,44; +3,60] | +29 |
| > 14 | 1 901 | +0,11 [−1,89; +2,23] | −0,58 [−2,83; +1,60] | −0,47 [−1,81; +1,23] | +2 |

Z 43 klatek netto zysku względem B 41 przypada na piłki ≤ 14 px. Zysk na klatkę wynosi 0,082 w < 8 px i 0,008 w
pozostałych. Kierunek pasuje więc do hipotezy o małych piłkach, ale żaden koszyk nie ma dolnej granicy > 0. Reguła
zarejestrowana dla P2 wymagała dolnej granicy > 0 w < 8 px, więc nie jest spełniona. Karta mówi to samo o recall
(środek, ≥ 0,15, 1600 wobec B 1280): < 8 px 0,085 wobec 0,071, 8–12 px 0,417 wobec 0,383, 12–16 px 0,545 wobec 0,523,
16–24 px 0,751 wobec 0,752.

## Filtr 45 px przy inferencji 1600

Ramki w zrzucie są w pikselach natywnej klatki 1920×1080 dla każdego imgsz, bo Ultralytics przeskalowuje wynik do
oryginału. Filtr liczy dłuższy bok × 1080 / wysokość klatki, więc reguła się nie zmienia. Sprawdzenie na zrzutach
(kandydaci piłki ≥ 0,15, wszystkie klatki gier 2+3 z jedną piłką):

| zrzut | imgsz | kandydaci | na prawdziwej piłce | dłuższy bok / prawda, mediana [p5; p95] | usunięte: wszystkie / na piłce / poza piłką |
|---|---|---|---|---|---|
| B | 1280 | 16 326 | 12 946 | 1,050 [0,835; 1,463] | 12,81% / 0,062% / 61,63% |
| 1600 | 1600 | 18 057 | 13 314 | 1,046 [0,845; 1,475] | 18,28% / 0,045% / 69,45% |

Ramki na prawdziwej piłce mają przy 1600 tę samą wielkość względem prawdy co przy 1280. Filtr nie usuwa prawie żadnej
prawdziwej piłki (0,045%). Częściej usuwa za to fałszywe piłki, bo przy 1600 jest więcej dużych fałszywych kandydatów,
głównie na sędziach gry 3.

## Fałszywe piłki na sędziach (opisowo, gra 3, na klatkę)

| | surowe ≥ 0,15 | po filtrze 45 px | piłka śledzona |
|---|---|---|---|
| dd2c | 0,204 | 0,0001 | 0,0004 |
| B | 0,158 | 0,0006 | 0,0006 |
| 1600 | **0,238** | 0,0006 | 0,0022 |

Przy 1600 korzyść z negatywów B znika: surowych fałszywych piłek na sędziach jest więcej niż przy dd2c. Negatywy były
wycinkami w powiększeniu 1,25–2×, a przy 1600 sędziowie są o 25% więksi. Filtr usuwa prawie wszystkie te pomyłki,
ale tracker wybiera sędziego 4× częściej niż przy B (0,22% klatek).

## Karty i strażnicy (gry 2+3; Ekstraklasa dev)

| | mAP50-95 ludzi (Δ vs dd2c) | Ekstraklasa mAP50-95 (Δ) | AP50 piłki (Δ) | Eks. AP50 piłki | fałszywe / klatkę ≥ 0,15 | ms/klatkę (karta) |
|---|---|---|---|---|---|---|
| dd2c 1280 | 0,5137 | 0,4240 | 0,296 | | 0,208 | 19,6 |
| B 1280 | 0,5382 (+0,0245) | 0,4163 (−0,0077) | 0,304 (+0,008) | 0,286 | 0,190 | 19,7 |
| 1600 | 0,5557 (+0,0420) | 0,4115 (−0,0125) | 0,307 (+0,011) | 0,317 | 0,245 | 28,5 |

Strażnicy 1600 względem dd2c są spełnieni (ludzie ≥ −0,01, Ekstraklasa ≥ −0,02). Względem B: ludzie +0,0175,
Ekstraklasa −0,0048, AP50 piłki +0,003. Klatki Ekstraklasy mają 720p, więc 1600 oznacza tam powiększenie.

## Wagi

- 1600: `runs/bd15_1600/yolo11l_1600/weights/last.pt`, sha256
  `e15c60d5c07395e9e23acae345a4b8fd4adeb2d02d98e4a522124cafcf58aaab`

Wyniki: zrzut `ball/15/1600/dump25`, karty `ball/15/1600/card` (vs dd2c) i `ball/15/1600_vs_B_card`, odczyt
`ball/15/read/` (`read.md`, `read.json`, `table.csv`, `by_size.csv`, `referees.csv`, tory), logi i czasy GPU
`ball/15/logs/`, łańcuch `ball/15/run_gpu.sh`, opis przebiegu `runs/bd15_1600/RUN.md`.

## Wnioski

1. **1600 nie daje nowego kandydata.** Względem B 1280 jest +1,01 pp, a względem fazy 1 +0,40 pp. Oba przedziały
   obejmują 0.
2. Zysk jest prawdziwy, ale lokalny. Na grze 2 1600 bije zarówno B, jak i fazę 1 (dolna granica > 0). Na grze 3 traci
   to, co negatywy B odzyskały na sędziach w czerni. Wynik łączny zależy więc od tego, czy uda się utrzymać efekt
   negatywów przy 1600, np. przez negatywy w skali 1600. To trop na później, nie ten ticket.
3. Zysk punktowo skupia się w małych i średnich piłkach (≤ 14 px: 41 z 43 klatek netto), ale za mało klatek < 8 px,
   żeby to rozstrzygnąć. Zgodnie z rejestracją nie ma wniosku o P2.
4. Koszt 26–29 ms/klatkę mieści się w limicie 40 ms.

## Odstępstwa i uwagi

- Batch 3 zamiast 4 (OOM przy 1600). lr0 bez zmian, bo efektywny batch po akumulacji to 63 wobec 64.
- Zrzut 25 fps obejmuje 58 klipów GSR valid (jak w 14), ale śledzone i czytane były tylko klipy gier 2+3. Karty bez
  `--confirm`. Gier 4, 5 ani GSR test nie czytano.
- Przed odczytem `read` był uruchomiony na sucho z zrzutem B w miejscu 1600 (sprawdzenie kodu, odtworzył wiersze 14).
  Poprawka kontroli: porównanie torów B bez kolumny czasu ściennego (commit dad754d).
- Ruch kamery jest z 04 (dd2c), tak jak w 13 i 14. Ludzie dla `track_ball` pochodzą ze zrzutu 1600.

## GPU

Probe pamięci i smoke 18:50–18:56. Jednostka `bd15-gpu` 18:58–22:05: trening 2 h 36 min, zrzut 25 fps 19 min, karta
8 min, karta vs B 3,5 min. Razem **ok. 3,2 GPU-h**. Blokada ustawiona przed startem i zwolniona przez łańcuch o 22:05.
