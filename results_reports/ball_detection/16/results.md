# ball-detection 16: koniec fazy 2, odczyt xG (opisowo)

2026-10-26. Żaden z ticketów 11–15 nie wygrał (komentarz orkiestratora 22:20), więc zgodnie z rejestracją 16 nie
ma kandydata fazy 2 ani złożenia, a **gry 4, gry 5 i GSR test nie czytano**. Ten ticket robi dwie rzeczy: odczyt xG
opisany niżej i raport fazy [`PHASE2.md`](../PHASE2.md). Rejestracja odczytu xG: komentarz ticketu 16 z 22:16,
zapisany przed jakimkolwiek przebiegiem na klatkach strzałów. Kod: `scripts/ball_xg.py` (`detect`, `read`), testy
`tests/test_ball_xg.py`. Wyniki leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/16/xg/` (`read.md`,
`read.json`, `detect.json`, `candidates.parquet`, galeria).

## Co dziś robi pipeline xG

Sprawdzone w worktree `research-xg-detector` (HEAD 0d1a194, `xgpoc/detector` i `src` bez lokalnych zmian). Cache
`cache/xgd_full4/E3` używa dd2c 1280 na klatkach 720p. Ludzie mają próg 0,3, piłka `class_conf` 0,15, NMS IoU 0,5.
Piłką klatki jest najpewniejszy kandydat, rzutowany ze środka ramki. **Filtra 45 px nie ma.** Miejsce strzału
wyznacza zamrożona reguła `feet_t0`: piłka pochodzi z pierwszej klatki z piłką na boisku w kolejności 0, −400,
−800 ms; strzelcem jest gracz, którego ramka w obrazie leży najbliżej środka piłki; miejscem strzału są jego stopy.

## Populacja i ramiona

**Populacja:** 7 062 strzały bez karnych z 328 meczów rozwojowych, czyli populacja treningowa stosu xG. Żaden strzał
nie pochodzi z 172 odłożonych meczów. Skrypt czyta tylko kolumny indeksu bez etykiet. Nie czyta goli, wyniku ani
StatsBomb.

| ramię | piłka na klatce (0, −400 i −800 ms) |
|---|---|
| X0, xG dziś | top-1 dd2c ≥ 0,15, bez filtra |
| P1, kandydat fazy 1 (pojedyncza klatka) | X0 po filtrze 45 px (dłuższy bok przeliczony na 1080p; przy 720p próg to 30 px) |
| S2, drugi stopień 13 | kandydaci dd2c ≥ 0,01 z tego samego wywołania detektora; wycinek, ResNet18 (`ball/13/model/best.pt`, sha256 09e8d644…), fuzja z `ball/13/fusion/fusion.json`, ramka przesunięta regresorem (wiersz (ii) z 13); potem filtr 45 px i top-1 fuzji ≥ 0,15 |

Ludzie, drużyny, punkty boiska i homografie pochodzą z cache. Ramiona różnią się tylko piłką, rzutowaną środkiem
ramki przez homografię tej samej klatki, tak jak w pakiecie. Cechy liczy zamrożony kod xG
(`location.feature_rows(..., FEET_T0)`). Klatek −800 ms, których potrzebowało któreś ramię, a których nie było w
cache, było 36. Przeszły przez analizator xG (`extract3.ball_analyzer`, dd2c 1280).

**Kontrola:** X0 odtwarza cache `cache/xgd_full4_feet_t0/E3/features.parquet` (sha256 93b4b580…) bez żadnej
różnicy, na 14 124 wierszach (t0 i −400 ms, wszystkie kolumny cech, tolerancja 1e-6).

## Wynik (wiersz t0 reguły `feet_t0`)

Przedziały 95%: bootstrap par po meczach (328), 2 000 losowań, seed 0.

| miara | X0 xG dziś | P1 faza 1 | S2 drugi stopień 13 | P1 − X0, pp [95%] | S2 − P1, pp [95%] |
|---|---|---|---|---|---|
| `ball_ok` (jest piłka z 0 / −400 / −800 ms) | 92,42% | 91,96% | **97,07%** | −0,47 [−0,71; −0,26] | **+5,11 [+4,51; +5,73]** |
| piłka własna t0 | 61,33% | 60,29% | **80,83%** | −1,03 [−1,35; −0,75] | **+20,53 [+19,23; +21,78]** |
| zapas −400 ms | 26,59% | 27,13% | 14,13% | +0,54 [+0,29; +0,77] | −13,00 [−14,09; −11,96] |
| zapas −800 ms | 4,50% | 4,53% | 2,11% | +0,03 [−0,07; +0,13] | −2,42 [−2,89; −1,98] |
| brak piłki | 7,58% | 8,04% | 2,93% | +0,47 [+0,26; +0,71] | −5,11 [−5,73; −4,51] |
| proxy: stopy strzelca ≤ 2 m od rzutu piłki | 51,50% | 51,78% | 51,46% | +0,09 [−0,09; +0,28] | +1,48 [+0,71; +2,24] |

Proxy liczę wśród strzałów z piłką. Różnice parami obejmują strzały, które mają piłkę w obu ramionach. Poziomy
dotyczą różnych zbiorów strzałów, bo S2 ma piłkę także w trudniejszych strzałach. Mediana odległości stóp od piłki
wynosi 1,91 / 1,89 / 1,91 m.

**Zmiana miejsca strzału** (wśród 6 990 strzałów z miejscem strzału w obu ramionach; bez prawdy, to nie jest błąd):

| | inne miejsce (inny strzelec) | > 1 m | > 5 m | mediana przesunięcia, gdy się zmienia |
|---|---|---|---|---|
| P1 vs X0 | 0,82% [0,53; 1,17] | 0,80% | 0,62% [0,35; 0,95] | 10,8 m |
| S2 vs P1 | **16,05% [15,00; 17,16]** | 15,36% [14,32; 16,42] | **9,23% [8,45; 10,02]** | 6,1 m |

**Piłka t0 P1 vs S2** (zgodna, gdy środek S2 leży w promieniu max(3 px przy 1080p, pół przekątnej ramki P1)):

| przypadek | strzały | miejsce strzału się zmienia (> 5 m) | P1 brał piłkę z | proxy P1 → S2 |
|---|---|---|---|---|
| ta sama piłka | 4 126 (58,4%) | 1,1% (0,2%) | t0 | 61,1% → 60,4% |
| tylko S2 ma piłkę t0 | **1 477 (20,9%)** | 54,9% (31,2%) | −400 ms: 1 085, −800 ms: 160, brak: 232 | 37,5% → 43,1% |
| obie, różne piłki | 168 (2,4%) | 55,4% (30,4%) | t0 | 47,6% → 48,2% |
| tylko P1 | 9 | | t0 | |
| żadne | 1 282 (18,2%) | 13,8% (10,0%) | −400 / −800 ms albo brak | 31,6% → 30,4% |

Filtr 45 px (P1 vs X0) usuwa piłkę t0 X0 w 98 strzałach (1,4%). Zmienia miejsce strzału w 0,8% strzałów, a gdy je
zmienia, to daleko (mediana 10,8 m).

## Wnioski (opisowo)

1. **S2 znajduje piłkę t0 dużo częściej.** Piłka własna t0 rośnie z 60,3% do 80,8%, zapas z −400/−800 ms spada z
   31,7% do 16,2%, a brak piłki z 8,0% do 2,9%. Zysk pochodzi z kandydatów dd2c poniżej 0,15, które fuzja
   podnosi powyżej progu: mediana fuzji tych piłek to 0,44. To ten sam mechanizm co w wierszu pojedynczej klatki z 13
   (+10,8 pp na grach 2+3).
2. **Taka zmiana wyraźnie przesuwa xG:** inne miejsce strzału ma 16% strzałów, w tym 9% o więcej niż 5 m. Prawie
   całość pochodzi ze strzałów, w których P1 brał piłkę z −400 ms albo nie miał jej wcale.
3. **Nie wiemy, czy nowe miejsce jest lepsze.** Brak prawdy na tych strzałach. Proxy (stopy strzelca blisko rzutu
   piłki) nie spada, a w strzałach, w których tylko S2 ma piłkę t0, rośnie z 37,5% do 43,1%. To słaby sygnał, nie
   dowód. Galeria 40 niezgodności (24 różne piłki, 8 z 9 przypadków „tylko P1”, 8 „tylko S2”) czeka na
   obejrzenie przez Jakuba: `ball/16/xg/gallery/` (`index.csv`; pomarańczowy P1, niebieski S2, liczba to pewność).
4. **Filtr 45 px w xG prawie nic nie zmienia** (1,4% piłek t0, 0,8% miejsc strzału). Strzały xG pochodzą z lig
   z kamerą szeroką, gdzie piłka rzadko przekracza 30 px przy 720p.

Następny krok należy do linii xG: odczyt C6 stosu na danych rozwojowych z piłką S2, czyli zmiana wejścia reguły
`feet_t0`. Zamrożony wariant bierze flagi źródła piłki t0 z E3 (dd2c), więc trzeba też zdecydować, czy flagi
pójdą za S2. Wniosek potwierdzający wymaga świeżego zbioru testowego, bo 172 mecze są zużyte.

## Odstępstwa i uwagi

1. **Błąd `feet_t0` względem StatsBomb nie jest mierzony.** Parowanie SoccerNet–StatsBomb obejmuje tylko 135 meczów
   ewaluacyjnych, a wszystkie leżą w 172 odłożonych. Żaden z 328 meczów rozwojowych nie ma meczu StatsBomb. Zakres
   zabrania czytania 172 meczów, więc zatrzymałem się na tej części (rejestracja) i podaję zmianę miejsca zamiast
   błędu.
2. **Smoke na 40 pierwszych strzałach populacji** (`ball/16/smoke`) przed pełnym przebiegiem, po rejestracji.
   Sprawdzał kod i kontrolę X0. Wyszedł w nim błąd w tabeli zgodności piłki t0: `~` na serii typu object oznaczał
   wszystkie wspólne piłki jako „różne”. Poprawka nie zmienia żadnej miary ani progu. Nic nie było dobierane.
3. **Rozbicie według zgodności piłki t0** (tabela „Piłka t0 P1 vs S2”, kolumny miejsca i proxy) dopisałem po
   pierwszym pełnym odczycie. Jest opisowe i nie ma go w rejestracji.
4. Godzina rejestracji: nagłówek komentarza miał najpierw 22:25. Zapis powstał o 22:16 (`date`), poprawione na 22:16.
5. Drugi stopień był uczony i kalibrowany na innych danych: GSR (1080p), SoccerNet-v3 i TeamTrack, fuzja na grze 9.
   Klatki xG mają 720p. Wycinki skalują się wysokością klatki (`image_scale`), ale kalibracja fuzji na klatkach xG
   nie jest sprawdzona.
6. Czytany jest tylko wiersz t0. S2 zmienia też piłkę na −400 ms, czyli wiersz −400 ms stosu.
7. Wyniki 16 oraz kod gier 4/5 i GSR test: bez odczytów. Zrzuty z 10 (`ball/10/dump25_holdout`,
   `card_dump_dd2c_holdout`) zostają nieczytane.

## Koszt i GPU

dd2c 1280 na klatce 720p: 12,9 ms/klatkę. Drugi stopień: 1,5 ms na klatkę z kandydatami. Analizator xG na 36
klatkach −800 ms: 109 ms/klatkę (punkty boiska i homografia). `agent-state gpu` było wolne przed startem. Blokada
trwała ok. 22:16–22:24 (smoke i jednostka `bd16-xg`, 21 186 klatek), razem **ok. 0,13 GPU-h**. Odczyt szedł na CPU.
