# ball-detection 17: dokładność lokalizacji piłki (progi, percentyle, rozbicia, zgodność referencji)

2026-10-27. Odczyt opisowy, tylko CPU, bez testów i bez doboru. Rejestracja w komentarzu ticketu 17 z 13:54, przed
pierwszym odczytem. Kod: `scripts/ball_accuracy.py`, testy `tests/test_ball_accuracy.py`. Tabele:
`/mnt/storage/tactifoot_research/detector_data/ball/17/`.

- **Predykcje (zamrożone, punkt = środek ramki):** top-1 dd2c 1280 ≥ 0,15 bez filtra; kandydat fazy 1 (tory 09);
  drugi stopień 13 na pojedynczej klatce (fuzja + regresor, filtr 45 px, top-1 ≥ 0,15 z `ball/13/dump25_regress`);
  dodatkowo drugi stopień 13 śledzony (wiersz (ii) z 13). Drugi stopień tylko na GSR valid, bo na teście nie ma jego
  zrzutu.
- **Klatki:** mianownik reguły 02 (co 5. klatka z jedną piłką prawdy na boisku). Klatka bez homografii albo bez piłki
  to „bez pozycji”.
- **Zbiory:** gry 2+3 (główne), cały GSR valid (gry 2, 3, 5 razem, jak w 09), GSR test opisowo na torach 09 (D4).
  Gry 4 nie czytano, gry 5 osobno też nie.
- **Przedziały:** 95%, bootstrap po klipach (2 000 losowań, seed 0, losowania karty). Rozbicia podaję jako wartości
  punktowe.

Progi 0,5 m i 1 m mierzą zgodność z referencją GSR (`bbox_pitch`), a nie dokładność fizyczną. Pokazuje to pierwsza
tabela.

## 1. Zgodność referencji (pierwszy wynik)

Ramkę prawdy piłki z GSR rzutuję naszą zamrożoną homografią klatki w dwóch punktach (środek i dół-środek ramki) i
porównuję z `bbox_pitch` z GSR, klatka po klatce. Wynik łączy trzy rzeczy: błąd homografii, konwencję anotacji i
wysokość piłki. Nie jest to „podłoga” i nie odejmuję go od błędów predykcji.

Liczone wśród klatek z homografią:

| zbiór | punkt | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m | p95 m |
|---|---|---|---|---|---|---|---|---|
| gry 2+3 | środek | 25,1 [19,9; 31,7] | 53,4 [47,6; 59,9] | 80,8 [76,8; 84,5] | 93,6 [90,2; 96,2] | 0,94 | 3,31 | 6,84 |
| gry 2+3 | dół | 35,7 [29,8; 42,5] | 61,5 [56,2; 67,0] | 80,5 [76,5; 84,1] | 93,2 [89,8; 95,9] | 0,71 | 3,44 | 7,14 |
| cały valid | środek | 23,4 | 51,3 | 78,8 | 93,6 [91,3; 95,5] | 0,97 | 3,56 | 6,05 |
| cały valid | dół | 33,6 | 58,6 | 78,4 | 93,0 | 0,78 | 3,78 | 6,52 |
| GSR test | środek | 25,8 | 56,8 | 83,0 | 94,0 [92,2; 95,5] | 0,86 | 3,14 | 5,66 |
| GSR test | dół | 43,0 [38,2; 47,5] | 68,6 | 84,5 | 94,2 | 0,61 | 3,01 | 5,79 |

Bez homografii jest 8,2% klatek na grach 2+3, 6,7% na całym valid i 13,0% na teście.

Co z tego wynika:
- Przy prawdzie rzutowanej naszą homografią mediana różnicy to 0,6–0,9 m.
- Około 6% klatek z homografią jest dalej niż 5 m od `bbox_pitch`.
- Progi 0,5 m i 1 m leżą więc w szumie referencji. Próg 5 m mieści około 94% klatek z homografią.

Rozbicie na gry 2+3, punkt dolny:
- piłka w powietrzu: p50 2,20 m, ≤ 1 m tylko 17,8%;
- piłka na ziemi: p50 0,62 m;
- pole karne: p50 0,44 m;
- reszta boiska: p50 0,85 m.

Dla piłki w powietrzu `bbox_pitch` nie jest więc zwykłym rzutem dołu ramki naszą homografią.

## 2. Odsetki i percentyle predykcji

### Gry 2+3 (39 klipów, 4 250 klatek, 347 bez homografii)

| wiersz | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m [95%] | p90 m [95%] | p95 m | bez pozycji |
|---|---|---|---|---|---|---|---|---|
| top-1 dd2c | 12,1 | 27,0 [22,7; 31,9] | 43,6 | 51,6 [44,5; 59,3] | 1,19 [1,05; 1,36] | 15,22 [9,07; 21,34] | 23,30 | 36,7 |
| kandydat fazy 1 | 16,4 | 37,3 [32,5; 42,4] | 58,7 | **71,2** [64,6; 77,5] | 1,13 [0,99; 1,28] | 9,03 [4,84; 12,03] | 14,87 | 17,0 |
| 13, pojedyncza klatka | 15,3 | 33,4 [28,9; 38,5] | 52,7 | 62,4 [56,7; 68,3] | 1,07 [0,94; 1,19] | 7,44 [4,26; 10,30] | 15,06 | 29,0 |
| 13, śledzony (ii) | 16,5 | 37,9 [33,8; 42,4] | 59,9 | 72,7 [66,9; 78,2] | 1,14 [1,01; 1,27] | 9,07 [5,39; 10,98] | 15,46 | 15,0 |

Odsetki w tej tabeli liczone są wśród wszystkich klatek. Percentyle liczone są wśród klatek z pozycją.

Wśród klatek z pozycją:

| wiersz | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | > 5 m |
|---|---|---|---|---|---|
| top-1 dd2c | 19,1 | 42,7 | 68,9 | 81,6 | 18,4 |
| kandydat fazy 1 | 19,8 | 44,9 | 70,7 | 85,8 | 14,2 |
| 13, pojedyncza klatka | 21,5 | 47,0 | 74,2 | 87,9 [83,8; 91,3] | 12,1 |
| 13, śledzony (ii) | 19,4 | 44,6 | 70,4 | 85,4 | 14,6 |

### Cały GSR valid (58 klipów, 6 404 klatek) i GSR test (49 klipów, 5 277 klatek; drugi odczyt, opisowo)

| zbiór | wiersz | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m [95%] | bez pozycji | ≤ 5 m wśród z pozycją |
|---|---|---|---|---|---|---|---|---|
| cały valid | top-1 dd2c | 31,2 | 49,6 | 60,2 [53,6; 66,1] | 1,14 | 9,06 [5,39; 14,82] | 30,0 | 86,0 |
| cały valid | kandydat fazy 1 | 39,3 | 61,9 | **76,3** [71,0; 80,9] | 1,12 | 6,16 [4,55; 10,02] | 13,3 | 88,0 |
| cały valid | 13, pojedyncza klatka | 36,1 | 56,9 | 69,1 [63,8; 73,8] | 1,09 | 5,42 [4,25; 8,91] | 22,5 | 89,2 |
| cały valid | 13, śledzony (ii) | 39,4 | 62,6 | 77,0 [72,4; 81,1] | 1,13 | 6,59 [4,80; 9,67] | 11,9 | 87,4 |
| GSR test | top-1 dd2c | 40,1 | 60,2 | 69,0 [63,7; 73,7] | 0,92 | 3,89 [3,10; 4,78] | 25,3 | 92,3 |
| GSR test | kandydat fazy 1 | 44,5 [39,4; 49,5] | 66,6 | **77,1** [71,5; 81,7] | 0,94 | 4,56 [3,63; 5,87] | 15,1 | 90,8 |

Dla porównania zgodność referencji (środek ramki, wśród wszystkich klatek) na ≤ 5 m wynosi:
- gry 2+3: 85,9%;
- cały valid: 87,4%;
- test: 81,8%.

## 3. Rozbicia

Opis, bez testów. Pełne tabele: `ball/17/breakdown.csv` i `ball/17/read.md`. Poniżej kandydat fazy 1 na grach 2+3;
wiersze referencji są w `reference_breakdown.csv`.

| cecha | poziom | klatki | bez pozycji | ≤ 1 m | ≤ 5 m | p50 m | > 5 m wśród z pozycją |
|---|---|---|---|---|---|---|---|
| status | zmierzona | 2 625 | 2,9 | 45,3 | 87,7 | 1,07 | 9,8 |
| status | interpolowana | 1 027 | 4,8 | 38,5 | 70,8 | 1,53 | 25,7 |
| status | brak | 598 | 100 | 0 | 0 | — | — |
| pokrycie przez ludzi | brak | 2 575 | 12,9 | 38,5 | 75,0 | 1,14 | 14,0 |
| pokrycie przez ludzi | < 0,5 | 385 | 16,4 | 37,4 | 71,9 | 1,14 | 14,0 |
| pokrycie przez ludzi | ≥ 0,5 | 1 290 | 25,3 | 34,9 | 63,6 | 1,10 | 14,7 |
| w powietrzu | ziemia | 3 433 | 17,2 | 41,0 | 71,5 | 1,01 | 13,6 |
| w powietrzu | w powietrzu | 476 | 9,0 | 14,9 | 72,7 | 2,72 | 20,1 |
| wielkość | < 8 px | 238 | 44,5 | 21,8 | 35,3 | 2,41 | 36,4 |
| wielkość | 8–12 px | 1 464 | 25,1 | 37,3 | 64,4 | 1,00 | 14,0 |
| wielkość | ≥ 24 px | 131 | 0,0 | 35,9 | 96,2 | 1,28 | 3,8 |
| strefa | pole karne | 965 | 15,1 | 35,4 | 71,6 | 1,16 | 15,6 |
| strefa | reszta | 3 285 | 17,5 | 37,8 | 71,1 | 1,12 | 13,7 |
| moment | strzał ±1 s | 42 | 0,0 | 47,6 | 92,9 | 1,09 | 7,1 |
| moment | gol ±1 s | 23 | 0,0 | 39,1 | 87,0 | 1,58 | 13,0 |

Obserwacje:
- **Interpolowane pozycje** to główne źródło błędów > 5 m w każdym zbiorze, z tym samym obrazem: 25,7% na grach 2+3,
  25,8% na całym valid i 26,8% na teście, wobec 9,8% / 8,1% / 6,9% dla pozycji zmierzonych. Domyka to 18.
- **Piłka w powietrzu** prawie nie trafia w progi ≤ 1 m: 14,9% na grach 2+3, z medianą 2,7 m. Referencja ma tu ten
  sam problem (p50 2,2 m). Przy progu 5 m piłka w powietrzu wypada podobnie jak piłka na ziemi (72,7% wobec
  71,5%).
- **Małe piłki (< 8 px)** wypadają najsłabiej: 35,3% ≤ 5 m. Najczęściej brakuje dla nich pozycji (44,5%), a gdy
  pozycja jest, 36% z nich leży dalej niż 5 m.
- **Pokrycie przez ludzi ≥ 0,5** głównie odbiera pozycję: 25,3% bez pozycji wobec 12,9% bez pokrycia. Dokładność
  pozycji, które są, prawie się nie zmienia.
- **Drugi stopień 13 na pojedynczej klatce** ma najmniejszy p90 (7,4 m na grach 2+3) i najwyższy odsetek ≤ 5 m wśród
  klatek z pozycją (87,9%). Częściej jednak nie daje pozycji wcale (29,0% wobec 17,0%).
- **Strefa:** na teście w polu karnym kandydat daje 92,1% ≤ 5 m, a na reszcie boiska 73,0%. Na grach 2+3 różnicy
  między strefami nie ma.

## 4. Momenty strzałów

**GSR.** Każdy klip GSR to 30 s wokół jednej akcji SoccerNet. Pole `info` podaje `action_class`, `action_position` i
`clip_start` w ms połowy, więc klatkę akcji wyznacza wzór (action_position − clip_start) / 40 + 1. We wszystkich 107
klipach akcja mieści się w oknie klipu.

Klipy ze strzałami:
- GSR valid: 9 klipów strzałów („Shots on/off target”) i 9 klipów goli;
- GSR test: 11 klipów strzałów, 1 karny i 1 gol.

W oknie ±1 s wokół akcji mianownik ma:
- gry 2+3: 42 klatki strzałów i 23 klatki goli;
- cały valid: 66 i 29;
- test: 70 i 3. Na teście z okolic gola zostają tylko 3 klatki, bo piłka w bramce leży poza boiskiem.

Kandydat fazy 1 przy strzale (punktowo, mało klatek):

| zbiór | ≤ 1 m | ≤ 5 m | p50 m |
|---|---|---|---|
| gry 2+3 | 47,6% | 92,9% | 1,09 |
| test | 42,9% | 94,3% | 1,11 |

Ograniczenie: znana jest tylko akcja środkowa klipu. GSR nie podaje meczu, więc Labels-v2 meczu (`soccernet_labels`
w danych xG) nie dają się przypisać i inne strzały w oknie 30 s pozostają nieoznaczone.

**Wiersz „moment strzału” na 172 (D4, opisowo).** Populacja: 2 721 strzałów bez karnych ze 135 meczów
ewaluacyjnych, sparowanych ze StatsBomb (`xgc_compare`, populacja `primary`). Porównuję piłkę t0 X0 (top-1 dd2c ≥ 0,15
z cache xG `xgd_full4/E3`, własna klatka t0, środek ramki) z miejscem strzału StatsBomb. Piłkę wyrównuję do bliższej
bramki według konwencji xG. StatsBomb służy tu tylko jako cel i jest drugą referencją z własnym szumem kodowania i
czasu. Bootstrap po meczach (2 000, seed 0).

| wiersz | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m [95%] | p90 m [95%] | bez pozycji |
|---|---|---|---|---|---|---|---|
| X0 piłka t0, wśród wszystkich strzałów | 4,5 | 16,0 | 34,6 | 49,8 [47,6; 52,1] | 1,78 [1,69; 1,91] | 10,48 [9,57; 11,23] | 37,2 [34,5; 39,7] |
| X0 piłka t0, wśród z pozycją | 7,2 | 25,4 | 55,1 | 79,2 [77,1; 81,3] | | | |
| `feet_t0` (miejsce strzału xG dziś, kontekst) | 22,7 | 45,0 | 66,9 | 84,8 [83,4; 86,2] | 1,16 [1,05; 1,25] | 6,79 [6,32; 7,27] | 0,5 |

Mediana `feet_t0` = 1,16 m, czyli liczba z feet-shot-location (kontrola zgodności z tamtą analizą). Piłka t0 ma
pozycję w 62,8% strzałów. Gdy ją ma, jest dalej od miejsca StatsBomb niż stopy strzelca (p50 1,78 m wobec 1,16 m).
**P1 i S2 nie wykonano:** cache trzyma tylko top-1, a zrzut 16 obejmuje tylko 328 meczów, więc obie wymagają nowej
detekcji na klatkach 172 meczów (GPU). Dłuższy bok piłki X0 t0 przekracza 45 px tylko w 24 strzałach (1,4% z
pozycją), więc P1 zmieniłby wynik najwyżej tam (albo usunął piłkę, albo podstawił następną). Różnica X0 → P1 na tym
wierszu jest zatem z góry mała. S2 wymaga GPU.

## Kontrole

- **Karty 09 co do cyfry:** odsetek ≤ 5 m wśród wszystkich i > 5 m wśród umieszczonych, wartość i przedział, dla
  top-1 i kandydata fazy 1. Zgodność na grach 2+3 (71,25% / 51,60%; 14,17%), na całym valid (76,30% / 60,18%;
  12,01%) i na teście (77,07% / 68,96%; 9,18%), 12 z 12 porównań równe.
- **Śledzony drugi stopień:** ponowne śledzenie `dump25_regress` z `ball/13/tune/chosen.json` daje na grach 2+3
  tory identyczne z `ball/13/read/track_ii` (29 250 klatek).
- **Definicje z 01:** poziomy wielkości, pokrycia i „w powietrzu” policzone tu są zgodne z tabelą prawdy ticketu 01
  (`ball/02/truth/balls_gsr_games_23.parquet`) na wszystkich 4 250 klatkach gier 2+3 (100% dla każdej cechy).
- **Testy:** `tests/test_ball_accuracy.py` (7 testów na syntetycznych danych: progi, percentyle, zgodność losowań z
  `ball_scorecard.placement_draws`, zgodność referencji środek/dół, strefa, moment, poziomy 01). Szybki zestaw:
  1 069 testów przechodzi, ruff czysty.

## Odstępstwa i uwagi

1. Drugi stopień 13 jest tylko na GSR valid, bo na teście nie ma jego zrzutu, a nowy zrzut wymagałby GPU. Wiersz
   śledzony (ii) dodałem opisowo; ticket wymaga tylko pojedynczej klatki. Na całym valid oznacza to pierwszy odczyt
   13 na klipach gry 5 jako części całego valid, tak jak w 09. Gry 5 osobno nie czytałem.
2. Pozycje piłki strzałów na 172 wyrównuję do bramki bliższej samej piłce (konwencja xG `attacks_right`), a nie do
   kierunku z `feet_t0`. Jeśli piłka jest blisko połowy, kierunek może się różnić.
3. Do wiersza 172 użyłem miejsca `feet_t0` jako kontekstu. W tickecie go nie było. To ta sama miara co w
   feet-shot-location.
4. Rozbicie „strzał” w GSR ma 42–70 klatek. To orientacja, a nie wynik.

## Odtworzenie

```bash
uv run python scripts/ball_accuracy.py read    # ok. 75 s CPU (12 procesów śledzenia drugiego stopnia)
uv run python scripts/ball_accuracy.py shots   # ok. 2 s
```

Poza gitem, w `ball/17/`:
- `read.json` i `read.md` (wszystkie liczby z przedziałami);
- `summary.csv`, `breakdown.csv`, `reference_breakdown.csv`;
- `placement_<grupa>_<wiersz>.parquet`;
- `truth_valid.parquet` i `truth_test.parquet` (cechy prawdy i błędy referencji na klatkę);
- `shots172.{json,md,parquet}`;
- `logs/`.

GPU: brak.
