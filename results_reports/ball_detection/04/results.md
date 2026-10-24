# ball-detection 04: śledzenie piłki w czasie (pomiar offline)

2026-10-24. Wagi `models/detector_yolo11l_1280_dd2c.pt` (sha256 05d706cf…), bez treningu i bez zmian w pakiecie
(decyzja 6). Porównanie główne zapisane w tickecie 04 (komentarz 21:50) przed odczytem; parametry dobrane tylko na
grach 2+3, i to tylko na klatkach spoza benchmarku. Gra 5 przeczytana raz, dla zwycięzcy. Przedziały 95%: bootstrap
po klipach (2 000 losowań, seed 0), karta `scripts/ball_scorecard.py`.

Kod: `scripts/ball_temporal.py` (`dump`, `motion`, `tune`, `track`, `breakdown`), testy `tests/test_ball_temporal.py`.

## Co mierzę

- **Zrzut:** dd2c w wierszu 1280 (natywne klatki, imgsz 1280, NMS IoU 0,7, `max_det` 300), wszystkie klasy
  ≥ 0,01, wszystkie 43 500 klatek 58 klipów GSR valid (25 fps), 19,15 ms/klatkę, 14 min GPU. Na klatkach co 5.
  zrzut jest identyczny ze zrzutem ticketu 01 (229 967 ramek, różnica 0,0).
- **Tracker:** pakietowy `track_ball` bez zmian, na `PipelineResult` zbudowanym z zrzutu (kandydaci piłki, ludzie
  ≥ 0,3). To port prototypu: odrzut nieruchomych kandydatów, punktów karnych i ramek piłki na człowieku, bramka stałej
  prędkości, restart tylko na pewnym kandydacie, usuwanie odstających, wypełnianie luk z flagą `interpolated`.
- **Bramka w metrach boiska.** Zamrożone homografie są tylko co 5. klatkę, więc pozostałe klatki dostają homografię
  przez kompensację ruchu kamery: homografia obraz→obraz między kolejnymi klatkami (narożniki poza ramkami ludzi,
  Lucas-Kanade, RANSAC) przenosi najbliższe kotwice sprzed i zza klatki (do 10 klatek), z liniowym mieszaniem.
  Na klatkach benchmarku homografia jest dokładnie zamrożona. Na 43 442 parach klatek nie ma ani jednej przerwy
  ruchu. Kotwica przeniesiona o 5 klatek różni się od następnej zamrożonej o 0,34 m (mediana klipów; bez
  kompensacji 0,80 m).
- **Kodowanie dla karty:** wybrana piłka jest jedynym kandydatem piłki ≥ 0,15 w klatce (zmierzona zachowuje ramkę,
  interpolowana dostaje 0,15), pozostałe kandydaty piłki są obcięte do 0,1499, inne klasy i Ekstraklasa jak w dd2c.
  Top-1 dd2c zakodowane tak samo daje identyczne umieszczenie co dd2c (różnica 0,0), więc służy za odniesienie dla
  trafień przy r px.

## Dobór parametrów (gry 2+3, klatki spoza benchmarku)

Miara doboru: odsetek klatek, w których środek wybranej piłki leży w r = max(5 px, średnica) od prawdy (16 966
klatek z jedną piłką na boisku). Top-1 dd2c: **56,61%**. Domyślne `track_ball`: **47,21%**, czyli gorzej niż top-1,
bo test nieruchomości z oknem 3,2 s odrzuca piłkę leżącą przed stałym fragmentem gry (SNGS-030, rzut wolny: 199 z
287 trafnych top-1 oznaczonych jako nieruchome). Dwa etapy siatki (bramka, maks. luka, próg restartu, okno testu
nieruchomości, `min_confidence`; 218 ustawień), potem jeden parametr naraz wokół najlepszego:

| parametr | wartości → trafienie (wybrane pogrubione) |
|---|---|
| bramka (m + m/s × dt) | 0,75+16,25: 59,60% · 1,5+32,5: 63,38% · 2,25+48,75: 64,89% · **3+65: 64,95%** · 4,5+97,5: 64,80% |
| maks. luka | 0,2 s: 60,54% · 0,5 s: 62,78% · 1 s: 64,33% · 1,5 s: 64,90% · **2 s: 64,95%** |
| próg restartu | **0,15: 64,95%** · 0,25: 64,15% · 0,35: 62,88% · 0,45: 62,01% · 0,6: 55,48% |
| okno testu nieruchomości | 3,2 s: 56,17% · 10 s: 60,95% · 20 s: 64,41% · **wył.: 64,95%** |
| `min_confidence` | 0,01: 62,53% · **0,05: 64,95%** · 0,12: 63,04% |

Wybrane: `min_confidence` 0,05, `restart_confidence` 0,15, `gate` 3 m, `gate_speed` 65 m/s, `max_gap` 2 s, test
nieruchomości wyłączony (`static_window` 1000 s), reszta domyślna. Bramka od 2,25+48,75 i luka od 1,5 s to plateau
(różnice ≤ 0,15 pp). Śledzenie kosztuje 0,17 ms/klatkę CPU (plus ok. 10 ms/klatkę na ruch kamery w połowie
rozdzielczości, jeden rdzeń).

## Wynik główny (gry 2+3, 4 250 klatek mianownika, 347 bez homografii)

| wariant | dobrze umieszczona [95%] | wśród z H | umieszczona | > 5 m wśród umieszczonych | mediana m | recall przy r px (środek) | recall IoU 0,5 |
|---|---|---|---|---|---|---|---|
| top-1 dd2c 1280 | 51,60% [44,5; 59,3] | 56,19% | 63,27% | 18,45% | 1,19 | 50,03% | 35,18% |
| **śledzenie** | **68,92%** [61,6; 75,8] | 75,04% | 84,71% | 18,64% | 1,22 | 56,34% | 36,10% |
| tylko zmierzone (bez interpolacji, opisowo) | 53,72% [46,6; 61,4] | 58,49% | 62,38% | 13,88% | 1,13 | 48,55% | 33,54% |

Parami względem top-1 (pp):

| wariant | dobrze umieszczona | > 5 m | recall przy r px | recall IoU |
|---|---|---|---|---|
| **śledzenie** | **+17,32 [+11,86; +22,82]** | +0,19 [−3,48; +4,30] | +6,31 [+1,96; +10,66] | +0,93 [−2,06; +3,81] |
| tylko zmierzone | +2,12 [−0,80; +4,40] | −4,56 [−7,90; −1,39] | −1,48 (punktowo) | −1,64 (punktowo) |

**Porównanie główne: sukces** (dolna granica +11,86 pp > 0). Strażnicy ludzi i Ekstraklasy: z konstrukcji bez
zmian (różnica 0,000). AP50 piłki opisowo: 0,308 wobec 0,296 dd2c (tylko kodowanie, nie detektor).
Recall „przy r px” dla dd2c liczony na wszystkich kandydatach ≥ 0,15 wynosi 51,61%; dla top-1 50,03%.

Klatki zmierzone i interpolowane osobno (status śledzenia na klatce benchmarku):

| split | status | klatki | śledzenie dobrze | śledzenie > 5 m | top-1 dobrze | zysk pp (z całości) |
|---|---|---|---|---|---|---|
| gry 2+3 | zmierzone | 2 737 (64,4%) | 83,4% | 13,4% | 76,8% | +4,24 |
| | interpolowane | 998 (23,5%) | 64,7% | 30,4% | 7,3% | +13,48 |
| | bez piłki | 515 (12,1%) | 0% | 0% | 3,3% | −0,40 |
| gra 5 | zmierzone | 1 824 (84,7%) | 91,7% | 5,5% | 87,4% | +3,62 |
| | interpolowane | 252 (11,7%) | 73,4% | 26,2% | 15,5% | +6,78 |
| | bez piłki | 78 (3,6%) | 0% | 0% | 34,6% | −1,25 |

## Odczyt potwierdzający (gra 5, raz) i bramki

- Gra 5 (2 154 klatki): 77,11% → **86,26%**, **+9,15 pp [+4,24; +14,58]: potwierdzony**. > 5 m 8,25% (top-1 7,26%).
- Cały GSR valid (6 404 klatki): 60,18% → **74,75%** [68,9; 79,9], +14,57 pp [+10,17; +18,92]; bramka 1280
  ≥ 68,18% i ambitna ≥ 73,18% osiągnięte. To wynik śledzenia, nie lepszego detektora: AP50 piłki 0,362 (< 0,45),
  a bramki planu dotyczą detektora w pipeline z `track_ball`, więc liczba mówi, ile daje tracker na dzisiejszym dd2c.

## Skąd zysk (gry 2+3, kategorie ticketu 01)

822 klatki odzyskane (top-1 źle, śledzenie dobrze), 86 straconych:

| źródło piłki śledzenia | odzyskane | z tego: kandydat dd2c w r < 0,15 / ≥ 0,15 / brak | stracone |
|---|---|---|---|
| interpolacja (wypełniona luka) | 613 | 256 / 11 / 346 | 40 |
| zmierzona, pewność < 0,15 | 183 | 173 / 0 / 10 | 15 |
| zmierzona ≥ 0,15, inna niż top-1 | 26 | 1 / 23 / 2 | 14 |
| brak piłki (top-1 było dobre) | | | 17 |

Po kategoriach ticketu 01 (zysk w pp całego mianownika): kandydat < 0,15 +10,07 (794 klatki, dobrze 7% → 61%), brak
kandydata +8,35 (1 008 klatek, 5% → 40%, prawie wyłącznie interpolacją), kandydat ≥ 0,15 −1,11 (85,1% → 83,2%).
Rozmiar: < 8 px +0,99 (7% → 25%), 8–12 px +8,52 (36% → 61%), 12–16 px +5,91, 16–24 px +1,79, ≥ 24 px +0,12.
Pokrycie ramkami ludzi ≥ 0,5: +7,20 (39% → 62%). Rozmyte +3,41 (44% → 69%). Dalsza tercja +5,62 (33% → 52%).
Pełna tabela: `ball/04/track/breakdown.md`.

## Wnioski

1. Śledzenie daje duży, potwierdzony zysk: +17,3 pp na grach 2+3, +9,2 pp na grze 5, 74,75% na całym GSR valid
   (dd2c 60,18%). Odsetek > 5 m wśród umieszczonych się nie zmienia, ale umieszczonych jest dużo więcej.
2. Zysk to głównie wypełnianie luk: interpolacja daje 13,5 z 17,3 pp, a wybór słabego kandydata (< 0,15) tylko
   4,0 pp. Bez interpolacji zysk jest niepewny (+2,1 pp [−0,8; +4,4]), za to spada odsetek > 5 m (−4,6 pp).
3. Interpolowane pozycje są zgrubne: 30% z nich jest dalej niż 5 m, a recall IoU śledzenia stoi w miejscu.
   Tabele powinny je oznaczać (flaga `interpolated` już jest).
4. Odpowiedź na pytanie ticketu: tracker naprawia dużą część pudeł z kandydatem < 0,15 (430 z 794 klatek) i część
   klatek bez kandydata (358 z 1 008), ale tylko przez sąsiednie klatki. Trening na małe piłki (< 12 px) dalej ma
   sens: tam odzysk jest najmniejszy (< 8 px: 25% dobrze), a piłka interpolowana nie daje pozycji w strzale, gdy
   piłki nie ma w otoczeniu t0.
5. Domyślne ustawienia `track_ball` są na GSR gorsze od top-1 (47,2% wobec 56,6% trafień na klatkach doboru). Główny
   powód to test nieruchomości z oknem 3,2 s, który odrzuca piłkę przed stałym fragmentem gry.

## Ograniczenia

- Test nieruchomości wyłączony wygrywa na GSR, gdzie klipy są wokół akcji (rzuty wolne, rożne, karne); na nagraniu
  z zapasowymi piłkami przy bandzie przez cały mecz okno 20 s może być bezpieczniejsze (−0,5 pp na doborze).
- Klatki doboru sąsiadują z klatkami benchmarku (te same klipy, inne klatki), więc dobór nie jest niezależny od
  odczytu głównego. Gra 5 jest od niego wolna i potwierdza zysk.
- Homografie śledzenia pochodzą z modelu punktów (kp03_r2_refined) i ruchu kamery, nie z prawdy; w pipeline
  homografia liczona co klatkę może być bardziej zaszumiona niż przeniesiona kotwica.
- Pozycje piłki w powietrzu są rzutowane na płaszczyznę boiska, jak w pakiecie.

## Domyślne parametry dla `ball-tracking/02`

`min_confidence` 0,05 (pipeline musi wtedy zostawiać kandydatów piłki od 0,05), `restart_confidence` 0,15,
`restart_after` 0,25 s, `gate` 3,0 m, `gate_speed` 65 m/s, `max_gap` 2,0 s, test nieruchomości wyłączony
(`static_window` ≥ długość klipu) albo 20 s przy nagraniach z zapasowymi piłkami, `spot_radius` 0,6,
`max_person_iou` 0,5, `isolation` 0,15. Komentarz z liczbami jest też w `.scratch/ball-tracking/issues/02-track-ball.md`.

## Ścieżki (poza gitem)

`/mnt/storage/tactifoot_research/detector_data/ball/04/`: `dump25/` (zrzut 25 fps, `dump.json`), `motion/` (ruch
kamery per klip, `summary.csv`), `tune/` (`sensitivity_stage{1,2,3}.csv|md`, `chosen.json`), `track/` (gry 2+3:
`tracks.parquet`, `top1.parquet`, zrzuty `dump_track|dump_measured|dump_top1`, `breakdown.md|csv`,
`breakdown_cross.csv`), `track_confirm/` (wszystkie gry, zrzut do odczytu potwierdzającego), `card_track/` (karta
porównania głównego), `card_track_vs_top1/`, `card_measured/`, `card_top1/`, `card_confirm/` (karta z `--confirm`),
`logs/`.

Odtworzenie: `uv run python scripts/ball_temporal.py dump` (GPU), `motion`, `tune --stage 1`, `tune --stage 2`,
`tune --stage 3`, `track --setting ball/04/tune/chosen.json`, potem `scripts/ball_scorecard.py card --dump
ball/04/track/dump_track --out ball/04/card_track` i `ball_temporal.py breakdown`.
