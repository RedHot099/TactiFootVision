# ball-detection 02: karta wyników piłki i reguła wyboru

2026-10-24. Bez treningu. GPU: trzy zrzuty predykcji (dd2c, run 1 best, model bazowy) i bezpośrednie
umieszczenie dd2c przy 1280, razem ok. 33 min (18:15–18:48), blokada `agent-state gpu` ustawiona i zwolniona.
Żaden nowy wariant nie był czytany; reguła wyboru jest w komentarzu ticketu 02 i w `RULE` w
`scripts/ball_scorecard.py`.

## Karta

```bash
uv run python scripts/ball_scorecard.py card --weights W --out OUT [--rows 1280 e3] [--confirm]  # zrzut (GPU) + karta
uv run python scripts/ball_scorecard.py card --dump DUMP --out OUT [--confirm] [--primary-row 1920] # karta ze zrzutu
uv run python scripts/ball_scorecard.py agree OUT_A OUT_B                                           # zgodność dwóch kart
uv run python scripts/ball_scorecard.py references                                                  # wiersze odniesienia
```

Wejście to zrzut w formacie ticketu 01 (`detector_bench.py dump`); `--weights` najpierw robi zrzut. Zrzut niesie w
`dump.json` wiersze (imgsz, NMS IoU 0,7 dla 1280 i 0,5 dla e3, `input_size`), `max_det`, klasy i od tego ticketu
mapowanie klas na klasy benchmarku, nazwę GPU i ms/klatkę (samo wywołanie modelu, batch 1). Karta odrzuca zrzut z
conf > 0,001 albo z innymi ustawieniami znanych wierszy.

Na każdy split (domyślnie tylko wybór = gry 2 i 3; `--confirm` dokłada grę 5 i cały GSR valid) i wiersz: AP50 /
AP50-95 piłki, P / R w punkcie pracy, mAP50-95 ludzi (średnia bramkarz / zawodnik / sędzia), recall obu reguł
z ticketu 01 (IoU 0,5 i odległość środka) przy 0,15 i 0,001, fałszywe piłki ≥ 0,15, recall w koszykach ticketu 01,
umieszczenie (mianownik, sufit, umieszczona, dobrze umieszczona wśród wszystkich i wśród klatek z homografią,
mediana, > 5 m; całość i blisko bramki), Ekstraklasa dev, koszt, różnice parami z dd2c i odczyt reguły.
Przedziały: bootstrap po klipach, 2 000 losowań, seed 0, te same losowania dla obu modeli. AP w losowaniu liczy
supervision na wylosowanych obrazach; dla oryginalnego zbioru to dokładnie liczba `evaluate`.

Kod: `scripts/ball_scorecard.py`; w `scripts/detector_bench.py` `score_dump` stoi teraz na `image_scores` /
`summarise_scores` (statystyki per obraz, więc split i losowanie liczą się tą samą drogą), `dump` zapisuje koszt i
mapowanie klas, `placement --row 1280` umieszcza piłkę z wiersza pipeline'u. Testy: `tests/test_ball_scorecard.py`.

## Zgodność

- **Wagi a zrzut (dd2c):** karta ze świeżego zrzutu wag i karta ze zrzutu ticketu 01: 2 888 liczb, największa różnica
  0,0 (`ball/02/agree_dd2c_weights_vs_dump01.json`).
- **Umieszczenie przy 1280 dwiema drogami:** z wag (`placement --row 1280`, model z conf 0,15 na każdej klatce) i
  z karty (zrzut, top-1 ≥ 0,15): ta sama flaga na 6 404 klatkach, różnica błędu 0,0 m, 60,18% w obu.
- **Pliki wyników** (cały GSR valid i Ekstraklasa dev, max |różnica| po klasach i miarach):

| model | GSR 1280 | GSR e3 | Ekstraklasa 1280 | Ekstraklasa e3 | umieszczenie e3 |
|---|---|---|---|---|---|
| dd2c (`final/candidate_valid/detection_valid.json`, `runs/compare/placement_run2c_epoch1`) | 0 | 0 | 0 | 0 | 6e-6 |
| run 1 best (`runs/run1/eval/…`) | 0 | 0 | 8e-4 | 3e-4 | 0 |
| model bazowy, yolo11m `best.pt` 7183d342 (`baseline/…`) | 0 | 0 | 3e-4 | 8e-5 | 2e-6 |

Model bazowy to „stary detektor” z raportu detector-data §4 (Roboflow yolo11m, sha256 w `baseline/*.json`). Liczby
całego zbioru zgadzają się z §4 (np. piłka 1280: dd2c 0,356, run 1 0,380, bazowy 0,141; umieszczenie e3 72,5 / 69,0 /
50,2%). Dla run 1 i modelu bazowego gra 5 weszła tylko do tej kontroli (zapisane same różnice); ich wiersze karty
są tylko na splicie wyboru.

## Wiersze odniesienia

Split wyboru (gry 2+3: 39 klipów, 4 250 klatek mianownika, 347 bez homografii, sufit 91,8%):

| model | wiersz | AP50 piłki | P / R | ludzie mAP50-95 | dobrze umieszczona [95%] | wśród z homografią | > 5 m | vs dd2c 1280, pp [95%] | ms/klatkę |
|---|---|---|---|---|---|---|---|---|---|
| dd2c | 1280 | 0,296 | 0,490 / 0,361 | 0,514 | **51,60%** [44,5; 59,3] | 56,19% | 18,45% | 0 | 19,6 |
| dd2c | e3 | 0,309 | 0,523 / 0,377 | 0,560 | 52,14% [44,5; 60,4] | 56,78% | 19,83% | (vs dd2c e3) 0 | 40,2 |
| run 1 best | 1280 | 0,329 | 0,585 / 0,337 | 0,537 | 48,59% [41,3; 55,7] | 52,91% | 15,44% | −3,01 [−5,54; −0,57] | 20,3 |
| run 1 best | e3 | 0,257 | 0,442 / 0,302 | 0,547 | 42,56% [33,9; 51,2] | 46,35% | 31,43% | (vs dd2c e3) −9,58 [−12,49; −7,18] | 39,7 |
| bazowy | 1280 | 0,093 | 0,369 / 0,146 | 0,120 | 24,75% [16,3; 34,0] | 26,95% | 26,18% | −26,85 [−31,80; −21,96] | 19,7 |
| bazowy | e3 | 0,080 | 0,310 / 0,137 | 0,108 | 24,42% [16,1; 33,3] | 26,59% | 32,90% | (vs dd2c e3) −27,72 | 40,0 |

dd2c, odczyt odniesienia (wyjątek dla gry 5 w spec):

| split | wiersz | AP50 piłki | dobrze umieszczona [95%] | wśród z homografią | umieszczona | > 5 m | mianownik (bez H) |
|---|---|---|---|---|---|---|---|
| gra 5 | 1280 | 0,490 | 77,11% [71,6; 82,0] | 80,05% | 83,15% | 7,26% | 2 154 (79) |
| cały GSR valid | 1280 | 0,356 | **60,18%** [53,6; 66,1] | 64,47% | 69,96% | 13,97% | 6 404 (426), sufit 93,3% |
| cały GSR valid | e3 | 0,373 | 61,95% [54,9; 68,5] | 66,36% | 72,55% | 14,61% | 6 404 (426) |

Ekstraklasa dev (dd2c 1280 / e3): AP50 piłki 0,279 / 0,335, mAP50-95 0,424 / 0,422. Pełna tabela z medianą, strefą
bramki i Ekstraklasą dla wszystkich modeli: `ball/02/reference_rows.md` (+ `.csv`).

Wniosek z wierszy: przy 1280 dd2c umieszcza piłkę trafnie w 60,18% klatek, 1,8 pp mniej niż przy e3; na splicie
wyboru różnica 1280 vs e3 to tylko 0,5 pp. Run 1 ma wyższe AP50 piłki niż dd2c na grach 2+3 (0,329 vs 0,296), ale
dobrze umieszczona jest niżej (−3,0 pp): AP i miara główna nie idą razem. Koszt e3 to ok. 2× koszt 1280.

## Reguła wyboru (skrót; pełny tekst w komentarzu ticketu 02)

- Główna: zysk parami nad dd2c 1280 w dobrze umieszczonych, gry 2+3, dolna granica 95% > 0. Jedno porównanie
  główne na ticket: 03 = imgsz 1920 natywnie vs dd2c 1280; 04 = wybrane śledzenie vs top-1 przy 1280;
  05 = B − A po ostatniej epoce (`last.pt`) obu ramion; 06 analogicznie do 05. Reszta opisowa, w 03: 1600, kafle
  2×2 / 3×2 + pełna klatka, TTA z odbiciem i post-filtr kandydatów piłki o dłuższym boku > 45 px (próg z prawdy:
  maks. 43,0 px na splicie wyboru, 17 z 40 931 piłek GSR train powyżej 45 px).
- Zwycięzca = sukces główny + strażnicy względni: mAP50-95 ludzi spada najwyżej o 0,01, Ekstraklasa mAP50-95
  najwyżej o 0,02.
- Odczyt potwierdzający raz na zwycięzcę: gra 5, potwierdzony przy dolnej granicy > 0.
- Bramki przy odczycie potwierdzającym na całym GSR valid: **68,18% przy 1280** (ambitnie 73,18%), 70% / 75% przy
  e3, AP50 piłki ≥ 0,45; Ekstraklasa AP50 piłki ≥ 0,40 tylko kierunek.

## Ścieżki (poza gitem)

`/mnt/storage/tactifoot_research/detector_data/ball/02/`: `dd2c/` (zrzut z wag, karta 3 splitów,
`placement_direct_1280/`), `dd2c_dump01/` (karta ze zrzutu ticketu 01), `run1_best/`, `base/` (zrzuty i karty
splitu wyboru), `truth/` (tabele cech piłek prawdy na split), `reference_rows.md|csv`, `reference_check.json`,
`agree_dd2c_weights_vs_dump01.json`, `logs/`.
