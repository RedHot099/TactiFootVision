# ball-detection 25: wiersz modelu bazowego na GSR test i całym GSR valid (opisowy)

2026-10-29. Jeden odczyt, nic nie jest wybierane. Stary detektor (YOLO11m, Roboflow `football_yolo`, `best.pt`
sha256 7183d342) liczony dokładnie główną miarą, jak dd2c w ticketach 02 i 09. Ten odczyt czyta grę 5 dla modelu
bazowego oraz ponownie GSR test, już po zamknięciu badań: jest opisowy i niczego nie wybiera.

Miara: top-1 piłka o conf ≥ 0,15, wejście 1280, **bez** filtra 45 px, środek ramki przez zamrożone homografie
`kp03_r2_refined`; dobrze umieszczona = ≤ 5 m od prawdy, klatki bez homografii to pudła; mianownik = co 5. klatka
z dokładnie jedną piłką na boisku. Przedziały: bootstrap po klipach, 2 000 losowań, seed 0 (te same losowania
dla wszystkich kolumn danego splitu). „> 5 m" to udział wśród klatek z umieszczoną piłką (jak w ticketach 02 i 09).

## Wiersze

| split | klatki (bez H) | bazowy: dobrze umieszczona [95%] | dd2c top-1 | pipeline (dd2c + 45 px + `track_ball`) | bazowy → dd2c top-1, pp [95%] | bazowy → pipeline, pp [95%] | > 5 m: bazowy / dd2c / pipeline |
|---|---|---|---|---|---|---|---|
| GSR test (49 klipów) | 5 277 (688) | **48,44%** [42,4; 54,2] | 68,96% | 77,07% | **+20,52** [+15,78; +25,30] | **+28,63** [+23,67; +33,53] | 20,08% / 7,71% / 9,18% |
| cały GSR valid (58 klipów) | 6 404 (426) | **37,49%** [29,4; 45,2] | 60,18% | 76,30% | **+22,69** [+19,04; +26,77] | **+38,80** [+32,18; +45,66] | 20,71% / 13,97% / 12,01% |

Różnica „> 5 m" (pp, wariant − bazowy, [95%]): test dd2c −12,37 [−16,37; −8,77], pipeline −10,90 [−14,78; −7,35];
valid dd2c −6,73 [−10,86; −3,13], pipeline −8,70 [−12,53; −5,51].

## Kontrole (wszystkie zgodne co do 4 cyfr)

- Wiersz wyboru (gry 2+3, 4 250 klatek) z tego samego zrzutu modelu bazowego: **24,75%** [16,3; 34,0], > 5 m 26,18%
  = wiersz z ticketu 02.
- dd2c top-1 z `ball/09/read_*`: test 68,96%, valid 60,18%. Pipeline: test 77,07%, valid 76,30%. Wszystkie jak w ticketach
  02 i 09 (`controls` w `read.json`).
- Klatki modelu bazowego i plików ticketu 09 identyczne (`clip`, `frame`), sprawdzane w kodzie; mianowniki 5 277 i 6 404.

## Jak liczono

- Valid: istniejący zrzut `ball/02/base/dump` (obejmuje cały GSR valid, także grę 5), bez inferencji.
- Test: zrzut wiersza 1280 na klatkach GSR test (co 5.; `detector_bench.dump`, conf 0,001, to samo ustawienie co
  zrzut valid), potem top-1 ≥ 0,15. GPU: 198 s (3,3 min), blokada `agent-state gpu` ustawiona i zwolniona.
  Test dd2c w tickecie 09 szedł z 25 fps zrzutu tej samej klatki; top-1 jest ten sam (kontrola 68,96%).
- Kod: `scripts/ball_baseline_row.py`, test `tests/test_ball_baseline_row.py`. Wyniki:
  `/mnt/storage/tactifoot_research/detector_data/ball/25/` (`read.json`, `placement_*_base.parquet`, zrzut testu).

## Wniosek

Stary detektor umieszcza piłkę trafnie w 48,4% klatek testu i 37,5% całego valid; dd2c dodaje 20–23 pp, końcowy
pipeline 29–39 pp. Co piąta umieszczona piłka modelu bazowego leży dalej niż 5 m (dd2c: 8–14%). Odczyt opisowy.
