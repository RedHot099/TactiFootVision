# ball-detection: raport końcowy fazy

2026-10-25. Branch `research/ball-detection`, tickety 01–05, 08 i 09 (`.scratch/ball-detection/issues/`). Przez całą
fazę ten sam detektor: dd2c (`models/detector_yolo11l_1280_dd2c.pt`, YOLO11l 1280, sha256 05d706cf…). Duże wyniki
leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/`.

## Wynik w skrócie

- Najwięcej dało śledzenie piłki w czasie. dd2c 1280 + filtr 45 px + `track_ball` (ustawienie z 04) umieszcza piłkę
  dobrze na **76,30%** klatek całego GSR valid (dd2c top-1: 60,18%). W drugim odczycie GSR test: **77,07% wobec
  68,96%, +8,11 pp [+5,73; +10,63]**.
- Filtr 45 px nic nie kosztuje i pomaga wszędzie: sam dd2c +0,64 pp, ze śledzeniem +1,55 pp, odsetek > 5 m spada.
- Sam detektor nie osiąga bramki planu: najlepszy potwierdzony wariant bez trackera (kafle 2×2 + filtr) ma 66,21%
  na całym GSR valid, a AP50 piłki ≥ 0,45 nie osiągnął żaden wariant.

## Cel i reguła

Cel: więcej klatek z trafną pozycją piłki na boisku, bez straty na ludziach. Miara główna (reguła z ticketu 02,
komentarz 2026-10-24 21:02): **zysk parami nad dd2c 1280 w odsetku klatek z piłką dobrze umieszczoną** (≤ 5 m od
prawdy na zamrożonych homografiach `kp03_r2_refined`). Mianownik: klatki co 5. z dokładnie jedną piłką na boisku
(GSR valid: 6 404 klatki, z tego 426 bez homografii liczonych jako pudło; sufit 93,3%). Przedziały 95%: bootstrap po
klipach, 2 000 losowań, seed 0. Wybór na grach 2+3 GSR valid, potwierdzenie raz na zwycięzcę na grze 5, jedno
porównanie główne na ticket, zapisane przed odczytem. Strażnicy: mAP50-95 ludzi ≥ −0,01, Ekstraklasa dev mAP50-95
≥ −0,02; bramka na całym GSR valid 68,18% przy 1280 (ambitnie 73,18%), AP50 piłki ≥ 0,45. Punkt wyjścia dd2c 1280:
60,18% na całym GSR valid, 51,60% na grach 2+3, 77,11% na grze 5.

## Wyniki ticketów

| ticket | co | porównanie główne, pp [95%] | werdykt | raport |
|---|---|---|---|---|
| 01 | analiza błędów dd2c | brak; pudła: 43% bez kandydata, 33% kandydat < 0,15, 24% zła ramka; 49% pudeł to piłki < 12 px; rozmycie 19% pudeł; 55% umieszczeń > 5 m to sędziowie w czerni z klasą „ball” | | [01](01/results.md) |
| 02 | karta wyników i reguła | dd2c 1280: 60,18% (cały valid); bramka 68,18% / 73,18% | reguła zapisana | [02](02/results.md) |
| 03 | inferencja bez treningu | 1920 natywnie vs 1280: −0,21 [−2,87; +2,10] | niespełniona | [03](03/results.md) |
| 04 | śledzenie offline (`track_ball`) | gry 2+3 +17,32 [+11,86; +22,82]; gra 5 +9,15 [+4,24; +14,58] | sukces, potwierdzony | [04](04/results.md) |
| 05 | dotrenowanie, nadpróbkowanie koszyków ×2 | B − A po ostatniej epoce +0,26 [−0,80; +1,35] | brak efektu | [05](05/results.md) |
| 08 | kafle 2×2 + filtr 45 px | gra 5 +5,29 [+3,15; +8,02] | potwierdzony; bramka (66,21%) niespełniona | [08](08/results.md) |
| 09 | drugi odczyt GSR test, kandydat wideo | GSR test +8,11 [+5,73; +10,63] | opisowy | [09](09/results.md) |

Opisowo z 03 (gry 2+3): sam filtr 45 px na dd2c +0,64 pp [+0,12; +1,18] (> 5 m 18,5% → 9,6%), 1600 + TTA + filtr
+6,45 pp, kafle 2×2 + filtr +6,40 pp (poza rejestracją, potwierdzone potem w 08).

## Kandydat wideo na GSR valid (opisowo)

dd2c 1280, kandydaci ≥ 0,01 przy 25 fps, filtr 45 px, `track_ball` z ustawieniem 04, homografie przeniesione ruchem
kamery; zrzut 25 fps z ticketu 04. Para: top-1 dd2c 1280 ≥ 0,15 na tych samych klatkach.

| split | top-1 dd2c | kandydat [95%] | zysk parami, pp [95%] | wśród z homografią | > 5 m wśród umieszczonych |
|---|---|---|---|---|---|
| gry 2+3 | 51,60% | 71,25% [64,6; 77,5] | +19,65 [+13,81; +25,60] | 56,19% → 77,58% | 18,45% → 14,17% |
| cały GSR valid | 60,18% | **76,30%** [71,0; 80,9] | +16,11 [+11,35; +20,52] | 64,47% → 81,73% | 13,97% → 12,01% |

Sam filtr na śledzeniu (kandydat vs śledzenie z 04 bez filtra): gry 2+3 +2,33 pp [+0,72; +4,16], cały valid
+1,55 pp [+0,49; +2,81]; > 5 m −4,47 / −2,89 pp. Gra 5 nie ma tu osobnego wiersza.

## Drugi odczyt GSR test

Zarejestrowany w tickecie 09 przed odczytem; jeden przebieg, nic nie dobierane na teście. dd2c był już wybrany na
GSR test w detector-data, więc **to nie jest świeży zbiór, a wynik jest opisowy**. Klatki 25 fps pochodzą z SoccerNet
Tracking test (te same klipy; wszystkie 7 350 lokalnych klatek GSR test bajt w bajt identyczne), etykiety i
homografie z GSR test. 49 klipów, 5 277 klatek mianownika, 688 bez homografii (sufit 87,0%).

| | top-1 dd2c 1280 | kandydat | różnica parami, pp [95%] |
|---|---|---|---|
| **dobrze umieszczona** | 68,96% [63,7; 73,7] | **77,07%** [71,5; 81,7] | **+8,11 [+5,73; +10,63]** |
| wśród klatek z homografią | 79,30% | 88,62% | +9,33 [+6,58; +12,21] |
| > 5 m wśród umieszczonych | 7,71% | 9,18% | +1,47 [+0,46; +2,73] |

Zmierzone 78,5% klatek (kandydat dobrze 89,1%, top-1 85,9%, +2,46 pp), interpolowane 10,3% (69,6% dobrze, 25,5%
> 5 m, top-1 14,4%, +5,67 pp), bez piłki 11,2% (−0,02 pp). Zysk na teście jest mniejszy niż na valid, bo test jest
dla dd2c łatwiejszy (jak gra 5 w 04). Na teście trochę rośnie odsetek pozycji > 5 m, a na valid ten odsetek spadał.

## Rekomendacja dla pipeline'u

1. **Wideo: dd2c 1280 + filtr 45 px + `track_ball`** z ustawieniem 04: `min_confidence` 0,05 (detektor musi
   zostawiać kandydatów piłki od 0,05), `restart_confidence` 0,15, `restart_after` 0,25 s, `gate` 3,0 m,
   `gate_speed` 65 m/s, `max_gap` 2,0 s, test nieruchomości wyłączony (albo 20 s przy zapasowych piłkach przy
   bandzie), reszta domyślna. Koszt: detekcja 19,6 ms/klatkę GPU, śledzenie 0,2 ms CPU. Ruch kamery (ok. 10 ms
   CPU) był potrzebny tylko dlatego, że homografie mieliśmy co 5. klatkę; pipeline z homografią na każdej klatce go
   nie potrzebuje.
2. **Filtr 45 px zawsze**, przed `track_ball` (opcja detektora, np. `ball_max_side_px=45` przy 1080p, skalowana
   wysokością klatki). Nic nie kosztuje, a ludzie i Ekstraklasa się nie zmieniają.
3. **Pojedyncze klatki bez trackera** (np. t0 strzału w xG): kafle 2×2 + filtr, potwierdzone w 08 (+5,29 pp na
   grze 5), za 3,8× czasu detekcji. W wideo kafli nie dokładać: ze śledzeniem gra 5 dała −1,25 pp [−2,79; +0,24].
4. Pozycje interpolowane oznaczać flagą `interpolated` (pakiet już ją ma), bo co czwarta leży dalej niż 5 m.

## Co nie działało

- **Wyższa rozdzielczość jednego wywołania** (03): 1920 natywnie −0,21 pp, 1600 +0,96 pp (przedział obejmuje 0).
  Detektor znajduje więcej piłek, ale też więcej pewnych fałszywych, więc top-1 częściej trafia w fałszywą.
- **Nadpróbkowanie koszyków porażek ×2** (05): B − A +0,26 pp [−0,80; +1,35] po ostatniej epoce.
- **Epoka bez mozaiki na restartowanym harmonogramie** (05): −8,5 pp wobec dd2c w obu ramionach, fałszywe piłki
  rosną z 0,2 do 0,5 na klatkę.
- Kafle bez filtra (03: −0,73 pp) i kafle dokładane do śledzenia (08: gra 5 −1,25 pp).
- Bramki 68,18% ani AP50 piłki ≥ 0,45 nie osiągnął żaden wariant samego detektora; bramkę daje dopiero tracker.
  Piłek < 8 px nie wykrywa żaden wariant (recall IoU ≤ 0,06).

## Otwarte tropy

- **1600 + TTA + filtr** (03: +6,45 pp na grach 2+3, 55,5 ms/klatkę): niezarejestrowany, bez odczytu
  potwierdzającego. Grę 5 czytały już 04 i 08, więc kolejny odczyt na niej słabiej chroni przed doborem po danych.
- **Poprawiony przepis dotrenowania:** `close_mosaic` 0 albo `warmup_epochs` dopasowane do długości przebiegu, z
  B − A po epoce 0 jako porównaniem głównym (opisowo B e0 miało +2,82 pp nad A). Do tego **twarde negatywy**:
  sędziowie i liniowi w czerni z klasą „piłka” (55% umieszczeń > 5 m w 01).
- **06 SoccerSynth jako mały udział** (rozmycie to 19,3% pudeł, próg 15% przekroczony z małym zapasem) i **07
  ablacja źródeł**: backlog, nieuruchomione.
- Drugi stopień dla piłki (klasyfikator na wycinkach) oraz głowica P2 / trening przy 1600: nie próbowane.
- Skutek kandydata końcowego w xG (`ball_ok`, `feet_t0` na strzałach dev): w tej fazie nie mierzony.

## Zużycie GPU (RTX 4070 Ti SUPER)

| ticket | 01 | 02 | 03 | 04 | 05 | 08 | 09 | razem |
|---|---|---|---|---|---|---|---|---|
| GPU-h | 0,17 | ok. 0,55 | ok. 0,5 | 0,23 | ok. 8,3 | ok. 1,2 | 0,22 | **ok. 11,2** |

## Licencje

Wagi dd2c i wszystkie warianty tej fazy są **tylko do badań** (decyzja 11): trenowano je na danych SoccerNet
objętych NDA, na źródłach CC BY-NC i na źródłach bez licencji. GSR test i SoccerNet Tracking test posłużyły tylko do
oceny. StatsBomb, ręczne anotacje i 172 mecze xG nie były używane.
