# ball-detection 18: wiarygodność interpolacji (krótsze luki, wynik niepewności, odrzucanie, ryzyko–pokrycie)

2026-10-27. Tylko CPU, gry 2+3 GSR valid (39 klipów, 4 250 klatek mianownika). Rejestracja w komentarzu ticketu 18
z 14:12, przed pierwszym odczytem. Kod: `scripts/ball_reliability.py`, testy `tests/test_ball_reliability.py`. Tabele:
`/mnt/storage/tactifoot_research/detector_data/ball/18/` (`read.json`, `curves.csv`, `levels.csv`,
`equal_coverage.csv`, `score_summary.csv`, `scored_out_of_game.parquet`, tory wariantów).

**Wynik w skrócie.** Bramka D2 niespełniona, więc punktu pracy nie ma: ustawienia 04 zostają bez zmian, a gry 4 nie
czytano. Wynik niepewności (regresja logistyczna) jest lepszy od samej długości luki: przy odrzuceniu 10% pozycji
ma 10,6% > 5 m wśród przekazanych wobec 12,1%, różnica −1,5 pp [−2,2; −0,9]. Na torach `max_gap` 2 s z odrzucaniem
krzywa leży też poniżej krzywej samego `max_gap`. Bramka nie przechodzi, bo próg wybrany na jednej grze nie
przenosi się na drugą. Gra 3 ma dużo więcej długich luk niż gra 2, a większość interpolacji w długich lukach jest
tam dobra.

## Ustawienia

- **Tory:** zrzut `ball/04/dump25`, filtr 45 px, ruch kamery `ball/04/motion` i `track_ball` z
  `ball/04/tune/chosen.json`. Zmienia się tylko `max_gap` (0,25 / 0,5 / 1 / 2 s). `max_gap` 2 s to kandydat fazy 1.
- **Klatki:** mianownik reguły 02 (zamrożone homografie, środek ramki), jak w 17. Pozycja to klatka z piłką i
  homografią. Pokrycie to odsetek klatek mianownika z przekazaną pozycją.
- **Cechy wyniku niepewności:** status, długość luki, czas do pomiaru, prędkość implikowana, pewność (dla
  interpolowanej: słabszy koniec luki), ruch kamery i wielkość ramki. Wszystkie są dostępne przy inferencji.
  Linia bazowa to sama długość luki.
- **Ocena:** procedura naprzemienna. Model uczony na grze 2 jest oceniany na grze 3 i odwrotnie, a próg wybierany jest
  na grze uczącej. Przedziały 95% pochodzą z bootstrapu po klipach (2 000 losowań, seed 0).

## 1. Kontrole

- `max_gap` 2 s bez odrzucania daje tory identyczne z `ball/09/track_valid` (gry 2+3, cała tabela).
- Ten sam przebieg bez filtra daje tory identyczne z `ball/04/track`.
- Odczyt odtwarza 17: kandydat fazy 1 ma 71,2% ≤ 5 m i 14,2% > 5 m wśród klatek z pozycją.

## 2. Sam `max_gap` (bez odrzucania), gry 2+3

| `max_gap` | pokrycie | > 5 m wśród przekazanych | > 2 m wśród przekazanych | > 5 m na wszystkie klatki | dobrze umieszczone | strata vs faza 1 |
|---|---|---|---|---|---|---|
| 0,25 s | 67,0 | 9,5 | 24,6 | 6,4 | 60,6 | 10,7 pp |
| 0,5 s | 72,1 | 11,0 | 26,1 | 8,0 | 64,2 | 7,1 pp |
| 1 s | 78,7 | 13,3 | 28,3 | 10,5 | 68,3 | 3,0 pp |
| 2 s (faza 1) | 83,0 | 14,2 | 29,3 | 11,8 | 71,2 | 0 |

Krótsza luka usuwa złe pozycje, ale usuwa ich mniej niż dobrych. Przejście z 2 s na 0,5 s usuwa 3,8 pp złych
pozycji na wszystkie klatki i 7,1 pp dobrze umieszczonych.

Gry różnią się mocno:

| gra | `max_gap` 0,25 s: pokrycie / > 5 m | 2 s: pokrycie / > 5 m | dobrze umieszczone 0,25 → 2 s |
|---|---|---|---|
| 2 (1 852 klatki) | 86,6 / 10,7 | 90,7 / 12,5 | 77,3 → 79,4 |
| 3 (2 398 klatek) | 51,8 / 8,0 | 77,1 / 15,7 | 47,7 → 65,0 |

W grze 2 interpolacja prawie nic nie zmienia. W grze 3 długie luki dają 17 pp dobrze umieszczonych.

## 3. Wynik niepewności vs linia bazowa (poza grą, łącznie 2+3)

Porównanie zarejestrowane na torach `max_gap` 2 s: odsetek > 5 m wśród przekazanych przy odrzuceniu 10% pozycji.

| wynik | > 5 m wśród przekazanych |
|---|---|
| model | 10,6 [6,0; 16,1] |
| długość luki | 12,1 [7,6; 17,7] |
| różnica | **−1,5 pp [−2,2; −0,9]** |

Model bije linię bazową, bo górna granica jest mniejsza od 0. To samo widać na pozostałych wariantach (opisowo):

| `max_gap` | AURC model / linia bazowa | log-loss model / linia bazowa | Brier model / linia bazowa | odsetek > 5 m |
|---|---|---|---|---|
| 0,25 s | 0,068 / 0,116 | 0,330 / 0,320 | 0,085 / 0,087 | 9,5 |
| 0,5 s | 0,066 / 0,102 | 0,340 / 0,346 | 0,089 / 0,098 | 11,0 |
| 1 s | 0,071 / 0,105 | 0,370 / 0,390 | 0,101 / 0,118 | 13,3 |
| 2 s | 0,079 / 0,109 | 0,424 / 0,466 | 0,117 / 0,141 | 14,2 |

**Kalibracja** (`max_gap` 2 s, poza grą): model przenoszony między grami jest za pewny w górnych przedziałach.

| P(> 5 m) | pozycje | średnie P | odsetek > 5 m |
|---|---|---|---|
| 0,0–0,1 | 2 030 | 0,03 | 0,07 |
| 0,1–0,2 | 648 | 0,14 | 0,10 |
| 0,2–0,3 | 220 | 0,24 | 0,15 |
| 0,3–0,4 | 137 | 0,35 | 0,31 |
| 0,4–0,5 | 113 | 0,45 | 0,33 |
| 0,5–0,6 | 97 | 0,55 | 0,53 |
| 0,6–0,7 | 68 | 0,65 | 0,40 |
| 0,7–0,8 | 80 | 0,75 | 0,43 |
| 0,8–0,9 | 60 | 0,85 | 0,45 |
| 0,9–1,0 | 75 | 0,95 | 0,56 |

Ranking pozycji działa, ale wartości P > 0,6 trzeba czytać jako „około połowa złych”.

Współczynniki modelu (cechy standaryzowane, tory 2 s, opisowo; gra 2 / gra 3):
- prędkość implikowana: +0,62 / +1,23;
- ruch kamery: −0,63 / −0,75;
- pewność: −0,10 / −0,65;
- długość luki: +0,21 / +0,32;
- czas do pomiaru: +0,16 / +0,29;
- wielkość: −0,20 / −0,31.

Flagi braku prędkości i przerwy ruchu kamery są stałe na pozycjach mianownika, więc mają współczynnik 0.

## 4. Krzywe ryzyko–pokrycie

Pełne krzywe dla > 5 m i > 2 m są w `curves.csv`. Poniżej stan przy stałym odsetku odrzuconych pozycji (poza grą,
łącznie 2+3, opisowo; `levels.csv`). Kolumny „strata” i „> 5 m wśród odrzuconych” mówią, ile dobrych pozycji
kosztuje odrzucanie.

Tory `max_gap` 2 s (kandydat fazy 1):

| odrzucone pozycje | wynik | pokrycie | > 5 m wśród przekazanych | > 2 m wśród przekazanych | > 5 m na wszystkie | > 5 m wśród odrzuconych | strata dobrze umieszczonych |
|---|---|---|---|---|---|---|---|
| 0 | — | 83,0 | 14,2 | 29,3 | 11,8 | — | 0 |
| 2,5% | model | 80,9 | 13,2 | 28,3 | 10,7 | 52,4 | 1,0 pp |
| 5% | model | 78,9 | 12,3 | 27,2 | 9,7 | 50,1 | 2,1 pp |
| 10% | model | 74,7 | 10,6 | 25,6 | 7,9 | 46,7 | 4,4 pp |
| 20% | model | 66,4 | 8,1 | 22,9 | 5,4 | 38,5 | 10,2 pp |
| 5% | długość luki | 78,9 | 13,4 | 28,4 | 10,6 | 29,3 | 2,9 pp |
| 10% | długość luki | 74,7 | 12,1 | 27,2 | 9,0 | 32,8 | 5,6 pp |

**Przy równym pokryciu** (`equal_coverage.csv`): krzywe interpolowane przy pokryciach punktów samego `max_gap`, > 5 m
wśród przekazanych.

| tory | przy pokryciu `max_gap` 1 s (78,7) | przy pokryciu 0,5 s (72,1) | przy pokryciu 0,25 s (67,0) |
|---|---|---|---|
| sam `max_gap` | 13,3 | 11,0 | 9,5 |
| 2 s + model | 12,3 | 9,4 | 8,1 |
| 2 s + długość luki | 13,4 | 11,4 | 10,6 |
| 1 s + model | — | 9,8 | 8,3 |

Odrzucanie modelem na torach 2 s leży poniżej krzywej samego `max_gap` w całym zakresie. Odrzucanie długością luki
leży na tej krzywej albo nad nią.

W ramach straty ≤ 2 pp dobrze umieszczonych (D2) mieści się jednak tylko odrzucenie około 5% pozycji. Daje to
spadek z 14,2% do około 12,3% > 5 m wśród przekazanych (−13% względnie), a połowa odrzuconych pozycji jest wtedy
zła.

## 5. Reguła i bramka D2 (poza grą)

Na grze uczącej reguła wybrała (warunki: ≥ 50% złych wśród odrzuconych i strata ≤ 2 pp; potem najmniejszy odsetek
> 5 m wśród przekazanych):

| wynik | uczony na grze 2 (oceniany na 3) | uczony na grze 3 (oceniany na 2) |
|---|---|---|
| model | `max_gap` 1 s, τ 0,37 | `max_gap` 2 s, τ 0,48 |
| długość luki | `max_gap` 0,5 s, τ 0,44 | `max_gap` 2 s, τ 0,44 |
| sam `max_gap` | 0,5 s | 2 s |

Łącznie poza grą (2+3, model; w nawiasach bootstrap po klipach):

| miara | punkt modelu | kandydat fazy 1 |
|---|---|---|
| pokrycie | 70,4 [63,5; 77,3] | 83,0 |
| > 5 m wśród przekazanych | 9,3 [5,2; 14,2] | 14,2 |
| > 5 m na wszystkie klatki | 6,5 [3,6; 10,2] | 11,8 |
| dobrze umieszczone | 63,9 [56,7; 71,1] | 71,2 |
| różnica dobrze umieszczonych | **−7,3 pp [−10,1; −5,0]** | — |
| > 5 m wśród odrzuconych (357 pozycji) | **47,9 [35,9; 59,0]** | — |

Bramka:
- (a) poniżej krzywej `max_gap`: spełniona (9,3% wobec 10,5% na krzywej przy tym pokryciu);
- (b) ≥ 50% odrzuconych złych: **niespełniona** (47,9%);
- (c) strata ≤ 2 pp: **niespełniona** (7,3 pp).

Bramka jest więc **niespełniona**. Linia bazowa i sam `max_gap` też jej nie przechodzą (strata 7,1 i 6,7 pp).

Każdy kierunek osobno (opisowo):
- **Gra 2 z regułą z gry 3:** spełnia wszystkie trzy warunki. Strata 0,8 pp, 68% odrzuconych złych, > 5 m 10,9% wobec
  12,5%.
- **Gra 3 z regułą z gry 2:** nie spełnia. Reguła z gry 2 wybiera `max_gap` 1 s, co na grze 3 samo kosztuje 5,3 pp.
  Próg 0,37 odrzuca tam 310 pozycji, z czego 45% jest złych. Strata wynosi 12,4 pp.

Przyczyną jest różnica gier: w grze 2 długie luki prawie nie występują, w grze 3 są częste i zwykle dobre.

## 6. Punkt pracy, gra 4

- Punktu pracy nie ma: `ball/18/chosen.json` nie powstał.
- Ustawienia `track_ball` zostają jak w 04 (`max_gap` 2 s, bez odrzucania).
- Zgodnie z D1 gry 4 nie czytano (`ball_reliability.py holdout` odmawia bez `chosen.json`). Gry 5 nie czytano ani
  nie śledzono.

## Rekomendacja dla `ball-tracking/02` (bez `src/`)

- `track_ball` bez zmian: `max_gap` 2 s, bez odrzucania.
- Wynik niepewności można udostępnić jako pole pozycji (P(> 5 m) z cech toru). Nie powinien działać jako domyślny
  filtr. Model porządkuje pozycje lepiej niż długość luki: −1,5 pp przy 10% odrzuconych, a krzywa leży poniżej
  krzywej `max_gap`. Kalibracja między grami jest jednak słaba, a w granicy 2 pp straty mieści się tylko ok. 5%
  odrzuconych pozycji.
- Użytkownik, który woli mniej złych pozycji kosztem pokrycia (np. statystyki pozycyjne), może progować to pole sam.
- Przy przenoszeniu do pakietu potrzebne są cechy z `position_features`: gap, czas do pomiaru, prędkość implikowana z
  końców luki, pewność końców, ruch kamery i średnica. Współczynniki trzeba uczyć na danych z więcej niż dwóch gier.

## Odstępstwa i uwagi

- Prędkość implikowana i ruch kamery liczone są na homografiach propagowanych 04 i na ruchu kamery 04. Oba są
  dostępne przy inferencji.
- Krzywe łączą prawdopodobieństwa dwóch modeli (uczonych na grze 2 i na grze 3) na jednej osi progu. To typowe w
  walidacji krzyżowej.
- Bez GPU. Cały odczyt trwa około minuty na CPU.
