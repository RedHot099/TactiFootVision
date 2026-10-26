# ball-detection 14: poprawiony przepis fine-tune, ramiona A i B (twarde negatywy sędziów)

2026-10-26. Odczyt tylko na grach 2+3 GSR valid (39 klipów, 4 250 klatek mianownika). Rejestracja: komentarz ticketu
14 z 11:17, zapisany przed treningiem (commit 5af3c4e). Kod: `scripts/ball_recipe.py` (negatywy, listy, odczyt),
`RUNS["bd14_*"]` w `scripts/detector_train.py`, testy `tests/test_ball_recipe.py`. Duże wyniki leżą pod
`/mnt/storage/tactifoot_research/detector_data/ball/14/`, a wagi pod `runs/bd14_A` i `runs/bd14_B`.

**Przepis wspólny:** start z dd2c (sha256 05d706cf…), 1280, 1 epoka, `close_mosaic` 0 (mozaika do końca),
`warmup_epochs` 0, lr0 0,0005 (stałe przez epokę), EMA. Punkt kontrolny to `last.pt` (EMA po epoce), ustalony przed
treningiem. **A:** lista run 2c bez 13 500 klatek gry 4, czyli 100 698 linii. **B:** A plus 4 362 wycinki z
sędziami i liniowymi w czerni, bez piłki (105 060 linii), nic więcej.

**Pipeline odczytu:** zrzut 25 fps, filtr 45 px, `track_ball` z ustawieniem 04, ruch kamery 04, środek ramki.
Odniesienie fazy 1 to `ball/09/track_valid`. Kontrola: dd2c czytany tym kodem daje tory identyczne z 09 (0 z 29 250
klatek różnych) i 71,25%, czyli wiersz 09 co do cyfry.

## Wynik

- **Porównanie główne B − A (pipeline, gry 2+3): +2,26 pp [+0,92; +3,72].** Dolna granica jest > 0, więc **B jest
  lepsze i to B idzie jako przepis do ticketu 15.**
- **Ticket nie wygrywa.** B względem kandydata fazy 1: 70,64% wobec 71,25%, **−0,61 pp [−2,00; +0,69]**. Bariera
  jest spełniona (> 5 m wśród wszystkich −0,82 pp), ale dolna granica jest < 0. Strażnicy B względem dd2c są
  spełnieni.
- **Kontrola stabilności: sam przepis nie jest stabilny.** A traci względem dd2c: w pipeline'ie −2,87 pp
  [−5,02; −0,95] wobec kandydata fazy 1, na pojedynczej klatce −1,81 pp [−3,19; −0,51]. Na pojedynczej klatce rośnie
  też odsetek pozycji > 5 m (+5,65 pp). Strata pochodzi prawie w całości z gry 3, gdzie A daje więcej fałszywych piłek
  na sędziach (0,37 na klatkę wobec 0,20 przy dd2c, przed filtrem). Negatywy w B tę stratę odrabiają, ale nie dają
  zysku ponad dd2c.

| wiersz (gry 2+3) | odniesienie | wariant | zysk, pp [95%] | > 5 m wśród wszystkich, Δ pp [95%] | bariera |
|---|---|---|---|---|---|
| **B − A, pipeline (główne)** | 68,38% | 70,64% | **+2,26 [+0,92; +3,72]** | −0,75 [−2,70; +1,02] | tak |
| A − faza 1, pipeline (stabilność) | 71,25% | 68,38% | −2,87 [−5,02; −0,95] | −0,07 [−1,57; +1,53] | tak |
| **B − faza 1, pipeline (reguła 11)** | 71,25% | 70,64% | **−0,61 [−2,00; +0,69]** | −0,82 [−2,82; +1,02] | tak |
| B − A, pojedyncza klatka | 49,79% | 51,91% | +2,12 [+0,61; +3,91] | −7,39 [−10,66; −4,62] | tak |
| A − top-1 dd2c, pojedyncza klatka | 51,60% | 49,79% | −1,81 [−3,19; −0,51] | +5,65 [+3,25; +8,56] | **nie** |
| B − top-1 dd2c, pojedyncza klatka | 51,60% | 51,91% | +0,31 [−0,59; +1,21] | −1,74 [−2,62; −0,95] | tak |

Wiersze pojedynczej klatki to top-1 ≥ 0,15 bez filtra (wiersz 1280). Karta `ball_scorecard.py card` i zrzut 25 fps
dają te same liczby (−1,81, +0,31, +2,12).

## Rozbicie po grach (opisowo)

| wiersz | gra 2 | gra 3 |
|---|---|---|
| B − A, pipeline | −0,54 [−1,34; +0,18] | **+4,42 [+2,48; +6,71]** |
| A − faza 1, pipeline | +0,11 [−1,62; +1,56] | −5,17 [−8,36; −2,29] |
| B − faza 1, pipeline | −0,43 [−2,16; +0,97] | −0,75 [−2,92; +1,31] |
| B − A, pojedyncza klatka | −0,38 [−0,98; +0,17] | +4,05 [+1,70; +6,64] |
| A − top-1 dd2c | +0,54 [−0,05; +1,09] | −3,63 [−5,69; −1,44] |
| B − top-1 dd2c | +0,16 [−0,39; +0,74] | +0,42 [−1,07; +1,98] |

Cały efekt B − A pochodzi z gry 3, czyli z gry z sędziami w czerni. Na grze 2 A i B nie różnią się od dd2c.

## Fałszywe piłki na sędziach (wszystkie klatki z adnotacją, 25 fps)

Fałszywa piłka to kandydat ≥ 0,15 dalej niż r = max(5 px, średnica) od każdej prawdziwej piłki, ze środkiem w ramce
sędziego z GSR.

| gra 3 | na sędzim / klatkę | klatki z taką piłką | wszystkie fałszywe / klatkę |
|---|---|---|---|
| dd2c, surowe | 0,204 | 15,05% | 0,264 |
| A, surowe | 0,368 | 24,44% | 0,444 |
| B, surowe | **0,158** | 11,67% | 0,213 |
| dd2c / A / B po filtrze 45 px | 0,0001 / 0,0013 / 0,0006 | 0,01% / 0,13% / 0,05% | 0,045 / 0,049 / 0,042 |
| dd2c / A / B, piłka śledzona | 0,0004 / 0,0030 / 0,0006 | 0,04% / 0,30% / 0,06% | 0,285 / 0,262 / 0,238 |

Na grze 2 takich pomyłek prawie nie ma (≤ 0,0007 na klatkę u wszystkich). Negatywy zmniejszają surowe fałszywe piłki
na sędziach poniżej poziomu dd2c (−23%), a względem A o 57%. Filtr 45 px i tak usuwa prawie wszystkie, dlatego w
pipeline'ie B zaledwie odrabia stratę A.

## Karty i strażnicy (wiersz 1280, gry 2+3; Ekstraklasa dev)

| | mAP50-95 ludzi (Δ vs dd2c) | Ekstraklasa mAP50-95 (Δ) | AP50 piłki (Δ) | recall piłki | fałszywe / klatkę ≥ 0,15 | ms/klatkę |
|---|---|---|---|---|---|---|
| dd2c | 0,5137 | 0,4240 | 0,296 | | 0,208 | 19,6 |
| A | 0,5180 (+0,0044) | 0,4348 (+0,0108) | 0,233 (−0,063) | 0,346 | 0,321 | 19,6 |
| B | 0,5382 (+0,0245) | 0,4163 (−0,0077) | 0,304 (+0,008) | 0,361 | 0,190 | 19,7 |

Strażnicy B względem dd2c są spełnieni (ludzie ≥ −0,01, Ekstraklasa ≥ −0,02). B względem A: ludzie +0,020,
Ekstraklasa −0,019 (tuż nad progiem). Ball AP50 na Ekstraklasie 0,286 (A 0,324). Koszt detekcji się nie zmienia:
zrzut 25 fps 17,5 / 17,6 ms na klatkę.

## Wagi

- A: `runs/bd14_A/yolo11l_1280/weights/last.pt`, sha256 `f291ee3d00b998c0fd82aa0905dfc6d08ad9747e25b87d1a1c7a7500bb9b629e`
- B: `runs/bd14_B/yolo11l_1280/weights/last.pt`, sha256 `bbf163dd85a60dad8d9e855ffa41c3cc52e511f852ce90a3a3a65631c5ebb28e`

Wyniki: zrzuty `ball/14/{A,B}/dump25`, karty `ball/14/{A,B}/card` (vs dd2c) i `ball/14/B_vs_A_card`, odczyt
`ball/14/read/` (`read.md`, `read.json`, `table.csv`, `referees.csv`, tory), negatywy `ball/14/negatives/`, logi i
czasy GPU `ball/14/logs/`.

## Wnioski

1. **Przepis dla 15: B.** B − A ma dolną granicę > 0 w porównaniu głównym.
2. Poprawiony przepis bez negatywów (A) psuje dd2c na grze 3: więcej fałszywych piłek na sędziach w czerni i mniej
   dobrze umieszczonych (−2,87 pp w pipeline'ie). Mozaika do końca i brak rozgrzewki nie wystarczają: jedna epoka
   dalszego treningu na tym samym widoku (bez gry 4) przesuwa model w stronę sędziów jako piłek.
3. Negatywy działają tam, gdzie miały działać (gra 3, surowe fałszywe piłki na sędziach −57% wobec A, −23% wobec
   dd2c). Nie dają jednak nowego kandydata, bo filtr 45 px kandydata fazy 1 usuwał już prawie wszystkie te pomyłki.
4. Ticket 15 (1600 px) dostaje przepis B. Wyniku 15 trzeba porównywać z B 1280, a nie z dd2c. Przy A − dd2c ujemnym
   przepis sam w sobie kosztuje ok. 2–3 pp na grze 3.

## Odstępstwa i uwagi

- Negatywy to wycinki (powiększenie 1,25–2× wobec pełnej klatki), nie całe klatki. Epoka B jest o 4,3% dłuższa: to
  dodane negatywy.
- Reguła „w czerni” to statystyka pikseli istniejącej ramki sędziego (mediana HSV V < 80, S < 120 w torsie), nie
  etykieta AI. Progi ustawione na wycinkach SoccerNet-v3 z treningu przed jakimkolwiek treningiem. Negatywy pochodzą
  głównie z SoccerNet-v3 (3 869 z 4 362), bo sędziowie GSR train gier 6+9 nie są w czerni.
- Zrzut 25 fps obejmuje 58 klipów GSR valid (jak w 04/13), ale śledzone i czytane były tylko klipy gier 2+3. Karty
  bez `--confirm`. Gry 4, gry 5 ani GSR test nie czytano.
- Ruch kamery jest z 04 (dd2c), tak jak w 13. Ludzie dla `track_ball` pochodzą ze zrzutu danego ramienia.

## GPU

Smoke B 11:14–11:16. Jednostka `bd14-gpu` 11:18–15:23: trening A 1 h 31 min, trening B 1 h 35 min, zrzuty 25 fps
2 × 13 min, karty 2 × 14 min + karta B vs A 4 min. Razem **ok. 4,1 GPU-h**. Blokada ustawiona przed startem i
zwolniona przez łańcuch o 15:23.
