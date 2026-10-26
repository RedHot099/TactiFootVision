# ball-detection 13: drugi stopień na wycinkach (rescorer + regresor środka)

2026-10-26. Tylko gry 2+3 GSR valid (39 klipów, 4 250 klatek mianownika, 347 bez homografii). Projekt według
decyzji rady (runda 2, `.scratch/ball-detection/council/round2-13-decision.md`). Rejestracja w komentarzach ticketu
13 z 10:25, 10:42 i 10:49, wszystkie przed pierwszym odczytem drugiego stopnia na grach 2+3. Kod:
`scripts/ball_crops.py`, testy `tests/test_ball_crops.py`. Duże wyniki: `/mnt/storage/tactifoot_research/detector_data/ball/13/`.

**Odniesienie:** kandydat fazy 1, czyli tory z 09 (`ball/09/track_valid`: dd2c 1280, kandydaci ≥ 0,01 przy 25 fps z
`ball/04/dump25`, filtr 45 px, `track_ball` z `ball/04/tune/chosen.json`, ruch kamery `ball/04/motion`), punkt =
środek ramki (11). Miara: odsetek dobrze umieszczonych (≤ 5 m), mianownik reguły 02, bootstrap parami po klipach
(2 000, seed 0). Wygrana: dolna granica > 0 i bariera (> 5 m wśród wszystkich klatek mianownika rośnie ≤ 1 pp).
Kontrola: kandydat fazy 1 czytany tym kodem daje wiersz `selection` karty 09 co do cyfry (157 liczb).

## Wynik

**Ticket nie wygrał.** Wiersz główny (ii), czyli rescoring + regresor z progami dobranymi na klatkach spoza
benchmarku, daje 72,66% wobec 71,25%: **+1,41 pp [−0,54; +3,34]**, a bariera jest spełniona (+0,61 pp). Dolna
granica jest poniżej zera. Na pojedynczej klatce drugi stopień daje dużo: top-1 po fuzji 62,40% wobec top-1 dd2c
51,60%, **+10,80 pp [+7,99; +13,97]**.

| wiersz | dobrze umieszczone | zysk vs faza 1, pp [95%] | > 5 m wśród wszystkich (Δ pp) | bariera | wygrana |
|---|---|---|---|---|---|
| kandydat fazy 1 (odniesienie) | 71,25% [64,6; 77,5] | — | 11,76% | — | — |
| (i) rescoring, progi dobrane | 72,14% [65,7; 78,0] | +0,89 [−0,81; +2,57] | 12,12% (+0,35 [−2,44; +2,90]) | tak | nie |
| **(ii) rescoring + regresor, progi dobrane (główny)** | **72,66% [66,9; 78,2]** | **+1,41 [−0,54; +3,34]** | 12,38% (+0,61 [−1,73; +2,96]) | tak | **nie** |
| rada dosłownie: (i) + progi 04 (0,05 / 0,15) | 70,12% [64,1; 75,8] | −1,13 [−2,99; +0,65] | 19,15% (+7,39 [+4,45; +10,83]) | **nie** | nie |
| rada dosłownie: (ii) + progi 04 | 70,47% [64,7; 76,0] | −0,78 [−2,88; +1,29] | 18,80% (+7,04 [+4,18; +10,52]) | **nie** | nie |
| sam regresor (conf dd2c, ramki przesunięte, ustawienie 04) | 71,62% [65,2; 77,7] | +0,38 [−0,02; +1,05] | 11,74% (−0,02 [−0,24; +0,24]) | tak | nie |
| (ii) vs (i) | 72,66% vs 72,14% | +0,52 [−0,14; +1,61] | +0,26 [−0,22; +0,97] | — | — |

Regresor wchodzi do wiersza głównego, bo (ii) > (i) w punkcie (+0,52 pp). Sam regresor na zrzucie dd2c daje
+0,38 pp z dolną granicą −0,02.

**Pojedyncza klatka** (top-1 po fuzji i filtrze 45 px vs top-1 dd2c ≥ 0,15 bez filtra, jak w 09):

| wiersz | dobrze umieszczone | zysk vs top-1 dd2c, pp [95%] | > 5 m wśród wszystkich (Δ pp) |
|---|---|---|---|
| top-1 dd2c ≥ 0,15 | 51,60% [44,5; 59,3] | — | 11,67% |
| rescoring, top-1 ≥ 0,15 | 62,40% [56,7; 68,3] | **+10,80 [+7,99; +13,97]** | 8,64% (−3,04 [−5,65; −0,76]) |
| rescoring + regresor, top-1 ≥ 0,15 | 62,42% [56,7; 68,3] | +10,82 [+7,99; +14,01] | 8,61% (−3,06 [−5,67; −0,82]) |

Próg 0,15 wybrała siatka {0,15 … 0,95} na klatkach spoza benchmarku. Top-1 po fuzji jest wciąż o 9 pp gorszy od
kandydata fazy 1. Śledzenie dalej daje dużo, ale po rescoringu dodaje mniej niż na samym dd2c (+10,2 pp zamiast
+19,7 pp).

**Gdzie jest zysk (ii) vs faza 1:** zmierzone 63,8% klatek (faza 1: 61,8%), dobrze 87,9% (87,7%), +0,33 pp
wszystkich klatek. Interpolowane 24,3% (24,2%), +1,22 pp. Bez piłki 11,9% (14,1%), −0,14 pp.

**Rozróżnianie kandydatów** (kandydaci piłki klatek co 5. gier 2+3 po filtrze, 7 573 prawdziwych i 3 794
fałszywych wg etykiet decyzji 3): przy progu, który zostawia 99% prawdziwych, conf dd2c zostawia 98,7% fałszywych,
a fuzja 62,5%.

## Krok 0: bramka oracle

Oracle (prawdziwi kandydaci 0,99, pozostali 0,001, tracker 04 i filtr bez zmian): 77,39%, **+6,14 pp [+4,15;
+8,40]** nad kandydatem fazy 1, czyli powyżej progu +3 pp, więc trening ruszył. Oracle z idealnym środkiem: +5,98 pp.
Pojedyncza klatka oracle: +7,36 pp nad top-1 dd2c. Wśród 500 klatek z pozycją fazy 1 > 5 m kandydat dd2c ≤ 5 m jest
w 27,4%, a wśród 375 klatek bez pozycji (z homografią) w 18,1%. Drugi stopień zrealizował ok. 1/4 sufitu oracle
(+1,41 z +6,14 pp). Tabela: `ball/13/oracle/oracle.md`.

## Co zbudowano

- **Widok** (`ball/13/view`): trening SoccerNet-v3 (26 361 obrazów z piłką, wszystkie z widoku run 2c), TeamTrack
  (4 065), GSR train gra 6 (13 583 klatek z piłką, 19 klipów). Kalibracja i wczesne zatrzymanie: gra 9 (20 klipów),
  co 5. klatka z piłką (2 816). Bez gry 4, GSR valid i GSR test (`check_view`, test). dd2c widział wszystkie te obrazy
  (widok run 2c), więc kandydaci są tam lepsi niż na nowych danych: na grze 6 jest 28 908 pozytywnych kandydatów i
  tylko 4 972 negatywnych (po przerzedzeniu), na grze 9 77% kandydatów to piłka.
- **Kandydaci:** dd2c 1280 ≥ 0,01, 154 202 kandydatów na 46 825 obrazach (16,7 ms/obraz).
- **Etykiety** (decyzja 3): r = max(3 px · skala, pół przekątnej prawdy), d ≤ r pozytyw, d ≥ 2r negatyw, pas
  pomijany. Pozytywy: kandydaci d ≤ r i jeden wycinek z przesuniętym środkiem wokół każdej prawdziwej piłki.
  Negatywy tylko z kandydatów dd2c; na GSR łańcuchy tej samej pomyłki w kolejnych klatkach przerzedzone (co 12.
  klatka, najwyżej 10). Zbiór treningowy: 117 405 pozytywów, 29 503 negatywów. Negatywów jest mniej niż 3 ×
  pozytywy, więc w stracie ważą tak, żeby w każdym źródle było 1:3.
- **Model:** ResNet18 (ImageNet), głowica klasyfikacji i przesunięcia środka; wycinek max(32 px, 3 × dłuższy bok)
  → 96×96, RGB. Ważone BCE + 5 × SmoothL1 (β 0,05). AdamW, one-cycle, 12 epok najwyżej; wybrana epoka 4 z 8 wg log
  loss pewności po fuzji na kalibracji. Wagi `ball/13/model/best.pt`, sha256
  `09e8d6441e843f98690c0e65b6e5b2830629ae29d37b3b8608f0639b673da179`.
- **Fuzja** (`ball/13/fusion/fusion.json`): sigmoid(0,620 · logit(p_crop) + 0,578 · logit(p_dd2c) + 3,084), zamrożona
  przed odczytem. Log loss na kalibracji: conf dd2c po rekalibracji 0,482, fuzja 0,247. Regresor: MAE środka 2,14 px
  wobec 2,87 px bez przesunięcia.
- **Zrzuty 25 fps GSR valid** (58 klipów, format `ball/04/dump25`): `ball/13/dump25_rescore` (conf = fuzja),
  `ball/13/dump25_regress` (conf = fuzja, ramka przesunięta), `ball/13/dump25_move` (conf dd2c, ramka przesunięta).
  `ball_temporal.py track --dump ball/13/dump25_rescore --post-filter --setting ball/13/tune/chosen.json --games 2 3
  --no-card-dumps` daje tory identyczne z torami odczytu (sprawdzone).
- **Progi trackera po fuzji** (`ball/13/tune`): siatka zarejestrowana o 10:49, dobór na 16 966 klatkach spoza
  benchmarku gier 2+3 miarą ticketu 04. Wybrane `min_confidence` 0,3, `restart_confidence` 0,5 (trafienie 68,5%);
  reszta ustawienia 04 bez zmian. Przy progach 04 trafienie 66,2%.

## Koszt

dd2c 1280 13,94 ms + drugi stopień 1,28 ms = **15,22 ms/klatkę** (300 klatek SNGS-021, batch 1, synchronizacja
CUDA, bez dekodowania obrazu; RTX 4070 Ti SUPER), limit 40 ms. W przebiegu zrzutu drugi stopień kosztował 1,88
ms na klatkę z kandydatami. GPU w tickecie: ok. 0,35 h (kandydaci 13 min, trening 2 × ok. 1–2 min, zrzut 3 min).

## Odstępstwa i uwagi

1. **Progi trackera dobrane po fuzji** zamiast „progi 04 bez zmian” z decyzji rady (pkt 5). Fuzja jest
   prawdopodobieństwem skalibrowanym na grze 9, gdzie 77% kandydatów to piłka. Na kalibracji fuzja ≥ 0,15 ma 92%
   kandydatów, a najniższa możliwa wartość (ok. 0,08) jest powyżej `min_confidence` 0,05. Treść ticketu przewiduje
   dobór progów na grach 2+3 z siatki zapisanej przed doborem; tak zrobiłem (rejestracja 10:49). Wiersz rady
   dosłownie jest w tabeli: −1,13 pp i bariera przekroczona (+7,39 pp > 5 m). Wiersz główny jest odczytem po doborze
   dwóch progów na sąsiednich klatkach tych samych gier, jak w 04.
2. **Wczesne zatrzymanie** po log loss pewności po fuzji, nie surowego p_crop (surowe p_crop jest źle skalibrowane z
   konstrukcji: trening 1:3, kalibracja 77% pozytywów). Pierwszy przebieg z kryterium surowym zatrzymał się na
   epoce 0 (`ball/13/model_try1_rawloss`, nieużywany). Zmiana przed odczytem 2+3.
3. **Proporcja 1:3 przez wagi**, bo negatywów jest za mało (dd2c widział obrazy treningowe).
4. `ball_temporal.py read --split valid` czyta też grę 5 (grupa „full”), więc odczyt robi `ball_crops.py read` tylko
   na grach 2+3 (jak 11) tym samym kodem karty (`placement_rows`, `Draws`, `paired`).
5. Wiersz ze śledzeniem z 12 nie powstał: 12 nie jest zamknięty. Zrzut `dump25_regress` z ustawieniem
   `ball/13/tune/chosen.json` może pójść do 12 i 16 bez zmian w kodzie.
6. Gry 4, gry 5 ani GSR test nie czytano. Zrzuty zawierają klipy gry 5 (cały GSR valid), ale bez odczytu.

## Wniosek

Drugi stopień dobrze oddziela piłkę od fałszywych kandydatów na pojedynczej klatce (+10,8 pp top-1, przy 99%
prawdziwych odrzuca 37% fałszywych zamiast 1%). Po śledzeniu z ustawieniem 04 i progami dobranymi po fuzji zysk
spada do +1,4 pp z przedziałem obejmującym zero. Tracker z fazy 1 odzyskiwał już większość tego, co rescoring
poprawia. Kandydat fazy 1 zostaje. Kierunki: wiersz pojedynczej klatki jest istotny dla t0 strzału w xG (16), a
łączny dobór bramki i luki trackera po fuzji nie był tu zarejestrowany.
