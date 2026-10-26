# ball-detection: raport fazy 2

2026-10-26. Branch `research/ball-detection`, tickety 10–16 (`.scratch/ball-detection/issues/`). Odniesieniem dla
każdego ticketu był kandydat fazy 1: dd2c 1280 (`models/detector_yolo11l_1280_dd2c.pt`, sha256 05d706cf…), kandydaci
piłki ≥ 0,01 przy 25 fps, filtr 45 px, `track_ball` z ustawieniem z ticketu 04 i środek ramki jako punkt rzutowania.
Raport fazy 1: [REPORT.md](REPORT.md). Duże wyniki leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/`.

## Wynik w skrócie

- **Faza 2 nie dała nowego kandydata.** Żaden z ticketów 11–15 nie spełnił reguły wygranej, czyli dolnej granicy
  > 0 na grach 2+3 względem kandydata fazy 1. Nie było więc złożenia ani potwierdzeń. **Gry 4 i 5 zostały
  nieprzeczytane.** Kandydat fazy 1 zostaje kandydatem pipeline'u piłki w wideo.
- Najbliżej wygranej były: punkty namalowane w śledzeniu (+1,62 pp [−0,07; +4,88], prawie cały zysk z jednego
  klipu), drugi stopień na wycinkach (+1,41 pp [−0,54; +3,34]) i trening 1600 px (+0,40 pp [−1,60; +2,35] względem
  fazy 1, +1,01 pp [−1,12; +3,15] względem tego samego przepisu przy 1280).
- **Na pojedynczej klatce drugi stopień z 13 daje dużo:** top-1 po fuzji 62,40% wobec 51,60% dla top-1 dd2c, czyli
  +10,80 pp [+7,99; +13,97] na grach 2+3, za 1,3 ms/klatkę. W xG (strzały rozwojowe, opisowo) piłka własna t0
  rośnie z 60,3% do 80,8%, a miejsce strzału zmienia się w 16% strzałów. Czy na lepsze, tego bez prawdy nie wiemy.
- Twarde negatywy sędziów w czerni działają w przepisie 1280 (B − A +2,26 pp [+0,92; +3,72]), ale tylko odrabiają
  stratę poprawionego przepisu. Filtr 45 px kandydata fazy 1 usuwał już prawie wszystkie te pomyłki.

## Cel i reguły fazy 2

Cel (`spec-phase2.md`): sprawdzić, czy któryś z kierunków otwartych w fazie 1 daje lepszego kandydata pipeline'u
piłki w wideo, zmierzonego tą samą miarą i potwierdzonego na danych, na których nic nie było dobierane. Reguły ustalił
właściciel 2026-10-26 (grilling rund 1–2, `spec.md`, „Decyzje właściciela (2026-10-26, faza 2)”):

- **Miara** bez zmian względem fazy 1: odsetek dobrze umieszczonych klatek (≤ 5 m na zamrożonych homografiach
  `kp03_r2_refined`). Mianownik reguły 02: klatki co 5. z dokładnie jedną piłką na boisku; klatki bez homografii
  liczą się jako pudło. Przedziały 95%: bootstrap parami po klipach, 2 000 losowań, seed 0. Prawdą jest dół ramki z
  `bbox_pitch`.
- **Odniesienie:** kandydat fazy 1. Obok zawsze wiersz pojedynczej klatki względem top-1 dd2c (≥ 0,15).
- **Ticket wygrywa**, gdy na grach 2+3 GSR valid (39 klipów, 4 250 klatek mianownika) dolna granica zysku jest > 0 i
  spełniona jest bariera: odsetek klatek mianownika z pozycją > 5 m rośnie najwyżej o 1 pp.
- **Faza wygrywa**, gdy kandydat końcowy (złożenie zwycięzców, zamrożone) ma na grze 4 dolną granicę > 0, a na grze 5
  dolną granicę > 0 i punkt ≥ +2 pp, z barierą na obu grach. Gdy nic nie wygrało, gry 4 i 5 nie są czytane.
- **Zbiory:** wszystko, co dobierane, dobiera się na grach 2+3. Gra 4 GSR train (18 klipów) to holdout wyłączony z
  każdego widoku treningowego fazy 2. Gra 5 to jeden ostry odczyt. GSR test nie jest już czytany.
- Strażnicy dla nowych detektorów: mAP50-95 ludzi ≥ −0,01, Ekstraklasa dev mAP50-95 ≥ −0,02 względem dd2c. Limit
  kosztu kandydata wideo: ≤ 40 ms/klatkę GPU. Trening 1600 tylko po osobnej zgodzie Jakuba (zgoda 15:50).
- StatsBomb tylko do walidacji. Bez etykietowania przez AI. Kod tylko w `scripts/`, bez zmian w `src/`.

## Wyniki ticketów (gry 2+3, względem kandydata fazy 1: 71,25% [64,6; 77,5])

| ticket | co | porównanie główne, pp [95%] | bariera (> 5 m, Δ pp) | werdykt | raport |
|---|---|---|---|---|---|
| 10 | gra 4 jako holdout: homografie, zrzuty, kod odczytu | brak odczytu predykcji; sufit 89,3% [83,6; 94,6] (gry 2+3: 91,8%) | | przygotowane, nieczytane | [10](10/results.md) |
| 11 | punkt rzutowania: dół ramki / przesunięty środek | dół ramki −0,68 [−1,03; −0,37]; k = 0,25 −0,21 [−0,41; −0,05] | +0,68 / +0,21 | nie wygrał, zostaje środek | [11](11/results.md) |
| 12 | poprawki śledzenia (4 wiersze + wszystkie) | najlepiej punkty namalowane +1,62 [−0,07; +4,88]; wszystkie +0,78 [−1,21; +4,12] | −0,89 / −2,14 | nie wygrał, tracker fazy 1 | [12](12/results.md) |
| 13 | drugi stopień na wycinkach (rescorer + regresor) | +1,41 [−0,54; +3,34]; pojedyncza klatka +10,80 [+7,99; +13,97] vs top-1 dd2c | +0,61 | nie wygrał | [13](13/results.md) |
| 14 | poprawiony przepis, A vs B (negatywy sędziów) | B − A +2,26 [+0,92; +3,72]; B − faza 1 −0,61 [−2,00; +0,69] | −0,82 | B lepsze od A; nie wygrał | [14](14/results.md) |
| 15 | trening i inferencja 1600 px (przepis B) | 1600 − B 1280 +1,01 [−1,12; +3,15]; vs faza 1 +0,40 [−1,60; +2,35] | −0,28 | nie wygrał | [15](15/results.md) |
| 16 | złożenie, gra 4, gra 5, xG, raport | brak zwycięzców: bez złożenia i potwierdzeń; xG opisowo | | faza bez nowego kandydata | [16](16/results.md) |

Każdy ticket miał kontrolę regresji: kandydat fazy 1 przeczytany nowym kodem odtwarza kartę 09 co do cyfry (157 liczb),
a wyłączone zmiany dają tory identyczne z 04/09.

## Stan kierunków

| kierunek | ticket | stan | najważniejsza liczba |
|---|---|---|---|
| holdout gra 4 (homografie, zrzuty, odczyt) | 10 | gotowy, nieczytany | 2 062 klatki mianownika, 221 bez homografii |
| punkt rzutowania: dół ramki | 11 | nie wygrał | −0,68 pp [−1,03; −0,37] |
| punkt rzutowania: środek przesunięty (k = 0,25, po doborze) | 11 | nie wygrał | −0,21 pp [−0,41; −0,05] |
| śledzenie: interpolacja w obrazie | 12 | nie wygrał | −0,19 pp [−0,40; +0,02] |
| śledzenie: korytarz wokół ścieżki | 12 | nie wygrał | +0,09 pp [−0,09; +0,28] |
| śledzenie: granice restartów | 12 | nie wygrał | −0,61 pp [−1,03; −0,28], ale > 5 m −1,48 pp [−2,41; −0,78] |
| śledzenie: punkty namalowane | 12 | nie wygrał | +1,62 pp [−0,07; +4,88] (jeden klip, SNGS-056) |
| śledzenie: wszystkie poprawki | 12 | nie wygrał | +0,78 pp [−1,21; +4,12] |
| drugi stopień w wideo (z trackerem) | 13 | nie wygrał | +1,41 pp [−0,54; +3,34]; oracle +6,14 pp |
| drugi stopień na pojedynczej klatce | 13 | opisowo, bez reguły wygranej | +10,80 pp [+7,99; +13,97] vs top-1 dd2c |
| poprawiony przepis bez negatywów (A) | 14 | niestabilny | −2,87 pp [−5,02; −0,95] vs faza 1 |
| przepis z negatywami sędziów (B) | 14 | lepszy od A, nie wygrał | −0,61 pp [−2,00; +0,69] vs faza 1 |
| trening 1600 px | 15 | nie wygrał | +0,40 pp [−1,60; +2,35] vs faza 1 |
| głowa P2 | 15 | nie uruchomiony | reguła z 15 niespełniona: < 8 px +8,22 [−1,14; +16,13] na 146 klatkach |
| złożenie zwycięzców, gra 4, gra 5 | 16 | nie uruchomiony | brak zwycięzców |
| SoccerSynth (06), ablacja źródeł (07), sieci wieloklatkowe, samouczenie | — | poza zakresem fazy 2 | |

Co z tego wynika:

- **Punkt rzutowania nie jest źródłem błędów > 5 m (11).** Rzut środka piłki ląduje dalej od kamery o medianę
  +0,38 m, zgodnie z geometrią. Rozrzut wzdłuż promienia kamery (rozstęp kwartylowy ok. 1,4 m) jest jednak kilka
  razy większy, a ogon błędów leży po stronie kamery. Przesunięcie w dół wypycha więc klatki z 4–5 m za próg.
- **Tracker fazy 1 odzyskuje już większość tego, co dałby lepszy detektor (13, 14, 15).** Drugi stopień poprawia
  top-1 na pojedynczej klatce o 10,8 pp, a w pipeline'ie zostaje z tego +1,4 pp. Tracker dokłada po rescoringu
  +10,2 pp zamiast +19,7 pp. Drugi stopień zrealizował ok. 1/4 sufitu oracle (+1,41 z +6,14 pp).
- **Sędziowie w czerni to problem gry 3 i filtra.** W 14 negatywy zmniejszyły surowe fałszywe piłki na sędziach o
  57% względem A i o 23% względem dd2c, ale po filtrze 45 px było ich już prawie zero. W 15 (1600) negatywy z
  wycinków w skali 1280 przestały działać: surowych fałszywych piłek na sędziach było 0,238 na klatkę wobec 0,158
  w B, i gra 3 zjadła zysk gry 2 (+2,48 pp [+0,57; +4,72]).
- **Granice restartów (12) wymieniają pokrycie na precyzję:** −0,61 pp dobrze umieszczonych za −1,48 pp pozycji
  > 5 m. Miara fazy liczy brak piłki jako pudło, a złą pozycję nie karze dodatkowo, więc tej zmiany nie nagradza.

## xG (opisowo, ticket 16)

Szczegóły: [16/results.md](16/results.md). Rejestracja w tickecie 16 (22:16) przed przebiegiem. Strzały: 7 062
strzały bez karnych z 328 meczów rozwojowych, żaden z 172 odłożonych meczów. Reguła miejsca strzału: zamrożona
`feet_t0` (piłka z 0, potem −400, potem −800 ms; miejsce strzału = stopy gracza najbliżej piłki w obrazie). Pipeline
xG dziś: top-1 dd2c 1280 ≥ 0,15 na klatce 720p, bez filtra 45 px. Kontrola: ramię „xG dziś” odtwarza cache xG bez
żadnej różnicy (14 124 wiersze cech).

| wiersz t0 | xG dziś | kandydat fazy 1 (+ filtr 45 px) | drugi stopień 13 | drugi stopień − faza 1, pp [95%] |
|---|---|---|---|---|
| `ball_ok` | 92,42% | 91,96% | 97,07% | +5,11 [+4,51; +5,73] |
| piłka własna t0 | 61,33% | 60,29% | 80,83% | +20,53 [+19,23; +21,78] |
| zapas −400 ms | 26,59% | 27,13% | 14,13% | −13,00 [−14,09; −11,96] |
| zapas −800 ms | 4,50% | 4,53% | 2,11% | −2,42 [−2,89; −1,98] |
| brak piłki | 7,58% | 8,04% | 2,93% | −5,11 [−5,73; −4,51] |
| inne miejsce strzału niż faza 1 (> 5 m) | | | 16,05% (9,23%) | |

Przedziały to bootstrap par po meczach (2 000, seed 0).

- Drugi stopień znajduje piłkę t0 w 1 477 strzałach (20,9%), w których kandydat fazy 1 brał piłkę z −400 ms (1 085),
  z −800 ms (160) albo nie miał jej wcale (232). W ponad połowie z nich zmienia się strzelec, a w 31% miejsce
  strzału przesuwa się o > 5 m.
- **Błędu `feet_t0` względem StatsBomb nie zmierzyłem.** Parowanie ze StatsBomb istnieje tylko dla 135 meczów
  ewaluacyjnych, a wszystkie należą do 172 odłożonych. Na meczach rozwojowych nie ma więc prawdy. Proxy bez prawdy
  (stopy strzelca ≤ 2 m od rzutu piłki) nie spada: +1,48 pp [+0,71; +2,24] na strzałach z piłką w obu ramionach.
  Galeria 40 niezgodności piłki t0 do obejrzenia: `ball/16/xg/gallery/`.
- Filtr 45 px zmienia w xG mało: usuwa piłkę t0 w 1,4% strzałów, a miejsce strzału zmienia w 0,8%.

## Rekomendacja dla `ball-tracking/02`

1. **Wideo: kandydat fazy 1 bez zmian.** dd2c 1280 + filtr 45 px + `track_ball` z ustawieniem 04 (`min_confidence`
   0,05, `restart_confidence` 0,15, `restart_after` 0,25 s, `gate` 3,0 m, `gate_speed` 65 m/s, `max_gap` 2,0 s, test
   nieruchomości wyłączony, punkt rzutowania = środek ramki). Szczegóły i koszt w [REPORT.md](REPORT.md),
   „Rekomendacja dla pipeline'u”. Poprawek śledzenia z 12 nie przenosić jako domyślnych.
2. **Pojedyncze klatki (np. t0 strzału w xG): drugi stopień z 13 jako opcja, nie domyślnie.** Wycinek
   max(32 px przy 1080p, 3 × dłuższy bok) → 96×96, ResNet18 (`ball/13/model/best.pt`, sha256 09e8d644…), fuzja
   sigmoid(0,620 · logit(p_crop) + 0,578 · logit(p_dd2c) + 3,084), regresor środka, filtr 45 px, top-1 fuzji ≥ 0,15.
   Kosztuje +1,3 ms/klatkę. Kafle 2×2 z fazy 1 kosztowały 3,8× czasu detekcji. Zastrzeżenia: +10,8 pp pochodzi z
   gier 2+3, na których dobrano próg 0,15 (na klatkach spoza benchmarku). Nie ma potwierdzenia na grze 4 ani 5. W
   xG zmienia miejsce strzału w 16% strzałów, a nie wiemy, czy na lepsze. Domyślnym wyborem w xG może się stać
   dopiero po odczycie C6 linii xG na danych rozwojowych.
3. Wagi drugiego stopnia, B i 1600 są tylko do badań (jak dd2c). Do pakietu przenosi się kod opcji, nie wagi.

## Tropy na później

- **Negatywy sędziów w skali 1600** (15): przy 1600 sędziowie są o 25% więksi niż w wycinkach negatywów. Gra 2
  pokazała zysk 1600 (+2,48 pp [+0,57; +4,72]), a gra 3 go zjadła. Kolejny krok to 1600 z negatywami wyciętymi w
  skali 1600.
- **Punkty namalowane w śledzeniu** (12, poprawka 4): +1,62 pp, z czego +67 z 69 klatek w jednym klipie ze stałym
  fragmentem gry. Ważne dla xG (rzuty karne, wznowienia). Wymaga nowego, zarejestrowanego odczytu poza grami 2+3, a
  to jest decyzja o zbiorach oceny dla Jakuba.
- **Małe piłki** (15): zysk 1600 skupia się punktowo w piłkach < 8 px (+8,22 pp, 146 klatek, 3,4% mianownika) i
  ≤ 14 px (41 z 43 klatek netto). To za mało klatek, żeby rozstrzygnąć sprawę P2. Potrzebny zbiór z większą liczbą
  małych piłek.
- **Drugi stopień i tracker razem** (13): oracle pokazuje +6,14 pp sufitu, zrealizowano +1,41 pp. Bramki i luki
  trackera dobierane wspólnie z fuzją nie były zarejestrowane.
- **Granice restartów dla xG** (12): mniej pewnych, ale złych pozycji (−1,48 pp > 5 m). Ma sens tam, gdzie zła
  pozycja kosztuje więcej niż jej brak.
- **Drugi stopień w stosie xG:** odczyt C6 na danych rozwojowych z piłką S2 (linia xG). Trzeba też zdecydować, czy
  flagi źródła piłki t0 pójdą za S2.

## Zbiory nieprzeczytane w fazie 2

- **Gry 4 i 5 zostały nieprzeczytane w fazie 2.** Nie było kandydata końcowego, więc zgodnie z rejestracją 16 nie
  było odczytów potwierdzających. Wiersza pojedynczej klatki z 13 też nie czytano na grze 4. Gra 5 była czytana w
  fazie 1 (04, 08), więc dla przyszłej fazy jest słabszą ochroną przed doborem po danych niż gra 4. GSR test nie
  był czytany w fazie 2.
- **Gotowe dla przyszłego odczytu gry 4 (10):** zamrożone homografie `bench/gsr_holdout_homographies.parquet`
  (sha256 cd413832…), zrzut 25 fps dd2c `ball/10/dump25_holdout`, ruch kamery `ball/10/motion_holdout`, zrzut karty
  dd2c `ball/10/card_dump_dd2c_holdout` i polecenia odczytu w [10/results.md](10/results.md). Gra 4 nie trafiła
  do żadnego widoku treningowego fazy 2 (13, 14, 15).
- **Uwaga o homografiach gry 4:** model punktów boiska (sha256 5be9954a…) był trenowany na 2 474 klatkach gry 4, a
  to te same klatki co 5., na których leży mianownik. Homografie gry 4 są więc dla modelu punktów „in-sample”.
  Błąd punktów w pikselach jest mniejszy niż na valid (mediana 6,4 vs 7,8 px), ale w metrach przewagi nie ma
  (mediana błędu osób 0,76 vs 0,59 m), bo dopasowanie do linii zadziałało tylko na 25% klatek (na valid 78%). Sufit
  wynosi 89,3% wobec 91,8% na grach 2+3. Porównanie parami używa tych samych homografii po obu stronach, więc to
  dotyczy poziomu i sufitu, a nie różnicy. Do tego dd2c widział grę 4 w widoku run 2c: odczyt jest zachowawczy dla
  wariantów trenowanych bez gry 4 i neutralny dla wariantów tylko inferencyjnych.

## Koszt (ms/klatkę GPU, RTX 4070 Ti SUPER)

| wariant | ms/klatkę | źródło |
|---|---|---|
| kandydat fazy 1: dd2c 1280 + filtr + `track_ball` | 19,6 (karta) / 17,6 (zrzut 25 fps) + 0,14 CPU tracker | 09, 12, 14 |
| + drugi stopień 13 | 13,94 + 1,28 = 15,22 (pomiar batch 1 z synchronizacją) | 13 |
| B 1280 | 19,7 (karta) / 17,6 (zrzut) | 14 |
| 1600 | 28,5 (karta) / 25,9 (zrzut) | 15 |
| xG: dd2c 1280 na 720p + drugi stopień | 12,9 + 1,5 | 16 |

Wszystkie warianty mieszczą się w limicie 40 ms/klatkę.

## Zużycie GPU

| ticket | 10 | 11 | 12 | 13 | 14 | 15 | 16 | razem |
|---|---|---|---|---|---|---|---|---|
| GPU-h | 0,25 | 0 (CPU) | 0 (CPU) | 0,35 | 4,1 | 3,2 | 0,13 | **ok. 8,0** |

Razem z fazą 1 (ok. 11,2 GPU-h) linia ball-detection zużyła ok. 19,2 GPU-h.

## Licencje

Wagi B, 1600 i drugiego stopnia mają ten sam status co dd2c: **tylko do badań** (decyzja 11). Trenowano je na danych
SoccerNet objętych NDA, na źródłach CC BY-NC i na źródłach bez licencji. Strzały xG (mecze rozwojowe) posłużyły tylko
do opisowego odczytu wyboru piłki. Nie czytano etykiet goli, StatsBomb, gier 4 i 5 ani GSR test.
