# ball-detection 12: poprawki śledzenia piłki

2026-10-26. Tylko CPU, tylko gry 2+3 GSR valid (39 klipów, 4 250 klatek mianownika, 347 bez homografii). Rejestracja i
siatka: komentarz w tickecie 12 (10:25), przed strojeniem i przed odczytem. Kod: `scripts/ball_track2.py` (`check`,
`tune`, `read`), testy `tests/test_ball_track2.py`, opcja `ball_temporal.py track --fixes PLIK`. Wyniki:
`ball/12/read.md`, `ball/12/read.json`, siatka `ball/12/tune/grid.md`, wybrane ustawienie `ball/12/chosen.json`.

**Odniesienie:** kandydat fazy 1, czyli tory 09 (`ball/09/track_valid`: dd2c 1280, kandydaci ≥ 0,01 przy 25 fps,
filtr 45 px, `track_ball` z `ball/04/tune/chosen.json`, ruch kamery `ball/04/motion`), punkt rzutowania = środek ramki
(ticket 11). Ustawienie 04 się nie zmienia. Zmieniają się tylko cztery poprawki w kopii logiki `track_ball`
(`track_ball2`). Pakiet (`src/`) jest nietknięty.

**Kontrole:**
- Wszystkie poprawki wyłączone dają tory identyczne z 09 (z filtrem) i z `ball/04/track` (bez filtra). Porównanie
  objęło status, ramkę i pewność w 29 250 klatkach gier 2+3, różnic jest 0. Top-1 też jest identyczny.
- Odczyt torów „wyłączone” odtwarza wiersz `selection` karty 09 (`ball/09/read_valid/read.json`) co do cyfry, 157
  liczb.
- `ball_temporal track --fixes` daje tory identyczne z `ball_track2` (wiersz „wszystkie”, 0 różnic).

## Poprawki

1. **Interpolacja w obrazie:** luka a → b jest wypełniana w pikselach klatki k jako (1 − s) · M_a→k(środek a) +
   s · M_b→k(środek b), gdzie M to złożony ruch kamery. Pozycję na boisku daje homografia klatki k. Przy przerwie
   ruchu w luce działa stara interpolacja na boisku, ale na grach 2+3 nie zdarzyło się to ani razu.
2. **Korytarz:** w klatkach luki używalny kandydat w promieniu `corridor` m od ścieżki interpolowanej staje się
   piłką zmierzoną.
3. **Granice restartów:** `_follow` zapisuje restarty. Luka między dwoma odcinkami nie jest wypełniana.
4. **Namalowane punkty:** kandydat przy punkcie karnym lub środkowym nie jest odrzucany. Może być wybrany przez
   bramkę w promieniu `spot_gate` m od przewidywania, ale nie może zrestartować toru. Wyjątek: kandydat, który nie
   jest statyczny w oknie `spot_window` s, jest zwykłym kandydatem.

## Dobór (klatki spoza benchmarku, gry 2+3)

Kryterium: odsetek dobrze umieszczonych (≤ 5 m) na 16 966 klatkach spoza benchmarku. Pozycja to rzut środka ramki
homografią przeniesioną ruchem kamery. Klatki benchmarku nie są czytane przy doborze. Pełna siatka (52 ustawienia) jest
w `ball/12/tune/grid.md`.

| wiersz | wybrane ustawienie | spoza benchmarku: dobrze | > 5 m |
|---|---|---|---|
| wyłączone | — | 73,03% | 12,55% |
| 1 interpolacja w obrazie | — | 72,95% | 12,63% |
| 2 korytarz | corridor 5 m (1 / 2 / 3 m: 73,03 / 73,07 / 73,12%) | 73,15% | 12,43% |
| 3 granice restartów | — | 72,28% | 10,87% |
| 4 punkty | spot_gate 1 m, spot_window 1,0 s (okno off: 73,15%) | 74,70% | 11,60% |
| wszystkie | corridor 3 m, spot_gate 1 m, spot_window 1,0 s (remis z 5 m, wygrywa mniejszy) | 73,69% | 10,19% |

`spot_gate` 1, 2 i 4 m dają prawie to samo. Prawie cały zysk poprawki 4 pochodzi z wyjątku „porusza się”, a nie z
samej bramki.

## Porównanie główne (zarejestrowane, odczyt po doborze)

**Żaden wiersz nie wygrał.** Żaden przedział zysku nie ma dolnej granicy > 0, więc `ball/12/chosen.json` = wszystkie
poprawki wyłączone (tracker fazy 1). Bariera jest spełniona wszędzie.

| wiersz | dobrze umieszczone | zysk vs faza 1, pp [95%] | > 5 m wśród wszystkich, pp [95%] | wygrana | vs top-1 dd2c, pp | > 5 m wśród umieszczonych | CPU ms/klatkę |
|---|---|---|---|---|---|---|---|
| faza 1 (odniesienie) | 71,25% [64,6; 77,5] | — | 11,76% | — | +19,65 [+13,81; +25,60] | 14,17% | 0,140 |
| 1 interpolacja w obrazie | 71,06% [64,5; 77,4] | −0,19 [−0,40; +0,02] | +0,19 [−0,02; +0,40] | nie | +19,46 | 14,40% | 0,153 |
| 2 korytarz (5 m) | 71,34% [64,6; 77,6] | +0,09 [−0,09; +0,28] | −0,09 [−0,28; +0,09] | nie | +19,74 | 14,06% | 0,153 |
| 3 granice restartów | 70,64% [64,0; 77,0] | **−0,61 [−1,03; −0,28]** | **−1,48 [−2,41; −0,78]** | nie | +19,04 | 12,71% | 0,142 |
| 4 punkty (1 m, 1,0 s) | 72,87% [67,1; 78,3] | **+1,62 [−0,07; +4,88]** | −0,89 [−3,00; +0,28] | nie | +21,27 | 12,98% | 0,170 |
| wszystkie | 72,02% [66,2; 77,6] | +0,78 [−1,21; +4,12] | **−2,14 [−4,42; −0,57]** | nie | +20,42 | 11,79% | 0,175 |

Wiersz pojedynczej klatki (top-1 dd2c ≥ 0,15) się nie zmienia, bo śledzenie nie zmienia detektora: 51,60%
[44,5; 59,3]. Koszt CPU to czas samego trackera na klatkę (bez budowy wyniku i rzutowania). Wszystkie poprawki razem
kosztują +0,035 ms/klatkę.

## Co robi każda poprawka (opisowo)

- **4 punkty:** +69 klatek netto, z czego +67 w jednym klipie (SNGS-056, gra 3). W tym klipie piłka leży przy
  namalowanym punkcie. Pozostałe klipy dają +3, +1 i −2. Stąd szeroki przedział: zysk jest prawdziwy, ale opiera się
  na jednym stałym fragmencie gry. Na grach 2+3 piłka prawdy jest w promieniu 0,6 m od punktu w 164 klatkach
  benchmarku (6 klipów), więc ten przypadek jest rzadki, ale ważny dla xG (rzuty karne, wznowienia). Przejścia
  statusów: 27 klatek „brak” → zmierzona (wszystkie dobrze), 33 interpolowane → zmierzone (14 → 31 dobrze), a w 2 623
  klatkach zmierzonych w obu wariantach wybór innej piłki daje 2 301 → 2 323 dobrze.
- **3 granice restartów:** blokuje 82 luki między odcinkami. Na klatkach mianownika 97 klatek interpolowanych
  zamienia się w „brak”. Wcześniej 26 z nich było dobrze, a 63 dalej niż 5 m. Poprawka zmniejsza więc liczbę pewnych,
  ale złych pozycji (−1,48 pp > 5 m wśród wszystkich, przedział cały poniżej zera), za cenę −0,61 pp dobrze
  umieszczonych. Miara główna liczy brak piłki jako pudło, ale nie jako błąd > 5 m. Dlatego tej wymiany nie
  nagradza.
- **1 interpolacja w obrazie:** przerwy ruchu nie zdarzyły się w żadnej luce. Na 1 027 interpolowanych klatkach
  mianownika daje 719 dobrze wobec 727 na boisku. Punkt z luki zależy tu od homografii klatki k, a nie od uśrednienia
  homografii końców. Zaszumienie zamrożonych homografii klatka do klatki szkodzi więc trochę bardziej niż liniowość
  ruchu na boisku. W obrazie jest lepiej (trafienie środka spoza benchmarku: 68,9% vs 66,5%), ale miara na boisku tego
  nie widzi.
- **2 korytarz:** 281 klatek luk dostaje zmierzonego kandydata (przy 5 m). Na mianowniku to 29 klatek (19 → 20 dobrze)
  i drobne przesunięcia ścieżek: +0,09 pp.

**Klatki zmierzone i interpolowane (gry 2+3, klatki mianownika):**

| wiersz | zmierzone: udział / dobrze / > 5 m | interpolowane: udział / dobrze / > 5 m | brak: udział |
|---|---|---|---|
| faza 1 | 61,8% / 87,7% / 9,5% | 24,2% / 70,8% / 24,4% | 14,1% |
| 1 interpolacja w obrazie | 61,8% / 87,7% / 9,5% | 24,2% / 70,0% / 25,2% | 14,1% |
| 2 korytarz | 62,4% / 87,5% / 9,7% | 23,5% / 71,2% / 23,9% | 14,1% |
| 3 granice restartów | 61,8% / 87,7% / 9,5% | 21,9% / 75,4% / 20,2% | 16,4% |
| 4 punkty | 63,1% / 88,7% / 8,5% | 23,5% / 71,6% / 23,5% | 13,3% |
| wszystkie | 63,7% / 88,2% / 9,0% | 20,7% / 76,7% / 18,9% | 15,6% |

Interpolowane w fazie 1 są tu dobrze w 70,8% (24,4% > 5 m), czyli podobnie jak na teście (69,6% / 25,5%). Granice
restartów podnoszą jakość interpolacji do 75,4% / 20,2%, bo usuwają luki łączące dwa różne obiekty.

## Rekomendacja

- Dla ticketów 13–16 tracker zostaje bez zmian (`ball/12/chosen.json`: wszystkie poprawki wyłączone).
- Poprawka 4 (punkty) jest najbliżej wygranej (+1,62 pp, dolna granica −0,07). Jej zysk pochodzi prawie w całości z
  jednego klipu. Na grach 2+3 nie da się go potwierdzić. Gdyby rada chciała ją sprawdzić dalej, trzeba nowego,
  zarejestrowanego odczytu na danych spoza gier 2+3. To decyzja o zbiorach oceny, nie decyzja techniczna.
- Poprawka 3 jest zmianą precyzja ↔ pokrycie: −1,48 pp pozycji > 5 m za −0,61 pp dobrze umieszczonych. Dla xG
  (pewna zła pozycja piłki jest gorsza niż jej brak) może mieć sens, ale miara fazy 2 tego nie nagradza.
- Poprawki 1 i 2 nie dają zysku na boisku.

**Odstępstwa:** brak. Decyzje techniczne (opisane w rejestracji): dobór na klatkach spoza benchmarku z homografią
przeniesioną ruchem kamery (jak 04, bez czytania klatek benchmarku); restart zawsze traktowany jako inny obiekt;
korytarz używa tych samych reguł `_usable` co bramka. Gier 4 i 5 oraz GSR test nie śledzono i nie czytano (homografie
zamrożone tylko dla klipów gier 2+3, jak w 11).
