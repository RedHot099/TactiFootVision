# ball-detection 11: audyt punktu rzutowania piłki

2026-10-26. Tylko CPU, tylko gry 2+3 GSR valid (39 klipów, 4 250 klatek mianownika, 347 bez homografii). Rejestracja
w treści ticketu 11, potwierdzona komentarzem 09:42 przed odczytem wariantów. Kod: `scripts/ball_projection.py`
(`read`), testy `tests/test_ball_projection.py`. Wyniki: `ball/11/read.md`, `ball/11/read.json`.

**Odniesienie:** kandydat fazy 1, czyli tory z ticketu 09 (`ball/09/track_valid`: dd2c 1280, kandydaci ≥ 0,01 przy
25 fps z `ball/04/dump25`, filtr 45 px, `track_ball` z `ball/04/tune/chosen.json`, ruch kamery `ball/04/motion`).
Piłka jest rzutowana ze środka ramki. Prawda to dół ramki z `bbox_pitch`.

**Wariant:** punkt (cx, cy + k · h/2) ramki wybranej piłki w pikselach 1080p. k = 0 to środek, k = 1 to dół ramki
(wariant a). Wariant (b) to k z siatki {0,25; 0,5; 0,75; 1} o największym punktowym zysku na grach 2+3, więc jego
odczyt jest odczytem po doborze. Punkt wybiera się w zrzucie wyboru piłki (kolumny `point_x`/`point_y`). Odczyt
(`detector_bench.placement_rows`, `ball_scorecard.Draws` / `placement_draws` / `paired`) się nie zmienia. Śledzenie
nie jest uruchamiane ponownie: `track_ball` bramkuje na środkach ramek, więc w każdej klatce wybrana piłka jest ta
sama co w 09, a wariant zmienia tylko punkt. Przedziały 95%: bootstrap po klipach, 2 000 losowań, seed 0.

**Kontrola:** wariant „środek” (k = 0) przez nowy kod odtwarza wiersz `selection` karty 09
(`ball/09/read_valid/read.json`) co do cyfry: 157 liczb identycznych (liczności, tor, top-1, zysk parami, wiersze
statusów), w tym 71,25% vs 51,60% i +19,65 pp [+13,81; +25,60].

## Porównanie główne (zarejestrowane)

**Żaden wariant nie wygrał. Każde przesunięcie punktu w dół obniża odsetek dobrze umieszczonych, a przedział jest
cały poniżej zera. Dla ticketów 12–16 zostaje środek ramki.**

| k | tor: dobrze umieszczone | zysk vs środek, pp [95%] | > 5 m wśród wszystkich klatek (Δ pp) | bariera ≤ +1 pp | wygrana | mediana błędu, m (Δ) | pojedyncza klatka: top-1 dd2c | top-1 zysk vs środek, pp |
|---|---|---|---|---|---|---|---|---|
| 0 (środek, odniesienie) | 71,25% [64,6; 77,5] | — | 11,76% | — | — | 1,128 | 51,60% [44,5; 59,3] | — |
| 0,25 = wariant (b), po doborze | 71,04% [64,4; 77,4] | **−0,21 [−0,41; −0,05]** | 11,98% (+0,21 [+0,05; +0,41]) | tak | nie | 1,050 (−0,078 [−0,101; −0,036]) | 51,60% | +0,00 [−0,21; +0,23] |
| 0,5 | 70,92% [64,2; 77,3] | −0,33 [−0,57; −0,12] | 12,09% (+0,33) | tak | nie | 1,014 (−0,114) | 51,51% | −0,09 [−0,38; +0,26] |
| 0,75 | 70,71% [64,1; 77,1] | −0,54 [−0,82; −0,29] | 12,31% (+0,54) | tak | nie | 0,996 (−0,133) | 51,48% | −0,12 [−0,47; +0,30] |
| 1 = wariant (a), dół ramki | 70,56% [63,9; 77,0] | **−0,68 [−1,03; −0,37]** | 12,45% (+0,68 [+0,37; +1,03]) | tak | nie | 1,013 (−0,115 [−0,186; −0,025]) | 51,36% [44,3; 59,1] | −0,24 [−0,64; +0,22] |

Wariant (b) wybrał k = 0,25, bo przy wszystkich k zysk jest ujemny, a 0,25 traci najmniej. Bariera jest spełniona
wszędzie (najwyżej +0,68 pp), ale żaden wariant nie ma dolnej granicy > 0. Wiersz pojedynczej klatki (top-1 dd2c
≥ 0,15 z punktem k wobec top-1 dd2c ze środkiem) też nie zyskuje: punkty od 0,00 do −0,24 pp, wszystkie przedziały
obejmują zero. Tor z k = 1 wobec top-1 ze środkiem (jak karta 09): +18,96 pp [+13,12; +24,95] zamiast +19,65 pp.

Skąd strata: przy k = 1 na granicy 5 m traci się 37 klatek (16 zmierzonych i 21 interpolowanych; mediana ich błędu
przy środku 4,64–4,76 m), a zyskuje 8 (5 i 3). Netto to −29 z 4 250 klatek. Przy k = 0,25: −11 i +2. Przesunięcie
w dół na obrazie przesuwa punkt na boisku w stronę kamery. Klatki z błędem 4–5 m przy środku leżą w 67% po
stronie kamery (85 klatek), a 36 z 37 straconych ma predykcję bliżej kamery niż prawda (mediana −4,2 m wzdłuż
promienia). Ogon błędów jest po stronie kamery (średnia składowej wzdłuż promienia −0,24 m przy medianie
+0,34 m), więc przesunięcie w dół wypycha go za 5 m. Jednocześnie mediana błędu spada o ok. 0,1 m. Środek rzutu jest więc trochę dokładniejszy w masie
rozkładu, ale ta różnica jest dużo mniejsza od progu 5 m i miara jej nie widzi.

## Diagnostyka (opisowo)

Zmierzone klatki toru (status „measured”) z homografią i pozycją: 2 550. Błąd (predykcja − prawda) na boisku
rozłożony na składową wzdłuż promienia od rzutu kamery na boisko (+ = dalej od kamery) i poprzeczną. Pozycja
kamery pochodzi z rozkładu zamrożonych homografii (720p, punkt główny w środku obrazu, piksele kwadratowe,
ogniskowa z ortogonalności r1 ⟂ r2), mediana na klip. Rozkład udał się dla 92,8% homografii. Kamera stoi przy
x 52,6 m (linia środkowa), y 88,1 m (ok. 20 m za linią boczną), na wysokości 13,3 m (9,3–15,5 m na klip; gra 2
ok. 9,7 m, gra 3 ok. 15 m). Oczekiwane przesunięcie rzutu środka piłki leżącej na murawie (r = 0,11 m) wynosi
d · r / (H − r).

Składowa wzdłuż promienia, klatki z błędem ≤ 5 m, w kwartylach odległości prawdy od kamery:

| odległość od kamery | n (k = 0) | środek: mediana [q25; q75] | dół ramki: mediana [q25; q75] | k = 0,25: mediana | oczekiwane przesunięcie środka |
|---|---|---|---|---|---|
| wszystkie | 2 301 | +0,38 [−0,39; +0,98] | −0,26 [−0,94; +0,35] | +0,22 | 0,54 |
| 17–41 m | 610 | +0,14 [−0,67; +0,72] | −0,24 [−1,01; +0,35] | +0,06 | 0,32 |
| 41–58 m | 559 | +0,30 [−0,77; +0,86] | −0,27 [−1,29; +0,24] | +0,15 | 0,51 |
| 58–74 m | 575 | +0,64 [−0,02; +1,16] | −0,11 [−0,75; +0,48] | +0,44 | 0,68 |
| 74–110 m | 557 | +0,47 [+0,04; +1,16] | −0,37 [−0,81; +0,29] | +0,25 | 0,81 |

- Przesunięcie systematyczne istnieje i ma oczekiwany kierunek. Rzut środka ląduje dalej od kamery o medianę
  +0,38 m. Rośnie ono z odległością (+0,14 m blisko, +0,47–0,64 m daleko), co zgadza się co do rzędu wielkości z
  geometrią (0,32–0,81 m). Dół ramki przerzuca je na drugą stronę (−0,26 m). Zero wypada w okolicy k ≈ 0,5.
- Składowa poprzeczna nie ma przesunięcia (mediana −0,02 m [−0,31; +0,24]) i we wszystkich wariantach jest taka
  sama.
- Rozrzut wzdłuż promienia (rozstęp kwartylowy ok. 1,4 m) jest kilka razy większy od samego przesunięcia, więc
  przesunięcie nie wyjaśnia błędów > 5 m. Te biorą się z wyboru złej piłki, z interpolacji i z homografii, a nie z
  punktu ramki.

## Rekomendacja

Dla ticketów 12–16 punktem rzutowania zostaje **środek ramki** (bez zmian). Opisowo: przesunięcie środka o ok.
0,4 m dalej od kamery jest realne i k ≈ 0,5–0,75 zmniejsza medianę błędu o ok. 0,1 m. Mogłoby to mieć znaczenie
tam, gdzie liczy się dokładność w metrach (np. cechy xG), ale według zarejestrowanej miary ≤ 5 m nic nie zyskuje.

## Zgodność z zasadami

Czytane były tylko gry 2+3. Klipy gry 5 dostały brak homografii przed `placement_rows`, a ich wiersze zostały
odrzucone przed jakąkolwiek liczbą, także w diagnostyce. Gry 4, gry 5 i GSR test nie czytano. GPU nie było używane.
Nie ma zmian w `src/`.
