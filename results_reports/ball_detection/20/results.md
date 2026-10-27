# ball-detection 20: xG z piłką śledzoną na klipie strzału (ramię T)

2026-10-27. Rejestracja: komentarz ticketu 20 z 14:23, zapisany przed odczytem log-loss T. Kod:
`scripts/ball_xg_tracked.py` (`dump`, `track`, `read`, `detect172`, `read172`; środowisko xG `run.sh`), testy
`tests/test_ball_xg_tracked.py`. Wyniki leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/20/`
(`dump/`, `tracks.parquet`, `track.json`, `xg/read.json`, `xg/breakdown.csv`, `xg/probabilities.csv`,
`172/read172.json`, `172/read172.md`).

## Ramię T

Populacja, foldy, procedura C6 i reguła D3 są takie same jak w 19: 7 062 strzały bez karnych z 328 meczów
rozwojowych, ramię stosu ALL, ΔLL wobec F, bootstrap par po meczach (2 000 losowań, seed 0).

Okno to klip strzału od t0 − 2 s do t0 włącznie (k = −50…0, 25 fps). Żadna klatka po t0 nie jest czytana. Okno
zaczyna się od pierwszej klatki po ostatnim cięciu przed t0 (cięcia z 19). Detektor to dd2c 1280 z fazy 1
(imgsz 1280, NMS IoU 0,7, kandydaci ≥ 0,01). Klatki k = 0 / −10 / −20 są podmienione na jpg `+0` / `-400` / `-800`,
które analizował cache xG. Kotwice homografii to homografie cache xG (k = 0 / −10 / −20) i model punktów `sn960`
(ustawienia `extract3.COMMON`) na pozostałych klatkach co 5. Ruch kamery i `propagate` działają jak w 04. Potem
idzie filtr 45 px i `track_ball` z `ball/04/tune/chosen.json`.

W klatkach T piłka wierszy 0 / −400 / −800 ms jest zamieniona na piłkę toru, jeśli tor ją ma (zmierzoną albo
interpolowaną). Wiersz bez piłki w torze zostaje z piłką F. Dalej działa konstrukcja F: `FEET_T0`, `e3_table` z
flagami źródła liczonymi z klatek T i pełny odczyt stosu w foldach.

- **Pokrycie:** T ma 6 947 strzałów. 115 klipów z nieregularnym fps nie ma T i zostaje z klatkami F (żaden strzał
  nie jest usuwany). Kontrola położenia t0 (klatki 100 / 90 wobec `+0.jpg` / `-400.jpg`) przeszła we wszystkich
  6 947 klipach. Mediana okna to 51 klatek, mediana kotwic 11, przerwy ruchu kamery w 0,03% par klatek.
- **Piłka T przy t0:** tor ma zmierzoną piłkę w 4 755 strzałach. W 2 192 strzałach tor kończy się przed t0, więc
  zostaje piłka F. Przyczynowy tor nie interpoluje na końcu okna, dlatego przy t0 nie ma piłek interpolowanych.
  W 692 strzałach T daje piłkę t0 tam, gdzie F jej nie miało. Wśród strzałów, w których obie mają piłkę t0, T
  wybiera inną piłkę niż F (środek dalej niż 3 px) w 7,6% przypadków. Przy −400 ms: 5 494 zmierzone, 1 086
  interpolowanych, 482 z F.
- **Miejsce strzału:** T zmienia je w 12,0% strzałów. Odległość do bramki zmienia się o więcej niż 5 m w 3,9%
  strzałów (mediana zmiany 3,1 m, gdy jest).

### Kontrole (wszystkie przeszły)

- F odtwarza `results/xgs_stack_feet_t0_flags/summary.json`, a jego prawdopodobieństwa ALL są bit w bit takie jak
  w 19. ΔLL, przedziały i p dla C i S2 są równe `ball/19/xg/read.json`.
- Homografia t0 dopasowana tym skryptem z moich punktów (co 50. strzał, 137 strzałów) wobec homografii cache: mediana
  i p90 odległości na siatce 0,00 m, maksimum 1,40 m (jeden przypadek). Top-1 dd2c ≥ 0,15 przy t0 z mojego zrzutu
  (NMS 0,7) to ta sama piłka co w cache (NMS 0,5) w 100% strzałów.
- Projekt T traci tylko `shot_masked@0` (zerowa wariancja), tak jak F.

### Wynik (ramię stosu ALL, ΔLL wobec F, wspólny Holm dla C, S2 i T)

| ramię | LL | AUC | Brier | kalibracja: nachylenie / wyraz wolny | ΔLL vs F [95%] | p poprawy | p TOST ±0,002 | werdykt D3 |
|---|---|---|---|---|---|---|---|---|
| F | 0,3314 | 0,760 | 0,0983 | 0,93 / −0,12 | – | | | |
| C (19) | 0,3330 | 0,756 | 0,0988 | 0,95 / −0,08 | +0,0015 [+0,0001; +0,0029] | 0,98 | 0,25 | nierozstrzygnięte |
| S2 (19) | 0,3324 | 0,762 | 0,0987 | 0,94 / −0,10 | +0,0010 [−0,0014; +0,0035] | 0,77 | 0,23 | nierozstrzygnięte |
| **T** | 0,3314 | 0,761 | 0,0984 | 0,99 / −0,02 | **−0,0000 [−0,0026; +0,0025]** | 0,49 | 0,062 | **nierozstrzygnięte** |
| T_flagiF (opisowo) | 0,3319 | 0,760 | 0,0986 | 0,98 / −0,03 | +0,0005 [−0,0019; +0,0029] | | | poza rodziną |
| R (odniesienie) | 0,3375 | 0,747 | 0,1004 | 0,98 / −0,04 | +0,0061 [+0,0029; +0,0090] | | | bez werdyktu |

T na poziomach Holma (k = 3 / 2 / 1): [−0,0031; +0,0030] / [−0,0029; +0,0028] / [−0,0026; +0,0025]; przedział 90%
[−0,0022; +0,0021].

- **Piłka śledzona nie poprawia xG wobec stóp.** Punkt jest równy zeru (−0,00002). p poprawy wynosi 0,49, więc
  poprawy nie ma przy żadnej korekcie.
- **Równoważności też nie ma.** Przedział 90% T wychodzi o 0,0002 poza ±0,002 z obu stron (p TOST 0,062). Holm
  wymaga przy najmniejszym p progu 0,05/3. Werdykt to „nierozstrzygnięte”, a T jest najbliżej równoważności z trzech
  ramion.
- **Tabela rodziny (D3):** C, S2 i T są nierozstrzygnięte. Żadne ramię piłki nie wnosi wartości. Tabela 19 dostała
  wiersz T.
- Kalibracja T jest lepsza niż F (nachylenie 0,99 wobec 0,93), ale log-loss i Brier stoją w miejscu. Wewnętrzne CV
  wybiera dla T we wszystkich 5 foldach C = 0,01, czyli silniejszą karę niż dla F (0,1–0,3). Mniejsze ramiona stosu
  też leżą przy F: G +0,0002 [−0,0015; +0,0020], GS +0,0005, Q +0,0004.
- F jest faworyzowane, bo reguła F była dobierana na tych 328 meczach (jak w 19).

### Rozbicia (opisowe, ALL, ΔLL T − F [95%])

| podział | poziom | strzały (gole) | T | T_flagiF |
|---|---|---|---|---|
| odległość F | < 11 m | 1 600 (412) | +0,0011 [−0,0078; +0,0100] | +0,0030 [−0,0049; +0,0111] |
| | 11–18 m | 2 091 (318) | −0,0017 [−0,0064; +0,0025] | −0,0015 [−0,0060; +0,0027] |
| | 18–25 m | 1 988 (123) | +0,0004 [−0,0023; +0,0029] | +0,0007 [−0,0017; +0,0031] |
| | ≥ 25 m | 1 318 (45) | +0,0008 [−0,0022; +0,0042] | +0,0004 [−0,0025; +0,0035] |
| piłka wysoko (proxy głowy) | wysoko | 1 539 (249) | −0,0029 [−0,0100; +0,0035] | −0,0024 [−0,0086; +0,0035] |
| | nisko | 4 958 (568) | +0,0011 [−0,0011; +0,0033] | +0,0014 [−0,0007; +0,0035] |
| | brak piłki F | 565 (85) | −0,0024 [−0,0172; +0,0140] | −0,0000 [−0,0151; +0,0168] |
| piłka T przy t0 | zmierzona przez tor | 4 755 (575) | −0,0001 [−0,0028; +0,0024] | +0,0003 [−0,0022; +0,0025] |
| | tor bez piłki (piłka F) | 2 192 (311) | −0,0001 [−0,0055; +0,0058] | +0,0007 [−0,0046; +0,0061] |
| | brak T (nieregularny fps) | 115 (16) | +0,0031 [−0,0160; +0,0178] | +0,0060 [−0,0115; +0,0190] |

Żadna warstwa nie ma przedziału w całości poniżej ani powyżej zera. Strzały bez T mają te same klatki co F, a mimo
to ΔLL wynosi tam +0,0031, bo zmienia się cały model. To pokazuje rząd szumu procedury w małych warstwach.

## Opisowo (D4): piłka P1 i S2 na klatkach strzałów 172 meczów wobec StatsBomb

Detekcja 16 bez zmian (dd2c ≥ 0,01 i drugi stopień 13 na jpg 0 / −400 / −800 ms oraz 51 potrzebnych klatek
−800 ms) objęła 3 005 strzałów bez karnych ze 135 meczów ewaluacyjnych. To w nich leżą pary StatsBomb; 37 meczów
„gate” z puli 172 nie było czytanych. Odczyt idzie na 2 721 sparowanych strzałach z 17. StatsBomb służy tylko jako
cel, nic nie jest dobierane. Kontrole: wiersz X0 jest równy wierszowi 17, a miejsce X0 jest równe cache `feet_t0`
(oba dokładnie). Bootstrap po meczach, wartości wśród wszystkich strzałów:

| wiersz | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m (z pozycją) | p90 m | bez pozycji |
|---|---|---|---|---|---|---|---|
| X0 piłka t0 (17) | 4,5 [3,7; 5,4] | 16,0 [14,2; 17,7] | 34,6 [32,4; 36,9] | 49,8 [47,6; 52,1] | 1,78 [1,69; 1,91] | 10,48 | 37,2 [34,5; 39,7] |
| P1 piłka t0 | 4,5 [3,7; 5,4] | 16,0 [14,2; 17,7] | 34,3 [32,1; 36,7] | 49,1 [46,8; 51,4] | 1,78 [1,68; 1,90] | 10,50 | 37,9 [35,1; 40,5] |
| S2 piłka t0 | 5,4 [4,5; 6,4] | 18,7 [16,8; 20,6] | 41,5 [39,1; 43,9] | 62,1 [60,1; 64,1] | 1,99 [1,86; 2,12] | 11,63 | 17,3 [15,6; 19,1] |
| miejsce `feet_t0` przy piłce X0 (F) | 22,7 [20,4; 25,1] | 45,0 [42,2; 48,0] | 66,9 [64,8; 69,1] | 84,8 [83,4; 86,2] | 1,16 [1,05; 1,25] | 6,79 | 0,5 |
| miejsce `feet_t0` przy piłce P1 | 22,6 [20,3; 25,0] | 44,9 [42,0; 47,9] | 66,9 [64,7; 69,0] | 84,7 [83,3; 86,1] | 1,16 [1,06; 1,25] | 6,79 | 0,5 |
| miejsce `feet_t0` przy piłce S2 | 23,0 [20,6; 25,5] | 46,2 [43,2; 49,2] | 69,0 [66,8; 71,1] | 85,4 [83,9; 86,9] | 1,09 [1,02; 1,20] | 6,82 | 0,5 |

- **Filtr P1 przy t0 prawie nic nie zmienia.** Usuwa część piłek (bez pozycji 37,9% wobec 37,2%), a miejsce
  strzału zmienia się w 0,3% strzałów.
- **S2 daje pozycję piłki dużo częściej.** Bez pozycji jest 17,3% strzałów wobec 37,9%, a ≤ 5 m wśród wszystkich
  62,1% wobec 49,1%. Dokładają się jednak gorsze trafienia: mediana błędu wśród strzałów z pozycją rośnie z 1,78 do
  1,99 m, a odsetek ≤ 5 m wśród nich spada z 79,0% do 75,2%.
- **Miejsce strzału przy piłce S2** zmienia się w 16,5% strzałów. Mediana błędu wobec StatsBomb wynosi 1,09 m
  [1,02; 1,20] wobec 1,16 m przy F, a odsetek ≤ 2 m 69,0% wobec 66,9%. Przedziały mocno się nakładają, różnicy
  parami nie testowałem. To odczyt opisowy na zbiorze zużytym.

## Wariant dwustronny ±2 s (rekonstrukcja offline, opisowo, poza rodziną)

Ten sam tor, ale okno sięga też 2 s za t0 (k = −50…+50), do ostatniej klatki przed pierwszym cięciem po t0 (ten sam
detektor cięć co w 19). Zrzut klatek po t0 i kotwice co 5 klatek liczyłem tak samo jak w T, a odczyt
(`read --two-sided`, `xg_two_sided/read.json`) tym samym kodem. Kontrole F, C i S2 przeszły tak samo. Wariant
korzysta z klatek po strzale, więc nie nadaje się do przewidywania przed strzałem. Werdyktu D3 tu nie ma.

- Piłka t0: 4 845 zmierzonych, 1 938 interpolowanych (tor ma piłkę po t0), 164 strzały z piłką F. Nowa piłka t0
  tam, gdzie F jej nie miało: 2 489. Miejsce strzału zmienia się w 23,1% strzałów (> 5 m w odległości: 5,9%).
- ΔLL vs F (ALL): **−0,0010 [−0,0040; +0,0019]** (p jednostronne 0,26). T_flagiF −0,0004 [−0,0033; +0,0023].
  LL 0,3304, AUC 0,764, Brier 0,0981, kalibracja 0,99 / −0,02.
- Punkt jest niżej niż w T przyczynowym, ale przedział obejmuje zero z dużym zapasem. Nawet piłka zrekonstruowana
  z klatkami po strzale nie daje wykrywalnej poprawy wobec stóp.

## Odstępstwa i uwagi

1. Klipy z nieregularnym fps (115) nie mają T, zamiast wyboru klatek po czasie. Taki był wybór rejestracji.
2. Pierwsze uruchomienie zrzutu (6 min GPU) przerwałem: `Executor.map` trzymał w pamięci całe okna 100 strzałów
   (15 GB na proces), a NMS ultralytics bez załadowanego torchvision kosztował 22 ms na klatkę. Częściowy zrzut
   skasowałem i zrobiłem od nowa z ograniczonym wyprzedzaniem i NMS torchvision (te same ramki, 17,5 wobec 44 ms).
3. Pierwszy odczyt `read` zatrzymał się na kontroli F: `read_csv` bez `float_precision="round_trip"` zmieniał
   ostatni bit prawdopodobieństw z 19. Kontrola zatrzymała się, zanim dopasowany został jakikolwiek wynik T.
   Poprawiłem wczytywanie i puściłem odczyt raz.
4. Wiersze 172 obejmują 135 meczów ewaluacyjnych (pary StatsBomb), nie wszystkie 172.
5. Ludzie do maski linii w kotwicach pochodzą z mojego zrzutu (NMS 0,7, ≥ 0,3), a nie z analizatora xG. Kontrola
   homografii t0 pokazuje, że wynik jest ten sam.

## Koszt

GPU (pod blokadą): zrzut T 1,56 h (dwa procesy, 360 tys. klatek dd2c przy 26,6 ms, 61 tys. klatek punktów),
detekcja 172 0,04 h, przerwany start, test szybkości i profil 0,16 h. Razem ok. 1,8 GPU-h do odczytu T. Wariant dwustronny: zrzut klatek po t0 1,61 GPU-h. **Razem ok. 3,4 GPU-h** (limit 8).
CPU: śledzenie T ok. 8,2 h CPU (11 procesów, ok. 45 min), dwustronne ok. 16 h CPU (14 procesów, ok. 70 min), odczyty
xG po ok. 2 min.
