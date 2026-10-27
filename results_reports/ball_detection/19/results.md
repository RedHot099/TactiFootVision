# ball-detection 19: lokalizacja piłki w xG wobec stóp strzelca (ramiona CPU) i wykonalność ramienia T

2026-10-27. Rejestracja: komentarz ticketu 19 z 13:55, zapisany przed odczytem log-loss ramion C i S2. Kod:
`scripts/ball_xg_compare.py` (`read` w środowisku xG, `clips` w środowisku repo), testy
`tests/test_ball_xg_compare.py`. Wyniki leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/19/`
(`xg/read.json`, `xg/breakdown.csv`, `xg/probabilities.csv`, `xg/relation.parquet`, `clips/clips.parquet`,
`clips/clips.json`, `clips/cuts_gallery/`). Tylko CPU, GPU nie było używane.

## Populacja i procedura

Populacja to 7 062 strzały bez karnych z 328 meczów rozwojowych (902 gole), czyli populacja xg-stack z tymi samymi
foldami. Procedura C6 jest zamrożona (`research/xgs_stack_protocol_c6.json`, kandydat `bc`). Każde ramię przechodzi
cały odczyt `stack.analysis.analyse` od nowa, poza foldem. Etykiety (gol) czytane są w pieczęci, bez zmian. Braki
obsługuje sama procedura (średnia z treningu i wskaźnik braku), żaden strzał nie jest usuwany. StatsBomb nie jest
czytany. Model główny to ramię stosu **ALL**. ΔLL liczę parami wobec F: bootstrap po meczach, 2 000 losowań, seed 0.

| ramię | wejście tabeli E3 stosu |
|---|---|
| R | reguła sprzed `feet_t0`: piłka z −400, potem −800, potem 0 ms, miejsce strzału = rzut piłki (`xgd_full4/E3`). Wiersz odniesienia |
| F | `feet_t0`, zamrożona (`xgd_full4_feet_t0/E3`) |
| C | F plus 11 kolumn relacji piłka–stopy z klatki t0 reguły F, dodanych liniowo do ALL: odległość stopy–piłka (m), wysokość piłki względem ramki strzelca, przesunięcie boczne, odstęp od ramki, pewność piłki, źródło piłki (własna / −400 / −800 / brak) i 2 wskaźniki braku. Żadna kolumna nie wypadła przez zerową wariancję |
| S2 | F ze strzelcem wybranym piłką drugiego stopnia 13 (kandydaci z 16). Kolumny źródła piłki zostają z F, więc różni się tylko strzelec i to, co z niego wynika |
| S2_reguła | opisowo: `feet_t0` na klatkach z piłką S2, z flagami źródła liczonymi z S2 (jak w 16) |

## Kontrole (wszystkie przeszły przed odczytem C i S2)

- Odczyt zamrożonego E3 (`xgd_full3`) odtwarza `results/xgs_stack/real/summary.json` dokładnie.
- R odtwarza `results/xgs_stack_dd2c/summary.json` dokładnie: ALL − R_stos **−0,0056**.
- F odtwarza `results/xgs_stack_feet_t0_flags/summary.json` dokładnie: ALL − R_stos **−0,0116**, czyli wynik dev z
  efd7908. Replay reguły na klatkach daje `xgd_full4_feet_t0/E3/features.parquet` bez różnicy
  (`stack_location.e3_table`, tolerancja 0). Projekt F różni się od protokołu tylko wypadnięciem `shot_masked@0`
  (zerowa wariancja), tak jak w `feet_rerun.py`.

R_stos to samo z_g_combo (G-combo), a nie ramię lokalizacji R.

## Wynik (ramię stosu ALL, ΔLL wobec F)

| ramię | LL | AUC | Brier | kalibracja: nachylenie / wyraz wolny | ΔLL vs F [95%] | p poprawy | p TOST ±0,002 | werdykt D3 (Holm na C, S2, T; T z ticketu 20) |
|---|---|---|---|---|---|---|---|---|
| F | 0,3314 | 0,760 | 0,0983 | 0,93 / −0,12 | – | | | |
| R (odniesienie) | 0,3375 | 0,747 | 0,1004 | 0,98 / −0,04 | +0,0061 [+0,0029; +0,0090] | | | bez werdyktu |
| **C** | 0,3330 | 0,756 | 0,0988 | 0,95 / −0,08 | **+0,0016 [+0,0001; +0,0029]** | 0,98 | 0,25 | **nierozstrzygnięte** |
| **S2** | 0,3324 | 0,762 | 0,0987 | 0,94 / −0,10 | **+0,0010 [−0,0014; +0,0035]** | 0,77 | 0,23 | **nierozstrzygnięte** |
| **T** (ticket 20, piłka śledzona przyczynowo) | 0,3314 | 0,761 | 0,0984 | 0,99 / −0,02 | **−0,0000 [−0,0026; +0,0025]** | 0,49 | 0,062 | **nierozstrzygnięte** |
| S2_reguła (opisowo) | 0,3320 | 0,762 | 0,0986 | 0,95 / −0,08 | +0,0006 [−0,0019; +0,0031] | | | poza rodziną |

Wiersz T dopisany 2026-10-27 przez ticket 20 (`results_reports/ball_detection/20/results.md`). Holm liczony wspólnie
dla C, S2 i T z prawdziwym p_T: żadne ramię nie wnosi wartości ani nie jest równoważne; wszystkie trzy są
nierozstrzygnięte. T: przedziały na poziomach Holma [−0,0031; +0,0030] / [−0,0029; +0,0028] / [−0,0026; +0,0025].

Przedziały na poziomach Holma (k = 3 / 2 / 1, poziom dwustronny 1 − 0,05/k): C [−0,0003; +0,0031] /
[−0,0001; +0,0031] / [+0,0001; +0,0029]; S2 [−0,0020; +0,0041] / [−0,0017; +0,0039] / [−0,0014; +0,0035].

- **Żadne ramię piłki nie wnosi wartości.** Poprawy nie ma przy żadnej wartości p_T: p wynosi 0,98 i 0,77, a Holm
  odrzuca najwyżej przy p ≤ 0,025. Równoważności w ±0,002 też nie ma: p TOST 0,25 i 0,23 przekracza 0,05 nawet bez
  korekty. Werdykt „nierozstrzygnięte” jest więc ostateczny dla C i S2 niezależnie od wyniku T z ticketu 20.
- **C jest raczej gorsze od F.** Punkt +0,0016, a przedział 95% bez korekty leży nad zerem. Po korekcie Holma
  (98,3%) obejmuje zero. Dodatkowe kolumny relacji nie pomagają i kosztują trochę wariancji. Wewnętrzne CV wybiera
  dla C silniejszą karę (C = 0,01 w 3 z 5 foldów, w F 0,1–0,3).
- **S2 zmienia miejsce strzału, ale nie log-loss.** Miejsce strzału przy t0 zmienia się w 16,4% strzałów; odległość
  do bramki zmienia się o więcej niż 5 m w 5,3% strzałów (mediana zmiany 2,9 m, gdy jest). W mniejszych ramionach
  stosu S2 leży przy F: G −0,0003 [−0,0018; +0,0013], GS −0,0001, Q −0,0000 (opisowo). Plus w ALL idzie razem ze
  zmianą wyboru C w foldach (S2: 0,01 / 1,0 / 0,01 / 0,01 / 0,03). Wnioskuję, że to szum procedury, a nie sygnał.
- **F jest faworyzowane:** reguła F była dobierana na tych samych 328 meczach (efd7908). Porównanie jest
  przechylone na korzyść F, a mimo to ramiona piłki nie wychodzą poza ±0,0035.
- R − F = +0,0061 [+0,0029; +0,0090]. To ta sama różnica co między dd2c a `feet_t0` w efd7908 (−0,0056 → −0,0116),
  teraz policzona parami.

### Rozbicia (opisowe, ALL, ΔLL vs F [95%])

| podział | poziom | strzały (gole) | C | S2 |
|---|---|---|---|---|
| odległość F `dist@0` | < 11 m | 1 600 (412) | +0,0027 [−0,0012; +0,0065] | +0,0041 [−0,0037; +0,0118] |
| | 11–18 m | 2 091 (318) | +0,0004 [−0,0026; +0,0032] | +0,0024 [−0,0027; +0,0081] |
| | 18–25 m | 1 988 (123) | +0,0030 [+0,0011; +0,0048] | −0,0018 [−0,0050; +0,0013] |
| | ≥ 25 m | 1 318 (45) | −0,0002 [−0,0032; +0,0026] | −0,0005 [−0,0045; +0,0037] |
| | brak | 65 (4) | −0,0014 [−0,0060; +0,0039] | −0,0012 [−0,0071; +0,0065] |
| piłka wysoko (proxy głowy, `bf_height` ≥ 0,6) | wysoko | 1 539 (249) | +0,0014 [−0,0028; +0,0056] | +0,0001 [−0,0059; +0,0061] |
| | nisko | 4 958 (568) | +0,0017 [−0,0000; +0,0032] | +0,0007 [−0,0016; +0,0031] |
| | brak piłki | 565 (85) | +0,0010 [−0,0039; +0,0055] | +0,0063 [−0,0105; +0,0255] |
| źródło piłki F przy t0 | własna | 4 331 (531) | +0,0013 [−0,0005; +0,0029] | +0,0008 [−0,0018; +0,0037] |
| | −400 ms | 1 862 (240) | +0,0029 [+0,0002; +0,0057] | +0,0002 [−0,0042; +0,0045] |
| | −800 ms | 304 (46) | −0,0027 [−0,0130; +0,0080] | −0,0015 [−0,0191; +0,0137] |

W żadnej warstwie ramię piłki nie ma przedziału poniżej zera. Głowy: na 328 meczach nie ma etykiety części ciała,
bo StatsBomb obejmuje tylko 172 mecze, więc rozbicie idzie po proxy wysokości piłki w obrazie. Według tego proxy
piłka jest wysoko w 23,7% strzałów z piłką.

## StatsBomb (opisowo, tylko 172)

Błąd miejsca strzału względem StatsBomb pochodzi z efd7908 (sparowane strzały 172 meczów): mediana R (dd2c) 2,62 m,
F 1,16 m; ≥ 5 m: 32% i 15%; głowy: 13,04 m i 2,40 m. C ma miejsce strzału F. **S2 na 172 nie jest policzone:**
wymaga detekcji dd2c i drugiego stopnia na klatkach 172 meczów, a to praca na GPU (około 11 500 klatek, około
0,05 GPU-h). Proponuję zrobić to w 20. Cech StatsBomb nie użyto nigdzie.

## Wykonalność ramienia T (ticket 20)

Skrypt: `clips`, 7 062 klipy strzałów rozwojowych, tylko CPU, bez xG.

- **Klipy:** 8 s, 720p, od t0 − 4 s do t0 + 4 s (`collect_soccernet.cut_clip`). Każdy strzał ma pełne okno
  [−2 s; 0], czyli 51 klatek. Żaden klip nie zaczyna się później niż t0 − 4 s. Klipy skrócone na końcu (112 z
  < 190 klatek) nie przeszkadzają wariantowi przyczynowemu.
- **fps:** 6 947 klipów (98,4%) ma 25 fps. 115 klipów (1,6%, 11 meczów, głównie Serie A 2016/17) ma nieregularny
  fps, ok. 16–19, czasem 24,9. Indeks klatki 25 fps jest tam błędny. W 20 klatki trzeba wybierać po znacznikach czasu
  albo traktować te strzały jako brak T i brać wartość z F.
- **Położenie t0:** na losowej próbie 200 strzałów (seed 0) klatka 100 klipu pasuje do `+0.jpg`, a klatka 90 do
  `-400.jpg`, w 196 przypadkach dokładnie (mediana |Δ| szarości 1,5 na 255). Pozostałe 4 to klipy z nieregularnym fps.
- **Cięcia (ContentDetector HSV, próg 27):** cięcie w [−2 s; 0] ma 7,2% strzałów, w ostatniej sekundzie 2,3%, w
  ostatnich 0,4 s 0,5%. Obejrzałem 3 z 12 klatek galerii (`clips/cuts_gallery/`): to prawdziwe cięcia, zbliżenie lub
  inny kąt → kamera główna. Tor przed cięciem nie nadaje się do śledzenia, więc T musi zaczynać tor od ostatniego
  cięcia. Powtórek przed t0 w oknie nie widać. Kamera główna jest przy t0, homografia działa w 98% klatek t0.
- **Koszt GPU (szacunek z pomiarów 09/13/16 i `xgd_full4`; bez nowego pomiaru):**

| składnik (7 062 strzały, przyczynowo [−2 s; 0]) | klatki | ms/klatkę | GPU-h |
|---|---|---|---|
| dd2c ≥ 0,01 przy 1280 na 720p | 360 162 | 12,9 (16) – 20,9 (09, z I/O) | 1,3 – 2,1 |
| kotwice homografii co 5 klatek (punkty boiska; t0 i −400 ms są w cache) | 63 558 | 109 (16) – 143 (`xgd_full4`, pełny analizator) | 1,9 – 2,5 |
| razem, kotwice co 5 | | | **3,2 – 4,6** |
| razem, kotwice co 10 (REACH 10 z 04) | 28 248 kotwic | | 2,2 – 3,2 |
| opcje: wariant dwustronny ±2 s (offline, opisowo) | +50 klatek/strzał | | +3,2 – 4,6 |
| opcje: T na 172 dla błędu względem StatsBomb (3 819 strzałów) | | | +1,7 – 2,5 |
| opcje: S2 na 172 (3 klatki/strzał) | 11 457 | ok. 15 | 0,05 |

  Ruch kamery i śledzenie liczą się na CPU (ok. 10 ms/klatkę). Analizator xG mierzy też ludzi i drużyny, więc
  kotwice są szacowane z góry.
- **Miejsce:** na Storage jest 241 GB wolnego (nie 768 GB, jak w pamięci). T nie potrzebuje zrzutu klatek, bo
  klatki dekoduje się z klipów na `/home` (18 GB). Kandydaci i homografie zajmą ok. 100 MB.

**Werdykt: T jest wykonalne.** Główne T przyczynowe na 328 meczach kosztuje 3,2–4,6 GPU-h (kotwice co 5 klatek), czyli
mieści się w limicie 8 GPU-h (D6). Do tego dochodzi S2 na 172 (0,05 GPU-h). Na wariant dwustronny albo T na 172
starczy budżetu tylko przy kotwicach co 10 klatek. Warunki dla 20:

1. klatki wybierać po znacznikach czasu (1,6% klipów nie ma 25 fps);
2. tor zaczynać od ostatniego cięcia przed t0;
3. bez piłki w torze zostawiać wartość F, a braki obsługiwać tak samo jak w procedurze.

C i S2 dały „nierozstrzygnięte” z punktem powyżej zera. Uruchomienie 20 zależy od decyzji orkiestratora i
właściciela; D6 nie stawia warunku sygnału z C i S2.

## Odstępstwa i uwagi

1. W wierszu R spec i ticket mówią „rzut piłki t0 z zapasem −400/−800”. Zamrożona reguła sprzed `feet_t0` (E3)
   bierze jednak najpierw piłkę z −400 ms. Użyłem reguły E3 i odnotowałem to w rejestracji.
2. W rejestracji było „próbka 200 do dopasowania t0”. Dopasowanie poszło na tej próbie, a cięcia i fps liczyłem na
   wszystkich 7 062 klipach.
3. Smoke `clips` na 20 strzałach (`/tmp`) przed pełnym przebiegiem; odczyt xG puściłem raz, bez smoke.
4. W teście Holma poprawiłem oczekiwaną wartość w moim własnym teście (błąd testu, nie kodu) przed jakimkolwiek
   odczytem.
5. Wiersze stosu R_stos i S są identyczne we wszystkich ramionach, bo nie czytają E3, więc ich ΔLL = 0.

## Koszt

CPU: odczyt xG ok. 4,5 min (5 pełnych odczytów stosu), klipy ok. 6 min na 6 procesach. GPU: 0 (`agent-state gpu`
wolne przez cały ticket).
