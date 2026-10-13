# Audyt 3: pozycje z detektora z nowym modelem punktów boiska wobec StatsBomb (pitch-keypoints, ticket 04)

Jakub Eichner, 2026-10-13 (czas UTC). Gałąź `research/xg-detector`, faza pitch-keypoints.

Data: StatsBomb. Pozycje odniesienia (miejsce strzału, bramkarz, obrońcy) pochodzą z freeze frames w otwartych danych StatsBomb.

Audyt 3 powtarza audyty 1 i 2 badania xg-detector (`results_reports/xg_detector_2026-10/AUDIT.md`) z nowym modelem punktów kluczowych boiska z ticketu 03 tej fazy. Właściciel zniósł dla niego zakaz trzeciego audytu (Q26). Reguła wyboru konfiguracji i bramka są te same co w audytach 1–2 i są zamrożone. StatsBomb służy tylko do pomiaru: żaden model nie dostaje tych danych, a po audycie niczego nie dostrojono.

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `claims.toml` mówi, z którego pliku wyników pochodzi liczba. Skrypt `scripts/check_xgd_audit.py` odczytuje każdą liczbę ze źródła i kończy się błędem przy niezgodności.

## 1. Wynik: bramka STOP<!--n:a3.gate-->

Wybrana jest konfiguracja E3<!--n:a3.chosen-->. Wszystkie trzy konfiguracje mają pokrycie lokalizacji strzału co najmniej 70%<!--n:a3.cov_min-->, więc decyduje najniższa mediana błędu: E1 9.24 m<!--n:a3.e1.med2-->, E2 5.22 m<!--n:a3.e2.med2-->, E3 5.18 m<!--n:a3.e3.med2-->.

| Warunek | Próg | E3<!--n:a3.chosen--> | Spełniony |
|---|---|---|---|
| pokrycie lokalizacji strzału | ≥ 70%<!--n:a3.cov_min--> | 99.5%<!--n:a3.g.cov--> | tak |
| mediana błędu lokalizacji | ≤ 3 m<!--n:a3.med_max--> | 5.2 m<!--n:a3.g.med--> | nie |
| informacja o bramkarzu (wykryty albo zastępczy) | ≥ 50%<!--n:a3.gk_min--> | 98.9%<!--n:a3.g.gk--> | tak |

W porównaniu z wcześniejszymi audytami (konfiguracje wybrane przez tę samą regułę):

| Audyt | konfiguracja | pokrycie lokalizacji | mediana błędu lokalizacji | informacja o bramkarzu |
|---|---|---|---|---|
| 1 (ticket 02) | C4 | 71.8% [67.3; 76.2]<!--n:a1.c4.cov--> | 12.1 m [11.4; 12.8]<!--n:a1.c4.med--> | 60.2%<!--n:a1.c4.gk.cov1--> |
| 2 (ticket 09) | D3 | 77.4% [74.3; 80.5]<!--n:a3.d3.cov--> | 10.5 m [9.9; 11.1]<!--n:a3.d3.med--> | 66.5%<!--n:a3.d3.gk.cov1--> |
| 3 (ten audyt) | E3 | 99.5% [99.2; 99.8]<!--n:a3.e3.cov--> | 5.2 m [4.7; 5.8]<!--n:a3.e3.med--> | 98.9%<!--n:a3.e3.gk.cov1--> |

Mediana spadła o połowę, a pokrycie jest prawie pełne, ale próg 3 m<!--n:a3.med_max--> nie jest osiągnięty. Według rady (C3) podejście 2 pozostaje zamknięte, a A1 nie jest rozwijane od nowa.

## 2. Co się zmieniło od audytu 2

**Model punktów kluczowych.** YOLO11m-pose przy 960 px, uczony na klatkach SoccerNet GSR i Calibration-2023 (ticket 03 tej fazy, `models/pitch_yolo11m_pose_sn960.pt`). Na GSR valid, przy ustawieniach wybranych w tickecie 03, homografia jest na 93.0% [89.2; 95.9]<!--n:t3.avail--> klatek, a mediana błędu stóp graczy wynosi 0.62 m [0.56; 0.71]<!--n:t3.med-->. Model D3 miał tam medianę 1.66 m<!--n:t3.d3.med-->.

**Wejście w pakiecie.** Model uczył się na kwadratowym płótnie, czyli klatce dopełnionej szarymi pasami do kwadratu, tak jak Ultralytics przygotowuje obrazy treningowe. Pakiet dostał opcję `resize="square"` (commit `2e9469b`): `tv.load_model("yolo_pose", W, conf=0.1, imgsz=960, resize="square")`. Na 1 500<!--n:sq.frames--> klatkach strzałów z meczów treningowych daje dokładnie te same punkty co wejście, na którym model oceniono w tickecie 03: instancja na 1 500<!--n:sq.new--> klatkach, największa różnica położenia 0.0 px<!--n:sq.max-->.

**Homografia.** Próg punktów kluczowych 0.3<!--n:dc.kp-->, RANSAC 2 m<!--n:dc.ransac-->, doprecyzowanie liniami boiska z maską bez ramek osób z detektora. Detektor, drużyny i wybór bramkarza są takie jak w C4 i D3.

## 3. Konfiguracje E1–E3

Konfiguracje zadeklarowano przed jakimkolwiek odczytem danych StatsBomb w tym tickecie, w pliku `research/pitchkp_audit3_configs.json` (SHA-256 `f7f23eaede754326a24b6d177c1e83e1eb6158c2cb22f1e36b81794604f8a8b7`, commit `e1d41e0`). Skrypt audytu odmawia pracy, gdy plik ma inny skrót. Różnią się tylko źródłem lokalizacji strzału, czyli kolejnością klatek, z których brana jest piłka (rada C3, pytanie 3):

| Konfiguracja | kolejność klatek piłki | rola |
|---|---|---|
| E1 | t0, −400 ms, −800 ms | reguła D3 z audytu 2, zmieniony tylko model i jego ustawienia |
| E2 | −400 ms, −800 ms | stała kolejność bez piłki z t0 |
| E3 | −400 ms, −800 ms, t0 | jak E2, piłka z t0 na końcu |

Bez piłki w żadnej z tych klatek lokalizacją jest strzelec wybrany regułą flagi (najbliższy środka bramki), jak w audytach 1–2. Zawodnicy, drużyny i bramkarz zawsze pochodzą z klatki t0.

**Skąd pomysł z −400 ms.** Pomysł brania piłki z klatki −400 ms wziął się z diagnostyki audytu 2 na danych StatsBomb: w D3 piłka z −400 ms miała medianę błędu 6.0 m<!--n:dg2.m400.med-->, a piłka z t0 11.9 m<!--n:dg2.t0.med-->. Te liczby nie uzasadniają E2 i E3. Uzasadnienie pochodzi wyłącznie z pozycji piłki w SoccerNet GSR (`scripts/pk_ball_evidence.py`, wyniki w `results/pk_ball_evidence`, bez danych StatsBomb i bez GSR test):
- **Czas strzału według prawdy GSR.** W 21<!--n:ev.clips--> klipach GSR train i valid z akcją "Shots on target" albo "Shots off target", zakotwiczonych tak jak strzały indeksu, reguła kopnięcia na torze piłki z adnotacji znajduje kopnięcie w 11<!--n:ev.kicks--> klipach, w medianie 240 ms<!--n:ev.kick_ms--> przed t0. Piłka z −400 ms leży w medianie 2.0 m<!--n:ev.to400--> od miejsca kopnięcia, z −800 ms 3.4 m<!--n:ev.to800-->, a z t0 5.1 m<!--n:ev.to0-->. Między −400 ms a t0 piłka przesuwa się w medianie o 4.0 m<!--n:ev.move400-->, między −800 ms a −400 ms o 2.4 m<!--n:ev.move800-->. To ustala kolejność: −400 ms, −800 ms, t0.
- **Pipeline na klipach strzałów GSR valid.** Na 9 klipach reguła E1 jest w medianie 1.9 m<!--n:ev.e1--> od miejsca kopnięcia, E2 4.5 m<!--n:ev.e2--> (po 6 porównanych klipów). To za mało klipów, żeby uszeregować reguły. E1 zostaje regułą odniesienia.
- **Umieszczenie piłki na GSR valid.** Na 6 404<!--n:ev.ball.frames--> klatkach z piłką na boisku detektor umieszcza piłkę na 50.2% [42.1; 58.2]<!--n:ev.ball.placed--> klatek. Mediana błędu wynosi 1.42 m [1.22; 1.76]<!--n:ev.ball.med-->, a 25.1%<!--n:ev.ball.far--> umieszczonych piłek leży ponad 5 m od prawdziwej. To zwykle inny obiekt. Pozycję strzału ogranicza więc detektor piłki, nie homografia.

Reguła klatki kopnięcia z toru piłki (dozwolona przez radę) nie została zadeklarowana. Na klipach strzałów GSR valid jej mediana błędu wynosi 13.6 m<!--n:ev.kick--> (4 porównane klipy), bo detektor umieszcza piłkę tylko na około połowie klatek okna.

## 4. Populacja i metoda

Jak w audytach 1–2: 2 721<!--n:a3.shots--> sparowanych strzałów bez karnych w 135<!--n:a3.matches--> meczach, klatka t0, przedziały 95%<!--n:a3.ci--> z bootstrapu po meczach, 5 000<!--n:a3.n_boot--> losowań, te same co wcześniej. D3 z audytu 2 przechodzi ten audyt jako odniesienie i nie bierze udziału w wyborze. Jego 390<!--n:a3.ref.rows--> wierszy metryk jest identycznych z audytem 2 (różni się 0<!--n:a3.ref.diff-->). Cechy wszystkich konfiguracji odtworzone z zapisanych klatek są identyczne z zapisanymi.

## 5. Lokalizacja strzału

| Konfiguracja | pokrycie | mediana błędu | p90 | < 1 m | < 2 m | < 5 m | poza boiskiem |
|---|---|---|---|---|---|---|---|
| E1 | 99.5%<!--n:a3.e1.cov1--> | 9.2 m [8.8; 9.9]<!--n:a3.e1.med--> | 23.9 m<!--n:a3.e1.p90--> | 7.9%<!--n:a3.e1.lt1--> | 16.4%<!--n:a3.e1.lt2--> | 32.2%<!--n:a3.e1.lt5--> | 7.2%<!--n:a3.e1.off--> |
| E2 | 99.5%<!--n:a3.e2.cov1--> | 5.2 m [4.7; 5.9]<!--n:a3.e2.med--> | 21.1 m<!--n:a3.e2.p90--> | 10.4%<!--n:a3.e2.lt1--> | 26.5%<!--n:a3.e2.lt2--> | 49.0%<!--n:a3.e2.lt5--> | 5.3%<!--n:a3.e2.off--> |
| **E3** | 99.5%<!--n:a3.e3.cov1--> | 5.2 m [4.7; 5.8]<!--n:a3.e3.med--> | 21.3 m<!--n:a3.e3.p90--> | 10.4%<!--n:a3.e3.lt1--> | 26.6%<!--n:a3.e3.lt2--> | 49.3%<!--n:a3.e3.lt5--> | 5.6%<!--n:a3.e3.off--> |
| D3 (audyt 2) | 77.4%<!--n:a3.d3.cov1--> | 10.5 m [9.9; 11.1]<!--n:a3.d3.med--> | 27.5 m<!--n:a3.d3.p90--> | 4.6%<!--n:a3.d3.lt1--> | 10.7%<!--n:a3.d3.lt2--> | 26.2%<!--n:a3.d3.lt5--> | 8.1%<!--n:a3.d3.off--> |

Sam nowy model (E1 wobec D3) podnosi pokrycie do prawie pełnego, ale medianę obniża tylko o nieco ponad metr. Większość poprawy daje piłka z klatki −400 ms (E2, E3). Mimo to połowa strzałów ma błąd ponad 5 m.

**Według źródła lokalizacji i pewności instancji**, jak w audytach 1–2:

| Konfiguracja | piłka: n, mediana | flaga: n, mediana | piłka nie z t0: n, mediana | instancja ≥ 0.5<!--n:a3.kp-->: n, mediana |
|---|---|---|---|---|
| E1 | 2 544<!--n:a3.e1.ball.n-->, 9.3 m<!--n:a3.e1.ball.med--> | 164<!--n:a3.e1.flag.n-->, 8.9 m<!--n:a3.e1.flag.med--> | 705<!--n:a3.e1.fb.n-->, 4.3 m<!--n:a3.e1.fb.med--> | 2 707<!--n:a3.e1.kp.n-->, 9.2 m<!--n:a3.e1.kp.med--> |
| E2 | 2 464<!--n:a3.e2.ball.n-->, 4.7 m<!--n:a3.e2.ball.med--> | 244<!--n:a3.e2.flag.n-->, 9.0 m<!--n:a3.e2.flag.med--> | 2 464<!--n:a3.e2.fb.n-->, 4.7 m<!--n:a3.e2.fb.med--> | 2 707<!--n:a3.e2.kp.n-->, 5.2 m<!--n:a3.e2.kp.med--> |
| E3 | 2 544<!--n:a3.e3.ball.n-->, 4.9 m<!--n:a3.e3.ball.med--> | 164<!--n:a3.e3.flag.n-->, 8.9 m<!--n:a3.e3.flag.med--> | 2 464<!--n:a3.e3.fb.n-->, 4.7 m<!--n:a3.e3.fb.med--> | 2 707<!--n:a3.e3.kp.n-->, 5.2 m<!--n:a3.e3.kp.med--> |
| D3 | 1 976<!--n:a3.d3.ball.n-->, 10.5 m<!--n:a3.d3.ball.med--> | 131<!--n:a3.d3.flag.n-->, 11.8 m<!--n:a3.d3.flag.med--> | 730<!--n:a3.d3.fb.n-->, 6.6 m<!--n:a3.d3.fb.med--> | 2 084<!--n:a3.d3.kp.n-->, 10.5 m<!--n:a3.d3.kp.med--> |

**Kontrola konwencji.** Po odbiciu wykrytego y mediana błędu E3<!--n:a3.chosen--> rośnie do 16.7 m<!--n:a3.e3.mirror-->. Mediana błędu ze znakiem wynosi +0.3 m<!--n:a3.e3.dx--> wzdłuż x i −0.2 m<!--n:a3.e3.dy--> wzdłuż y.

## 6. Czy wykryte pozycje niosą informację: korelacje i stałe zgadywanie

Stałe zgadywanie to jedna pozycja dla wszystkich strzałów: mediana pozycji StatsBomb w populacji, oceniana na tych samych strzałach co detektor (wyrocznia, zna rozkład prawdziwych pozycji).

| Konfiguracja | Spearman x | Spearman y | Pearson x | Pearson y | mediana błędu: detektor | stała |
|---|---|---|---|---|---|---|
| E1 | 0.51 [0.48; 0.54]<!--n:a3.e1.rx--> | 0.49 [0.45; 0.52]<!--n:a3.e1.ry--> | 0.03 [−0.04; 0.15]<!--n:a3.e1.px--> | 0.07 [0.06; 0.13]<!--n:a3.e1.py--> | 9.2 m [8.8; 9.9]<!--n:a3.e1.med--> | 10.3 m [10.1; 10.6]<!--n:a3.e1.const--> |
| E2 | 0.65 [0.62; 0.68]<!--n:a3.e2.rx--> | 0.64 [0.61; 0.67]<!--n:a3.e2.ry--> | 0.21 [0.14; 0.51]<!--n:a3.e2.px--> | 0.16 [0.12; 0.44]<!--n:a3.e2.py--> | 5.2 m [4.7; 5.9]<!--n:a3.e2.med--> | 10.3 m [10.1; 10.6]<!--n:a3.e2.const--> |
| E3 | 0.66 [0.63; 0.68]<!--n:a3.e3.rx--> | 0.64 [0.61; 0.67]<!--n:a3.e3.ry--> | 0.20 [0.13; 0.39]<!--n:a3.e3.px--> | 0.16 [0.13; 0.31]<!--n:a3.e3.py--> | 5.2 m [4.7; 5.8]<!--n:a3.e3.med--> | 10.3 m [10.1; 10.6]<!--n:a3.e3.const--> |
| D3 | 0.41 [0.37; 0.45]<!--n:a3.d3.rx--> | 0.46 [0.42; 0.50]<!--n:a3.d3.ry--> | 0.05 [0.01; 0.18]<!--n:a3.d3.px--> | 0.09 [0.07; 0.17]<!--n:a3.d3.py--> | 10.5 m [9.9; 11.1]<!--n:a3.d3.med--> | 10.7 m [10.4; 11.0]<!--n:a3.d3.const--> |

W audycie 2 mediana błędu D3 była równa stałemu zgadywaniu. W E2 i E3 jest o połowę mniejsza od niego, a korelacja rang x wzrosła z 0.41<!--n:a3.d3.rx1--> do 0.66<!--n:a3.e3.rx1-->. Lokalizacja strzału niesie teraz wyraźną informację. Pearson pozostaje niski, bo kilka procent pozycji leży daleko poza boiskiem.

## 7. Bramkarz

| Konfiguracja | pokrycie | mediana | < 2 m | wykryty: n, mediana | zastępczy, ta sama drużyna: n, mediana | zastępczy, inna: n, mediana | stała: mediana, < 2 m |
|---|---|---|---|---|---|---|---|
| E1 | 98.8%<!--n:a3.e1.gk.cov1--> | 6.6 m [6.1; 7.1]<!--n:a3.e1.gk.med--> | 25.6%<!--n:a3.e1.gk.lt2--> | 497<!--n:a3.e1.gkd.n-->, 10.6 m<!--n:a3.e1.gkd.med--> | 1 176<!--n:a3.e1.same.n-->, 6.1 m<!--n:a3.e1.same.med--> | 1 012<!--n:a3.e1.other.n-->, 5.9 m<!--n:a3.e1.other.med--> | 2.0 m<!--n:a3.e1.gkc.med-->, 49.8%<!--n:a3.e1.gkc.lt2--> |
| E2 | 98.9%<!--n:a3.e2.gk.cov1--> | 6.6 m [6.0; 7.0]<!--n:a3.e2.gk.med--> | 26.6%<!--n:a3.e2.gk.lt2--> | 499<!--n:a3.e2.gkd.n-->, 10.6 m<!--n:a3.e2.gkd.med--> | 1 098<!--n:a3.e2.same.n-->, 6.1 m<!--n:a3.e2.same.med--> | 1 091<!--n:a3.e2.other.n-->, 5.8 m<!--n:a3.e2.other.med--> | 2.0 m<!--n:a3.e2.gkc.med-->, 49.9%<!--n:a3.e2.gkc.lt2--> |
| E3 | 98.9%<!--n:a3.e3.gk.cov1--> | 6.5 m [5.9; 6.9]<!--n:a3.e3.gk.med--> | 27.1%<!--n:a3.e3.gk.lt2--> | 499<!--n:a3.e3.gkd.n-->, 10.6 m<!--n:a3.e3.gkd.med--> | 1 092<!--n:a3.e3.same.n-->, 6.0 m<!--n:a3.e3.same.med--> | 1 097<!--n:a3.e3.other.n-->, 5.5 m<!--n:a3.e3.other.med--> | 2.0 m<!--n:a3.e3.gkc.med-->, 49.9%<!--n:a3.e3.gkc.lt2--> |
| D3 | 66.5%<!--n:a3.d3.gk.cov1--> | 8.6 m [8.1; 9.1]<!--n:a3.d3.gk.med--> | 16.7%<!--n:a3.d3.gk.lt2--> | 298<!--n:a3.d3.gkd.n-->, 14.8 m<!--n:a3.d3.gkd.med--> | 819<!--n:a3.d3.same.n-->, 8.2 m<!--n:a3.d3.same.med--> | 690<!--n:a3.d3.other.n-->, 8.2 m<!--n:a3.d3.other.med--> | 1.9 m<!--n:a3.d3.gkc.med-->, 51.5%<!--n:a3.d3.gkc.lt2--> |

Bramkarz jest dokładniejszy niż w audycie 2, ale nadal trzy razy gorszy od stałego zgadywania. Pozycja x bramkarza prawie nie koreluje z prawdą (Spearman E3 0.07 [0.02; 0.12]<!--n:a3.e3.gk.rx-->, y 0.31 [0.27; 0.36]<!--n:a3.e3.gk.ry-->). Gorzej wypada bramkarz wykryty jako klasa "goalkeeper" niż zastępczy, czyli zawodnik najbliżej bramki. Homografia na GSR myli się w medianie o mniej niż metr (sekcja 2), więc ten błąd pochodzi prawdopodobnie z wyboru zawodnika i z chwili klatki t0, a nie z homografii.

## 8. Obrońcy w stożku strzału

| Konfiguracja | n | zgodność dokładna | zgodność ≥ 1 | zgodność ≥ 2 | średnia różnica (detektor − StatsBomb) |
|---|---|---|---|---|---|
| E1 | 2 690<!--n:a3.e1.cone.n--> | 49.6% [47.7; 51.6]<!--n:a3.e1.cone.exact--> | 53.3%<!--n:a3.e1.cone.ge1--> | 84.2%<!--n:a3.e1.cone.ge2--> | −0.51<!--n:a3.e1.cone.diff--> |
| E2 | 2 690<!--n:a3.e2.cone.n--> | 49.0% [47.1; 51.0]<!--n:a3.e2.cone.exact--> | 53.8%<!--n:a3.e2.cone.ge1--> | 83.3%<!--n:a3.e2.cone.ge2--> | −0.47<!--n:a3.e2.cone.diff--> |
| E3 | 2 690<!--n:a3.e3.cone.n--> | 49.0% [47.0; 50.9]<!--n:a3.e3.cone.exact--> | 53.8%<!--n:a3.e3.cone.ge1--> | 83.3%<!--n:a3.e3.cone.ge2--> | −0.47<!--n:a3.e3.cone.diff--> |
| D3 | 1 812<!--n:a3.d3.cone.n--> | 48.0% [45.6; 50.3]<!--n:a3.d3.cone.exact--> | 51.3%<!--n:a3.d3.cone.ge1--> | 84.4%<!--n:a3.d3.cone.ge2--> | −0.54<!--n:a3.d3.cone.diff--> |

Macierz pomyłek E3<!--n:a3.chosen--> dla "co najmniej jeden obrońca w stożku":

| | StatsBomb: brak | StatsBomb: ≥ 1 |
|---|---|---|
| detektor: brak | 1 174<!--n:a3.cf.00--> | 1 022<!--n:a3.cf.01--> |
| detektor: ≥ 1 | 220<!--n:a3.cf.10--> | 274<!--n:a3.cf.11--> |

Zgodność stożka prawie się nie zmieniła. Detektor nadal częściej nie widzi obrońcy, którego StatsBomb ma w stożku, niż odwrotnie.

## 9. Źródło piłki: t0 wobec −400 ms (opisowo, po wyborze)

Skrypt `scripts/pk_audit3_diag.py` dzieli błędy tego audytu z tymi samymi losowaniami. Podział nie wpływa na wybór ani na bramkę.

| Konfiguracja | źródło | udział strzałów | n | mediana błędu | mediana błędu x |
|---|---|---|---|---|---|
| E1 | piłka w t0 | 67.6%<!--n:dg3.e1.ball_t0.share--> | 1 839<!--n:dg3.e1.ball_t0.n--> | 10.6 m [10.1; 11.1]<!--n:dg3.e1.ball_t0.med--> | +1.9 m<!--n:dg3.e1.ball_t0.dx--> |
| E1 | piłka z −400 ms | 21.3%<!--n:dg3.e1.ball_m400.share--> | 580<!--n:dg3.e1.ball_m400.n--> | 3.9 m [3.2; 4.6]<!--n:dg3.e1.ball_m400.med--> | +0.3 m<!--n:dg3.e1.ball_m400.dx--> |
| E1 | piłka z −800 ms | 4.6%<!--n:dg3.e1.ball_m800.share--> | 125<!--n:dg3.e1.ball_m800.n--> | 7.5 m [4.9; 9.9]<!--n:dg3.e1.ball_m800.med--> | −0.5 m<!--n:dg3.e1.ball_m800.dx--> |
| E1 | strzelec z flagi | 6.0%<!--n:dg3.e1.flag.share--> | 164<!--n:dg3.e1.flag.n--> | 8.9 m [7.7; 10.4]<!--n:dg3.e1.flag.med--> | +4.4 m<!--n:dg3.e1.flag.dx--> |
| E3 | piłka z −400 ms | 81.2%<!--n:dg3.e3.ball_m400.share--> | 2 209<!--n:dg3.e3.ball_m400.n--> | 4.3 m [3.8; 4.7]<!--n:dg3.e3.ball_m400.med--> | +0.2 m<!--n:dg3.e3.ball_m400.dx--> |
| E3 | piłka z −800 ms | 9.4%<!--n:dg3.e3.ball_m800.share--> | 255<!--n:dg3.e3.ball_m800.n--> | 9.5 m [7.5; 10.8]<!--n:dg3.e3.ball_m800.med--> | −0.6 m<!--n:dg3.e3.ball_m800.dx--> |
| E3 | piłka w t0 | 2.9%<!--n:dg3.e3.ball_t0.share--> | 80<!--n:dg3.e3.ball_t0.n--> | 8.9 m [6.6; 11.8]<!--n:dg3.e3.ball_t0.med--> | +2.0 m<!--n:dg3.e3.ball_t0.dx--> |
| E3 | strzelec z flagi | 6.0%<!--n:dg3.e3.flag.share--> | 164<!--n:dg3.e3.flag.n--> | 8.9 m [7.7; 10.4]<!--n:dg3.e3.flag.med--> | +4.4 m<!--n:dg3.e3.flag.dx--> |

W E1 ta sama homografia daje 10.6 m<!--n:dg3.e1.ball_t0.med1--> z piłką w t0 i 3.9 m<!--n:dg3.e1.ball_m400.med1--> z piłką z −400 ms. Piłka w t0 leży bliżej bramki niż miejsce strzału StatsBomb (dodatni błąd x), czyli zwykle już leci. Zgadza się to z czasem kopnięcia według prawdy GSR (sekcja 3). W E3 piłka z −400 ms ma błąd 4.3 m<!--n:dg3.e3.ball_m400.med1-->: nawet dla właściwej klatki pozostaje kilka metrów. Składają się na nie pomyłki detektora piłki (na GSR co czwarta umieszczona piłka jest innym obiektem), nieznana dokładna chwila strzału (−400 ms to stała, a kopnięcie wypada różnie) i szum czasowy obu adnotacji.

**Szum czasowy (opisowo).** Mediana błędu lokalizacji E3<!--n:a3.chosen--> według |rezydualnego czasu| parowania: do 50 ms 5.0 m<!--n:a3.tm.t1.med--> (2 125<!--n:a3.tm.t1.n--> strzałów), 50 ms do 200 ms 5.7 m<!--n:a3.tm.t2.med--> (509<!--n:a3.tm.t2.n-->), ponad 200 ms 10.4 m<!--n:a3.tm.t3.med--> (74<!--n:a3.tm.t3.n-->).

## 10. Mecze z klatkami w treningu modelu (warunek S)

Członek rady S prosił o podział bramki na mecze, których klatki z Calibration-2023 były w treningu modelu punktów kluczowych, i pozostałe. Wszystkie 135<!--n:dg3.matches--> meczów audytu należy do roli ewaluacyjnej planu podziału xg-vendor. Rada (C3, pytanie 2) wyłączyła ich klatki z treningu i walidacji modelu, więc w treningu jest 0<!--n:dg3.train--> z nich, a w walidacji 0<!--n:dg3.val-->. Dopasowanie nazw meczów działa: znajduje klatki Calibration-2023 w treningu dla 213<!--n:dg3.plan.train--> meczów treningowych planu. Bramka dotyczy więc w całości meczów, których model nie widział w Calibration-2023: pokrycie 99.5%<!--n:dg3.other.cov-->, mediana 5.2 m [4.7; 5.8]<!--n:dg3.other.med-->, bramkarz 98.9%<!--n:dg3.other.gk-->. GSR nie podaje nazw meczów, więc nie da się sprawdzić, czy klatki któregoś z tych meczów były w GSR train.

## 11. Pełna ekstrakcja E3<!--n:a3.chosen-->

E3<!--n:a3.chosen--> przeszła po wszystkich strzałach indeksu w t0 i −400 ms, z zespołami w obu klatkach; klatkę −800 ms analizujemy tylko po piłkę. Przebieg szedł w częściach po 500<!--n:fc3.chunk--> strzałów (wznawialnie) i trwał łącznie 64 min<!--n:fu3.seconds--> (346 ms<!--n:fu3.ms--> na strzał), na 24 273<!--n:fu3.frames--> klatkach. Cechy t0 porównaliśmy z pamięcią audytu 3 na 2 754<!--n:fu3.ref--> sparowanych strzałach: różnią się w 0<!--n:fu3.ref_diff--> strzałach. Pamięć jest w `cache/xgd_full3/E3` i nie nadpisuje D3 (`cache/xgd_full2/D3`).

Pokrycie według roli w planie podziału xg-vendor (udział strzałów):

| Rola | klatka | strzały | mecze | homografia | piłka | lokalizacja strzału | strzelec z flagi | bramkarz | stożek |
|---|---|---|---|---|---|---|---|---|---|
| trening | t0 | 7 166<!--n:fu3.training.t0.shots--> | 328<!--n:fu3.training.t0.matches--> | 98.6%<!--n:fu3.training.t0.homography_ok--> | 93.0%<!--n:fu3.training.t0.ball_ok--> | 98.9%<!--n:fu3.training.t0.shot_ok--> | 5.9%<!--n:fu3.training.t0.shooter_by_goal--> | 98.2%<!--n:fu3.training.t0.gk_ok--> | 98.2%<!--n:fu3.training.t0.def_cone--> |
| trening | −400 ms | 7 166<!--n:fu3.training.m400.shots--> | 328<!--n:fu3.training.m400.matches--> | 98.0%<!--n:fu3.training.m400.homography_ok--> | 89.2%<!--n:fu3.training.m400.ball_ok--> | 98.2%<!--n:fu3.training.m400.shot_ok--> | 9.0%<!--n:fu3.training.m400.shooter_by_goal--> | 97.7%<!--n:fu3.training.m400.gk_ok--> | 97.7%<!--n:fu3.training.m400.def_cone--> |
| bramka | t0 | 825<!--n:fu3.gate.t0.shots--> | 37<!--n:fu3.gate.t0.matches--> | 99.4%<!--n:fu3.gate.t0.homography_ok--> | 95.9%<!--n:fu3.gate.t0.ball_ok--> | 99.6%<!--n:fu3.gate.t0.shot_ok--> | 3.8%<!--n:fu3.gate.t0.shooter_by_goal--> | 99.2%<!--n:fu3.gate.t0.gk_ok--> | 99.3%<!--n:fu3.gate.t0.def_cone--> |
| bramka | −400 ms | 825<!--n:fu3.gate.m400.shots--> | 37<!--n:fu3.gate.m400.matches--> | 98.9%<!--n:fu3.gate.m400.homography_ok--> | 92.7%<!--n:fu3.gate.m400.ball_ok--> | 99.4%<!--n:fu3.gate.m400.shot_ok--> | 6.7%<!--n:fu3.gate.m400.shooter_by_goal--> | 98.8%<!--n:fu3.gate.m400.gk_ok--> | 98.8%<!--n:fu3.gate.m400.def_cone--> |
| ewaluacja | t0 | 3 039<!--n:fu3.evaluation.t0.shots--> | 135<!--n:fu3.evaluation.t0.matches--> | 99.1%<!--n:fu3.evaluation.t0.homography_ok--> | 93.1%<!--n:fu3.evaluation.t0.ball_ok--> | 99.3%<!--n:fu3.evaluation.t0.shot_ok--> | 6.3%<!--n:fu3.evaluation.t0.shooter_by_goal--> | 98.7%<!--n:fu3.evaluation.t0.gk_ok--> | 98.7%<!--n:fu3.evaluation.t0.def_cone--> |
| ewaluacja | −400 ms | 3 039<!--n:fu3.evaluation.m400.shots--> | 135<!--n:fu3.evaluation.m400.matches--> | 98.4%<!--n:fu3.evaluation.m400.homography_ok--> | 90.0%<!--n:fu3.evaluation.m400.ball_ok--> | 98.6%<!--n:fu3.evaluation.m400.shot_ok--> | 8.6%<!--n:fu3.evaluation.m400.shooter_by_goal--> | 97.6%<!--n:fu3.evaluation.m400.gk_ok--> | 97.5%<!--n:fu3.evaluation.m400.def_cone--> |
| wszystkie | t0 | 11 030<!--n:fu3.all.t0.shots--> | 500<!--n:fu3.all.t0.matches--> | 98.8%<!--n:fu3.all.t0.homography_ok--> | 93.2%<!--n:fu3.all.t0.ball_ok--> | 99.1%<!--n:fu3.all.t0.shot_ok--> | 5.9%<!--n:fu3.all.t0.shooter_by_goal--> | 98.4%<!--n:fu3.all.t0.gk_ok--> | 98.4%<!--n:fu3.all.t0.def_cone--> |
| wszystkie | −400 ms | 11 030<!--n:fu3.all.m400.shots--> | 500<!--n:fu3.all.m400.matches--> | 98.2%<!--n:fu3.all.m400.homography_ok--> | 89.7%<!--n:fu3.all.m400.ball_ok--> | 98.4%<!--n:fu3.all.m400.shot_ok--> | 8.7%<!--n:fu3.all.m400.shooter_by_goal--> | 97.7%<!--n:fu3.all.m400.gk_ok--> | 97.7%<!--n:fu3.all.m400.def_cone--> |

Z tych cech może korzystać dalsza praca rozwojowa; według rady (C3) nie służą one ponownemu rozwojowi A1, bo bramka dała STOP<!--n:a3.gate-->.

## 12. Co z tego wynika

1. **Bramka STOP<!--n:a3.gate-->.** Nowy model punktów kluczowych i piłka z klatki −400 ms obniżyły medianę błędu lokalizacji z 10.5 m<!--n:a3.d3.med1--> do 5.2 m<!--n:a3.g.med-->, przy prawie pełnym pokryciu. Próg 3 m<!--n:a3.med_max--> nie jest osiągnięty. Według rady (C3) podejście 2 pozostaje zamknięte, a A1 nie jest rozwijane od nowa.
2. **Homografia nie jest już głównym źródłem błędu.** Na GSR stopy graczy lądują w medianie 0.62 m<!--n:t3.med1--> od prawdy, a w audycie sam model (E1 wobec D3) obniżył medianę tylko o nieco ponad metr. Pozostały błąd pochodzi najpewniej z wyboru klatki i z detektora piłki: na GSR valid detektor znajduje piłkę na połowie klatek, a co czwarta umieszczona piłka jest innym obiektem.
3. **Dalsza praca nad produktem (poza tym badaniem).** Następną dźwignią jest detektor piłki uczony na klatkach SoccerNet z piłką z GSR, a potem reguła chwili strzału z toru piłki, gdy piłka będzie wykrywana na większości klatek. Bramkarz wymaga lepszego wyboru zawodnika: detektor rzadko oznacza bramkarza, a gdy go oznaczy, pozycja jest gorsza niż u zastępczego.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `cache/xgd_full3/E3/complete.json` | `2eefc603593e7f6f15c0af2f8f47bc65d30ce5c3b8cf83c9206b527aa3b75ff8` | rekord ukończenia |
| `research/pitchkp_audit3_configs.json` | `f7f23eaede754326a24b6d177c1e83e1eb6158c2cb22f1e36b81794604f8a8b7` | deklaracja E1–E3 |
| `results/pitchkp_gsr_eval/valid/summary.csv` | `f8a287179189727d771314d6ab9a8a3f2a6f6fc91fe057feff8e9906fa5c35e3` | ocena homografii na GSR valid |
| `results/pk_ball_evidence/ball.csv` | `dcaecfae0ba5ff31f512b37e28cd60ecc7b428dee05e6c879bfc4853a12982b6` | umieszczenie piłki na GSR valid |
| `results/pk_ball_evidence/summary.json` | `4b5873b6612a9ab6057a0ef93c83a6836cb09f92bef3c24f28d7e2630a1f98c9` | dowody z GSR: podsumowanie |
| `results/pk_candidates/y11m_960_e18_square/retune_valid/summary.csv` | `4327764fc6c983c79044a482bfea4f7cdc92e08c602cd4607a022a263cffb197` | ocena homografii na GSR valid |
| `results/pk_square_input/complete.json` | `8b6e86455948a1f3fc87d62aee8ff6c07a7fda96cac394863f4e53f154b2749d` | rekord ukończenia |
| `results/xgd_audit/metrics.csv` | `094eb796175da304a1e650f2aaa09500d08878d2562887ea2f56788137077280` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `results/xgd_audit2_diag/diag.csv` | `32f45780139e42676b25fd81417d5bd9c2ef5f22b88995af9e59b2a47bd1776e` | podziały błędów (opisowo) |
| `results/xgd_audit3/choice.json` | `2714e7dad473f0d02e79359a98c6d8e0c9356d25877c03de1d73be13bf25ef02` | wybór konfiguracji i bramka |
| `results/xgd_audit3/complete.json` | `dae5c570a9b6a869744469f38b20135b11ecba2d14051291dd153844bbb7cbbb` | rekord ukończenia |
| `results/xgd_audit3/cone_confusion.csv` | `328dde2fc490bd685961d1b0ab5b16bdb7e014169a346142accd17d02f7117c5` | macierze pomyłek stożka |
| `results/xgd_audit3/metrics.csv` | `38d2d5b4d67761d4a80487b76e8b5fa9ff0a2ac15cafcdb50ef2523f8ace6d93` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `results/xgd_audit3/timing.csv` | `60749694bf14ab14c21f9cf20cd96a2b264737263bcb8b337548f22039dae463` | błędy według rezyduum czasu |
| `results/xgd_audit3_diag/calibration_plan.csv` | `de5348a116f36eb11e774a5d704581776f3c4d693005f382f6f3c26e80f8b770` | mecze planu według klatek Calibration-2023 |
| `results/xgd_audit3_diag/complete.json` | `eb8d7a515a123a3049845301a7346b51938a9fa1a35d38750a2555d957f35561` | rekord ukończenia |
| `results/xgd_audit3_diag/diag.csv` | `dcea6970d6d044c1b83a9e7c1ce6eff81757a48659151d5c9bdbc28085798eb5` | podziały błędów (opisowo) |
| `results/xgd_full3/coverage_by_role.csv` | `a3a26ef214644a6eff4564da655e0b911abd2a5f37bd241c4b646750a548b499` | pokrycie według roli i klatki |
| `xgpoc/detector/audit.py` | `c40fe17f84549d7b51f5805791e57ee7f76c09a8e08d52fe3c202e25383b6e14` | kod audytu: poziom przedziałów |
