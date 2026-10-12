# Audyt pozycji z detektora wobec StatsBomb (xg-detector, ticket 02 i ticket 09)

Jakub Eichner, 2026-10-12 (audyt 1: ticket 02, audyt 2: ticket 09, czas UTC). Gałąź `research/xg-detector`, badanie xg-detector.

Data: StatsBomb. Pozycje odniesienia (miejsce strzału, bramkarz, obrońcy) pochodzą z freeze frames w otwartych danych StatsBomb.

Audyt mierzy, jak dokładnie pipeline detektora umieszcza na boisku miejsce strzału, bramkarza i obrońców. To pomiar diagnostyczny. Żaden model nie dostaje danych StatsBomb, ani jako wejścia, ani jako celu. Audyt nie czyta żadnej kolumny z golem ani wynikiem strzału. Wybór konfiguracji i bramka widzą tylko pokrycie i błędy pozycji.

Sekcje 1–9 opisują audyt 1 (konfiguracje C1–C4). Sekcja 10 opisuje audyt 2 na homografii poprawionej przez ticket 08 (konfiguracje D1–D4). To ostatni audyt. Jego bramka też daje STOP<!--n:a2.gate-->, a wybrana konfiguracja to D3<!--n:a2.chosen-->.

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `claims.toml` mówi, z którego pliku wyników pochodzi liczba. Skrypt `scripts/check_xgd_audit.py` odczytuje każdą liczbę ze źródła i kończy się błędem przy niezgodności.

## 1. Wynik: bramka STOP<!--n:au.gate-->

Według zamrożonej reguły wybrana jest konfiguracja C4<!--n:au.chosen-->. Tylko ona ma pokrycie lokalizacji strzału co najmniej 70%<!--n:au.cov_min--> strzałów. Reguła wyboru i bramka używają estymat punktowych. Przedział pokrycia C4<!--n:au.chosen-->, 71.8% [67.3; 76.2]<!--n:au.c4.cov-->, obejmuje próg, więc kwalifikacja C4<!--n:au.chosen--> jest na granicy. Bramka dla podejścia 2 wymaga trzech warunków:

| Warunek | Próg | C4<!--n:au.chosen--> | Spełniony |
|---|---|---|---|
| pokrycie lokalizacji strzału | ≥ 70%<!--n:au.cov_min--> | 71.8%<!--n:au.g.cov--> | tak |
| mediana błędu lokalizacji | ≤ 3 m<!--n:au.med_max--> | 12.1 m<!--n:au.g.med--> | nie |
| informacja o bramkarzu (wykryty albo zastępczy) | ≥ 50%<!--n:au.gk_min--> | 60.2%<!--n:au.g.gk--> | tak |

Mediana błędu jest około czterokrotnie wyższa od progu. Zgodnie ze specyfikacją podejście 2 (ticket 05) kończy się na tym raporcie. Podejście 1 używa C4<!--n:au.chosen-->.

Najbardziej prawdopodobna przyczyna to homografia. Bramkarz stoi zwykle kilka metrów od bramki, a mediana błędu jego pozycji to 6.9 m [6.4; 7.4]<!--n:au.c4.gk.med--> (sekcja 4). Na przykładowych klatkach linie boiska rzutowane homografią często mijają namalowane linie. Część błędu bierze się z wejścia modelu punktów kluczowych (sekcja 7).

## 2. Populacja i metoda

- **Strzały.** Sparowane strzały bez karnych: 2 721<!--n:au.shots--> strzałów z 135<!--n:au.matches--> meczów. Freeze frame ma 2 721<!--n:au.ff--> z nich, bramkarza we freeze frame 2 719<!--n:au.sbgk-->.
- **Klatka.** Klatka kotwicy SoccerNet (t0). C2 i C4 biorą piłkę z klatek −400 ms i −800 ms, gdy t0 jej nie ma.
- **Współrzędne.** Metry boiska 105 × 68, wyrównane tak, że atakowana bramka leży przy x = 105. StatsBomb przeliczamy z jardów wokół środka bramki (`conventions.statsbomb_to_metres`).
- **Błąd.** Odległość euklidesowa w metrach. Pokrycie to udział wszystkich 2 721<!--n:au.shots--> strzałów, dla których pipeline daje daną wielkość.
- **Przedziały.** Percentylowe przedziały 95%<!--n:au.ci--> z bootstrapu po meczach, 5 000<!--n:au.n_boot--> losowań, te same dla każdej konfiguracji.
- **Szum czasowy.** Błędy zawierają szum adnotacji czasu obu źródeł, około 1 s. W tym czasie zawodnicy przesuwają się o 3–6 m, a piłka po strzale dalej. Sama homografia daje jednak większy błąd (sekcje 4 i 7).

## 3. Lokalizacja strzału

Cztery konfiguracje zadeklarowane przed audytem: C1 (wejście detektora 640, próg piłki 0.30<!--n:cfg.c1.ball-->, bez piłki z wcześniejszych klatek), C2 (jak C1, z piłką z wcześniejszych klatek), C3 (wejście 1920, próg piłki 0.15<!--n:cfg.c3.ball-->, bez), C4 (jak C3, z).

| Konfiguracja | pokrycie | mediana błędu | p90 | < 1 m | < 2 m | < 5 m |
|---|---|---|---|---|---|---|
| C1 | 60.4% [55.3; 65.5]<!--n:au.c1.cov--> | 15.8 m [15.0; 16.4]<!--n:au.c1.median--> | 27.0 m<!--n:au.c1.p90--> | 0.1%<!--n:au.c1.lt1--> | 0.5%<!--n:au.c1.lt2--> | 6.0%<!--n:au.c1.lt5--> |
| C2 | 63.9% [59.0; 68.8]<!--n:au.c2.cov--> | 12.6 m [11.6; 13.3]<!--n:au.c2.median--> | 25.5 m<!--n:au.c2.p90--> | 0.3%<!--n:au.c2.lt1--> | 1.3%<!--n:au.c2.lt2--> | 12.1%<!--n:au.c2.lt5--> |
| C3 | 60.4% [55.3; 65.5]<!--n:au.c3.cov--> | 13.4 m [12.9; 14.2]<!--n:au.c3.median--> | 27.2 m<!--n:au.c3.p90--> | 0.2%<!--n:au.c3.lt1--> | 1.6%<!--n:au.c3.lt2--> | 11.1%<!--n:au.c3.lt5--> |
| C4 | 71.8% [67.3; 76.2]<!--n:au.c4.cov--> | 12.1 m [11.4; 12.8]<!--n:au.c4.median--> | 27.2 m<!--n:au.c4.p90--> | 0.4%<!--n:au.c4.lt1--> | 1.7%<!--n:au.c4.lt2--> | 14.5%<!--n:au.c4.lt5--> |

Udziały < 1/2/5 m liczymy wśród strzałów z lokalizacją.

**Według źródła lokalizacji** (decyzja Q23). "Piłka" to lokalizacja z piłki, własnej albo z wcześniejszej klatki. "Flaga" to strzelec wybrany bez piłki, czyli zawodnik najbliżej środka bramki. "Wcześniejsza klatka" to podzbiór "piłki". Kolumna "instancja ≥ 0.5<!--n:au.kp-->" (decyzja Q25) zostawia strzały, w których każda homografia za porównywaną pozycją ma pewność instancji punktów kluczowych co najmniej 0.5<!--n:au.kp-->.

| Konfiguracja | piłka: n, mediana | flaga: n, mediana | wcześniejsza klatka: n, mediana | instancja ≥ 0.5<!--n:au.kp-->: n, mediana |
|---|---|---|---|---|
| C1 | 142<!--n:au.c1.ball.n-->, 9.8 m<!--n:au.c1.ball.med--> | 1 502<!--n:au.c1.flag.n-->, 16.2 m<!--n:au.c1.flag.med--> | 0, — | 1 444<!--n:au.c1.kp05.n-->, 15.7 m<!--n:au.c1.kp05.med--> |
| C2 | 692<!--n:au.c2.ball.n-->, 8.7 m<!--n:au.c2.ball.med--> | 1 048<!--n:au.c2.flag.n-->, 15.8 m<!--n:au.c2.flag.med--> | 550<!--n:au.c2.fallback.n-->, 8.5 m<!--n:au.c2.fallback.med--> | 1 500<!--n:au.c2.kp05.n-->, 12.0 m<!--n:au.c2.kp05.med--> |
| C3 | 1 129<!--n:au.c3.ball.n-->, 13.6 m<!--n:au.c3.ball.med--> | 515<!--n:au.c3.flag.n-->, 13.4 m<!--n:au.c3.flag.med--> | 0, — | 1 444<!--n:au.c3.kp05.n-->, 13.1 m<!--n:au.c3.kp05.med--> |
| C4 | 1 833<!--n:au.c4.ball.n-->, 12.1 m<!--n:au.c4.ball.med--> | 121<!--n:au.c4.flag.n-->, 11.9 m<!--n:au.c4.flag.med--> | 704<!--n:au.c4.fallback.n-->, 10.4 m<!--n:au.c4.fallback.med--> | 1 639<!--n:au.c4.kp05.n-->, 11.3 m<!--n:au.c4.kp05.med--> |

W C1 i C2 lokalizacja z piłki jest o kilka metrów lepsza niż z flagi. W C3 i C4 obie mają podobną medianę: niższy próg piłki i większe wejście dają więcej piłek, ale nie dokładniejszych. Próg instancji 0.5<!--n:au.kp--> zmniejsza medianę najwyżej o metr.

**Kontrola konwencji.** Po odbiciu wykrytego y (y → 68 − y) mediana błędu C4 rośnie do 18.3 m<!--n:au.c4.mirror-->. Oś y nie jest więc odwrócona. Mediana błędu ze znakiem w C4 wynosi +2.1 m<!--n:au.c4.dx--> wzdłuż x (dodatni: bliżej bramki niż w StatsBomb) i −2.2 m<!--n:au.c4.dy--> wzdłuż y.

## 4. Bramkarz

Bramkarz to wykryty bramkarz najbliżej atakowanej bramki albo zastępczy: zawodnik najbliżej bramki na atakowanej połowie, inny niż strzelec. "Zastępczy, ta sama drużyna" znaczy, że jego klaster stroju to klaster strzelca. Błąd liczymy na strzałach z bramkarzem we freeze frame.

| Konfiguracja | pokrycie | mediana | < 2 m | wykryty: n, mediana | zastępczy, ta sama drużyna: n, mediana | zastępczy, inna: n, mediana | instancja ≥ 0.5<!--n:au.kp-->: mediana |
|---|---|---|---|---|---|---|---|
| C1 | 60.4%<!--n:au.c1.gk.cov--> | 7.9 m [7.6; 8.1]<!--n:au.c1.gk.med--> | 4.9%<!--n:au.c1.gk.lt2--> | 1<!--n:au.c1.gkd.n-->, 26.5 m<!--n:au.c1.gkd.med--> | 908<!--n:au.c1.same.n-->, 7.6 m<!--n:au.c1.same.med--> | 733<!--n:au.c1.other.n-->, 8.1 m<!--n:au.c1.other.med--> | 8.0 m<!--n:au.c1.gkkp.med--> |
| C2 | 60.3%<!--n:au.c2.gk.cov--> | 6.7 m [6.3; 7.0]<!--n:au.c2.gk.med--> | 7.0%<!--n:au.c2.gk.lt2--> | 1<!--n:au.c2.gkd.n-->, 26.5 m<!--n:au.c2.gkd.med--> | 884<!--n:au.c2.same.n-->, 6.5 m<!--n:au.c2.same.med--> | 755<!--n:au.c2.other.n-->, 6.9 m<!--n:au.c2.other.med--> | 6.6 m<!--n:au.c2.gkkp.med--> |
| C3 | 60.2%<!--n:au.c3.gk.cov--> | 7.6 m [7.2; 8.1]<!--n:au.c3.gk.med--> | 6.2%<!--n:au.c3.gk.lt2--> | 274<!--n:au.c3.gkd.n-->, 12.1 m<!--n:au.c3.gkd.med--> | 697<!--n:au.c3.same.n-->, 6.4 m<!--n:au.c3.same.med--> | 666<!--n:au.c3.other.n-->, 7.0 m<!--n:au.c3.other.med--> | 7.6 m<!--n:au.c3.gkkp.med--> |
| C4 | 60.2%<!--n:au.c4.gk.cov--> | 6.9 m [6.4; 7.4]<!--n:au.c4.gk.med--> | 7.6%<!--n:au.c4.gk.lt2--> | 274<!--n:au.c4.gkd.n-->, 12.1 m<!--n:au.c4.gkd.med--> | 695<!--n:au.c4.same.n-->, 5.5 m<!--n:au.c4.same.med--> | 666<!--n:au.c4.other.n-->, 6.5 m<!--n:au.c4.other.med--> | 6.9 m<!--n:au.c4.gkkp.med--> |

Wykryty bramkarz ma błąd większy niż zastępczy. Zastępczy z klastra strzelca nie jest gorszy od zastępczego z drugiego klastra, co zgadza się z decyzją z ticketu 01, by nie filtrować go po drużynie.

**Według źródła lokalizacji** (Q23). Lokalizacja strzału wpływa na wybór bramkarza zastępczego i na wierzchołek stożka, więc oba błędy dzielimy tak samo jak w sekcji 3.

| Konfiguracja | bramkarz, piłka: n, mediana | bramkarz, flaga: n, mediana | stożek, piłka: n, dokładna, ≥ 1 | stożek, flaga: n, dokładna, ≥ 1 |
|---|---|---|---|---|
| C1 | 142<!--n:au.c1.gk.ball.n-->, 4.0 m<!--n:au.c1.gk.ball.med--> | 1 500<!--n:au.c1.gk.flag.n-->, 8.2 m<!--n:au.c1.gk.flag.med--> | 142<!--n:au.c1.cone.ball.n-->, 62.0%<!--n:au.c1.cone.ball.exact-->, 69.7%<!--n:au.c1.cone.ball.ge1--> | 1 502<!--n:au.c1.cone.flag.n-->, 50.1%<!--n:au.c1.cone.flag.exact-->, 51.3%<!--n:au.c1.cone.flag.ge1--> |
| C2 | 594<!--n:au.c2.gk.ball.n-->, 4.7 m<!--n:au.c2.gk.ball.med--> | 1 046<!--n:au.c2.gk.flag.n-->, 8.0 m<!--n:au.c2.gk.flag.med--> | 596<!--n:au.c2.cone.ball.n-->, 51.3%<!--n:au.c2.cone.ball.exact-->, 60.9%<!--n:au.c2.cone.ball.ge1--> | 1 048<!--n:au.c2.cone.flag.n-->, 51.0%<!--n:au.c2.cone.flag.exact-->, 52.2%<!--n:au.c2.cone.flag.ge1--> |
| C3 | 1 127<!--n:au.c3.gk.ball.n-->, 6.7 m<!--n:au.c3.gk.ball.med--> | 510<!--n:au.c3.gk.flag.n-->, 9.0 m<!--n:au.c3.gk.flag.med--> | 1 128<!--n:au.c3.cone.ball.n-->, 46.5%<!--n:au.c3.cone.ball.exact-->, 52.1%<!--n:au.c3.cone.ball.ge1--> | 510<!--n:au.c3.cone.flag.n-->, 49.8%<!--n:au.c3.cone.flag.exact-->, 51.0%<!--n:au.c3.cone.flag.ge1--> |
| C4 | 1 518<!--n:au.c4.gk.ball.n-->, 6.7 m<!--n:au.c4.gk.ball.med--> | 117<!--n:au.c4.gk.flag.n-->, 8.6 m<!--n:au.c4.gk.flag.med--> | 1 522<!--n:au.c4.cone.ball.n-->, 46.6%<!--n:au.c4.cone.ball.exact-->, 52.8%<!--n:au.c4.cone.ball.ge1--> | 116<!--n:au.c4.cone.flag.n-->, 52.6%<!--n:au.c4.cone.flag.exact-->, 53.4%<!--n:au.c4.cone.flag.ge1--> |

Przedziały tych podzbiorów są w `results/xgd_audit/metrics.csv`.

## 5. Obrońcy w stożku strzału

Liczba obrońców z pola w trójkącie lokalizacja strzału – słupek – słupek, na strzałach z liczbą z obu źródeł.

| Konfiguracja | n | zgodność dokładna | zgodność ≥ 1 | zgodność ≥ 2 | średnia różnica (detektor − StatsBomb) |
|---|---|---|---|---|---|
| C1 | 1 644<!--n:au.c1.cone.n--> | 51.2% [48.8; 53.4]<!--n:au.c1.cone.exact--> | 52.9%<!--n:au.c1.cone.ge1--> | 85.2%<!--n:au.c1.cone.ge2--> | −0.59<!--n:au.c1.cone.diff--> |
| C2 | 1 644<!--n:au.c2.cone.n--> | 51.2% [48.9; 53.4]<!--n:au.c2.cone.exact--> | 55.4%<!--n:au.c2.cone.ge1--> | 84.2%<!--n:au.c2.cone.ge2--> | −0.48<!--n:au.c2.cone.diff--> |
| C3 | 1 638<!--n:au.c3.cone.n--> | 47.6% [45.0; 50.0]<!--n:au.c3.cone.exact--> | 51.8%<!--n:au.c3.cone.ge1--> | 83.1%<!--n:au.c3.cone.ge2--> | −0.43<!--n:au.c3.cone.diff--> |
| C4 | 1 638<!--n:au.c4.cone.n--> | 47.1% [44.6; 49.4]<!--n:au.c4.cone.exact--> | 52.8%<!--n:au.c4.cone.ge1--> | 82.7%<!--n:au.c4.cone.ge2--> | −0.36<!--n:au.c4.cone.diff--> |

Macierz pomyłek C4 dla "co najmniej jeden obrońca w stożku":

| | StatsBomb: brak | StatsBomb: ≥ 1 |
|---|---|---|
| detektor: brak | 696<!--n:au.cf.00--> | 625<!--n:au.cf.01--> |
| detektor: ≥ 1 | 148<!--n:au.cf.10--> | 169<!--n:au.cf.11--> |

Zgodność ≥ 1 jest bliska rzutu monetą. Detektor częściej nie widzi obrońcy, którego StatsBomb ma w stożku, niż odwrotnie.

## 6. Szum czasowy (opisowo)

Mediana błędu lokalizacji C4 według |rezydualnego czasu| parowania xg-vendor. Nic nie jest wybierane na tej podstawie.

| |rezyduum| | strzały z lokalizacją | mediana błędu |
|---|---|---|
| do 50 ms | 1 548<!--n:au.tm.t1.n--> | 11.9 m<!--n:au.tm.t1.med--> |
| 50 ms do 200 ms | 349<!--n:au.tm.t2.n--> | 13.1 m<!--n:au.tm.t2.med--> |
| ponad 200 ms | 57<!--n:au.tm.t3.n--> | 14.5 m<!--n:au.tm.t3.med--> |

Błąd nie maleje przy dobrze dopasowanym czasie. Rezyduum mierzy jednak tylko zgodność zegarów po przesunięciu połowy, a nie opóźnienie adnotacji względem kontaktu z piłką.

## 7. Diagnostyka: wejście modelu punktów kluczowych (poza wyborem i bramką)

Model punktów kluczowych uczono na klatkach rozciągniętych do 640 × 640 (eksport Roboflow). Pakiet, a więc C1–C4, podaje mu klatkę z letterboxem. Na przykładowych klatkach punkty z letterboxa leżą obok namalowanych punktów boiska, a punkty z rozciągniętej klatki na nich. Diagnostyka `scripts/xgd_kp_diag.py` powtarza C4 z rozciągniętym wejściem: te same detekcje i drużyny, nowe punkty kluczowe, nowa homografia ("C4s"). To nie jest piąta konfiguracja. Bramka zostaje STOP<!--n:au.gate-->.

| | C4 (letterbox) | C4s (rozciągnięte) |
|---|---|---|
| homografia w t0 (sparowane strzały) | 60.0%<!--n:au.c4.hom--> | 66.9%<!--n:kd.hom--> |
| pokrycie lokalizacji strzału | 71.8% [67.3; 76.2]<!--n:au.c4.cov--> | 73.8% [70.6; 76.9]<!--n:kd.cov--> |
| mediana błędu lokalizacji | 12.1 m [11.4; 12.8]<!--n:au.c4.median--> | 10.7 m [10.1; 11.2]<!--n:kd.median--> |
| lokalizacja z piłki: mediana | 12.1 m<!--n:au.c4.ball.med--> | 10.7 m<!--n:kd.ball.med--> |
| błąd < 5 m | 14.5%<!--n:au.c4.lt5--> | 24.0%<!--n:kd.lt5--> |
| informacja o bramkarzu | 60.2%<!--n:au.c4.gk.cov--> | 67.1%<!--n:kd.gk.cov--> |
| mediana błędu bramkarza | 6.9 m [6.4; 7.4]<!--n:au.c4.gk.med--> | 8.2 m [7.8; 8.8]<!--n:kd.gk.med--> |
| zgodność dokładna stożka | 47.1% [44.6; 49.4]<!--n:au.c4.cone.exact--> | 48.1%<!--n:kd.cone.exact--> |

Rozciągnięte wejście daje więcej homografii i trochę mniejszy błąd lokalizacji, ale błąd bramkarza rośnie. Do progu 3 m<!--n:au.med_max--> jest daleko w obu wariantach. W 1 650<!--n:kd.padded--> strzałach C4s nie ma w pamięci podręcznej klatki wcześniejszej, której mógłby potrzebować; liczy się wtedy jako klatka bez piłki. Na przykładowych klatkach z rozciągniętym wejściem rzut linii boiska nadal często mija namalowane linie. Wąskim gardłem jest więc jakość punktów kluczowych na klatkach SoccerNet, nie sam sposób skalowania.

## 8. Pełna ekstrakcja C4

C4 przeszła po wszystkich strzałach indeksu w t0 i −400 ms. Klatkę −800 ms analizujemy tylko po piłkę. Przebieg trwał 25 min<!--n:fu.seconds--> (137 ms<!--n:fu.ms--> na strzał) i przeanalizował 27 430<!--n:fu.frames--> klatek. Cechy t0 pełnego przebiegu porównaliśmy z pamięcią audytu na 2 754<!--n:fu.ref--> sparowanych strzałach: różnią się w 0<!--n:fu.ref_diff--> strzałach.

Pokrycie według roli w planie podziału xg-vendor (udział strzałów):

| Rola | klatka | strzały | mecze | homografia | piłka | lokalizacja strzału | strzelec z flagi | bramkarz | stożek |
|---|---|---|---|---|---|---|---|---|---|
| trening | t0 | 7 166<!--n:fu.training.t0.shots--> | 328<!--n:fu.training.t0.matches--> | 60.7%<!--n:fu.training.t0.homography_ok--> | 68.3%<!--n:fu.training.t0.ball_ok--> | 72.1%<!--n:fu.training.t0.shot_ok--> | 3.8%<!--n:fu.training.t0.shooter_by_goal--> | 60.6%<!--n:fu.training.t0.gk_ok--> | 60.7%<!--n:fu.training.t0.def_cone--> |
| trening | −400 ms | 7 166<!--n:fu.training.m400.shots--> | 328<!--n:fu.training.m400.matches--> | 62.1%<!--n:fu.training.m400.homography_ok--> | 63.3%<!--n:fu.training.m400.ball_ok--> | 68.2%<!--n:fu.training.m400.shot_ok--> | 4.9%<!--n:fu.training.m400.shooter_by_goal--> | 62.0%<!--n:fu.training.m400.gk_ok--> | 62.0%<!--n:fu.training.m400.def_cone--> |
| bramka | t0 | 825<!--n:fu.gate.t0.shots--> | 37<!--n:fu.gate.t0.matches--> | 51.3%<!--n:fu.gate.t0.homography_ok--> | 62.2%<!--n:fu.gate.t0.ball_ok--> | 64.0%<!--n:fu.gate.t0.shot_ok--> | 1.8%<!--n:fu.gate.t0.shooter_by_goal--> | 51.0%<!--n:fu.gate.t0.gk_ok--> | 51.3%<!--n:fu.gate.t0.def_cone--> |
| bramka | −400 ms | 825<!--n:fu.gate.m400.shots--> | 37<!--n:fu.gate.m400.matches--> | 54.5%<!--n:fu.gate.m400.homography_ok--> | 58.4%<!--n:fu.gate.m400.ball_ok--> | 61.5%<!--n:fu.gate.m400.shot_ok--> | 3.0%<!--n:fu.gate.m400.shooter_by_goal--> | 54.5%<!--n:fu.gate.m400.gk_ok--> | 54.5%<!--n:fu.gate.m400.def_cone--> |
| ewaluacja | t0 | 3 039<!--n:fu.evaluation.t0.shots--> | 135<!--n:fu.evaluation.t0.matches--> | 59.0%<!--n:fu.evaluation.t0.homography_ok--> | 66.2%<!--n:fu.evaluation.t0.ball_ok--> | 70.5%<!--n:fu.evaluation.t0.shot_ok--> | 4.3%<!--n:fu.evaluation.t0.shooter_by_goal--> | 58.7%<!--n:fu.evaluation.t0.gk_ok--> | 58.8%<!--n:fu.evaluation.t0.def_cone--> |
| ewaluacja | −400 ms | 3 039<!--n:fu.evaluation.m400.shots--> | 135<!--n:fu.evaluation.m400.matches--> | 60.9%<!--n:fu.evaluation.m400.homography_ok--> | 61.4%<!--n:fu.evaluation.m400.ball_ok--> | 67.0%<!--n:fu.evaluation.m400.shot_ok--> | 5.6%<!--n:fu.evaluation.m400.shooter_by_goal--> | 60.6%<!--n:fu.evaluation.m400.gk_ok--> | 60.6%<!--n:fu.evaluation.m400.def_cone--> |
| wszystkie | t0 | 11 030<!--n:fu.all.t0.shots--> | 500<!--n:fu.all.t0.matches--> | 59.5%<!--n:fu.all.t0.homography_ok--> | 67.3%<!--n:fu.all.t0.ball_ok--> | 71.1%<!--n:fu.all.t0.shot_ok--> | 3.8%<!--n:fu.all.t0.shooter_by_goal--> | 59.4%<!--n:fu.all.t0.gk_ok--> | 59.4%<!--n:fu.all.t0.def_cone--> |
| wszystkie | −400 ms | 11 030<!--n:fu.all.m400.shots--> | 500<!--n:fu.all.m400.matches--> | 61.2%<!--n:fu.all.m400.homography_ok--> | 62.4%<!--n:fu.all.m400.ball_ok--> | 67.3%<!--n:fu.all.m400.shot_ok--> | 4.9%<!--n:fu.all.m400.shooter_by_goal--> | 61.1%<!--n:fu.all.m400.gk_ok--> | 61.1%<!--n:fu.all.m400.def_cone--> |

## 9. Decyzje dla właściciela (po audycie 1)

Właściciel zdecydował po audycie 1 (Q26, Q27): poprawić homografię (ticket 08) i przeprowadzić ostatni audyt 2 (sekcja 10); podejście 1 czeka na audyt 2.

1. **Bramka STOP<!--n:au.gate-->.** Podejście 2 kończy się na tym raporcie, chyba że właściciel zmieni specyfikację.
2. **Wejście punktów kluczowych.** Pakiet podaje modelowi letterbox, a model uczono na klatkach rozciągniętych. To błąd produktu niezależny od tego badania. Poprawka sama nie przechodzi bramki (sekcja 7).
3. **Podejście 1 z C4.** Cechy C4 mają duży błąd pozycji, a modele dostaną je z flagami pokrycia. Właściciel decyduje, czy podejście 1 rusza na tych cechach, czy czeka na lepszą homografię (`line-refined-homography`, poprawka wejścia, inny model punktów kluczowych). Lepsza homografia wymagałaby nowego zestawu konfiguracji zadeklarowanego przed ponownym audytem.

## 10. Audyt 2 (ticket 09): homografia po poprawkach

Po audycie 1 właściciel postanowił (Q26), że homografia zostanie zdiagnozowana i poprawiona w pakiecie (ticket 08), a potem odbędzie się jeden, ostatni audyt na konfiguracjach zadeklarowanych przed nim. Ticket 08 poprawił wejście modelu punktów kluczowych (klatka rozciągnięta do 640 × 640, jak w treningu), odrzucił osobliwe dopasowania i dodał opcjonalne doprecyzowanie homografii liniami boiska. Konfiguracje D1–D4 wybrał na metryce bez etykiet, na klatkach meczów treningowych, i zapisał w `research/xg_detector_audit2_configs.json` (SHA-256 `b7bfcffb8e36d22c23eca13a98eee887fcd1bfe6aa36435ba47a3a8f30cf4448`, commit `4a60000`). Skrypt audytu odmawia pracy, gdy plik ma inny skrót. Po audycie nie dodano żadnej konfiguracji. Reguła wyboru i bramka są te same co w audycie 1. Trzeciego audytu nie będzie.

### 10.1 Wynik: bramka STOP<!--n:a2.gate-->

Wybrana jest konfiguracja D3<!--n:a2.chosen-->. Wszystkie cztery mają pokrycie lokalizacji strzału co najmniej 70%<!--n:au.cov_min-->, więc wszystkie kwalifikują się do wyboru, a decyduje najniższa mediana błędu.

| Warunek | Próg | D3<!--n:a2.chosen--> | Spełniony |
|---|---|---|---|
| pokrycie lokalizacji strzału | ≥ 70%<!--n:au.cov_min--> | 77.4%<!--n:a2.g.cov--> | tak |
| mediana błędu lokalizacji | ≤ 3 m<!--n:au.med_max--> | 10.5 m<!--n:a2.g.med--> | nie |
| informacja o bramkarzu (wykryty albo zastępczy) | ≥ 50%<!--n:au.gk_min--> | 66.5%<!--n:a2.g.gk--> | tak |

Mediany D1–D4 różnią się o ułamki metra, a ich przedziały prawie się pokrywają. D3<!--n:a2.chosen--> wygrywa z D1 o kilka centymetrów (10.55 m<!--n:a2.g.med2--> wobec 10.57 m<!--n:a2.d1.med2-->). Reguła używa estymat punktowych, więc wybór jest zgodny z regułą, ale między D1–D4 praktycznie nie ma różnicy w błędzie. Zgodnie ze specyfikacją podejście 2 (ticket 05) kończy się na tym raporcie. Podejście 1 (ticket 03) używa cech D3<!--n:a2.chosen--> z pełnej ekstrakcji (sekcja 10.8), bez względu na wynik bramki.

### 10.2 Konfiguracje i populacja

Wspólne dla D1–D4: detektor C4 (wejście 1920, próg piłki 0.15<!--n:cfg.d1.ball-->, piłka z klatek −400 ms i −800 ms, gdy t0 jej nie ma), model punktów kluczowych na klatce rozciągniętej, bezstanowa homografia każdej klatki.

| Konfiguracja | próg punktów kluczowych | doprecyzowanie liniami |
|---|---|---|
| D1 | 0.5<!--n:cfg.d1.kp--> | nie |
| D2 | 0.3<!--n:cfg.d2.kp--> | nie |
| D3 | 0.5<!--n:cfg.d1.kp--> | tak |
| D4 | 0.3<!--n:cfg.d2.kp--> | tak |

Populacja i metoda jak w sekcji 2: 2 721<!--n:a2.shots--> sparowanych strzałów bez karnych, klatka t0, przedziały 95%<!--n:au.ci--> z bootstrapu po meczach, 5 000<!--n:a2.n_boot--> losowań, te same co w audycie 1. C4 z audytu 1 przechodzi audyt razem z D1–D4 jako odniesienie i nie bierze udziału w wyborze. Jej 255<!--n:a2.ref.rows--> wierszy metryk jest identycznych z audytem 1 (różni się 0<!--n:a2.ref.diff-->).

### 10.3 Lokalizacja strzału

| Konfiguracja | pokrycie | mediana błędu | p90 | < 1 m | < 2 m | < 5 m | poza boiskiem |
|---|---|---|---|---|---|---|---|
| D1 | 77.4% [74.3; 80.5]<!--n:a2.d1.cov--> | 10.6 m [9.9; 11.0]<!--n:a2.d1.median--> | 27.5 m<!--n:a2.d1.p90--> | 1.9%<!--n:a2.d1.lt1--> | 6.9%<!--n:a2.d1.lt2--> | 24.4%<!--n:a2.d1.lt5--> | 9.9%<!--n:a2.d1.off--> |
| D2 | 86.5% [84.1; 88.9]<!--n:a2.d2.cov--> | 10.7 m [10.1; 11.1]<!--n:a2.d2.median--> | 27.5 m<!--n:a2.d2.p90--> | 1.8%<!--n:a2.d2.lt1--> | 6.9%<!--n:a2.d2.lt2--> | 23.1%<!--n:a2.d2.lt5--> | 10.1%<!--n:a2.d2.off--> |
| D3 | 77.4% [74.3; 80.5]<!--n:a2.d3.cov--> | 10.5 m [9.9; 11.1]<!--n:a2.d3.median--> | 27.5 m<!--n:a2.d3.p90--> | 4.6%<!--n:a2.d3.lt1--> | 10.7%<!--n:a2.d3.lt2--> | 26.2%<!--n:a2.d3.lt5--> | 8.1%<!--n:a2.d3.off--> |
| D4 | 86.5% [84.1; 88.9]<!--n:a2.d4.cov--> | 10.6 m [10.1; 11.2]<!--n:a2.d4.median--> | 27.5 m<!--n:a2.d4.p90--> | 4.0%<!--n:a2.d4.lt1--> | 10.1%<!--n:a2.d4.lt2--> | 24.9%<!--n:a2.d4.lt5--> | 8.2%<!--n:a2.d4.off--> |
| C4 (audyt 1) | 71.8% [67.3; 76.2]<!--n:a2.c4.cov--> | 12.1 m [11.4; 12.8]<!--n:a2.c4.median--> | 27.2 m<!--n:a2.c4.p90--> | 0.4%<!--n:a2.c4.lt1--> | 1.7%<!--n:a2.c4.lt2--> | 14.5%<!--n:a2.c4.lt5--> | 13.5%<!--n:a2.c4.off--> |

"Poza boiskiem" to udział strzałów z lokalizacją, które homografia umieściła poza prostokątem boiska. Niektóre z nich leżą setki metrów od boiska.

Poprawki pakietu podniosły pokrycie i obniżyły medianę o około półtora metra, a udział błędów < 5 m wzrósł mniej więcej o dziesięć punktów procentowych. Doprecyzowanie liniami (D3, D4) zwiększa udział błędów < 1 m i < 2 m, na przykład z 6.9%<!--n:a2.d1.lt2--> do 10.7%<!--n:a2.d3.lt2--> dla < 2 m, ale nie zmienia mediany. Do progu 3 m<!--n:au.med_max--> nadal jest daleko.

**Według źródła lokalizacji** (Q23) **i pewności instancji** (Q25), jak w sekcji 3:

| Konfiguracja | piłka: n, mediana | flaga: n, mediana | wcześniejsza klatka: n, mediana | instancja ≥ 0.5<!--n:au.kp-->: n, mediana |
|---|---|---|---|---|
| D1 | 1 976<!--n:a2.d1.ball.n-->, 10.5 m<!--n:a2.d1.ball.med--> | 131<!--n:a2.d1.flag.n-->, 11.0 m<!--n:a2.d1.flag.med--> | 730<!--n:a2.d1.fallback.n-->, 6.9 m<!--n:a2.d1.fallback.med--> | 2 084<!--n:a2.d1.kp05.n-->, 10.5 m<!--n:a2.d1.kp05.med--> |
| D2 | 2 214<!--n:a2.d2.ball.n-->, 10.6 m<!--n:a2.d2.ball.med--> | 141<!--n:a2.d2.flag.n-->, 11.1 m<!--n:a2.d2.flag.med--> | 787<!--n:a2.d2.fallback.n-->, 7.3 m<!--n:a2.d2.fallback.med--> | 2 314<!--n:a2.d2.kp05.n-->, 10.6 m<!--n:a2.d2.kp05.med--> |
| D3 | 1 976<!--n:a2.d3.ball.n-->, 10.5 m<!--n:a2.d3.ball.med--> | 131<!--n:a2.d3.flag.n-->, 11.8 m<!--n:a2.d3.flag.med--> | 730<!--n:a2.d3.fallback.n-->, 6.6 m<!--n:a2.d3.fallback.med--> | 2 084<!--n:a2.d3.kp05.n-->, 10.5 m<!--n:a2.d3.kp05.med--> |
| D4 | 2 214<!--n:a2.d4.ball.n-->, 10.6 m<!--n:a2.d4.ball.med--> | 141<!--n:a2.d4.flag.n-->, 11.7 m<!--n:a2.d4.flag.med--> | 787<!--n:a2.d4.fallback.n-->, 7.3 m<!--n:a2.d4.fallback.med--> | 2 314<!--n:a2.d4.kp05.n-->, 10.5 m<!--n:a2.d4.kp05.med--> |
| C4 | 1 833<!--n:a2.c4.ball.n-->, 12.1 m<!--n:a2.c4.ball.med--> | 121<!--n:a2.c4.flag.n-->, 11.9 m<!--n:a2.c4.flag.med--> | 704<!--n:a2.c4.fallback.n-->, 10.4 m<!--n:a2.c4.fallback.med--> | 1 639<!--n:a2.c4.kp05.n-->, 11.3 m<!--n:a2.c4.kp05.med--> |

Lokalizacja z piłki wcześniejszej klatki jest wyraźnie lepsza niż średnio (sekcja 10.7). Próg instancji 0.5<!--n:au.kp--> prawie nie zmienia mediany.

**Kontrola konwencji.** Po odbiciu wykrytego y mediana błędu D3<!--n:a2.chosen--> rośnie do 17.0 m<!--n:a2.d3.mirror-->. Mediana błędu ze znakiem wynosi +0.9 m<!--n:a2.d3.dx--> wzdłuż x i −0.5 m<!--n:a2.d3.dy--> wzdłuż y; w C4 było to +2.1 m<!--n:a2.c4.dx--> i −2.2 m<!--n:a2.c4.dy-->.

### 10.4 Czy wykryte pozycje niosą informację: korelacje i stałe zgadywanie

Dwa sprawdziany, które w audycie 1 były tylko liczbami doraźnymi:
- **korelacja** wykrytych współrzędnych ze współrzędnymi StatsBomb na tych samych strzałach. Pearson reaguje na kilka pozycji wyrzuconych daleko poza boisko, Spearman (na rangach) nie, więc podajemy oba. W bootstrapie Spearman liczy rangi w każdym losowaniu od nowa;
- **stałe zgadywanie**: jedna pozycja dla wszystkich strzałów, mediana pozycji StatsBomb w całej populacji, oceniana na dokładnie tych strzałach, które porównujemy dla detektora. To wyrocznia: zna rozkład prawdziwych pozycji. Detektor gorszy od niej nie mówi o pozycji nic ponad typowe miejsce.

| Konfiguracja | strzał: Pearson x | Pearson y | Spearman x | Spearman y | mediana błędu: detektor | stała |
|---|---|---|---|---|---|---|
| D1 | 0.24 [0.16; 0.32]<!--n:a2.d1.corr.x--> | 0.20 [0.15; 0.28]<!--n:a2.d1.corr.y--> | 0.41 [0.37; 0.45]<!--n:a2.d1.rank.x--> | 0.46 [0.42; 0.50]<!--n:a2.d1.rank.y--> | 10.6 m [9.9; 11.0]<!--n:a2.d1.median--> | 10.7 m [10.4; 11.0]<!--n:a2.d1.const.med--> |
| D2 | 0.28 [0.20; 0.36]<!--n:a2.d2.corr.x--> | 0.25 [0.21; 0.30]<!--n:a2.d2.corr.y--> | 0.40 [0.36; 0.44]<!--n:a2.d2.rank.x--> | 0.43 [0.39; 0.46]<!--n:a2.d2.rank.y--> | 10.7 m [10.1; 11.1]<!--n:a2.d2.median--> | 10.5 m [10.2; 10.8]<!--n:a2.d2.const.med--> |
| D3 | 0.05 [0.01; 0.18]<!--n:a2.d3.corr.x--> | 0.09 [0.07; 0.17]<!--n:a2.d3.corr.y--> | 0.41 [0.37; 0.45]<!--n:a2.d3.rank.x--> | 0.46 [0.42; 0.50]<!--n:a2.d3.rank.y--> | 10.5 m [9.9; 11.1]<!--n:a2.d3.median--> | 10.7 m [10.4; 11.0]<!--n:a2.d3.const.med--> |
| D4 | 0.02 [0.00; 0.14]<!--n:a2.d4.corr.x--> | 0.02 [0.02; 0.13]<!--n:a2.d4.corr.y--> | 0.42 [0.38; 0.45]<!--n:a2.d4.rank.x--> | 0.43 [0.39; 0.46]<!--n:a2.d4.rank.y--> | 10.6 m [10.1; 11.2]<!--n:a2.d4.median--> | 10.5 m [10.2; 10.8]<!--n:a2.d4.const.med--> |
| C4 | 0.07 [0.04; 0.41]<!--n:a2.c4.corr.x--> | 0.04 [−0.02; 0.35]<!--n:a2.c4.corr.y--> | 0.43 [0.40; 0.46]<!--n:a2.c4.rank.x--> | 0.42 [0.38; 0.46]<!--n:a2.c4.rank.y--> | 12.1 m [11.4; 12.8]<!--n:a2.c4.median--> | 10.6 m [10.3; 11.0]<!--n:a2.c4.const.med--> |

| Konfiguracja | bramkarz: Spearman x | Spearman y | mediana błędu: detektor | stała | < 2 m: detektor | stała |
|---|---|---|---|---|---|---|
| D1 | −0.00 [−0.06; 0.05]<!--n:a2.d1.gk.rank.x--> | 0.25 [0.21; 0.30]<!--n:a2.d1.gk.rank.y--> | 8.2 m [7.7; 8.8]<!--n:a2.d1.gk.med--> | 1.9 m [1.8; 2.1]<!--n:a2.d1.gkconst.med--> | 8.6%<!--n:a2.d1.gk.lt2--> | 51.5%<!--n:a2.d1.gkconst.lt2--> |
| D2 | −0.01 [−0.06; 0.04]<!--n:a2.d2.gk.rank.x--> | 0.22 [0.17; 0.26]<!--n:a2.d2.gk.rank.y--> | 8.3 m [7.8; 8.7]<!--n:a2.d2.gk.med--> | 2.0 m [1.9; 2.1]<!--n:a2.d2.gkconst.med--> | 7.8%<!--n:a2.d2.gk.lt2--> | 51.1%<!--n:a2.d2.gkconst.lt2--> |
| D3 | 0.00 [−0.06; 0.06]<!--n:a2.d3.gk.rank.x--> | 0.25 [0.19; 0.29]<!--n:a2.d3.gk.rank.y--> | 8.6 m [8.1; 9.1]<!--n:a2.d3.gk.med--> | 1.9 m [1.8; 2.1]<!--n:a2.d3.gkconst.med--> | 16.7%<!--n:a2.d3.gk.lt2--> | 51.5%<!--n:a2.d3.gkconst.lt2--> |
| D4 | 0.01 [−0.05; 0.06]<!--n:a2.d4.gk.rank.x--> | 0.22 [0.18; 0.26]<!--n:a2.d4.gk.rank.y--> | 8.5 m [8.1; 9.0]<!--n:a2.d4.gk.med--> | 2.0 m [1.9; 2.1]<!--n:a2.d4.gkconst.med--> | 16.1%<!--n:a2.d4.gk.lt2--> | 51.1%<!--n:a2.d4.gkconst.lt2--> |
| C4 | 0.01 [−0.05; 0.07]<!--n:a2.c4.gk.rank.x--> | 0.34 [0.29; 0.38]<!--n:a2.c4.gk.rank.y--> | 6.9 m [6.4; 7.4]<!--n:a2.c4.gk.med--> | 2.0 m [1.9; 2.1]<!--n:a2.c4.gkconst.med--> | 7.6%<!--n:a2.c4.gk.lt2--> | 50.5%<!--n:a2.c4.gkconst.lt2--> |

- **Miejsce strzału.** Korelacja rang jest umiarkowana w obu osiach, więc wykryta lokalizacja niesie część informacji. Pearson z audytu 1 (0.07 [0.04; 0.41]<!--n:a2.c4.corr.x--> i 0.04 [−0.02; 0.35]<!--n:a2.c4.corr.y--> dla C4) zaniżał ją przez pozycje poza boiskiem. W D1–D4 mediana błędu detektora jest jednak praktycznie równa medianie stałego zgadywania, a w C4 była od niej wyższa. W medianie wykryta lokalizacja nie jest więc bliżej prawdy niż typowe miejsce strzału.
- **Bramkarz.** Stałe zgadywanie jest trzy do czterech razy dokładniejsze niż detektor. Bramkarz stoi zwykle blisko środka bramki, a błąd detektora pochodzi prawdopodobnie z homografii i z wyboru zawodnika (bramkarz zastępczy nie zawsze jest bramkarzem). Pozycja x bramkarza nie koreluje z prawdą wcale. W D1–D4 błąd bramkarza jest większy niż w C4, tak jak w diagnostyce C4s z sekcji 7.

### 10.5 Bramkarz

| Konfiguracja | pokrycie | mediana | < 2 m | wykryty: n, mediana | zastępczy, ta sama drużyna: n, mediana | zastępczy, inna: n, mediana | instancja ≥ 0.5<!--n:au.kp-->: mediana |
|---|---|---|---|---|---|---|---|
| D1 | 66.6%<!--n:a2.d1.gk.cov--> | 8.2 m [7.7; 8.8]<!--n:a2.d1.gk.med--> | 8.6%<!--n:a2.d1.gk.lt2--> | 298<!--n:a2.d1.gkd.n-->, 14.7 m<!--n:a2.d1.gkd.med--> | 824<!--n:a2.d1.same.n-->, 7.5 m<!--n:a2.d1.same.med--> | 688<!--n:a2.d1.other.n-->, 7.9 m<!--n:a2.d1.other.med--> | 8.2 m<!--n:a2.d1.gkkp.med--> |
| D2 | 76.0%<!--n:a2.d2.gk.cov--> | 8.3 m [7.8; 8.7]<!--n:a2.d2.gk.med--> | 7.8%<!--n:a2.d2.gk.lt2--> | 349<!--n:a2.d2.gkd.n-->, 14.9 m<!--n:a2.d2.gkd.med--> | 927<!--n:a2.d2.same.n-->, 7.6 m<!--n:a2.d2.same.med--> | 791<!--n:a2.d2.other.n-->, 7.7 m<!--n:a2.d2.other.med--> | 8.2 m<!--n:a2.d2.gkkp.med--> |
| D3 | 66.5%<!--n:a2.d3.gk.cov--> | 8.6 m [8.1; 9.1]<!--n:a2.d3.gk.med--> | 16.7%<!--n:a2.d3.gk.lt2--> | 298<!--n:a2.d3.gkd.n-->, 14.8 m<!--n:a2.d3.gkd.med--> | 819<!--n:a2.d3.same.n-->, 8.2 m<!--n:a2.d3.same.med--> | 690<!--n:a2.d3.other.n-->, 8.2 m<!--n:a2.d3.other.med--> | 8.6 m<!--n:a2.d3.gkkp.med--> |
| D4 | 76.0%<!--n:a2.d4.gk.cov--> | 8.5 m [8.1; 9.0]<!--n:a2.d4.gk.med--> | 16.1%<!--n:a2.d4.gk.lt2--> | 349<!--n:a2.d4.gkd.n-->, 15.2 m<!--n:a2.d4.gkd.med--> | 937<!--n:a2.d4.same.n-->, 8.0 m<!--n:a2.d4.same.med--> | 779<!--n:a2.d4.other.n-->, 8.1 m<!--n:a2.d4.other.med--> | 8.4 m<!--n:a2.d4.gkkp.med--> |
| C4 | 60.2%<!--n:a2.c4.gk.cov--> | 6.9 m [6.4; 7.4]<!--n:a2.c4.gk.med--> | 7.6%<!--n:a2.c4.gk.lt2--> | 274<!--n:a2.c4.gkd.n-->, 12.1 m<!--n:a2.c4.gkd.med--> | 695<!--n:a2.c4.same.n-->, 5.5 m<!--n:a2.c4.same.med--> | 666<!--n:a2.c4.other.n-->, 6.5 m<!--n:a2.c4.other.med--> | 6.9 m<!--n:a2.c4.gkkp.med--> |

**Według źródła lokalizacji** (Q23):

| Konfiguracja | bramkarz, piłka: n, mediana | bramkarz, flaga: n, mediana | stożek, piłka: n, dokładna, ≥ 1 | stożek, flaga: n, dokładna, ≥ 1 |
|---|---|---|---|---|
| D1 | 1 684<!--n:a2.d1.gk.ball.n-->, 8.0 m<!--n:a2.d1.gk.ball.med--> | 126<!--n:a2.d1.gk.flag.n-->, 11.5 m<!--n:a2.d1.gk.flag.med--> | 1 687<!--n:a2.d1.cone.ball.n-->, 48.2%<!--n:a2.d1.cone.ball.exact-->, 51.6%<!--n:a2.d1.cone.ball.ge1--> | 125<!--n:a2.d1.cone.flag.n-->, 49.6%<!--n:a2.d1.cone.flag.exact-->, 49.6%<!--n:a2.d1.cone.flag.ge1--> |
| D2 | 1 931<!--n:a2.d2.gk.ball.n-->, 8.0 m<!--n:a2.d2.gk.ball.med--> | 136<!--n:a2.d2.gk.flag.n-->, 11.4 m<!--n:a2.d2.gk.flag.med--> | 1 933<!--n:a2.d2.cone.ball.n-->, 48.6%<!--n:a2.d2.cone.ball.exact-->, 52.3%<!--n:a2.d2.cone.ball.ge1--> | 135<!--n:a2.d2.cone.flag.n-->, 54.1%<!--n:a2.d2.cone.flag.exact-->, 54.1%<!--n:a2.d2.cone.flag.ge1--> |
| D3 | 1 681<!--n:a2.d3.gk.ball.n-->, 8.3 m<!--n:a2.d3.gk.ball.med--> | 126<!--n:a2.d3.gk.flag.n-->, 11.6 m<!--n:a2.d3.gk.flag.med--> | 1 687<!--n:a2.d3.cone.ball.n-->, 47.8%<!--n:a2.d3.cone.ball.exact-->, 51.4%<!--n:a2.d3.cone.ball.ge1--> | 125<!--n:a2.d3.cone.flag.n-->, 49.6%<!--n:a2.d3.cone.flag.exact-->, 49.6%<!--n:a2.d3.cone.flag.ge1--> |
| D4 | 1 929<!--n:a2.d4.gk.ball.n-->, 8.3 m<!--n:a2.d4.gk.ball.med--> | 136<!--n:a2.d4.gk.flag.n-->, 11.4 m<!--n:a2.d4.gk.flag.med--> | 1 933<!--n:a2.d4.cone.ball.n-->, 48.4%<!--n:a2.d4.cone.ball.exact-->, 51.9%<!--n:a2.d4.cone.ball.ge1--> | 135<!--n:a2.d4.cone.flag.n-->, 54.1%<!--n:a2.d4.cone.flag.exact-->, 54.1%<!--n:a2.d4.cone.flag.ge1--> |
| C4 | 1 518<!--n:a2.c4.gk.ball.n-->, 6.7 m<!--n:a2.c4.gk.ball.med--> | 117<!--n:a2.c4.gk.flag.n-->, 8.6 m<!--n:a2.c4.gk.flag.med--> | 1 522<!--n:a2.c4.cone.ball.n-->, 46.6%<!--n:a2.c4.cone.ball.exact-->, 52.8%<!--n:a2.c4.cone.ball.ge1--> | 116<!--n:a2.c4.cone.flag.n-->, 52.6%<!--n:a2.c4.cone.flag.exact-->, 53.4%<!--n:a2.c4.cone.flag.ge1--> |

Przedziały tych podzbiorów są w `results/xgd_audit2/metrics.csv`.

### 10.6 Obrońcy w stożku strzału

| Konfiguracja | n | zgodność dokładna | zgodność ≥ 1 | zgodność ≥ 2 | średnia różnica (detektor − StatsBomb) |
|---|---|---|---|---|---|
| D1 | 1 812<!--n:a2.d1.cone.n--> | 48.3% [46.0; 50.5]<!--n:a2.d1.cone.exact--> | 51.4%<!--n:a2.d1.cone.ge1--> | 84.1%<!--n:a2.d1.cone.ge2--> | −0.54<!--n:a2.d1.cone.diff--> |
| D2 | 2 068<!--n:a2.d2.cone.n--> | 49.0% [46.8; 51.1]<!--n:a2.d2.cone.exact--> | 52.4%<!--n:a2.d2.cone.ge1--> | 84.2%<!--n:a2.d2.cone.ge2--> | −0.54<!--n:a2.d2.cone.diff--> |
| D3 | 1 812<!--n:a2.d3.cone.n--> | 48.0% [45.6; 50.3]<!--n:a2.d3.cone.exact--> | 51.3%<!--n:a2.d3.cone.ge1--> | 84.4%<!--n:a2.d3.cone.ge2--> | −0.54<!--n:a2.d3.cone.diff--> |
| D4 | 2 068<!--n:a2.d4.cone.n--> | 48.7% [46.5; 50.9]<!--n:a2.d4.cone.exact--> | 52.0%<!--n:a2.d4.cone.ge1--> | 84.4%<!--n:a2.d4.cone.ge2--> | −0.55<!--n:a2.d4.cone.diff--> |
| C4 | 1 638<!--n:a2.c4.cone.n--> | 47.1% [44.6; 49.4]<!--n:a2.c4.cone.exact--> | 52.8%<!--n:a2.c4.cone.ge1--> | 82.7%<!--n:a2.c4.cone.ge2--> | −0.36<!--n:a2.c4.cone.diff--> |

Macierz pomyłek D3<!--n:a2.chosen--> dla "co najmniej jeden obrońca w stożku":

| | StatsBomb: brak | StatsBomb: ≥ 1 |
|---|---|---|
| detektor: brak | 804<!--n:a2.cf.00--> | 764<!--n:a2.cf.01--> |
| detektor: ≥ 1 | 119<!--n:a2.cf.10--> | 125<!--n:a2.cf.11--> |

Zgodność stożka prawie się nie zmieniła. Jak w audycie 1, detektor częściej nie widzi obrońcy, którego StatsBomb ma w stożku, niż odwrotnie.

**Szum czasowy (opisowo).** Mediana błędu lokalizacji D3<!--n:a2.chosen--> według |rezydualnego czasu| parowania: do 50 ms 10.3 m<!--n:a2.tm.t1.med--> (1 674<!--n:a2.tm.t1.n--> strzałów), 50 ms do 200 ms 11.5 m<!--n:a2.tm.t2.med--> (377<!--n:a2.tm.t2.n-->), ponad 200 ms 13.1 m<!--n:a2.tm.t3.med--> (56<!--n:a2.tm.t3.n-->).

### 10.7 Skąd błąd (opisowo, po wyborze)

Ten podział nie wpływa na wybór ani na bramkę i nie dodaje konfiguracji. Skrypt `scripts/xgd_audit2_diag.py` dzieli błędy z audytu 2 z tymi samymi losowaniami.

**Doprecyzowanie liniami.** D3 i D4 mają te same punkty kluczowe co D1 i D2, więc homografię t0 można przypisać do jednej z trzech grup. Grupa "zmieniona" to te, które doprecyzowanie zmieniło, czyli linie boiska ją potwierdziły. Grupa "bez zmian" to te, w których doprecyzowanie wróciło do dopasowania z punktów. Trzecia grupa nie ma homografii t0 (lokalizacja może wtedy pochodzić z piłki wcześniejszej klatki). Każdą grupę porównujemy na tych samych strzałach:

| Homografia t0 (D3 wobec D1) | udział strzałów | strzały z lokalizacją | mediana błędu D1 | mediana błędu D3 | bramkarz D1 | bramkarz D3 |
|---|---|---|---|---|---|---|
| zmieniona | 37.5% [33.8; 41.3]<!--n:dg.d3.refined.share--> | 1 020<!--n:dg.d3.refined.n--> | 10.3 m [9.5; 10.9]<!--n:dg.d3.refined.d1.med--> | 10.3 m [9.2; 11.1]<!--n:dg.d3.refined.d3.med--> | 7.4 m<!--n:dg.d3.refined.d1.gk--> | 8.0 m<!--n:dg.d3.refined.d3.gk--> |
| bez zmian | 29.4% [26.2; 32.6]<!--n:dg.d3.kept.share--> | 799<!--n:dg.d3.kept.n--> | 11.2 m [10.5; 12.0]<!--n:dg.d3.kept.d1.med--> | 11.2 m [10.5; 12.0]<!--n:dg.d3.kept.d3.med--> | 9.1 m<!--n:dg.d3.kept.d1.gk--> | 9.1 m<!--n:dg.d3.kept.d3.gk--> |
| brak homografii t0 | 33.1% [29.7; 36.7]<!--n:dg.d3.no_fit.share--> | 288<!--n:dg.d3.no_fit.n--> | 9.2 m [7.6; 10.6]<!--n:dg.d3.no_fit.d1.med--> | 8.9 m [7.4; 10.5]<!--n:dg.d3.no_fit.d3.med--> | — | — |

| Homografia t0 (D4 wobec D2) | udział strzałów | strzały z lokalizacją | mediana błędu D2 | mediana błędu D4 | bramkarz D2 | bramkarz D4 |
|---|---|---|---|---|---|---|
| zmieniona | 42.2% [38.3; 46.2]<!--n:dg.d4.refined.share--> | 1 148<!--n:dg.d4.refined.n--> | 10.0 m [9.2; 10.9]<!--n:dg.d4.refined.d2.med--> | 10.0 m [9.1; 10.8]<!--n:dg.d4.refined.d4.med--> | 7.2 m<!--n:dg.d4.refined.d2.gk--> | 7.7 m<!--n:dg.d4.refined.d4.gk--> |
| bez zmian | 34.1% [30.7; 37.6]<!--n:dg.d4.kept.share--> | 927<!--n:dg.d4.kept.n--> | 11.3 m [10.6; 12.0]<!--n:dg.d4.kept.d2.med--> | 11.3 m [10.6; 12.0]<!--n:dg.d4.kept.d4.med--> | 9.3 m<!--n:dg.d4.kept.d2.gk--> | 9.2 m<!--n:dg.d4.kept.d4.gk--> |
| brak homografii t0 | 23.7% [20.6; 27.2]<!--n:dg.d4.no_fit.share--> | 280<!--n:dg.d4.no_fit.n--> | 9.7 m [8.6; 11.3]<!--n:dg.d4.no_fit.d2.med--> | 9.4 m [8.4; 11.4]<!--n:dg.d4.no_fit.d4.med--> | — | — |

Grupy wyznacza homografia t0. Strzał umieszczony piłką wcześniejszej klatki ma homografię tamtej klatki, zmienioną albo nie.

Tam, gdzie linie potwierdzają doprecyzowaną homografię, błąd lokalizacji jest taki sam jak bez doprecyzowania, a błąd bramkarza trochę większy. Lepsza zgodność z liniami nie przekłada się więc na dokładniejsze pozycje w audycie. Pozostały błąd nie pochodzi głównie z dopasowania homografii.

**Źródło lokalizacji.** Skąd pochodzi lokalizacja strzału w D3<!--n:a2.chosen--> i w C4:

| Źródło | D3<!--n:a2.chosen-->: udział, n | D3<!--n:a2.chosen-->: mediana błędu | D3<!--n:a2.chosen-->: mediana błędu x | C4: n | C4: mediana błędu |
|---|---|---|---|---|---|
| piłka w t0 | 45.8%<!--n:dg.src.d3.ball_t0.share-->, 1 246<!--n:dg.src.d3.ball_t0.n--> | 11.9 m [11.3; 12.6]<!--n:dg.src.d3.ball_t0.med--> | +1.6 m<!--n:dg.src.d3.ball_t0.dx--> | 1 129<!--n:dg.src.c4.ball_t0.n--> | 13.6 m [12.8; 14.6]<!--n:dg.src.c4.ball_t0.med--> |
| piłka z −400 ms | 18.6%<!--n:dg.src.d3.ball_m400.share-->, 505<!--n:dg.src.d3.ball_m400.n--> | 6.0 m [5.4; 7.0]<!--n:dg.src.d3.ball_m400.med--> | −0.0 m<!--n:dg.src.d3.ball_m400.dx--> | 512<!--n:dg.src.c4.ball_m400.n--> | 9.4 m [8.7; 10.4]<!--n:dg.src.c4.ball_m400.med--> |
| piłka z −800 ms | 8.3%<!--n:dg.src.d3.ball_m800.share-->, 225<!--n:dg.src.d3.ball_m800.n--> | 8.9 m [7.2; 10.6]<!--n:dg.src.d3.ball_m800.med--> | −1.0 m<!--n:dg.src.d3.ball_m800.dx--> | 192<!--n:dg.src.c4.ball_m800.n--> | 12.2 m [10.8; 15.0]<!--n:dg.src.c4.ball_m800.med--> |
| strzelec z flagi | 4.8%<!--n:dg.src.d3.flag.share-->, 131<!--n:dg.src.d3.flag.n--> | 11.8 m [8.9; 12.7]<!--n:dg.src.d3.flag.med--> | +6.4 m<!--n:dg.src.d3.flag.dx--> | 121<!--n:dg.src.c4.flag.n--> | 11.9 m [10.7; 14.7]<!--n:dg.src.c4.flag.med--> |

Piłka z klatki −400 ms daje mniej więcej o połowę mniejszy błąd niż piłka w t0, w tym samym pipeline (homografia tamtej klatki, te same ustawienia). Wygląda więc na to, że pipeline umieszcza piłkę znacznie bliżej miejsca strzału, gdy piłka jest jeszcze przy strzelcu. Piłka w t0 leży średnio bliżej bramki niż miejsce strzału StatsBomb (dodatni błąd x), co pasuje do piłki już lecącej albo do innego wykrycia. To obserwacja, nie dowód. Grupa −400 ms jest wybrana: wchodzą do niej tylko klatki t0 bez piłki na boisku, a więc inne ujęcia.

### 10.8 Pełna ekstrakcja D3<!--n:a2.chosen-->

D3<!--n:a2.chosen--> przeszła po wszystkich strzałach indeksu w t0 i −400 ms. Klatkę −800 ms analizujemy tylko po piłkę. Przebieg trwał 55 min<!--n:fu2.seconds--> (301 ms<!--n:fu2.ms--> na strzał) i przeanalizował 26 615<!--n:fu2.frames--> klatek. Cechy t0 porównaliśmy z pamięcią audytu 2 na 2 754<!--n:fu2.ref--> sparowanych strzałach: różnią się w 0<!--n:fu2.ref_diff--> strzałach. Z tych cech korzysta podejście 1 (ticket 03), także po wyniku STOP<!--n:a2.gate-->.

Pokrycie według roli w planie podziału xg-vendor (udział strzałów):

| Rola | klatka | strzały | mecze | homografia | piłka | lokalizacja strzału | strzelec z flagi | bramkarz | stożek |
|---|---|---|---|---|---|---|---|---|---|
| trening | t0 | 7 166<!--n:fu2.training.t0.shots--> | 328<!--n:fu2.training.t0.matches--> | 71.5%<!--n:fu2.training.t0.homography_ok--> | 76.8%<!--n:fu2.training.t0.ball_ok--> | 81.4%<!--n:fu2.training.t0.shot_ok--> | 4.6%<!--n:fu2.training.t0.shooter_by_goal--> | 71.2%<!--n:fu2.training.t0.gk_ok--> | 71.3%<!--n:fu2.training.t0.def_cone--> |
| trening | −400 ms | 7 166<!--n:fu2.training.m400.shots--> | 328<!--n:fu2.training.m400.matches--> | 72.7%<!--n:fu2.training.m400.homography_ok--> | 72.1%<!--n:fu2.training.m400.ball_ok--> | 78.3%<!--n:fu2.training.m400.shot_ok--> | 6.2%<!--n:fu2.training.m400.shooter_by_goal--> | 72.5%<!--n:fu2.training.m400.gk_ok--> | 72.5%<!--n:fu2.training.m400.def_cone--> |
| bramka | t0 | 825<!--n:fu2.gate.t0.shots--> | 37<!--n:fu2.gate.t0.matches--> | 68.1%<!--n:fu2.gate.t0.homography_ok--> | 78.3%<!--n:fu2.gate.t0.ball_ok--> | 80.7%<!--n:fu2.gate.t0.shot_ok--> | 2.4%<!--n:fu2.gate.t0.shooter_by_goal--> | 68.1%<!--n:fu2.gate.t0.gk_ok--> | 68.1%<!--n:fu2.gate.t0.def_cone--> |
| bramka | −400 ms | 825<!--n:fu2.gate.m400.shots--> | 37<!--n:fu2.gate.m400.matches--> | 71.9%<!--n:fu2.gate.m400.homography_ok--> | 73.6%<!--n:fu2.gate.m400.ball_ok--> | 78.2%<!--n:fu2.gate.m400.shot_ok--> | 4.6%<!--n:fu2.gate.m400.shooter_by_goal--> | 71.9%<!--n:fu2.gate.m400.gk_ok--> | 71.9%<!--n:fu2.gate.m400.def_cone--> |
| ewaluacja | t0 | 3 039<!--n:fu2.evaluation.t0.shots--> | 135<!--n:fu2.evaluation.t0.matches--> | 65.7%<!--n:fu2.evaluation.t0.homography_ok--> | 71.7%<!--n:fu2.evaluation.t0.ball_ok--> | 76.5%<!--n:fu2.evaluation.t0.shot_ok--> | 4.8%<!--n:fu2.evaluation.t0.shooter_by_goal--> | 65.3%<!--n:fu2.evaluation.t0.gk_ok--> | 65.5%<!--n:fu2.evaluation.t0.def_cone--> |
| ewaluacja | −400 ms | 3 039<!--n:fu2.evaluation.m400.shots--> | 135<!--n:fu2.evaluation.m400.matches--> | 66.8%<!--n:fu2.evaluation.m400.homography_ok--> | 66.8%<!--n:fu2.evaluation.m400.ball_ok--> | 73.2%<!--n:fu2.evaluation.m400.shot_ok--> | 6.4%<!--n:fu2.evaluation.m400.shooter_by_goal--> | 66.4%<!--n:fu2.evaluation.m400.gk_ok--> | 66.5%<!--n:fu2.evaluation.m400.def_cone--> |
| wszystkie | t0 | 11 030<!--n:fu2.all.t0.shots--> | 500<!--n:fu2.all.t0.matches--> | 69.6%<!--n:fu2.all.t0.homography_ok--> | 75.5%<!--n:fu2.all.t0.ball_ok--> | 80.0%<!--n:fu2.all.t0.shot_ok--> | 4.5%<!--n:fu2.all.t0.shooter_by_goal--> | 69.4%<!--n:fu2.all.t0.gk_ok--> | 69.4%<!--n:fu2.all.t0.def_cone--> |
| wszystkie | −400 ms | 11 030<!--n:fu2.all.m400.shots--> | 500<!--n:fu2.all.m400.matches--> | 71.0%<!--n:fu2.all.m400.homography_ok--> | 70.7%<!--n:fu2.all.m400.ball_ok--> | 76.9%<!--n:fu2.all.m400.shot_ok--> | 6.2%<!--n:fu2.all.m400.shooter_by_goal--> | 70.7%<!--n:fu2.all.m400.gk_ok--> | 70.8%<!--n:fu2.all.m400.def_cone--> |

### 10.9 Co z tego wynika

1. **Bramka STOP<!--n:a2.gate-->, ostatecznie.** Podejście 2 (ticket 05) kończy się na tym raporcie. Trzeciego audytu nie będzie (Q26).
2. **Podejście 1 na cechach D3<!--n:a2.chosen-->.** Mają więcej pokrycia niż C4, ale w medianie lokalizacja strzału nie jest dokładniejsza od stałego zgadywania, a bramkarz jest dużo gorszy od niego. Umiarkowana korelacja rang zostawia w cechach trochę sygnału. Modele dostaną je z flagami pokrycia.
3. **Dalsza praca nad produktem (poza tym badaniem).** Sekcja 10.7 pokazuje, że po poprawkach z ticketu 08 lepsza zgodność homografii z liniami boiska sama nie zmniejsza błędu audytu. Większy wpływ może mieć wybór klatki i piłki: piłka sprzed t0 jest bliżej miejsca strzału. Nowy model punktów kluczowych mógłby nie wystarczyć.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `cache/xgd_extract/C1/complete.json` | `b112ff22432883ba8526509ea775b9622c09230ab2b9404e93d8a3fe5e8b5aec` | rekord ukończenia |
| `cache/xgd_extract/C3/complete.json` | `6e228ae0bc7f6c99c5d3770f953ecaba65a567b1fa0bd0bed8dff4bb6ea95b1b` | rekord ukończenia |
| `cache/xgd_extract/C4/complete.json` | `e2e47c3e2cd55f7d2d54fda56b8a309ddafb5060e34dd625895cd12c3fa580c5` | rekord ukończenia |
| `cache/xgd_extract2/D1/complete.json` | `e1988d257d4cbf55de9304714b34f5c29f663e3a3c6433d917bc113708efa225` | rekord ukończenia |
| `cache/xgd_extract2/D2/complete.json` | `ab0127fd24842bb3bf7a88f7a1fadd1b7534b7d44dfc40b3ae3fdf205cf8cfff` | rekord ukończenia |
| `cache/xgd_full/C4/complete.json` | `effe9a665a944276d488268bdc1c4f64ecb09eaf27780f22e305bb374c083a28` | rekord ukończenia |
| `cache/xgd_full2/D3/complete.json` | `2f35b5361d178e9360993653b4fd833310842fc83ceb5efda0445da5d86b3ca1` | rekord ukończenia |
| `research/xg_detector_audit2_configs.json` | `b7bfcffb8e36d22c23eca13a98eee887fcd1bfe6aa36435ba47a3a8f30cf4448` | deklaracja D1–D4 (ticket 08) |
| `results/xgd_audit/choice.json` | `0e7bc0b9910bd32deeef34628cf2bfa4aa2817f9bb198c51483bad7b6dd56f71` | wybór konfiguracji i bramka |
| `results/xgd_audit/complete.json` | `876e4a7717487b6c4db58d7d500aafeca15c03cbced397a81f029afc21f97e15` | rekord ukończenia |
| `results/xgd_audit/cone_confusion.csv` | `dbdf6ef578a8a174ae487c26c49219f023c0edc8a51c02d82b322ee043fadaa3` | macierze pomyłek stożka |
| `results/xgd_audit/metrics.csv` | `094eb796175da304a1e650f2aaa09500d08878d2562887ea2f56788137077280` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `results/xgd_audit/timing.csv` | `d2bc32453c5ead72aeffeda31c0709e037d6f7b1651a4c9ae2561bceb84a3e45` | błędy według rezyduum czasu |
| `results/xgd_audit2/choice.json` | `0d5d8275322ced76a87bdc29b590e1fa97ec1c96c9d82cf4378ccd227f5432c9` | wybór konfiguracji i bramka |
| `results/xgd_audit2/complete.json` | `6e5da126ab55d46197edabe47eda34288ae7510ec842ff07947b7ee8e8829641` | rekord ukończenia |
| `results/xgd_audit2/cone_confusion.csv` | `c288007efbd8c6acb5f7e9fb3bf0d6bb1a70d4639a574e984a72478fe9291c2a` | macierze pomyłek stożka |
| `results/xgd_audit2/metrics.csv` | `e422a0a2311a924a4b4e2f2ed895b264360437f8bd2ddf93fd37e156722420ab` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `results/xgd_audit2/timing.csv` | `9ade735a5e5c4cdd15aa207a036cecef070059d03646452a6ccb051255a13311` | błędy według rezyduum czasu |
| `results/xgd_audit2_diag/complete.json` | `60360dc69d2a3aeceeec77fa488789986527a36ea175eaec89134041444cbef3` | rekord ukończenia |
| `results/xgd_audit2_diag/diag.csv` | `32f45780139e42676b25fd81417d5bd9c2ef5f22b88995af9e59b2a47bd1776e` | podziały błędów audytu 2 (opisowo) |
| `results/xgd_full/coverage_by_role.csv` | `7bda2b30e3e85a96db88131e8340808ff49a7f810a2ce0fe45955f593e7fe12c` | pokrycie według roli i klatki |
| `results/xgd_full2/coverage_by_role.csv` | `785fd8815cc8e78c03dafb61c16a1991662a67442fe0cafdb04f58e7d865b7ed` | pokrycie według roli i klatki |
| `results/xgd_kp_diag/complete.json` | `c271936e2e47d55bce966f6af7de2c44322265398872c2c162f35c1a5c62f2d4` | rekord ukończenia |
| `results/xgd_kp_diag/metrics.csv` | `e6680930f8248ff390b1d1c0cf4b3498ac400313b77a1d49e35b28ecebe859d3` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `xgpoc/detector/audit.py` | `c40fe17f84549d7b51f5805791e57ee7f76c09a8e08d52fe3c202e25383b6e14` | kod audytu: poziom przedziałów |
