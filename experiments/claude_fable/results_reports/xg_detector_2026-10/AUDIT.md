# Audyt pozycji z detektora wobec StatsBomb (xg-detector, ticket 02)

Jakub Eichner, 2026-10-12. Gałąź `research/xg-detector`, badanie xg-detector, ticket 02.

Data: StatsBomb. Pozycje odniesienia (miejsce strzału, bramkarz, obrońcy) pochodzą z freeze frames w otwartych danych StatsBomb.

Audyt mierzy, jak dokładnie pipeline detektora umieszcza na boisku miejsce strzału, bramkarza i obrońców. To pomiar diagnostyczny. Żaden model nie dostaje danych StatsBomb, ani jako wejścia, ani jako celu. Audyt nie czyta żadnej kolumny z golem ani wynikiem strzału. Wybór konfiguracji i bramka widzą tylko pokrycie i błędy pozycji.

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

## 9. Decyzje dla właściciela

1. **Bramka STOP<!--n:au.gate-->.** Podejście 2 kończy się na tym raporcie, chyba że właściciel zmieni specyfikację.
2. **Wejście punktów kluczowych.** Pakiet podaje modelowi letterbox, a model uczono na klatkach rozciągniętych. To błąd produktu niezależny od tego badania. Poprawka sama nie przechodzi bramki (sekcja 7).
3. **Podejście 1 z C4.** Cechy C4 mają duży błąd pozycji, a modele dostaną je z flagami pokrycia. Właściciel decyduje, czy podejście 1 rusza na tych cechach, czy czeka na lepszą homografię (`line-refined-homography`, poprawka wejścia, inny model punktów kluczowych). Lepsza homografia wymagałaby nowego zestawu konfiguracji zadeklarowanego przed ponownym audytem.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `cache/xgd_extract/C1/complete.json` | `b112ff22432883ba8526509ea775b9622c09230ab2b9404e93d8a3fe5e8b5aec` | rekord ukończenia |
| `cache/xgd_extract/C3/complete.json` | `6e228ae0bc7f6c99c5d3770f953ecaba65a567b1fa0bd0bed8dff4bb6ea95b1b` | rekord ukończenia |
| `cache/xgd_extract/C4/complete.json` | `e2e47c3e2cd55f7d2d54fda56b8a309ddafb5060e34dd625895cd12c3fa580c5` | rekord ukończenia |
| `cache/xgd_full/C4/complete.json` | `effe9a665a944276d488268bdc1c4f64ecb09eaf27780f22e305bb374c083a28` | rekord ukończenia |
| `results/xgd_audit/choice.json` | `0e7bc0b9910bd32deeef34628cf2bfa4aa2817f9bb198c51483bad7b6dd56f71` | wybór konfiguracji i bramka |
| `results/xgd_audit/complete.json` | `876e4a7717487b6c4db58d7d500aafeca15c03cbced397a81f029afc21f97e15` | rekord ukończenia |
| `results/xgd_audit/cone_confusion.csv` | `dbdf6ef578a8a174ae487c26c49219f023c0edc8a51c02d82b322ee043fadaa3` | macierze pomyłek stożka |
| `results/xgd_audit/metrics.csv` | `094eb796175da304a1e650f2aaa09500d08878d2562887ea2f56788137077280` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `results/xgd_audit/timing.csv` | `d2bc32453c5ead72aeffeda31c0709e037d6f7b1651a4c9ae2561bceb84a3e45` | błędy według rezyduum czasu |
| `results/xgd_full/coverage_by_role.csv` | `7bda2b30e3e85a96db88131e8340808ff49a7f810a2ce0fe45955f593e7fe12c` | pokrycie według roli i klatki |
| `results/xgd_kp_diag/complete.json` | `c271936e2e47d55bce966f6af7de2c44322265398872c2c162f35c1a5c62f2d4` | rekord ukończenia |
| `results/xgd_kp_diag/metrics.csv` | `e6680930f8248ff390b1d1c0cf4b3498ac400313b77a1d49e35b28ecebe859d3` | metryki: konfiguracja × wielkość × podzbiór × statystyka |
| `xgpoc/detector/audit.py` | `891c55eba77b309fefccce7661849512eead082e74ac7d0de8a9e85511f15c96` | kod audytu: poziom przedziałów |
