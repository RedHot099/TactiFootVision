# Model punktów kluczowych boiska dla klatek SoccerNet: raport fazy (pitch-keypoints, ticket 06)

Jakub Eichner, 2026-10-13 (czas UTC). Gałąź `research/xg-detector`, faza pitch-keypoints, tickety 01–05.

Raport zamyka fazę. Szczegóły są w dwóch notatkach obok: `DATA_CARD.md` (zbiór danych, ticket 01) i `AUDIT3.md` (audyt 3, ticket 04). Przebiegi i decyzje są w `research/quality_v13_decisions.tsv`.

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `report_claims.toml` mówi, z którego pliku wyników pochodzi liczba. Skrypt `scripts/check_xgd_audit.py` odczytuje każdą liczbę ze źródła i kończy się błędem przy niezgodności. Pilnuje też rozdziału liczb: wyniki wyboru na zbiorach walidacyjnych stoją tylko pod nagłówkami z „wyniki rozwojowe”, a jedyna ocena wybranego modelu na zbiorach testowych tylko pod nagłówkami z „wynik testu”.

## 1. Odpowiedź i wynik testu w skrócie

**Model.** Nowy model punktów kluczowych boiska to YOLO11m-pose przy 960 px, uczony na klatkach SoccerNet z etykietami wyprowadzonymi z adnotacji linii boiska. Oceniono go na GSR test raz. Homografia jest na 85.5% [79.8; 90.5]<!--n:ct.avail--> klatek, a stopy graczy lądują w medianie 0.46 m [0.42; 0.50]<!--n:ct.med--> od prawdy (p90 1.47 m [1.23; 1.88]<!--n:ct.p90-->). Cel rady C3 (dostępność ≥ 85%<!--n:c3.avail-->, mediana ≤ 1.0 m<!--n:c3.med-->, p90 ≤ 3 m) jest spełniony, ale dostępność przechodzi ledwo: punkt leży tuż nad progiem, a przedział schodzi poniżej niego. Dotychczasowy model w konfiguracji D3 miał na tym samym teście 75.6%<!--n:bt.d3.avail1-->, 1.88 m<!--n:bt.d3.med1--> i p90 9.03 m<!--n:bt.d3.p90-->.

**Audyt 3.** Z nowym modelem i piłką z klatki −400 ms mediana błędu miejsca strzału wobec StatsBomb spadła z 10.5 m<!--n:a3.d3.med1--> (audyt 2) do 5.2 m<!--n:a3.g.med--> przy pokryciu 99.5%<!--n:a3.g.cov-->. Bramka wymaga ≤ 3 m<!--n:a3.med_max-->, więc wynik to STOP<!--n:a3.gate-->. Podejście 2 pozostaje zamknięte, A1 nie jest rozwijane od nowa, a ticket 05 zamknięto bez pracy.

**Co ogranicza teraz.** Homografia nie jest już głównym źródłem błędu. Ogranicza detektor piłki: na GSR valid umieszcza piłkę na 50.2%<!--n:ev.placed--> klatek, a 25.1%<!--n:ev.far--> umieszczonych piłek leży ponad 5 m od prawdziwej. Ogranicza też wybór bramkarza: jego mediana błędu to 6.5 m<!--n:a3.e3.gk-->, a stałego zgadywania 2.0 m<!--n:a3.e3.gkc-->.

## 2. Pytanie, decyzje właściciela i rada

### 2.1 Pytanie

Badanie xg-detector zamknęło podejście 2 (obraz → wykryte pozycje → xG) na bramce. Mediana błędu miejsca strzału wobec StatsBomb wyniosła 12.1 m<!--n:a1.c4.med1--> w audycie 1 i 10.5 m<!--n:a3.d3.med1--> w audycie 2, przy progu 3 m<!--n:a3.med_max-->. Ticket 08 tamtego badania wskazał główną przyczynę: model punktów kluczowych boiska, YOLOv8n-pose uczony na 316<!--n:hl.images--> obrazach Roboflow, myli się na klatkach SoccerNet w medianie o 48 px<!--n:rf.stretch.med-->. Pakiet nie miał też niezależnego pomiaru homografii w metrach. Audyty mieszały homografię, detekcję piłki i czas strzału.

Pytania tej fazy: czy model uczony na klatkach SoccerNet daje homografię dokładną w metrach? Czy to wystarczy, żeby podejście 2 przeszło zamrożoną bramkę?

### 2.2 Decyzje właściciela (2026-10-13)

Po raporcie xg-detector właściciel odpowiedział na otwarte punkty:

1. Przegląd raportu i kodu: na razie w porządku.
2. Model punktów kluczowych uczony na klatkach podobnych do SoccerNet: „zrób ile potrzeba, żeby osiągnąć zadowalający wynik”. To znosi regułę Q26 (brak trzeciego audytu) dla audytu z nowym modelem.
3. Poprawki pakietu do mastera: później.
4. Odłożone punkty (klip wideo, porównanie ze znanym pozycyjnym modelem xG, ocena homografii na GSR): tylko te, które dają realny zysk. Orkiestrator uznał ocenę na GSR za potrzebną, bo to niezależna prawda w metrach. Pozycyjny model xG miał być punktem odniesienia podejścia 2, gdyby się otworzyło. Klip nie daje zysku badawczego i pozostaje odłożony.

Pozostałe reguły wiążące obowiązują dalej: żaden model nie widzi danych StatsBomb, bramka i reguła wyboru konfiguracji są zamrożone, test na 172<!--n:pl.heldout--> meczach odłożonych odbywa się raz, G-combo zostaje bez zmian.

### 2.3 Rada: decyzja C3

Rada z badania xg-detector ma trzy miejsca: M (metodolog, Codex), V (inżynier widzenia komputerowego, Claude Opus) i S (sceptyk, Claude Fable). Przed treningiem zdecydowała:

| Pytanie | Głosy M / V / S | Decyzja |
|---|---|---|
| Q1: cel modelu | a / d / a | **(a)** na GSR test: dostępność ≥ 85%<!--n:c3.avail-->, mediana ≤ 1.0 m<!--n:c3.med-->, p90 ≤ 3 m, maska doprecyzowania z detektora pakietu. Stop przy pierwszym z warunków: 3 rodziny modeli, 30<!--n:c3.budget--> GPU-godzin, kandydat spełniający cel na GSR valid. Na teście jeden kandydat. |
| Q2: higiena danych | b / b / b | **(b)** wyłączyć rozpoznawalne klatki 172<!--n:pl.heldout--> meczów odłożonych i klatki Calibration z meczów GSR valid i test |
| Q3: konfiguracje audytu 3 | b / b / b | **(b)** E1–E4 mogą zmieniać źródło miejsca strzału, z uzasadnieniem na pozycjach piłki z GSR albo bez etykiet. Jedna konfiguracja to reguła audytu 2 z samą zamianą modelu. Audyt 3 rusza, jeśli model bije D3 na GSR valid, nawet bez spełnienia celu. |
| Q4: dalszy test | c / b / c | **(c)** bez drugiego testu na 172<!--n:pl.heldout--> meczach odłożonych, dalsza praca tylko w rozwoju. Drugi test zostaje do decyzji właściciela. |
| Czy rozwijać A1 od nowa, gdy audyt 3 da STOP | nie / nie / nie | nie |

Przy Q1 V głosował za własnym wariantem z warunkiem na klatkach przy bramce. Ta warstwa i progi pętli bez etykiet, które proponował S, weszły do raportu jako liczby podawane obok wyniku, nie jako wymogi. Przy Q4 V chciał dopuścić test samego podejścia 2. Rada zażądała też, żeby raport podał, skąd wziął się pomysł piłki z −400 ms (sekcja 9). Decyzja C3 nie trafiła wtedy do dziennika decyzji; wiersz dopisano przy tym raporcie.

## 3. Dane (ticket 01)

**Źródła.** To dane publiczne, pobierane bez hasła NDA. Calibration pobrano pakietem `SoccerNet`. GSR train i test pobrano z tego samego serwera co pakiet, ale ze wznawianiem, bo downloader pakietu przerywał duże pliki; GSR valid był już na dysku.
- **SoccerNet Game State Reconstruction 2024 (GSR):** półminutowe klipy ligi szwajcarskiej z 2019 roku, z kamery głównej, po trzy mecze na podział. Każda klatka ma adnotacje linii boiska i pozycje każdej osoby na boisku (`bbox_pitch`). Zostały adnotacje i co piąta klatka.
- **SoccerNet Camera Calibration 2023:** klatki transmisji z meczów SoccerNet-v2 (sześć rozgrywek, lata 2014–2017), także powtórki, z adnotacjami linii.

**Etykiety.** Dla każdego obrazu dopasowano homografię klatka → boisko do adnotowanych linii: punkty na prostych, przecięcia i styczne okręgów. Przez tę homografię rzutowano 32 wierzchołki `tv.pitch.SoccerPitch`. Wierzchołek poza kadrem ma widoczność 0. Nic nie oznaczano ręcznie. Obrazy zapisano w rozdzielczości 960 × 540.

**Odrzucanie.** Obraz zostaje, gdy spełnia wszystkie warunki (w pikselach 720p):
- dopasowanie istnieje;
- RMS reszt ≤ 4 px, a największa reszta ≤ 16 px<!--n:ds.maxres-->;
- odchylenie każdego widocznego wierzchołka ≤ 10 px<!--n:ds.sd-->;
- widok jest od strony kamery głównej;
- lustrzane adnotacje dają etykiety zgodne z `flip_idx`.

Widoki z przeciwnej strony i wzdłuż boiska, głównie powtórki z Calibration, odpadają. SoccerNet nazywa linie w układzie boiska, więc ich etykiety byłyby obrócone o ćwierć lub pół obrotu wobec podobnych obrazów z kamery głównej.

| źródło i podział | obrazy | klipy lub mecze | po regule | po wyłączeniach C3 | podział zbioru |
|---|---|---|---|---|---|
| GSR train | 8 550<!--n:ds.gtr.images--> | 57<!--n:ds.gtr.groups--> | 8 082<!--n:ds.gtr.kept--> | 8 082<!--n:ds.gtr.after--> | train |
| GSR valid | 8 700<!--n:ds.gva.images--> | 58<!--n:ds.gva.groups--> | 8 243<!--n:ds.gva.kept--> | 8 243<!--n:ds.gva.after--> | val |
| GSR test | 7 350<!--n:ds.gte.images--> | 49<!--n:ds.gte.groups--> | 6 653<!--n:ds.gte.kept--> | 6 653<!--n:ds.gte.after--> | test_gsr (tylko ocena) |
| Calibration train | 16 463<!--n:ds.ctr.images--> | 290<!--n:ds.ctr.groups--> | 10 537<!--n:ds.ctr.kept--> | 7 585<!--n:ds.ctr.after--> | train |
| Calibration valid | 3 212<!--n:ds.cva.images--> | 55<!--n:ds.cva.groups--> | 2 063<!--n:ds.cva.kept--> | 1 658<!--n:ds.cva.after--> | val |
| Calibration test | 3 141<!--n:ds.cte.images--> | 54<!--n:ds.cte.groups--> | 2 005<!--n:ds.cte.kept--> | 2 005<!--n:ds.cte.after--> | test_calib (tylko ocena) |

Regułę spełniły 37 583<!--n:ds.kept--> obrazy. Decyzja C3 wyłączyła z nich 3 357<!--n:ds.excluded-->. Zbiór ma 34 226<!--n:ds.images--> obrazów: train 15 667<!--n:ds.train-->, val 9 901<!--n:ds.val-->, test_gsr 6 653<!--n:ds.test_gsr--> i test_calib 2 005<!--n:ds.test_calib-->. Klip GSR i mecz Calibration leżą zawsze w jednym podziale.

**Wyłączenia C3.** Każdy mecz Calibration dopasowano do planu podziału xg-vendor po lidze, sezonie, dacie i drużynach.
- Z treningu wypadło 2 952<!--n:ds.excl.train--> obrazy z 76<!--n:ds.excl.train.games--> meczów ewaluacyjnych, z walidacji 405<!--n:ds.excl.val--> obrazów z 9 meczów ewaluacyjnych.
- test_calib zachowuje 18<!--n:ds.cte.eval--> meczów ewaluacyjnych i 15<!--n:ds.cte.gate--> bramkowych. Służą tylko do oceny.
- GSR nie podaje nazw meczów, więc nie da się go dopasować po nazwie. Nie może się jednak pokrywać z planem ani z Calibration: GSR to liga szwajcarska z 2019 roku, a plan i Calibration to mecze SoccerNet-v2 z lat 2014–2017.
- Duplikatów jednego obrazu w dwóch podziałach nie znaleziono.

**Kontrola konwencji i podłoga błędu.** Stopy osób z GSR, rzutowane homografią etykiety, leżą w medianie 0.35 m<!--n:ds.feet.train--> (train), 0.37 m<!--n:ds.feet.valid--> (valid) i 0.34 m<!--n:ds.feet.test--> (test) od pozycji `bbox_pitch`. Na GSR valid przy odwróconej osi y wychodzi 18.4 m<!--n:ds.axis.y-->, przy odwróconej osi x 82.2 m<!--n:ds.axis.x-->. Etykiety z lustrzanych adnotacji zgadzają się z `flip_idx` z dokładnością do 0.57 px<!--n:ds.flip--> na każdym zachowanym obrazie. Te około 0.35 m<!--n:ds.feet.train--> to błąd samej etykiety wobec pozycji GSR, czyli podłoga dla celu 1.0 m<!--n:c3.med-->: model uczony na tych etykietach nie powinien zejść w tej metryce wyraźnie niżej. Część tego błędu to szum samego GSR, bo w kilku klipach `bbox_pitch` odbiega o kilka metrów od namalowanych linii.

## 4. Ocena w metrach na GSR i punkt odniesienia (ticket 02)

**Metryka.**
- Klatki: co piąta klatka każdego klipu, w 720p. Punkty modelu idą do `HomographyEstimator` pakietu.
- Błąd osoby: dolny środek `bbox_image` każdej osoby na boisku (do 3 m za liniami) rzutuje się na boisko i porównuje z dolnym środkiem `bbox_pitch`, w metrach. Błąd klatki to mediana po jej osobach.
- Dostępność: udział ocenianych klatek z homografią. Klatka bez homografii liczy się jako porażka. Mediana i p90 są liczone po klatkach z homografią.
- Przedziały 95%<!--n:bv.level-->: bootstrap po klipach, 2 000<!--n:bv.draws--> losowań, seed 0.
- Podłoga samej metryki: homografia dopasowana do stóp tej samej klatki (z pominięciem ocenianej osoby) daje 0.12 m<!--n:bv.loo-->.

**Punkt odniesienia** to dotychczasowy model (YOLOv8n-pose, wejście rozciągnięte, próg instancji 0.1<!--n:bv.inst-->) w konfiguracjach D1–D4 z audytu 2. D1 i D3 mają próg punktów 0.5<!--n:bv.d1.kp-->, D2 i D4 próg 0.3<!--n:bv.d2.kp-->. D3 i D4 doprecyzowują homografię liniami boiska, z maską bez ramek osób z detektora pakietu (C3). D3 wybrał audyt 2 i jest punktem odniesienia. Wiersz „etykieta” to homografia z ticketu 01, czyli wyrocznia.

| metoda | valid: dostępność | valid: mediana | valid: p90 | test: dostępność | test: mediana | test: p90 |
|---|---|---|---|---|---|---|
| D1 | 82.0%<!--n:bv.d1.avail--> | 2.74 m [2.46; 3.10]<!--n:bv.d1.med--> | 6.54 m<!--n:bv.d1.p90--> | 75.6%<!--n:bt.d1.avail--> | 3.40 m [2.77; 4.09]<!--n:bt.d1.med--> | 8.74 m<!--n:bt.d1.p90--> |
| D2 | 87.1%<!--n:bv.d2.avail--> | 2.96 m [2.63; 3.29]<!--n:bv.d2.med--> | 7.14 m<!--n:bv.d2.p90--> | 83.2%<!--n:bt.d2.avail--> | 3.71 m [3.07; 4.49]<!--n:bt.d2.med--> | 10.11 m<!--n:bt.d2.p90--> |
| **D3** | 82.0%<!--n:bv.d3.avail--> | 1.66 m [1.33; 1.99]<!--n:bv.d3.med--> | 6.43 m<!--n:bv.d3.p90--> | 75.6%<!--n:bt.d3.avail1--> | 1.88 m [0.95; 3.02]<!--n:bt.d3.med--> | 9.03 m<!--n:bt.d3.p90--> |
| D4 | 87.1%<!--n:bv.d4.avail--> | 1.88 m [1.58; 2.31]<!--n:bv.d4.med--> | 7.18 m<!--n:bv.d4.p90--> | 83.2%<!--n:bt.d4.avail--> | 2.54 m [1.36; 3.53]<!--n:bt.d4.med--> | 10.39 m<!--n:bt.d4.p90--> |
| etykieta | 95.2%<!--n:bv.lab.avail--> | 0.37 m [0.35; 0.40]<!--n:bv.lab.med--> | 0.96 m<!--n:bv.lab.p90--> | 90.5%<!--n:bt.lab.avail--> | 0.34 m [0.32; 0.37]<!--n:bt.lab.med--> | 0.78 m<!--n:bt.lab.p90--> |

GSR valid ma 58<!--n:bv.clips--> klipów i 8 538<!--n:bv.frames--> ocenianych klatek, GSR test 49<!--n:bt.clips--> klipów i 7 260<!--n:bt.frames--> klatek. Najlepsza dotychczasowa konfiguracja, D3, jest na teście daleko od celu C3 na wszystkich trzech wielkościach.

- **Maska z detektora czy z prawdy GSR.** Sparowana różnica mediany D3 (maska z ramek GSR minus maska z detektora) wynosi −0.03 m [−0.11; +0.05]<!--n:bv.mask--> na valid i +0.13 m [−0.04; +0.20]<!--n:bt.mask--> na test. Maska z prawdy nie pomaga mierzalnie.
- **Przy bramce.** Ta warstwa to klatki, w których mediana x osób leży najwyżej 30 m<!--n:bv.near_m--> od linii bramkowej: 4 118<!--n:bv.near.frames--> klatek na valid i 3 732<!--n:bt.near.frames--> na test. D3 ma tu dostępność 65.1%<!--n:bv.near.avail--> i medianę 2.92 m<!--n:bv.near.med--> na valid oraz 59.2%<!--n:bt.near.avail--> i 2.48 m<!--n:bt.near.med--> (p90 58.8 m<!--n:bt.near.p90-->) na test. Widoki podobne do klatek strzałów były trudniejszą połową.
- **Według meczów.** Mediana D3 na trzech meczach testu wynosi 0.45 m<!--n:bt.g11.med-->, 2.60 m<!--n:bt.g7.med--> i 4.54 m<!--n:bt.g8.med-->. Najgorszy z nich ma dostępność 61.8%<!--n:bt.g8.avail--> i p90 59.4 m<!--n:bt.g8.p90-->. Rozrzut między meczami jest większy, niż sugerują przedziały z bootstrapu po klipach.
- **Punkty w pikselach.** Błąd liczono w 720p wobec etykiet z ticketu 01, dla punktów z pewnością ≥ 0.5<!--n:bv.kpconf-->. Mediana wynosi 32.7 px<!--n:bv.px--> na GSR valid, 37.9 px<!--n:bt.px--> na GSR test i 56.3 px<!--n:bct.px--> na Calibration test. Model znajduje odpowiednio 74.6%<!--n:bv.recall-->, 66.1%<!--n:bt.recall--> i 53.0%<!--n:bct.recall--> widocznych wierzchołków.

## 5. Trening i wybór na walidacji (wyniki rozwojowe, ticket 03)

**Uczenie.** YOLO11m-pose startował z wag COCO i był uczony przez `tv.train` pakietu, przy wejściu 960 i partii 8. Mozaikę, mixup, cutmix i copy-paste wyłączono. Augmentacja: odbicia poziome z `flip_idx`, skala 0.3<!--n:tr.scale-->, przesunięcie 0.1<!--n:tr.translate-->, zmiany barwy HSV. Trening szedł na 15 667<!--n:ds.train--> obrazach (GSR train i Calibration train), a walidacja na GSR valid i Calibration valid.

**Fałszywa diagnoza i jej koszt.**
1. Recall ramek w walidacji Ultralytics spadł z 0.90<!--n:tr.recall--> do mniej niż połowy, a wejście letterbox pakietu nie znajdowało instancji na większości klatek GSR valid. Przebieg zatrzymano po 18<!--n:tr.epoch--> epokach z 40<!--n:tr.epochs-->; zajął 2.06<!--n:tr.gpuh--> GPU-godziny.
2. Pierwsza diagnoza była błędna: ramka całego boiska miała przekraczać zasięg głowicy ramek. Dostrajanie ze stałymi wyśrodkowanymi ramkami trwało 946 s<!--n:tr.cbox--> czasu GPU, zanim je przerwano.
3. Prawdziwa przyczyna: Ultralytics uczy na kwadratowym płótnie, z szarymi pasami nad i pod panoramiczną klatką, a model polega na tych pasach. Wejście pakietu (`stretch=False`) i walidacja Ultralytics dopełniają klatkę tylko do wielokrotności kroku sieci (960 × 544). Na takim wejściu model znajdował instancję na 20%<!--n:tr.rect--> obrazów, a na kwadratowym na 100%<!--n:tr.square-->.
4. Ewaluator dostał wejście `square`. Na nim epoka 18<!--n:tr.epoch--> spełniła cel na walidacji, więc pozostałych epok nie trenowano.

**Kandydaci na GSR valid** (maska z detektora; `kp05` to próg punktów 0.5<!--n:cv.kp05-->, RANSAC 10 m<!--n:cv.ransac-->):

| kandydat | wejście | dostępność | mediana | p90 | < 1 m |
|---|---|---|---|---|---|
| D3 (odniesienie) | rozciągnięte | 82.0%<!--n:bv.d3.avail--> | 1.66 m<!--n:bv.d3.med1--> | 6.43 m<!--n:bv.d3.p90--> | 30.5%<!--n:bv.d3.lt1--> |
| epoka 18<!--n:tr.epoch-->, `kp05` z doprecyzowaniem | letterbox pakietu | 35.4%<!--n:lb.avail--> | 1.00 m<!--n:lb.med--> | 2.75 m<!--n:lb.p90--> | 17.6%<!--n:lb.lt1--> |
| epoka 18<!--n:tr.epoch-->, `kp05` z doprecyzowaniem | kwadrat | 89.6%<!--n:cv.r.avail--> | 0.62 m [0.56; 0.71]<!--n:cv.r.med--> | 2.21 m<!--n:cv.r.p90--> | 61.4%<!--n:cv.r.lt1--> |
| epoka 18<!--n:tr.epoch-->, `kp05` bez doprecyzowania | kwadrat | 89.6%<!--n:cv.k.avail--> | 0.97 m<!--n:cv.k.med--> | 2.57 m<!--n:cv.k.p90--> | 46.2%<!--n:cv.k.lt1--> |

Mediana błędu punktów w 720p wynosi 7.8 px<!--n:cv.px--> na GSR valid (D3: 32.7 px<!--n:bv.px-->) i 9.3 px [8.7; 10.1]<!--n:ccv.px--> na Calibration valid (D3: 58.4 px [49.3; 69.4]<!--n:bcv.px-->).

**Reguła stopu.** Pierwszy kandydat pierwszej rodziny spełnił cel na walidacji. Poszukiwania skończyły się więc po 2.6<!--n:tr.stop--> GPU-godziny z budżetu 30<!--n:c3.budget-->. Druga rodzina (model map ciepła) nie ruszyła.

**Dostrojenie homografii.** Regułę wyboru zapisano w commicie `c18fbfd` przed przebiegiem: spośród metod z dostępnością ≥ 85%<!--n:c3.avail--> i p90 ≤ 3 m na GSR valid wygrywa najniższa mediana, a przy różnicy do 0.01 m<!--n:rt.tie--> wyższa dostępność. Siatka miała trzy progi punktów, trzy limity RANSAC i doprecyzowanie włączone lub wyłączone, razem 18<!--n:rt.methods--> metod. Wszystkie spełniły oba progi. Wybrana metoda ma próg punktów 0.3<!--n:rt.kp-->, RANSAC 2 m i doprecyzowanie z maską z detektora: dostępność 93.0% [89.2; 95.9]<!--n:rt.avail-->, mediana 0.62 m [0.56; 0.71]<!--n:rt.med-->, p90 2.35 m [1.91; 2.97]<!--n:rt.p90-->.

**Warunek audytu 3 (C3).** Sparowana różnica mediany wybranej metody wobec D3 na GSR valid wynosi −1.04 m [−1.34; −0.74]<!--n:rt.paired-->. Model bije D3, więc audyt 3 rusza.

## 6. Wynik testu na GSR i Calibration (ticket 03)

Wybrany model z wybraną metodą oceniono raz na GSR test: 49<!--n:ct.clips--> klipów z trzech meczów, 7 260<!--n:ct.frames--> ocenianych klatek. Powód przebiegu jest zapisany w jego rekordzie. Żaden inny kandydat ani ustawienie nie dotknęło GSR test. Wcześniej oceniono na nim tylko punkt odniesienia (ticket 02).

### 6.1 Wynik testu wobec celu C3

| | dostępność | mediana | p90 | < 1 m | < 2 m |
|---|---|---|---|---|---|
| cel C3 | ≥ 85%<!--n:c3.avail--> | ≤ 1.0 m<!--n:c3.med--> | ≤ 3 m | | |
| **nowy model** | **85.5% [79.8; 90.5]<!--n:ct.avail-->** | **0.46 m [0.42; 0.50]<!--n:ct.med-->** | **1.47 m [1.23; 1.88]<!--n:ct.p90-->** | 70.8%<!--n:ct.lt1--> | 79.2%<!--n:ct.lt2--> |
| D3 | 75.6% [69.3; 81.5]<!--n:bt.d3.avail--> | 1.88 m [0.95; 3.02]<!--n:bt.d3.med--> | 9.03 m [7.33; 18.91]<!--n:bt.d3.p90ci--> | 31.5%<!--n:bt.d3.lt1--> | 38.2%<!--n:bt.d3.lt2--> |
| różnica sparowana, model − D3 | +10.0 pp [+2.9; +17.1]<!--n:ct.pd.avail--> | −1.42 m [−2.54; −0.50]<!--n:ct.pd.med--> | −7.56 m [−17.26; −5.88]<!--n:ct.pd.p90--> | +39.2 pp [+31.9; +47.0]<!--n:ct.pd.lt1--> | +41.0 pp<!--n:ct.pd.lt2--> |
| etykieta (wyrocznia) | 90.5%<!--n:bt.lab.avail--> | 0.34 m<!--n:bt.lab.med1--> | 0.78 m<!--n:bt.lab.p90--> | 85.0%<!--n:bt.lab.lt1--> | 89.0%<!--n:bt.lab.lt2--> |

Wszystkie trzy progi są spełnione na wartościach punktowych. Mediana leży blisko błędu samej etykiety.

**Margines dostępności jest wąski.** Punkt wynosi 85.5%<!--n:ct.avail1--> przy progu 85%<!--n:c3.avail-->, a dolna granica przedziału to 79.8%<!--n:ct.avail.lo-->. Inny zestaw klipów mógł dać wynik pod progiem. Mediana i p90 mają duży zapas.

### 6.2 Obok wyniku testu: warstwy, mecze, piksele

- **Przy bramce** (3 732<!--n:ct.near.frames--> klatki): dostępność 97.1%<!--n:ct.near.avail-->, mediana 0.42 m<!--n:ct.near.med-->, p90 1.00 m<!--n:ct.near.p90-->. D3 miał tu 59.2%<!--n:bt.near.avail-->, 2.48 m<!--n:bt.near.med--> i 58.8 m<!--n:bt.near.p90-->. Sparowana różnica mediany to −2.05 m [−3.71; −0.73]<!--n:ct.near.pd-->. Klatki przy bramce są teraz łatwiejszą połową, odwrotnie niż dla D3.
- **Według meczów:**

  | mecz | dostępność | mediana | p90 | mediana D3 |
  |---|---|---|---|---|
  | mecz 7 | 89.7%<!--n:ct.g7.avail--> | 0.44 m<!--n:ct.g7.med--> | 1.51 m<!--n:ct.g7.p90--> | 2.60 m<!--n:bt.g7.med--> |
  | mecz 8 | 80.1%<!--n:ct.g8.avail--> | 0.54 m<!--n:ct.g8.med--> | 2.26 m<!--n:ct.g8.p90--> | 4.54 m<!--n:bt.g8.med--> |
  | mecz 11 | 88.2%<!--n:ct.g11.avail--> | 0.39 m<!--n:ct.g11.med--> | 0.84 m<!--n:ct.g11.p90--> | 0.45 m<!--n:bt.g11.med--> |

  Mecz 8, najtrudniejszy już dla D3, jest jedynym poniżej progu dostępności.
- **Punkty w pikselach (720p).** Na GSR test mediana wynosi 7.2 px [6.7; 7.7]<!--n:ct.px-->, a model znajduje 93.4%<!--n:ct.recall--> widocznych wierzchołków (D3: 37.9 px<!--n:bt.px-->, 66.1%<!--n:bt.recall-->).
- **Calibration-2023 test w pikselach** (2 005<!--n:cct.frames--> obrazów z 54<!--n:cct.games--> meczów, oceniony raz): mediana 9.0 px [8.5; 9.6]<!--n:cct.px-->, p90 27.2 px<!--n:cct.p90-->. Model znajduje 93.1%<!--n:cct.recall--> widocznych wierzchołków, a fałszywych punktów jest 3.6%<!--n:cct.spurious-->. D3 miał 56.3 px<!--n:bct.px--> i 53.0%<!--n:bct.recall-->.

## 7. Pętla bez etykiet z ticketu 08

Metryka zgodności rzutowanych linii boiska z namalowanymi, z ticketu 08 badania xg-detector, na tych samych 1 500<!--n:lf.frames--> klatkach t0 z meczów treningowych. Przebieg nie czyta StatsBomb; z indeksu czyta tylko identyfikatory strzałów, mecze i ścieżki klatek. „Dobra zgodność” to zgodność ≥ 0.5<!--n:lf.good-->.

| metoda | homografia | dobra zgodność linii | mediana zgodności |
|---|---|---|---|
| nowy model, same punkty (próg 0.3<!--n:lf.kp-->, RANSAC 2 m) | 98.2%<!--n:lf.k.hom--> | 78.7%<!--n:lf.k.good--> | 0.72<!--n:lf.k.agr--> |
| dotychczasowy model, same punkty | 71.3%<!--n:lf.c.hom--> | 0.9%<!--n:lf.c.good--> | 0.20<!--n:lf.c.agr--> |
| nowy model z doprecyzowaniem | 98.2%<!--n:lf.r.hom--> | 85.0%<!--n:lf.r.good--> | 0.91<!--n:lf.r.agr--> |
| dotychczasowy model z doprecyzowaniem | 71.3%<!--n:lf.cr.hom--> | 36.3%<!--n:lf.cr.good--> | 0.61<!--n:lf.cr.agr--> |

Na foldach potwierdzających dobrą zgodność ma 72.9%<!--n:lf.k.conf--> klatek z samych punktów i 78.5%<!--n:lf.r.conf--> z doprecyzowaniem. Lustrzanych dopasowań nie ma. Na 773<!--n:lf.res.frames--> klatkach z homografią potwierdzoną liniami punkty leżą w medianie 7.2 px<!--n:lf.res.px--> od niej (720p). W tickecie 08 dotychczasowy model miał w tej mierze 48 px<!--n:rf.stretch.med-->.

## 8. Zmiana w pakiecie (ticket 04)

Model potrzebuje kwadratowego płótna także przy inferencji.
- **Opcja `resize`.** Backendy Ultralytics w pakiecie przyjmują teraz `resize="letterbox" | "square" | "stretch"` zamiast `stretch: bool`, bez warstwy zgodności (commit `2e9469b`). `"square"` dopełnia klatkę do wyśrodkowanego szarego kwadratu, tak jak Ultralytics przy treningu, i przenosi ramki i punkty z powrotem do pikseli klatki. Testy z atrapą predyktora sprawdzają dopełnienie poziomych i pionowych klatek oraz powrót współrzędnych.
- **Równość z ocenionym wejściem.** `tv.load_model("yolo_pose", W, conf=0.1, imgsz=960, resize="square")` daje na 1 500<!--n:sq.frames--> klatkach strzałów z meczów treningowych dokładnie te punkty, które oceniono w tickecie 03. Instancja jest na 1 500<!--n:sq.new--> klatkach, a największa różnica położenia to 0.0 px<!--n:sq.max-->.
- **Pliki przebiegu.** Ten sam commit przełącza README, oba pliki przebiegu i `examples/04` na nowy model z jego ustawieniami homografii (`min_confidence=0.3`, `ransac_threshold=2.0`, `refine_with_lines=True`). Plik `models/pitch_yolo11m_pose_sn960.pt` istnieje tylko lokalnie, jak poprzedni model. Notatniki zostają przy modelu Roboflow z `resize="stretch"`.
- **Skutek dla zamrożonego testu xg-detector.** Commit `2e9469b` zmienił `src/tactifoot_vision/models/ultralytics.py`, który należy do przypiętego zamknięcia kodu testu A1, a commit `89bd0ee` dostosował wywołanie w `xgpoc/detector/extract.py` (konfiguracje D bez zmian). `xgd_test.py --check` przechodzi więc już tylko przy zamrożonym commicie `e14bb79`. Test się odbył i nie jest powtarzany. Pliki przypięte listą skrótów testu są niezmienione.

## 9. Audyt 3 (ticket 04)

Szczegóły są w `AUDIT3.md`.

**Deklaracja przed odczytem StatsBomb.** Trzy konfiguracje E1–E3 zapisano w pliku `research/pitchkp_audit3_configs.json` (SHA-256 `f7f23eaede754326a24b6d177c1e83e1eb6158c2cb22f1e36b81794604f8a8b7`, commit `e1d41e0`) przed jakimkolwiek odczytem danych StatsBomb w tickecie 04. Skrypty audytu i ekstrakcji odmawiają pracy z plikiem o innym skrócie. Wspólne ustawienia: detektor i drużyny jak w C4 i D3, nowy model na kwadratowym wejściu, próg punktów 0.3<!--n:dc.kp-->, RANSAC 2 m, doprecyzowanie liniami. Konfiguracje różnią się tylko kolejnością klatek, z których piłka daje miejsce strzału:
- E1: t0, −400 ms, −800 ms. To reguła D3 z audytu 2; zmieniono tylko model i jego ustawienia.
- E2: −400 ms, −800 ms.
- E3: −400 ms, −800 ms, t0.

Reguły klatki kopnięcia z toru piłki nie zadeklarowano. Na klipach strzałów GSR valid jej mediana błędu wynosiła 13.6 m<!--n:ev.kick-->.

**Skąd pomysł z −400 ms.** Wziął się z diagnostyki audytu 2 na danych StatsBomb. Tam w D3 piłka z −400 ms miała medianę błędu 6.0 m<!--n:dg2.m400.med-->, a piłka z t0 11.9 m<!--n:dg2.t0.med-->; grupa −400 ms nie była losowa. Tamte liczby nie uzasadniają E2 i E3. Uzasadnienie pochodzi z prawdy GSR, bez StatsBomb. W 21<!--n:ev.clips--> klipach strzałów GSR train i valid reguła na torze piłki z adnotacji znalazła kopnięcie w 11<!--n:ev.kicks--> klipach, w medianie 240 ms<!--n:ev.kick_ms--> przed t0. Piłka leży od miejsca kopnięcia w medianie 2.0 m<!--n:ev.to400--> w klatce −400 ms, 3.4 m<!--n:ev.to800--> w −800 ms i 5.1 m<!--n:ev.to0--> w t0. Pomysł przyszedł jednak ze StatsBomb, więc audyt 3 nie jest od niego w pełni niezależny.

**Wynik.** Audyt objął te same 2 721<!--n:a3.shots--> sparowanych strzałów bez karnych z 135<!--n:a3.matches--> meczów co audyty 1–2, z 5 000<!--n:a3.n_boot--> losowaniami meczów. D3 przeliczony w tym audycie odtwarza wszystkie 390<!--n:a3.ref.rows--> wierszy metryk audytu 2. Reguła wyboru (najniższa mediana wśród konfiguracji z pokryciem ≥ 70%<!--n:a3.cov_min-->) wybrała E3<!--n:a3.chosen-->: E1 9.24 m<!--n:a3.e1.med2-->, E2 5.22 m<!--n:a3.e2.med2-->, E3 5.18 m [4.68; 5.85]<!--n:a3.e3.med2ci-->.

| audyt | konfiguracja | pokrycie miejsca strzału | mediana błędu miejsca strzału | stałe zgadywanie | informacja o bramkarzu | mediana błędu bramkarza (stałe zgadywanie) |
|---|---|---|---|---|---|---|
| 1 | C4 | 71.8%<!--n:a1.c4.cov--> | 12.1 m [11.4; 12.8]<!--n:a1.c4.med--> | 10.6 m<!--n:a1.c4.const--> | 60.2%<!--n:a1.c4.gk.cov--> | 6.9 m<!--n:a1.c4.gk--> (2.0 m<!--n:a1.c4.gkc-->) |
| 2 | D3 | 77.4%<!--n:a3.d3.cov--> | 10.5 m [9.9; 11.1]<!--n:a3.d3.med--> | 10.7 m<!--n:a3.d3.const--> | 66.5%<!--n:a3.d3.gk.cov--> | 8.6 m<!--n:a3.d3.gk--> (1.9 m<!--n:a3.d3.gkc-->) |
| 3 | E1 | 99.5%<!--n:a3.e1.cov--> | 9.2 m [8.8; 9.9]<!--n:a3.e1.med--> | 10.3 m<!--n:a3.e1.const--> | 98.8%<!--n:a3.e1.gk.cov--> | 6.6 m<!--n:a3.e1.gk--> (2.0 m<!--n:a3.e1.gkc-->) |
| 3 | E2 | 99.5%<!--n:a3.e2.cov--> | 5.2 m [4.7; 5.9]<!--n:a3.e2.med--> | 10.3 m<!--n:a3.e2.const--> | 98.9%<!--n:a3.e2.gk.cov--> | 6.6 m<!--n:a3.e2.gk--> (2.0 m<!--n:a3.e2.gkc-->) |
| 3 | **E3** | 99.5%<!--n:a3.e3.cov--> | 5.2 m [4.7; 5.8]<!--n:a3.e3.med--> | 10.3 m<!--n:a3.e3.const--> | 98.9%<!--n:a3.e3.gk.cov--> | 6.5 m<!--n:a3.e3.gk--> (2.0 m<!--n:a3.e3.gkc-->) |

**Bramka: STOP<!--n:a3.gate-->.** Pokrycie 99.5%<!--n:a3.g.cov--> spełnia próg ≥ 70%<!--n:a3.cov_min-->, a informacja o bramkarzu 98.9%<!--n:a3.g.gk--> próg ≥ 50%<!--n:a3.gk_min-->. Mediana 5.2 m<!--n:a3.g.med--> przekracza jednak próg ≤ 3 m<!--n:a3.med_max-->.

**Co mówią liczby.**
- Sam nowy model (E1 wobec D3) obniża medianę o nieco ponad metr i podnosi pokrycie do prawie pełnego.
- Większość poprawy daje piłka z −400 ms. W E1 ta sama homografia daje 10.6 m<!--n:dg3.e1.t0--> z piłką w t0 i 3.9 m<!--n:dg3.e1.m400--> z piłką z −400 ms.
- Wykryte miejsce strzału ma teraz o połowę mniejszy błąd niż stałe zgadywanie. W audycie 2 było mu równe.

**Co ogranicza teraz.**
- **Detektor piłki.** Na 6 404<!--n:ev.frames--> klatkach GSR valid z piłką na boisku detektor umieszcza piłkę na 50.2%<!--n:ev.placed--> klatek. Z umieszczonych piłek 25.1%<!--n:ev.far--> leży ponad 5 m od prawdziwej, zwykle na innym obiekcie. Nawet z właściwej klatki (E3, piłka z −400 ms) zostaje 4.3 m<!--n:dg3.e3.m400--> błędu.
- **Wybór bramkarza.** Bramkarz E3 myli się w medianie o 6.5 m<!--n:a3.e3.gk-->, a stałe zgadywanie o 2.0 m<!--n:a3.e3.gkc-->. Zawodnik oznaczony przez detektor jako bramkarz wypada gorzej (10.6 m<!--n:a3.e3.gkd.med-->) niż zastępczy, czyli zawodnik najbliżej bramki. Homografia myli się na GSR o mniej niż metr, więc ten błąd pochodzi z wyboru zawodnika i z chwili klatki t0.

**Mecze audytu a trening modelu.** Wszystkie 135<!--n:a3.matches--> meczów audytu to mecze ewaluacyjne, które decyzja C3 wyłączyła z treningu i walidacji modelu. W treningu nie ma żadnego z nich. Bramka dotyczy więc w całości meczów, których model nie widział w Calibration-2023.

## 10. Co z tego wynika

1. **Podejście 2 pozostaje zamknięte**, bo bramka dała STOP<!--n:a3.gate--> przy 5.2 m<!--n:a3.g.med-->.
2. **A1 nie jest rozwijane od nowa** z nową geometrią. Rada zdecydowała to jednogłośnie przed audytem.
3. **Ticket 05** (dalszy model xG z nową geometrią) zamknięto bez pracy. Nic nie trenowano.
4. **Ekstrakcja E3** wszystkich 11 030<!--n:fu3.shots--> strzałów indeksu leży w `cache/xgd_full3/E3` i może służyć dalszej pracy rozwojowej.
5. **Model i opcja `resize="square"`** są w pakiecie, w osobnym commicie gotowym do przeniesienia na master.

## 11. Ograniczenia

- **GSR to wąska prawda.** To liga szwajcarska z 2019 roku, gra otwarta, kamera główna i po trzy mecze na podział. Test to niecałe pięćdziesiąt klipów z trzech meczów, a rozrzut między meczami jest większy, niż mówią przedziały po klipach. Klatki strzałów z meczów SoccerNet-v2 są często ciaśniejsze. Calibration test, z meczów sześciu rozgrywek, mierzy tylko piksele.
- **Wąski margines dostępności** (sekcja 6.1): punkt leży tuż nad progiem, a dolna granica przedziału pod nim.
- **Etykiety i pozycje GSR mają własny błąd.** Etykieta myli się wobec GSR o około 0.35 m<!--n:ds.feet.train-->, więc mediana z testu jest blisko tej podłogi. W kilku klipach `bbox_pitch` z GSR odbiega o metry od namalowanych linii.
- **Progi odrzucania etykiet częściowo ustalone na GSR valid.** Próg RMS podniesiono z 3 do 4 px po sprawdzeniu pozycji na GSR valid. Kontrola stóp nie jest więc na GSR valid w pełni niezależna od reguły; na GSR train i test jest.
- **Jeden przebieg treningu.** Wybrano epokę 18<!--n:tr.epoch--> przerwanego przebiegu, bez powtórzeń z innym seedem.
- **Przeglądy tylko modelami Anthropic.** Przegląd kodu (`/code-review`) każdego ticketu robił Claude, bez drugiego dostawcy. W radzie jedno miejsce (M) zajmuje model Codex.
- **Pomysł −400 ms pochodzi ze StatsBomb** (sekcja 9).
- **`xgd_test.py --check` przechodzi tylko przy `e14bb79`** (sekcja 8).
- **Decyzja C3 trafiła do dziennika z opóźnieniem.** Była zapisana w notatkach rady i w specyfikacji fazy, a wiersz w `research/quality_v13_decisions.tsv` dopisano przy tym raporcie.

## 12. Otwarte dla właściciela

1. **Detektor piłki uczony na klatkach SoccerNet** to następna dźwignia: GSR ma ramki i pozycje piłki. To praca produktowa, nierozpoczęta. Dopiero z nim ma sens reguła chwili strzału z toru piłki.
2. **Drugi test na 172<!--n:pl.heldout--> meczach odłożonych** (C3 Q4) to decyzja właściciela, bo reguła jednego testu należy do niego. Po STOP w audycie 3 podejście 2 nie ma kandydata do takiego testu.
3. **Commity pakietu do mastera.** Chodzi o `2e9469b` (opcja `resize`) razem z poprawkami z xg-detector (lista w raporcie xg-detector, sekcja 8). `2e9469b` przełącza też README, oba pliki przebiegu i `examples/04` na `models/pitch_yolo11m_pose_sn960.pt`, który istnieje tylko lokalnie. Master potrzebuje więc sposobu dystrybucji tych wag albo powrotu tych plików do starego modelu.
4. **Notatniki** `00_overview`, `04_inference` i `05_visualisation` mają stare wyjścia i model Roboflow. Trzeba je przebudować (`notebooks/build/build.py --execute`), ewentualnie z nowym modelem.
5. **Dłuższy trening (opcjonalnie).** Przebieg zatrzymano po 18<!--n:tr.epoch--> z 40<!--n:tr.epochs--> epok. Reszta harmonogramu może poprawić dostępność, ale GSR test jest dla tego kandydata zużyty, więc wynik byłby tylko rozwojowy.
6. **Widoki z przeciwnej strony boiska** (ticket 01, opcja b): można je obrócić do konwencji kamery głównej, jeśli okażą się potrzebne.
7. **Przegląd termonuklearny** uruchamia właściciel.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `research/pitchkp_audit3_configs.json` | `f7f23eaede754326a24b6d177c1e83e1eb6158c2cb22f1e36b81794604f8a8b7` | deklaracja E1–E3 |
| `research/quality_v13_decisions.tsv` | `81958d09522092ae41d688cbf18f7b529be7dad2ae283b29a0d87b183a2b5bea` | dziennik decyzji: rada C3, przebiegi treningu |
| `research/xg_detector_test_rule.json` | `4d593f52286e8c0ea46a39c8c07097c70de5be15905b924190d1745578fa87ca` | liczba meczów odłożonych (reguła testu xg-detector) |
| `results/pitchkp_gsr_eval/calib_test/keypoints_summary.csv` | `1bfac1eafd8dd9384467e7f8ed9b3dd9c1877ed20ba99723b48fa2418155245a` | ticket 02: błąd punktów D3 na Calibration test |
| `results/pitchkp_gsr_eval/calib_valid/keypoints_summary.csv` | `dff7a283b62d730f6a2db2b8b9fa71ed2472529d25846a7c6ddfb6ec50d6c672` | ticket 03: błąd punktów D3 na Calibration valid |
| `results/pitchkp_gsr_eval/test/keypoints_summary.csv` | `cb2c51873492a32c8e9ebe0c0e01278ac92fb36b1cc75261e2cc6fd56b135ac2` | ticket 02: błąd punktów D3 na GSR test |
| `results/pitchkp_gsr_eval/test/paired.csv` | `0d2761de7539e6ac99b2391447927bd643268004f633dcae8281870fefbe6237` | ticket 02: różnice sparowane na GSR test |
| `results/pitchkp_gsr_eval/test/per_game.csv` | `d303e6e6388e6700cce064114ab6ca090ad4f1385daf88e82acb9a576b87df0b` | ticket 02: D3 według meczów GSR test |
| `results/pitchkp_gsr_eval/test/summary.csv` | `a10f3e0925f894e84b039c5dd253098dd99b52213317ad924f5c376223afc469` | ticket 02: D1–D4 i etykieta na GSR test |
| `results/pitchkp_gsr_eval/valid/complete.json` | `6ec5ad771e0ce6bdab8e3c9d69625e9a3304d1c65f8deed5b6a6842882d25d72` | ticket 02: parametry metryki, konfiguracje |
| `results/pitchkp_gsr_eval/valid/keypoints_summary.csv` | `bee8d0d5d3167bed7c8df476d67d282b54ebd270c624cd2961f4ea895af71655` | ticket 02: błąd punktów D3 na GSR valid |
| `results/pitchkp_gsr_eval/valid/paired.csv` | `c8f4fec808d66c437148e99bf76045dbec3dad5ffe98c75fb0ed7d17b37ea747` | ticket 02: różnice sparowane na GSR valid |
| `results/pitchkp_gsr_eval/valid/summary.csv` | `f8a287179189727d771314d6ab9a8a3f2a6f6fc91fe057feff8e9906fa5c35e3` | ticket 02: D1–D4 i etykieta na GSR valid |
| `results/pk_ball_evidence/ball.csv` | `dcaecfae0ba5ff31f512b37e28cd60ecc7b428dee05e6c879bfc4853a12982b6` | ticket 04: umieszczenie piłki na GSR valid |
| `results/pk_ball_evidence/summary.json` | `4b5873b6612a9ab6057a0ef93c83a6836cb09f92bef3c24f28d7e2630a1f98c9` | ticket 04: czas strzału i reguły piłki na GSR |
| `results/pk_candidates/y11m_960_e18/valid/summary.csv` | `d337e701271ca58854f58b6d93a1ca7fffba7c578eb2877173babc7e2fdefb74` | ticket 03: kandydat z wejściem letterbox, GSR valid |
| `results/pk_candidates/y11m_960_e18_square/calib_test/keypoints_summary.csv` | `3b6960069c30a4fc33ce0972ca7f5b3e09426375dcbfeed641652b557bdbd0fb` | ticket 03: błąd punktów na Calibration test |
| `results/pk_candidates/y11m_960_e18_square/calib_valid/keypoints_summary.csv` | `8c5063857833b436d59aab33e2ed12b2d217a1c683d2e83d36af431ec563f51e` | ticket 03: błąd punktów kandydata, Calibration valid |
| `results/pk_candidates/y11m_960_e18_square/retune_valid/complete.json` | `d95b5b5b1a5ab96e540e9c2d5fa81dc846481dc4726d71529ac627e69e72812d` | ticket 03: metody dostrojenia |
| `results/pk_candidates/y11m_960_e18_square/retune_valid/paired.csv` | `7417410f34b7f05a77753adc8f8f23cdf8bbba82113bdc47ed3b35b11e66bd20` | ticket 03: różnica wobec D3, GSR valid |
| `results/pk_candidates/y11m_960_e18_square/retune_valid/summary.csv` | `4327764fc6c983c79044a482bfea4f7cdc92e08c602cd4607a022a263cffb197` | ticket 03: dostrojenie homografii, GSR valid |
| `results/pk_candidates/y11m_960_e18_square/test/keypoints_summary.csv` | `5267b6076337750761eafa6724c789fe099edea83411619a9302281eb043479e` | ticket 03: błąd punktów na GSR test |
| `results/pk_candidates/y11m_960_e18_square/test/paired.csv` | `51822e74e65ce80304e385083feefb68395f37886bc542e09c4666f7c0c08575` | ticket 03: różnice wobec D3 na GSR test |
| `results/pk_candidates/y11m_960_e18_square/test/per_game.csv` | `fe258b66da4e9e43c3bdbf4e9287e241a8897c54ba720fcc925ac2b4bb34031f` | ticket 03: według meczów GSR test |
| `results/pk_candidates/y11m_960_e18_square/test/summary.csv` | `bed9c07e4902e0601e16d157c2440f1aeb663a2ceecc7ef02e3fc96a883b3db2` | ticket 03: jedyna ocena na GSR test |
| `results/pk_candidates/y11m_960_e18_square/valid/complete.json` | `66a1c065125f5e7b813ef284be94ba81ee23fd11840d90d0c13ce09f86d22961` | ticket 03: metody kandydata |
| `results/pk_candidates/y11m_960_e18_square/valid/keypoints_summary.csv` | `ef0a29419e85585070488833067c961a85e6c063f2dd78e99d3e44e68358b627` | ticket 03: błąd punktów kandydata, GSR valid |
| `results/pk_candidates/y11m_960_e18_square/valid/summary.csv` | `4ed4d88548e28e63d5681daf852f884765c5ad79d5dc1b7d4495a973450c62a0` | ticket 03: kandydat z wejściem kwadratowym, GSR valid |
| `results/pk_dataset/complete.json` | `a96d9b146239a1d74668a3d6c7ceaa4d06467a683d017799a4ee0cea5f7c93b7` | ticket 01: rekord ukończenia zbioru |
| `results/pk_dataset/labelled.csv` | `712a50b41ac556aa4f925f24b285ebe85163abf4fafe1b3a2280288da202c90b` | ticket 01: obrazy po regule i wyłączenia C3 |
| `results/pk_dataset/summary.json` | `88f2e084aa67822cab144d897747aa5cc5f782ed4f01f045311daf23bce07811` | ticket 01: liczności, reguła odrzucania, kontrole |
| `results/pk_label_free/y11m_960_e18_square/complete.json` | `49b75bf66c85bc3006ad1a33bc688a1343710fdcdc3a7b98f378f839acd50d7d` | ticket 03: parametry pętli, reszty punktów |
| `results/pk_label_free/y11m_960_e18_square/summary.csv` | `8881f32dda2c67a3cee88133a5a879531f50f916a6fef9aeb1e512f92f31ffd4` | ticket 03: pętla bez etykiet |
| `results/pk_square_input/complete.json` | `8b6e86455948a1f3fc87d62aee8ff6c07a7fda96cac394863f4e53f154b2749d` | ticket 04: równość wejścia kwadratowego w pakiecie |
| `results/xgd_audit/metrics.csv` | `094eb796175da304a1e650f2aaa09500d08878d2562887ea2f56788137077280` | audyt 1: metryki |
| `results/xgd_audit2/metrics.csv` | `e422a0a2311a924a4b4e2f2ed895b264360437f8bd2ddf93fd37e156722420ab` | audyt 2: stałe zgadywanie dla C4 z audytu 1 |
| `results/xgd_audit2_diag/diag.csv` | `32f45780139e42676b25fd81417d5bd9c2ef5f22b88995af9e59b2a47bd1776e` | audyt 2: podziały błędów (opisowo) |
| `results/xgd_audit3/choice.json` | `2714e7dad473f0d02e79359a98c6d8e0c9356d25877c03de1d73be13bf25ef02` | audyt 3: wybór konfiguracji i bramka |
| `results/xgd_audit3/complete.json` | `dae5c570a9b6a869744469f38b20135b11ecba2d14051291dd153844bbb7cbbb` | audyt 3: populacja, parametry bramki, odtworzenie D3 |
| `results/xgd_audit3/metrics.csv` | `38d2d5b4d67761d4a80487b76e8b5fa9ff0a2ac15cafcdb50ef2523f8ace6d93` | audyt 3: metryki |
| `results/xgd_audit3_diag/diag.csv` | `dcea6970d6d044c1b83a9e7c1ce6eff81757a48659151d5c9bdbc28085798eb5` | audyt 3: podziały błędów (opisowo) |
| `results/xgd_full3/coverage_by_role.csv` | `a3a26ef214644a6eff4564da655e0b911abd2a5f37bd241c4b646750a548b499` | ticket 04: pełna ekstrakcja E3 |
| `results/xgd_homography_dev/labels/summary.json` | `a55a4fe165dc37ec13d819dfc2ddb6032463c2d6692b3b585307ab412337bb61` | ticket 08 xg-detector: obrazy treningowe dotychczasowego modelu |
| `results/xgd_report_figures/figures.json` | `cc9ad66d2437510b2fcc176eeff3d85f656933a49609955d66d61088feaf5646` | ticket 08 xg-detector: błąd punktów dotychczasowego modelu na SoccerNet |
| `xgpoc/pitchkp/gsr_eval.py` | `03b098eb74ea7602b02cde9c77a6526bf12ce9e2c30d365fc5aeaafd71a13745` | kod oceny na GSR: poziom przedziałów |
