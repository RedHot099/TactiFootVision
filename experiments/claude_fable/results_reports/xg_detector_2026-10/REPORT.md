# xG z wejściami z detektora wobec G-combo: raport końcowy (xg-detector, ticket 07)

Jakub Eichner, 2026-10-13 (czas UTC). Gałąź `research/xg-detector`, badanie xg-detector, tickety 01–09.

Raport zamyka badanie. Szczegóły audytów są w notatce `AUDIT.md` obok. Reguła testu jest w `research/xg_detector_test_rule.md`, a decyzje w `research/quality_v13_decisions.tsv`.

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `report_claims.toml` mówi, z którego pliku wyników pochodzi liczba. Skrypt `scripts/check_xgd_audit.py` odczytuje każdą liczbę ze źródła i kończy się błędem przy niezgodności. Ten sam skrypt pilnuje rozdziału liczb: liczby rozwojowe (walidacja krzyżowa na meczach treningowych i moc policzona z nich) stoją tylko pod nagłówkami z „wyniki rozwojowe”, a liczby z testu tylko pod nagłówkami z „wynik testu”.

## 1. Odpowiedź i wynik testu w skrócie

**Pytanie 1: czy jawna struktura z detektora poprawia model obrazowy?** Na meczach testowych poprawy nie wykazano. Jedyny wariant w teście, A1 (G-combo z cechami geometrycznymi z detektora w głowicy), dał ΔLL = +0.0009 [−0.0012; +0.0030]<!--n:ts.d95--> względem G-combo (przedział 95%<!--n:rule.success_level-->). Według zamrożonej reguły to odczyt **equivalent within δ<!--n:ts.label-->**: bez sukcesu i bez pogorszenia, a przedział 90%<!--n:rule.eq_level--> mieści się w granicy równoważności δ = 0.005<!--n:rule.delta-->. Równoważność jest odczytem opisowym, nie sukcesem. Wynik dotyczy detektora i homografii D3 w stanie z audytu 2, nie struktury z detektora w ogóle (sekcja 6.4).

**Pytanie 2: czy łańcuch obraz → pozycje z detektora → xG działa na klatkach transmisji?** Nie w obecnym stanie pipeline'u. Oba audyty zatrzymały podejście 2 na bramce. Po poprawkach homografii mediana błędu miejsca strzału wynosi 10.5 m<!--n:a2.g.med-->, a próg to 3 m<!--n:au.med_max-->. Wykryte miejsce strzału nie jest w medianie bliżej prawdy niż stałe zgadywanie (sekcja 3.4). Podejście 2 nie powstało (sekcja 4).

## 2. Pytanie i protokół

### 2.1 Pytania

Badanie xg-vendor pokazało, że z osadzenia G-combo da się odczytać geometrię strzału (odległość, obrońców w stożku), a sama klatka nie dodaje nic do pozycji z danych zdarzeń. G-combo widzi tylko piksele i uczy się z jednego bitu na strzał. Pipeline tactifoot ma detektor (YOLO11m: piłka, bramkarz, zawodnik, sędzia), punkty kluczowe boiska z homografią do metrów i przypisanie drużyn przez SigLIP. Stąd dwa pytania:

1. Czy podanie modelowi obrazowemu jawnej struktury z detektora poprawia jego xG? Kandydaci to cechy geometryczne w metrach (A1), raster boiska z góry (A2) i tokeny detekcji (A3).
2. Czy model dwuetapowy, obraz → pozycje na boisku → xG, w ogóle działa na klatkach transmisji i czy dodaje coś do G-combo (podejście 2)?

### 2.2 Ustalenia właściciela (Q1–Q27)

Plan ustalono w wywiadzie z właścicielem 2026-10-12 (Q1–Q22). Pięć decyzji zapadło później, po ekstrakcji z ticketu 01 i po audycie 1 (Q23–Q27).

- **Podział danych (Q1).** Cały rozwój idzie w walidacji krzyżowej na 328<!--n:pl.training--> meczach treningowych, w foldach według gospodarza z planu podziału xg-vendor. Test odbywa się raz, na 172<!--n:rule.matches--> meczach odłożonych: 135<!--n:pl.evaluation--> ewaluacyjnych i 37<!--n:pl.gate--> bramkowych. Nowe modele ich nie widzą.
- **Lekki protokół (Q2).** Przed testem regułę porównania zamraża się w commicie. Pełnej prerejestracji jak w xg-vendor nie ma.
- **Bez StatsBomb w modelach (Q3).** Żaden nowy model nie widzi danych StatsBomb, ani jako wejścia, ani jako celu. Freeze frame'y StatsBomb służą tylko do pomiaru błędu wykrytych pozycji.
- **Sukces (Q4, Q21).** Górna granica przedziału 95%<!--n:rule.success_level--> z bootstrapu po meczach dla ΔLL (wariant − G-combo, log-loss na strzałach bez karnych) leży poniżej zera. Do testu idą najwyżej dwa warianty z korektą Bonferroniego; przy jednym wariancie zostaje przedział 95%<!--n:rule.success_level-->.
- **Koszt GPU (Q5, Q18).** W rozwoju pięć foldów i jeden seed. Pełne zespoły dziesięciu modeli tylko dla wariantów idących do testu. Zgoda na GPU dla całego planu, jedno zadanie naraz, każde w dzienniku.
- **Ekstraktor (Q7, Q15, Q19).** Modele i homografia z pakietu są jedynym źródłem prawdy. Klatki t0 i −400 ms, klatka −800 ms tylko do dociągania piłki, nigdy klatka po strzale. Cechy to wielkości z H3 w metrach z flagami pokrycia.
- **Bramka podejścia 2 (Q10), zamrożona przed audytem.** Mediana błędu miejsca strzału ≤ 3 m<!--n:au.med_max--> przy pokryciu ≥ 70%<!--n:au.cov_min--> sparowanych strzałów bez karnych i informacja o bramkarzu dla ≥ 50%<!--n:au.gk_min--> z nich.
- **Warianty podejścia 1 (Q11, Q12, Q13, Q20).** Najpierw A1 i A2 z recepturą G-combo bez zmian. Strzał bez detekcji zachowuje obraz i dostaje wyuczony wektor „brak”. A3 rusza tylko wtedy, gdy A1 albo A2 poprawi log-loss out-of-fold o co najmniej 0.002<!--n:rule.trigger--> punktowo.
- **Wybór konfiguracji (Q22).** Audyt wybiera spośród najwyżej czterech konfiguracji zadeklarowanych z góry, wyłącznie po błędzie pozycji w metrach, bez żadnej etykiety gola.

Po pierwszych ekstrakcjach właściciel zostawił regułę strzelca bez piłki (Q23), przyjął obecne pokrycie homografii (Q24) i próg instancji punktów kluczowych (Q25). Po audycie 1 postanowił poprawić homografię i przeprowadzić jeden, ostatni audyt 2 (Q26). Podejście 1 czekało na audyt 2 (Q27). Trzeciego audytu nie ma.

### 2.3 Mandat rady i decyzje C1, C2

Wieczorem 2026-10-12 właściciel przekazał prowadzenie tej fazy orkiestratorowi. Jego słowa: „przejął stery i z pomocą agentów, których będziesz traktować jako radę decyzyjną potraktował ten problem jako /goal i doprowadził tę fazę researchu do końca”. Rada ma trzy miejsca i każde głosuje niezależnie: M (metodolog i statystyk, Codex), V (inżynier widzenia komputerowego, Claude Opus) i S (sceptyk, Claude Fable). Decyduje większość. Rada nie może zmienić reguł wiążących właściciela: zakazu StatsBomb w modelach, zamrożonej bramki, braku trzeciego audytu, protokołu testu i G-combo bez zmian jako punktu odniesienia.

| Decyzja | Pytanie | Głosy M / V / S | Wynik |
|---|---|---|---|
| C1 | Jak prowadzić podejście 1 po audycie 2 | b / d / b | **(b)**: A1 i A2 zgodnie z planem, po zamrożeniu w commicie reguły sanitacji bez etykiet; bez wariantu A0 |
| C2 | Testować A1 czy zamknąć podejście 1 w rozwoju | c / c / b | **(c)**: test A1 jako zespołu dziesięciu modeli przy niezmienionej regule sukcesu, z opisowym odczytem równoważności (TOST, δ = 0.005<!--n:rule.delta-->) ustalonym przed testem |

Przy C1 wszyscy trzej uznali, że pozycje rzutowane setki metrów poza boisko nie mogą trafić do głowicy. Wariant A0, który proponował V, odrzucili M i S: dodawał ścieżkę selekcji i omijał bramkę A3. Przy C2 S głosował za zamknięciem bez testu, bo moc testu jest mała. Ten argument wszedł do reguły jako warunek: oczekiwaną moc podano przed testem (sekcja 6.2).

## 3. Audyty pozycji wobec StatsBomb

Audyty mierzą, jak dokładnie pipeline umieszcza na boisku miejsce strzału, bramkarza i obrońców. Prawdą są freeze frame'y StatsBomb dla 2 721<!--n:au.shots--> sparowanych strzałów bez karnych z 135<!--n:au.matches--> meczów ewaluacyjnych, klatka t0. Tylko te mecze mają freeze frame'y; pula treningowa (7 166<!--n:pl.pool_shots--> strzałów) nie ma takiej prawdy. To pomiar diagnostyczny. Audyt nie czyta żadnej kolumny z golem, a wybór konfiguracji i bramka widzą tylko pokrycie i błędy. Błędy zawierają szum czasu adnotacji obu źródeł, około 1 s. W tym czasie zawodnicy przesuwają się o 3–6 m.

### 3.1 Audyt 1 (ticket 02): STOP<!--n:au.gate--> przy 12.1 m<!--n:au.g.med-->

Spośród C1–C4 tylko C4<!--n:au.chosen--> miała pokrycie miejsca strzału co najmniej 70%<!--n:au.cov_min-->: 71.8% [67.3; 76.2]<!--n:au.c4.cov-->. Mediana błędu miejsca strzału wyniosła 12.1 m [11.4; 12.8]<!--n:au.c4.median--> przy progu 3 m<!--n:au.med_max-->. Informacja o bramkarzu: 60.2%<!--n:au.g.gk-->, mediana błędu bramkarza 6.9 m [6.4; 7.4]<!--n:au.c4.gk.med-->. Bramka: STOP<!--n:au.gate-->.

### 3.2 Poprawki homografii (ticket 08) i ich przyczyny

Ticket 08 szukał przyczyn na metryce bez etykiet i bez StatsBomb: zgodności rzutowanych linii boiska z namalowanymi liniami, na 1 500<!--n:hd.frames--> klatkach t0 z meczów treningowych. Sprawdził pięć podejrzeń:

1. **Wejście modelu punktów kluczowych: udowodnione, poprawione.** Model uczono na obrazach rozciągniętych do 640 × 640, a pakiet podawał mu klatkę z letterboxem. Na 316<!--n:hl.images--> własnych obrazach treningowych modelu mediana błędu punktów w rozdzielczości 1280 na 720 to 25.3 px<!--n:hl.letterbox--> z letterboxem i 9.5 px<!--n:hl.stretch--> z rozciągnięciem. Poprawka w pakiecie: `stretch=True` (`e927f40`).
2. **Definicje wierzchołków boiska: sprawdzone, bez błędu.** Wierzchołki pakietu zgadzają się z etykietami treningowymi modelu, mediana przesunięcia 0.21 m<!--n:hl.vertex-->.
3. **Kolejność punktów i lustrzane odbicia.** Kolejność jest dobra. Doprecyzowanie liniami dawało czasem obraz lustrzany, bo linie boiska są symetryczne. Pakiet odrzuca teraz takie wyniki, a pilnują tego testy regresyjne. Rzadkie lustrzane dopasowania z samych punktów są nadal przyjmowane (sekcja 8).
4. **Progi i RANSAC.** Zmiana progów punktów i limitu RANSAC prawie nie zmienia zgodności z liniami. Znalazł się za to błąd dopasowań osobliwych: gdy prawie wszystkie punkty leżą na jednej linii boiska, OpenCV zwraca macierz, która rzutuje całą klatkę na tę linię. Poprawka: `8994cc6`.
5. **Jakość modelu na klatkach SoccerNet: główna przyczyna, nie do poprawienia bez trenowania.** Nawet z rozciągniętym wejściem punkty kluczowe leżą w medianie 48 px<!--n:rf.stretch.med--> od punktów potwierdzonych liniami (312<!--n:rf.stretch.frames--> klatek; z letterboxem 86 px<!--n:rf.letterbox.med-->). W promieniu dziesięciu pikseli jest tylko 4%<!--n:rf.stretch.lt10--> z nich.

Ticket 08 dodał też opcjonalne doprecyzowanie homografii liniami boiska. Na klatkach treningowych homografia jest teraz w 71.3%<!--n:hd.d1.hom--> klatek zamiast 62.3%<!--n:hd.c4.hom-->. Dobrą zgodność z liniami miało 0.0%<!--n:hd.c4.good--> klatek C4, 0.9%<!--n:hd.d1.good--> po poprawce wejścia i 36.3%<!--n:hd.d3.good--> z doprecyzowaniem (konfiguracja D3). Błąd skali sprawdzany bez linii, mediana |log| stosunku wysokości zawodnika, spadł z 0.32<!--n:hd.c4.height--> do 0.12<!--n:hd.d3.height-->. Na tej metryce ticket 08 zadeklarował D1–D4 przed audytem 2.

### 3.3 Audyt 2 (ticket 09): STOP<!--n:a2.gate--> przy 10.5 m<!--n:a2.g.med-->

Wszystkie cztery konfiguracje D1–D4 przekroczyły 70%<!--n:au.cov_min--> pokrycia. Reguła wybrała D3<!--n:a2.chosen-->, z najniższą medianą; mediany D1–D4 różnią się o ułamki metra. Dla D3<!--n:a2.chosen-->: pokrycie miejsca strzału 77.4% [74.3; 80.5]<!--n:a2.d3.cov-->, mediana błędu 10.5 m [9.9; 11.1]<!--n:a2.d3.median-->, informacja o bramkarzu 66.5%<!--n:a2.g.gk-->. Bramka: STOP<!--n:a2.gate-->, ostatecznie.

### 3.4 Stałe zgadywanie i korelacje (opisowo)

Stałe zgadywanie to jedna pozycja dla wszystkich strzałów: mediana pozycji StatsBomb w całej populacji, oceniana na tych samych strzałach co detektor. To wyrocznia, bo zna rozkład prawdziwych pozycji.

| Wielkość (D3<!--n:a2.chosen-->) | detektor | stałe zgadywanie |
|---|---|---|
| mediana błędu miejsca strzału | 10.5 m [9.9; 11.1]<!--n:a2.d3.median--> | 10.7 m [10.4; 11.0]<!--n:a2.d3.const.med--> |
| mediana błędu bramkarza | 8.6 m [8.1; 9.1]<!--n:a2.d3.gk.med--> | 1.9 m [1.8; 2.1]<!--n:a2.d3.gkconst.med--> |

| Korelacja z pozycją StatsBomb (D3<!--n:a2.chosen-->) | x | y |
|---|---|---|
| miejsce strzału, Spearman | 0.41 [0.37; 0.45]<!--n:a2.d3.rank.x--> | 0.46 [0.42; 0.50]<!--n:a2.d3.rank.y--> |
| miejsce strzału, Pearson | 0.05 [0.01; 0.18]<!--n:a2.d3.corr.x--> | 0.09 [0.07; 0.17]<!--n:a2.d3.corr.y--> |
| bramkarz, Spearman | 0.00 [−0.06; 0.06]<!--n:a2.d3.gk.rank.x--> | 0.25 [0.19; 0.29]<!--n:a2.d3.gk.rank.y--> |

W medianie wykryte miejsce strzału nie jest bliżej prawdy niż typowe miejsce strzału, a bramkarz jest kilka razy gorszy od stałego zgadywania. Korelacja rang miejsca strzału jest umiarkowana, więc cechy niosą trochę informacji. Pearson jest niski przez pozycje wyrzucone setki metrów poza boisko.

Skąd zostaje błąd (opisowo, po wyborze). Tam, gdzie linie potwierdziły doprecyzowaną homografię (37.5%<!--n:dg.refined.share--> strzałów), mediana błędu D1 i D3 jest taka sama: 10.3 m<!--n:dg.refined.d1--> i 10.3 m<!--n:dg.refined.d3-->. Piłka z klatki −400 ms daje medianę 6.0 m [5.4; 7.0]<!--n:dg.ball_m400-->, a piłka z t0 11.9 m [11.3; 12.6]<!--n:dg.ball_t0-->. Grupa −400 ms jest wybrana (t0 nie miało piłki na boisku), więc to obserwacja, nie dowód. Wskazuje jednak, że duża część błędu bierze się z wyboru klatki i piłki, a nie z samego dopasowania homografii.

## 4. Podejście 2: zamknięte przez bramkę

Bramka z Q10 dała STOP<!--n:au.gate--> w audycie 1 i STOP<!--n:a2.gate--> w audycie 2. Zgodnie ze specyfikacją podejście 2 (ticket 05) zamknięto bez kodu. Nie powstał model xG z samych wykrytych pozycji ani jego stos z G-combo, więc raport nie ma opisowego wyniku podejścia 2, o który prosił ticket 07. Do testu poszedł jeden wariant, z przedziałem 95%<!--n:rule.success_level-->.

## 5. Podejście 1: wyniki rozwojowe

### 5.1 A1 i A2 wobec G-combo seed 0 (wyniki rozwojowe, ticket 03)

Przed treningiem (decyzja C1) zamrożono regułę sanitacji. Pozycja ponad 3 m poza boiskiem staje się brakiem. Bramkarz ponad 3 m poza polem karnym jest brakiem dla cech bramkarza. Liczniki liczą tylko poprawnych zawodników i są brakiem, nie zerem, gdy nie ma homografii albo drużyny strzelca. Hiperparametry gałęzi zapisano przed pierwszym przebiegiem, bez siatki. A1 to cechy geometryczne z t0 i −400 ms ze wskaźnikami braku, wpięte w głowicę G-combo. A2 to mała sieć CNN uczona od zera na rastrze boiska. Oba warianty trenowano z recepturą G-combo, pięć foldów i seed 0.

Wyniki out-of-fold na 7 062<!--n:ap.shots--> strzałach bez karnych z 328<!--n:ap.matches--> meczów treningowych (902<!--n:ap.goals--> gole). Przedziały 95%<!--n:ap.level--> z bootstrapu po meczach, te same losowania dla każdego wiersza. Δ = model − G-combo seed 0.

| Model | LL | ΔLL [przedział] | AUC | ECE | nachylenie kalibracji |
|---|---|---|---|---|---|
| G-combo seed 0 (punkt odniesienia) | 0.3446<!--n:ap.g0.ll--> | — | 0.732<!--n:ap.g0.auc--> | 0.014<!--n:ap.g0.ece--> | 0.89<!--n:ap.g0.slope--> |
| próg szumu: G-combo seed 1 | 0.3440<!--n:ap.g1.ll--> | −0.0006 [−0.0031; +0.0021]<!--n:ap.g1.d--> | 0.732<!--n:ap.g1.auc--> | 0.008<!--n:ap.g1.ece--> | 0.92<!--n:ap.g1.slope--> |
| A1 (cechy geometryczne) | 0.3439<!--n:ap.a1.ll--> | −0.0007 [−0.0032; +0.0018]<!--n:ap.a1.d--> | 0.734<!--n:ap.a1.auc--> | 0.012<!--n:ap.a1.ece--> | 0.87<!--n:ap.a1.slope--> |
| A2 (raster CNN) | 0.3446<!--n:ap.a2.ll--> | −0.0000 [−0.0021; +0.0020]<!--n:ap.a2.d--> | 0.731<!--n:ap.a2.auc--> | 0.012<!--n:ap.a2.ece--> | 0.88<!--n:ap.a2.slope--> |

- **Próg szumu.** Drugi seed samego G-combo zmienia log-loss o −0.0006<!--n:ap.g1.d_point-->, czyli tyle, ile A1 (−0.0007<!--n:ap.a1.d_point-->). Liczba rady z C1, czyli G-combo seed 0 − seed 1 na wszystkich strzałach puli razem z karnymi, to −0.0004 [−0.0032; +0.0023]<!--n:ap.council-->.
- **Wyzwalacz A3 niespełniony.** Próg to poprawa o 0.002<!--n:ap.trigger--> punktowo. A1 dał −0.0007<!--n:ap.a1.d_point-->, A2 −0.0000<!--n:ap.a2.d_point-->. Czy uruchomiono A3 (ticket 04): nie<!--n:ap.runs04-->.
- **Wybór rozwojowy.** Niższy log-loss out-of-fold miał A1<!--n:ap.selected-->. Poprawka mieści się w szumie seeda, a oba przedziały zawierają zero.

### 5.2 Zespoły dziesięciu modeli out-of-fold (wyniki rozwojowe, ticket 06)

Do testu A1 wytrenowano też z seedem 1, pięć foldów, ta sama receptura. Odtworzenie wszystkich dziesięciu modeli z dysku w nowym procesie dało maksymalną różnicę logitu 2.4 × 10⁻⁷<!--n:rp.replay-->. Oba zespoły skalibrowano tą samą procedurą co `ensemble.json` G-combo. Wyniki out-of-fold na tych samych strzałach: A1 ma LL 0.3419<!--n:en.a1.ll--> i AUC 0.736<!--n:en.a1.auc-->, G-combo LL 0.3427<!--n:en.g.ll--> i AUC 0.735<!--n:en.g.auc-->, ΔLL −0.0008 [−0.0026; +0.0011]<!--n:en.d-->. Platta dopasowano w obu ramionach na tych samych predykcjach, które się ocenia, więc to liczby opisowe.

## 6. Test na meczach odłożonych

### 6.1 Reguła (ticket 06, zamrożona przed pierwszą predykcją testową A1)

- **Ramiona.** A1 i G-combo, każde jako zespół dziesięciu modeli (pięć foldów × seedy 0 i 1). Agregacja jest ta sama (logity podzielone przez temperaturę, potem średnia) i kalibracja też (Platt na out-of-fold puli, korekta częstości, osobna ścieżka karnych).
- **Populacja.** Wszystkie strzały bez karnych z 172<!--n:rule.matches--> meczów odłożonych: 3 819<!--n:rule.shots--> strzałów, w tym 3 005<!--n:rule.eval_shots--> z meczów ewaluacyjnych i 814<!--n:rule.gate_shots--> z meczów bramkowych. Predykcje G-combo pochodzą z tabeli ewaluacyjnej xg-vendor (2 754<!--n:gt.paired--> sparowane strzały), z zapisanych predykcji bramkowych (825<!--n:gt.gate-->) i z inferencji tymi samymi modelami (285<!--n:gt.unpaired--> pozostałych). Ticket 06 sprawdził każde źródło.
- **Metryka.** Log-loss ważony strzałami, ΔLL = LL(A1) − LL(G-combo). Wartość ujemna znaczy, że A1 jest lepszy.
- **Bootstrap.** 5 000<!--n:rule.draws--> sparowanych losowań meczów, seed 0, przedziały percentylowe.
- **Sukces:** górna granica przedziału 95%<!--n:rule.success_level--> poniżej zera. To jedyne kryterium potwierdzające. **Pogorszenie:** dolna granica przedziału 95%<!--n:rule.success_level--> powyżej zera. **Równoważność (opisowo, C2):** przedział 90%<!--n:rule.eq_level--> z tych samych losowań leży w całości wewnątrz (−δ, +δ) dla δ = 0.005<!--n:rule.delta-->. Nigdy nie liczy się jako sukces i nie ratuje nieudanej reguły sukcesu.
- **Etykiety odczytu, w kolejności pierwszeństwa:** success and equivalent, success, worse, equivalent within δ, inconclusive. Raport podaje też najmniejszy margines równoważności: większą z wartości bezwzględnych granic przedziału 90%<!--n:rule.eq_level-->.
- **Zamrożenie.** Lista skrótów plików wejściowych, zamknięcie kodu skryptu testu i odcisk klatek t0 populacji są w `research/xg_detector_test_hashes.json` (commit `c4f66fe`). Skrypt odmawia pracy przy każdej niezgodności. Orkiestrator sprawdził regułę wobec C2 w zastępstwie właściciela przed testem.

### 6.2 Moc oczekiwana (wyniki rozwojowe, policzona przed testem)

Moc policzono z rozwojowej wariancji sparowanej A1 seed 0 − G-combo seed 0 (błąd standardowy z bootstrapu 0.00127<!--n:pw.se_dev-->), przeskalowanej do populacji testu. Oczekiwany błąd standardowy ΔLL w teście to 0.00173<!--n:pw.se-->. Przybliżenie normalne:

| zakładana prawdziwa ΔLL | P(sukces) | P(pogorszenie) | P(równoważność w δ) |
|---|---|---|---|
| −0.0007<!--n:pw.d0--> (wynik rozwojowy) | 6.0%<!--n:pw.p0--> | 0.9%<!--n:pw.w0--> | 75%<!--n:pw.e0--> |
| −0.002<!--n:pw.d1--> (wyzwalacz A3) | 21.1%<!--n:pw.p1--> | 0.1%<!--n:pw.w1--> | 53%<!--n:pw.e1--> |

**Zastrzeżenie mocy z reguły.** Moc reguły sukcesu jest mała. Przy prawdziwej różnicy rzędu rozwojowego sukces wypadłby w około 6.0%<!--n:pw.p0--> przypadków, więc ewentualny sukces trzeba by czytać z tym zastrzeżeniem. Brak sukcesu jest oczekiwanym wynikiem nawet wtedy, gdy A1 jest naprawdę odrobinę lepszy. Informację przy wyniku zerowym daje odczyt równoważności.

### 6.3 Wynik testu

Test uruchomiono raz, 2026-10-13 od 03:16:43 do 03:30:47 UTC, przy commicie e14bb79<!--n:ts.git-->, po udanym `--check`. Drzewo robocze było czyste (zmiany w plikach śledzonych: nie<!--n:ts.dirty-->). Skrypt zapisał predykcje obu ramion i ich rekord przed odczytem etykiet. Nic nie odrzucono i niczego nie powtórzono.

Populacja: 3 819<!--n:ts.shots--> strzałów bez karnych z 172<!--n:ts.matches--> meczów, 488<!--n:ts.goals--> goli.

| Wielkość | Wartość |
|---|---|
| ΔLL (A1 − G-combo) | +0.0009<!--n:ts.d--> |
| przedział 95%<!--n:rule.success_level--> (reguła sukcesu) | [−0.0012; +0.0030]<!--n:ts.ci95--> |
| przedział 90%<!--n:rule.eq_level--> (odczyt równoważności) | [−0.0008; +0.0027]<!--n:ts.ci90--> |
| sukces | nie<!--n:ts.success--> |
| pogorszenie | nie<!--n:ts.worse--> |
| równoważność w δ = 0.005<!--n:rule.delta--> | tak<!--n:ts.equivalent--> |
| **odczyt** | **equivalent within δ<!--n:ts.label-->** |
| najmniejszy margines równoważności | 0.0027<!--n:ts.margin--> |

| Ramię | LL | AUC |
|---|---|---|
| A1, zespół dziesięciu modeli | 0.3386<!--n:ts.a1.ll--> | 0.739<!--n:ts.a1.auc--> |
| G-combo, zespół dziesięciu modeli | 0.3376<!--n:ts.g.ll--> | 0.742<!--n:ts.g.auc--> |

**Rekord ukończenia** (`results/xgd_test/complete.json`) wskazuje listę skrótów testu e1011b08920755762b79c154886d09f2b7e326cb866e033fc9c360dd6c428c91<!--n:ts.hashlist-->. Ta lista przypina każde wejście: regułę, moc, dwadzieścia modeli z rekordami, kalibracje obu zespołów, predykcje G-combo, tabelę D3 po sanitacji, plan podziału i foldy, indeks, maskę nakładki, wagi DINOv2 i odcisk klatek. Rekord zawiera też skróty kodu załadowanego w procesie (równe zamrożonemu zamknięciu), commit i skróty plików wyjściowych. Kontrola ścieżki przed testem: model A1 z foldu 0 i seeda 0 odtworzył swoje logity na strzałach puli z różnicą 6.9 × 10⁻⁸<!--n:ts.path-->.

### 6.4 Odczyt według reguły (wynik testu)

Odczyt to **equivalent within δ<!--n:ts.label-->**. Przedział 95%<!--n:rule.success_level--> zawiera zero, więc reguła sukcesu nie jest spełniona i pogorszenia też nie stwierdzono. Przedział 90%<!--n:rule.eq_level--> leży wewnątrz (−δ, +δ), więc opisowy odczyt równoważności zachodzi. Najmniejszy margines, przy którym by zachodził, to 0.0027<!--n:ts.margin-->. Zgodnie z regułą równoważność nie jest sukcesem. Wynik czytamy z zastrzeżeniem mocy z sekcji 6.2.

Zakres, zdanie ustalone w regule: „The result applies to the D3 detector and homography as audited in ticket 09, not to detector-derived structure in general.” Wynik dotyczy detektora i homografii D3 w stanie z audytu 2 (ticket 09), a nie struktury z detektora w ogóle.

## 7. Ograniczenia

- **Szum czasu adnotacji.** Błędy audytów zawierają przesunięcie czasu adnotacji SoccerNet i StatsBomb, około 1 s, w którym zawodnicy przesuwają się o 3–6 m. To zawyża błędy, ale nie tłumaczy ich wielkości: bramkarz stoi zwykle blisko bramki, a stałe zgadywanie jest kilka razy lepsze od detektora.
- **Dane treningowe detektora.** Większość obrazów treningowych detektora to klipy Lecha Poznań, głównie z Ekstraklasy (Roboflow, licencja Creative Commons). Źródło pozostałych jest nieznane. Klatek SoccerNet nie ma ani w danych detektora, ani w danych modelu punktów kluczowych. Badanie używa gotowych modeli, nie nagrań z Ekstraklasy.
- **Etykiety meczów ewaluacyjnych były widziane w xg-vendor.** Żaden model tego badania nie był na nich wybierany, ale wyniki G-combo na tych meczach były znane. Mecze bramkowe G-combo ocenił w bramce ponownego treningu xg-vendor.
- **Wybór konfiguracji na klatkach meczów testowych (Q22).** Oba audyty wybierały konfigurację na sparowanych strzałach z meczów ewaluacyjnych, tylko po błędzie pozycji, bez etykiet goli.
- **Margines δ dobrany po rozwoju.** δ = 0.005<!--n:rule.delta--> przyjęto po wynikach rozwojowych i pożyczono z xg-vendor. Jest luźny: to około 1.5%<!--n:rule.delta_share--> log-loss G-combo i kilka razy więcej niż różnica rozwojowa.
- **Zmienność ponownego treningu.** Bootstrap po meczach losuje mecze dla ustalonych modeli. Inny zestaw seedów dałby inny zespół, a przedziały tej zmienności nie obejmują.
- **Rozwój na pojedynczych modelach.** Porównania A1 i A2 z G-combo oraz wyzwalacz A3 opierały się na jednym seedzie na fold. Różnica A1 była tej samej wielkości co różnica między dwoma seedami samego G-combo.
- **Kod seeda 1.** Modele A1 z seedem 1 trenowano przy kodzie z ticketu 06. Ma on inne skróty niż kod seeda 0 (refaktoryzacja `9b7eea9`, która nie zmienia wyników, i blokada etykiet), ale wejścia i wyjścia są te same: tabela po sanitacji odtwarza się identycznie, a odtworzenia seeda 0 przechodzą przy nowym kodzie. Foldy 1–4 obu seedów mają w rekordach `dirty: true` przez edycję plików, których trening nie ładuje.
- **Mała moc** (sekcja 6.2).

## 8. Otwarte dla właściciela

1. **Model punktów kluczowych uczony na klatkach podobnych do SoccerNet.** To prawdziwa dźwignia dla homografii: obecny model myli się w medianie o 48 px<!--n:rf.stretch.med--> na klatkach SoccerNet, a o 9.5 px<!--n:hl.stretch--> na własnych obrazach. Trzeba przy tym pamiętać o sekcji 3.4: część błędu miejsca strzału pochodzi z wyboru klatki i piłki. Wyniki takiego modelu mogłyby wejść do tego badania tylko przez trzeci audyt, a ten wyklucza reguła Q26. To decyzja właściciela i praca produktowa.
2. **Poprawki pakietu do mastera** (osobne commity w `src/tactifoot_vision`). Z ticketu 08: wejście `stretch=True` modelu punktów (`e927f40`), odrzucanie osobliwych homografii (`8994cc6`), `pitch_markings` w `tv.pitch` (`99bd7a0`), opcjonalne doprecyzowanie liniami z kontrolami (`4725862`, `30702ec`, `9a6289b`, `639b86f`, `ec487a6`, `0243771`), dokumentacja (`08c9606`) i poprawki po przeglądzie (`56e288f`). Z ticketu 01: bezstanowa homografia (`446d51a`), próg pewności na klasę (`66239e0`), tryb drużyn z jednej klatki (`316993b`), analiza pojedynczej klatki z dociąganiem piłki (`c59f976`) i poprawki po przeglądzie (`6936ca3`). Otwarte przy nich: odrzucanie lustrzanych dopasowań bez doprecyzowania (wymaga odwrócenia fixture'ów testowych) i `stretch` dla detektora.
3. **Notatniki** `00_overview`, `04_inference` i `05_visualisation` mają nowy kod i stare wyjścia. Trzeba je wykonać ponownie (`notebooks/build/build.py --execute`) tam, gdzie jest `models/pitch_yolov8n_pose.pt`.
4. **SciPy w `pyproject.toml`.** Doprecyzowanie liniami używa SciPy, który przychodzi ze scikit-learn. Jawna zależność wymaga `uv lock`, a ten nie działał bez sieci.
5. **Przegląd termonuklearny** uruchamia właściciel.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `research/xg_detector_test_rule.json` | `4d593f52286e8c0ea46a39c8c07097c70de5be15905b924190d1745578fa87ca` | parametry zamrożonej reguły testu (ticket 06) |
| `research/xg_detector_test_rule.md` | `87c97619a96b381324a1dce1b17eefec2c1ada5214267e974b3a0c4cf8174e4c` | tekst zamrożonej reguły testu (ticket 06) |
| `results/xgd_a1/replay_ensemble.json` | `729529d1601651ddd2a79b70b57a2813ff45d2d2093f494f8074cf4e783ccf0b` | ticket 06: odtworzenie dziesięciu modeli A1 |
| `results/xgd_a1_ensemble/comparison.json` | `2debf5719f9731aff43efbd42a808203f57d8a6f6cb4e870c02262234baa4443` | ticket 06: zespoły dziesięciu modeli out-of-fold |
| `results/xgd_approach1/comparison.csv` | `263f6621a72aa2a94b61b2972788bec2f84ce42a9592dd52ef7e4a9c73b258ce` | ticket 03: wyniki out-of-fold A1, A2, G-combo |
| `results/xgd_approach1/summary.json` | `b61c8482481649893366f16a16ef18d84afaf27b1f9faabb085e794bc7ed663f` | ticket 03: populacja rozwojowa, próg szumu, wyzwalacz A3, wybór |
| `results/xgd_audit/choice.json` | `0e7bc0b9910bd32deeef34628cf2bfa4aa2817f9bb198c51483bad7b6dd56f71` | audyt 1: wybór konfiguracji i bramka |
| `results/xgd_audit/complete.json` | `876e4a7717487b6c4db58d7d500aafeca15c03cbced397a81f029afc21f97e15` | audyt 1: populacja i parametry bramki |
| `results/xgd_audit/metrics.csv` | `094eb796175da304a1e650f2aaa09500d08878d2562887ea2f56788137077280` | audyt 1: metryki |
| `results/xgd_audit2/choice.json` | `0d5d8275322ced76a87bdc29b590e1fa97ec1c96c9d82cf4378ccd227f5432c9` | audyt 2: wybór konfiguracji i bramka |
| `results/xgd_audit2/metrics.csv` | `e422a0a2311a924a4b4e2f2ed895b264360437f8bd2ddf93fd37e156722420ab` | audyt 2: metryki, korelacje, stałe zgadywanie |
| `results/xgd_audit2_diag/diag.csv` | `32f45780139e42676b25fd81417d5bd9c2ef5f22b88995af9e59b2a47bd1776e` | audyt 2: podziały błędów (opisowo) |
| `results/xgd_gcombo_test/checks.json` | `e4f502e0b1ab07ca1d5391e090fd721f75e96bb0f0ddcc33318a3325f0b90a01` | ticket 06: źródła predykcji testowych G-combo |
| `results/xgd_homography_dev/labels/summary.json` | `a55a4fe165dc37ec13d819dfc2ddb6032463c2d6692b3b585307ab412337bb61` | ticket 08: błąd punktów na obrazach treningowych modelu, wierzchołki |
| `results/xgd_homography_dev/summary.csv` | `0369a175a078e4c8f8c6d48d4576d85cf9c0c7de09ad476d411bbb582559e93f` | ticket 08: metryka linii na klatkach treningowych |
| `results/xgd_report_figures/figures.json` | `8d70c0017a13249aed3223d42ecbf595218400d5d59b29be1c0955b6dc465409` | mediana błędu punktów kluczowych na klatkach SoccerNet (z ticketu 08) |
| `results/xgd_test/complete.json` | `640946699319841cf0d895db78a1c2f5296f2380490956cdb6aaded28a88ed8c` | ticket 07: rekord ukończenia testu |
| `results/xgd_test/result.json` | `ee3fc40671bb8a500b250e33012bc5ad15f0e11958ddb8da22f87db0111565c3` | ticket 07: wynik testu |
| `results/xgd_test_power/power.json` | `1fc98b8c11182b8f2047c1c52c0deded2a8735a1d8e3a59774bd33b9ebbe2473` | ticket 06: moc oczekiwana |
| `results/xgv_split_plan/plan.json` | `80162d10c59908893d86e0bc905be1623cae9d20ab17dd4f5607df42f8f4f3ce` | plan podziału xg-vendor: liczby meczów i strzałów według roli |
