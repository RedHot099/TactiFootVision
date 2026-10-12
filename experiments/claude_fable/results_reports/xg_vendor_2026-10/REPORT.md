# xG z jednej klatki transmisji wobec xG dostawcy i xG pozycyjnego: raport badania xg-vendor

Jakub Eichner, 2026-10-12. Gałąź badawcza `claude-fable/xg-poc`, badanie xg-vendor, tickety 01–16.

Data: StatsBomb. Wszystkie liczby o strzałach StatsBomb, xG dostawcy i modelu pozycyjnym pochodzą z otwartych danych StatsBomb (commit `4b73468f`). W publikacji każda tabela i każdy rysunek z tych danych dostaje podpis "Data: StatsBomb" i logo StatsBomb.

Badanie pyta, jak xG z jednej klatki transmisji wypada wobec xG, które sprzedaje dostawca danych (StatsBomb), i wobec xG liczonego z danych pozycyjnych StatsBomb, na tych samych strzałach. Model klatki to G-combo, dziesięć modeli foldu dostrojonego DINOv2-L. Plan analizy ustaliła pre-rejestracja `research/xg_vendor_preregistration.md` z commita `cc82d22`. Właściciel zatwierdził ją 2026-10-12, zanim połączyliśmy etykiety. Etykiety połączyliśmy raz i analiza przebiegła raz.

**Trzy rodzaje liczb.** Raport ich nie miesza.

- Wyniki konfirmacyjne pochodzą z jedynego przebiegu na zatwierdzonym kodzie (`results/xgv_confirmatory/`). Sekcje z nimi mają w tytule dopisek "wyniki konfirmacyjne". H2 i H3 są konfirmacyjne, H1 i H4 opisowe, choć liczone w tym samym przebiegu.
- Wejścia zamrożone przed łączeniem etykiet: parowanie, bramki K0 i K1, test bramkowy retreningu i werdykt źródła czasu. Żadne z nich nie czytało etykiet tej analizy.
- Wyniki rozwojowe pochodzą z prototypu, z tabeli deweloperskiej i z symulacji mocy. Liczyły je modele, które widziały te mecze w treningu. Stoją tylko w sekcji 11, która ma w tytule dopisek "wyniki rozwojowe".

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `claims.toml` w tym folderze mówi, z którego pliku wyników i z którego miejsca ta liczba pochodzi. Skrypt `scripts/check_vendor_report.py` odczytuje każdą liczbę ze źródła. Kończy się błędem przy niezgodności, przy liczbie bez znacznika, przy wyniku konfirmacyjnym poza sekcją konfirmacyjną i przy pliku, którego hash nie zgadza się z tabelą źródeł na końcu raportu. Przedziały w nawiasach kwadratowych to percentylowe przedziały 95%<!--n:pg.ci95--> z bootstrapu po meczach. Bootstrap ma 5 000<!--n:run.n_boot--> losowań, te same dla każdego kontrastu i każdej hipotezy. Różnice log-lossu liczymy jako A − B, więc ujemna różnica znaczy, że A przewiduje lepiej.

## 1. Wnioski (wyniki konfirmacyjne)

- **H2, konfirmacyjna.** Dodanie klatki do modelu pozycyjnego zmieniło log-loss o −0.0004 [−0.0026; +0.0018]<!--n:cf.h2.p.diff-->. Test równoważności TOST przy δ = 0.005<!--n:run.delta--> wykazał równoważność, p = 0.0006<!--n:cf.h2.p.tost-->. Odczyt według reguły pre-rejestracji brzmi "równoważne przy δ: klatka nie dodaje więcej niż δ"<!--n:cf.h2.p.reading-->. Najmniejszy margines, przy którym równoważność zachodzi, to 0.0022<!--n:cf.h2.p.margin-->.
- **H3, konfirmacyjna.** Z embeddingu modelu foldu 0 sonda liniowa odczytuje odległość strzału, R² 0.711 [0.678; 0.739]<!--n:cf.h3.f0.dist--> przy progu 0.50<!--n:cf.h3.thr.dist-->, i obecność obrońcy w stożku strzału, AUC 0.761 [0.743; 0.778]<!--n:cf.h3.f0.cone--> przy progu 0.70<!--n:cf.h3.thr.cone-->. Oba cele dostały odczyt "koduje". Kontrola negatywna przeszła dla wszystkich trzech enkoderów.
- **H1, opisowa.** Sama klatka ma log-loss wyższy o +0.0283 [+0.0168; +0.0392]<!--n:cf.c.p.frame_vendor--> niż xG dostawcy i o +0.0272 [+0.0162; +0.0380]<!--n:cf.c.p.frame_ff--> niż nasz model pozycyjny. Przedział różnicy dostawca − pozycje, −0.0010 [−0.0076; +0.0057]<!--n:cf.c.p.vendor_ff-->, obejmuje zero.
- **H4, opisowa.** Wśród strzałów bez gola różnica logitów klatka − dostawca jest na strzałach celnych wyższa o +0.056 [+0.004; +0.108]<!--n:cf.h4.main--> logitu przy stałych kontrolach. To zgadza się z dwoma wyjaśnieniami, których dane nie rozdzielają. Klatka może czytać kierunek piłki po kontakcie. Może też widzieć coś sprzed kontaktu, czego dostawca nie zapisuje, na przykład równowagę strzelca.

## 2. Przebieg badania

Pre-rejestracja ustaliła hipotezy, populacje, etykiety, margines δ, rekalibrację, bootstrap, progi sond, szablony tabel S1–S9 i obie narracje każdej hipotezy konfirmacyjnej. Lista hashy `research/xg_vendor_prereg_hashes.json` przypina kod analizy (32<!--n:pg.code_files--> pliki), tabelę predykcji bez etykiet, plan podziału, predykcje komparatora, werdykt źródła czasu i dowody z dwóch dry runów na permutowanych etykietach.

Właściciel zatwierdził decyzje A–F tak, jak je zalecono:

| Decyzja | Wartość |
|---|---|
| A, bootstrap szczebli rekalibrowanych | modele dopasowywane od nowa w każdym losowaniu (`refit`) |
| B, wiersz wtórny z uszkodzoną klatką | wykluczony; klatka pokazuje inną chwilę niż strzał |
| C, cele wtórne bez klatki | zostają w tabeli, wypadają z H1 i H2, tabela S1 je liczy |
| D, gole StatsBomb bez gola SoccerNet | wykluczone według flagi zamrożonego parowania |
| F, próg kontroli negatywnej sond | R² kontroli najwyżej 0.005<!--n:run.gate_thr--> |
| E | zatwierdzenie całego dokumentu |

Etykiety połączył logowany krok 2026-10-12 o 11:57:06 UTC, po rekordzie pre-rejestracji z 09:08:21 UTC (`results/xgv_confirmatory/label_join_log.jsonl`). Skrypt `scripts/vendor_confirm.py` przed połączeniem przeliczył listę hashy i porównał ją bajt w bajt z zatwierdzoną. Hash dokumentu pre-rejestracji w zapisie łączenia to 312e4e4a3282<!--n:run.prereg_sha-->…. Trzy skrypty analizy zakończyły się kodem 0. Nie było powtórki ani odstępstwa. Rekord ukończenia `results/xgv_confirmatory/complete.json` wskazuje commit `cc82d22` z czystym drzewem.

## 3. Wejścia zamrożone przed łączeniem etykiet

### 3.1 Parowanie i bramka K0

Otwarte dane StatsBomb pokrywają 135<!--n:pa.matches--> z 500<!--n:pa.sn_matches--> meczów SoccerNet: sezon 2015/16 pięciu lig i sezony La Ligi z Barceloną. Mecze paruje reguła na metadanych Labels-v2: te same rozgrywki, data w granicy jednego dnia, dokładny wynik i podobieństwo nazw klubów z listą aliasów. Strzały paruje przypisanie jeden do jednego w oknie 2.5 s<!--n:pa.window--> po wyznaczeniu offsetu każdej połowy. Druga implementacja reguł, napisana bez wglądu w pierwszą, dała te same pary.

- Sparowanych jest 2 754<!--n:pa.paired--> z 3 039<!--n:pa.index_shots--> strzałów indeksu SoccerNet, czyli 90.62%<!--n:pa.rate-->. K0 wymaga co najmniej 85%<!--n:pa.k0_rate-->.
- Gol SoccerNet i gol StatsBomb zgadzają się na 2741/2754<!--n:pa.goal_agree--> parach, czyli 99.53%<!--n:pa.goal_rate-->. K0 wymaga co najmniej 99%<!--n:pa.k0_goal-->.
- K0 przechodzi<!--n:pa.k0-->.

### 3.2 Komparator pozycyjny i bramka K1

Komparator to gradient boosting na zdarzeniach i freeze frame StatsBomb. Uczył się raz, na 36 626<!--n:co.pool_shots--> strzałach z 1 491<!--n:co.pool_matches--> meczów spoza 135<!--n:pa.matches--> meczów badania. Jego log-loss w walidacji krzyżowej puli wynosi 0.24607<!--n:co.k1_ff-->, a xG dostawcy na tych samych strzałach 0.24732<!--n:co.k1_vendor-->. Różnica to −0.0012<!--n:co.k1_gap-->, a K1 dopuszcza najwyżej +0.01<!--n:co.k1_margin-->. K1 przechodzi<!--n:co.k1_pass-->. Wartość K1 to minimum po siatce ustawień, więc jest trochę za optymistyczna. Pre-rejestracja szacuje tę przewagę z selekcji na rząd jednej dziesięciotysięcznej log-lossu.

Szczeble komparatora to w drabinie: stała, odległość i kąt, zdarzenia oraz freeze frame. Freeze frame nazywamy też "pozycjami".

### 3.3 G-combo wytrenowany od nowa i test bramkowy

Archiwalny G-combo miał 72<!--n:sp.eval_train--> mecze badania w zbiorze treningowym i 31<!--n:sp.eval_valid--> w walidacyjnym. Dlatego wytrenowaliśmy go od nowa archiwalną recepturą na 328<!--n:sp.training--> meczach spoza badania. Test bramkowy porównał nowy model z archiwalnym na 37<!--n:gc.matches--> meczach, których żaden z nich nie widział, z 825<!--n:gc.shots--> strzałami.

| Metryka | Nowy | Archiwalny | Różnica | Tolerancja |
|---|---|---|---|---|
| log-loss | 0.32655<!--n:gc.ll_new--> | 0.32380<!--n:gc.ll_old--> | +0.00275 [−0.00096; +0.00650]<!--n:gc.ll_diff--> | +0.010<!--n:gc.tol_ll--> |
| AUC | 0.79240<!--n:gc.auc_new--> | 0.79884<!--n:gc.auc_old--> | −0.00643 [−0.01721; +0.00465]<!--n:gc.auc_diff--> | −0.020<!--n:gc.tol_auc--> |

Test bramkowy przechodzi<!--n:gc.pass-->. Rozstrzygają estymaty punktowe, przedziały tylko informują.

### 3.4 Test źródła czasu

Kotwica strzału SoccerNet i czas strzału StatsBomb zgadzają się bardzo ściśle. Mogło to znaczyć, że obie pochodzą z jednego zegara. Test porównał więc inne zdarzenia, przede wszystkim zmiany zawodników, które SoccerNet oznacza wtedy, gdy pokazuje je transmisja. Regułę zapisaliśmy przed porównaniem (`research/xg_vendor_timing_rule.md`).

| Wielkość | Wartość | Próg reguły |
|---|---|---|
| zmiany w zbiorze reguły | 686<!--n:ti.n_subs--> | co najmniej 50<!--n:ti.min_subs--> |
| zmiany w ±200 ms<!--n:ti.agree_ms--> | 13.3%<!--n:ti.subs_share--> | od 90%<!--n:ti.shared--> wspólne źródło |
| mediana \|r\| zmian | 1788 ms<!--n:ti.subs_median--> | od 1000 ms<!--n:ti.lag--> niezależne adnotacje |
| strzały w ±200 ms, offset leave-one-out | 96.6%<!--n:ti.shots--> | co najmniej 80%<!--n:ti.kick--> |
| rzuty rożne w ±200 ms | 82.1%<!--n:ti.corners--> | co najmniej 80%<!--n:ti.kick--> |

Werdykt brzmi "independent"<!--n:ti.verdict-->. Czas StatsBomb to osobna adnotacja tej samej transmisji, a nie kopia zegara SoccerNet. Kopiowany zegar nie dawałby zmian przesuniętych o sekundy. Margines rzutów rożnych nad progiem był cienki, 0.021<!--n:pg.corner_margin-->. Werdykt ma dwa skutki. Ramię reszty czasowej w H4 biegnie, zawsze z ograniczeniem do ±200 ms jako wrażliwością. Klatki populacji wtórnej, wycięte w czasie StatsBomb, mogą leżeć kilkaset milisekund od kontaktu (sekcja 12).

## 4. Populacje i ich audyt (wyniki konfirmacyjne: S1)

Populacja pierwotna to strzały sparowane. Populacja wtórna to strzały StatsBomb, których SoccerNet nie oznaczył, i gole SoccerNet bez adnotacji strzału, odzyskane jeden do jednego. Populacja łączona to obie razem. H2 rozstrzyga populacja pierwotna, łączona służy tylko do sprawdzenia odporności. Etykieta gola pochodzi zawsze z SoccerNet (decyzja D4).

Tabela S1 z przebiegu konfirmacyjnego. Każdy wiersz liczy się raz, pod pierwszą przyczyną od lewej:

| Populacja | Wybrane | Wykluczone przy etykietach | Karne | Klatka wykluczona | Bez klatki | Analizowane | Gole | Mecze |
|---|---|---|---|---|---|---|---|---|
| pierwotna | 2 754<!--n:cf.s1.p.chosen--> | 0<!--n:cf.s1.p.label--> | 33<!--n:cf.s1.p.pen--> | 0<!--n:cf.s1.p.fx--> | 0<!--n:cf.s1.p.nf--> | 2 721<!--n:cf.s1.p.n--> | 374<!--n:cf.s1.p.goals--> | 135<!--n:cf.s1.p.matches--> |
| łączona | 3 601<!--n:cf.s1.c.chosen--> | 12<!--n:cf.s1.c.label--> | 40<!--n:cf.s1.c.pen--> | 1<!--n:cf.s1.c.fx--> | 147<!--n:cf.s1.c.nf--> | 3 401<!--n:cf.s1.c.n--> | 417<!--n:cf.s1.c.goals--> | 135<!--n:cf.s1.c.matches--> |

Wiersze populacji łączonej spoza populacji pierwotnej mają klatkę w czasie StatsBomb plus offset połowy, nie w kotwicy SoccerNet (sekcja 12).

Skąd biorą się te liczby:

- **Pierwotna.** Strzały sparowane leżą w meczach treningowych, walidacyjnych i testowych oficjalnego podziału SoccerNet: 1 478<!--n:pa.p_train-->, 639<!--n:pa.p_valid--> i 637<!--n:pa.p_test-->. Po odjęciu 33<!--n:cf.s1.p.pen--> karnych zostaje 2 721<!--n:cf.s1.p.n--> strzałów.
- **Wtórna.** Do populacji wtórnej należy 797<!--n:pa.secondary--> strzałów tylko ze StatsBomb i 50<!--n:pa.recovered--> odzyskanych goli. Odzyskane gole to 50<!--n:pa.recovered--> z 74<!--n:pa.dropped--> goli SoccerNet bez adnotacji strzału. Klatki tych strzałów wycięliśmy z ponownie pobranych połówek meczów SoccerNet: 209<!--n:rd.cut--> połówek, żadnej odrzuconej przez sumę kontrolną.
- **Bez klatki.** 147<!--n:cf.s1.c.nf--> celów wtórnych leży po końcu nagrania połowy, bo StatsBomb liczy doliczony czas, a wiele nagrań SoccerNet kończy się wraz z regulaminowym czasem połowy. Wszystkie to strzały tylko ze StatsBomb, z 86<!--n:pg.nf_matches--> meczów. Wypadają z każdego szczebla H1 i H2. Populacji łączonej brakuje więc 147<!--n:cf.s1.c.nf--> z 797<!--n:pa.secondary--> strzałów tylko ze StatsBomb, głównie z doliczonego czasu.
- **Klatka wykluczona.** Jeden strzał tylko ze StatsBomb (Stoke City przeciw Chelsea, pierwsza połowa) ma klatkę uszkodzoną przy dekodowaniu, z zegarem transmisji kilkanaście sekund przed strzałem. Wypada z populacji łączonej (decyzja B).
- **Wykluczone przy etykietach.** 12<!--n:cf.s1.c.label--> strzałów tylko ze StatsBomb to gole StatsBomb, którym nie odpowiada żaden gol SoccerNet. Etykieta SoccerNet mówiłaby "brak gola", więc zostają w tabeli, ale wypadają z analiz (decyzja D).
- **Karne.** 33<!--n:cf.s1.p.pen--> karne populacji pierwotnej i 7<!--n:pg.recovered_pen--> karnych wśród odzyskanych goli raportujemy osobno (sekcja 10).
- Rachunek populacji łączonej: 3 601<!--n:cf.s1.c.chosen--> wybranych − 12<!--n:cf.s1.c.label--> − 40<!--n:cf.s1.c.pen--> − 1 − 147<!--n:cf.s1.c.nf--> = 3 401<!--n:cf.s1.c.n--> analizowanych.

Czego populacja pierwotna nie widzi. SoccerNet oznacza tylko część strzałów StatsBomb: z zablokowanych ma 515/890<!--n:pa.blocked-->, z niecelnych "Wayward" 47/146<!--n:pa.wayward-->. Z 875<!--n:pa.sb_unpaired--> strzałów StatsBomb bez pary 797<!--n:pa.secondary--> tworzy populację wtórną, 50<!--n:pa.recovered--> to odzyskane gole, 26<!--n:pa.mispairs--> leży w ciągu 10 s<!--n:pa.mispair_s--> od niesparowanego strzału SoccerNet i wypada jako możliwa błędna para, a 2 to karne.

Z drugiej strony 285<!--n:ti.unpaired--> strzałów SoccerNet nie ma pary. 36<!--n:ti.r_shot--> z nich ma strzał StatsBomb w ciągu 10 s<!--n:pa.mispair_s-->. Pozostałe 249<!--n:pa.no_sb_10s--> klasyfikuje najbliższe zdarzenie StatsBomb dowolnego typu. 176<!--n:ti.r_other--> ma inne zdarzenie w ciągu 2.5 s<!--n:pa.window-->, zwykle podanie albo przyjęcie, więc StatsBomb zapisał tę akcję jako podanie. 52<!--n:ti.r_dead--> przypada na martwą piłkę: aut, spalony albo faul. 21<!--n:ti.r_none--> nie ma żadnego zdarzenia w ciągu 2.5 s<!--n:pa.window-->.

## 5. Co widzi które źródło

Tabela opisuje informację, a nie wyniki. "Zdarzenia" i "pozycje" to szczeble naszego komparatora. O modelu dostawcy wiemy tyle, ile podaje StatsBomb.

| Informacja | Klatka (G-combo) | Zdarzenia | Pozycje (freeze frame) | xG dostawcy |
|---|---|---|---|---|
| położenie strzelca na boisku | pośrednio, z perspektywy kamery | tak, współrzędne | tak | tak |
| część ciała, technika, typ strzału | część ciała zwykle widać (sekcja 8), techniki nie zawsze | tak | tak | tak |
| podanie przed strzałem: dośrodkowanie, prostopadłe, wycofanie | nie | tak | tak | tak |
| presja, strzał z pierwszej piłki | pośrednio, jeśli rywal jest w kadrze | tak, jako flagi StatsBomb | tak | tak |
| obrońcy i bramkarz w chwili strzału | tylko ci w kadrze kamery | nie | tak, zawodnicy zapisani przez StatsBomb | dane je mają; jak model ich używa, nie wiemy |
| wysokość piłki przy uderzeniu | nie, jedna klatka nie ma głębi | nie | nie | tak, według opisu modelu dostawcy |
| postawa i równowaga strzelca | tak, jeśli kamera je pokazuje | nie | nie | nie |
| to, co dzieje się po kontakcie | możliwe, gdy kotwica leży po kontakcie (H4) | nie, pola po kontakcie wyłączone | nie | nie, xG liczy się przed strzałem |

## 6. H1: drabina informacji (wyniki konfirmacyjne, opisowe)

Każdy szczebel jest rekalibrowany tak samo: Platt dopasowany krzyżowo na pięciu foldach meczów populacji, z foldami wspólnymi dla wszystkich szczebli. Stosy to regresja logistyczna na logitach dwóch szczebli, dopasowana na tych samych foldach. H1 jest opisowa. Żadne zdanie tej sekcji nie opiera się na wartości p.

| Szczebel | Log-loss, pierwotna | AUC, pierwotna | Log-loss, łączona |
|---|---|---|---|
| stała | 0.4006 [0.3751; 0.4271]<!--n:cf.m.p.constant--> | 0.477<!--n:cf.a.p.constant--> | 0.3722<!--n:cf.m.c.constant--> |
| odległość i kąt | 0.3518 [0.3257; 0.3783]<!--n:cf.m.p.dist_angle--> | 0.741<!--n:cf.a.p.dist_angle--> | 0.3309<!--n:cf.m.c.dist_angle--> |
| zdarzenia | 0.3333 [0.3076; 0.3591]<!--n:cf.m.p.event--> | 0.778<!--n:cf.a.p.event--> | 0.3091<!--n:cf.m.c.event--> |
| pozycje (freeze frame) | 0.3275 [0.3016; 0.3538]<!--n:cf.m.p.freeze_frame--> | 0.788<!--n:cf.a.p.freeze_frame--> | 0.3033<!--n:cf.m.c.freeze_frame--> |
| xG dostawcy | 0.3264 [0.3016; 0.3515]<!--n:cf.m.p.vendor--> | 0.792<!--n:cf.a.p.vendor--> | 0.3015<!--n:cf.m.c.vendor--> |
| klatka (G-combo) | 0.3547 [0.3298; 0.3802]<!--n:cf.m.p.frame--> | 0.736<!--n:cf.a.p.frame--> | 0.3297<!--n:cf.m.c.frame--> |
| klatka + pozycje | 0.3271 [0.3017; 0.3538]<!--n:cf.m.p.frame_ff--> | 0.789<!--n:cf.a.p.frame_ff--> | 0.3028<!--n:cf.m.c.frame_ff--> |
| klatka + dostawca | 0.3258 [0.3016; 0.3510]<!--n:cf.m.p.frame_vendor--> | 0.792<!--n:cf.a.p.frame_vendor--> | 0.3007<!--n:cf.m.c.frame_vendor--> |

Wiersze populacji łączonej spoza populacji pierwotnej mają klatkę w czasie StatsBomb plus offset połowy, nie w kotwicy SoccerNet (sekcja 12). Stała po rekalibracji ma AUC poniżej poziomu losowego, bo pięć foldów ma różne częstości goli. To artefakt foldów, nie informacja.

Kontrasty kluczowe, log-loss A − B:

| Kontrast | Pierwotna | Łączona |
|---|---|---|
| klatka − dostawca | +0.0283 [+0.0168; +0.0392]<!--n:cf.c.p.frame_vendor--> | +0.0282 [+0.0184; +0.0377]<!--n:cf.c.c.frame_vendor--> |
| klatka − pozycje | +0.0272 [+0.0162; +0.0380]<!--n:cf.c.p.frame_ff--> | +0.0264 [+0.0167; +0.0355]<!--n:cf.c.c.frame_ff--> |
| klatka − odległość i kąt | +0.0029 [−0.0083; +0.0137]<!--n:cf.c.p.frame_da--> | −0.0012 [−0.0105; +0.0075]<!--n:cf.c.c.frame_da--> |
| dostawca − pozycje | −0.0010 [−0.0076; +0.0057]<!--n:cf.c.p.vendor_ff--> | −0.0018 [−0.0073; +0.0037]<!--n:cf.c.c.vendor_ff--> |
| pozycje − zdarzenia | −0.0059 [−0.0109; −0.0009]<!--n:cf.c.p.ff_event--> | −0.0058 [−0.0101; −0.0013]<!--n:cf.c.c.ff_event--> |
| zdarzenia − odległość i kąt | −0.0184 [−0.0272; −0.0101]<!--n:cf.c.p.event_da--> | −0.0219 [−0.0300; −0.0146]<!--n:cf.c.c.event_da--> |
| odległość i kąt − stała | −0.0488 [−0.0608; −0.0376]<!--n:cf.c.p.da_const--> | −0.0413 [−0.0506; −0.0323]<!--n:cf.c.c.da_const--> |

Co z tego widać. Przedziały kolejnych kroków drabiny, od stałej do odległości i kąta, do zdarzeń i do pozycji, leżą w obu populacjach w całości poniżej zera. Największy jest krok do odległości i kąta, najmniejszy krok od zdarzeń do pozycji. Przedział różnicy dostawca − pozycje obejmuje zero. Przedziały różnic klatki wobec pozycji i wobec dostawcy leżą w całości powyżej zera, a przedział różnicy klatki wobec samej odległości i kąta obejmuje zero. W AUC klatka traci do dostawcy −0.056 [−0.080; −0.031]<!--n:cf.c.p.auc_frame_vendor-->.

Wariant surowy, bez rekalibracji, pokazuje przesunięcie populacji. Modele uczone na StatsBomb przewidują w populacji pierwotnej za mało goli: średnie p minus częstość goli wynosi −0.0328 [−0.0456; −0.0205]<!--n:cf.r.p.vendor--> dla dostawcy i −0.0274 [−0.0399; −0.0155]<!--n:cf.r.p.ff--> dla pozycji. Klatka, uczona na etykietach SoccerNet, ma −0.0041 [−0.0182; +0.0092]<!--n:cf.r.p.frame-->. SoccerNet pomija dużą część zablokowanych i niecelnych strzałów, więc częstość goli w jego strzałach jest wyższa niż w StatsBomb. Pełne tabele S2–S4 obu wariantów i obu populacji są w `results/xgv_confirmatory/shells.md`.

## 7. H2: czy klatka dodaje coś do pozycji (wyniki konfirmacyjne)

Wielkość to Δ = log-loss(klatka + pozycje) − log-loss(pozycje), oba modele rekalibrowane. Rozstrzyga TOST na populacji pierwotnej przy δ = 0.005<!--n:run.delta-->, na poziomie 5%<!--n:pg.alpha--> na stronę. Obok stoi jednostronny test przewagi. Populacja łączona to sprawdzenie odporności, a klatka + dostawca − dostawca jest opisowa.

| Kontrast | Populacja | Rola | Δ | Δ, 90%<!--n:pg.ci90--> | p TOST | p przewagi | Najmniejszy margines | Odczyt |
|---|---|---|---|---|---|---|---|---|
| klatka + pozycje − pozycje | pierwotna | konfirmacyjny | −0.0004 [−0.0026; +0.0018]<!--n:cf.h2.p.diff--> | [−0.0022; +0.0015]<!--n:cf.h2.p.ci90--> | 0.0006<!--n:cf.h2.p.tost--> | 0.5274<!--n:cf.h2.p.sup--> | 0.0022<!--n:cf.h2.p.margin--> | równoważne przy δ |
| klatka + pozycje − pozycje | łączona | odporność | −0.0005 [−0.0025; +0.0012]<!--n:cf.h2.c.diff--> | [−0.0021; +0.0010]<!--n:cf.h2.c.ci90--> | 0.0000<!--n:cf.h2.c.tost--> | 0.4030<!--n:cf.h2.c.sup--> | 0.0021<!--n:cf.h2.c.margin--> | równoważne przy δ |
| klatka + dostawca − dostawca | pierwotna | opisowy | −0.0007 [−0.0034; +0.0020]<!--n:cf.fv.p.diff--> | [−0.0028; +0.0016]<!--n:cf.fv.p.ci90--> | 0.0042<!--n:cf.fv.p.tost--> | 0.4384<!--n:cf.fv.p.sup--> | 0.0029<!--n:cf.fv.p.margin--> | równoważne przy δ |
| klatka + dostawca − dostawca | łączona | opisowy | −0.0008 [−0.0032; +0.0012]<!--n:cf.fv.c.diff--> | [−0.0027; +0.0010]<!--n:cf.fv.c.ci90--> | 0.0018<!--n:cf.fv.c.tost--> | 0.3158<!--n:cf.fv.c.sup--> | 0.0027<!--n:cf.fv.c.margin--> | równoważne przy δ |

Wiersze populacji łączonej spoza populacji pierwotnej mają klatkę w czasie StatsBomb plus offset połowy, nie w kotwicy SoccerNet (sekcja 12).

Na populacji pierwotnej równoważność zachodzi, przewagi nie ma, a dolna granica przedziału 95%<!--n:pg.ci95--> nie leży powyżej zera. Reguła z sekcji 9 pre-rejestracji daje więc odczyt "równoważne przy δ: klatka nie dodaje więcej niż δ". Pre-rejestracja napisała dla niego narrację z góry. Wstawiamy tylko liczby:

> Na 2 721<!--n:cf.s1.p.n--> strzałach sparowanych w 135<!--n:cf.s1.p.matches--> meczach dodanie jednej klatki transmisji do modelu pozycyjnego zmieniło log-loss o Δ = −0.0004<!--n:cf.h2.p.point--> (95%<!--n:pg.ci95--> [−0.0026; +0.0018]<!--n:cf.h2.p.ci95-->). Test równoważności wykluczył zysk i stratę większe niż δ = 0.005<!--n:run.delta--> (p TOST = 0.0006<!--n:cf.h2.p.tost-->). Klatka nie niesie informacji o szansie gola ponad to, co zawierają zdarzenia i freeze frame StatsBomb, w granicy δ. Przy tym klatka sama jest wyraźnie słabsza od pozycji (S4). Najprostsze wyjaśnienie: to, co klatka widzi (położenie strzelca, obrońców i bramkarza), dostawca zapisuje dokładniej, a czego dostawca nie zapisuje, jedna klatka nie pokazuje. Najmniejszy margines równoważności wynosi 0.0022<!--n:cf.h2.p.margin-->, więc twierdzenie o braku zysku ma tę precyzję i nie większą. Populacja łączona dała równoważne przy δ: klatka nie dodaje więcej niż δ<!--n:cf.h2.c.reading-->.

Odczyty obu populacji są takie same, więc nie ma rozbieżności do nazwania. Krzywe równoważności każdej pary szczebli są w `results/xgv_confirmatory/ladder/equivalence_curves.csv`.

Własności bootstrapu. Decyzja A wybrała bootstrap, który w każdym losowaniu dopasowuje od nowa rekalibrację i stosy. SD bootstrapu różnicy H2 wynosi tu 0.00111<!--n:cf.h2.p.sd-->. Przy prawdziwym efekcie bliskim zeru ten bootstrap jest za szeroki (liczby rozwojowe w sekcji 11). Twierdzenie o równoważności to tylko utrudnia, więc go nie osłabia. Test przewagi jest jednak konserwatywny i mały rzeczywisty zysk z klatki łatwiej przeoczyć.

## 8. H3: co klatka koduje (wyniki konfirmacyjne)

Sonda to ridge na embeddingu klatki z offsetu 0, w zagnieżdżonej walidacji krzyżowej grupowanej po meczach. Cele liczymy z freeze frame i lokalizacji strzału StatsBomb. Rozstrzyga model foldu 0, model foldu 1 to wrażliwość, a zamrożony DINOv2-L z tymi samymi wagami co szkielet G-combo to odniesienie. Konfirmacyjne są dwa cele z poprawką Bonferroniego: każdy rozstrzyga dolna granica przedziału 95%<!--n:pg.ci95-->, czyli test jednostronny na poziomie 2.5%<!--n:pg.bonf-->.

**Kontrola negatywna.** Sonda na odległości permutowanej po wierszach musi dać R² najwyżej 0.005<!--n:run.gate_thr-->. Dała −0.0006<!--n:cf.ctl.f0--> dla foldu 0, +0.0005<!--n:cf.ctl.f1--> dla foldu 1 i −0.0002<!--n:cf.ctl.dino--> dla DINOv2. Bramka przeszła dla wszystkich trzech enkoderów, więc wszystkie odczyty są ważne.

**Cele konfirmacyjne.** Sondy liczą 2 721<!--n:cf.h3.n--> strzałów pierwotnych bez karnych.

| Cel | Próg | G-combo fold 0 | Odczyt | G-combo fold 1 | Odczyt | DINOv2 |
|---|---|---|---|---|---|---|
| odległość, R² | 0.50<!--n:cf.h3.thr.dist--> | 0.711 [0.678; 0.739]<!--n:cf.h3.f0.dist--> | koduje<!--n:cf.h3.f0.dist.reading--> | 0.716 [0.685; 0.743]<!--n:cf.h3.f1.dist--> | koduje<!--n:cf.h3.f1.dist.reading--> | 0.684 [0.656; 0.711]<!--n:cf.h3.d.dist--> |
| obrońca w stożku, AUC | 0.70<!--n:cf.h3.thr.cone--> | 0.761 [0.743; 0.778]<!--n:cf.h3.f0.cone--> | koduje<!--n:cf.h3.f0.cone.reading--> | 0.758 [0.741; 0.775]<!--n:cf.h3.f1.cone--> | koduje<!--n:cf.h3.f1.cone.reading--> | 0.735 [0.716; 0.754]<!--n:cf.h3.d.cone--> |

Oba cele konfirmacyjne dostały odczyt "koduje", więc obowiązuje narracja "klatka koduje geometrię", napisana przed wynikiem:

> Z embeddingu jednego modelu foldu G-combo da się liniowo odczytać odległość strzału (R² = 0.711 [0.678; 0.739]<!--n:cf.h3.f0.dist-->, próg 0.50<!--n:cf.h3.thr.dist-->) i obecność obrońcy w stożku strzału (AUC = 0.761 [0.743; 0.778]<!--n:cf.h3.f0.cone-->, próg 0.70<!--n:cf.h3.thr.cone-->). Klatka zawiera więc te wielkości pozycyjne, z których korzysta dostawca. Z ośmiu celów opisowych model foldu 0 koduje kąt, liczbę obrońców w stożku, co najmniej dwóch obrońców w stożku, głębokość bramkarza i strzał głową, przy odległości bramkarza od bramki odczyt jest nierozstrzygnięty, a bocznego odsunięcia bramkarza i liczby rywali blisko strzelca nie koduje. Sonda liniowa daje tylko dolne ograniczenie tego, co embedding zawiera.

**G-combo wobec zamrożonego DINOv2.** To wynik drugorzędny, ustalony z góry, bez poprawki na wielokrotność. Na obu celach konfirmacyjnych i dla obu modeli foldu odczyt brzmi "G > DINOv2". Dla foldu 0 różnica wynosi +0.027 [+0.011; +0.042]<!--n:cf.d.f0.dist--> R² odległości i +0.025 [+0.013; +0.038]<!--n:cf.d.f0.cone--> AUC obrońcy w stożku, dla foldu 1 +0.032 [+0.018; +0.046]<!--n:cf.d.f1.dist--> i +0.023 [+0.010; +0.035]<!--n:cf.d.f1.cone-->. Obowiązuje więc narracja "G > DINOv2":

> Dostrajanie na etykiecie gola zwiększyło liniową czytelność odległości strzału i obecności obrońcy w stożku o +0.027 [+0.011; +0.042]<!--n:cf.d.f0.dist--> (R²) i +0.025 [+0.013; +0.038]<!--n:cf.d.f0.cone--> (AUC) względem zamrożonego szkieletu z tymi samymi wagami. Model uczył się więc reprezentować geometrię strzału, a nie tylko korzystać z cech, które szkielet już miał. Wcześniejsza obserwacja z prototypu, że dostrajanie nie pomaga, pochodziła z sondy na wymieszanych przestrzeniach embeddingów pięciu modeli i jej nie potwierdzamy.

**Cele opisowe.** Próg to R² 0.25<!--n:cf.h3.thr.angle--> albo AUC 0.70<!--n:cf.h3.thr.head-->, a odczyt idzie za dolną granicą jak wyżej. Znak † oznacza dolną granicę w odległości do 0.02<!--n:pg.near--> od progu; pre-rejestracja każe takie granice oznaczyć.

| Cel | G-combo fold 0 | Odczyt, fold 0 | Odczyt, fold 1 | G − DINOv2, fold 0 / fold 1 |
|---|---|---|---|---|
| kąt, R² | 0.450 [0.409; 0.489]<!--n:cf.h3.f0.angle--> | koduje | koduje | G > DINOv2 / G > DINOv2 |
| liczba obrońców w stożku, R² | 0.367 [0.319; 0.409]<!--n:cf.h3.f0.ff_def_cone--> | koduje | koduje † | G > DINOv2 / brak różnicy |
| co najmniej dwóch obrońców w stożku, AUC | 0.849 [0.826; 0.871]<!--n:cf.h3.f0.ge2--> | koduje | koduje | G > DINOv2 / brak różnicy |
| głębokość bramkarza, R² | 0.325 [0.269; 0.377]<!--n:cf.h3.f0.gk_depth--> | koduje † | nierozstrzygnięty † | G > DINOv2 / G > DINOv2 |
| boczne odsunięcie bramkarza, R² | 0.025 [0.004; 0.063]<!--n:cf.h3.f0.gk_off--> | nie koduje | nie koduje | G > DINOv2 / G > DINOv2 |
| odległość bramkarza od bramki, R² | 0.318 [0.244; 0.384]<!--n:cf.h3.f0.gk_dist--> | nierozstrzygnięty † | nierozstrzygnięty | G > DINOv2 / brak różnicy |
| rywale w promieniu 3 jednostek, R² | 0.185 [0.149; 0.218]<!--n:cf.h3.f0.opp--> | nie koduje | nie koduje | G > DINOv2 / G > DINOv2 |
| strzał głową, AUC | 0.906 [0.883; 0.927]<!--n:cf.h3.f0.head--> | koduje | koduje | brak różnicy / G > DINOv2 |

Modele foldu różnią się tylko na celach opisowych i w odczytach różnicy. Odczyty celów konfirmacyjnych są w obu foldach takie same. Progi opisowe ustaliliśmy, znając liczby rozwojowe prototypu (sekcja 12). Przedziały sond są warunkowe: losowania przeważają stałe predykcje out-of-fold, a sondy nie są dopasowywane od nowa. Pomijają więc zmienność wyboru α i współczynników i są węższe niż pełna niepewność. Pełne tabele S6 i S7 są w `results/xgv_confirmatory/shells.md`.

## 9. H4: diagnostyka przecieku (wyniki konfirmacyjne, opisowe)

H4 pyta, czy klatka czyta coś, co dzieje się po kontakcie. Populacja to 2 347<!--n:cf.h4.n--> strzałów pierwotnych bez gola i bez karnych, z czego 1 079<!--n:cf.h4.on--> celnych według SoccerNet. SoccerNet liczy strzały zablokowane jako celne (decyzja D4). Stałe kontrole to logit xG dostawcy, część ciała i typ strzału. H4 nie ma progów ani werdyktu.

| Część | Model | Współczynnik |
|---|---|---|
| celność | różnica logitów klatka − dostawca na celności, z kontrolami (wiersz główny) | +0.056 [+0.004; +0.108]<!--n:cf.h4.main--> |
| celność | to samo bez kontroli | −0.034 [−0.107; +0.036]<!--n:cf.h4.main_u--> |
| kontrast offsetów | średnia G(0) − G(−400 ms) | +0.057 [+0.041; +0.073]<!--n:cf.h4.off_mean--> |
| kontrast offsetów | kontrast na celności, z kontrolami | +0.025 [−0.009; +0.057]<!--n:cf.h4.off_on--> |
| kontrast offsetów | kontrast na celności, bez kontroli | +0.041 [+0.007; +0.073]<!--n:cf.h4.off_on_u--> |

Dodatni współczynnik celności zgadza się z odczytem kierunku piłki po kontakcie. Zgadza się też z informacją sprzed kontaktu, której dostawca nie ma, na przykład z równowagą strzelca. Dodatni kontrast offsetów na strzałach celnych ma te same dwa wyjaśnienia i jeszcze trzecie: przez ostatnie 400 ms przed kotwicą akcja się rozwija. Bez kontroli przedział współczynnika celności obejmuje zero.

Ramię reszty czasowej biegnie, bo werdykt źródła czasu brzmi "independent". Reszta to kotwica SoccerNet minus czas StatsBomb po offsecie połowy. Wszystkie wiersze i wrażliwość ograniczona do ±200 ms stoją obok siebie:

| Model, z kontrolami | Wszystkie wiersze | W ±200 ms |
|---|---|---|
| nachylenie różnicy logitów na reszcie [logit/s] | +0.545 [+0.259; +0.857]<!--n:cf.h4.slope--> | −0.212 [−0.822; +0.388]<!--n:cf.h4.slope_200--> |
| kotwica ponad 40 ms przed StatsBomb | −0.073 [−0.150; +0.007]<!--n:cf.h4.early--> | +0.005 [−0.072; +0.081]<!--n:cf.h4.early_200--> |
| kotwica ponad 40 ms po StatsBomb | −0.047 [−0.114; +0.020]<!--n:cf.h4.late--> | −0.058 [−0.128; +0.012]<!--n:cf.h4.late_200--> |
| strzały | 2 347<!--n:cf.h4.n--> | 2 284<!--n:cf.h4.n200--> |

Na wszystkich wierszach przedział nachylenia leży powyżej zera. W ograniczeniu do ±200 ms jest szeroki i obejmuje zero. Pełna tabela S8, także bez kontroli, jest w `results/xgv_confirmatory/shells.md`.

## 10. Rzuty karne (wyniki konfirmacyjne, opisowe)

Karne nie wchodzą do H1–H4. Siedem karnych populacji łączonej spoza pierwotnej to odzyskane gole z klatką w czasie StatsBomb plus offset połowy. Klatka nie ocenia karnego: dostaje stałą częstość karnych z kalibracji G-combo. Prawdopodobieństwa są surowe, bez przedziałów.

| Populacja | Karne | Gole (SoccerNet) | Śr. xG dostawcy | Log-loss dostawcy | Śr. p klatki | Log-loss klatki | Śr. p pozycji | Log-loss pozycji |
|---|---|---|---|---|---|---|---|---|
| pierwotna | 33<!--n:cf.pen.p.n--> | 22<!--n:cf.pen.p.goals--> | 0.783<!--n:cf.pen.p.vp--> | 0.6727<!--n:cf.pen.p.vll--> | 0.697<!--n:cf.pen.p.fp--> | 0.6387<!--n:cf.pen.p.fll--> | 0.758<!--n:cf.pen.p.pp--> | 0.6578<!--n:cf.pen.p.pll--> |
| łączona | 40<!--n:cf.pen.c.n--> | 29<!--n:cf.pen.c.goals--> | 0.783<!--n:cf.pen.c.vp--> | 0.5977<!--n:cf.pen.c.vll--> | 0.697<!--n:cf.pen.c.fp--> | 0.5900<!--n:cf.pen.c.fll--> | 0.758<!--n:cf.pen.c.pp--> | 0.5911<!--n:cf.pen.c.pll--> |

## 11. Kontekst z modeli, które widziały te mecze (wyniki rozwojowe)

Te liczby nie są wynikami badania. Liczyły je archiwalny G-combo albo prototyp, a archiwalny G-combo widział w treningu większość tych meczów. Podajemy je, bo wyjaśniają wybory pre-rejestracji.

**Wybór δ.** Reguła spisana przed symulacją mocy brała najmniejszy margines z mocą co najmniej 0.80<!--n:dv.target--> przy prawdzie rozwojowej. Moc TOST wyszła 0.284<!--n:dv.pow002-->, 0.509<!--n:dv.pow003--> i 0.849<!--n:dv.pow005--> przy marginesach 0.002, 0.003 i 0.005<!--n:dv.margins-->, więc reguła wybrała δ = 0.005<!--n:dv.chosen-->. Prawda rozwojowa zakładała zysk z klatki około dwa razy większy, niż pokazało H2 na tabeli deweloperskiej. Przy założeniu, że klatka nic nie dodaje, już margines 0.002<!--n:dv.m002--> miał moc 0.949<!--n:dv.pow_null-->. Właściciel zostawił regułę bez zmian. Stąd "równoważne przy δ" to słabe twierdzenie, a raport podaje obok najmniejszy margines równoważności.

**H2 na tabeli deweloperskiej.** Archiwalny G-combo dał −0.0007 [−0.0031; +0.0016]<!--n:dv.h2--> z najmniejszym marginesem 0.0028<!--n:dv.h2_margin-->, a klatka − dostawca wyszła +0.0274 [+0.0159; +0.0382]<!--n:dv.frame_vendor-->.

**Bootstrap z dopasowaniem w losowaniu.** Rozrzut próbkowania różnicy H2 w symulacji mocy (1 000<!--n:dv.n_sims--> badań na scenariusz) wynosił 0.00119<!--n:dv.sd_dev--> przy prawdzie rozwojowej i 0.00042<!--n:dv.sd_null--> przy prawdzie zerowej. Kontrola na 80<!--n:dv.rc.sims--> symulowanych badaniach dała średnie SD bootstrapu z dopasowaniem w losowaniu 0.00120<!--n:dv.rc.dev_refit--> przy prawdzie rozwojowej, czyli zgodne z rozrzutem, i 0.00072<!--n:dv.rc.null_refit--> przy prawdzie zerowej, czyli wyraźnie za szerokie. Losowania leżały tam średnio o +0.00025<!--n:dv.rc.null_shift--> nad estymatą. Bootstrap ze stałymi predykcjami dał 0.00103<!--n:dv.rc.dev_fixed--> i 0.00036<!--n:dv.rc.null_fixed-->, czyli za mało w obu scenariuszach. Dlatego test przewagi wybranego bootstrapu jest konserwatywny.

**Sondy na modelach foldu.** Prototyp mieszał w jednej sondzie embeddingi pięciu modeli foldu i nie widział przewagi dostrojonego G-combo nad DINOv2. Sonda na jednym modelu foldu 0, na 489<!--n:dv.probe_n--> strzałach, pokazała dla odległości różnicę +0.071 [+0.025; +0.122]<!--n:dv.probe_dist-->. Dlatego pre-rejestracja napisała obie narracje o dostrajaniu neutralnie.

## 12. Ryzyka i ograniczenia

- **Słaby test bramkowy.** Retrening sprawdziło 825<!--n:gc.shots--> strzałów z 37<!--n:gc.matches--> meczów. Przedział różnicy log-lossu obejmuje zarówno brak straty, jak i stratę ponad dwa razy większą od punktowej (sekcja 3.3). Bramka łapie tylko zepsuty retrening, nie małą stratę jakości. Wyniki H1–H4 dotyczą modelu wytrenowanego od nowa, który może być trochę słabszy od archiwalnego.
- **Konwencja kotwicy populacji wtórnej.** Klatki strzałów wtórnych wycięliśmy w czasie StatsBomb plus offset połowy, a nie w kotwicy SoccerNet. Werdykt "independent" znaczy, że czas StatsBomb to osobna adnotacja, więc te klatki mogą leżeć kilkaset milisekund od kontaktu. Margines rzutów rożnych w teście źródła czasu był przy tym cienki. Populacja łączona jest więc, jak ustaliła pre-rejestracja, tylko sprawdzeniem odporności; H2 rozstrzyga populacja pierwotna.
- **Kluby dzielone między sezonami.** Mecze badania leżą poza treningiem każdego modelu, ale ich kluby grają w meczach treningowych z innych sezonów, jak w archiwalnym podziale testowym. Model może znać stadiony, stroje i grafikę transmisji tych klubów. Tej zależności nie mierzyliśmy.
- **H2 dotyczy naszego komparatora.** Pozycje w H2 to szczebel freeze frame naszego modelu, który dorównuje dostawcy w K1. Model dostawcy nie ma wersji "z klatką" dopasowanej krzyżowo, więc klatka + dostawca − dostawca jest tylko opisowa.
- **Słabe twierdzenie H2.** Równoważność przy δ wyklucza tylko zysk większy niż δ, a δ wybrała reguła mocy liczonej przy zysku z klatki dwa razy większym, niż pokazały dane rozwojowe (sekcja 11). Precyzję twierdzenia wyznacza najmniejszy margines równoważności z sekcji 7.
- **Bootstrap.** Wybrany bootstrap odtwarzał w symulacji rozrzut próbkowania przy prawdzie rozwojowej, ale przy prawdzie zerowej był za szeroki i przesunięty w górę (liczby w sekcji 11). Test przewagi jest więc konserwatywny. Przedziały H3 pomijają zmienność dopasowania sondy.
- **Progi opisowe H3** ustaliliśmy, znając liczby rozwojowe prototypu i ticketu 09. Progi konfirmacyjne pochodzą z planu sprzed prototypu.
- **Kogo nie ma w populacji pierwotnej.** To strzały, które oznaczył SoccerNet, a SoccerNet pomija dużą część zablokowanych i niecelnych strzałów (sekcja 4). Populacja łączona częściowo to naprawia, ale traci strzały z doliczonego czasu, których nagrania SoccerNet nie obejmują.
- **Wysokość piłki.** Klatka nie widzi wysokości piłki przy uderzeniu, a model dostawcy ją zna (sekcja 5).
- **Wartość K1** jest trochę za optymistyczna, bo to wynik najlepszego ustawienia z siatki.

## 13. Udostępnienie danych

Specyfikacja zakładała udostępnienie mapy meczów, tabeli par i cech pochodnych (decyzja D1). Wieczorem 2026-10-11 właściciel to zawęził: pakiet nie zawiera żadnych danych StatsBomb. Zawiera nasz kod, nasze predykcje i identyfikatory meczów i zdarzeń StatsBomb jako klucze złączenia. Umowa StatsBomb (punkty `1.2.1` i `7`) zabrania przekazywania danych osobom trzecim bez pisemnej zgody, a cechy z freeze frame i xG dostawcy to dane StatsBomb albo wartości z nich liczone. Z wideo SoccerNet pakiet ma tylko nasze predykcje, bez klatek, klipów, embeddingów i wag.

Pakiet buduje `scripts/vendor_release.py` do `results/xgv_release_v1/`, poza gitem. Zawiera mapę meczów, tabelę par z identyfikatorami i czasami SoccerNet, odzyskane gole, predykcje G-combo przy obu offsetach dla każdego wiersza ewaluacji, plan podziału retreningu i kod. Predykcje klatki są bez routingu karnych, bo flaga karnego to adnotacja SoccerNet albo wartość StatsBomb. Nie zawiera xG dostawcy, wyników strzałów, lokalizacji, cech freeze frame, celów sond, predykcji komparatora, offsetów połówek ani reszt czasowych. Offset i reszta razem z czasem SoccerNet oddają czas StatsBomb. Nie ma w nim też treści adnotacji SoccerNet. To, w której tabeli stoi czas, mówi tylko, czy był to strzał z indeksu, czy gol bez adnotacji strzału. Status klatki ma tylko dwie wartości, ok albo brak klatki, bo powód braku wynika z czasu StatsBomb. `MANIFEST.md` pakietu wymienia każdą kolumnę z jej pochodzeniem i każdą pominiętą kolumnę źródłową z powodem. README opisuje, jak odtworzyć resztę z otwartych danych StatsBomb naszym kodem.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Co z niego pochodzi |
|---|---|---|
| `research/xg_vendor_preregistration.md` | `312e4e4a3282be7e8c2d884a73ee06a7a827311bf52805331ca06b56f34ac76b` | stałe pre-rejestracji: poziomy testów, liczba plików kodu, progi, liczności z decyzji C |
| `results/xgv_confirmatory/complete.json` | `2188f91314ae935890ba8091677e1192711bb948db1e81a2f983622da56739da` | parametry przebiegu konfirmacyjnego: losowania, δ, próg kontroli sond |
| `results/xgv_confirmatory/label_join_log.jsonl` | `6b41632ac1329b226b8b7e2ec92570cdbe9c0be9030e15991eb2be2e149cc152` | zapis łączenia etykiet, hash pre-rejestracji |
| `results/xgv_confirmatory/shells.md` | `5afcc4730ca8c2b9d3fd5ad0908462a97522be08550c3077f07600a2c8e519de` | odczyty H2 i H3 wpisane przez kod pre-rejestracji |
| `results/xgv_confirmatory/ladder/h2.json` | `9d5b459b3e307e8e1bbaeec2ac7e54b1941649844d183b3771b9ceecf889e217` | H2, klatka + dostawca, S1, karne |
| `results/xgv_confirmatory/ladder/metrics.csv` | `17e95f8b11483b64ce3b9a4b3cdf6406ac943758af19de88ccb96037b0af8bb4` | H1: metryki szczebli |
| `results/xgv_confirmatory/ladder/contrasts.csv` | `6dbfcf1d51086c2f115af4c750bef4c79b302eb47624b64a4fd5c82d48e41e07` | H1: kontrasty szczebli |
| `results/xgv_confirmatory/probes/probes.csv` | `266e190098d9363f3b2e0d4d282f628abf9686961de1a90a7652ed0f6f70ec98` | H3: sondy i kontrola negatywna |
| `results/xgv_confirmatory/probes/probe_differences.csv` | `2334de9e7aa98ab4631601d4adda30b72f196108fff242a0917f078ce55930d6` | H3: G-combo − DINOv2 |
| `results/xgv_confirmatory/leakage/leakage.csv` | `11e15eb84cb1bc258959e3f6d5f2ff1776d42a84ccc28947b520086929c569f0` | H4 |
| `results/xgv_pairing/pairing_report.json` | `6ed6b7615082fe15ac1e317f4d9955c5308c543de675ad3dbd0c8357a9aa0a9e` | parowanie, K0, populacje |
| `results/xgv_redownload.run.log` | `50ccc6eead00aac696d47394a1844dcb47836dcad60afbcb7d4c749e9849b6d3` | ponowne pobranie połówek SoccerNet |
| `results/xgv_comparator/report.json` | `17d474c404ceeb1b4e82738cd42a2b04f96e90e6962f59dd18e9bbc82d60cf7b` | komparator pozycyjny, K1 |
| `results/xgv_split_plan/plan.json` | `80162d10c59908893d86e0bc905be1623cae9d20ab17dd4f5607df42f8f4f3ce` | plan podziału retreningu |
| `results/xgv_g_combo/gate.json` | `af1fdc72af6e232a328be3efd7f974a932d512a6ac71bf0081b96344ec003fe9` | test bramkowy retreningu |
| `results/xgv_timing_source/timing_verdict.json` | `3903e8f7c7f35f2d9059850ec62fa7c69bed74ba597e6f28e2adb83440a66a59` | werdykt źródła czasu |
| `results/xgv_timing_source/unpaired_reasons.csv` | `bb393b4aa89c8ee6856e1cff6c0e8ba3376f1496b3973c335da0a911a9f6311b` | przyczyny dla niesparowanych strzałów SoccerNet |
| `results/xgv_power/decision.json` | `af4080beb5100a06fa28accab6663e96200cb9d81a7df458c7b5d979ff1afedf` | wybór δ (rozwojowe) |
| `results/xgv_power/power.csv` | `bb59d755cf9f0f8062e119825f92fd6414f12901645fe277b22b319cb749d9db` | tabela mocy (rozwojowe) |
| `results/xgv_ladder_dev/h2.json` | `482b3b184d777a9b5cc7e72e726eab0cb30947663a7f2d240ab2e4608e9e768c` | H2 na tabeli deweloperskiej (rozwojowe) |
| `results/xgv_ladder_dev/contrasts.csv` | `bc3d22cfc4beafcc850f8a98fd62186f45427c5e4a9052fe1f9d012720d9e52e` | kontrasty na tabeli deweloperskiej (rozwojowe) |
| `results/xgv_refit_check/refit_check.json` | `1766a63191882710dba3cabdf534f41320b2eeac38581a378bd1be918ad009b4` | kontrola bootstrapu (rozwojowe) |
| `results/xgv_probes_dev/probe_differences.csv` | `965ea40a164b705e9ed7fd50dd8b03b870b7c231c8f4b0927d0458afa3f2e512` | sondy na tabeli deweloperskiej (rozwojowe) |
