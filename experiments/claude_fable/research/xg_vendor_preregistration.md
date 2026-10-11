# Pre-rejestracja: xG z jednej klatki transmisji wobec xG dostawcy i xG pozycyjnego (xg-vendor, ticket 14)

Status: **SZKIC, runda robocza z 2026-10-11.** To jeszcze nie jest pre-rejestracja. Brakuje tabeli łączonej bez etykiet (ticket 13), decyzji właściciela o δ (sekcja 10), końcowych hashy, commita i zatwierdzenia przez właściciela. Każde miejsce, które czeka na te kroki, ma znacznik „DO UZUPEŁNIENIA”. Skrypt `scripts/vendor_prereg_hashes.py` liczy te znaczniki i nie wystawi rekordu pre-rejestracji, dopóki choć jeden zostaje.

Do chwili zatwierdzenia żadna etykieta nie zostaje połączona z nowymi predykcjami. Jedynym przebiegiem analizy na tych predykcjach jest dry run na etykietach i celach sond permutowanych w obrębie meczów (sekcja 15).

Wszystkie liczby oznaczone „deweloperskie” pochodzą z prototypu, z tabeli deweloperskiej (ticket 07) albo z symulacji mocy (ticket 08). Modele, które je dały, widziały w treningu większość tych meczów. To nie są wyniki.

## 0. Co zostaje do finalizacji

1. Ticket 13: tabela łączona bez etykiet. Jej sha256 trafia do pinu `combined_label_free_table` w bloku parametrów (sekcja 16). Wiersze pierwotne mają być bajt w bajt takie jak w tabeli z ticketu 11. DO UZUPEŁNIENIA
2. Ticket 13: reguła etykiet populacji wtórnej wdrożona w kroku łączenia etykiet (sekcja 4) i wymieniona w kodzie analizy. DO UZUPEŁNIENIA
3. Dry run na tabeli łączonej: `scripts/vendor_dryrun.py --pin combined_label_free_table --table-dir <katalog tabeli łączonej> --out results/xgv_dryrun_combined`. DO UZUPEŁNIENIA
4. Decyzja właściciela o δ (sekcja 10), wpisana jako `margin` w bloku parametrów. DO UZUPEŁNIENIA
5. Końcowe hashe: `./run.sh python scripts/vendor_prereg_hashes.py`. Polecenie przelicza listę hashy kodu, pinów i dowodów, a potem odtwarza blok hashy w sekcji 17.
6. Commit dokumentu, listy hashy i kodu. Potem `./run.sh python scripts/vendor_prereg_hashes.py --record results/xgv_prereg/preregistration_record.json`, które wystawia rekord pre-rejestracji z czasem tego commita.
7. Zatwierdzenie przez właściciela. Dopiero potem ticket 15 łączy etykiety.

## 1. Pytanie i hipotezy

Pytanie: jak xG z jednej klatki transmisji (G-combo) wypada wobec xG sprzedawanego przez dostawcę danych (StatsBomb) i wobec xG liczonego z danych pozycyjnych (zdarzenia i freeze frame StatsBomb), na tych samych strzałach? I dlaczego tak wypada?

| Hipoteza | Treść | Status |
|---|---|---|
| H1 | Drabina informacji: log-loss, Brier, AUC i kalibracja każdego szczebla od stałej do stosów, na tych samych strzałach | opisowa |
| H2 | Czy klatka dodaje coś do pozycji: klatka + pozycje wobec samych pozycji, oba modele rekalibrowane | konfirmacyjna (populacja pierwotna) |
| H3 | Jakie wielkości pozycyjne klatka koduje: sondy liniowe z embeddingu klatki do wielkości z freeze frame | konfirmacyjna (dwa cele, sekcja 11) |
| H4 | Diagnostyka przecieku: czy klatka czyta kierunek piłki po kontakcie | opisowa |

Kontrola błędu:

- H2 to jeden test TOST na poziomie 5% na stronę.
- W H3 konfirmacyjne są dwa cele. Każdy rozstrzyga jednostronny test na poziomie 2,5% (dolna granica przedziału 95%). Poprawka Bonferroniego daje więc błąd rodzinny w H3 co najwyżej 5%.
- H2 i H3 to osobne pytania, bez poprawki między nimi.
- Reszta jest opisowa. Raportujemy przedziały 95%. Żadne twierdzenie nie opiera się na wartości p z części opisowej.

## 2. Pojęcia i źródła

Słownictwo idzie za `GLOSSARY.md`, sekcja xG. Używamy pojęć: kotwica strzału, offset, model foldu, G-combo, xG dostawcy, komparator pozycyjny, drabina informacji, strzał sparowany, offset połowy, reszta czasowa i test bramkowy (gate test).

| Źródło | Co daje | Ticket |
|---|---|---|
| Parowanie SoccerNet–StatsBomb, zamrożone | 135 meczów i 2 754 strzały sparowane. Do tego 50 odzyskanych goli i 797 strzałów tylko ze StatsBomb | 02, 03 |
| Wersja danych StatsBomb `statsbomb_open_v1` (commit open-data 4b73468f) | xG dostawcy, zdarzenia, freeze frame | 01 |
| Komparator pozycyjny, zamrożony | szczeble: stała, odległość i kąt, zdarzenia, freeze frame | 06 |
| G-combo wytrenowany ponownie bez 135 meczów nakładki | logity klatki przy offsetach 0 i −400 ms, embeddingi | 10, 11 |
| Test źródła czasu | werdykt „independent” | 04 |
| Reguła δ i symulacja mocy | tabela mocy; reguła wybiera δ = 0,005 | 08 |

Atrybucja: każda tabela i każdy rysunek z danych StatsBomb dostaje podpis „Data: StatsBomb open data” (decyzja D1, `research/xg_vendor_statsbomb_terms.md`).

## 3. Populacje

**Pierwotna.** Strzały sparowane: 2 754 strzały z indeksu v1.2 w 135 meczach (train 1 478, valid 639, test 637). Kotwicą jest kotwica SoccerNet (`anchor_convention = soccernet_anchor`). Wszystkie 2 754 mają klatkę przy offsetach 0 i −400 ms. Rzutów karnych jest 33. Flaga karnego z indeksu zgadza się na wszystkich 2 754 strzałach z flagą StatsBomb i z typem strzału „Penalty”.

**Wtórna** (ticket 13). Należą do niej:

- 797 strzałów tylko ze StatsBomb: każdy strzał StatsBomb ze sparowanego meczu, który nie jest karnym, nie został sparowany ani odzyskany i nie leży w ciągu 10 s od niesparowanego strzału SoccerNet tej samej połowy;
- 50 odzyskanych goli.

Nie należą:

- 26 możliwych błędnych par (strzały w ciągu 10 s od niesparowanego strzału SoccerNet);
- 2 niesparowane karne;
- gole samobójcze, które w StatsBomb nie są zdarzeniami Shot.

Klatki populacji wtórnej wycina się w czasie StatsBomb plus offset połowy (`anchor_convention = statsbomb_time`). Werdykt „independent” znaczy, że te klatki mogą leżeć kilkaset ms od kontaktu. Raport musi to napisać. Strzał wtórny bez klatki zostaje w tabeli ze statusem klatki i wypada z analiz; S1 go liczy.

**Łączona** = pierwotna ∪ wtórna. Liczności po tickecie 13: DO UZUPEŁNIENIA

**Wykluczenia z analiz:**

| Analiza | Populacja | Wykluczone |
|---|---|---|
| H1, H2 | pierwotna (H2: konfirmacyjnie) i łączona (odporność) | karne (`is_penalty`), wiersze bez ocenionej klatki przy offsecie 0 |
| H3 | pierwotna | karne (typ strzału StatsBomb „Penalty”, zmienna bez etykiety; na pierwotnej ten sam zbiór co `is_penalty`), wiersze bez embeddingu; dla każdego celu wiersze bez jego wartości |
| H4 | pierwotna, tylko strzały bez gola SoccerNet | karne, wiersze bez ocenionej klatki; w kontraście offsetów także wiersze bez logitu przy −400 ms |

Pominiętej klatki ani brakującego celu niczym nie zastępujemy.

**Mecze i losowania.** Strzały wtórne leżą w tych samych 135 meczach. Losowania bootstrapu zależą tylko od zbioru meczów tabeli (sekcja 7), więc tabela pierwotna i łączona mają te same losowania. Przebieg konfirmacyjny liczy wszystko na tabeli łączonej. Jeśli zbiór meczów tabeli łączonej różni się od 135 meczów tabeli pierwotnej, odstępstwo trafia do dziennika (sekcja 19), a obowiązują losowania tabeli łączonej.

## 4. Etykiety (decyzja D4)

Wiersze pierwotne biorą etykiety z zamrożonej tabeli par (`results/xgv_pairing/pairs.parquet`), w kodzie `xgpoc.vendor.labels.primary_labels`:

- `goal_sn`: gol z indeksu v1.2, czyli etykieta, na której uczył się G;
- `on_target_sn`: etykieta SoccerNet „Shots on target”, która liczy strzały zablokowane jako celne;
- `sb_outcome`: wynik strzału StatsBomb;
- `is_penalty`: flaga karnego z indeksu v1.2.

Trzynaście par, w których gol SoccerNet i gol StatsBomb się nie zgadzają (6 goli tylko w SoccerNet, 7 tylko w StatsBomb), zachowuje etykietę SoccerNet.

Wiersze wtórne dostają etykiety tak (wdrożenie: ticket 13):

- odzyskany gol: `goal_sn` = 1;
- strzał tylko ze StatsBomb: `goal_sn` = 0, chyba że regułą odzyskiwania przypisano mu gol SoccerNet;
- gole StatsBomb bez żadnego gola SoccerNet (12 wierszy wtórnych) trafiają na listę z powodem i wypadają z analiz;
- `on_target_sn` jest pusty, bo definicja SoccerNet istnieje tylko dla strzałów, które SoccerNet oznaczył;
- `is_penalty` = 0, bo populacja wtórna nie ma karnych.

Mechanizm wykluczenia 12 goli w kodzie analizy: DO UZUPEŁNIENIA

Etykiety łączy osobny, logowany krok `xgpoc.vendor.evaltable.join_labels`:

- zapisuje sha256 tabeli bez etykiet, sha256 etykiet, czas i hash pre-rejestracji;
- odmawia łączenia etykiet z tabelą konfirmacyjną bez rekordu pre-rejestracji, z rekordem roboczym albo przed czasem commita;
- przy każdym otwarciu tabeli walidator sprawdza ten sam warunek.

## 5. Predykcje i bramki wejścia

Predykcje tabeli:

- **Klatka.** Dziesięć modeli foldu ponownie wytrenowanego G-combo. Ich logity po skalowaniu temperaturą są uśredniane, a potem przechodzą jedną kalibrację Platta z puli treningowej, z korektą częstości i routingiem karnych (ticket 10). `frame_logit_0` i `frame_logit_m400` to logity tej skalibrowanej predykcji przy offsetach 0 i −400 ms.
- **xG dostawcy.** xG StatsBomb, nieobcięte w tabeli. Analiza obcina je do [10⁻⁶, 1 − 10⁻⁶].
- **Szczeble komparatora.** Stała, odległość i kąt, zdarzenia, freeze frame (= „pozycje” w H2). Predykcje z zamrożonego modelu, który nie widział żadnego meczu nakładki.

Bramki przed analizą. Wszystkie przeszły; wartości są zamrożone:

| Bramka | Reguła | Wartość | Wynik |
|---|---|---|---|
| K0, parowanie (ticket 02) | ≥ 85% strzałów indeksu sparowanych i ≥ 99% zgodności goli | 2 754 / 3 039 = 90,62%; 2 741 / 2 754 = 99,53% | przechodzi |
| K1, komparator (ticket 06) | log-loss CV puli szczebla freeze frame najwyżej 0,01 gorszy niż dostawcy na tej samej puli | 0,24607 wobec 0,24732, różnica −0,0012 (36 626 strzałów, 1 491 meczów) | przechodzi |
| Test bramkowy retreningu (ticket 05/10) | na 37 meczach bramkowych log-loss najwyżej +0,010 i AUC najwyżej −0,020 względem archiwalnego G-combo (punktowo) | log-loss 0,32655 wobec 0,32380 (+0,00275, 95% [−0,00096; +0,00650]); AUC 0,79240 wobec 0,79884 (−0,00643, 95% [−0,01721; +0,00465]) | przechodzi |
| Test źródła czasu (ticket 04) | reguła `research/xg_vendor_timing_rule.md`, sha256 42dcf710… | zmiany: mediana \|r\| 1 788 ms, 13,3% w ±200 ms; strzały (LOO) 96,6%, rogi 82,1% w ±200 ms | „independent” |

Wartość K1 to minimum po siatce, więc jest optymistyczna przez selekcję, o rząd 10⁻⁴. Margines rogów nad progiem 0,80 w teście źródła czasu jest cienki (0,021); raport to podaje.

## 6. Rekalibracja i stosy (decyzja D6)

- Każdy szczebel pojedynczy dostaje rekalibrację Platta: regresję logistyczną etykiety na logicie szczebla, bez regularyzacji (C = 10⁶). Dopasowanie jest krzyżowe (cross-fit) na pięciu foldach grupowanych po meczach.
- Foldy: posortowane unikalne mecze populacji, permutacja z `default_rng(0)`, fold = pozycja mod 5 (`ladder.match_folds`).
  - Foldy zależą tylko od meczów populacji, więc dzielą je wszystkie szczeble i oba stosy.
  - Populacja pierwotna dostaje te same foldy w tabeli pierwotnej i łączonej. Populacja łączona dostaje własne foldy, według tej samej reguły.
  - Żaden strzał nie jest oceniany modelem dopasowanym na jego meczu.
- Stos to regresja logistyczna etykiety na surowych logitach dwóch szczebli (klatka + pozycje, klatka + dostawca), dopasowana krzyżowo na tych samych foldach. Liczy się jako wariant rekalibrowany.
- Logit klatki jest obcinany jak każde prawdopodobieństwo (`ladder.frame_feature`).
- Wariant surowy (bez rekalibracji) dla sześciu szczebli pojedynczych jest drugorzędny. Pokazuje niedoszacowanie modeli uczonych na StatsBomb w tej populacji.

## 7. Wnioskowanie

- **Bootstrap meczowy.**
  - Losowania liczy `xgpoc.vendor.draws.table_draws`, czyli `match_draw_counts`: klucze `np.unique` meczów całej tabeli, `default_rng(boot_seed = 0)`, każde losowanie to `rng.choice(M, M)`. Liczba losowań `n_boot` = 5 000.
  - Mecz wylosowany k razy wchodzi z wagą k.
  - Te same losowania obsługują każdą populację, każdy szczebel, każdy kontrast i każdą hipotezę (H1–H4). Podzbiór czyta kolumny swoich meczów.
  - Predykcje z dopasowania krzyżowego są w losowaniach stałe. Każdą metrykę liczy się od nowa z wagami meczów.
  - Przedziały są percentylowe.
- **TOST** przy marginesie d, na poziomie 5% na stronę:
  - równoważność, gdy co najwyżej 5% losowań różnicy leży na lub poniżej −d i co najwyżej 5% na lub powyżej d, czyli gdy przedział 90% mieści się w (−d, d);
  - p TOST = max(odsetek losowań ≤ −d, odsetek ≥ d).
- **Test przewagi**, jednostronny, dla log-lossu (A − B): p = odsetek losowań ≥ 0, przewaga przy p ≤ 0,05. Dla AUC: odsetek ≤ 0.
- **Krzywe równoważności**: p TOST na siatce marginesów od 0 do 0,01 co 0,0001, dla każdej pary szczebli. Najmniejszy margines to pierwszy punkt siatki z równoważnością.
- **Znana granica.** W tickecie 08 bootstrap ze stałymi predykcjami był o około 8% węższy od rzeczywistego rozrzutu próbkowania (12% przy prawdzie zerowej). TOST jest więc lekko liberalny. Raport to podaje i nie poprawia.
- **Zatrzymanie.** Jeśli któreś losowanie nie ma gola albo ma same gole, kod przerywa analizę. Raport opisuje wtedy przerwę i nie podstawia innej metody.

## 8. H1: drabina informacji (opisowa)

- Szczeble w kolejności: stała, odległość i kąt, zdarzenia, freeze frame, xG dostawcy, klatka (sigmoid `frame_logit_0`), klatka + pozycje, klatka + dostawca.
- Metryki: log-loss, Brier, AUC (remisy liczone jako połowa), nachylenie i wyraz wolny kalibracji (ważona regresja logistyczna etykiety na logicie predykcji), średnie p minus częstość goli. Każda z przedziałem 95%.
- Warianty rekalibrowany i surowy, w populacji pierwotnej i łączonej. Tabele S1–S3.
- Kontrasty: każda para szczebli jednego wariantu × (log-loss, Brier, AUC), z przedziałami 95% i 90%, testem przewagi i najmniejszym marginesem równoważności (plik `contrasts.csv`). Krzywe równoważności są w `equivalence_curves.csv`. Osiem kontrastów kluczowych pokazuje S4.
- H1 jest opisowa. Opisujemy przedziały. Nie twierdzimy, że szczebel A „wygrywa”, na podstawie wartości p.

## 9. H2: czy klatka dodaje coś do pozycji (konfirmacyjna)

**Wielkość.** Δ = log-loss(klatka + pozycje) − log-loss(pozycje). Pozycje to szczebel freeze frame, oba modele są rekalibrowane (sekcja 6). Ujemne Δ znaczy, że klatka pomaga.

**Test konfirmacyjny.** Na populacji pierwotnej TOST przy marginesie δ (sekcja 10) na poziomie 5% na stronę. Obok raportujemy jednostronny test przewagi (5%). Rozstrzyga wyłącznie populacja pierwotna.

**Odczyt** (stosowany w kolumnie „Odczyt” S5, kod `shells.h2_reading`):

| Równoważność przy δ (p TOST ≤ 0,05) | Przewaga (p przewagi ≤ 0,05) | Dolna granica 95% > 0 | Odczyt |
|---|---|---|---|
| tak | tak | – | poprawa istotna, mniejsza niż δ |
| nie | tak | – | klatka dodaje; równoważności przy δ brak |
| tak | nie | tak | równoważne przy δ; stos istotnie gorszy |
| tak | nie | nie | równoważne przy δ: klatka nie dodaje więcej niż δ |
| nie | nie | tak | stos istotnie gorszy; równoważności przy δ brak |
| nie | nie | nie | nierozstrzygnięty |

„Stos istotnie gorszy” znaczy, że dodanie klatki pogarsza log-loss (koszt dopasowania drugiego współczynnika). To nie dowodzi, że klatka niesie informację.

**Odporność.** Ta sama analiza na populacji łączonej, z tym samym δ i tym samym odczytem. Nie zmienia werdyktu z populacji pierwotnej. Gdy odczyty się różnią, raport podaje oba i nazywa różnicę.

**Klatka + dostawca − dostawca.** Ta sama wielkość, opisowo, na obu populacjach. Odpowiada na pytanie, czy klatka dodaje coś do kupionych danych dostawcy.

## 10. Decyzja właściciela: δ

Reguła z ticketu 08 (`research/xg_vendor_delta_rule.md`, sha256 766ee11b…, ustalona przed symulacją) brzmi: δ to najmniejszy margines z {0,002; 0,003; 0,005} z mocą decyzyjną co najmniej 0,80. Scenariusz decyzyjny to populacja pierwotna i prawda deweloperska, a moc decyzyjna to minimum mocy dla klatki archiwalnej i dla klatki słabszej o 0,002. Reguła wybrała **δ = 0,005**. Jej krok 6 wymaga, żeby właściciel zatwierdził δ, zanim δ trafi do tego dokumentu.

Moc TOST z symulacji (1 000 badań na scenariusz, błąd Monte Carlo ≤ 0,016; liczby deweloperskie):

| Scenariusz | δ = 0,002 | δ = 0,003 | δ = 0,005 | Średnie symulowane H2 |
|---|---|---|---|---|
| pierwotna, prawda deweloperska, klatka archiwalna | 0,284 | 0,509 | 0,849 | −0,0013 |
| pierwotna, prawda deweloperska, klatka słabsza | 0,308 | 0,556 | 0,883 | −0,0012 |
| pierwotna, prawda zerowa, klatka archiwalna | 0,949 | 0,993 | 1,000 | +0,0002 |
| pierwotna, prawda zerowa, klatka słabsza | 0,965 | 0,996 | 1,000 | +0,0002 |
| łączona, prawda deweloperska, archiwalna / słabsza | 0,263 / 0,304 | 0,535 / 0,569 | 0,883 / 0,905 | −0,0014 / −0,0013 |
| łączona, prawda zerowa, archiwalna / słabsza | 0,989 / 0,989 | 0,999 / 0,999 | 1,000 / 1,000 | +0,0002 |

Problem, który znalazł ticket 08:

- Prawda deweloperska to regresja dopasowana w próbie na tych samych 2 708 strzałach. Zakłada zysk z klatki około −0,0013, czyli mniej więcej dwa razy więcej, niż pokazało H2 na danych deweloperskich: −0,0007 [−0,0031; +0,0016], najmniejszy margines 0,0028.
- Sam ten wybór scenariusza rozstrzyga o δ = 0,005.
- Przy założeniu „klatka nic nie dodaje” już δ = 0,002 ma moc 0,95.

Opcje (dokument nie wybiera):

- **A. Reguła bez zmian: δ = 0,005.**
  - Plusy: reguła została ustalona przed symulacją i nie zmienia się po obejrzeniu tabeli mocy. Moc 0,85 przy zysku dwa razy większym od deweloperskiego.
  - Minusy: „równoważne przy 0,005” to słabe twierdzenie. 0,005 log-lossu to około jedna czwarta różnicy klatka − dostawca z prototypu (+0,0235 na teście). Równoważność przy 0,005 nie wyklucza zysku rzędu dobrych cech pozycyjnych (freeze frame − zdarzenia w prototypie: od −0,007 do +0,006).
- **B. Poprawiony scenariusz decyzyjny.** Nowa notatka reguły (z nowym hashem; obie wersje idą do raportu) zastępuje prawdę deweloperską prawdą ściągniętą do zysku obserwowanego (−0,0007) albo prawdą zerową.
  - Przy prawdzie zerowej reguła daje δ = 0,002 (moc 0,949 i 0,965).
  - Dla prawdy ściągniętej do −0,0007 mocy nie policzono. Leży między wierszami deweloperskimi i zerowymi (przy 0,002: między 0,28 a 0,95; przy 0,003: między 0,51 a 0,99) i wymaga jednego przebiegu symulacji (około 14 min CPU) przed wyborem.
  - Plusy: mocniejsze twierdzenie przy prawdopodobnym zysku bliskim zera.
  - Minusy: zmiana reguły po obejrzeniu tabeli mocy. To jawne odstępstwo od D5, a raport musi je opisać. Gdy prawdziwy zysk wynosi około −0,001, δ = 0,002 łatwo daje wynik nierozstrzygnięty.

Niezależnie od opcji raport podaje najmniejszy margines równoważności i całą krzywą równoważności H2, więc czytelnik może przyłożyć własny próg.

**Decyzja właściciela:** δ = DO UZUPEŁNIENIA (opcja A albo B, z datą). Wybrana wartość trafia do pola `margin` bloku parametrów (sekcja 16). Do tego czasu dry run używa `dry_run_margin` = 0,005.

## 11. H3: co klatka koduje (konfirmacyjna)

**Cele.** Wielkości policzone z freeze frame i lokalizacji strzału StatsBomb (`xgpoc.vendor.probe_targets`), w jednostkach boiska StatsBomb (jardy):

- `dist`: odległość do środka bramki;
- `angle`: kąt widzenia słupków;
- `ff_def_cone`: liczba obrońców z pola w trójkącie strzelec–słupki, z progami `def_cone_ge1` (≥ 1) i `def_cone_ge2` (≥ 2);
- `gk_depth`: głębokość bramkarza;
- `gk_off_line_abs`: boczne odsunięcie bramkarza od linii strzelec–środek bramki;
- `gk_dist_goal`: odległość bramkarza od środka bramki;
- `ff_opp_3m`: liczba rywali w promieniu 3 jednostek od strzelca;
- `head`: strzał głową.

Cele binarne mierzy AUC wyniku sondy, ciągłe R² predykcji out-of-fold. Obok R² raportujemy ρ Spearmana.

**Sonda** (`xgpoc.vendor.probes`):

- ridge w postaci dualnej na standaryzowanym embeddingu z offsetu 0;
- zagnieżdżona walidacja krzyżowa grupowana po meczach: zewnętrznie `GroupKFold(5)`, wewnętrznie `GroupKFold(3)` do wyboru α z siatki 1…10⁶ (13 wartości), przy remisie mniejsze α;
- standaryzacja dopasowana tylko na wierszach, na których model się uczy.

Kontrola negatywna `dist_shuffled`: odległość permutowana po wierszach sondy (`default_rng(0)`). Musi dać R² ≤ 0 (punktowo) dla każdego enkodera. Jeśli dla któregoś enkodera nie da, H3 dla tego enkodera jest nieważna i raport to podaje.

**Enkodery** (embeddingi wszystkich 2 754 strzałów pierwotnych, ticket 11):

| Enkoder | Rola |
|---|---|
| G-combo, model foldu 0, seed 0 (`emb_g_fold0`, 5 120 wymiarów: token CLS i wyjścia zapytań attention pooling) | rozstrzyga |
| G-combo, model foldu 1, seed 0 (`emb_g_fold1`) | wrażliwość; nie zmienia werdyktu; rozbieżności są raportowane |
| zamrożony DINOv2-L (`emb_dinov2`, 2 048 wymiarów: CLS i średnia tokenów łatek), te same wagi co szkielet G | odniesienie |

Mecze ewaluacyjne leżą poza treningiem każdego modelu foldu. Embedding pochodzi więc z jednego modelu dla wszystkich strzałów. Prototyp mieszał modele foldów, czego specyfikacja teraz zabrania.

**Progi i rozstrzygnięcie.** Progi H3 konfirmacyjnych pochodzą z planu sprzed prototypu (R² odległości > 0,5, AUC obrońców w stożku > 0,7). Progi opisowe ustalono w tym dokumencie, przy znanych liczbach deweloperskich prototypu (sekcja 19).

| Cel | Metryka | Próg | Rola |
|---|---|---|---|
| dist | R² | 0,50 | konfirmacyjny |
| def_cone_ge1 | AUC | 0,70 | konfirmacyjny |
| def_cone_ge2, head | AUC | 0,70 | opisowy |
| angle, ff_def_cone, gk_depth, gk_off_line_abs, gk_dist_goal, ff_opp_3m | R² | 0,25 | opisowy |

Rozstrzyga **dolna granica** percentylowego przedziału 95%, nie estymata punktowa (kod `shells.h3_reading`):

- dolna granica ≥ próg: „koduje”;
- górna granica < próg: „nie koduje (poniżej progu)”;
- inaczej: „nierozstrzygnięty”.

Estymata punktowa jest raportowana obok, ale nie rozstrzyga. Dla celów opisowych odczyt liczy się tak samo, tylko bez statusu testu.

**Różnica G-combo − DINOv2.** Na tych samych wierszach, z tymi samymi losowaniami, dla wszystkich celów. Odczyt (`shells.difference_reading`):

- dolna granica > 0: „G > DINOv2”;
- górna granica < 0: „G < DINOv2”;
- inaczej: „brak rozróżnialnej różnicy”.

To wynik drugorzędny, ustalony z góry i bez korekty na wielokrotność. Dla dwóch celów konfirmacyjnych jest raportowany na pierwszym miejscu.

**Przedziały są warunkowe.** Losowania przeważają stałe predykcje out-of-fold. Sondy nie są dopasowywane od nowa (jak w `xgpoc.probe.readability`). Przedziały pomijają więc zmienność samego dopasowania sondy (wybór α, współczynniki) i są węższe niż pełna niepewność. Reguła rozstrzygnięcia mimo to używa tych przedziałów. Raport to podaje i oznacza dolne granice, które leżą w odległości do 0,02 od progu.

## 12. H4: diagnostyka przecieku (opisowa)

**Populacja.** Strzały pierwotne bez gola SoccerNet, bez karnych, z ocenioną klatką przy offsecie 0 (`xgpoc.vendor.leakage`).

**Celność.** `on_target_sn`, definicja SoccerNet „Shots on target”, która liczy strzały zablokowane jako celne (D4).

**Kontrole stałe:**

- logit xG dostawcy;
- część ciała StatsBomb: wskaźniki poziomów;
- typ strzału StatsBomb: wskaźniki poziomów.

Poziom odniesienia to poziom najczęstszy (przy remisie pierwszy alfabetycznie). Poziomy nieobecne w wierszach wypadają.

**Części:**

1. **Celność.** OLS różnicy logit(klatka) − logit(dostawca) na `on_target_sn` z kontrolami. To wiersz główny H4. Obok wiersz bez kontroli, jako odniesienie.
2. **Kontrast offsetów.** G(0) − G(−400 ms) na skalibrowanych logitach (`frame_logit_0 − frame_logit_m400`, `offset_logits = calibrated`). Raportujemy jego średnią i OLS kontrastu na `on_target_sn` z kontrolami i bez nich.
3. **Ramię reszty czasowej.** Werdykt ticketu 04 brzmi „independent” (reguła 42dcf710…), więc ramię jest włączone. Skrypt czyta rekord werdyktu i przyjmuje go tylko z tym hashem reguły (`--expected-rule-sha256`).
   - OLS różnicy logitów na reszcie czasowej w sekundach (obciętej do ±1 s) i, osobno, na wskaźnikach kotwicy wcześniejszej o ponad 40 ms i późniejszej o ponad 40 ms niż StatsBomb.
   - Każdy model z kontrolami i bez nich, na wszystkich wierszach.
   - Wrażliwość ustalona z góry: te same modele ograniczone do |reszta| ≤ 200 ms. Wrażliwość nie zastępuje wierszy głównych; raport podaje obie wersje obok siebie.

**Odczyt.** Bez progów i bez werdyktu. Przedziały 95% z tych samych losowań.

- Dodatni współczynnik celności z kontrolami jest zgodny z odczytem kierunku piłki po kontakcie. Jest też zgodny z informacją sprzed kontaktu, której dostawca nie ma, np. z równowagą strzelca.
- Dodatni kontrast offsetów na strzałach celnych ma te same dwa wyjaśnienia, a do tego rozwój akcji w ostatnich 400 ms.
- Raport zawsze wymienia oba wyjaśnienia.

## 13. Narracje

Obie wersje każdej narracji są napisane przed wynikiem. Raport użyje tej, którą wskaże odczyt z S5 i S6, bez zmian treści poza liczbami.

### H2

**Klatka nie dodaje więcej niż δ** (odczyt „równoważne przy δ: klatka nie dodaje więcej niż δ” na populacji pierwotnej):

> Na <n> strzałach sparowanych w 135 meczach dodanie jednej klatki transmisji do modelu pozycyjnego zmieniło log-loss o Δ = <Δ> (95% [<lo>; <hi>]). Test równoważności wykluczył zysk i stratę większe niż δ = <δ> (p TOST = <p>). Klatka nie niesie informacji o szansie gola ponad to, co zawierają zdarzenia i freeze frame StatsBomb, w granicy δ. Przy tym klatka sama jest wyraźnie słabsza od pozycji (S4). Najprostsze wyjaśnienie: to, co klatka widzi (położenie strzelca, obrońców i bramkarza), dostawca zapisuje dokładniej, a czego dostawca nie zapisuje, jedna klatka nie pokazuje. Najmniejszy margines równoważności wynosi <m>, więc twierdzenie o braku zysku ma tę precyzję i nie większą. Populacja łączona dała <odczyt łączonej>.

**Klatka dodaje** (odczyt z przewagą na populacji pierwotnej):

> Na <n> strzałach sparowanych dodanie jednej klatki do modelu pozycyjnego obniżyło log-loss o <|Δ|> (95% [<lo>; <hi>], p przewagi = <p>). Klatka niesie więc informację o szansie gola, której zdarzenia i freeze frame StatsBomb nie zawierają. Zysk jest <mniejszy niż δ = <δ> / większy niż δ>, czyli <rzędu / mniejszy niż> różnica między szczeblami zdarzeń i freeze frame (S4). H4 pokazuje, na ile ten zysk może pochodzić z treści po kontakcie: <wiersz główny H4>. Bez H4 nie wolno tego zysku przypisać informacji sprzed strzału. Populacja łączona dała <odczyt łączonej>.

**Nierozstrzygnięty:** „Dane nie rozstrzygają: Δ = <Δ> (95% [<lo>; <hi>]). Równoważność wykazuje się dopiero przy marginesie <m>, a przewagi nie ma.” Raport nie nazywa tego brakiem efektu.

### H3

**Klatka koduje geometrię** (oba cele konfirmacyjne „koduje”):

> Z embeddingu jednego modelu foldu G-combo da się liniowo odczytać odległość strzału (R² = <r> [<lo>; <hi>], próg 0,50) i obecność obrońcy w stożku strzału (AUC = <a> [<lo>; <hi>], próg 0,70). Klatka zawiera więc te wielkości pozycyjne, z których korzysta dostawca. <cele opisowe>. Sonda liniowa daje tylko dolne ograniczenie tego, co embedding zawiera.

**Klatka nie koduje** (choć jeden cel konfirmacyjny „nie koduje” albo „nierozstrzygnięty”):

> Z embeddingu G-combo nie da się liniowo odczytać <celu> na poziomie progu (<wartość> [<lo>; <hi>], próg <t>). Sonda liniowa daje dolne ograniczenie, więc wynik nie dowodzi, że informacji w embeddingu nie ma. Dowodzi tylko, że nie jest dostępna liniowo, w formie, w jakiej korzysta z niej model pozycyjny. To częściowo tłumaczy stratę klatki wobec pozycji w H1.

**Dostrajanie a zamrożony szkielet.** Obie wersje są napisane neutralnie, bo dane deweloperskie wskazują w dwie strony:

- prototyp (pięć modeli foldów wymieszanych w jednej sondzie, 2 090 strzałów) nie widział przewagi dostrojonego G-combo nad zamrożonym DINOv2;
- ticket 09 (jeden model foldu, 489 i 682 strzały) widział przewagę G-combo, np. odległość +0,071 [+0,025; +0,122] dla foldu 0.

Obie liczby są deweloperskie.

> (G > DINOv2) Dostrajanie na etykiecie gola zwiększyło liniową czytelność <celów> o <Δ> [<lo>; <hi>] względem zamrożonego szkieletu z tymi samymi wagami. Model uczył się więc reprezentować geometrię strzału, a nie tylko korzystać z cech, które szkielet już miał. Wcześniejsza obserwacja z prototypu, że dostrajanie nie pomaga, pochodziła z sondy na wymieszanych przestrzeniach embeddingów pięciu modeli i jej nie potwierdzamy.

> (G ≈ DINOv2 albo G < DINOv2) Zamrożony DINOv2 koduje <cele> tak samo dobrze jak dostrojony G-combo (różnica <Δ> [<lo>; <hi>]) <albo: lepiej>. Geometria strzału jest dostępna już w szkielecie, a dostrajanie na etykiecie gola jej nie wzmacnia. Przewaga G-combo widziana w danych deweloperskich ticketu 09 na małych podzbiorach się nie potwierdziła.

## 14. Szablony tabel

Kod `xgpoc.vendor.shells` wypełnia szablony z wyników skryptów analizy. Wiersze i kolumny są stałe. Komórka bez wartości dostaje „—”, a populacja pominięta w danym przebiegu dostaje „pominięta”. Tabele S2–S4 powtarzają się dla populacji pierwotnej i łączonej. Pełne tabele (wszystkie pary kontrastów, krzywe równoważności, wszystkie współczynniki H4, wszystkie podzbiory sond) zostają w plikach CSV przebiegu. Test `tests/test_vendor_prereg.py` sprawdza, że puste szablony poniżej są dokładnie tym, co renderuje kod.

#### S1. Populacje analizy H1/H2: liczności i wykluczenia

| Populacja | Wybrane | Karne | Bez klatki | Analizowane | Gole | Mecze |
|---|---|---|---|---|---|---|
| pierwotna |  |  |  |  |  |  |
| łączona |  |  |  |  |  |  |

#### S2. H1, drabina informacji, warianty rekalibrowane (populacja: pierwotna; ta sama tabela dla łączonej)

| Szczebel | Log-loss [95%] | Brier [95%] | AUC [95%] | Nachylenie kalibracji [95%] | Wyraz wolny kalibracji [95%] | Śr. p − częstość [95%] |
|---|---|---|---|---|---|---|
| stała |  |  |  |  |  |  |
| odległość i kąt |  |  |  |  |  |  |
| zdarzenia |  |  |  |  |  |  |
| freeze frame (pozycje) |  |  |  |  |  |  |
| xG dostawcy |  |  |  |  |  |  |
| klatka (G-combo) |  |  |  |  |  |  |
| klatka + pozycje |  |  |  |  |  |  |
| klatka + dostawca |  |  |  |  |  |  |

#### S3. H1, drabina informacji, warianty surowe (populacja: pierwotna; ta sama tabela dla łączonej)

| Szczebel | Log-loss [95%] | Brier [95%] | AUC [95%] | Nachylenie kalibracji [95%] | Wyraz wolny kalibracji [95%] | Śr. p − częstość [95%] |
|---|---|---|---|---|---|---|
| stała |  |  |  |  |  |  |
| odległość i kąt |  |  |  |  |  |  |
| zdarzenia |  |  |  |  |  |  |
| freeze frame (pozycje) |  |  |  |  |  |  |
| xG dostawcy |  |  |  |  |  |  |
| klatka (G-combo) |  |  |  |  |  |  |

#### S4. H1, kontrasty kluczowe, warianty rekalibrowane (populacja: pierwotna; ta sama tabela dla łączonej)

| Kontrast (A − B) | Δ log-loss [95%] | Δ log-loss, 90% | Najmniejszy margines równoważności | Δ AUC [95%] |
|---|---|---|---|---|
| klatka (G-combo) − xG dostawcy |  |  |  |  |
| klatka (G-combo) − freeze frame (pozycje) |  |  |  |  |
| xG dostawcy − freeze frame (pozycje) |  |  |  |  |
| freeze frame (pozycje) − zdarzenia |  |  |  |  |
| zdarzenia − odległość i kąt |  |  |  |  |
| odległość i kąt − stała |  |  |  |  |
| klatka + pozycje − klatka (G-combo) |  |  |  |  |
| klatka + dostawca − klatka (G-combo) |  |  |  |  |

#### S5. H2 i klatka + dostawca (δ z sekcji 10)

| Kontrast | Populacja | Rola | Log-loss A | Log-loss B | Δ [95%] | Δ, 90% | p TOST przy δ | p przewagi | Najmniejszy margines | Odczyt |
|---|---|---|---|---|---|---|---|---|---|---|
| klatka + pozycje − pozycje | pierwotna | konfirmacyjny |  |  |  |  |  |  |  |  |
| klatka + pozycje − pozycje | łączona | odporność |  |  |  |  |  |  |  |  |
| klatka + dostawca − dostawca | pierwotna | opisowy |  |  |  |  |  |  |  |  |
| klatka + dostawca − dostawca | łączona | opisowy |  |  |  |  |  |  |  |  |

#### S6. H3, sondy: G-combo fold 0 seed 0 (model rozstrzygający) wobec zamrożonego DINOv2; S7 to ta sama tabela dla foldu 1

| Cel | Metryka | n | G-combo [95%] | DINOv2 [95%] | G − DINOv2 [95%] | Próg | Rola | Odczyt progu | Odczyt różnicy |
|---|---|---|---|---|---|---|---|---|---|
| dist_shuffled | R2 |  |  |  |  | R2 ≤ 0 | kontrola negatywna |  |  |
| dist | R2 |  |  |  |  | 0.50 | konfirmacyjny |  |  |
| angle | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| ff_def_cone | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| def_cone_ge1 | AUC |  |  |  |  | 0.70 | konfirmacyjny |  |  |
| def_cone_ge2 | AUC |  |  |  |  | 0.70 | opisowy |  |  |
| gk_depth | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| gk_off_line_abs | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| gk_dist_goal | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| ff_opp_3m | R2 |  |  |  |  | 0.25 | opisowy |  |  |
| head | AUC |  |  |  |  | 0.70 | opisowy |  |  |

#### S8. H4, diagnostyka przecieku (opisowa)

„Kontrole: tak” oznacza logit xG dostawcy oraz wskaźniki części ciała i typu strzału.

| Część | Specyfikacja | Regresor | Kontrole | Ograniczenie | n | Współczynnik [95%] |
|---|---|---|---|---|---|---|
| celność | gap_on_target | celny (SN) | tak | brak |  |  |
| celność | gap_on_target_unadjusted | celny (SN) | nie | brak |  |  |
| kontrast offsetów | offset_mean | średnia G(0) − G(−400) | nie | brak |  |  |
| kontrast offsetów | offset_on_target | celny (SN) | tak | brak |  |  |
| kontrast offsetów | offset_on_target_unadjusted | celny (SN) | nie | brak |  |  |
| reszta czasowa | gap_residual_slope | reszta [s] | tak | brak |  |  |
| reszta czasowa | gap_residual_slope_unadjusted | reszta [s] | nie | brak |  |  |
| reszta czasowa | gap_early_late | kotwica > 40 ms przed SB | tak | brak |  |  |
| reszta czasowa | gap_early_late | kotwica > 40 ms po SB | tak | brak |  |  |
| reszta czasowa | gap_early_late_unadjusted | kotwica > 40 ms przed SB | nie | brak |  |  |
| reszta czasowa | gap_early_late_unadjusted | kotwica > 40 ms po SB | nie | brak |  |  |
| reszta czasowa | gap_residual_slope_within_200ms | reszta [s] | tak | reszta w ±200 ms |  |  |
| reszta czasowa | gap_residual_slope_unadjusted_within_200ms | reszta [s] | nie | reszta w ±200 ms |  |  |
| reszta czasowa | gap_early_late_within_200ms | kotwica > 40 ms przed SB | tak | reszta w ±200 ms |  |  |
| reszta czasowa | gap_early_late_within_200ms | kotwica > 40 ms po SB | tak | reszta w ±200 ms |  |  |
| reszta czasowa | gap_early_late_unadjusted_within_200ms | kotwica > 40 ms przed SB | nie | reszta w ±200 ms |  |  |
| reszta czasowa | gap_early_late_unadjusted_within_200ms | kotwica > 40 ms po SB | nie | reszta w ±200 ms |  |  |

## 15. Dry run

Dry run (`scripts/vendor_dryrun.py`) przechodzi całą ścieżkę konfirmacyjną na etykietach bez związku z predykcjami:

1. Tabela bez etykiet, sprawdzona z pinem, zostaje zapisana jeszcze raz z `table_kind = dryrun`. Kod sprawdza, że nic innego się nie zmienia.
2. Etykiety wierszy pierwotnych z tabeli par są permutowane w obrębie meczów (`xgpoc.vendor.dryrun.permute_labels`):
   - warstwy (mecz, karny);
   - jedna permutacja przenosi razem gol i wynik StatsBomb;
   - druga przenosi celność wśród wierszy pierwotnych;
   - `default_rng([permutation_seed, 1])`.
   Potem etykiety przechodzą przez logowany krok łączenia. Żadna niepermutowana etykieta nie jest łączona z predykcją ani wypisywana.
3. Cele sond: wiersze tabeli strzałów StatsBomb dla zdarzeń tabeli (lokalizacja, część ciała, freeze frame) są permutowane razem w warstwach (mecz, karny), `default_rng([permutation_seed, 2])`. Sondy liczą wszystkie cele z permutowanych wierszy.
4. Rekord roboczy pre-rejestracji (`draft = true`) bieżącego dokumentu podaje parametry. Dopóki δ jest otwarte, ten rekord niesie `dry_run_margin`. Rekord roboczy nie dopuszcza łączenia etykiet z tabelą konfirmacyjną.
5. Trzy skrypty analizy biegną z opcjami przebiegu konfirmacyjnego:
   - `vendor_ladder.py` z parametrami z rekordu;
   - `vendor_probes.py` z progami z bloku parametrów;
   - `vendor_leakage.py` z rekordem werdyktu przypiętym do hasha reguły i ze skalibrowanymi logitami offsetów.
6. Kod wypełnia szablony S1–S8 (`shells.md`) i zapisuje rekord ukończenia z hashem dokumentu i kodu.

Permutacja w obrębie meczów zachowuje różnice między meczami (częstość goli, średnią odległość). Liczby z dry run nie muszą więc leżeć przy zerze. Nie są wynikami i nie wolno ich cytować jako wyników.

Czego permutacja nie ukrywa:

- liczności w S1 (wiersze, karne, gole, mecze), znane z indeksu;
- wiersze H4, które nie czytają etykiety: średni kontrast offsetów i ramię reszty czasowej. Różnica logitów, kontrast i reszta czasowa są kolumnami tabeli bez etykiet. Etykieta wchodzi tylko przez wybór strzałów bez gola, a ten permutacja zmienia w niewielkim stopniu. Te wiersze dry run pokazuje więc w przybliżeniu w wartościach przebiegu konfirmacyjnego. H4 jest opisowa i żadna decyzja nie zależy od tych wierszy.

Przebiegi:

- tabela pierwotna, 2026-10-11, wyniki w `results/xgv_dryrun_primary/` (hash rekordu ukończenia w sekcji 17; 43 s CPU, 5 000 losowań).
  - Wypełnione są wszystkie szablony; populacja łączona jest oznaczona jako pominięta, bo tabela ma tylko wiersze pierwotne.
  - Analizowane: 2 721 strzałów w H1/H2, 2 721 (cele bramkarza 2 719) w H3, 2 347 strzałów bez gola w H4.
  - Kontrola negatywna sond przechodzi dla obu modeli foldu i DINOv2.
  - Ramię reszty czasowej uruchomiło się z werdyktem przypiętym do hasha reguły.
  - Odczyt H2 na etykietach permutowanych to „równoważne przy δ; stos istotnie gorszy”. Gdy nic nie przewiduje etykiety, stos płaci za drugi współczynnik. To zgadza się z dodatnią średnią H2 przy prawdzie zerowej w symulacji mocy (+0,0002).
- tabela łączona: DO UZUPEŁNIENIA (`results/xgv_dryrun_combined/`).

Przygotowanie dry run ujawniło dwa błędy ścieżki konfirmacyjnej, naprawione przed zamrożeniem kodu. Pierwszy: `vendor_probes.py` szukał względnych plików embeddingów w katalogu roboczym, a nie w katalogu tabeli, i przerwałby przebieg konfirmacyjny. Drugi: rekordy sond i przecieku oznaczały każdą tabelę inną niż deweloperska jako „confirmatory run”.

## 16. Parametry

Blok poniżej jest jedynym źródłem parametrów przebiegu konfirmacyjnego. `scripts/vendor_prereg_hashes.py` go czyta i sprawdza każdy pin z plikiem. Rekord pre-rejestracji przenosi `margin`, `n_boot`, `fold_seed` i `boot_seed`, a `vendor_ladder.py` czyta je z rekordu; z linii poleceń ich nie przyjmuje. `null` to pozycja do uzupełnienia. Pin `timing_rule` jest też hashem, z którym `vendor_leakage.py` przyjmuje werdykt.

<!-- prereg-parameters:begin -->
```json
{
  "margin": null,
  "dry_run_margin": 0.005,
  "n_boot": 5000,
  "fold_seed": 0,
  "boot_seed": 0,
  "permutation_seed": 0,
  "offset_logits": "calibrated",
  "h3_thresholds": {"dist": 0.5, "angle": 0.25, "ff_def_cone": 0.25, "def_cone_ge1": 0.7, "def_cone_ge2": 0.7, "gk_depth": 0.25,
                    "gk_off_line_abs": 0.25, "gk_dist_goal": 0.25, "ff_opp_3m": 0.25, "head": 0.7},
  "h3_confirmatory_targets": ["dist", "def_cone_ge1"],
  "pins": {
    "primary_label_free_table": {"path": "results/xgv_eval_primary/eval_table_labels_absent.parquet",
                                 "sha256": "829aa7edb59e99423cdf379c8cb656c59536a42bea38f4f64bec93cc9fd3c741"},
    "primary_label_free_sidecar": {"path": "results/xgv_eval_primary/eval_table_labels_absent.json",
                                   "sha256": "628b8ba7b93528a81eb9c26ad5e67485a4a4090c977e10a3dd3bb902bda438ed"},
    "combined_label_free_table": {"path": null, "sha256": null},
    "split_plan": {"path": "results/xgv_split_plan/plan.csv", "sha256": "a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a"},
    "split_plan_record": {"path": "results/xgv_split_plan/complete.json", "sha256": "9f828694f782a427b8ce6f0abcbf4e5cb439c74b221c34505432f48d7208d64b"},
    "comparator_predictions": {"path": "results/xgv_comparator/predictions.parquet",
                               "sha256": "6c66b2f5d4f25a6b60d4ed167435ecac49a5192e569adfac5f765f82003c1a56"},
    "comparator_record": {"path": "results/xgv_comparator/complete.json", "sha256": "84f16f555a2c308048a2fad2576c14dde2a1a0d86bb551efcb10869c7c8d2136"},
    "timing_verdict": {"path": "results/xgv_timing_source/timing_verdict.json",
                       "sha256": "3903e8f7c7f35f2d9059850ec62fa7c69bed74ba597e6f28e2adb83440a66a59"},
    "timing_rule": {"path": "research/xg_vendor_timing_rule.md", "sha256": "42dcf71018f28c078e5fa6ff83e71f137ac77e13c45fdb2534608314882029c5"},
    "delta_rule": {"path": "research/xg_vendor_delta_rule.md", "sha256": "766ee11b3ba5ea33d7a6e9dce8bce691fbc9468815cfe5f7b6b320032e5ff30a"},
    "power_decision": {"path": "results/xgv_power/decision.json", "sha256": "af4080beb5100a06fa28accab6663e96200cb9d81a7df458c7b5d979ff1afedf"},
    "pairs": {"path": "results/xgv_pairing/pairs.parquet", "sha256": "b07be928c8c7d457f80e049a411925234fafca1fe530fbbdcdb5d2b3979a64f5"},
    "pairing_record": {"path": "results/xgv_pairing/complete.json", "sha256": "8807ee42cba33062a07ca9136faab8b63427e2bd9a2e445ba902716fc14efc75"},
    "g_combo_ensemble": {"path": "results/xgv_g_combo/ensemble.json", "sha256": "045542a5439a586eac6c460e9fffe19eab150dd91de52cad92c1130e7f289f15"},
    "statsbomb_shots": {"path": "/home/kuba/projects/football_data/statsbomb_open_v1/tables/shots.parquet",
                        "sha256": "c11c77ab58e32924501488a596e88dda0082932b2b445d6e13abc0febbfb4aed"}
  }
}
```
<!-- prereg-parameters:end -->

Embeddingi (`emb_g_fold0.npy`, `emb_g_fold1.npy`, `emb_dinov2.npy`) przypina plik boczny tabeli: pin `primary_label_free_sidecar`, a po tickecie 13 plik boczny tabeli łączonej. `vendor_probes.py` sprawdza sha256 każdego pliku z plikiem bocznym.

## 17. Hashe

Blok poniżej generuje `scripts/vendor_prereg_hashes.py`; nie edytuje się go ręcznie. Lista obejmuje:

- piny z bloku parametrów, każdy sprawdzony z plikiem;
- kod analizy: skrypty `vendor_ladder.py`, `vendor_probes.py`, `vendor_leakage.py`, `vendor_dryrun.py`, `vendor_prereg_hashes.py` i każdy moduł pakietu `xgpoc`, który importują, przechodnio;
- rekordy ukończenia dry runów.

Hashe kodu w szkicu są robocze i zostaną przeliczone przy finalizacji. Kod kroku łączenia etykiet populacji wtórnej (ticket 13) dojdzie do listy skryptów wejściowych (`prereg.ENTRY_SCRIPTS`): DO UZUPEŁNIENIA

<!-- prereg-hashes:begin -->
Lista hashy: `research/xg_vendor_prereg_hashes.json`, sha256 `94825daabf62869cd0455a51b309cebe76e388765f40814f28d12cd8af13af0c`, status: **SZKIC**.

| Wejście | Plik | sha256 |
|---|---|---|
| combined_label_free_table | — | do uzupełnienia |
| comparator_predictions | results/xgv_comparator/predictions.parquet | `6c66b2f5d4f25a6b60d4ed167435ecac49a5192e569adfac5f765f82003c1a56` |
| comparator_record | results/xgv_comparator/complete.json | `84f16f555a2c308048a2fad2576c14dde2a1a0d86bb551efcb10869c7c8d2136` |
| delta_rule | research/xg_vendor_delta_rule.md | `766ee11b3ba5ea33d7a6e9dce8bce691fbc9468815cfe5f7b6b320032e5ff30a` |
| g_combo_ensemble | results/xgv_g_combo/ensemble.json | `045542a5439a586eac6c460e9fffe19eab150dd91de52cad92c1130e7f289f15` |
| pairing_record | results/xgv_pairing/complete.json | `8807ee42cba33062a07ca9136faab8b63427e2bd9a2e445ba902716fc14efc75` |
| pairs | results/xgv_pairing/pairs.parquet | `b07be928c8c7d457f80e049a411925234fafca1fe530fbbdcdb5d2b3979a64f5` |
| power_decision | results/xgv_power/decision.json | `af4080beb5100a06fa28accab6663e96200cb9d81a7df458c7b5d979ff1afedf` |
| primary_label_free_sidecar | results/xgv_eval_primary/eval_table_labels_absent.json | `628b8ba7b93528a81eb9c26ad5e67485a4a4090c977e10a3dd3bb902bda438ed` |
| primary_label_free_table | results/xgv_eval_primary/eval_table_labels_absent.parquet | `829aa7edb59e99423cdf379c8cb656c59536a42bea38f4f64bec93cc9fd3c741` |
| split_plan | results/xgv_split_plan/plan.csv | `a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a` |
| split_plan_record | results/xgv_split_plan/complete.json | `9f828694f782a427b8ce6f0abcbf4e5cb439c74b221c34505432f48d7208d64b` |
| statsbomb_shots | /home/kuba/projects/football_data/statsbomb_open_v1/tables/shots.parquet | `c11c77ab58e32924501488a596e88dda0082932b2b445d6e13abc0febbfb4aed` |
| timing_rule | research/xg_vendor_timing_rule.md | `42dcf71018f28c078e5fa6ff83e71f137ac77e13c45fdb2534608314882029c5` |
| timing_verdict | results/xgv_timing_source/timing_verdict.json | `3903e8f7c7f35f2d9059850ec62fa7c69bed74ba597e6f28e2adb83440a66a59` |
| dowód: dry_run_primary | results/xgv_dryrun_primary/complete.json | `66519c7b4f3888b83c9aa9fde09b1f9688e1515d263e519f4f2072ea4c472875` |
| dowód: dry_run_combined | results/xgv_dryrun_combined/complete.json | brak (do wykonania) |

Kod analizy (29 plików):

| Plik | sha256 |
|---|---|
| scripts/vendor_dryrun.py | `a5940853a4e7de1b86ba3a469c38dca3456bd863222e863b800545e408d4e87b` |
| scripts/vendor_ladder.py | `37c46bfffe455f527e330529e5fecd68e8066bef22719e85dced78ed5f683c83` |
| scripts/vendor_leakage.py | `821e9c94e0c528316d53b7c18245a6b8c3f22897abfe1c61b1391250af2f560d` |
| scripts/vendor_prereg_hashes.py | `134ec17a531c942d7079310c32c9dd4b59b09fce5a662fd8f7b32e6a308030bc` |
| scripts/vendor_probes.py | `e6041916ae4f46ea3e1e41efa78b4d334ccae283756a835b5846038d8f446590` |
| xgpoc/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| xgpoc/bundle.py | `05cc32bbb5ebf00089395ad659e24aec585d7f7f57da9672e04d8cb13c6a517d` |
| xgpoc/data/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| xgpoc/data/canonical_frames.py | `4d0c1c03f13e6282f3087877a2369bf9a7077a247ffe6349e8058a84e2bde46b` |
| xgpoc/data/datasets.py | `ff9bf7bc81491f9e13d4433a8f934e47bead61c85134a2e940b6494e91603c2d` |
| xgpoc/data/offset_frames.py | `cefb22618a37d910f1a77fa562abe83598dbfaf0af33b669f69975fcb823d195` |
| xgpoc/eval/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| xgpoc/eval/calibration.py | `e2e73f525ef6de22bf3cb121128430e1cbc75418152516d68aba7961dd189df6` |
| xgpoc/probe/__init__.py | `f651a9148861c806c0a4e3851509d456f3d278593f2b99fba861a08d33ecf4d2` |
| xgpoc/probe/readability.py | `540d23f699b7e884362f07c827a72f9710f6e96c9e161b7ce5c67d443bdc0116` |
| xgpoc/probe/table.py | `d1cc2d77e331cb5d37b898e1694341fbb9973a31a474eff3b3557a586bd58ceb` |
| xgpoc/vendor/__init__.py | `aa7741ce1a94ea50fc915f440569855ed2e33c435c81cb1c47fc0912168ea195` |
| xgpoc/vendor/draws.py | `fc88a16636cd4e558b5264081a7a6d245d6a4b981f0c1cad7b33ba9649bfe62c` |
| xgpoc/vendor/dryrun.py | `1074c98257a662c74e87da4f430a0882655c674d115afd5ac6f338cd3f55b8ab` |
| xgpoc/vendor/evaltable.py | `7cb1d108d9247f8b901208120bd4f95e95487a370ef64407269bc2b36b845157` |
| xgpoc/vendor/labels.py | `0ac7a6a81b1d0609252e0ff4bcd0fd4634c0c6a37b5c0da7cdae758ea4651421` |
| xgpoc/vendor/ladder.py | `585ae401fd0a9f9aac50d9395c74c1579da7dceed42b8db67d0ef082e3bf9465` |
| xgpoc/vendor/ladder_dev.py | `7f8a7404e1bfe9d2f1f4687e49383cd096ddaf318183057d6cfda0b17fb6d525` |
| xgpoc/vendor/leakage.py | `1206a8c1391f03f9cd3bf0d1b281d6e48899cb9acf6b6b39066f6cc902bf3be8` |
| xgpoc/vendor/prereg.py | `404380f551c826e7d6f951f675b2834c50a724eddca82f62cfe838511f2e6427` |
| xgpoc/vendor/probe_targets.py | `e9ec5bb125d9522885c924bd3acfdd76008790c00be97643e2b2d24fdc1e6293` |
| xgpoc/vendor/probes.py | `405e4b2835dbdd3fdeb1f8b101e6a7cda60d93e6b236aaf2567656c78db12706` |
| xgpoc/vendor/shells.py | `7cd0526d0d6f6551cf4c9dd264bc3a8f791af4164cc9d84bb1a8526e943f62ea` |
| xgpoc/vendor/timing_verdict.py | `ba50b1aea4c65ccb137f3c81ed06d301b260a7e48c7ab78c40a83b2d53fc3b02` |

Otwarte pozycje: margin; pin combined_label_free_table; 10 'DO UZUPEŁNIENIA' marks in the document; evidence dry_run_combined.
<!-- prereg-hashes:end -->

## 18. Przebieg konfirmacyjny (ticket 15)

Kolejność, bez innych kroków:

1. `./run.sh python scripts/vendor_prereg_hashes.py --check` musi przejść bez zmian.
2. Łączenie etykiet przez `evaltable.join_labels` z rekordem z `results/xgv_prereg/preregistration_record.json` (etykiety pierwotne: `labels.primary_labels`; wtórne: reguła ticketu 13).
3. `./run.sh python scripts/vendor_ladder.py --table-dir <tabela łączona> --table-name <tabela z etykietami> --prereg <rekord> --out results/xgv_ladder`.
4. `./run.sh python scripts/vendor_probes.py --table <tabela łączona> --name <tabela z etykietami> --prereg <rekord> --n-boot 5000 --seed 0 --thresholds <progi z bloku parametrów> --out results/xgv_probes`.
5. `./run.sh python scripts/vendor_leakage.py --table <tabela łączona> --name <tabela z etykietami> --prereg <rekord> --n-boot 5000 --seed 0 --offset-logits calibrated --timing-verdict results/xgv_timing_source/timing_verdict.json --expected-rule-sha256 42dcf71018f28c078e5fa6ff83e71f137ac77e13c45fdb2534608314882029c5 --out results/xgv_leakage`.
6. Wypełnienie szablonów (`shells.render`) i raport.

Każdy krok uruchamia się raz. Powtórka po błędzie technicznym (przerwany proces, brak dysku) jest dozwolona tylko z tymi samymi wejściami, a dziennik ją odnotowuje.

## 19. Odstępstwa i ograniczenia

**Odstępstwa.**

- Przed zatwierdzeniem: zmianę dokumentu wprowadza nowa wersja, a hash ją przypina.
- Po zatwierdzeniu: każda zmiana analizy jest oznaczona jako post hoc i raportowana obok analizy z pre-rejestracji. Werdykty H2 i H3 zostają werdyktami reguł tego dokumentu.
- Jeśli kroku nie da się wykonać (brak plików, błąd kodu, losowanie bez goli), agent zatrzymuje się i opisuje problem w tickecie. Nie dobiera zastępczej reguły.

Dziennik odstępstw: brak.

**Ograniczenia, które raport musi wymienić:**

- Progi opisowe H3 (0,25 dla R², 0,70 dla AUC) ustalono przy znanych liczbach deweloperskich prototypu i ticketu 09. Progi konfirmacyjne pochodzą z planu sprzed prototypu.
- Bootstrap ze stałymi predykcjami jest węższy od pełnej niepewności: w H2 o około 8–12% (ticket 08), w H3 o nieznaną część zmienności dopasowania sondy.
- H2 dotyczy szczebla freeze frame naszego komparatora, który dorównuje dostawcy (K1). Nie dotyczy modelu samego dostawcy, który nie ma krzyżowo dopasowanej wersji „z klatką”. Kontrast klatka + dostawca − dostawca jest tylko opisowy.
- Populacja pierwotna to strzały, które SoccerNet oznaczył. SoccerNet pomija około 42% strzałów zablokowanych i 68% niecelnych strzałów StatsBomb. Populacja łączona to częściowo naprawia, ale jej klatki mają kotwicę w czasie StatsBomb.
- Kluby z meczów ewaluacyjnych występują w meczach treningowych innych sezonów, jak w archiwalnym podziale testowym.
- Klatka nie widzi wysokości piłki. Dostawca zna wysokość kontaktu, technikę i presję (raport: tabela „co widzi które źródło”).
