# Pre-rejestracja: xG z jednej klatki transmisji wobec xG dostawcy i xG pozycyjnego (xg-vendor, ticket 14)

Status: **wersja końcowa, do zatwierdzenia przez właściciela (2026-10-12).** Dokument, lista hashy (sekcja 17) i kod analizy są zamrożone w jednym commicie przed jakimkolwiek łączeniem etykiet. Otwarte zostają tylko decyzje z sekcji 0. Każda decyzja, od której zależy przebieg analizy, jest parametrem bloku parametrów (sekcja 16) i ma tam wpisaną wartość zalecaną. Zatwierdzenie bez zmian i zatwierdzenie ze zmianą prowadzą przez procedurę z sekcji 0.2.

Do chwili zatwierdzenia żadna etykieta nie zostaje połączona z nowymi predykcjami. Jedynymi przebiegami analizy na tych predykcjach są dwa dry runy na etykietach i celach sond permutowanych w obrębie meczów (sekcja 15).

Wszystkie liczby oznaczone „deweloperskie” pochodzą z prototypu, z tabeli deweloperskiej (ticket 07) albo z symulacji mocy (ticket 08). Modele, które je dały, widziały w treningu większość tych meczów. To nie są wyniki.

## 0. Decyzje wymagane przed łączeniem etykiet

### 0.1 Decyzje

Decyzje A, B i F to parametry z wartością zalecaną; C i D to reguły w kodzie, które właściciel potwierdza; E to samo zatwierdzenie. Decyzje podjęte wcześniej i tylko wpisane tutaj: δ = 0,005 (sekcja 10), część konfirmacyjna H3 zawężona do `dist` i `def_cone_ge1` z poprawką Bonferroniego (sekcja 11), pakiet replikacyjny bez danych StatsBomb (ticket 16). Do decyzji zostaje:

| | Decyzja | Parametr | Wartość wpisana (zalecana) | Alternatywa | Czego dotyczy |
|---|---|---|---|---|---|
| A | Bootstrap szczebli rekalibrowanych (H1, H2) | `recalibrated_bootstrap` | `"refit"`: modele dopasowywane od nowa w każdym losowaniu | `"fixed"`: stałe predykcje out-of-fold | przedziały i p-wartości H2 obu populacji i wariantu rekalibrowanego H1 (sekcja 7.1) |
| B | Wiersz z uszkodzoną klatką | `frame_exclusions` | `{"1357b69b-baa6-406f-a238-12e493594942": "decode_corrupted_frame"}`: wiersz wypada | `{}`: wiersz zostaje | tylko populacja łączona (odporność H2, H1 opisowo) |
| C | 147 celów wtórnych bez klatki | reguła w kodzie (sekcja 3) | wiersze zostają w tabeli, wypadają z H1/H2 dla każdego szczebla, S1 je liczy, lista jest przypięta | opisowa drabina szczebli bez klatki na wszystkich 797 strzałach (niewdrożona) | tylko populacja łączona |
| D | 12 goli StatsBomb bez gola SoccerNet | reguła w kodzie (sekcja 4) | wykluczenie według flagi zamrożonego parowania (12 wierszy) | — | tylko populacja łączona |
| F | Próg bramki kontroli negatywnej sond (H3) | `h3_control_max_r2` | `0.005`: kontrola może dać R² do 0,005 | `0.0`: próg szkicu, R² ≤ 0 | ważność każdego odczytu H3 enkodera (sekcja 11) |
| E | Zatwierdzenie całości | — | — | — | cały dokument; potem przegląd właściciela („thermo-nuclear review”) |

**A. Bootstrap szczebli rekalibrowanych.** Szczegóły i liczby w sekcji 7.1. Krótko: bootstrap z dopasowaniem w każdym losowaniu (`refit`) odtwarza rozrzut próbkowania przy prawdzie deweloperskiej (SD 0,00120 wobec 0,00119), a przy prawdzie zerowej jest o około 70% za szeroki. Bootstrap ze stałymi predykcjami (`fixed`) jest o około 13% za wąski w obu scenariuszach. Konfirmacyjne twierdzenie H2 to równoważność przy δ = 0,005; za wąski przedział czyni TOST liberalnym, a za szeroki tylko konserwatywnym, przy δ = 0,005 praktycznie bez straty mocy. **Zalecenie: `refit`** (wpisane). Koszt: test przewagi jest konserwatywny, więc mały rzeczywisty zysk z klatki łatwiej przeoczyć. `fixed` zgadza się metodą z zamrożoną tabelą mocy (ticket 08), ale błąd ma w groźniejszą stronę.

**B. Wiersz z uszkodzoną klatką.** Audyt klatek i klipów ticketu 13 porównał klatkę i klip każdego z 847 celów wtórnych. Rozjechał się jeden: zdarzenie StatsBomb `1357b69b-baa6-406f-a238-12e493594942` (Stoke City 1-0 Chelsea, 1. połowa, 182,5 s, strzał tylko ze StatsBomb, typ „Open Play”). Ma klatki ze statusem „ok” i zgodnymi hashami, ale jego klip się nie wycina („Clip ends before the annotation anchor”). Klatka z offsetu 0 jest uszkodzona przy dekodowaniu: smugi i bloki na całym obrazie, kamera na kole środkowym, zegar transmisji 02:46 przy celu około 03:02. Offset tej połowy wynosi 0,36 s, więc nie tłumaczy różnicy 16 s. Wideo połowy jest w tym miejscu uszkodzone, a klatka pokazuje inny moment niż strzał. Wiersz został oceniony jak pozostałe (ticket 13 nie wymyślał reguły). Decyzja nie czyta żadnej etykiety: opiera się na audycie, który etykiet nie czyta, i na obrazie, który strzału nie pokazuje.
- Wykluczyć (wpisane, **zalecane**): wiersz jest traktowany jak wiersz bez klatki, w każdym szczeblu H1/H2 populacji łączonej, i liczony osobno w S1 (kolumna „Klatka wykluczona”). Za: reguła „wiersz bez klatki strzału wypada” (decyzja C) obejmuje ten przypadek co do treści; zostawienie oceniałoby model klatki na obrazie innej chwili.
- Zostawić (`frame_exclusions = {}`): wiersz wchodzi do analiz. Za: brak wykluczeń ręcznych; sygnał audytu (nieudany klip) nie wykrywa każdej uszkodzonej klatki, więc ewentualne inne takie klatki i tak zostają.
- Skutek: 1 z 700 ocenionych wierszy wtórnych, tylko populacja łączona. H2 konfirmacyjne, H3 i H4 liczą się na populacji pierwotnej i są nietknięte. Kod waliduje parametr: każdy klucz musi być wierszem wtórnym przypiętej tabeli łączonej z ocenioną klatką (`prereg.check_frame_exclusions`), a analiza odmawia klucza wiersza pierwotnego.

**C. 147 celów wtórnych bez klatki.** Wszystkie to strzały tylko ze StatsBomb; przy obu offsetach mają status `beyond_half_end`, czyli cel leży po końcu wideo połowy (doliczony czas). Leżą w 86 meczach (train 84, valid 30, test 33 wiersze). Reguła (sekcja 3): wiersz zostaje w tabeli ze statusem klatki i logitami NaN, wypada z H1/H2 populacji łączonej dla każdego szczebla (także dostawcy i pozycji, żeby wszystkie szczeble liczyły się na tych samych strzałach), S1 liczy go w kolumnie „Bez klatki”, a lista z przyczynami (`results/xgv_eval_combined/no_frame_targets.csv`) jest przypięta (pin `combined_no_frame_targets`). Niczego nie podstawiamy. Skutek: populacji łączonej brakuje 147 z 797 strzałów tylko ze StatsBomb (18%), głównie z doliczonego czasu; raport to podaje. **Zalecenie: potwierdzić regułę.** Alternatywa (osobna, opisowa drabina szczebli bez klatki na wszystkich 797) wymagałaby nowego kodu, nowych testów i powtórzenia procedury z 0.2.

**D. 12 goli StatsBomb bez gola SoccerNet.** Flaga `sb_goal_without_sn_goal` zamrożonego parowania oznacza 12 wierszy wtórnych (wcześniejsze notatki podawały 11). Kod idzie za flagą i odmawia, gdy flaga nie zgadza się z wynikiem StatsBomb (sekcja 4). Wiersze te zostają w tabeli, wypadają z analiz, a S1 je liczy. Imienna lista powstaje dopiero w kroku łączenia etykiet (ticket 15, kolumna `label_exclusion`), bo wymaga wyników strzałów. **Zalecenie: potwierdzić regułę z flagą (12).**

**F. Próg bramki kontroli negatywnej sond.** Szkic (runda poprawek 2026-10-11) ustalił twardą bramkę: jeśli kontrola `dist_shuffled` (odległość permutowana po wierszach) da punktowo R² > 0 dla enkodera, każdy odczyt H3 tego enkodera jest nieważny. Dry run na tabeli łączonej z tym progiem (`results/xgv_dryrun_combined_gate0/`) pokazał, że próg odpada przypadkiem: kontrola modelu foldu 0 dała R² = +0,0005 [−0,0025; +0,0027], choć wszystkie cele były tam szumem, a w dry runie pierwotnym ta sama kontrola przeszła. Kontrola na samym szumie (`results/xgv_control_gate_check/`): 50 permutacji odległości z dry runu (seedy 1–50), zamrożona procedura sond, tabela pierwotna bez etykiet, 2 721 wierszy. Bramka R² ≤ 0 odpada w 12% (fold 0), 10% (fold 1) i 12% (DINOv2) przebiegów; największe R² na szumie to 0,0017, mediana około −0,0015, żaden przebieg nie przekroczył 0,005. Przy progu 0 przebieg konfirmacyjny miałby więc około 1 do 8 szans, że sam szum unieważni konfirmacyjne H3 modelu foldu 0, a odczyt różnicy G − DINOv2 zależy jeszcze od bramki DINOv2.
- `0.005` (wpisane, **zalecane**): bramka łapie przeciek, który daje kontroli choćby pół procent wariancji, czyli trzy razy więcej niż największy wynik szumu; przeciek tego rzędu nie przesuwa żadnego odczytu wobec progów 0,25–0,50. Na szumie fałszywe odrzucenie 0 na 150.
- `0.0`: próg szkicu. Najsurowszy, ale z około 10–12% fałszywych unieważnień na enkoder.
- Decyzja nie czyta żadnej etykiety ani żadnego prawdziwego celu sondy: dry run permutuje cele, a kontrola na szumie permutuje je jeszcze raz.

**E. Zatwierdzenie.** Właściciel zatwierdza dokument (z decyzjami A–D i F) i wpisuje zatwierdzenie w tickecie 14. Ticket 15 nie startuje bez tego wpisu.

### 0.2 Procedura po decyzjach

Wartości wpisane w bloku parametrów weszły do dry runów, listy hashy i commita pre-rejestracji.

Zatwierdzenie bez zmian w bloku parametrów:

1. Właściciel wpisuje zatwierdzenie w tickecie 14.
2. `./run.sh python scripts/vendor_prereg_hashes.py --check` przechodzi bez zmian.
3. `./run.sh python scripts/vendor_prereg_hashes.py --record results/xgv_prereg/preregistration_record.json` wystawia rekord pre-rejestracji z czasem commita dokumentu.
4. Ticket 15 (sekcja 18).

Zmiana decyzji A, B lub F (zmiana parametru, kod się nie zmienia), raz, przed rekordem:

1. Zmienić tylko pole `recalibrated_bootstrap`, `frame_exclusions` albo `h3_control_max_r2` w bloku parametrów i zapisać zmianę w dzienniku odstępstw (sekcja 19) jako decyzję właściciela przed łączeniem etykiet.
2. Przenieść `results/xgv_dryrun_primary` i `results/xgv_dryrun_combined` do folderów z dopiskiem `_superseded`; nie usuwać.
3. Powtórzyć oba dry runy, tymi samymi poleceniami co w sekcji 15.
4. `./run.sh python scripts/vendor_prereg_hashes.py`, potem `--check`. Lista hashy liczy dry run jako dowód tylko wtedy, gdy przebieg użył bieżącego kodu, bieżących parametrów i tabeli swojego pinu (`prereg.evidence_mismatch`). Dotyczy to każdego pola bloku poza pinami, także `dry_run_margin` i `permutation_seed`. Zmiana parametru bez nowych dry runów zostawia listę w statusie szkicu, a wtedy `--record` i `vendor_confirm.py` odmawiają.
5. Osobny commit dokumentu i listy hashy, przed jakimkolwiek łączeniem etykiet.
6. Kroki 1–4 z poprzedniej listy.

Zmiana decyzji C albo D zmienia kod analizy. Wymaga nowego ticketu: kod, testy, ta sama procedura z nowymi hashami kodu.

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
| Ponowne pobranie SoccerNet i ocena populacji wtórnej | klatki w czasie StatsBomb, logity, embeddingi, tabela łączona bez etykiet | 12, 13 |
| Test źródła czasu | werdykt „independent” | 04 |
| Reguła δ i symulacja mocy | tabela mocy; reguła wybiera δ = 0,005 | 08 |

Atrybucja: każda tabela i każdy rysunek z danych StatsBomb dostaje podpis „Data: StatsBomb open data” (decyzja D1, `research/xg_vendor_statsbomb_terms.md`).

## 3. Populacje

**Pierwotna.** Strzały sparowane: 2 754 strzały z indeksu v1.2 w 135 meczach (train 1 478, valid 639, test 637). Kotwicą jest kotwica SoccerNet (`anchor_convention = soccernet_anchor`). Wszystkie 2 754 mają klatkę przy offsetach 0 i −400 ms. Rzutów karnych jest 33. Flaga karnego z indeksu zgadza się na wszystkich 2 754 strzałach z flagą StatsBomb i z typem strzału „Penalty”.

**Wtórna** (ticket 13). Należą do niej:

- 797 strzałów tylko ze StatsBomb: każdy strzał StatsBomb ze sparowanego meczu, który nie jest karnym, nie został sparowany ani odzyskany i nie leży w ciągu 10 s od niesparowanego strzału SoccerNet tej samej połowy;
- 50 odzyskanych goli; 7 z nich StatsBomb oznacza jako rzuty karne.

Nie należą:

- 26 możliwych błędnych par (strzały w ciągu 10 s od niesparowanego strzału SoccerNet);
- 2 niesparowane karne;
- gole samobójcze, które w StatsBomb nie są zdarzeniami Shot.

Klatki populacji wtórnej wycina się w czasie StatsBomb plus offset połowy (`anchor_convention = statsbomb_time`). Werdykt „independent” znaczy, że te klatki mogą leżeć kilkaset ms od kontaktu. Raport musi to napisać.

Ticket 13 ocenił klatki 700 z 847 celów wtórnych przy obu offsetach: 50 odzyskanych goli i 650 strzałów tylko ze StatsBomb. Pozostałe 147 (wszystkie tylko ze StatsBomb) nie ma klatki przy żadnym offsecie, bo cel leży po końcu wideo połowy (`frame_status = 0:beyond_half_end;-400:beyond_half_end`).

**Reguła dla wierszy bez klatki (decyzja C).** Wiersz bez ocenionej klatki przy offsecie 0 zostaje w tabeli ze statusem klatki i logitami NaN. Wypada z H1/H2 dla każdego szczebla, także szczebli bez klatki, żeby cała drabina liczyła się na tych samych strzałach. S1 liczy go w kolumnie „Bez klatki”. Lista z przyczynami to `results/xgv_eval_combined/no_frame_targets.csv` (pin `combined_no_frame_targets`).

**Wykluczenia klatek z bloku parametrów (decyzja B).** Wiersz wtórny wymieniony w `frame_exclusions` jest traktowany jak wiersz bez klatki i liczony osobno (S1, kolumna „Klatka wykluczona”). Wartość wpisana wyklucza jeden wiersz z uszkodzoną klatką (sekcja 0.1).

**Łączona** = pierwotna ∪ wtórna, tabela `results/xgv_eval_combined/eval_table_labels_absent.parquet` (pin `combined_label_free_table`, ticket 13). Liczności:

| Populacja | Wiersze | train / valid / test | Z ocenioną klatką | Bez klatki |
|---|---|---|---|---|
| pierwotna | 2 754 | 1 478 / 639 / 637 | 2 754 | 0 |
| odzyskane gole | 50 | 19 / 15 / 16 | 50 | 0 |
| tylko StatsBomb | 797 | 427 / 216 / 154 | 650 | 147 |
| razem | 3 601 | | 3 454 | 147 |

Wiersze pierwotne tabeli łączonej są bajt w bajt tabelą ticketu 11 (pin `primary_label_free_table`; dowód w `results/xgv_eval_combined/complete.json`, `summary.primary_identity`). Macierze embeddingów mają 3 454 wiersze: najpierw macierze ticketu 11, potem 700 wierszy wtórnych; wiersze bez klatki nie mają embeddingu. Karne populacji wtórnej to 7 odzyskanych goli (typ strzału StatsBomb „Penalty”).

**Wykluczenia z analiz:**

| Analiza | Populacja | Wykluczone |
|---|---|---|
| H1, H2 | pierwotna (H2: konfirmacyjnie) i łączona (odporność) | wykluczenia przy etykietach (`label_exclusion`), karne (`is_penalty`), wykluczenia klatek (`frame_exclusions`), wiersze bez ocenionej klatki przy offsecie 0; każdy szczebel na tych samych wierszach |
| H3 | pierwotna | karne (typ strzału StatsBomb „Penalty”, zmienna bez etykiety; na pierwotnej ten sam zbiór co `is_penalty`), wiersze bez embeddingu; dla każdego celu wiersze bez jego wartości |
| H4 | pierwotna, tylko strzały bez gola SoccerNet | karne, wiersze bez ocenionej klatki; w kontraście offsetów także wiersze bez logitu przy −400 ms |

Pominiętej klatki ani brakującego celu niczym nie zastępujemy.

**Mecze i losowania.** Strzały wtórne leżą w tych samych 135 meczach; zbiór meczów tabeli łączonej jest równy zbiorowi tabeli pierwotnej (sprawdzone na tabelach bez etykiet). Losowania bootstrapu zależą tylko od zbioru meczów tabeli (sekcja 7), więc tabela pierwotna i łączona mają te same losowania. Przebieg konfirmacyjny liczy wszystko na tabeli łączonej. Gdyby zbiór meczów tabeli łączonej różnił się od 135 meczów tabeli pierwotnej, odstępstwo trafia do dziennika (sekcja 19), a obowiązują losowania tabeli łączonej; S1 obu dry runów podaje liczbę meczów.

## 4. Etykiety (decyzja D4)

Wiersze pierwotne biorą etykiety z zamrożonej tabeli par (`results/xgv_pairing/pairs.parquet`), w kodzie `xgpoc.vendor.labels.primary_labels`:

- `goal_sn`: gol z indeksu v1.2, czyli etykieta, na której uczył się G;
- `on_target_sn`: etykieta SoccerNet „Shots on target”, która liczy strzały zablokowane jako celne;
- `sb_outcome`: wynik strzału StatsBomb;
- `is_penalty`: flaga karnego z indeksu v1.2;
- `label_exclusion`: puste (każda para wchodzi do analiz ze swoją etykietą SoccerNet).

`is_penalty` nie jest wynikiem strzału, tylko warunkiem znanym przed strzałem: strzał jest rzutem karnym. Leży wśród kolumn etykiet, bo pochodzi z tych samych źródeł w kroku łączenia. Służy wyłącznie do routingu: logity klatki karnych to stała częstość karnych z kalibracji G-combo, a analizy wyjmują karne z H1–H4 i raportują je osobno (S9). Żadna metryka nie używa go jako celu.

Trzynaście par, w których gol SoccerNet i gol StatsBomb się nie zgadzają (6 goli tylko w SoccerNet, 7 tylko w StatsBomb), zachowuje etykietę SoccerNet.

Wiersze wtórne dostają etykiety z tabeli populacji niesparowanych strzałów StatsBomb (`results/xgv_pairing/sb_unpaired.parquet`, plik zamrożonego parowania), w kodzie `xgpoc.vendor.labels.secondary_labels`:

- odzyskany gol: `goal_sn` = 1, bo reguła odzyskiwania przypisała mu gol SoccerNet;
- strzał tylko ze StatsBomb: `goal_sn` = 0, bo żaden gol SoccerNet nie jest mu przypisany;
- gole StatsBomb bez żadnego gola SoccerNet (flaga `sb_goal_without_sn_goal` parowania; w zamrożonym parowaniu 12 wierszy wtórnych, wszystkie z wynikiem StatsBomb „Goal”) zostają w tabeli z `goal_sn` = 0 i kodem `label_exclusion = sb_goal_without_sn_goal`. Każda analiza je pomija, a S1 je liczy (kolumna „Wykluczone przy etykietach”). Kod odmawia, gdy flaga nie zgadza się z wynikiem StatsBomb;
- `on_target_sn` jest pusty, bo definicja SoccerNet istnieje tylko dla strzałów, które SoccerNet oznaczył;
- `is_penalty`: flaga karnego StatsBomb. Odzyskane gole obejmują 7 karnych; strzały tylko ze StatsBomb nie mają karnych z definicji populacji.

`xgpoc.vendor.labels.table_labels` składa obie reguły w kolejności wierszy tabeli. Walidator tabeli przyjmuje tylko znane kody wykluczenia (`evaltable.LABEL_EXCLUSIONS`).

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
  - Wariant rekalibrowany (szczeble pojedyncze i oba stosy, więc też H2), według parametru `recalibrated_bootstrap` (decyzja A, sekcja 7.1):
    - `refit` (wpisane): w każdym losowaniu rekalibracja i stosy są dopasowywane od nowa na wylosowanych meczach (`xgpoc.vendor.refit`). Foldy zostają te same, kopie meczu leżą w jego foldzie, wagą strzału jest liczba wylosowań jego meczu. Model to ten sam cel co regresja logistyczna sklearn z C = 10⁶, rozwiązany do zbieżności metodą Newtona (tak samo liczy się estymata punktowa, `ladder.crossfit`). Każdą metrykę liczy się z predykcji out-of-fold dopasowanych w danym losowaniu.
    - `fixed`: predykcje out-of-fold z próby obserwowanej zostają stałe, losowanie tylko je przeważa (metoda zamrożonej symulacji mocy, `ladder.h2_draws`).
    - Estymata punktowa nie zależy od wyboru.
  - Wariant surowy: nic nie jest dopasowywane, zapisane prawdopodobieństwa dostają wagi meczów.
  - Przedziały są percentylowe.
- **TOST** przy marginesie d, na poziomie 5% na stronę:
  - równoważność, gdy co najwyżej 5% losowań różnicy leży na lub poniżej −d i co najwyżej 5% na lub powyżej d, czyli gdy przedział 90% mieści się w (−d, d);
  - p TOST = max(odsetek losowań ≤ −d, odsetek ≥ d).
- **Test przewagi**, jednostronny, dla log-lossu (A − B): p = odsetek losowań ≥ 0, przewaga przy p ≤ 0,05. Dla AUC: odsetek ≤ 0.
- **Krzywe równoważności**: p TOST na siatce marginesów od 0 do 0,01 co 0,0001, dla każdej pary szczebli. Najmniejszy margines to pierwszy punkt siatki z równoważnością.
- **Zatrzymanie.** Jeśli któreś losowanie nie ma gola albo ma same gole, kod przerywa analizę. Raport opisuje wtedy przerwę i nie podstawia innej metody.

### 7.1 Decyzja właściciela A: bootstrap z dopasowaniem w losowaniu czy ze stałymi predykcjami

Ticket 08 liczył moc bootstrapem ze stałymi predykcjami. Przegląd kodu (2026-10-11) kazał dopasowywać rekalibrację i stosy w każdym losowaniu, bo stałe predykcje pomijają zmienność samego dopasowania. Kontrola na bazie deweloperskiej ticketu 08 (`results/xgv_refit_check/refit_check.json`; 80 symulowanych badań na scenariusz, 500 losowań; liczby deweloperskie) pokazuje, że żaden wariant nie jest dokładny:

| SD różnicy H2 | Prawda deweloperska (zysk z klatki około −0,0012) | Prawda zerowa (klatka nic nie dodaje) |
|---|---|---|
| rzeczywisty rozrzut próbkowania (1 000 badań ticketu 08; w tych 80 badaniach) | 0,00119 (0,00099) | 0,00042 (0,00044) |
| `fixed`: średnie SD bootstrapu | 0,00103 (−13%) | 0,00036 (−14%) |
| `refit`: średnie SD bootstrapu | 0,00120 (+1%) | 0,00072 (+71%) |
| `refit`: średnia losowań minus estymata punktowa | +0,00022 | +0,00025 |

Procenty liczone są względem rozrzutu z 1 000 badań. Przesunięcie losowań `refit` w górę bierze się stąd, że zduplikowane mecze zmniejszają efektywną liczbę meczów i zwiększają koszt dopasowania drugiego współczynnika stosu.

Skutki dla rozstrzygnięć H2:

| | `refit` (wpisane) | `fixed` |
|---|---|---|
| TOST przy δ = 0,005 (twierdzenie konfirmacyjne) | konserwatywny przy prawdzie bliskiej zeru; bez praktycznej straty mocy, bo górny koniec przedziału 90% leży daleko od 0,005 (dry run pierwotny: +0,0019) | lekko liberalny w obu scenariuszach |
| test przewagi | konserwatywny przez przesunięcie i szerokość; mały rzeczywisty zysk łatwiej przeoczyć | bliższy nominalnemu, lekko liberalny |
| zgodność z tabelą mocy (sekcja 10) | nie; rzeczywista moc przy prawdzie deweloperskiej może być nieco niższa niż 0,849 | tak, to metoda symulacji mocy |

**Zalecenie: `refit`.** Twierdzenie konfirmacyjne to równoważność. Za wąski przedział daje fałszywą równoważność, a za szeroki tylko utrudnia jej wykazanie; przy δ = 0,005 tej straty praktycznie nie widać. Przy prawdzie deweloperskiej `refit` odtwarza rozrzut próbkowania prawie dokładnie. Cena to konserwatywny test przewagi; raport podaje ją wprost.

Wybór obejmuje każdy szczebel wariantu rekalibrowanego (H1, H2 i klatka + dostawca, obie populacje). Wariant surowy niczego nie dopasowuje i od wyboru nie zależy. Kod: `ladder.analyse(..., bootstrap=...)`, parametr z rekordu pre-rejestracji. Raport podaje własności wybranego wariantu z tej tabeli (sekcja 19) i niczego nie poprawia po fakcie.

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

Reguła z ticketu 08 (`research/xg_vendor_delta_rule.md`, sha256 766ee11b…, ustalona przed symulacją) brzmi: δ to najmniejszy margines z {0,002; 0,003; 0,005} z mocą decyzyjną co najmniej 0,80. Scenariusz decyzyjny to populacja pierwotna i prawda deweloperska, a moc decyzyjna to minimum mocy dla klatki archiwalnej i dla klatki słabszej o 0,002. Reguła wybrała **δ = 0,005**. Jej krok 6 wymaga, żeby właściciel zatwierdził δ, zanim δ trafi do tego dokumentu. Właściciel zatwierdził δ = 0,005 wieczorem 2026-10-11 (opcja A poniżej).

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

Opcje przedstawione właścicielowi przed decyzją (wybrał A):

- **A. Reguła bez zmian: δ = 0,005.**
  - Plusy: reguła została ustalona przed symulacją i nie zmienia się po obejrzeniu tabeli mocy. Moc 0,85 przy zysku dwa razy większym od deweloperskiego.
  - Minusy: „równoważne przy 0,005” to słabe twierdzenie. 0,005 log-lossu to około jedna czwarta różnicy klatka − dostawca z prototypu (+0,0235 na teście). Równoważność przy 0,005 nie wyklucza zysku rzędu dobrych cech pozycyjnych (freeze frame − zdarzenia w prototypie: od −0,007 do +0,006).
- **B. Poprawiony scenariusz decyzyjny.** Nowa notatka reguły (z nowym hashem; obie wersje idą do raportu) zastępuje prawdę deweloperską prawdą ściągniętą do zysku obserwowanego (−0,0007) albo prawdą zerową.
  - Przy prawdzie zerowej reguła daje δ = 0,002 (moc 0,949 i 0,965).
  - Dla prawdy ściągniętej do −0,0007 mocy nie policzono. Leży między wierszami deweloperskimi i zerowymi (przy 0,002: między 0,28 a 0,95; przy 0,003: między 0,51 a 0,99) i wymaga jednego przebiegu symulacji (około 14 min CPU) przed wyborem.
  - Plusy: mocniejsze twierdzenie przy prawdopodobnym zysku bliskim zera.
  - Minusy: zmiana reguły po obejrzeniu tabeli mocy. To jawne odstępstwo od D5, a raport musi je opisać. Gdy prawdziwy zysk wynosi około −0,001, δ = 0,002 łatwo daje wynik nierozstrzygnięty.

Niezależnie od opcji raport podaje najmniejszy margines równoważności i całą krzywą równoważności H2, więc czytelnik może przyłożyć własny próg.

**Decyzja właściciela (2026-10-11, wieczór):** δ = 0,005, opcja A, czyli reguła z ticketu 08 bez zmian. Wartość stoi w polu `margin` bloku parametrów (sekcja 16). Moc z tabeli powyżej policzono bootstrapem ze stałymi predykcjami. Przy wpisanej wartości decyzji A (`refit`, sekcja 7.1) bootstrap analizy dopasowuje modele w każdym losowaniu, więc rzeczywista moc przy prawdzie deweloperskiej może być nieco niższa niż 0,849. Dopasowanie krzyżowe liczy się teraz do zbieżności (wcześniej sklearn z domyślną tolerancją, około 10⁻³ w prawdopodobieństwie od optimum), więc powtórka symulacji dałaby liczby różne w ostatnich miejscach.

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

Kontrola negatywna `dist_shuffled`: odległość permutowana po wierszach sondy (`default_rng(0)`). Musi dać R² ≤ `h3_control_max_r2` (punktowo) dla każdego enkodera; wartość wpisana 0,005, decyzja F (sekcja 0.1). Jeśli dla któregoś enkodera nie da, H3 dla tego enkodera jest nieważna i raport to podaje. To twarda bramka w kodzie (`probes.control_gate`): `vendor_probes.py` zapisuje wyniki z bramką oznaczoną jako niezaliczona i kończy się kodem 3, a szablony S6 i S7 wpisują dla każdego celu tego enkodera „nieważny: kontrola negatywna nie przechodzi” zamiast odczytu progu; tak samo odczyt różnicy, gdy bramki nie przechodzi enkoder albo DINOv2.

**Enkodery** (embeddingi wszystkich 2 754 strzałów pierwotnych, ticket 11):

| Enkoder | Rola |
|---|---|
| G-combo, model foldu 0, seed 0 (`emb_g_fold0`, 5 120 wymiarów: token CLS i wyjścia zapytań attention pooling) | rozstrzyga |
| G-combo, model foldu 1, seed 0 (`emb_g_fold1`) | wrażliwość; nie zmienia werdyktu; rozbieżności są raportowane |
| zamrożony DINOv2-L (`emb_dinov2`, 2 048 wymiarów: CLS i średnia tokenów łatek), te same wagi co szkielet G | odniesienie |

Mecze ewaluacyjne leżą poza treningiem każdego modelu foldu. Embedding pochodzi więc z jednego modelu dla wszystkich strzałów. Prototyp mieszał modele foldów, czego specyfikacja teraz zabrania.

**Progi i rozstrzygnięcie.** Decyzja właściciela (2026-10-11, wieczór): część konfirmacyjna H3 to dwa cele, `dist` i `def_cone_ge1`, z poprawką Bonferroniego (każdy na poziomie 2,5% jednostronnie, sekcja 1). Pozostałe osiem celów jest opisowych: `def_cone_ge2` i `head` z progiem AUC 0,70 oraz `angle`, `ff_def_cone`, `gk_depth`, `gk_off_line_abs`, `gk_dist_goal` i `ff_opp_3m` z progiem R² 0,25. Progi konfirmacyjne pochodzą z planu sprzed prototypu (R² odległości > 0,5, AUC obrońców w stożku > 0,7). Progi opisowe ustalono w tym dokumencie, przy znanych liczbach deweloperskich prototypu (sekcja 19).

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

Kod `xgpoc.vendor.shells` wypełnia szablony S1–S9 z wyników skryptów analizy. Wiersze i kolumny są stałe. Komórka bez wartości dostaje „—”, a populacja pominięta w danym przebiegu dostaje „pominięta”. Tabele S2–S4 powtarzają się dla populacji pierwotnej i łączonej. Pełne tabele (wszystkie pary kontrastów, krzywe równoważności, wszystkie współczynniki H4, wszystkie podzbiory sond) zostają w plikach CSV przebiegu. Test `tests/test_vendor_prereg.py` sprawdza, że puste szablony poniżej są dokładnie tym, co renderuje kod.

#### S1. Populacje analizy H1/H2: liczności i wykluczenia

Każdy wiersz liczy się raz, pod pierwszą przyczyną z kolejności kolumn: wykluczenie przy etykietach, karny, klatka wykluczona (`frame_exclusions`), brak ocenionej klatki.

| Populacja | Wybrane | Wykluczone przy etykietach | Karne | Klatka wykluczona | Bez klatki | Analizowane | Gole | Mecze |
|---|---|---|---|---|---|---|---|---|
| pierwotna |  |  |  |  |  |  |  |  |
| łączona |  |  |  |  |  |  |  |  |

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
| dist_shuffled | R2 |  |  |  |  | R2 ≤ 0.005 | kontrola negatywna |  |  |
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

#### S9. Rzuty karne, raportowane osobno (poza H1–H4; prawdopodobieństwa surowe, bez przedziałów)

Karne nie wchodzą do H1–H4 (specyfikacja: karne raportowane osobno). S9 podaje ich liczbę, gole SoccerNet, skuteczność oraz średnie prawdopodobieństwo i log-loss surowych predykcji dostawcy, klatki (stała częstość karnych z routingu) i szczebla freeze frame, bez rekalibracji i bez przedziałów (`ladder.penalty_summary`).

| Populacja | Karne | Gole (SN) | Skuteczność | Śr. xG dostawcy | Log-loss dostawcy | Śr. p klatki | Log-loss klatki | Śr. p pozycji | Log-loss pozycji |
|---|---|---|---|---|---|---|---|---|---|
| pierwotna |  |  |  |  |  |  |  |  |  |
| łączona |  |  |  |  |  |  |  |  |  |

## 15. Dry run

Dry run (`scripts/vendor_dryrun.py`) przechodzi całą ścieżkę konfirmacyjną na etykietach bez związku z predykcjami:

1. Tabela bez etykiet, sprawdzona z pinem, zostaje zapisana jeszcze raz z `table_kind = dryrun`. Kod sprawdza, że nic innego się nie zmienia.
2. Etykiety wszystkich wierszy, zbudowane regułami przebiegu konfirmacyjnego (`labels.table_labels`), są permutowane w obrębie meczów (`xgpoc.vendor.dryrun.permute_labels`); kody wykluczenia zostają przy swoich wierszach:
   - warstwy (mecz, karny);
   - jedna permutacja przenosi razem gol i wynik StatsBomb;
   - druga przenosi celność wśród wierszy pierwotnych;
   - `default_rng([permutation_seed, 1])`.
   Potem etykiety przechodzą przez logowany krok łączenia. Żadna niepermutowana etykieta nie jest łączona z predykcją ani wypisywana.
3. Cele sond: wiersze tabeli strzałów StatsBomb dla zdarzeń tabeli (lokalizacja, część ciała, freeze frame) są permutowane razem w warstwach (mecz, karny), `default_rng([permutation_seed, 2])`. Sondy liczą wszystkie cele z permutowanych wierszy.
4. Rekord roboczy pre-rejestracji (`draft = true`) bieżącego dokumentu podaje parametry z bloku parametrów (δ = 0,005, decyzje A i B z wartościami wpisanymi). Rekord roboczy nie dopuszcza łączenia etykiet z tabelą konfirmacyjną.
5. Trzy skrypty analizy biegną z tymi samymi argumentami co w przebiegu konfirmacyjnym (`confirm.analysis_commands`): każdy czyta z rekordu wszystkie swoje parametry, z linii poleceń nie przyjmuje żadnego:
   - `vendor_ladder.py`: δ, liczba losowań, ziarna, bootstrap szczebli rekalibrowanych, wykluczenia klatek;
   - `vendor_probes.py`: liczba losowań, ziarno bootstrapu, progi H3, cele konfirmacyjne i próg bramki kontroli;
   - `vendor_leakage.py`: liczba losowań, ziarno, logity offsetów i hash reguły, z którym przyjmuje rekord werdyktu.
   Niezaliczona bramka kontroli sond (kod 3) jest zapisywana, nie przerywa przebiegu.
6. Kod wypełnia szablony S1–S9 (`shells.md`) i zapisuje rekord ukończenia z hashem dokumentu, kodem (sha256 każdego pliku domknięcia kodu), parametrami i hashem tabeli. Lista hashy liczy przebieg jako dowód tylko wtedy, gdy te trzy rzeczy zgadzają się z bieżącymi (sekcja 17).

Permutacja w obrębie meczów zachowuje różnice między meczami (częstość goli, średnią odległość). Liczby z dry run nie muszą więc leżeć przy zerze. Nie są wynikami i nie wolno ich cytować jako wyników.

Czego permutacja nie ukrywa:

- liczności w S1 (wiersze, karne, gole, mecze), znane z indeksu;
- wiersze H4, które nie czytają etykiety: średni kontrast offsetów i ramię reszty czasowej. Różnica logitów, kontrast i reszta czasowa są kolumnami tabeli bez etykiet. Etykieta wchodzi tylko przez wybór strzałów bez gola, a ten permutacja zmienia w niewielkim stopniu. Te wiersze dry run pokazuje więc w przybliżeniu w wartościach przebiegu konfirmacyjnego. H4 jest opisowa i żadna decyzja nie zależy od tych wierszy.

Przebiegi, od najstarszego. Obowiązują dwa ostatnie; ich rekordy ukończenia są dowodami w sekcji 17. Starsze foldery zostają jako historia.

- `results/xgv_dryrun_primary_bdc3af3/`: tabela pierwotna, pierwsza wersja kodu (commit bdc3af3; stałe predykcje w losowaniach). Odczyt H2 „równoważne przy δ; stos istotnie gorszy”.
- `results/xgv_dryrun_primary_16a3c9d/`: tabela pierwotna po rundzie poprawek (dopasowanie w każdym losowaniu). H2 +0,0005 [−0,0003; +0,0022], SD bootstrapu 0,00064 wobec 0,00022 przy stałych predykcjach w pierwszej wersji.
- `results/xgv_dryrun_primary_gate0/` i `results/xgv_dryrun_combined_gate0/` (2026-10-12): kod końcowy z bramką kontroli R² ≤ 0. Na tabeli łączonej bramka modelu foldu 0 odpadła na szumie (R² kontroli +0,0005), `vendor_probes.py` zakończył się kodem 3, a szablony S6 i S7 wpisały „nieważny” zgodnie z regułą. Z tego wynikła decyzja F (sekcja 0.1); oba przebiegi zastąpiły następne.
- **`results/xgv_dryrun_primary/`** (2026-10-12, 80 s CPU): tabela pierwotna, kod i parametry końcowe (5 000 losowań, `refit`, próg bramki 0,005).
  - S1: 2 754 wybrane, 33 karne, 2 721 analizowanych, 374 gole (permutacja w warstwach samej tabeli pierwotnej nie zmienia ich liczby), 135 meczów; populacja łączona oznaczona jako pominięta.
  - H2 na etykietach permutowanych: Δ = +0,0005 [−0,0003; +0,0022], 90% [−0,0001; +0,0019], SD bootstrapu 0,00064, p TOST 0,0002, p przewagi 0,93, odczyt „równoważne przy δ: klatka nie dodaje więcej niż δ”.
  - H3: 2 721 strzałów (cele bramkarza 2 719); bramka kontroli przechodzi dla wszystkich trzech enkoderów (największe R² kontroli −0,0006).
  - H4: 2 347 strzałów bez gola w 135 meczach, 2 283 z resztą w ±200 ms; ramię reszty czasowej uruchomione z werdyktem „independent” przyjętym pod hashem reguły 42dcf710….
  - S9: 33 karne, 22 gole.
- **`results/xgv_dryrun_combined/`** (2026-10-12, 130 s CPU): tabela łączona, te same parametry.
  - S1, populacja łączona: 3 601 wybranych, 12 wykluczonych przy etykietach (kody wykluczenia zostają przy swoich wierszach), 40 karnych, 1 klatka wykluczona (decyzja B), 147 bez klatki (decyzja C), 3 401 analizowanych, 135 meczów. Populacja pierwotna: 2 721 analizowanych.
  - Liczba goli w obu populacjach (318 pierwotna, 398 łączona) nie jest prawdziwa: w tej tabeli warstwy permutacji obejmują wiersze pierwotne i wtórne razem, więc gole przechodzą między populacjami.
  - H2 na etykietach permutowanych: pierwotna Δ = −0,0005 [−0,0023; +0,0015], łączona −0,0002 [−0,0015; +0,0013]; oba odczyty „równoważne przy δ: klatka nie dodaje więcej niż δ”. Klatka + dostawca: tak samo.
  - H3: sondy liczą tylko wiersze pierwotne (847 wierszy wtórnych pominiętych); bramka kontroli przechodzi (największe R² kontroli: fold 0 +0,0005, fold 1 0,0000, DINOv2 −0,0009, próg 0,005).
  - H4: 2 403 strzały bez gola (inny wybór niż w dry runie pierwotnym, z tego samego powodu co gole), ramię reszty czasowej uruchomione.
  - S9: 40 karnych (33 pierwotne i 7 odzyskanych goli).
- Na celach permutowanych w obrębie meczów sonda przewiduje między meczami odwrotnie (np. AUC `def_cone_ge1` dla foldu 0: 0,47 [0,45; 0,49] w dry runie łączonym). To znane obciążenie walidacji grupowanej, gdy cel niesie tylko różnice między meczami, a nie błąd kodu; liczby dry runu nie są wynikami.

Przygotowanie dry run ujawniło dwa błędy ścieżki konfirmacyjnej, naprawione przed zamrożeniem kodu. Pierwszy: `vendor_probes.py` szukał względnych plików embeddingów w katalogu roboczym, a nie w katalogu tabeli, i przerwałby przebieg konfirmacyjny. Drugi: rekordy sond i przecieku oznaczały każdą tabelę inną niż deweloperska jako „confirmatory run”.

## 16. Parametry

Blok poniżej jest jedynym źródłem parametrów przebiegu konfirmacyjnego. `scripts/vendor_prereg_hashes.py` go czyta i sprawdza każdy pin z plikiem. Rekord pre-rejestracji przenosi `margin`, `n_boot`, `fold_seed`, `boot_seed`, `offset_logits`, `h3_thresholds`, `h3_confirmatory_targets`, `h3_control_max_r2`, `recalibrated_bootstrap`, `frame_exclusions` i hash pinu `timing_rule`. `vendor_ladder.py`, `vendor_probes.py` i `vendor_leakage.py` czytają je z rekordu, wymagają co najmniej 5 000 losowań i dla tabeli dry run albo konfirmacyjnej nie przyjmują ich z linii poleceń ani opcji deweloperskich. `null` to pozycja do uzupełnienia (żadna nie zostaje). Hash pinu `timing_rule` jest tym, z którym `vendor_leakage.py` przyjmuje werdykt.

Decyzje właściciela z sekcji 0 to pola `recalibrated_bootstrap` (A), `frame_exclusions` (B) i `h3_control_max_r2` (F), z wartościami zalecanymi. `frame_exclusions` przyjmuje tylko kody z `evaltable.FRAME_EXCLUSIONS` i tylko klucze wierszy wtórnych z ocenioną klatką przypiętej tabeli łączonej. Piny `combined_no_frame_targets` i `combined_frame_clip_audit` przypinają listę 147 celów bez klatki i wynik audytu klatek i klipów, na których opierają się decyzje B i C.

<!-- prereg-parameters:begin -->
```json
{
  "margin": 0.005,
  "dry_run_margin": 0.005,
  "n_boot": 5000,
  "fold_seed": 0,
  "boot_seed": 0,
  "permutation_seed": 0,
  "offset_logits": "calibrated",
  "h3_thresholds": {"dist": 0.5, "angle": 0.25, "ff_def_cone": 0.25, "def_cone_ge1": 0.7, "def_cone_ge2": 0.7, "gk_depth": 0.25,
                    "gk_off_line_abs": 0.25, "gk_dist_goal": 0.25, "ff_opp_3m": 0.25, "head": 0.7},
  "h3_confirmatory_targets": ["dist", "def_cone_ge1"],
  "h3_control_max_r2": 0.005,
  "recalibrated_bootstrap": "refit",
  "frame_exclusions": {"1357b69b-baa6-406f-a238-12e493594942": "decode_corrupted_frame"},
  "pins": {
    "primary_label_free_table": {"path": "results/xgv_eval_primary/eval_table_labels_absent.parquet",
                                 "sha256": "829aa7edb59e99423cdf379c8cb656c59536a42bea38f4f64bec93cc9fd3c741"},
    "primary_label_free_sidecar": {"path": "results/xgv_eval_primary/eval_table_labels_absent.json",
                                   "sha256": "628b8ba7b93528a81eb9c26ad5e67485a4a4090c977e10a3dd3bb902bda438ed"},
    "combined_label_free_table": {"path": "results/xgv_eval_combined/eval_table_labels_absent.parquet",
                                  "sha256": "f1e7c71f59a57f610bcc8fdf7caf385f689d994ab85606df0805cc842ad50ab3"},
    "combined_label_free_sidecar": {"path": "results/xgv_eval_combined/eval_table_labels_absent.json",
                                    "sha256": "efe0dce7741621df4052cb93b9e2ce010e11fbaa7fd8c563a7ca1ea313795ad2"},
    "combined_no_frame_targets": {"path": "results/xgv_eval_combined/no_frame_targets.csv",
                                  "sha256": "b5d0378a6285ed94e0a5b25fe8e6304d6333ce93215c6eb5b3b21c245a34e9ad"},
    "combined_frame_clip_audit": {"path": "results/xgv_eval_combined/frame_clip_disagreements.csv",
                                  "sha256": "127cba0f7b99b29392bc047a7d3733275df2407785ee7d631fc01eb69243e4a7"},
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

Embeddingi (`emb_g_fold0.npy`, `emb_g_fold1.npy`, `emb_dinov2.npy`) przypina plik boczny tabeli: piny `primary_label_free_sidecar` i `combined_label_free_sidecar`. `vendor_probes.py` sprawdza sha256 każdego pliku z plikiem bocznym.

## 17. Hashe

Blok poniżej generuje `scripts/vendor_prereg_hashes.py`; nie edytuje się go ręcznie. Lista obejmuje:

- piny z bloku parametrów, każdy sprawdzony z plikiem;
- kod analizy: skrypty `vendor_confirm.py` (weryfikacja hashy i łączenie etykiet obiema regułami), `vendor_ladder.py`, `vendor_probes.py`, `vendor_leakage.py`, `vendor_dryrun.py`, `vendor_prereg_hashes.py` i każdy moduł pakietu `xgpoc`, który importują, przechodnio;
- rekordy ukończenia dry runów. Rekord liczy się jako dowód tylko wtedy, gdy dry run użył dokładnie bieżącego domknięcia kodu, bieżących parametrów (bez pinów) i tabeli swojego pinu (`prereg.evidence_mismatch`). Inaczej lista ma status szkicu z pozycją „evidence … is stale”, a `--record` i `vendor_confirm.py` odmawiają.

Kod kroku łączenia etykiet obu populacji leży na liście skryptów wejściowych (`prereg.ENTRY_SCRIPTS`: `vendor_confirm.py`, przez nią `xgpoc.vendor.labels` i `xgpoc.vendor.confirm`). Zmiana decyzji A, B lub F zmienia blok parametrów, więc ten blok trzeba wygenerować ponownie według sekcji 0.2.

<!-- prereg-hashes:begin -->
Lista hashy: `research/xg_vendor_prereg_hashes.json`, sha256 `0db46ea942629e4fc57f485bfc193d88973e41b24f9567c027c0b45e824f558e`, status: **KOMPLETNA**.

| Wejście | Plik | sha256 |
|---|---|---|
| combined_frame_clip_audit | results/xgv_eval_combined/frame_clip_disagreements.csv | `127cba0f7b99b29392bc047a7d3733275df2407785ee7d631fc01eb69243e4a7` |
| combined_label_free_sidecar | results/xgv_eval_combined/eval_table_labels_absent.json | `efe0dce7741621df4052cb93b9e2ce010e11fbaa7fd8c563a7ca1ea313795ad2` |
| combined_label_free_table | results/xgv_eval_combined/eval_table_labels_absent.parquet | `f1e7c71f59a57f610bcc8fdf7caf385f689d994ab85606df0805cc842ad50ab3` |
| combined_no_frame_targets | results/xgv_eval_combined/no_frame_targets.csv | `b5d0378a6285ed94e0a5b25fe8e6304d6333ce93215c6eb5b3b21c245a34e9ad` |
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
| dowód: dry_run_primary | results/xgv_dryrun_primary/complete.json | `a5752e242098b07d16d4addb743987d31ac73ce18d9395d2de4e177d5125a6ac` |
| dowód: dry_run_combined | results/xgv_dryrun_combined/complete.json | `885b00bc295778b1793342bffc9069bff88786bcbb94d0cab183b76f51b0ed9b` |

Kod analizy (32 plików):

| Plik | sha256 |
|---|---|
| scripts/vendor_confirm.py | `5f63c56ee81ed2e339974db0b51d46c91bf74700a8c23eb9aa48b8b071997271` |
| scripts/vendor_dryrun.py | `774d617247bc2125350314cb836b7a26ebcbba2867f280361bea026c07003fe0` |
| scripts/vendor_ladder.py | `89d5fd33b82b91537e795561a3e432953b9cca881bbfbdd51e99e3de912235ab` |
| scripts/vendor_leakage.py | `e1e9fc9213d04ed98470a09e496b11ecebe6822709d329e55cb7a2232117a31e` |
| scripts/vendor_prereg_hashes.py | `804e4b88c138d15ff1e35cfb169aba8f59ee0e1a05daeb8e4e3064c182e80d60` |
| scripts/vendor_probes.py | `07de430c650c127010395de6d64149e4b36a4dbd03370ff3ac649b618eb242b0` |
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
| xgpoc/vendor/confirm.py | `411466f6391f55d1a83661565ec0479c8e439521960461b47baa96fe0e0ef86e` |
| xgpoc/vendor/draws.py | `fc88a16636cd4e558b5264081a7a6d245d6a4b981f0c1cad7b33ba9649bfe62c` |
| xgpoc/vendor/dryrun.py | `6665a9bcd661da3ed90069736eeefb6e02254c543c6d1137e39800cf057d38d4` |
| xgpoc/vendor/evaltable.py | `e6220255f3dc4723a40f4f37ba1044aa33cc016ca30d8746aca3c29d5cbc31f7` |
| xgpoc/vendor/labels.py | `b719774f7913a9c19aa24eb2b57f1501210bb395b2c6151757348e7395ccda0b` |
| xgpoc/vendor/ladder.py | `a2de6fff3ade4b7741896aaf6f78df1071d3e67e1e5db983ea434d0bbd5428b1` |
| xgpoc/vendor/ladder_dev.py | `7f8a7404e1bfe9d2f1f4687e49383cd096ddaf318183057d6cfda0b17fb6d525` |
| xgpoc/vendor/leakage.py | `f65f2bc368dbe4761550a81733c2a354ea4d64555cdcefa81ff421c78afec134` |
| xgpoc/vendor/prereg.py | `0f170950b5b8f107705038a9d130e4a432d8dbcd6b2bbd45da0d00bd3cea3fd0` |
| xgpoc/vendor/probe_targets.py | `e9ec5bb125d9522885c924bd3acfdd76008790c00be97643e2b2d24fdc1e6293` |
| xgpoc/vendor/probes.py | `90b61163f06ea5218f2c1d3bf3d62142d82d0ba77a938637c8945be75a050565` |
| xgpoc/vendor/refit.py | `1adeb13c9ad1c51e21bca3f9c211d0852c0e4723012eb5f85066bf588a6e6a02` |
| xgpoc/vendor/shells.py | `991573ed8e3c36206916b7458c29baa6e5348f2609fc13972fa33ac01865e198` |
| xgpoc/vendor/timing_verdict.py | `ba50b1aea4c65ccb137f3c81ed06d301b260a7e48c7ab78c40a83b2d53fc3b02` |
<!-- prereg-hashes:end -->

## 18. Przebieg konfirmacyjny (ticket 15)

Kolejność, bez innych kroków:

1. Zatwierdzenie właściciela jest wpisane w tickecie 14, a rekord pre-rejestracji wystawiony według sekcji 0.2. `./run.sh python scripts/vendor_prereg_hashes.py --check` musi przejść bez zmian.
2. `./run.sh python scripts/vendor_confirm.py --record results/xgv_prereg/preregistration_record.json --table-dir <katalog tabeli łączonej> --out results/xgv_confirmatory`. Skrypt (`xgpoc.vendor.confirm`), zanim przeczyta jakąkolwiek etykietę:
   - sprawdza, że rekord nie jest roboczy i wskazuje commit;
   - przelicza listę hashy i porównuje ją bajt w bajt z zatwierdzoną: każdy pin (tabela łączona bez etykiet, podział, predykcje komparatora, werdykt i reguła czasu, parowanie i reszta pinów) z plikiem, każdy plik kodu analizy i rekordy dry runów;
   - sprawdza, że rekord zgadza się z dokumentem (hash dokumentu, przypięta tabela, δ, losowania, ziarna, logity offsetów, progi i cele H3, próg bramki kontroli, hash reguły czasu, bootstrap szczebli rekalibrowanych, wykluczenia klatek), że dokument przy zapisanym commicie jest plikiem roboczym, a dokument, lista hashy i kod są zatwierdzone i niezmienione;
   - sprawdza, że łączona tabela jest przypiętą, a pliki etykiet (`pairs.parquet`, `sb_unpaired.parquet`) to pliki przypiętego parowania.
   Przy jakiejkolwiek niezgodności odmawia i nie łączy etykiet. Potem łączy etykiety raz (`labels.table_labels`, `evaltable.join_labels` z rekordem), uruchamia `vendor_ladder.py`, `vendor_probes.py` i `vendor_leakage.py` z parametrami z rekordu i wypełnia szablony S1–S9 (`shells.md`).
3. Raport.

Każdy krok uruchamia się raz. Powtórka po błędzie technicznym (przerwany proces, brak dysku) jest dozwolona tylko z tymi samymi wejściami, a dziennik ją odnotowuje.

## 19. Odstępstwa i ograniczenia

**Odstępstwa.**

- Przed zatwierdzeniem: zmianę dokumentu wprowadza nowa wersja, a hash ją przypina.
- Po zatwierdzeniu: każda zmiana analizy jest oznaczona jako post hoc i raportowana obok analizy z pre-rejestracji. Werdykty H2 i H3 zostają werdyktami reguł tego dokumentu.
- Jeśli kroku nie da się wykonać (brak plików, błąd kodu, losowanie bez goli), agent zatrzymuje się i opisuje problem w tickecie. Nie dobiera zastępczej reguły.

Dziennik odstępstw: brak. Zmiana decyzji A, B lub F przed rekordem (sekcja 0.2) trafia tutaj z datą i nie jest odstępstwem od pre-rejestracji, bo zapada przed łączeniem etykiet.

**Ujawnienia dotyczące samej pre-rejestracji.**

- Uprzedniości reguł δ (`research/xg_vendor_delta_rule.md`) i źródła czasu (`research/xg_vendor_timing_rule.md`) nie da się sprawdzić w gicie co do minuty. Obie reguły napisano przed liczbami, które rozstrzygały, ale w tej samej sesji roboczej, i trafiły do commitów razem z wynikami (tickety 08 i 04). Ich hashe przypina skrypt (`power.RULE_SHA256`, `timing_source`), który odmawia innej wersji, ale kolejność „reguła, potem liczby” w obrębie sesji poświadcza tylko opis w ticketach. Właściwa pre-rejestracja dostaje własny commit przed jakimkolwiek łączeniem etykiet (sekcja 0, krok 6), więc dla H1–H4 ta luka nie występuje.
- Próg bramki kontroli (decyzja F) zmieniono po dry runie na tabeli łączonej, w którym bramka R² ≤ 0 odpadła na celach będących szumem. Zmiana opiera się wyłącznie na zachowaniu kontroli na szumie, nie na żadnym prawdziwym celu sondy ani etykiecie.
- Decyzje B i C zapadły po obejrzeniu tabeli bez etykiet: statusów klatek, audytu klatek i klipów oraz jednej klatki z offsetu 0, która strzału nie pokazuje. Żadna etykieta ani wynik strzału nie zostały przy tym odczytane. Wiersz z decyzji B wskazał audyt (jedyna niezgodność klatki i klipu na 847 celach), a nie wartość predykcji.
- Dry runy uruchomiono na kodzie i parametrach z commita pre-rejestracji, przed commitem; rekordy ukończenia podają więc stan drzewa „zmieniony”, a o tożsamości kodu świadczą hashe plików w rekordzie, porównane z listą hashy.
- Kod rekordów ukończenia (`complete.json`) powtarza się w skryptach ticketów 01–13, a funkcja `file_sha256` z `xgpoc.vendor.statsbomb_fetch` dubluje `xgpoc.bundle.sha256_file`. Celowo tego nie refaktoryzujemy: każdy z tych plików jest plikiem źródłowym, którego hash zapisały gotowe wyniki (parowanie, test czasu, plan podziału, komparator, retrening G, tabela pierwotna, wersja danych StatsBomb), a plan podziału i trwające pobieranie SoccerNet porównują te hashe przy każdym użyciu. Refaktoryzacja zmieniłaby hashe i unieważniła te sprawdzenia.

**Ograniczenia, które raport musi wymienić:**

- Progi opisowe H3 (0,25 dla R², 0,70 dla AUC) ustalono przy znanych liczbach deweloperskich prototypu i ticketu 09. Progi konfirmacyjne pochodzą z planu sprzed prototypu.
- H1 i H2: własności bootstrapu wybranego w decyzji A (sekcja 7.1). Przy `refit`: SD zgodne z rozrzutem próbkowania przy prawdzie deweloperskiej, ale o około 70% za szerokie i przesunięte o około +0,0002 przy prawdzie zerowej. Przy `fixed`: około 13% za wąskie w obu scenariuszach. H3: sondy nie są dopasowywane w losowaniach, więc przedziały pomijają zmienność dopasowania sondy.
- H2 dotyczy szczebla freeze frame naszego komparatora, który dorównuje dostawcy (K1). Nie dotyczy modelu samego dostawcy, który nie ma krzyżowo dopasowanej wersji „z klatką”. Kontrast klatka + dostawca − dostawca jest tylko opisowy.
- Populacja pierwotna to strzały, które SoccerNet oznaczył. SoccerNet pomija około 42% strzałów zablokowanych i 68% niecelnych strzałów StatsBomb. Populacja łączona to częściowo naprawia, ale jej klatki mają kotwicę w czasie StatsBomb, a 147 z 797 strzałów tylko ze StatsBomb (głównie z doliczonego czasu) nie ma klatki i wypada (decyzja C).
- Kluby z meczów ewaluacyjnych występują w meczach treningowych innych sezonów, jak w archiwalnym podziale testowym.
- Klatka nie widzi wysokości piłki. Dostawca zna wysokość kontaktu, technikę i presję (raport: tabela „co widzi które źródło”).
