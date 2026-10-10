# Udział klatek po kontakcie, test znanej odpowiedzi i werdykt (xg-credibility, ticket 10)

Wynik deweloperski: wszystkie mecze były widziane we wcześniejszych rundach. Analiza wdraża zatwierdzony protokół `research/xgc_direction_probe_protocol.md` (SHA-256 `23c8bd25…0775a0`, commit `41dfd12`, zatwierdzony 2026-10-09). Wejście: pliki robocze ticketu 09 (`results/xgc_readability`, hashe zgodne z jego rekordem ukończenia) i tabela predykcji z ticketu 08 (hashe sprawdzone; zamrożone embeddingi są potrzebne do testu znanej odpowiedzi). Krzywe D przeliczone tutaj są identyczne z krzywymi ticketu 09, w punkcie i w każdym z 2 000 losowań meczów. Przedziały to percentylowe 95% z tych samych losowań co w tickecie 09. Werdykt jest tylko raportowany i nie zmienia żadnego modelu.

## Werdykt

**Niekonkluzywny (reguła 1).** Odczyt F nie przeszedł dwóch z czterech bramek: kontroli pozytywnej (połowa szerokości przedziału U_F 0,218 > 0,15) i testu znanej odpowiedzi (dolna granica A poniżej 0). Placebo i kontrola negatywna przeszły. Według protokołu ten wynik nie pozwala ani stwierdzić przecieku po kontakcie, ani go wykluczyć. Wyniki G i X poniżej są opisowe.

Powód jest ten sam, który ticket 09 pokazał na krzywych AUC: zamrożony DINOv2-L słabo czyta celność nawet po kontakcie (AUC F 0,56–0,61 w oknie POST, poniżej najsłabszego scenariusza symulacji, 0,69–0,70). Przy tak małym skoku Δ_F przedział U_F jest szeroki, a test +200 ms nie ma mocy. Symulacja przewidziała ten przypadek, a bramka „połowa szerokości ≤ 0,15” zadziałała tak, jak miała.

## Wejścia reguł

| Wejście | Wartość | Przedział 95% | Warunek protokołu | Wynik |
|---|---|---|---|---|
| Δ_F | 0,060 | [0,046; 0,074] | dolna granica > 0 | spełniony |
| U_F | 0,036 | [−0,221; 0,216] | połowa szerokości ≤ 0,15 | 0,218, niespełniony |
| Kontrola pozytywna F | | | oba warunki | **nie przechodzi** |
| A (F, sondy ponownie uczone) | 0,027 | [−0,067; 0,148] | dolna granica > 0 | niespełniony |
| A_pred (F) | 0,031 | [0,000; 0,054] | | |
| A / A_pred (F) | 0,87 | | 0,5–1,5 | spełniony |
| Test znanej odpowiedzi F | | | oba warunki | **nie przechodzi** |
| π(−800) | 0,025 | [−0,131; 0,183] | \|π\| > 0,10 i przedział wyklucza 0 → porażka | przechodzi |
| π(−600) | 0,132 | [−0,088; 0,342] | jak wyżej | przechodzi |
| Δ_neg | 0,0002 | [−0,011; 0,011] | \|wartość\| > 0,10·Δ_F = 0,006 i przedział wyklucza 0 → porażka | przechodzi |
| D_neg(0) | 0,004 | [−0,005; 0,012] | jak wyżej | przechodzi |
| U_G | 0,039 | [−0,040; 0,112] | reguła 3: dolna granica > 0,10 | niespełniony |
| U_X | 0,024 | [−0,104; 0,136] | jak wyżej | niespełniony |
| U_G − U_F | 0,002 | [−0,180; 0,253] | reguła 3: dolna granica > 0 | niespełniony |
| U_X − U_F | −0,012 | [−0,220; 0,244] | jak wyżej | niespełniony |
| Nachylenie plateau F, (D(+800) − D(+400)) / Δ_F | 1,16 | [0,93; 1,51] | raportowane | czytelność nadal rośnie |

Kontrola pozytywna G (Δ_G 0,173 [0,150; 0,197], połowa szerokości U_G 0,076) i X (Δ_X 0,258 [0,215; 0,303], połowa szerokości 0,120) przechodzi. Gdyby F przeszedł bramki, reguła 3 i tak by nie zadziałała: dolne granice U_G i U_X są poniżej 0,10, a różnice U_G − U_F i U_X − U_F obejmują 0. Reguła 2 też by nie zadziałała (dolna granica U_F −0,221). Tego nie można jednak czytać jako „brak przecieku”: reguła 4 wymaga przejścia wszystkich bramek.

## Udział U, Δ i t50 po odczytach

| Odczyt | Strzały | Δ | U | Połowa szerokości U | t50 [ms] |
|---|---|---|---|---|---|
| F (sonda, DINOv2-L zamrożony) | 9 307 | 0,060 [0,046; 0,074] | 0,036 [−0,221; 0,216] | 0,218 | > +400 [+398; > +400] |
| G (głowa celności G-combo) | 9 307 | 0,173 [0,150; 0,197] | 0,039 [−0,040; 0,112] | 0,076 | +397 [+362; > +400] |
| X (logit xG G-combo) | 9 307 | 0,258 [0,215; 0,303] | 0,024 [−0,104; 0,136] | 0,120 | +380 [+318; > +400] |
| X0 (logit xG kontroli bez głowy) | 9 307 | 0,292 [0,243; 0,341] | 0,027 [−0,093; 0,132] | 0,113 | +377 [+335; > +400] |
| Gp (sonda, embedding G-combo) | 7 487 | 0,158 [0,135; 0,182] | −0,028 [−0,166; 0,089] | 0,128 | > +400 [+381; > +400] |
| G0p (sonda, embedding kontroli) | 7 487 | 0,155 [0,132; 0,179] | −0,020 [−0,170; 0,100] | 0,135 | > +400 [+389; > +400] |

Krzywe S i U: `share_curves.png`, `s_curves.csv`, `shares.csv`. Punktowo wszystkie odczyty mają U blisko 0, a przełom t50 leży około +400 ms albo później. Przy kontaktach skupionych tuż po kotwicy, jak w scenariuszach symulacji, t50 wynosiłby około +50…+100 ms. Tak późny przełom wskazuje raczej na powolne narastanie czytelności kierunku w uśrednionych embeddingach niż na kontakty po +400 ms. Krzywe F, Gp i G0p nie mają plateau do +1000 ms.

## Test znanej odpowiedzi (+200 ms)

Połowa strzałów populacji (4 601 z 9 307, `default_rng(1)` na strzałach posortowanych po `shot_id`) dostała klatki z offsetu o + 200 ms. Estymator potrzebuje partnera +200 dla PRE (−1600…−600 → −1400…−400), dla 0 (→ +200), dla POST (+400, +600, +800 → +600, +800, +1000) i dla okna KA (−400…+200 → −200…+400). Wszystkie te offsety są w siatce. Offsety +240…+360 i +1000 nie mają partnera, zostały bez zmian i nie wchodzą do statystyki. Sondy F zostały wytrenowane od nowa z tym samym C = 1e−4. Ten sam potok bez przesunięcia kotwicy odtwarza logity F z ticketu 09 co do bitu (maksymalna różnica 0).

| Odczyt | A | A_pred | A / A_pred | Zmiana w kotwicy S_KA(0) − S(0) | Przewidywanie ½·(S(200) − S(0)) | Wynik |
|---|---|---|---|---|---|---|
| F (bramka) | 0,027 [−0,067; 0,148] | 0,031 [0,000; 0,054] | 0,87 | 0,044 [−0,068; 0,206] | 0,042 [−0,026; 0,105] | nie przechodzi |
| G (opisowo) | 0,061 [0,026; 0,096] | 0,054 [0,042; 0,064] | 1,13 | 0,084 [0,030; 0,138] | 0,055 [0,023; 0,084] | przeszedłby |
| X (opisowo) | 0,079 [0,028; 0,137] | 0,062 [0,047; 0,075] | 1,28 | 0,106 [0,031; 0,188] | 0,055 [0,014; 0,091] | przeszedłby |

Stosunek A / A_pred dla F mieści się w tolerancji, ale A nie odróżnia się od 0. Porażka tej bramki wynika z braku mocy testu przy późnym wzroście czytelności, a nie z wady estymatora. Protokół spodziewa się A_pred ≈ 0,16, gdy cały wzrost czytelności mieści się w oknie. Tu A_pred wynosi 0,03, bo wzrost S_F zaczyna się dopiero za oknem KA (po +200 ms), więc przesunięcie kotwicy o 200 ms prawie nie zmienia S w oknie. G i X czytają lepiej i wykrywają przesunięcie z ilorazem bliskim 1. Wykres: `known_answer.png`, tabela: `known_answer.csv`.

## G-combo a kontrola bez głowy celności

Na wspólnych osiach (`head_free_comparison.png`) krzywe obu par prawie się pokrywają:

- sondy na embeddingach: U_Gp − U_G0p = −0,008 [−0,092; 0,081];
- logity xG: U_X − U_X0 = −0,003 [−0,048; 0,038].

Głowa celności nie nauczyła G czytać kierunku piłki w klatce kotwicy ponad to, co ma kontrola bez głowy. W oknie POST X0 jest nawet nieco wyżej niż X (Δ_X0 0,292 wobec Δ_X 0,258). Liczby są opisowe i nie wchodzą do reguł.

## Warstwy (opisowo)

Sparowane różnice U_warstwa − U_całość (`differences.csv`):

| Warstwa | U_F | U_G | U_X | U_G − U_G[całość] | U_X − U_X[całość] |
|---|---|---|---|---|---|
| Dobitki (653 strzały, 237 meczów) | −0,47 [−9,91; 9,00] | 0,256 [−0,075; 0,486] | 0,264 [−0,179; 0,533] | 0,217 [−0,097; 0,432] | 0,241 [−0,178; 0,505] |
| Podział test (1 820 strzałów, 95 meczów) | 0,026 [−0,606; 0,358] | 0,225 [0,089; 0,354] | 0,281 [0,081; 0,445] | 0,187 [0,063; 0,316] | 0,257 [0,066; 0,430] |
| Bez cięć (8 854 strzały) | −0,064 [−0,354; 0,129] | 0,016 [−0,069; 0,092] | −0,003 [−0,141; 0,111] | −0,023 [−0,045; −0,004] | −0,027 [−0,073; 0,013] |

- W podziale test U_G i U_X są wyraźnie wyższe niż w całej populacji, ale nie z powodu kotwicy. Przegląd kodu (Fable, 2026-10-10) pokazał, że w tej warstwie okno PRE ma niższy i bardziej płaski poziom D (dla G średnio 0,011 wobec 0,033 w całej populacji), więc ekstrapolowana prosta P(0) leży nisko i podnosi D − P równo od −400 ms, bez narastania przy kotwicy. Samo D_G(0) w podziale test (0,058) nie jest wyższe niż w całej populacji (około 0,068). To efekt linii bazowej w warstwie z 95 meczami albo przypadek przy 4 warstwach × 6 odczytach, a nie efekt czasu kontaktu. Nie wynika też mechanicznie z oceny strzałów test średnią z dziesięciu modeli: D jest liniowe w logitach, więc D warstwy to średnia D poszczególnych modeli. Porównywalne grupy (train, valid, foldy 0–4) dają U_G od −0,09 do +0,07. Obserwacja nie zmienia werdyktu i nie jest dowodem przecieku.
- U_X0 w podziale test też jest wyższe (0,252 [0,056; 0,417]), więc efekt nie pochodzi z głowy celności.
- Dobitki: Δ_F nie odróżnia się od 0, więc U_F nie ma tu sensu.

## Wrażliwość na przypisanie goli (opisowo)

Sondy pozostają te same (populacja główna); zmienia się tylko zbiór ocenianych strzałów bez gola. Ograniczenia skrajne dają każdy gol z łańcucha dobitek strzałowi z najniższym („worst”) albo najwyższym („best”) xG G-combo w kotwicy, czyli strzałowi, który najbardziej psuje albo poprawia log-loss G-combo.

| Reguła | Strzały | Zmiana wobec głównej (+/−) | U_F | Δ_F | t50_F | U_X |
|---|---|---|---|---|---|---|
| najbliższy (główna) | 9 307 | 0 / 0 | 0,036 [−0,221; 0,216] | 0,060 [0,046; 0,074] | > +400 | 0,024 [−0,104; 0,136] |
| najwcześniejszy w łańcuchu | 9 308 | +94 / −93 | 0,013 [−0,267; 0,200] | 0,057 [0,044; 0,071] | > +400 | −0,035 [−0,201; 0,101] |
| ostatni w łańcuchu | 9 307 | 0 / 0 | 0,036 [−0,221; 0,216] | 0,060 [0,046; 0,074] | > +400 | 0,024 [−0,104; 0,136] |
| łańcuchy wyłączone | 8 654 | 0 / −653 | 0,086 [−0,159; 0,256] | 0,062 [0,047; 0,076] | > +400 | 0,038 [−0,097; 0,152] |
| ograniczenie „worst” | 9 307 | +76 / −76 | 0,025 [−0,248; 0,209] | 0,058 [0,044; 0,072] | > +400 | −0,020 [−0,182; 0,112] |
| ograniczenie „best” | 9 307 | +20 / −20 | 0,028 [−0,241; 0,211] | 0,060 [0,045; 0,073] | > +400 | 0,013 [−0,117; 0,126] |

Reguła „ostatni” jest z konstrukcji tą samą regułą co „najbliższy” (ticket 03). Żadna reguła nie przesuwa wyniku: przedziały U_F są tak samo szerokie, a Δ_F zmienia się najwyżej o 0,003 (`sensitivity.csv`).

## Ograniczenia

- Sonda wykrywa tylko przeciek kierunku piłki. Inne sygnały po kontakcie, na przykład reakcja bramkarza, nie muszą korelować z celnością strzałów bez gola.
- U to udział efektywny. Gdy czytelność narasta przez kilka klatek, U jest dolnym ograniczeniem odsetka klatek kotwicy po kontakcie (w symulacji 0,37 przy rzeczywistych 0,50).
- Narastanie czytelności tuż przed kontaktem (zamach, ustawienie ciała) zawyża U o 0,04–0,17. Próg 0,10 chroni werdykt tylko przy braku kontaktów przed kotwicą.
- Odczyty G i X są uczone w offsecie 0, więc ich czytelność może mieć szczyt w kotwicy z samej zgodności rozkładu. Trend z okna PRE tego nie usuwa, więc U_G i U_X mogą być zawyżone. Dlatego reguła 3 wymaga, żeby U_G lub U_X było wyższe od U_F.
- Estymator zakłada, że embedding reaguje na treść po kontakcie addytywnie, a czas kontaktu nie zależy od celności. Symulacja tego nie sprawdza.
- Modele G uczono na JPEG ze źródła, a każda klatka sondy to PNG zdekodowany z klipu (ponowne kodowanie crf-26). To przesunięcie domeny per strzał o około 0,01–0,13 logitu, obecne przy każdym offsecie i prawie stałe między offsetami. Trend z okna PRE je usuwa, więc kontrasty między offsetami w pierwszym przybliżeniu się nie zmieniają.
- Sondy wybrały dolną krawędź siatki C, a czytelność F jest niska (AUC ≤ 0,61 do +800 ms). Uśredniony embedding zamrożonego DINOv2-L słabo widzi kierunek małej piłki. To jest główna przyczyna werdyktu „niekonkluzywny”.
- Czytelność F nie osiąga plateau do +1000 ms (nachylenie plateau 1,16 Δ_F). Symulacja badała spadek plateau (zawyża U o 0,03–0,05), a nie dalszy wzrost.
- Wykluczenia z populacji per offset (184 strzały, 1,9%, od 17 w offsecie 0 do 102 w +280 ms) podaje `results_reports/readability_2026-10/exclusions.csv`.
- Wszystkie mecze były widziane we wcześniejszych rundach. Wyniki są deweloperskie.

## Odczytanie protokołu w miejscach niejednoznacznych

1. Losowanie połowy w teście znanej odpowiedzi: `n` to strzały populacji analizy (9 307 pełnych przypadków bez gola), posortowane po `shot_id`. Sondy KA, tak jak w kontroli negatywnej ticketu 09, uczą się i oceniają tylko na populacji.
2. Bramka testu znanej odpowiedzi używa ilorazu punktowego A / A_pred, tak jak symulacja. Przedział ilorazu jest raportowany opisowo (F: [−5,6; 11,4]).
3. U dla kontroli negatywnej nie jest raportowane: przy Δ_neg ≈ 0 krzywa S nie ma sensu, a bramka czyta tylko Δ_neg i D_neg(0).
4. Warstwy dostają U, Δ, t50, krzywe S i sparowane różnice. Test znanej odpowiedzi liczony jest tylko w populacji głównej, bo tylko tam jest bramką.
5. Ograniczenia skrajne we wrażliwości na przypisanie goli: model to G-combo, xG w kotwicy (`xg_logit`, offset 0, out-of-fold). Kalibracja jest monotoniczna i nie zmienia wyboru strzału w łańcuchu. Jedyny rzut karny w łańcuchu dobitek (train) nie może dostać gola, bo protokół wyłącza rzuty karne.
6. Wrażliwość używa tych samych losowań meczów co populacja główna. Żaden strzał nowej populacji nie leży poza kluczami losowań (kolumna `outside_bootstrap_keys` = 0).

## Pliki

- `shares.csv`: Δ, U z połową szerokości, kontrola pozytywna, t50 per odczyt i warstwa; kolumna `role`.
- `s_curves.csv`, `s_curves_known_answer.csv`: S(o) z przedziałami, bez przesunięcia i po przesunięciu kotwicy (F, G, X). W drugim pliku kolumna `partnered` = False oznacza offsety bez partnera +200 (+240…+360, +1000): mieszanka połowy przesuniętej i nieprzesuniętej, poza każdą statystyką.
- `known_answer.csv`: A, A_pred, iloraz, zmiana w kotwicy.
- `differences.csv`: sparowane różnice U (wejścia reguły 3, porównanie z kontrolą bez głowy, warstwy).
- `sensitivity.csv`: wiersze wrażliwości na przypisanie goli.
- `summary.json`: werdykt, bramki, wszystkie wejścia reguł, statystyki każdego odczytu. `complete.json`: hashe źródeł, wejść (ticket 08 i ticket 09), plików roboczych i wyników oraz hash protokołu.
- Wykresy: `share_curves.png`, `known_answer.png`, `head_free_comparison.png`, `u_overview.png`. Krzywe D i AUC są w `results_reports/readability_2026-10`.

Odtworzenie: `CUDA_VISIBLE_DEVICES="" ./run.sh python scripts/report_post_contact_share.py`. Tylko CPU, kilka minut (hashowanie plików ticketu 08 i dwa dopasowania sondy F z jednym C; z pamięci podręcznej `results/xgc_post_contact` szybciej).
