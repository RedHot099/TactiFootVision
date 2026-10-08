# Protokół sondy kierunku: xg-credibility, ticket 06

Status: projekt do zatwierdzenia przez właściciela. Protokół powstał przed wycięciem jakiejkolwiek klatki z offsetem i przed obliczeniem jakiegokolwiek wyniku sondy.

Pytanie: ile klatek kotwicy pokazuje grę po kontakcie? Kierunek piłki rozstrzyga się w momencie kontaktu. Jeśli status celności da się odczytać z klatki kotwicy lepiej, niż wynika z trendu sprzed kontaktu, to klatki kotwicy niosą informację, która istnieje dopiero po kontakcie. To jest przeciek po kontakcie. Sonda nie ma ręcznych etykiet kontaktu. Czas kontaktu wyprowadza z tego, jak czytelność celności zmienia się wzdłuż osi offsetów.

Powód do pytania dały wstępne kontrole AI. Na 17 strzałach z przedziałem kontaktu kotwica wypadła przed przedziałem raz, w przedziale 13 razy, po przedziale 3 razy. Takie liczby niczego nie rozstrzygają, ale nie pozwalają też założyć, że problem nie istnieje.

Werdykt sondy jest wyłącznie raportowany. Nie zmienia żadnego modelu, kalibracji, danych treningowych, kotwic ani etykiet.

Wszystkie liczby poniżej są ustalone z góry. Agenci ticketów 07–10 wdrażają je bez własnych wyborów. Każdą konieczną zmianę obsługuje sekcja "Odstępstwa".

## Pojęcia

Słownictwo za `GLOSSARY.md`, sekcja xG. Offset to czas klatki względem kotwicy strzału w ms, ujemny przed kotwicą. Klatka kotwicy to klatka z offsetem 0. Kontakt to uderzenie piłki przy strzale. Strzał celny to strzał oznaczony w źródle jako `Shots on target`.

## Populacja

- Indeks `index_v1_2.parquet`, w którym gol należy do najbliższego strzału.
- Strzały bez gola (`goal == False`) i bez rzutów karnych (`is_penalty == False`): 9 491 strzałów w 500 meczach. Celnych jest 4 283, niecelnych 5 208. Według podziału: train 5 752, valid 1 816, test 1 923.
- Etykieta sondy: `y = 1` dla `label == "Shots on target"`, inaczej `y = 0`.
- Każdy wynik sondy pochodzi z predykcji out-of-fold, zgodnie z tabelą predykcji per strzał z ticketu 08. Strzały train i valid ocenia para modeli obu seedów z foldu, który wyłączył ich mecz. Strzały test oceniają wszystkie dziesięć modeli. Logity są uśredniane przed kalibracją.
- Analiza obejmuje pełne przypadki. Strzał wchodzi do każdej krzywej tylko wtedy, gdy ma klatki we wszystkich 30 offsetach siatki, a każda klatka ma zdekodowany czas odległy od nominalnego offsetu o co najwyżej 20 ms. Dzięki temu populacja jest identyczna w każdym offsecie. Raport podaje liczbę wykluczeń dla każdego offsetu. Jeśli wykluczenia obejmą więcej niż 5% populacji, analiza się zatrzymuje i wraca do właściciela.

## Enkodery i odczyty

Trzy enkodery z tabeli predykcji per strzał:

| Klucz | Model | Co daje |
|---|---|---|
| `g_combo` | G-combo z ticketu 04, głowa celności z wagą 0,5 | logit xG, logit celności, embedding |
| `g_combo_no_head` | kontrola z ticketu 05, waga głowy celności 0 | logit xG, embedding |
| `dinov2_frozen` | DINOv2-L z tym samym hashem wag co szkielet G, bez dostrajania | embedding |

Embedding G to wektor wchodzący do MLP głowy, czyli token CLS i wyjścia zapytań attention pooling. Embedding `dinov2_frozen` to konkatenacja tokenu CLS i średniej tokenów łatek z ostatniej warstwy, 2 048 wymiarów. Obraz przechodzi to samo przetwarzanie co wejście G-combo przy inferencji: maska nakładek, bez augmentacji.

Z enkoderów powstaje sześć odczytów:

| Odczyt | Źródło | Rola |
|---|---|---|
| F | sonda liniowa na `dinov2_frozen` | główny; estymator, bramki, test znanej odpowiedzi |
| G | logit celności `g_combo` | reguła "przeciek nauczony tylko przez G" |
| X | logit xG `g_combo` | studium zdarzeń xG; reguła "tylko przez G" |
| X0 | logit xG `g_combo_no_head` | studium zdarzeń xG kontroli; opisowy |
| Gp | sonda liniowa na embeddingu `g_combo` | porównanie z kontrolą; opisowy |
| G0p | sonda liniowa na embeddingu `g_combo_no_head` | porównanie z `g_combo`; opisowy |

## Siatka offsetów

30 offsetów w ms:

```
−1600, −1400, −1200, −1000, −800, −600,
−400, −360, −320, −280, −240, −200, −160, −120, −80, −40,
0, +40, +80, +120, +160, +200, +240, +280, +320, +360, +400,
+600, +800, +1000
```

Role offsetów:

- okno przed kontaktem PRE = {−1600, −1400, −1200, −1000, −800, −600}; z niego liczony jest trend i placebo;
- okno gęste DENSE = od −400 do +400 co 40 ms, czyli co klatkę przy 25 fps;
- okno po kontakcie POST = {+400, +600, +800}; to kontrola pozytywna i poziom plateau;
- +1000 służy tylko jako partner +800 w teście znanej odpowiedzi;
- okno testu znanej odpowiedzi KA = od −400 do +200 co 40 ms, razem 16 offsetów.

Siatka zawiera offset+200 dla każdego offsetu, którego używa estymator, czyli PRE, 0 i POST, oraz test znanej odpowiedzi, czyli KA. Offsety od +240 do +360 nie mają partnera +200. Służą tylko krzywym opisowym i wyznaczeniu przełomu t50. Klipy mają 8 s z kotwicą w 4. sekundzie, więc cała siatka mieści się w klipie.

## Metryka czytelności

Dla odczytu r, offsetu o i strzału i niech `z_r(i, o)` będzie logitem odczytu: sondy, głowy celności albo xG. Logit jest centrowany w meczu:

```
ż_r(i, o) = z_r(i, o) − średnia z_r(j, o) po strzałach j populacji z tego samego meczu
```

Czytelność to różnica średnich centrowanych logitów między strzałami celnymi i niecelnymi:

```
D_r(o) = średnia ż_r(i, o) po y = 1 − średnia ż_r(i, o) po y = 0
```

Centrowanie w meczu usuwa różnice między meczami, takie jak kamera czy częstość strzałów celnych. Sprawia też, że skala logitów różnych modeli foldów nie miesza się w obrębie meczu. W warstwach średnia meczowa pochodzi z całej populacji meczu, a różnica średnich tylko ze strzałów warstwy.

Dodatkowo, wyłącznie opisowo, raport podaje AUC celności po offsetach dla każdego odczytu poza X i X0, na logitach bez centrowania.

### Sondy liniowe

Sonda jest stała w czasie: jedna sonda na fold, wytrenowana raz i zastosowana bez zmian do wszystkich 30 offsetów. Symulacja pokazała, że sonda dopasowywana osobno w każdym offsecie zaniża udział o 0,07–0,23, bo jej log-loss nie jest liniowy w udziale klatek po kontakcie.

- Wiersze treningowe: klatki z offsetów PRE ∪ POST, razem 9 offsetów, ze strzałów z meczów treningowych sondy. Każdy strzał daje 9 wierszy z tą samą etykietą.
- Model: `StandardScaler` dopasowany na wierszach treningowych, potem `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)` bez wag klas.
- C wybierane raz na enkoder z siatki {1e−4, 3e−4, 1e−3, 3e−3, 1e−2, 3e−2, 1e−1}. Kryterium to średni log-loss w walidacji krzyżowej po foldach opisanych niżej, na danych bez przesunięcia i z prawdziwymi etykietami. Przy remisie wygrywa mniejsze C. Wybrane C obowiązuje też w teście znanej odpowiedzi i w kontroli negatywnej.
- Odczyt F: foldy zewnętrzne to domowe foldy G z pliku `cache/folds_v12_home5.csv`, przypisane meczom train i valid. Sonda foldu k uczy się na strzałach train i valid spoza foldu k i ocenia fold k. Strzał test dostaje średnią logitów pięciu sond.
- Gp i G0p: embeddingi pochodzą z modelu z foldu k i seedu 0, który wyłączył mecz strzału, więc tylko strzały train i valid mają embeddingi out-of-fold. Sonda działa w obrębie foldu k na jego wyłączonych strzałach. Foldy wewnętrzne: mecze foldu k posortowane leksykograficznie, fold wewnętrzny = pozycja mod 5. Gp i G0p nie obejmują strzałów test.

## Estymator udziału klatek po kontakcie

Dla odczytu r:

1. Trend sprzed kontaktu: prosta `P_r(o) = a + b·o` dopasowana metodą najmniejszych kwadratów do punktów `(o, D_r(o))` dla o ∈ PRE, z równymi wagami.
2. Skok kontroli pozytywnej: `Δ_r = średnia D_r(o) po o ∈ POST − P_r(0)`.
3. Znormalizowana krzywa: `S_r(o) = (D_r(o) − P_r(o)) / Δ_r`.
4. Udział klatek kotwicy po kontakcie: `U_r = S_r(0)`.
5. Przełom: `t50_r` to pierwsze przecięcie poziomu 0,5 przez S_r w oknie DENSE, z interpolacją liniową między sąsiednimi offsetami. Gdy S_r(−400) ≥ 0,5, t50 = "≤ −400". Gdy S_r nie osiąga 0,5 do +400, t50 = "> +400".

Wielkość szacowana to efektywny udział: średnia czytelność kierunku w klatkach kotwicy, względem plateau po kontakcie. Gdy kierunek staje się czytelny już w pierwszej klatce po kontakcie, U równa się odsetkowi klatek kotwicy po kontakcie. Gdy czytelność narasta przez kilka klatek, U jest dolnym ograniczeniem tego odsetka. Strzały, których kierunku nigdy nie widać, na przykład zablokowane albo zasłonięte, nie przesuwają U, o ile ich czas kontaktu nie zależy od tego, czy są czytelne.

t50 odpowiada mediacie czasu, w którym kierunek staje się czytelny. Opisuje "przełom" ze specyfikacji. Reguły werdyktu opierają się na U, bo przełom przed kotwicą to tyle co U > 0,5, a za przeciek uznajemy już znacznie mniejszy udział.

## Bootstrap

- 2 000 losowań, klastrowanie po meczach.
- Sekwencja losowań jak w `game_draws` z `scripts/report_complexity.py`, czyli ta sama co w `scripts/significance.py`. Klucze to `np.unique` z identyfikatorów meczów populacji, `rng = np.random.default_rng(0)`, a każde losowanie to `rng.choice(M, M, replace=True)`. Mecz wylosowany k razy wchodzi z wagą k.
- Sondy nie są dopasowywane od nowa w losowaniach. Losowania przeważają logity out-of-fold per strzał, a wszystkie statystyki, czyli D, P, Δ, S, U, t50 i statystyki kontroli, liczy się od nowa w każdym losowaniu. W symulacji błąd standardowy z takiego bootstrapu odbiegał od rozrzutu między replikacjami o najwyżej 12% wszędzie tam, gdzie kontrola pozytywna przechodzi.
- Do percentyli t50 wartość "≤ −400" wchodzi jako −400, a "> +400" jako +440.
- Przedziały percentylowe 95%: `np.percentile(x, [2.5, 97.5])`.
- Wszystkie odczyty, warstwy i kontrole używają tych samych losowań, więc różnice, takie jak U_G − U_F, są sparowane.
- Losowanie, w którym warstwa nie ma strzału celnego albo niecelnego, pomija się dla tej warstwy. Raport podaje liczbę pominięć.

## Kontrole

### Placebo

Prosta z offsetów {−1600, −1400, −1200, −1000}, przedłużona do −800 i −600. Reszta placebo `π(o) = (D_F(o) − P'_F(o)) / Δ_F` dla o ∈ {−800, −600}. Kontrola placebo nie przechodzi, gdy dla któregoś z dwóch offsetów |π(o)| > 0,10 i przedział 95% π(o) wyklucza 0.

### Kontrola pozytywna

Przechodzi, gdy spełnione są oba warunki:

- dolna granica 95% Δ_F > 0;
- połowa szerokości przedziału 95% U_F ≤ 0,15.

Drugi warunek odróżnia płaską kontrolę od słabej. W symulacji przy czytelności po kontakcie na poziomie AUC 0,69–0,70 połowa szerokości wynosiła 0,22–0,39, a estymator był obciążony do −0,06. Przy AUC od 0,72 wzwyż połowa szerokości wynosiła 0,02–0,13, a obciążenie mieściło się w ±0,03. Te same warunki z odczytem G lub X zamiast F definiują kontrolę pozytywną dla G i X.

### Kontrola negatywna

Etykiety celności permutowane w obrębie meczu: `np.random.default_rng(2)`, mecze w porządku posortowanym, w każdym meczu permutacja etykiet strzałów posortowanych po `shot_id`. Cały potok odczytu F powtarza się na permutowanych etykietach: trening sond z tym samym C, D, Δ. Kontrola nie przechodzi, gdy dla `Δ_neg` albo `D_neg(0)` zachodzą jednocześnie dwa warunki: wartość bezwzględna przekracza 0,10·Δ_F, a przedział 95% wyklucza 0.

### Test znanej odpowiedzi (+200 ms)

- Połowa strzałów: `np.random.default_rng(1).random(n) < 0.5` na strzałach posortowanych po `shot_id`.
- Dla wybranych strzałów każda klatka z offsetem o zostaje zastąpiona klatką z offsetem o+200, jeśli taki offset jest w siatce. Offsety bez partnera, czyli +240…+360 i +1000, zostają bez zmian i nie wchodzą do statystyki.
- Cały potok odczytu F powtarza się na danych przesuniętych: trening sond z tym samym C, D, P, Δ, S. Wynik to S_KA.
- Statystyka: `A = średnia (S_KA(o) − S_F(o)) po o ∈ KA`.
- Przewidywanie z danych bez przesunięcia: `A_pred = ½ · średnia (S_F(o+200) − S_F(o)) po o ∈ KA`. Gdy cały wzrost czytelności mieści się w oknie, A_pred ≈ 0,16 niezależnie od tego, gdzie leżą kontakty.
- Test przechodzi, gdy dolna granica 95% A > 0 i 0,5 ≤ A / A_pred ≤ 1,5.
- Raport podaje też zmianę w samej kotwicy, `S_KA(0) − S_F(0)`, obok przewidywania `½·(S_F(200) − S_F(0))`.
- Dla odczytów G i X test liczy się opisowo, na logitach z offsetu o+200, bez ponownego treningu.

## Reguły werdyktu

Wszystkie granice to percentylowe przedziały 95% z bootstrapu. Reguły stosuje się w kolejności. Werdykt wynika wyłącznie z populacji głównej. Warstwy i wrażliwość na przypisanie goli są opisowe.

1. **Niekonkluzywny.** Odczyt F nie przechodzi którejkolwiek z czterech bramek: kontroli pozytywnej, testu znanej odpowiedzi, placebo albo kontroli negatywnej. Raport wskazuje, które bramki nie przeszły. Wyniki G i X idą do raportu jako opisowe.
2. **Przeciek.** Dolna granica U_F > 0,10.
3. **Przeciek nauczony tylko przez G.** Reguła 2 nie zachodzi, a przynajmniej jeden z odczytów G lub X spełnia wszystkie trzy warunki:
   - przechodzi własną kontrolę pozytywną;
   - dolna granica U_r > 0,10;
   - dolna granica różnicy U_r − U_F > 0.
4. **Brak dowodu przecieku.** Żadna z powyższych. Raport dodaje jedno z dwóch zdań: "udział powyżej 0,10 wykluczony", gdy górna granica U_F ≤ 0,10, albo "udział do <górna granica> nie wykluczony".

Skąd próg 0,10. W symulacji przy braku jakichkolwiek kontaktów przed kotwicą, ale z silnym narastaniem czytelności w ostatnich 200 ms przed kontaktem, czyli przy zamachu i ustawieniu ciała, U_F wynosiło średnio 0,09. Dolna granica przekroczyła 0,05 w 65% replikacji, a 0,10 w 1%. Przy progu 0,10 zamach sam nie daje werdyktu "przeciek". Ceną jest mniejsza moc: przy efektywnym udziale 0,16 i średniej sile sygnału reguła 2 zadziałała w 54% replikacji, przy 0,25 w 99%.

## Studium zdarzeń xG

Dla X i X0: D(o), trend P(o) z okna PRE, różnica `D(o) − P(o)` w logitach i S(o), wszystko z przedziałami. Odczyt X wchodzi do reguły 3. X0 jest opisowy: porównanie X z X0 pokazuje, czy głowa celności nauczyła xG reagować na kierunek.

## Porównanie z kontrolą bez głowy

Raport pokazuje na wspólnych wykresach krzywe D i S dla Gp i G0p oraz dla X i X0, z różnicami sparowanymi w bootstrapie: `U_Gp − U_G0p` i `U_X − U_X0`. Te liczby są opisowe i nie wchodzą do reguł.

## Warstwy

Każda warstwa dostaje te same statystyki co populacja główna, dla wszystkich odczytów, na tych samych losowaniach bootstrapu:

- strzały z dobitek: łańcuchy dobitek z ticketu 03, definicja z glosariusza; należą tu wszystkie strzały łańcucha z co najmniej dwoma strzałami;
- oficjalny podział test;
- strzały bez cięć: żadna z 30 klatek strzału nie ma flagi cięcia kamery.

Dla każdej warstwy raport podaje liczbę strzałów i meczów oraz sparowaną różnicę U_warstwa − U_całość z przedziałem. Warstwy nie zmieniają werdyktu.

## Wrażliwość na przypisanie goli

Ticket 10 przelicza U_F, Δ_F, t50_F i U_X dla populacji strzałów bez gola wyznaczonej każdą regułą z ticketu 03: najbliższy strzał, czyli reguła główna, najwcześniejszy strzał łańcucha, ostatni strzał łańcucha, łańcuchy wyłączone, oraz dla obu ograniczeń skrajnych. Sondy zostają wytrenowane na populacji głównej. Zmienia się tylko zbiór ocenianych strzałów. Wiersze są opisowe.

## Co raportujemy

- Krzywe D, S i AUC po offsetach dla każdego odczytu, w populacji głównej i w warstwach, z przedziałami.
- U, Δ i t50 dla każdego odczytu, z przedziałami.
- Wszystkie wejścia każdej bramki i reguły: π(−800), π(−600), Δ_neg, D_neg(0), A, A_pred, A/A_pred, połowa szerokości przedziału U_F, różnice U_G − U_F i U_X − U_F.
- Werdykt i numer reguły, która go dała.
- Nachylenie plateau: `D_F(+800) − D_F(+400)` jako ułamek Δ_F. W symulacji spadek czytelności po +400 ms zawyżał U o 0,03–0,05.
- Liczby wykluczeń z populacji dla każdego offsetu.
- Wyniki wrażliwości na przypisanie goli.
- Hash SHA-256 tego pliku z zatwierdzonego commita, w każdym rekordzie ukończenia wyników sondy.

We wcześniejszej analizie zamrożony G, uczony w kotwicy, oceniono na klatkach z −400 ms i dostał log-loss gorszy o 0,0235. Ta liczba nie mówi nic o przecieku, bo miesza przesunięcie rozkładu z informacją po kontakcie. Dlatego sonda porównuje strzały celne i niecelne wzdłuż offsetów, a nie surową jakość modelu.

## Ograniczenia, które raport musi wymienić

- Sonda wykrywa tylko przeciek kierunku piłki. Inne sygnały po kontakcie, np. reakcja bramkarza, nie muszą korelować z celnością strzału bez gola.
- U to efektywny udział. Przy czytelności narastającej przez trzy klatki i 30% strzałów nieczytelnych symulacja dała U ≈ 0,37 przy rzeczywistym odsetku klatek po kontakcie 0,50.
- Narastanie czytelności tuż przed kontaktem zawyża U o 0,04–0,17. Tyle dała symulacja z zamachem, który w ostatnich 200 ms dodaje 50–100% sygnału sprzed kontaktu. Próg 0,10 chroni werdykt tylko przy braku kontaktów przed kotwicą.
- Odczyty G i X są uczone w offsecie 0, więc ich czytelność może mieć szczyt w kotwicy z samej zgodności rozkładu. Trend z okna PRE tego nie usuwa i U_G oraz U_X mogą być zawyżone. Dlatego reguła 3 wymaga, żeby U_G lub U_X było wyższe od U_F.
- Estymator zakłada, że embedding reaguje na treść po kontakcie addytywnie, a czas kontaktu nie zależy od etykiety celności. Symulacja tego nie sprawdza.
- Wszystkie mecze były widziane we wcześniejszych rundach. Wyniki to wyniki deweloperskie.

## Symulacja przed zamrożeniem protokołu

Wyrzucany kod leży w `/home/kuba/projects/tactifoot_vision/.scratch/xg-credibility/prototype/`: `sim.py`, `battery.py`, tabele w `results_table.txt`. Do protokołu wchodzi tylko werdykt.

Model symulacji: 500 meczów, około 9,5 tys. strzałów, 45% celnych ze zmiennością między meczami, czasy kontaktu względem kotwicy z kilku rozkładów, cecha sprzed kontaktu z trendem, cecha kierunku po kontakcie, szum trwały w obrębie strzału, mnożnik jakości meczu. Sonda logistyczna i bootstrap jak wyżej, 200 replikacji na scenariusz, 1 000 losowań bootstrapu.

Scenariusze czasu kontaktu względem kotwicy, w ms:

| Rozkład | Odsetek klatek kotwicy po kontakcie |
|---|---|
| jednostajny +40…+400 | 0 |
| N(+80, 80) | 0,16 |
| 85% N(+60, 50) i 15% N(−300, 150) | 0,25 |
| N(0, 80) | 0,50 |
| N(−80, 200) | 0,66 |
| N(−80, 80) | 0,84 |
| N(−200, 120) | 0,95 |

Rozmycie: natychmiastowa czytelność albo narastanie przez 3 klatki i 30% strzałów nieczytelnych. Siła sygnału po kontakcie: AUC 0,96, 0,80, 0,72–0,74, 0,69–0,70 i płaski. Dodatkowo: zamach, spadek plateau, brak trendu, silny trend.

Wyniki:

- Forma estymatora. Stała sonda z różnicą centrowanych logitów i trendem liniowym była jedyną wersją bez istotnego obciążenia. Sonda dopasowywana w każdym offsecie, na log-lossie: obciążenie od −0,07 do −0,23 przy udziale powyżej zera. Dopasowanie schodka z jednym przełomem: od −0,05 do −0,20. Dopasowanie dystrybuanty normalnej: do −0,09 przy mieszaninie. Płaski poziom zamiast trendu: od +0,06 do +0,36.
- Obciążenie i pokrycie. Gdy kontrola pozytywna przechodzi, obciążenie względem efektywnego udziału mieści się w przedziale od −0,024 do +0,001, a pokrycie przedziałów 95% w przedziale 0,93–0,98. Błąd standardowy z bootstrapu zgadza się z empirycznym. Połowa szerokości przedziału wynosi 0,02 przy silnym sygnale, 0,03–0,10 przy średnim i 0,10–0,13 przy słabym.
- Test +200 ms. Gdy kontrola pozytywna przechodzi, A > 0 wykryto w 94–100% replikacji. Mediana A/A_pred wynosi 0,97–1,23, a 5. i 95. percentyl mieszczą się w 0,59–1,48.
- Słaba kontrola pozytywna. Przy AUC po kontakcie 0,69–0,70 połowa szerokości przedziału wynosi 0,22–0,39, obciążenie do −0,06, a test +200 ms wykrywa zmianę w 23–91% replikacji. Przy płaskiej kontroli U traci sens, jego odchylenie standardowe sięga 4–9. Bramka "połowa szerokości ≤ 0,15" odrzuca wszystkie te przypadki.
- Kontrole. Pokrycie zera w kontroli negatywnej: 0,83–0,95 dla D_neg(0) i 0,92–0,98 dla Δ_neg, przy średnim |Δ_neg| poniżej 0,002·Δ_F. Średnia największa reszta placebo: 0,03–0,06 przy średnim sygnale i 0,07–0,09 przy słabym.
- Zakłócenia. Zamach opisują ograniczenia. Spadek plateau o 30% na sekundę po +400 ms zawyża U o 0,03–0,05 i przesuwa A/A_pred do około 1,15–1,19.

## Odstępstwa

- Przed obliczeniem jakiegokolwiek wyniku sondy: zmianę protokołu zatwierdza właściciel jako poprawkę w nowym commicie. Wyniki zapisują hash poprawionego pliku.
- Po obliczeniu wyników: każda zmiana analizy jest oznaczona jako post hoc i raportowana obok wyników z protokołu. Werdykt pozostaje werdyktem z reguł tego protokołu.
- Jeśli wdrożenie okaże się niemożliwe, na przykład przez brak klatek ponad próg, brak zbieżności sondy przy każdym C albo brakujący odczyt, agent zatrzymuje się i opisuje problem w tickecie. Nie dobiera zastępczej reguły sam.
