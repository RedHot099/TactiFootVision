# Krzywe czytelności i studium zdarzeń xG (xg-credibility, ticket 09)

Wynik deweloperski: wszystkie mecze były widziane we wcześniejszych rundach. Analiza wdraża zatwierdzony protokół `research/xgc_direction_probe_protocol.md` (SHA-256 `23c8bd25…0775a0`, commit `41dfd12`, zatwierdzony 2026-10-09). Wejście: tabela predykcji per strzał z ticketu 08 (`results/xgc_offset_scores`, hashe sprawdzone przed odczytem) i indeks `index_v1_2.parquet`. Liczby pochodzą z plików CSV w tym folderze. Przedziały to percentylowe 95% z 2 000 losowań meczów; wszystkie odczyty i warstwy używają tych samych losowań.

Ten ticket nie liczy udziału klatek po kontakcie (S, U, t50), testu znanej odpowiedzi ani werdyktu. To jest ticket 10, który czyta pliki robocze opisane na końcu.

## Populacja

- Strzały bez gola i bez rzutów karnych: 9 491. Pełne przypadki (klatka w każdym z 30 offsetów, czas w granicy 20 ms): 9 307 strzałów w 491 meczach, w tym 4 190 celnych. Wykluczono 184 strzały (1,9%), poniżej progu zatrzymania 5%. Wykluczenia per offset: od 17 (offset 0) do 102 (+280 ms), `exclusions.csv`. Dziewięć meczów (pięć test, cztery train/valid) nie ma żadnego pełnego przypadku bez gola, więc bootstrap losuje 491 meczów.
- Strzały z golem (1 366 pełnych przypadków) nie wchodzą do populacji. Sondy je oceniają, żeby ticket 10 mógł przeliczyć wrażliwość na przypisanie goli.

## Sondy liniowe

| Odczyt | Wybrane C | Log-loss CV | Foldy |
|---|---|---|---|
| F (DINOv2-L zamrożony) | 1e−4 | 0,6886 | 5 |
| Gp (embedding G-combo) | 1e−4 | 0,6919 | 25 (5 × 5 wewnętrznych) |
| G0p (embedding kontroli) | 1e−4 | 0,6918 | 25 |

Każda sonda wybrała najmniejsze C z siatki protokołu, a log-loss jest tylko nieznacznie poniżej ln 2 = 0,6931. Sondy zbiegły przy każdym C (najwyżej 865 iteracji). Siatka jest ustalona protokołem, więc C nie schodzi niżej (`probe_selection.csv`).

## AUC celności po offsetach (opisowo, logity bez centrowania)

| Offset [ms] | F | G | Gp | G0p |
|---|---|---|---|---|
| −1600 | 0,526 [0,513; 0,539] | 0,522 [0,509; 0,534] | 0,519 [0,507; 0,531] | 0,515 [0,503; 0,527] |
| −600 | 0,536 [0,524; 0,549] | 0,540 [0,528; 0,551] | 0,536 [0,523; 0,547] | 0,527 [0,514; 0,539] |
| 0 | 0,536 [0,525; 0,549] | 0,551 [0,539; 0,563] | 0,538 [0,525; 0,551] | 0,532 [0,520; 0,545] |
| +200 | 0,544 [0,532; 0,556] | 0,558 [0,546; 0,570] | 0,554 [0,540; 0,567] | 0,543 [0,529; 0,556] |
| +400 | 0,555 [0,543; 0,568] | 0,577 [0,564; 0,589] | 0,576 [0,562; 0,589] | 0,572 [0,558; 0,586] |
| +600 | 0,591 [0,579; 0,603] | 0,607 [0,595; 0,619] | 0,615 [0,602; 0,629] | 0,614 [0,600; 0,628] |
| +800 | 0,611 [0,599; 0,624] | 0,624 [0,612; 0,635] | 0,655 [0,642; 0,668] | 0,653 [0,639; 0,667] |

Gp i G0p obejmują tylko strzały train i valid (7 487). Kontrola negatywna F przy permutowanych etykietach ma AUC 0,50–0,51 przy każdym offsecie. Wykres: `auc_by_offset.png`.

Celność jest słabo czytelna w każdym odczycie. AUC F w oknie POST (0,56–0,61) jest niższe niż w najsłabszym scenariuszu symulacji z protokołu (0,69–0,70).

## Krzywe D i studium zdarzeń xG

D(o) to różnica średnich centrowanych logitów między strzałami celnymi i niecelnymi, P(o) to prosta z okna PRE. Tabela podaje D − P w jednostkach logitu danego odczytu (`curves.csv`, `readability_curves.png`, `event_study_xg.png`).

| Offset [ms] | F | G | X | X0 | Gp | G0p |
|---|---|---|---|---|---|---|
| −400 | +0,000 [−0,008; +0,009] | −0,002 [−0,010; +0,006] | −0,008 [−0,027; +0,011] | −0,014 [−0,034; +0,005] | −0,003 [−0,016; +0,009] | −0,004 [−0,016; +0,008] |
| −200 | +0,001 [−0,010; +0,012] | −0,000 [−0,010; +0,010] | −0,005 [−0,029; +0,019] | −0,011 [−0,037; +0,016] | −0,007 [−0,022; +0,010] | −0,010 [−0,026; +0,006] |
| 0 | +0,002 [−0,011; +0,015] | +0,007 [−0,006; +0,020] | +0,006 [−0,024; +0,038] | +0,008 [−0,024; +0,043] | −0,004 [−0,023; +0,015] | −0,003 [−0,023; +0,017] |
| +80 | +0,006 [−0,008; +0,020] | +0,010 [−0,004; +0,025] | +0,007 [−0,025; +0,041] | +0,009 [−0,027; +0,046] | +0,006 [−0,014; +0,027] | +0,003 [−0,018; +0,023] |
| +160 | +0,005 [−0,010; +0,020] | +0,021 [+0,004; +0,037] | +0,024 [−0,011; +0,061] | +0,022 [−0,016; +0,062] | +0,007 [−0,015; +0,030] | +0,004 [−0,019; +0,026] |
| +200 | +0,007 [−0,008; +0,023] | +0,026 [+0,009; +0,043] | +0,035 [−0,003; +0,074] | +0,037 [−0,004; +0,078] | +0,021 [−0,002; +0,045] | +0,012 [−0,011; +0,034] |
| +280 | +0,013 [−0,003; +0,029] | +0,047 [+0,026; +0,066] | +0,080 [+0,036; +0,123] | +0,083 [+0,037; +0,129] | +0,031 [+0,006; +0,056] | +0,022 [−0,005; +0,047] |
| +400 | +0,019 [+0,002; +0,036] | +0,088 [+0,065; +0,111] | +0,144 [+0,097; +0,192] | +0,168 [+0,117; +0,217] | +0,069 [+0,040; +0,098] | +0,066 [+0,037; +0,093] |
| +800 | +0,086 [+0,062; +0,108] | +0,217 [+0,186; +0,249] | +0,275 [+0,211; +0,344] | +0,311 [+0,239; +0,383] | +0,226 [+0,191; +0,262] | +0,227 [+0,191; +0,262] |

W kotwicy żaden odczyt nie odstaje od trendu sprzed kontaktu: wszystkie przedziały D − P(0) zawierają 0. Odczytane z krzywych, poza protokołem: dolna granica D − P jest stale powyżej 0 od +160 ms (G), +240 ms (X i X0), +280 ms (Gp), +320 ms (G0p) i +400 ms (F). Przełom t50 liczy ticket 10.

Głowa celności a xG: krzywe X i X0 prawie się pokrywają, a X0 (kontrola bez głowy) jest w oknie POST nawet nieco wyżej niż X. Krzywe Gp i G0p są praktycznie identyczne. Sparowane różnice U_X − U_X0 i U_Gp − U_G0p liczy ticket 10.

## Kontrole

| Kontrola | Wartość | Przedział 95% | Próg protokołu | Wynik |
|---|---|---|---|---|
| Placebo π(−800) | 0,025 | [−0,131; 0,183] | \|π\| > 0,10 i przedział wyklucza 0 | przechodzi |
| Placebo π(−600) | 0,132 | [−0,088; 0,342] | jak wyżej | przechodzi (przedział zawiera 0) |
| Negatywna Δ_neg | 0,0002 | [−0,011; 0,011] | \|wartość\| > 0,10·Δ_F = 0,006 i przedział wyklucza 0 | przechodzi |
| Negatywna D_neg(0) | 0,004 | [−0,005; 0,012] | jak wyżej | przechodzi |
| Pozytywna, warunek 1: Δ_F | 0,060 | [0,046; 0,074] | dolna granica > 0 | spełniony |
| Pozytywna, warunek 1: Δ_G | 0,173 | [0,150; 0,197] | dolna granica > 0 | spełniony |
| Pozytywna, warunek 1: Δ_X | 0,258 | [0,215; 0,303] | dolna granica > 0 | spełniony |

Drugi warunek kontroli pozytywnej, połowa szerokości przedziału U ≤ 0,15, jest funkcją estymatora U. Ocenia go ticket 10, tak samo jak test znanej odpowiedzi. Placebo przy −600 ma szeroki przedział, bo Δ_F jest mały, a π dzieli resztę przez Δ_F.

Nachylenie plateau `(D(+800) − D(+400)) / Δ`: F 1,16 [0,93; 1,52], G 0,80 [0,68; 0,95], X 0,60 [0,47; 0,77] (`jumps.csv`). Krzywe nie osiągają plateau do +800 ms. Czytelność F rośnie dalej do +1000 ms. Protokół ostrzega, że zmiana poziomu po +400 ms przesuwa U (w symulacji spadek plateau zawyżał U o 0,03–0,05). Tu poziom rośnie, a nie spada.

## Warstwy (opisowo)

| Warstwa | Strzały | Mecze | Celne |
|---|---|---|---|
| Cała populacja | 9 307 | 491 | 4 190 |
| Dobitki (łańcuchy ≥ 2 strzałów, ticket 03) | 653 | 237 | 417 |
| Oficjalny podział test | 1 820 | 95 | 823 |
| Bez cięć (żadna z 30 klatek nie ma flagi cięcia) | 8 854 | 491 | 4 001 |

Gp i G0p: 7 487 strzałów, 555 dobitek, bez warstwy test (2 000 pominiętych losowań z definicji). W pozostałych warstwach żadne losowanie nie zostało pominięte (`strata.csv`, `strata.png`).

- Bez cięć: krzywe pokrywają się z całą populacją.
- Podział test: G i X leżą w kotwicy powyżej trendu, G +0,040 [+0,013; +0,068], X +0,079 [+0,018; +0,146]. F w tej warstwie nie odstaje: +0,002 [−0,027; +0,029]. Przegląd ticketu 10 (Fable, 2026-10-10) pokazał, że to nie jest szczyt w kotwicy: G jest powyżej trendu już od −400 ms, bez narastania, bo okno PRE tej warstwy ma niższy i bardziej płaski poziom, a ekstrapolowana prosta P(0) leży nisko. Samo D_G(0) nie jest wyższe niż w całej populacji. Ocena strzałów test średnią z dziesięciu modeli tego nie tłumaczy, bo D jest liniowe w logitach. Szczegóły w `results_reports/post_contact_share_2026-10/REPORT.md`.
- Dobitki: 653 strzały, przedziały szerokie. G0p ma w kotwicy D − P = −0,103 [−0,180; −0,021], czyli poniżej trendu.

Sparowane różnice U_warstwa − U_całość liczy ticket 10.

## Ograniczenia

- Sondy wybrały dolną krawędź siatki C, a ich czytelność jest niska (AUC F ≤ 0,61 do +800 ms). Uśredniony (pooled) embedding zamrożonego DINOv2-L słabo widzi kierunek małej piłki.
- Modele G uczono na JPEG ze źródła, a każda klatka sondy to PNG zdekodowany z klipu (ponowne kodowanie crf-26). Ticket 08 zmierzył przesunięcie 0,01–0,13 logitu, prawie stałe między offsetami. Trend z okna PRE je usuwa.
- Odczyty X i X0 używają `xg_logit_raw`: średniego surowego logitu modeli przed kalibracją. Skala różni się między grupami modeli (strzały test: dziesięć modeli, train/valid: dwa). Centrowanie w meczu usuwa przesunięcie, ale nie skalę. Mecz należy zawsze do jednej grupy.
- Pozostałe ograniczenia protokołu (sonda widzi tylko kierunek piłki, U jest udziałem efektywnym, zamach przed kontaktem) dotyczą estymatora z ticketu 10.

## Pliki

- `exclusions.csv`: wykluczenia per offset i łącznie.
- `probe_selection.csv`: log-loss CV dla każdego C, zbieżność, wybór.
- `curves.csv`: D, P, D − P z przedziałami, per odczyt (F, G, X, X0, Gp, G0p, F_neg), warstwa i offset; `skipped_draws`.
- `auc.csv`: AUC per odczyt (F, G, Gp, G0p, F_neg), warstwa i offset.
- `jumps.csv`: Δ z przedziałem, warunek 1 kontroli pozytywnej, nachylenie plateau. Kolumna `role`: `gate_input` dla F, G i X w populacji głównej, reszta opisowa.
- `strata.csv`: liczności warstw per odczyt.
- `summary.json`: bramki i ich wejścia, wybrane C, populacja. `complete.json`: hashe źródeł, wejść, plików roboczych i wyników oraz hash protokołu.
- Pliki robocze dla ticketu 10 w `results/xgc_readability/` (poza repozytorium; schemat w docstringu `scripts/report_readability.py`): `logits.parquet` (logity per strzał i offset, z golami i etykietą permutowaną) oraz `curves.npz` (liczności losowań, sumy per mecz i losowania D dla każdego odczytu i warstwy).

Odtworzenie: `CUDA_VISIBLE_DEVICES="" ./run.sh python scripts/report_readability.py`. Tylko CPU: około 50 minut z dopasowaniem sond, kilka minut z sondami z pamięci podręcznej.
