# Protokół testu transferu: leave-one-league-out

Status: projekt do akceptacji właściciela. Dopóki właściciel nie zatwierdzi commita z tym plikiem, żaden trening transferowy nie startuje.
Runda: xG credibility, prośba 3 recenzenta. Tickety: 11 (ten protokół i plan podziału), 12 (treningi), 13 (analiza).
Plan podziału: `research/xgc_transfer_plan/`, sha256 planu `33229bbfd38dcbbdcd2ba33524cdaf2ab5ae766485f6ba7da08e3dda40b7bc88`.

## 1. Pytanie

Ile tracimy w log-lossie, gdy model nie widział ani jednego meczu danej ligi ani jej klubów, wobec modelu uczonego na tej samej liczbie meczów, wśród których jest połowa meczów tej ligi? Ta różnica to luka transferu przy stałym rozmiarze treningu. Recenzent chciał ją zmierzyć na Ekstraklasie. Tych danych nie będzie, więc każdą z pięciu lig krajowych po kolei traktujemy jak nową ligę.

Wszystkie mecze były widziane we wcześniejszych rundach, a receptę G-combo dobrano na walidacji obejmującej wszystkie ligi. Każda liczba z tego testu jest więc wynikiem rozwojowym i tak ją oznaczamy.

## 2. Dane i pojęcia

Indeks: `index_v1_2.parquet`, sha256 `3968d532ed531eec4244e6e70967bfdffb568b6d3646f1596a494c00ea4a37ee`, 500 meczów, 11 030 strzałów. Liga i sezon pochodzą z dwóch pierwszych członów identyfikatora meczu. Kluby wyciąga zakotwiczony wzorzec `^\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$` z ostatniego członu.

**Normalizacja nazw klubów.** Nazwa przechodzi przez usunięcie akcentów, zwinięcie spacji i mapę aliasów `CLUB_ALIASES` w `xgpoc/transfer/plan.py`. Audyt z 2026-10-08 objął 133 nazwy i wszystkie sezony oraz rozgrywki. Każdy klub ma jedną pisownię, a nazwy z identyfikatorów zgadzają się z polami `gameHomeTeam`/`gameAwayTeam` w `Labels-v2.json` we wszystkich 500 meczach. Mapa aliasów jest więc pusta. Kod odrzuca plan, jeśli dwie nazwy wyglądają podobnie, a para nie ma aliasu ani nie przeszła przeglądu. Podobne są nazwy o podobieństwie pisowni co najmniej 0.75, nazwy, w których słowa jednej zawierają się w drugiej, skróty słowo po słowie ("Man Utd" i "Manchester United") i akronimy ("PSG" i "Paris Saint-Germain"). Przegląd uznał za różne kluby cztery pary: Atl. Madrid i Real Madrid, Dijon i Gijon, Everton i Verona, Manchester City i Manchester United. Test na fikcyjnym indeksie pokazuje, co ta reguła chroni. Wariant "Newcastle United" w Lidze Mistrzów bez aliasu wpada do puli, z aliasem znika.

Straż nie złapie przydomka bez wspólnych liter, np. "Spurs" i "Tottenham Hotspur". Na indeksie v1.2 gwarancją jest więc ręczny audyt, a nie straż. Audyt przejrzał wszystkie 133 nazwy. 21 klubów występuje tylko w Lidze Mistrzów i wszystkie są spoza pięciu lig, np. Ajax, Benfica, Celtic, PSV i Zenit. Każdy klub pięciu lig grający w Lidze Mistrzów ma tam tę samą nazwę co w lidze.

**Kluby ligi L** to wszystkie kluby z meczów ligi L w indeksie, ze wszystkich sezonów.

**Pula ligi L** to mecze, w których nie gra żaden klub ligi L. Wypadają więc mecze samej ligi L i każdy mecz Ligi Mistrzów z jej klubem. Liga Mistrzów nigdy nie jest ligą testową.

**Podzbiory A i B ligi L.** Mecze ligi L dzielimy losowo na dwie części. Losowanie idzie po meczach, osobno w każdym sezonie, z ziarnem planu 0, a mecze rozdajemy na przemian, więc A i B różnią się najwyżej o jeden mecz w każdym sezonie. To nie są połowy meczu. Okres gry nazywamy w tym projekcie "period" i nie ma on tu nic do rzeczy.

**Ramię "bez"** uczy się na 300 meczach z puli. **Ramię "z"** uczy się na 300 meczach, wśród których jest cały drugi podzbiór ligi L, a reszta pochodzi z puli.

**Role meczów w ramieniu.** Każde ramię dzieli swoje 300 meczów na trzy role:

- uczenie (train): 225 meczów, na nich idą kroki gradientu;
- wybór epoki (select): 30 meczów, na nich wybieramy epokę;
- kalibracja (calibration): 45 meczów, na nich dopasowujemy Platta.

Test to podzbiór h ligi L, wspólny dla obu ramion.

## 3. Plan podziału

Plan jest czystą funkcją indeksu i ziarna: `xgpoc.transfer.plan.build_plan`. Skrypt `scripts/make_transfer_plan.py` zapisuje `plan.csv` (liga, podzbiór, ramię, rola, źródło, mecz), `summary.csv`, `pools.csv` i `manifest.json` z hashami indeksu, kodu i planu. Skrypt nie nadpisze innego planu bez `--replace`.

Dla ligi L i podzbioru testowego h:

1. Pulę sortujemy i permutujemy ziarnem wyprowadzonym z (0, L, h).
2. Ramię "bez" to pierwsze 300 meczów tej kolejności.
3. Ramię "z" to drugi podzbiór ligi i pierwsze 300 − |drugi podzbiór| meczów tej samej kolejności. Mecze z puli w ramieniu "z" są więc podzbiorem ramienia "bez". Ramiona różnią się tylko tym, że mecze ligi zastępują ostatnie mecze puli. To ogranicza szum między ramionami do samej zamiany.
4. Role nadaje gładki ważony round-robin z wagami 15:2:3 wzdłuż kolejności puli. Każdy prefiks ma role w tych proporcjach z dokładnością do jednego meczu. Mecze puli w ramieniu "z" zachowują rolę z ramienia "bez", a drugi podzbiór ligi dostaje resztę kwot. Obie strony mają więc dokładnie 225, 30 i 45 meczów.

Rola idzie po meczu, nie po gospodarzu. Mecze kalibracyjne dzielą stadiony z meczami uczącymi, tak jak przy wdrożeniu kalibrowanym na własnych danych. Ramię "bez" uczy i kalibruje się bez stadionów ligi L, ramię "z" ma je w obu rolach. To jest część mierzonej luki. Archiwalne G-combo kalibrowało Platta na predykcjach out-of-fold z foldami po gospodarzu, co bliżej udaje nowy stadion. Tę różnicę zostawiam do decyzji właściciela (pkt 11).

Dlaczego 45 i 30 meczów. Platt ma dwa parametry, a 45 meczów to 910–1 060 strzałów i 120–150 goli. Wybór epoki to wybór jednej z pięciu, więc wystarcza 30 meczów, czyli 590–730 strzałów i 70–110 goli. Większe wycinki zabrałyby mecze z uczenia, a ono już ma 225 meczów zamiast ok. 320 w archiwalnych foldach.

Ramię "bez" dla podzbiorów A i B tej samej ligi korzysta z dwóch różnych losowań puli. Daje to 40 treningów zgodnie ze specyfikacją. Alternatywa ze wspólną kolejnością puli dla A i B dałaby jedno ramię "bez" na ligę i 30 treningów (pkt 11).

### Liczności planu

Pule i wykluczenia:

| liga | mecze | strzały | gole | kluby | odrzucone mecze LM | pula |
|---|---:|---:|---:|---:|---:|---:|
| Premier League | 95 | 2 025 | 270 | 23 | 24 | 381 |
| Ligue 1 | 38 | 726 | 108 | 21 | 19 | 443 |
| Bundesliga | 53 | 1 136 | 155 | 20 | 33 | 414 |
| Serie A | 96 | 2 339 | 290 | 22 | 17 | 387 |
| LaLiga | 117 | 2 671 | 369 | 26 | 43 | 340 |

Test i dawka ligi w ramieniu "z". Każde ramię w każdym wierszu ma 225/30/45 meczów w rolach uczenie/wybór/kalibracja.

| liga | podzbiór | test: mecze | strzały | gole | mecze ligi w "z" (uczenie/wybór/kalibracja) | udział w 300 |
|---|---|---:|---:|---:|---|---:|
| Premier League | A | 48 | 1 034 | 131 | 47 (35/5/7) | 16% |
| Premier League | B | 47 | 991 | 139 | 48 (36/5/7) | 16% |
| Ligue 1 | A | 19 | 358 | 59 | 19 (14/2/3) | 6% |
| Ligue 1 | B | 19 | 368 | 49 | 19 (14/2/3) | 6% |
| Bundesliga | A | 27 | 546 | 80 | 26 (19/3/4) | 9% |
| Bundesliga | B | 26 | 590 | 75 | 27 (20/3/4) | 9% |
| Serie A | A | 48 | 1 188 | 135 | 48 (36/5/7) | 16% |
| Serie A | B | 48 | 1 151 | 155 | 48 (36/5/7) | 16% |
| LaLiga | A | 59 | 1 303 | 169 | 58 (43/6/9) | 19% |
| LaLiga | B | 58 | 1 368 | 200 | 59 (44/6/9) | 20% |

Strzałów w 300 meczach ramienia jest 6 445–6 742, a ramiona tej samej ligi i podzbioru różnią się najwyżej o 134 strzały, czyli 2%. Szczegóły są w `summary.csv`.

Luka mierzy efekt zastąpienia 19–59 meczów puli meczami ligi, a nie efekt uczenia na całej lidze. Dla Ligue 1 dawka to 19 meczów.

## 4. Treningi

Jeden model na ramię, bez uśredniania foldów, ziarna treningu 0 i 1. Ziarno treningu nie zmienia planu, bo plan ma własne ziarno 0. Razem 5 lig × 2 podzbiory × 2 ramiona × 2 ziarna = 40 treningów.

Receptura jest ta sama co w tickecie 04: funkcja `recipe()` w `scripts/run_xgc_g.py`, zbudowana na etapie `g_combo` z `scripts/run_v12.py`. DINOv2-L ma dostrajane cztery ostatnie bloki i głowę. Dochodzą maska overlay, random resized crop w skali 0.85–1.0, pomocnicza głowa on-target z wagą 0.5 i EMA z zanikiem 0.998. Pozostałe ustawienia: 5 epok, batch 8, lr 3e-4, mnożnik lr trzonu 0.02, limit 45 min, bez TTA. `FT` w `run_v12.py` ma batch 16, ale dla ViT-L kod obcina go do 8. Batch 8 zapisała też archiwalna konfiguracja w `results/v12_cv/g_combo`. Kod trenera z ticketu 04 zmienia tylko listy meczów.

Epokę wybiera skalibrowany log-loss na wycinku wyboru: temperatura dopasowana na tym wycinku w każdej epoce, jak w `select_on="calibrated"`. Temperaturę potem odrzucamy. Ani wycinek kalibracji, ani test nie biorą udziału w wyborze epoki.

Zapisujemy predykcje modelu na wycinkach wyboru i kalibracji oraz na teście, z rekordem ukończenia: hashe źródeł, hash planu, hash commita protokołu. Wagi nie są potrzebne do analizy i nie zostają. Ticket 12 przed pierwszym treningiem odtwarza plan z indeksu i przerywa, jeśli sha256 `plan.csv` różni się od `33229bbfd38dcbbdcd2ba33524cdaf2ab5ae766485f6ba7da08e3dda40b7bc88`.

Budżet: ok. 12 min na trening, razem ok. 8 GPU h. Treningi nie mogą nakładać się na tickety 04, 05 i 08, a na dysku zostaje co najmniej 5 GB.

## 5. Kalibracja

Kalibracja jak w archiwum (`scripts/build_ensemble.py`), ale dopasowana wyłącznie na wycinku kalibracji danego ramienia:

1. Platt na logitach modelu ze wszystkich strzałów wycinka kalibracji.
2. Korekta priora z częstości goli na wycinku kalibracji do częstości goli we wszystkich 300 meczach ramienia.
3. Karne dostają częstość z rozkładu a posteriori Beta(prior 0.76, siła 40), uaktualnionego karnymi z 300 meczów ramienia. Archiwum brało karne z oficjalnego podziału train, tu odpowiednikiem jest całe ramię.

Sam Platt widzi tylko wycinek kalibracji. Częstość docelowa w kroku 2 i karne w kroku 3 biorą wszystkie 300 meczów ramienia, także uczące, tak jak archiwum brało częstość z train+valid. To odejście od historyjki 55 specyfikacji ("calibration fitted only on matches held out"), które raport i odpowiedź dla recenzenta podają wprost. Korekta priora z wycinka do tego samego wycinka niczego by nie zmieniła, a 45 meczów to ok. 10 karnych. Efekt jest mały. Przesunięcie logitu z korekty wynosi w planie 0.0003–0.13, mediana 0.05. Przy 0.05 log-loss zmienia się o ok. 0.0001, przy 0.13 o ok. 0.001, a oba ramiona przechodzą tę samą procedurę. Test ligi L nie wpływa na kalibrację w żadnym kroku.

## 6. Metryki

Strzał testowy ma predykcję z obu ramion swojego podzbioru. Każdy mecz ligi L jest w teście dokładnie raz, bo podzbiory A i B się zamieniają.

**Metryka główna.** Łączna luka log-lossu to log-loss ramienia "bez" minus log-loss ramienia "z" na wszystkich strzałach testowych pięciu lig razem, więc ligi ważą tyle, ile mają strzałów. Liczymy ją osobno dla ziarna 0 i 1, a wynikiem głównym jest średnia z obu. Przedział daje bootstrap po meczach warstwowany po lidze: w każdej lidze losujemy z powtórzeniami tyle meczów, ile liga ma. 2 000 losowań, ziarno 0, przedział percentylowy 95%. Oba ziarna treningu dostają w danym losowaniu te same mecze. Podajemy też lukę względną, czyli lukę podzieloną przez log-loss ramienia "z", liczoną wewnątrz bootstrapu.

Odczyt ustalamy z góry. Bootstrap po meczach nie widzi szumu treningu, a tu każde ramię to jeden model, więc reguła wymaga też zgody ziaren:

- przedział nie obejmuje zera, a luki obu ziaren mają ten sam znak: "luka transferu wykazana", z wielkością i przedziałem;
- przedział nie obejmuje zera, ale ziarna mają przeciwne znaki: "wynik niejednoznaczny między ziarnami";
- przedział obejmuje zero: "luki nie wykazano", z górną granicą jako największą luką zgodną z danymi. Nie piszemy "brak luki".

Luki obu ziaren i ich różnica stoją zawsze obok wyniku głównego. Obok stoi też punkt odniesienia dla szumu treningu z raportu punktu 1. Dwie wersje G z 5 foldami, różniące się tylko ziarnem, dzielą 0.0005 log-lossu. Pojedyncze modele foldowe tracą wobec G-combo od 0.0010 do 0.0083, choć różnią się też danymi uczącymi. Tu każde ramię to jeden model, więc luka rzędu 0.001 mieści się w zwykłym rozrzucie pojedynczych treningów i tak ją opisujemy.

**Metryki dodatkowe.** Liczymy je dla każdej ligi i łącznie, dla obu ramion i jako różnicę "bez" minus "z", z tym samym bootstrapem:

- AUC, Brier, ECE (10 równych przedziałów, jak w `xgpoc.eval.metrics`);
- nachylenie i wyraz wolny kalibracji z regresji logistycznej y ~ a + b·logit(p) na strzałach z gry (bez karnych) oraz średnie xG minus częstość goli;
- luka log-lossu w każdej lidze osobno;
- łączna luka ze wszystkimi ligami ważonymi po równo;
- łączna luka na samych strzałach z gry, bez karnych, które w obu ramionach dostają tę samą stałą częstość.

Ligue 1 ma 108 goli i dawkę 19 meczów, Bundesliga 155 goli i dawkę 26–27 meczów. Ich wyniki są orientacyjne i tak je opisujemy. Wnioski opieramy na wyniku łącznym.

Nachylenie ok. 1.2 z punktu 1 sprawdzamy tak. Jeśli nachylenie ramienia "bez" na lidze testowej odbiega od 1 wyraźnie bardziej niż nachylenie ramienia "z", część tej zbyt ostrożnej skali pochodzi z przesunięcia rozkładu. Jeśli oba ramiona mają podobne nachylenie, przyczyna leży gdzie indziej.

## 7. Krzywa rekalibracji

Pytanie od strony pakietu: ile meczów nowej ligi trzeba, żeby odzyskać lukę samą rekalibracją?

- k ∈ {0, 5, 10, 20} meczów z testowego podzbioru h.
- Na podzbiór h przypada K = największe k z siatki, przy którym w h zostaje co najmniej 10 meczów do oceny. Wychodzi K = 20 dla Premier League, Serie A i LaLiga, K = 10 dla Bundesligi i K = 5 dla Ligue 1.
- Jedno losowanie to losowa permutacja meczów h. Pierwsze K meczów to kandydaci do rekalibracji, k-te ustawienie bierze ich pierwsze k. Pozostałe mecze to zbiór oceny, ten sam dla każdego k w losowaniu, także dla k = 0.
- 200 losowań na (liga, podzbiór), ziarno wyprowadzone z (0, L, h). Te same losowania dla obu ramion i obu ziaren treningu.
- Rekalibracja główna: Platt a·z + b na logitach skalibrowanych już predykcji ramienia "bez", dopasowany na strzałach z gry z k meczów. Karne zostają przy swojej częstości. Jeśli k meczów nie ma ani jednego gola wśród strzałów z gry, losowanie zostaje przy k = 0 i liczymy takie przypadki.
- Rekalibracja dodatkowa: sam wyraz wolny, a = 1.
- Wiersz dodatkowy: ramię "z" rekalibrowane tymi samymi k meczami. Pokazuje, ile daje sama rekalibracja na tej lidze, niezależnie od transferu.

Dla każdego k raportujemy lukę na zbiorze oceny: log-loss "bez" po rekalibracji minus log-loss "z" bez rekalibracji. Podajemy średnią z losowań, rozrzut 5–95% losowań i odzyskany udział 1 − luka(k)/luka(0), liczony ze średnich po losowaniach, nie w każdym losowaniu osobno. Udział podajemy tylko wtedy, gdy średnia luka(0) > 0. Luka(0) liczymy na zbiorze oceny, czyli na podzbiorze h bez K meczów kandydujących, więc nie równa się luce głównej z pkt 6. Krzywa główna łączy Premier League, Serie A i LaLiga dla wszystkich k. Każda liga osobno ma krzywą do swojego K, a wszystkie pięć lig razem pokazujemy dla k ∈ {0, 5}.

## 8. Wrażliwość na przypisanie goli

Łączną lukę i luki ligowe liczymy ponownie według reguł z ticketu 03: najbliższy strzał (reguła obecna), najwcześniejszy strzał łańcucha dobitek, ostatni strzał łańcucha, łańcuchy pominięte. Dochodzą granice. Granice z ticketu 03 dotyczą jednego modelu, więc liczymy je osobno dla każdego ramienia i podajemy log-loss obu ramion pod każdą z nich. Dla samej luki rozszerzamy tę regułę na różnicę: dla każdego gola z łańcucha wybieramy strzał, który maksymalizuje albo minimalizuje sumę różnic log-lossu "bez" minus "z" w łańcuchu, uśrednioną po ziarnach. Zmieniamy wyłącznie etykiety testowe. Etykiety uczenia, wyboru i kalibracji zostają przy regule najbliższego strzału. To ograniczenie podajemy w raporcie.

## 9. Co raportujemy

- Główny wiersz: łączna luka log-lossu, średnia z ziaren, przedział 95%, luka względna, luki dla ziarna 0 i 1.
- Tabela ligowa: log-loss, AUC, Brier, ECE, nachylenie, wyraz wolny i średnie xG minus częstość goli dla obu ramion, różnice z przedziałami, oznaczenie "orientacyjne" przy Ligue 1 i Bundeslidze.
- Krzywa rekalibracji według pkt 7.
- Wiersze wrażliwości według pkt 8.
- Wykresy niezawodności dla obu ramion na każdej lidze.
- Folder wyników z rekordem ukończenia: hashe źródeł, hash planu, hash tego protokołu.

## 10. Zastrzeżenia do raportu

- Pięć lig dzieli realizację telewizyjną najwyższego poziomu i sezony 2014–2017. Kamery, grafika i jakość obrazu są do siebie podobne. Luka zmierzona tutaj jest więc prawdopodobnie mniejsza niż dla Ekstraklasy, z inną realizacją i innymi latami.
- Ramię "z" zawiera 19–59 meczów ligi, czyli 6–20% treningu. Luka dotyczy tej dawki.
- Ramię "z" widzi te same kluby, stadiony i sezony co testowy podzbiór, bo drugi podzbiór pochodzi z tych samych rozgrywek. Luka porównuje więc ligę nigdy niewidzianą z sytuacją "widzieliśmy te właśnie kluby w tym sezonie". To górny kraniec tego, co użytkownik nowej ligi może odzyskać z własnymi danymi. Z danymi z innego sezonu odzyskałby mniej.
- Receptę dobrano na danych z wszystkich lig, a wszystkie mecze widzieliśmy wcześniej. To wynik rozwojowy.
- Oba ziarna to tylko surowa miara szumu treningu. Przedział bootstrapowy obejmuje losowość meczów, nie treningu.

## 11. Decyzje dla właściciela

1. Rola po meczu czy po gospodarzu. Plan dzieli role po meczu, zgodnie z zadaniem. Archiwalne G-combo kalibrowało na foldach po gospodarzu. Podział po gospodarzu dałby ramieniu "bez" kalibrację lepiej przygotowaną na nowy stadion i prawdopodobnie mniejszą lukę. Rekomenduję zostawić podział po meczu, bo test pyta o model zbudowany na własnych danych.
2. Ramię "bez" osobno dla A i B (40 treningów, stan obecny) czy wspólne dla ligi (30 treningów, 6 GPU h).
3. Rekalibracja w krzywej: Platt jako główna, zgodnie ze specyfikacją, a sam wyraz wolny jako dodatkowa. Przy k = 5 to ok. 13 goli i Platt może pogorszyć wynik. Jeśli właściciel woli odwrotną kolejność, zmiana zostaje zapisana przed treningiem.
4. Proporcje ról 225/30/45.
5. Krzywa rekalibracji kończy się na k = 5 dla Ligue 1 i na k = 10 dla Bundesligi, bo podzbiór ma tam 19 i 26–27 meczów, a do oceny musi zostać co najmniej 10. Pełną krzywą do k = 20 mają tylko Premier League, Serie A i LaLiga.

## 12. Odstępstwa

Zmiana po akceptacji trafia do dziennika poniżej z datą i powodem. Wpis musi wejść do commita, zanim policzymy wyniki, których dotyczy. Dostaje też wiersz w `research/quality_v13_decisions.tsv`, a wyniki objęte zmianą noszą jej numer. Nieudany trening powtarzamy raz z tym samym ziarnem i tą samą konfiguracją. Jeśli znowu się nie uda, brakującą komórkę raportujemy i nie zastępujemy jej innym ziarnem. Nie zmieniamy receptury ani hiperparametrów po obejrzeniu czegokolwiek z testu. Trening, który dojdzie do limitu 45 min, raportujemy jako odstępstwo.

### Dziennik odstępstw

Brak wpisów.
