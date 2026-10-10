# Odpowiedź na recenzję z 2026-10-02: złożoność, kontakt i przypisanie goli, transfer

Jakub Eichner, 2026-10-10. Gałąź badawcza `claude-fable/xg-poc`, runda xg-credibility.

Recenzent przysłał trzy prośby. Ten raport odpowiada na każdą, w jego kolejności.

**Wszystkie liczby w raporcie to wyniki rozwojowe.** Każdy mecz SoccerNet, na którym liczymy, widzieliśmy we wcześniejszych rundach. Żadna liczba nie pochodzi z nowych danych. Sekcje z takimi liczbami mają w tytule dopisek "wyniki rozwojowe".

Każda liczba pochodzi z pliku w folderze wyników. Tabela [Źródła i sumy kontrolne](#źródła-i-sumy-kontrolne) na końcu podaje SHA-256 każdego z tych plików. W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`, a `claims.toml` w tym folderze mówi, z którego pliku i miejsca ta liczba pochodzi. Skrypt `scripts/check_reviewer_report.py` odczytuje każdą liczbę ze źródła. Kończy się błędem przy każdej niezgodności, przy liczbie dziesiętnej bez znacznika i przy pliku, którego hash nie zgadza się z tabelą. Przedziały w nawiasach kwadratowych to percentylowe przedziały 95%<!--n:proto.ci95--> z bootstrapu po meczach.

## Co zrobiliśmy zamiast tego, o co prosił recenzent

| Prośba recenzenta | Co było możliwe | Odpowiedź |
|---|---|---|
| 1. Pokazać, czy lepsze prawdopodobieństwa ensemble'u uzasadniają jego złożoność wobec dostrajanego DINOv2-L (G-combo): wykresy kalibracji, przedziały dla różnic log-loss i Briera | wszystko, o co prosił | sekcja 1 i [raport punktu 1](../ensemble_vs_g_2026-10/REPORT.md) |
| 2. Przed dalszymi eksperymentami sprawdzić moment kontaktu i przypisanie goli, zwłaszcza w łańcuchach dobitek; model ma oceniać sytuację w chwili kontaktu, bez przecieku po kontakcie | nikt nie mógł oglądać wideo, więc żadnego ręcznego sprawdzenia | sekcja 2: automatyczna sonda kierunku z protokołem zatwierdzonym przed pierwszym wynikiem i przeliczenie wyników pod różnymi regułami przypisania goli |
| 3. Ekstraklasa jako niezależny test transferu do innej ligi | danych z Ekstraklasy nie będzie | sekcja 3: leave-one-league-out na pięciu ligach krajowych SoccerNet |

## Protokoły zatwierdzone przed wynikami

Sonda kierunku i test transferu mają polskie protokoły. Właściciel zatwierdził je, zanim policzyliśmy jakikolwiek wynik. Od zatwierdzonego commita żaden z plików się nie zmienił. Hash pliku w tym commicie, hash pliku dziś, podany w tabeli źródeł, i hash zapisany w rekordzie ukończenia wyników są identyczne. Zatwierdzenie zapisaliśmy w trackerze zadań, w ticketach 06 i 11. W samym pliku nadal stoi "do zatwierdzenia", bo poprawka zmieniłaby jego hash.

| Protokół | Commit | SHA-256 w commicie | SHA-256 w rekordzie wyników | Zatwierdzony | Pierwszy wynik |
|---|---|---|---|---|---|
| sonda kierunku, `research/xgc_direction_probe_protocol.md` | `41dfd12`, 2026-10-08 | 23c8bd25934a<!--n:proto.probe.git-->… | 23c8bd25934a<!--n:proto.probe.record-->… | 2026-10-09 | 2026-10-10; klatki z offsetem wycięte dopiero po zatwierdzeniu |
| test transferu, `research/xgc_transfer_protocol.md` | `07dc33f`, 2026-10-08 | e5ae46624703<!--n:proto.transfer.git-->… | e5ae46624703<!--n:proto.transfer.record-->… | 2026-10-09 | 2026-10-09; 40<!--n:proto.t_fits--> treningów od 14:53 do 19:40 |

Protokoły zostawiały kilka szczegółów do odczytania przy wdrożeniu. Raporty wyników wymieniają każdy z nich i sposób odczytania. Żadna reguła werdyktu się nie zmieniła.

## 1. Złożoność: ensemble wobec G-combo (wyniki rozwojowe)

Pełna odpowiedź, z wykresami kalibracji, krzywą równoważności, przedziałami dla każdej instancji i kosztem, jest w [raporcie punktu 1](../ensemble_vs_g_2026-10/REPORT.md). Tu najważniejsze liczby. Test v1.2 ma 2 229<!--n:p1.n_test--> strzałów ze 100<!--n:p1.games--> meczów. Dodatnia różnica oznacza przewagę ensemble'u.

| Metryka | Ensemble − G-combo (przewaga ensemble'u) |
|---|---|
| AUC | +0.0010 [−0.0029; +0.0049]<!--n:p1.auc--> |
| redukcja log-loss | +0.0010 [−0.0006; +0.0026]<!--n:p1.ll--> |
| redukcja Briera | +0.00022 [−0.00032; +0.00081]<!--n:p1.brier--> |

- **Nie wykazaliśmy przewagi ensemble'u.** Żaden przedział nie wyklucza zera.
- **Krzywa równoważności.** Dla każdego marginesu z siatki liczymy dwa testy jednostronne, czyli TOST. [Wykres](../ensemble_vs_g_2026-10/equivalence.png) pokazuje całą krzywą. Najmniejszy margines, przy którym równoważność zachodzi, to 0.0024<!--n:p1.margin_ll--> dla log-loss i 0.0008<!--n:p1.margin_brier--> dla Briera, z 20 000 losowań. Przy 2 000 losowań i dziesięciu ziarnach bootstrapu wychodzi 0.0024–0.0025<!--n:p1.margin_ll_range--> i 0.0007–0.0008<!--n:p1.margin_brier_range-->. Możliwy zysk ensemble'u to więc mniej niż 3.6%<!--n:p1.margin_ll_share--> przewagi G-combo nad stałą prognozą w log-loss i mniej niż 4.3%<!--n:p1.margin_brier_share--> w Brierze. Granicę podajemy zamiast werdyktu, bo wynik tego porównania znaliśmy od września, zanim spisaliśmy jakikolwiek próg.
- **Punkty odniesienia.** Dwie instancje G z 5 foldami, różniące się tylko ziarnem, dzielą 0.00052<!--n:p1.seed_gap_ll--> log-loss i 0.00019<!--n:p1.seed_gap_brier--> Briera. Sparowany przedział ich różnicy log-loss to [−0.0022; +0.0012]<!--n:p1.seed_gap_ll_ci-->, więc ta para treningów mówi niewiele. G-combo zyskuje nad stałą prognozą 0.0675<!--n:p1.skill_ll--> log-loss. Koszt to szacunek przepustowości przy batchu 16 na losowych wagach, a nie zmierzona latencja wdrożenia. G-combo ze wspólnym zamrożonym trzonem liczy ok. 35.9<!--n:p1.ms_g10--> ms na strzał, ensemble 112.6–275.2<!--n:p1.ms_ens--> ms.
- **Kalibracja.** [Krzywe niezawodności](../ensemble_vs_g_2026-10/calibration_reliability.png) obejmują cały zakres prognoz. Nachylenie kalibracji G-combo wynosi 1.19 [1.07; 1.33]<!--n:p1.slope_g10-->, ensemble'u 1.23 [1.10; 1.37]<!--n:p1.slope_ens-->. Sparowana różnica, ensemble minus G-combo, to +0.037 [+0.007; +0.067]<!--n:p1.slope_diff-->, a różnica ECE −0.0007 [−0.0093; +0.0078]<!--n:p1.ece_diff-->. Ensemble nie jest więc lepiej skalibrowany. Oba modele dają zbyt ściśnięte prognozy. Najniższe 30%<!--n:p1.low30_pct--> prognoz G-combo to 659<!--n:p1.low30_shots--> strzałów, z których padło 9<!--n:p1.low30_goals--> goli wobec 29.9<!--n:p1.low30_xg--> przewidywanych. Rekalibracja dopasowana na połowie meczów testu poprawia drugą połowę średnio o +0.0002<!--n:p1.split_half--> log-loss i pomaga w 66%<!--n:p1.split_half_share--> podziałów. Kalibrację poprawimy na świeżych meczach odłożonych do kalibracji, nie na tym teście.
- **10 czy 5 modeli foldowych.** G-combo z 10 modeli zmniejsza log-loss względem średniej dwóch instancji z 5 modelami o +0.0006 [+0.0003; +0.0009]<!--n:p1.g5_vs_g10-->, ale przedział każdej z tych instancji osobno obejmuje zero. To kompromis kosztu, nie wykazany zysk.
- **Decyzja.** Modelem docelowym zostaje G-combo bez stacku. To nie dowód, że zysku z ensemble'u nie ma. Wiemy tylko, że jeśli jest, leży poniżej granicy z krzywej równoważności.

Jak te wyniki zależą od reguły przypisania goli, pokazuje sekcja 2.6.

## 2. Moment kontaktu, przeciek po kontakcie i przypisanie goli (wyniki rozwojowe)

### 2.1 Pytanie i metoda

Kotwica strzału to czas, który podaje źródło danych. Przybliża kontakt, ale może się od niego różnić o kilkaset milisekund. Recenzent ostrzegł, że przez taki błąd klatka wejściowa modelu może pokazywać grę już po kontakcie. Wstępne kontrole AI tę obawę wspierały. Na 17<!--n:proto.ai_n--> strzałach z oznaczonym przedziałem kontaktu kotwica wypadła po przedziale 3<!--n:proto.ai_after--> razy.

Nikt nie mógł oglądać wideo, więc sprawdzenie musiało działać bez ręcznych etykiet. Sonda kierunku opiera się na prostym fakcie. O tym, czy strzał jest celny, rozstrzyga wykonanie w chwili kontaktu, a celność oznacza źródło danych. Wśród strzałów bez gola sprawdzamy, od którego offsetu względem kotwicy celność da się odczytać z klatki. Jeśli klatki kotwicy często pokazują grę po kontakcie, celność jest w nich czytelniejsza, niż wynika z trendu sprzed kontaktu. Skok w kotwicy porównujemy z trzema kontrolami. Placebo sprawdza skoki we wczesnych offsetach, które leżą na pewno przed kontaktem. Kontrola pozytywna sprawdza, że w późnych offsetach, gdzie kierunek piłki musi już być widać, odczyt wyraźnie rośnie. Test znanej odpowiedzi przesuwa kotwicę połowy strzałów o +200 ms i sprawdza, czy estymator wykrywa to przesunięcie.

- **Enkodery.** Pierwszy to G-combo wytrenowany od nowa z zapisanymi wagami. Odtwarza archiwalny model, z log-loss na teście 0.3232<!--n:g.gate_ll--> wobec 0.3242<!--n:g.gate_ll_archived--> i AUC 0.7933<!--n:g.gate_auc--> wobec 0.7923<!--n:g.gate_auc_archived-->. Drugi to ta sama receptura bez głowy celności, trzeci to zamrożony, ogólny DINOv2-L. Werdykt opiera się na sondzie liniowej na zamrożonym DINOv2-L, czyli na odczycie F. Odczyty z G-combo uczono w kotwicy, więc ich czytelność może mieć tam szczyt z samej zgodności rozkładu. Dlatego protokół z góry uczynił je tylko opisowymi.
- **Offsety.** 30 klatek od −1600 do +1000 ms, gęsto co 40 ms między −400 a +400 ms. Wszystkie klatki wycina jeden ekstraktor z tych samych klipów.
- **Populacja.** 9 307<!--n:pc.shots--> strzałów bez gola i bez rzutów karnych z 491<!--n:pc.matches--> meczów, z klatką w każdym offsecie. Wykluczyliśmy 184<!--n:pc.excluded--> strzały, czyli 1.9%<!--n:pc.excluded_pct-->, mniej niż próg zatrzymania. Każdy strzał ocenia model, który nie widział jego meczu.

Szczegóły są w raportach [krzywych czytelności](../readability_2026-10/REPORT.md) z ticketu 09 i [udziału klatek po kontakcie z werdyktem](../post_contact_share_2026-10/REPORT.md) z ticketu 10.

### 2.2 Werdykt: niekonkluzywny<!--n:pc.verdict--> (reguła 1<!--n:pc.rule-->)

Odczyt F nie przeszedł dwóch z czterech bramek protokołu. Według protokołu taki wynik nie pozwala ani stwierdzić przecieku po kontakcie, ani go wykluczyć.

| Bramka | Warunek protokołu | Wejście | Wynik |
|---|---|---|---|
| kontrola pozytywna | dolna granica Δ_F > 0 i połowa szerokości przedziału U_F ≤ 0.15<!--n:proto.half_width--> | Δ_F 0.060 [0.046; 0.074]<!--n:pc.delta_f-->, połowa szerokości 0.218<!--n:pc.u_f_hw--> | nie przechodzi<!--n:pc.gate_pos--> |
| test znanej odpowiedzi (+200 ms) | dolna granica A > 0 i 0.5 ≤ A/A_pred ≤ 1.5<!--n:proto.ka_ratio--> | A 0.027 [−0.067; 0.148]<!--n:pc.a-->, A/A_pred 0.87<!--n:pc.ratio--> | nie przechodzi<!--n:pc.gate_ka--> |
| placebo | porażka, gdy \|π\| > 0.10<!--n:proto.placebo--> i przedział wyklucza 0 | | przechodzi<!--n:pc.gate_placebo--> |
| kontrola negatywna (etykiety permutowane w meczu) | porażka, gdy wartość przekracza 0.10·Δ_F<!--n:proto.neg--> i przedział wyklucza 0 | | przechodzi<!--n:pc.gate_neg--> |

Oszacowanie, o które prosiła specyfikacja, czyli udział klatek kotwicy pokazujących grę po kontakcie, wynosi w odczycie F 0.036 [−0.221; 0.216]<!--n:pc.u_f-->. Przedział jest za szeroki, żeby coś z niego wynikało.

**Dlaczego.** Zamrożony DINOv2-L słabo czyta celność nawet długo po kontakcie. AUC sondy F wynosi 0.536 [0.525; 0.549]<!--n:rd.auc_f0--> w kotwicy i 0.611 [0.599; 0.624]<!--n:rd.auc_f800--> przy +800 ms. To mniej niż w najsłabszym scenariuszu symulacji, na której sprawdzaliśmy estymator przed zamrożeniem protokołu. Tam AUC wynosiło 0.69–0.70<!--n:proto.sim_auc-->. Przy tak małym skoku przedział U_F jest szeroki, a test +200 ms traci moc. Czytelność F zaczyna rosnąć dopiero za oknem testu, więc przewidywany efekt A_pred wynosi tylko 0.031<!--n:pc.a_pred-->. Bramki zadziałały tak, jak je zaprojektowano. Właściciel z góry przyjął wynik "niekonkluzywny" jako ważny i nie zatwierdził zastępczego odczytu na wypadek słabej sondy F.

**Co werdykt pokazuje, a czego nie.**

- Nie pokazuje, że przecieku po kontakcie nie ma. Nie pokazuje też, że jest.
- Nie daje użytecznego oszacowania udziału klatek kotwicy po kontakcie.
- Pokazuje, że potok nie tworzy fałszywych skoków, bo placebo i kontrola negatywna przeszły. Tam, gdzie odczyt jest silny, czyli w G i X, test znanej odpowiedzi wykrywa przesunięcie kotwicy. Opisuje to sekcja 2.3.
- Werdykt jest tylko raportowany i nie zmienia żadnego modelu.

### 2.3 Wyniki opisowe G i X (poza werdyktem)

Odczyt G to głowa celności G-combo, X to logit xG G-combo. Oba czytają celność lepiej niż F i przechodzą własną kontrolę pozytywną. Połowa szerokości przedziału U wynosi 0.076<!--n:pc.hw_g--> dla G i 0.120<!--n:pc.hw_x--> dla X. Test znanej odpowiedzi wykrywa w nich przesunięcie +200 ms z ilorazem A/A_pred 1.13<!--n:pc.ratio_g--> i 1.28<!--n:pc.ratio_x-->.

| Odczyt | U, udział klatek kotwicy po kontakcie | D − P w kotwicy [logit] | t50 [ms] |
|---|---|---|---|
| F (sonda, zamrożony DINOv2-L; werdykt) | 0.036 [−0.221; 0.216]<!--n:pc.u_f--> | +0.002 [−0.011; +0.015]<!--n:rd.dp0_f--> | > +400<!--n:pc.t50_f--> |
| G (głowa celności G-combo) | 0.039 [−0.040; 0.112]<!--n:pc.u_g--> | +0.007 [−0.006; +0.020]<!--n:rd.dp0_g--> | +397<!--n:pc.t50_g--> |
| X (logit xG G-combo) | 0.024 [−0.104; 0.136]<!--n:pc.u_x--> | +0.006 [−0.024; +0.038]<!--n:rd.dp0_x--> | +380<!--n:pc.t50_x--> |

D − P to różnica między strzałami celnymi i niecelnymi w kotwicy ponad trend sprzed kontaktu. t50 to offset, przy którym krzywa czytelności przechodzi połowę drogi do poziomu po kontakcie.

Punktowo U jest bliskie zera w każdym odczycie, D − P w kotwicy nie odróżnia się od trendu, a przełom t50 leży około +400 ms albo później. Gdyby kotwica często pokazywała grę po kontakcie, G i X, które kierunek piłki czytają, miałyby skok w offsecie 0. Takiego skoku nie widać. To jednak opis, nie werdykt. Gdyby F przeszedł bramki, reguła 3, czyli "przeciek nauczony tylko przez G", i tak by nie zadziałała. Dolne granice U_G i U_X leżą poniżej progu 0.10<!--n:proto.leak-->, a różnice U_G − U_F 0.002 [−0.180; 0.253]<!--n:pc.ug_uf--> i U_X − U_F −0.012 [−0.220; 0.244]<!--n:pc.ux_uf--> obejmują zero.

### 2.4 Czy głowa celności nauczyła G czytać kierunek piłki

Każdy model G uczy się z pomocniczą głową celności. Gdyby klatki kotwicy pokazywały kierunek piłki, ta głowa nagradzałaby jego odczytywanie. Kontrola bez głowy ma wagę tej głowy równą 0, a poza tym tę samą recepturę. Daje praktycznie te same krzywe:

- sondy na embeddingach: U_Gp − U_G0p = −0.008 [−0.092; 0.081]<!--n:pc.d_gp_g0p-->;
- logity xG: U_X − U_X0 = −0.003 [−0.048; 0.038]<!--n:pc.d_x_x0-->.

Po kontakcie xG kontroli reaguje na celność nawet nieco silniej. Skok Δ_X0 wynosi 0.292<!--n:pc.delta_x0-->, a Δ_X 0.258<!--n:pc.delta_x-->. Na teście kontrola ma log-loss wyższy o +0.0014 [−0.0001; +0.0029]<!--n:g0.ll--> i AUC zmienione o −0.0008 [−0.0045; +0.0027]<!--n:g0.auc-->, więc usunięcie głowy nie zmienia dokładności w mierzalny sposób. Nic nie wskazuje, że głowa celności nauczyła G czytać kierunek piłki ponad to, co robi kontrola.

### 2.5 Warstwy i klatki sondy

- **Oficjalny podział test.** Ta warstwa ma 1 820<!--n:pc.test_shots--> strzałów z 95<!--n:pc.test_matches--> meczów. U_G wynosi tu 0.225 [0.089; 0.354]<!--n:pc.test_ug-->, a U_X 0.281 [0.081; 0.445]<!--n:pc.test_ux-->, więcej niż w całej populacji. F tego nie pokazuje, U_F wynosi 0.026 [−0.606; 0.358]<!--n:pc.test_uf-->. Przegląd kodu ticketu 10 poprawił pierwszą interpretację. To nie jest skok w kotwicy. W tej warstwie okno sprzed kontaktu ma niższy i bardziej płaski poziom D. Dla G średnie D w tym oknie wynosi 0.011<!--n:rd.pre_g_test--> wobec 0.033<!--n:rd.pre_g_all--> w całej populacji. Ekstrapolowany trend P(0) leży więc nisko, a D − P podnosi się równo od −400 ms, bez narastania przy kotwicy. Samo D_G(0) nie jest podwyższone: w teście wynosi 0.058<!--n:rd.d0_g_test-->, w całej populacji 0.068<!--n:rd.d0_g_all-->. Kontrola bez głowy ma ten sam efekt, U_X0 wynosi tam 0.252<!--n:pc.test_ux0-->, więc nie pochodzi on z głowy celności. Obserwacja nie zmienia werdyktu i nie jest dowodem przecieku.
- **Łańcuchy dobitek.** To 653<!--n:pc.rb_shots--> strzały z 237<!--n:pc.rb_matches--> meczów. Skok Δ_F 0.041 [−0.011; 0.094]<!--n:pc.rb_delta_f--> nie odróżnia się od zera, więc U_F nic tu nie mówi. U_G 0.256 [−0.075; 0.486]<!--n:pc.rb_ug--> ma szeroki przedział.
- **Klatki bez cięć kamery.** Na 8 854<!--n:pc.nocut_shots--> strzałach krzywe pokrywają się z całą populacją.
- **Klatki sondy a klatki treningowe.** To sprawdzał ticket 08. Modele G uczono na JPEG-ach ze źródła, a każda klatka sondy to PNG zdekodowany z klipu, który przeszedł ponowne kodowanie z crf 26. Ten sam potok na JPEG-ach z indeksu odtwarza predykcje G-combo z treningu. Największa różnica logitu to 0.00043<!--n:os.jpeg_tv--> dla strzałów train i valid i 0.000039<!--n:os.jpeg_test--> dla testu, przy tolerancji 0.001<!--n:os.jpeg_tol-->. Na PNG w offsecie 0 logity różnią się od zapisanych średnio o 0.094<!--n:os.png_tv--> dla strzałów train i valid i o 0.083<!--n:os.png_test--> dla testu. To ta sama klatka w innym kodowaniu. Przegląd na kilku strzałach pokazał, że JPEG jest najbliższy PNG z offsetu 0, a nie z ±40 ms. Sprawdzenie całej populacji w tickecie 08 pokazało, że offset, którego logit leży najbliżej zapisanego, ma wyraźny szczyt w 0 i spada symetrycznie. Nie ma więc systematycznego przesunięcia czasu między klatkami treningowymi a kotwicą. Przesunięcie domeny jest prawie stałe między offsetami, a trend sprzed kontaktu je usuwa.

### 2.6 Wrażliwość na przypisanie goli

Łańcuch dobitek to kolejne strzały tej samej drużyny w tej samej połowie, każdy najwyżej 6 s po poprzednim, bez zmiany posiadania. SoccerNet v2 nie ma zdarzeń posiadania, więc warunek jest operacyjny. Dwa strzały łączymy, chyba że między nimi jest zdarzenie rywala, gol, rozpoczęcie gry, faul albo spalony. Obecna reguła indeksu przypisuje gola najbliższemu wcześniejszemu strzałowi tej samej drużyny w oknie 6 s. Odtworzyliśmy ją z adnotacji bez jednej różnicy. W teście są 64<!--n:p1.chains--> łańcuchy. Obejmują 130<!--n:p1.chain_shots--> z 2 229<!--n:p1.n_test--> strzałów i 26<!--n:p1.chain_goals--> z 296<!--n:p1.test_goals--> goli.

Sprawdzamy cztery reguły: najbliższy strzał, czyli obecną, najwcześniejszy strzał łańcucha, ostatni strzał łańcucha i łańcuchy wyłączone. "Ostatni" to z konstrukcji ten sam strzał co "najbliższy", bo gol kończy łańcuch. Granice skrajne dają każdego gola z łańcucha strzałowi najmniej albo najbardziej korzystnemu dla modelu. Zmieniamy tylko etykiety ewaluacji.

| Wynik | Obecna reguła | Pod innymi regułami i granicami |
|---|---|---|
| punkt 1: redukcja log-loss przez ensemble | +0.0010 [−0.0006; +0.0026]<!--n:p1.ll--> | od +0.0003<!--n:p1.attr_ll_worst--> do +0.0018<!--n:p1.attr_ll_best-->; przedział wyklucza zero tylko w granicy najkorzystniejszej dla ensemble'u, [+0.0001; +0.0034]<!--n:p1.attr_ll_best_ci--> |
| punkt 1: nachylenie kalibracji G-combo | 1.19 [1.07; 1.33]<!--n:p1.slope_g10--> | 1.00 [0.88; 1.13]<!--n:p1.attr_slope_earliest--> przy najwcześniejszym strzale |
| sonda: U_F | 0.036<!--n:pc.u_f_point--> | od 0.013<!--n:pc.attr_uf_min--> do 0.086<!--n:pc.attr_uf_max-->; Δ_F od 0.057<!--n:pc.attr_df_min--> do 0.062<!--n:pc.attr_df_max--> |
| transfer: łączna luka log-loss | +0.0041 [+0.0020; +0.0061]<!--n:tr.gap--> | od +0.0023 [+0.0001; +0.0044]<!--n:tr.attr_min--> do +0.0053<!--n:tr.attr_max-->; każdy przedział wyklucza zero |

Żaden wniosek nie zależy od reguły. Wyjątkiem jest nachylenie kalibracji ok. 1.2<!--n:p1.slope_g10_1d--> z punktu 1. Spada do 1.0<!--n:p1.attr_slope_earliest_1d-->, gdy gol należy do pierwszego strzału łańcucha. Zbyt ściśnięte prognozy G-combo na oficjalnym teście zależą więc w dużej części od tego, który strzał łańcucha dostaje etykietę gola. To pokazuje czułość, nie przyczynę.

## 3. Transfer między ligami (wyniki rozwojowe)

Danych z Ekstraklasy nie będzie, więc transfer mierzymy na pięciu ligach krajowych SoccerNet: Premier League, Ligue 1, Bundesliga, Serie A i LaLiga. Każdą ligę po kolei wyłączamy:

- ramię "bez" uczy się na 300<!--n:proto.t_matches--> meczach bez tej ligi i bez meczów Ligi Mistrzów z udziałem jej klubów;
- ramię "z" uczy się na tej samej liczbie meczów, wśród nich połowa meczów tej ligi;
- test to druga połowa ligi, potem połowy zamieniają się rolami, więc każdy mecz ligi jest w teście dokładnie raz;
- kalibracja korzysta tylko z meczów odłożonych z treningu ramienia;
- receptura G-combo, jeden model na ramię, ziarna 0 i 1, razem 40<!--n:proto.t_fits--> treningów.

Luka transferu to różnica log-loss przy tej samej liczbie meczów treningowych, więc nie miesza transferu z zyskiem z większego zbioru. Test to 8 897<!--n:tr.shots--> strzałów z 399<!--n:tr.matches--> meczów, 1 192<!--n:tr.goals--> gole. Szczegóły są w [raporcie transferu](../transfer_2026-10/REPORT.md).

**Wynik: luka transferu wykazana<!--n:tr.reading-->.** Model, który nie widział ligi ani jej klubów, ma log-loss wyższy o +0.0041 [+0.0020; +0.0061]<!--n:tr.gap-->, czyli o 1.2% [0.6%; 1.8%]<!--n:tr.rel-->. Ziarna treningu dają +0.0033<!--n:tr.seed0--> i +0.0049<!--n:tr.seed1-->. Przedział nie obejmuje zera, a oba ziarna mają ten sam znak, więc reguła protokołu daje "luka transferu wykazana". Dla porównania pojedynczy model foldowy traci wobec G-combo 0.0010–0.0083<!--n:tr.noise_single--> log-loss, a dwie instancje G z 5 foldami różnią się o 0.0005<!--n:tr.noise_g5-->.

| Liga | Strzały | Gole | Luka log-loss |
|---|---|---|---|
| Premier League | 2 025<!--n:tr.epl_shots--> | 270<!--n:tr.epl_goals--> | +0.0062 [+0.0014; +0.0111]<!--n:tr.epl--> |
| Ligue 1 (orientacyjnie) | 726<!--n:tr.l1_shots--> | 108<!--n:tr.l1_goals--> | +0.0078 [+0.0021; +0.0128]<!--n:tr.l1--> |
| Bundesliga (orientacyjnie) | 1 136<!--n:tr.bl_shots--> | 155<!--n:tr.bl_goals--> | +0.0031 [−0.0033; +0.0100]<!--n:tr.bl--> |
| Serie A | 2 339<!--n:tr.sa_shots--> | 290<!--n:tr.sa_goals--> | +0.0047 [+0.0012; +0.0083]<!--n:tr.sa--> |
| LaLiga | 2 671<!--n:tr.ll_shots--> | 369<!--n:tr.ll_goals--> | +0.0014 [−0.0023; +0.0051]<!--n:tr.ll--> |

W każdej lidze luka jest dodatnia w obu ziarnach. Najmniejsza luka pojedynczego ziarna to +0.0005<!--n:tr.min_seed-->, w LaLidze przy ziarnie 0. Ligue 1 i Bundesliga mają mało goli i małą dawkę ligi w ramieniu "z", więc są tylko orientacyjne. Wnioski opieramy na wyniku łącznym.

**Pozostałe metryki.** Różnica AUC, "bez" minus "z", wynosi −0.0068 [−0.0121; −0.0015]<!--n:tr.auc-->, różnica Briera +0.0014 [+0.0007; +0.0021]<!--n:tr.brier-->. Nachylenie kalibracji na strzałach z gry wynosi 0.88 [0.82; 0.96]<!--n:tr.slope_without--> dla ramienia "bez" i 0.95 [0.88; 1.03]<!--n:tr.slope_with--> dla ramienia "z". Nachylenia ok. 1.2<!--n:p1.slope_g10_1d--> z punktu 1 nie widać w żadnym ramieniu. Powyżej 1 jest tylko Premier League, z 1.13<!--n:tr.slope_epl_without--> w ramieniu "bez" i 1.04<!--n:tr.slope_epl_with--> w ramieniu "z". Ten test nie daje powodu, by przypisywać nachylenie z punktu 1 przesunięciu rozkładu. Sekcja 2.6 pokazuje, że zależy ono od przypisania goli.

**Krzywa rekalibracji.** Ile meczów nowej ligi trzeba, żeby odzyskać lukę samą rekalibracją? Krzywa główna łączy Premier League, Serie A i LaLigę, 4 292<!--n:tr.eval_shots--> strzały oceny. Wartości to luka wobec ramienia "z" bez rekalibracji, średnia z 200 losowań meczów. Rozrzut losowań pokazuje [wykres](../transfer_2026-10/recalibration_curve.png).

| k meczów | "bez" + Platt | odzysk | "bez" + sam wyraz wolny | odzysk | zmiana dla "z" + Platt |
|---|---|---|---|---|---|
| 0 | +0.0038<!--n:tr.rc_p0--> | | +0.0038<!--n:tr.rc_i0--> | | |
| 5 | +0.0132<!--n:tr.rc_p5--> | −248%<!--n:tr.rc_p5_rec--> | +0.0078<!--n:tr.rc_i5--> | −106%<!--n:tr.rc_i5_rec--> | +0.0123<!--n:tr.rc_w5--> |
| 10 | +0.0069<!--n:tr.rc_p10--> | −80%<!--n:tr.rc_p10_rec--> | +0.0044<!--n:tr.rc_i10--> | −17%<!--n:tr.rc_i10_rec--> | +0.0053<!--n:tr.rc_w10--> |
| 20 | +0.0040<!--n:tr.rc_p20--> | −7%<!--n:tr.rc_p20_rec--> | +0.0030<!--n:tr.rc_i20--> | +21%<!--n:tr.rc_i20_rec--> | +0.0024<!--n:tr.rc_w20--> |

Kilka meczów nowej ligi nie wystarczy, żeby rekalibracją odrobić lukę. Platt na 5 meczach kosztuje więcej niż sama luka. Ramię "z" rekalibrowane tymi samymi meczami traci prawie tyle samo, więc to szum dopasowania na małej próbie.

**Najważniejsze zastrzeżenie.** Pięć lig dzieli telewizyjną realizację najwyższego poziomu i sezony 2014–2017<!--n:proto.t_seasons-->. Dla Ekstraklasy, z inną realizacją i z innych lat, luka będzie prawdopodobnie większa.

## 4. Ograniczenia

1. **Czego sonda nie wykryje.**
   - Sonda wykrywa tylko przeciek kierunku piłki. Inne sygnały po kontakcie, na przykład reakcja bramkarza, nie muszą korelować z celnością strzałów bez gola.
   - Strzały zablokowane, zasłonięte i błędy etykiet rozmywają przełom. U to udział efektywny. Gdy czytelność narasta przez kilka klatek, U jest dolnym ograniczeniem odsetka klatek kotwicy po kontakcie. W symulacji wyszło U = 0.37<!--n:proto.eff_u--> przy rzeczywistym odsetku 0.50<!--n:proto.eff_true-->.
   - Narastanie czytelności tuż przed kontaktem, przy zamachu i ustawieniu ciała, zawyża U o 0.04–0.17<!--n:proto.windup-->. Próg reguły przecieku chroni werdykt tylko wtedy, gdy przed kotwicą nie ma kontaktów.
   - Odczyty G i X uczono w kotwicy, więc mogą mieć tam szczyt czytelności z samej zgodności rozkładu.
   - Estymator zakłada, że embedding reaguje na treść po kontakcie addytywnie, a czas kontaktu nie zależy od celności. Symulacja tego nie sprawdza.
   - W tej rundzie odczyt F był za słaby, żeby cokolwiek wykryć albo wykluczyć. Opisuje to sekcja 2.2.
2. **Przypisanie goli zmieniamy tylko w ewaluacji.** Modele uczono na etykietach reguły najbliższego strzału, więc wpływu reguły na trening nie sprawdzamy. Granice skrajne optymalizują wartość punktową, a przedziały bootstrapu liczą się przy stałych etykietach i nie obejmują niepewności samego przypisania.
3. **Pięć lig dzieli realizację.** Kamery, grafika, jakość obrazu i sezony są w nich podobne, więc luka transferu jest prawdopodobnie mniejsza niż dla Ekstraklasy. Ramię "z" widzi też kluby, stadiony i sezony testowej połowy ligi, więc luka to górny kraniec tego, co użytkownik nowej ligi odzyska własnymi danymi. Ligue 1 i Bundesliga są orientacyjne.
4. **Dane widziane wcześniej.** Każdy mecz widzieliśmy we wcześniejszych rundach, oficjalny test także. Porównanie ensemble'u z G-combo znaliśmy przed spisaniem jakiejkolwiek reguły, a recepturę G-combo dobraliśmy na walidacji ze wszystkich lig. Wszystkie liczby są rozwojowe i potwierdzić je mogą tylko nowe mecze.
5. **Szum treningu.** Przedziały bootstrapu obejmują losowość meczów, nie treningu. W teście transferu dwa ziarna to tylko surowa miara szumu treningu.
6. **Klatki sondy.** Modele G uczono na JPEG-ach, a sonda czyta PNG z klipów, co opisuje sekcja 2.5. Przesunięcie jest prawie stałe między offsetami, ale nie zerowe. Wykluczone strzały skupiają się w kilku meczach Serie A 2016/17, których klipy gubią klatki, więc populacja sondy nie jest losowym podzbiorem testu.

## 5. Czego ta runda nie rozstrzyga

Punkt 2 zostaje otwarty. Rozstrzygnięcie wymagałoby ręcznych etykiet kontaktu na próbie strzałów albo nowego, osobno zarejestrowanego odczytu, który widzi kierunek piłki lepiej niż uśredniony embedding zamrożonego DINOv2-L. W tej rundzie nie robiliśmy żadnego z nich. Do tego czasu nie zmieniamy ani kotwic, ani modeli. Opisowe wyniki G-combo nie pokazują skoku w kotwicy, ale z góry umówiliśmy się, że nie zastąpią werdyktu.

## Źródła i sumy kontrolne

Wszystkie ścieżki są względem `experiments/claude_fable`. Pliki z `results/` leżą poza repozytorium, w katalogu ignorowanym przez git. Ich hashe podajemy, żeby wynik dało się sprawdzić na maszynie, na której powstał. Folder punktu 1 nie ma rekordu ukończenia, więc jego pliki przypina tylko ta tabela. Pozostałe foldery mają `complete.json` z hashami plików. Skrypt sprawdza, że te hashe zgadzają się z tabelą.

| Plik | SHA-256 | Wynik rozwojowy | W repozytorium |
|---|---|---|---|
| `results_reports/ensemble_vs_g_2026-10/REPORT.md` | `7ce8932e3f36e01d9af4a62fd16f5590544b046bbe54fbc6f0cd4bd0a9ea5836` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/summary.json` | `586c4f01b0851b396b5aa3ed66bd413bb60eed3205df282446a33486aa4e089a` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/headline_ens_vs_g10.csv` | `6a062767c86fa6e921450943e09caa9ac3eea59c8840fad737d48db7a057b6c4` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/calibration.csv` | `d9ba090471fd3defc8ceb3f41f8310257c7e6e218e9c7c2ea2b26da7ed7800d5` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/calibration_diff.csv` | `7a0a1f9ae30ca26d48b061d64eb573accaffa99591b4495eb2f716d15fb57a34` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/ladder_vs_g10.csv` | `c947aec5a5be912f82aef252f41282f0be78a8c5f7ca8590a88d7f074f5ac239` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/attribution_sensitivity.csv` | `c19c5c1dfbb7a9bb793f605ac87325a689b27a3fcb972a94e113f7ec8e5b363e` | tak | tak |
| `results_reports/ensemble_vs_g_2026-10/attribution_recount.csv` | `ff997217e00d9b504d303007bc7f1b05f47e8ef33707ed5286ed6d78cacd113a` | tak | tak |
| `results_reports/readability_2026-10/auc.csv` | `d689049fd224846a5d2d2180d8d2f96d56098ddf145c1144f703979fcab8bc2a` | tak | tak |
| `results_reports/readability_2026-10/curves.csv` | `710ca527dfeafec64c05bc72703265cf16c986057a77be23f5d8b61d214b30dd` | tak | tak |
| `results_reports/post_contact_share_2026-10/summary.json` | `c68faad0141d7e702bde2acd04b615316034e5cf38e92737a3dbd94fd7251c50` | tak | tak |
| `results_reports/post_contact_share_2026-10/shares.csv` | `8d5a2871e376241575abc118454790358bb6d294b05cac6bfb5bdcf4edfe3a7f` | tak | tak |
| `results_reports/post_contact_share_2026-10/differences.csv` | `77a3baecae2a6969e2b202770f749b65d841270f2aed47218ec254f50171070e` | tak | tak |
| `results_reports/post_contact_share_2026-10/sensitivity.csv` | `cf5f2a3b365a9096fd95b5a370603d65c54d90c0358ef5c694dafdf32514b180` | tak | tak |
| `results_reports/post_contact_share_2026-10/complete.json` | `7d7cfcb636db0bda8aa9bcdb518fa4a5e16aa28e3cd1a40798bf9816910f969a` | tak | tak |
| `results_reports/transfer_2026-10/summary.json` | `aad5092e1b5ed1311d2c60af9b8e7186a0f11a529276b6d9b6ffcefc895e5eff` | tak | tak |
| `results_reports/transfer_2026-10/gaps.csv` | `0f9890ad80770ffd6de5b3d0f8640ebfd471479cc945fba161eb243843006b88` | tak | tak |
| `results_reports/transfer_2026-10/metrics.csv` | `82b3ef45fdc258de6af277fb7b02037ccdbe7f9d10b8a73ba4957b63ba2bd977` | tak | tak |
| `results_reports/transfer_2026-10/recalibration_curve.csv` | `55cbd21bb26972e55f5d29dad05d4335aed6aabcd0158a928473c4b181c56497` | tak | tak |
| `results_reports/transfer_2026-10/attribution_sensitivity.csv` | `d5758dd518c1ee677f849e120424dc7f4b7868c5cfd9694b120554b542d02593` | tak | tak |
| `results_reports/transfer_2026-10/complete.json` | `d472bba7b080ffb0fdccf1b068b2b5f48bd57b6e54455e59c4ed94ad7e6ef506` | tak | tak |
| `results/xgc_g_combo/gate.json` | `9c6f796921bef25bceb6755f9746960744749f982ea1d527f67d2b672865bd8a` | tak | nie |
| `results/xgc_g_combo_no_head/comparison.csv` | `88806f718e9f317476c5b25f17184b9f1ddee549742840be34d94e8c3fc03549` | tak | nie |
| `results/xgc_offset_scores/complete.json` | `bf0080e0bbdcb6763f15c5b1ec80d6e8e7f9e4cb30be0ef071503b84c2dab657` | tak | nie |
| `research/xgc_direction_probe_protocol.md` | `23c8bd25934a1fd7b63fe6513f9dd425899522641a19b068313981de7b0775a0` | nie (protokół) | tak |
| `research/xgc_transfer_protocol.md` | `e5ae46624703c4a2c6b520aa70d446f6f07a127b906913fdb41dba2daa077c0e` | nie (protokół) | tak |

Sprawdzenie:

```bash
cd experiments/claude_fable
./run.sh python scripts/check_reviewer_report.py results_reports/reviewer_response_2026-10/REPORT.md \
    --reply results_reports/reviewer_response_2026-10/reply.md
```

Każdy folder wyników ma w swoim `REPORT.md` polecenia, które odtwarzają jego liczby.
