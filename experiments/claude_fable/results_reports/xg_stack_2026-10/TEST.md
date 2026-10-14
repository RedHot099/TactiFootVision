# Test stackera xg-stack na meczach odłożonych

Data: 2026-10-14. Faza xg-stack, ticket 04, decyzja rady C7. Reguła testu: `research/xg_stack_test_rule.md` (parametry w `research/xg_stack_test_rule.json`). Lista skrótów zamrożona przed odczytem etykiet: `research/xg_stack_test_hashes.json`, SHA-256 `1dbfad14d288592721f2c56dedb8ffb5c2a94d3e89f373dcdb28f531e10b174b`. Raport fazy z wynikami rozwojowymi leży obok (`REPORT.md`).

**Zakres** (zdanie reguły): wynik dotyczy stackera C6 nad zamrożonym G-combo z geometrią E3 w postaci zaudytowanej w fazie pitch-keypoints, a nie struktury z detektora w ogóle.

## 1. Odpowiedź (wynik testu)

Zdanie zapisane w regule przed testem dla etykiety „equivalent within δ”, z liczbami testu:

> Test na 172<!--n:r.matches--> meczach odłożonych nie wykazał poprawy ani pogorszenia: ΔLL(G − R) = −0.00094 [−0.00211; +0.00022]<!--n:t.g.ci--> (95%<!--n:t.level--> przedział). 90%<!--n:r.tost--> przedział [−0.00192; +0.00004]<!--n:t.g.ci90--> mieści się w ±0.002<!--n:r.margin-->, więc G i R są równoważne w granicy δ = 0.002<!--n:r.margin--> (TOST, odczyt opisowy).

- **Etykieta:** „equivalent within δ”. Reguła sukcesu (górna granica przedziału 95%<!--n:t.level--> poniżej 0) nie jest spełniona. Pogorszenia też nie ma.
- **Liczby:** 3 819<!--n:t.shots--> strzałów w 172<!--n:t.matches--> meczach, 488<!--n:t.goals--> goli. Log-loss G wynosi 0.33687<!--n:t.G.ll-->, a R 0.33781<!--n:t.R.ll-->.
- **Równoważność:** przy δ = 0.005<!--n:r.also--> (marginesie testu A1) G i R też są równoważne. Najmniejszy margines, który obejmuje przedział 90%<!--n:r.tost-->, to 0.00192<!--n:t.margin-->.
- **Próg praktyczny:** punkt −0.00094<!--n:t.g.point--> leży powyżej progu −0.002<!--n:r.prac-->.
- **Moc:** moc obliczona przed testem dawała sukces z prawdopodobieństwem 0.197<!--n:b.pw.head--> (sekcja 3.3), więc brak sukcesu był wynikiem oczekiwanym.
- **Mecze zużyte:** 172<!--n:r.matches--> mecze odłożone są teraz zużyte dla wariantów wejść detektora (sekcja 6).
- **Odczyty opisowe** (sekcja 5) nie mogą zmienić tej odpowiedzi ani jej uratować.

## 2. Co testowano

**Polecenie właściciela.** 2026-10-14 właściciel napisał: „Zrób test na całym zbiorze meczy”. To polecenie zniosło zasadę jednego testu, ale tylko dla tego jednego testu. Rada C7 ustaliła, jak test przebiega. Reguła została zamrożona przed odczytem jakiejkolwiek etykiety testowej.

**Warianty.** Wariant potwierdzający jest jeden: **G**. To stacker C6 na zamrożonym logicie G-combo. Wejścia:
- `z_g_combo`;
- dystans E3 jako naturalny spline z 3 stopniami swobody;
- kąt E3 (zawinięty do [0, π]);
- wskaźnik braku pozycji strzału;
- wskaźnik braku piłki.

Punkt odniesienia **R** to `z_g_combo` bez kary. **GS** (G plus flagi SoccerNet) jest tylko opisowy.

**Dopasowanie końcowe.** R, G i GS dopasowano raz, na wszystkich 7 062<!--n:b.train.shots--> strzałach treningowych z 328<!--n:b.train.matches--> meczów:
- karę C wybrano walidacją krzyżową grupowaną po meczach, na 5 foldach gospodarzy;
- w G wybrane C = 0.003<!--n:c.G.c-->, w GS C = 0.001<!--n:c.GS.c-->, czyli dolna krawędź siatki, którą reguła dopuszcza;
- R jest bez kary;
- imputacja, węzły spline'u i standaryzacja pochodzą tylko z danych treningowych.

**Populacja.** Wszystkie strzały poza karnymi z 172<!--n:r.matches--> meczów odłożonych: 3 819<!--n:r.shots--> strzałów. To 3 005<!--n:r.shots.eval--> strzałów w 135<!--n:r.eval--> meczach ewaluacyjnych i 814<!--n:r.shots.gate--> w 37<!--n:r.gate--> meczach bramki. Ta sama populacja była w teście A1.

**Reguła testu (A1).**
- ΔLL = LL(G) − LL(R), log-loss ważony strzałami, prawdopodobieństwa przycięte na 1e−06<!--n:r.clip-->.
- Sparowany bootstrap meczów: 5 000<!--n:r.draws--> losowań, ziarno 0.
- **Sukces:** górna granica przedziału 95%<!--n:r.lvl--> poniżej 0 (poziom w mocy po bramce z sekcji 3.2).
- **Pogorszenie:** dolna granica powyżej 0.
- **Równoważność (opisowo):** TOST na przedziale 90%<!--n:r.tost-->, przy δ = 0.002<!--n:r.margin--> (wyznacza etykietę) i δ = 0.005<!--n:r.also-->.

## 3. Przed odczytem etykiet

Wszystko w tej sekcji zapisano, zanim przeczytano jakąkolwiek etykietę testową. Etykiety treningowe czytano wyłącznie wewnątrz pieczęci etykiet. Danych StatsBomb nie czytano wcale.

### 3.1 Kontrole bez etykiet

| Kontrola | Wynik | Warunek zatrzymania |
|---|---|---|
| odwrócenie Platta | największa różnica z logitami członków 2.7e−09<!--n:c.zinv--> na 1 098<!--n:c.zinv.n--> strzałach | powyżej 1e−06<!--n:c.zinv.tol--> |
| rozkład z: stosunek średnich (test / trening) | 0.982<!--n:c.zr.mean--> | poza [0.8; 1.25]<!--n:c.zr.bounds--> |
| rozkład z: stosunek odchyleń | 0.966<!--n:c.zr.sd--> | jak wyżej |
| pokrycie | 3 819<!--n:c.cov.shots--> strzałów, 172<!--n:c.cov.matches--> mecze, złączenia jeden do jednego | każde odstępstwo |
| pochodzenie | rozłączność z treningiem, kolumny zgodne z protokołem C6, flagi niezależne od „Goal” | każde odstępstwo |
| powtórka E3 | identyczna z cache | każda różnica |
| zawinięcie kąta | 9 strzałów przy t0, 8 przy −400 ms | tylko raport |

Wszystkie kontrole przeszły. Żadna nie była blisko swojego warunku zatrzymania.

### 3.2 Bramka syntetycznych zer

Wylosowano 5 000<!--n:b.gate.reps--> wektorów y* ~ Bernoulli(p_R) z losowym wyrazem wolnym meczu o SD 0.25<!--n:r.gate.sd--> (ziarno 301<!--n:b.gate.seed-->), na zapisanych predykcjach testowych. Reguła G dała fałszywy sukces w 0.0006<!--n:b.gate.fs--> przypadków przy poziomie 95%<!--n:r.lvl-->, czyli poniżej progu 0.05<!--n:r.gate.max-->. Poziom sukcesu pozostał więc 95%<!--n:b.gate.level--> i drabina 97.5%<!--n:r.ladder.0-->, 99%<!--n:r.ladder.1-->, 99.5%<!--n:r.ladder.2--> nie zadziałała.

### 3.3 Moc oczekiwana, policzona przed testem (wyniki rozwojowe)

Przybliżenie normalne. SD rozwojowego ΔLL dla G to 0.00043<!--n:b.dev.sd-->, a punkt rozwojowy −0.00065<!--n:b.dev.point-->. Odzysk C6 to 0.793<!--n:b.recovery--> efektu wszczepionego.

| Scenariusz SE | SE | P(sukces) w punkcie rozwojowym | P(sukces) przy −0.002<!--n:r.prac--> × odzysk | P(sukces) przy −0.002<!--n:r.prac--> |
|---|---|---|---|---|
| SD × √(N_dev / N_test), wiodący | 0.00058<!--n:b.se.shot--> | **0.197<!--n:b.pw.head-->** | 0.776<!--n:b.pw.shot.rec--> | 0.929<!--n:b.pw.shot.prac--> |
| SD × √(M_dev / M_test) | 0.00059<!--n:b.se.match--> | 0.192<!--n:b.pw.match.dev--> | 0.763<!--n:b.pw.match.rec--> | 0.922<!--n:b.pw.match.prac--> |
| SE testu A1 | 0.00108<!--n:b.se.a1--> | 0.086<!--n:b.pw.a1.dev--> | 0.311<!--n:b.pw.a1.rec--> | 0.456<!--n:b.pw.a1.prac--> |

To liczby planistyczne. SD rozwojowe pochodzi z dopasowań poza foldem na około czterech piątych strzałów treningowych, a dopasowanie końcowe używa wszystkich; stacker odzyskuje tylko część wszczepionego efektu; SE testu A1 dotyczy innego porównania (A1 wobec G-combo) i jest pesymistyczną wrażliwością.

- Przy punkcie rozwojowym P(równoważność w δ = 0.002<!--n:r.margin-->) wynosiła 0.748<!--n:b.pw.eq-->. Etykieta „equivalent within δ” była więc najbardziej prawdopodobnym odczytem.
- Przy −0.002<!--n:r.prac--> moc sukcesu wynosiła 0.929<!--n:b.pw.shot.prac-->. Efekt na poziomie progu praktycznego test by prawdopodobnie wykrył.

## 4. Wynik testu: odczyt główny

| Porównanie | ΔLL [95%<!--n:t.level-->] | przedział 90%<!--n:r.tost--> | etykieta |
|---|---|---|---|
| G − R | −0.00094 [−0.00211; +0.00022]<!--n:t.g.ci--> | [−0.00192; +0.00004]<!--n:t.g.ci90--> | equivalent within δ |

- **SD bootstrapu:** 0.00059<!--n:t.g.sd-->, zgodnie z przewidywaniem sprzed testu (0.00058<!--n:b.se.shot-->).
- **Równoważność:** przedział 90%<!--n:r.tost--> mieści się zarówno w ±0.002<!--n:r.margin-->, jak i w ±0.005<!--n:r.also-->.

## 5. Wynik testu: odczyty opisowe (nie zmieniają wyniku)

Żaden z tych odczytów nie jest sukcesem i żaden nie ratuje ani nie zmienia odpowiedzi z sekcji 1. Każdy jest **hipotezą sformułowaną na tych meczach**.

### 5.1 Pozostałe porównania i miary (hipoteza sformułowana na tych meczach)

| Porównanie (opisowe) | ΔLL [95%<!--n:t.level-->] | przedział 90%<!--n:r.tost--> |
|---|---|---|
| GS − R | −0.00086 [−0.00185; +0.00011]<!--n:t.gs.ci--> | [−0.00169; −0.00004]<!--n:t.gs.ci90--> |
| G − wdrożony G-combo | −0.00076 [−0.00207; +0.00052]<!--n:t.gc.ci--> | [−0.00187; +0.00032]<!--n:t.gc.ci90--> |
| R − wdrożony G-combo (dodane dla odniesienia, poza listą C7) | +0.00017 [−0.00037; +0.00073]<!--n:t.rc.ci--> | [−0.00028; +0.00064]<!--n:t.rc.ci90--> |

| Model | log-loss | AUC | Brier | nachylenie kalibracji |
|---|---|---|---|---|
| R | 0.33781<!--n:t.R.ll--> | 0.742<!--n:t.R.auc--> | 0.1000<!--n:t.R.brier--> | 1.11<!--n:t.R.slope--> |
| G | 0.33687<!--n:t.G.ll--> | 0.744<!--n:t.G.auc--> | 0.0997<!--n:t.G.brier--> | 1.11<!--n:t.G.slope--> |
| GS | 0.33694<!--n:t.GS.ll--> | 0.744<!--n:t.GS.auc--> | 0.0998<!--n:t.GS.brier--> | 1.11<!--n:t.GS.slope--> |
| wdrożony G-combo | 0.33763<!--n:t.C.ll--> | 0.742<!--n:t.C.auc--> | 0.0999<!--n:t.C.brier--> | 1.05<!--n:t.C.slope--> |

- **GS** jest opisowy i nie ma etykiety sukcesu. Jego przedział 90%<!--n:r.tost--> leży poniżej 0, ale przedział 90%<!--n:r.tost--> nie jest regułą sukcesu.
- **Nachylenie kalibracji** jest powyżej 1 we wszystkich modelach (interpretacja: na tych meczach predykcje są nieco zbyt ostrożne).

### 5.2 Mecze ewaluacyjne i mecze bramki (hipoteza sformułowana na tych meczach)

| Mecze | strzały | ΔLL(G − R) [95%<!--n:t.level-->] |
|---|---|---|
| ewaluacyjne | 3 005<!--n:t.ev.n--> | −0.00091 [−0.00219; +0.00038]<!--n:t.ev.ci--> |
| bramka | 814<!--n:t.gt.n--> | −0.00106 [−0.00388; +0.00147]<!--n:t.gt.ci--> |

Kierunek jest ten sam w obu grupach.

### 5.3 Kategorie E3 (hipoteza sformułowana na tych meczach)

O/E i ΔLL(G − R) w kategoriach z fazy rozwojowej. Przedziały pochodzą z tych samych losowań, ograniczonych do strzałów kategorii. Kategorie E3 są określone na 3 687<!--n:k.def--> strzałach: 132<!--n:c.miss--> strzały bez pozycji strzału E3 przy t0 nie należą ani do kategorii, ani do jej dopełnienia.

| Kategoria | strzały | O/E R | O/E G | ΔLL(G − R) [95%<!--n:t.level-->] |
|---|---|---|---|---|
| E3 poza polem karnym | 1 356<!--n:k.box.n--> | 0.95<!--n:k.box.R--> | 1.01<!--n:k.box.G--> | −0.00031 [−0.00146; +0.00090]<!--n:k.box.d--> |
| E3 kąt < 15° | 1 360<!--n:k.ang.n--> | 0.69<!--n:k.ang.R--> | 0.75<!--n:k.ang.G--> | −0.00162 [−0.00300; −0.00016]<!--n:k.ang.d--> |
| E3 dystans < 6 m | 304<!--n:k.dst.n--> | 1.23<!--n:k.dst.R--> | 1.04<!--n:k.dst.G--> | −0.00616 [−0.01829; +0.00523]<!--n:k.dst.d--> |
| rzut rożny w ostatnich 20 s | 484<!--n:k.cor.n--> | 0.95<!--n:k.cor.R--> | 0.92<!--n:k.cor.G--> | −0.00035 [−0.00476; +0.00375]<!--n:k.cor.d--> |
| bezpośredni rzut wolny | 274<!--n:k.fk.n--> | 0.82<!--n:k.fk.R--> | 0.82<!--n:k.fk.G--> | −0.00195 [−0.00727; +0.00281]<!--n:k.fk.d--> |

- **Kierunek jak w fazie rozwojowej:** G przesuwa O/E ku 1 poza polem karnym i przy dystansie < 6 m.
- **Kąt < 15°:** tu ΔLL ma przedział poniżej 0. To jedna kategoria z pięciu, bez korekty na wielokrotność, więc to hipoteza, nie wynik.
- **O/E przy kącie < 15°** pozostaje daleko od 1 w obu modelach (0.75<!--n:k.ang.G--> w G).

### 5.4 Zgodność ze StatsBomb na meczach ewaluacyjnych (hipoteza sformułowana na tych meczach; Data: StatsBomb.)

Jedno spojrzenie, wykonane po zapisaniu wyniku testu (`scripts/xgs_test_statsbomb.py`). Populacja to 2 721<!--n:s.shots--> sparowanych strzałów ze 135<!--n:r.eval--> meczów ewaluacyjnych, wzięte z tabeli xg-compare. StatsBomb nie jest wejściem, celem ani kryterium żadnego modelu.

**Korelacja Spearmana z xG dostawcy:**
- R: 0.679<!--n:s.rho.R-->;
- G: 0.688<!--n:s.rho.G-->;
- wdrożony G-combo: 0.679<!--n:s.rho.C--> (dodane dla odniesienia, poza listą C7).

R i G-combo mają tę samą kolejność, bo oba są rosnącą funkcją tego samego z.

**Luki w kategoriach xg-compare, w punktach xG.** Luka to 100 × (średnie p modelu − średnie xG dostawcy) w kategorii, liczona we współrzędnych StatsBomb. Kategoria „dystans < 6 yd” jest odrębna od kategorii E3 „dystans < 6 m”.

| Kategoria (StatsBomb) | strzały | luka R | luka G | luka G-combo | kontrast luki logitów R / G |
|---|---|---|---|---|---|
| poza polem karnym | 1 055<!--n:s.box.n--> | +2.7<!--n:s.box.R--> | +2.6<!--n:s.box.G--> | +2.6<!--n:s.box.C--> | 0.29<!--n:s.box.lcR--> / 0.25<!--n:s.box.lcG--> |
| kąt bramki < 15° | 614<!--n:s.ang.n--> | +6.1<!--n:s.ang.R--> | +5.8<!--n:s.ang.G--> | +6.1<!--n:s.ang.C--> | 1.06<!--n:s.ang.lcR--> / 1.01<!--n:s.ang.lcG--> |
| dystans < 6 yd | 71<!--n:s.dst.n--> | −16.9<!--n:s.dst.R--> | −16.7<!--n:s.dst.G--> | −15.3<!--n:s.dst.C--> | −1.42<!--n:s.dst.lcR--> / −1.41<!--n:s.dst.lcG--> |

- G zmniejsza luki xg-compare tylko nieznacznie.
- Model z klatki nadal przeszacowuje strzały pod ostrym kątem i niedoszacowuje strzały z bliska.
- Hipoteza sformułowana na tych meczach. Data: StatsBomb.

## 6. Zastrzeżenia

- **172<!--n:r.matches--> mecze odłożone są teraz zużyte dla wariantów wejść detektora.** To ich drugie spojrzenie na etykiety dla wariantu detektora; pierwszym był A1 w fazie xg-detector. Żaden kolejny wariant wejść detektora nie może być na nich testowany.
- Właściciel zniósł zasadę jednego testu wyłącznie dla tego testu.
- Konfigurację E3 wybrano na 135<!--n:r.eval--> z 172<!--n:r.matches--> tych meczów, według błędu pozycji względem StatsBomb, bez wyników meczów.
- Faza xg-compare patrzyła na wyniki tych meczów (eksploracyjnie). Zainspirowała wejścia i częściowo okna flag.
- Etykiety meczów ewaluacyjnych były widziane w xg-vendor. Mecze bramki ocenił G-combo w bramce retreningu xg-vendor.
- **z testowe i treningowe są różnej natury.** z testowe to średnia z dziesięciu członków G-combo, a z treningowe z dwóch. Kontrola rozkładu z (sekcja 3.1) ogranicza wynikające stąd przesunięcie skali.
- `z_g_combo` to uśredniony logit sprzed Platta. Epokę i temperaturę każdego członka wybrano na jego własnym foldzie (krok zależny od wyników, poprawione ujawnienie 3 rady C6).
- Flagi SoccerNet są wyłącznie badawcze, bo produkt potrzebowałby detektora akcji. GS jest opisowy.
- Bootstrap meczów losuje mecze dla ustalonych, dopasowanych modeli. Nie obejmuje zmienności samego dopasowania końcowego.
- W GS wybrane C leży na krawędzi siatki. Reguła to dopuszcza.
- Pochodzenie stackowania: wartości z treningowe pochodzą z modeli dopasowanych na pozostałych foldach treningowych. Dopasowanie końcowe używa wszystkich 7 062<!--n:b.train.shots--> strzałów treningowych, więc jego koszt estymacji jest niższy niż w fazie rozwojowej.

## 7. Wynik testu: interpretacja (oznaczona jako interpretacja)

- **Kierunek się powtórzył.** Na meczach odłożonych G daje ΔLL tego samego znaku i podobnej wielkości co w fazie rozwojowej (sekcja 8).
- **Brak dowodu poprawy.** Reguła sukcesu nie przeszła, więc test nie wykazał poprawy. Przedział 95%<!--n:t.level--> sięga −0.00211<!--n:t.g.lo-->, czyli za próg praktyczny −0.002<!--n:r.prac-->: test nie wykluczył poprawy na poziomie progu.
- **Równoważność jest odczytem opisowym.** Przedział 90%<!--n:r.tost--> mieści się w ±0.002<!--n:r.margin--> (TOST, najmniejszy margines 0.00192<!--n:t.margin-->), ale to nie jest wykluczenie efektu. Pokrycie granicy jednostronnej w symulacjach C6 było poniżej progu, więc raport nie używa sformułowania „brak poprawy większej niż X”.
- **Wniosek produktowy.** Test nie wykazał praktycznej korzyści z geometrii E3 dołożonej do G-combo w tej postaci.
- **Gdzie szukać dalej.** Odczyty opisowe kategorii (sekcja 5.3) i zgodności ze StatsBomb (sekcja 5.4) wskazują te same słabe miejsca co xg-compare: strzały pod ostrym kątem i z bliska (hipoteza sformułowana na tych meczach; Data: StatsBomb.). Kolejną dźwignią pozostaje percepcja: identyfikacja bramkarza, piłka i moment strzału.

## 8. Kontekst: wyniki rozwojowe

Dla porównania, przebieg rozwojowy (ticket 02 fazy xg-stack, poza foldem, 7 062<!--n:d.shots--> strzały treningowe): G ΔLL −0.00065 [−0.00151; +0.00020]<!--n:d.g.ci-->, GS −0.00048 [−0.00138; +0.00036]<!--n:d.gs.ci-->.

## 9. Otwarte punkty dla właściciela

- Zamknięcie linii stackera nad G-combo.
- Następny krok po stronie percepcji (bramkarz, piłka, moment strzału).
- Przegląd thermo-nuclear.

## Źródła i sumy kontrolne

| Plik | SHA-256 | Zawartość |
|---|---|---|
| `research/xg_stack_test_rule.json` | `02c4356b5e0d57c041ca9639d992f320a9f6ca053eaea3cae4a4a6c1f3cb9c78` | parametry reguły testu (C7) z blokiem sprzed odczytu: moc, bramka, poziom sukcesu |
| `results/xgs_test_report/readings.json` | `d6441534e1f5ae3395fb72a728202558647e2de4e9a534e3676acc0222a006a0` | wynik testu i spojrzenia StatsBomb, spłaszczone przez `scripts/xgs_test_report.py` z `results/xgs_test/result.json` i `results/xgs_test_statsbomb/agreement.json` |
| `results/xgs_test_report/before_scoring.json` | `182483ac30935374f28056ebfa58d2060bc288fd05a5f31fd46381b4b92f0633` | bramka i moc, zapisane przed testem (`results/xgs_test_prep/gate.json`, `power.json`), spłaszczone |
| `results/xgs_test/categories.csv` | `d49edf3bac54356e1dc7206443d8407d592407935f3e182ce1e9eceba436af03` | kategorie E3 testu (opisowe) |
| `results/xgs_test_prep/checks.json` | `615e64007d6fe16aa65a73c51f231ef3e59da2bcd789b91266c16b44ee082bab` | kontrole bez etykiet |
| `results/xgs_test_prep/final_fit.json` | `83eef14db2cb056f81086548d7961eb77b5bddf876439d4333048658f90ac08b` | dopasowanie końcowe na danych treningowych |
| `results/xgs_test_statsbomb/categories.csv` | `22cdfa42ea6081bc5af4a65f4735659426a7147904c87bd55d4570a67a0ade09` | luki w kategoriach StatsBomb (Data: StatsBomb.) |
| `results/xgs_stack/real/summary.json` | `e3245585febac3998900df55d51d3fab2035d54a53878dff984019d0e2a9234b` | przebieg rozwojowy (ticket 02) |
