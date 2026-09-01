# Co dalej z jakością xG — diagnoza i plan

Data: 2026-09-01. Stan wyjściowy: ensemble `dl_mask + finetune L/12 + GBM`,
**AUC 0.758 [0.734, 0.781], BSS 11.7%, log-loss 0.340, ECE 0.020** na 2 229 strzałach
ze 100 meczów testowych. Dokument porządkuje, *gdzie* konkretnie tracimy jakość i
które ruchy mają realną szansę to zmienić.

Wszystkie liczby poniżej pochodzą z artefaktów w `results_reports/figures_premerge/`
i z historii treningów w `results/premerge/`.

---

## 1. Diagnoza główna: to nie jest problem pojemności modelu

Historia walidacji fine-tuningu DINOv2-L (`results/premerge/g_final/*/seed0`):

| epoka | train loss | valid log-loss | valid AUC |
|---|---|---|---|
| 0 | 0.434 | 0.932 | 0.654 |
| **1** | 0.436 | **0.394** | 0.700 |
| 2 | 0.396 | 0.489 | **0.710** |
| 3 | 0.377 | 0.419 | 0.710 |
| 4 | 0.357 | 0.571 | 0.708 |
| 5 | 0.309 | 0.769 | 0.672 |

**Najlepszy model powstaje po 1–2 epokach i od epoki 3 tylko się psuje**, podczas gdy
strata treningowa spada dalej (0.44 → 0.26). Wynik 0.758 AUC pochodzi więc z modelu,
który praktycznie nie zdążył się nauczyć, zanim zaczął zapamiętywać. To samo w C
(`c_mask`: train loss 0.259 po 8 epokach) i, w łagodniejszej formie, w zamrożonym D.

Konsekwencja dla planowania: **wąskim gardłem jest liczba etykiet, a nie architektura.**
8 657 strzałów treningowych i 1 095 goli to za mało dla 56 M trenowalnych parametrów.
Każdy pomysł typu „większy backbone", „wyższa rozdzielczość", „model wideo" dokłada
pojemności do problemu, który polega na jej nadmiarze. Kolejność prac musi to odwracać:
najpierw **zwiększyć efektywną liczbę nadzorowanych przykładów i zmniejszyć efektywną
pojemność**, dopiero potem dokładać informacji wejściowej.

## 2. Budżet błędu — gdzie leżą te 758 natów

Test to 2 229 strzałów × log-loss 0.340 = **758 natów**. Względem częstości bazowej
liczonej w obrębie każdego kubełka model odzyskuje **~105 natów** (14%). Rozkład:

| podgrupa | n | udział w stracie | naty odzyskane vs. baza kubełka |
|---|---|---|---|
| geometria niedostępna | 519 | 27.0% | 28.6 |
| 16–25 m | 586 | 25.3% | 30.1 |
| **11–16 m** | **290** | **16.1%** | **4.8** |
| 25+ m | 506 | 16.1% | 24.9 |
| 6–11 m | 208 | 9.7% | 9.0 |
| 0–6 m | 120 | 5.8% | 8.4 |
| **karne** | **30** | 2.5% | **−2.7** |

Dwie rzeczy wychodzą natychmiast:

**(a) Pas 11–16 m jest martwym polem.** AUC w tym kubełku: C 0.623, D 0.614, G 0.655,
ENSEMBLE 0.646 — wobec 0.81–0.82 w kubełkach 0–6 m i 25+ m. Model odzyskuje tam 4.8 nata
ze 122. To jest strefa „róg pola karnego": dystans przestaje różnicować, a decyduje
ustawienie bramkarza, presja obrońcy i część ciała. Innymi słowy — **dyskryminacja
naszych modeli pochodzi głównie z rozpoznania typu sytuacji i odległości, a nie z
rzeczywistego zrozumienia szczegółu.** To najostrzejszy dostępny dowód na to, czego
brakuje.

**(b) Na karnych model jest gorszy od stałej.** 30 karnych, realna konwersja 0.767,
średnie xG: C 0.304, D 0.288, G 0.333, ENSEMBLE 0.574. Ujemne 2.7 nata oznacza, że
predykcja „częstość karnych z treningu = 0.628" pobiłaby model. `is_penalty` jest znane
na wejściu (jest już w cechach GBM), więc to jest strata do usunięcia zerowym kosztem.

## 3. Niewykorzystane zasoby, które już mamy na dysku

**(a) Etykieta „strzał celny" — 4× gęstszy sygnał, w ogóle nieużywana.**
Kolumna `label` w indeksie:

| label | nie-gol | gol |
|---|---|---|
| Shots on target | 4 334 | 1 462 |
| Shots off target | 5 205 | 29 |

Czyli P(gol \| celny) = 0.252, P(gol \| niecelny) = 0.0055. Cel „celny" ma 52.5%
pozytywów zamiast 13.3% — przy 8.7 tys. przykładów treningowych to jest największa
dostępna zmiana proporcji sygnału do parametrów. Uwaga metodologiczna: `label` jest
**wynikiem**, nie wejściem — wolno go użyć wyłącznie jako celu treningowego.

**(b) Siedem klatek na strzał** (`-1600 … +800`, 8.8 GB) i **klipy 8 s** (18 GB).
Krytyczna pułapka: `cut_clip(span_s=4.0)` tnie okno **[t−4 s, t+4 s]**, więc druga
połowa każdego klipu oraz klatki `+400`/`+800` zawierają rozstrzygnięcie (piłka w siatce,
obrona, reakcja). Każde ich użycie to jawny wyciek etykiety. Wolno korzystać wyłącznie
z `t ≤ 0`. To wymaga zapisania w kodzie jako twardego ograniczenia, bo „wczytaj wszystkie
klatki" jest tu najbardziej naturalnym i najbardziej błędnym odruchem.

**(c) `grass_ratio` bywa zepsute.** Kubełek „framing q1" (650 strzałów, 35% całej straty,
najgorsze AUC 0.713) miał być „ciasny kadr". Przegląd 24 klatek o najniższym
`grass_ratio` pokazuje co innego: to szerokie ujęcia z kamery głównej wycelowane w bramkę
przy trybunie, a w kilku przypadkach **detektor murawy zwraca 0.04 dla klatki niemal w
całości zielonej** (ciemne/wieczorne oświetlenie, nietypowy odcień). Podgrupa jest więc
mieszaniną „dużo publiczności w kadrze" i „awaria detektora" — wnioski z niej są obecnie
niepewne i wymagają naprawy metryki przed interpretacją.

## 4. Sufit pomiaru — czego i tak nie zobaczymy

95% CI dla AUC na teście: **±0.024**. Sparowany bootstrap grupowany po meczach wykrywa
różnice od ok. **0.019 AUC** (tyle wynosiła istotna różnica G vs. D: +0.0192 [+0.0006, +0.0366]).

To znaczy, że **każde usprawnienie poniżej ~0.02 AUC jest dla nas nieodróżnialne od szumu.**
Widać to zresztą w tabeli ensembli: dziesięć czołowych kombinacji mieści się w przedziale
0.3815–0.3823 walidacyjnego CV log-loss i 0.756–0.759 AUC — dalsze przebieranie w składzie
ensembla to już przebieranie w szumie.

Wniosek praktyczny: albo gonimy wyłącznie duże ruchy, albo najpierw **poprawiamy sam
przyrząd pomiarowy** (punkt M1 poniżej). Bez tego drugiego połowa planu jest niesprawdzalna.

---

## 5. Plan działań, w kolejności

### Poziom 0 — przyrząd (warunek wstępny dla reszty)

**M1. Walidacja krzyżowa 5-fold grupowana po meczach na train+valid.**
Dziś porównujemy modele na jednym zbiorze walidacyjnym (2 144 strzały, 100 meczów), a
wybór najlepszej epoki na tym samym zbiorze — stąd obserwowana wcześniej niestabilność
selekcji (konfiguracja najlepsza na teście miała najgorszy log-loss walidacyjny). CV po
400 meczach daje 4× więcej danych do porównań i uczciwe out-of-fold predykcje do
ensemblowania. Test pozostaje nietknięty do jednego finalnego pomiaru.
*Koszt:* 5× czas treningu. *Zysk:* zdolność odróżnienia zmian rzędu 0.01 AUC.

### Poziom 1 — atak na głód nadzoru (najwyższy oczekiwany zysk na godzinę GPU)

**P1. Faktoryzacja przez „celny strzał".** Dwie wersje do porównania:
- *multi-task*: wspólny backbone, dwie głowy, `L = BCE(gol) + λ·BCE(celny)`;
- *faktoryzacja*: `p(gol) = p(celny) · p(gol | celny)`, druga głowa uczona na 5 796
  strzałach celnych (25% pozytywów).

Druga wersja odwzorowuje faktyczną strukturę zdarzenia i jest zgodna z literaturowym
rozbiciem xG na xG = P(celny) × P(gol|celny). *Koszt:* ~2 h GPU. *To jest pierwsza rzecz
do zrobienia.*

**P2. Augmentacja czasowa.** W treningu losowa klatka z `{−800, −400, 0}`, w ewaluacji
zawsze `0`. Potraja liczbę widoków bez nowych etykiet i wprowadza realistyczny szum
anotacji. Mechanizm `alt` w `FeatureDataset` już to umożliwia (użyty dziś do odbicia).
**Twardy zakaz `+400`/`+800`.** *Koszt:* ~1 h (ekstrakcja cech dla dwóch dodatkowych
przesunięć + trening).

**P3. Regularyzacja zamiast pojemności.** Skoro optimum wypada w epoce 1–2, sensowne są:
LoRA/adaptery zamiast odmrażania całych bloków (setki tysięcy parametrów zamiast 56 M),
prawdziwy layer-wise LR decay zamiast płaskiego mnożnika 0.02, EMA/SWA wag, mixup na
poziomie cech, label smoothing, mocniejszy weight decay. Sweep 6–8 wariantów.
*Koszt:* ~4 h GPU. *Największa niepewność, ale też najbardziej bezpośrednio celuje w
zdiagnozowaną przyczynę.*

**P4. Hierarchia karnych.** `p = is_penalty ? p_pen : p_open`, gdzie `p_pen` to głowa
uczona wyłącznie na 86 karnych treningowych z silnym priorem na częstość bazową.
*Koszt:* minuty. *Zysk:* ~0.001 log-loss globalnie, ale usuwa jedyną podgrupę, w której
model przegrywa ze stałą — i to ta podgrupa najbardziej rzuca się w oczy na demie.

**P5. Krzywa uczenia: 25 / 50 / 75 / 100% meczów treningowych.** To jest **bramka
decyzyjna dla całego Poziomu 3**. Jeśli krzywa nadal rośnie stromo, priorytetem staje się
pozyskanie danych (SoccerReplay-1988, pozostałe sezony SoccerNet), a dalsze majstrowanie
przy architekturze jest marnowaniem czasu. Jeśli się wypłaszcza — odwrotnie.
*Koszt:* ~1.5 h GPU. *Najwyższy stosunek informacji do kosztu w całym planie.*

**P6. Krzywa czasowa: t = −1600, −1200, −800, −400, 0 na zamrożonym D.**
Podwójna rola: (a) audyt — jeśli jakość rośnie płynnie w miarę zbliżania się do t0, to
sygnał jest fizyczny; skokowy przyrost między −400 a 0 sugerowałby, że t0 podgląda
rozstrzygnięcie; (b) samodzielny wynik — „jak wcześnie da się przewidzieć xG".
*Koszt:* ~40 min. Klatki są już na dysku.

### Poziom 2 — atak na rozdzielczość tam, gdzie ma znaczenie

**P7. Dwustrumieniowe wejście: pełna klatka 448×252 + wycinek okolic bramki w pełnej
rozdzielczości.** To jest hipoteza wyjaśniająca, *dlaczego* skalowanie 448→672→896
zawiodło (0.732 → 0.724 → 0.713): jednolite powiększanie zużywa pojemność na murawę i
publiczność, a informacja krytyczna — postawa bramkarza, ustawienie obrońców — zajmuje
kilka procent kadru. Wycinek wyznaczamy z homografii (81% skuteczności) lub z detekcji
bramki; przy braku — fallback na sam pełny kadr. *Koszt:* ~4 h. *Bezpośrednio celuje w
martwe pole 11–16 m.*

**P8. Naprawa `grass_ratio` i powtórzenie analizy podgrup.** Bez tego kubełek odpowiadający
za 35% straty jest nieinterpretowalny. *Koszt:* ~1 h.

### Poziom 3 — nowa informacja (dopiero po bramce decyzyjnej P5)

**P9. Model wideo na oknie `t ∈ [−2 s, 0]`.** 8–16 klatek, backbone typu VideoMAE lub
lekki temporalny transformer nad cechami per klatka — ale tym razem z fine-tuningiem, bo
PoC F na zamrożonych cechach nie działał. Ryzyko przeuczenia jest tu największe w całym
planie (patrz sekcja 1), więc ma sens dopiero po P1–P3. **Okno musi być twardo ograniczone
do t ≤ 0 w kodzie, nie w komentarzu.**

**P10. Naprawa toru pozycyjnego.** 23% strzałów nie ma geometrii i odpowiada za 27% straty.
Kierunki: kalibracja boiska metodą dedykowaną piłce (TVCalib / PnLCalib) zamiast obecnej
homografii, jawna detekcja bramkarza jako osobna cecha, śledzenie piłki po 7 klatkach
zamiast detekcji na jednej. GBM ma AUC tylko 0.622, ale jest najbardziej zdekorelowanym
członkiem ensembla — jego waga 0.487 i cały zysk log-loss ensembla nad samym G (ΔAUC
dokładnie 0.000, Δlog-loss +0.0072) pochodzą właśnie stąd. Podniesienie go do ~0.70
przełożyłoby się na ensemble mocniej niż kolejne 0.01 AUC w torze obrazowym.

**P11. Pretrening domenowy** na action spotting SoccerNet (17 klas, te same 500 meczów,
etykiety już mamy) — adaptacja backbone'u do specyfiki transmisji przed fine-tuningiem xG.
Wymaga ekstrakcji klatek dla zdarzeń innych niż strzały.

**P12. Więcej danych.** SoccerReplay-1988 po NDA, pozostałe sezony SoccerNet, inne ligi.
Priorytet zależy w całości od wyniku P5.

---

## 6. Czego się realnie spodziewam

Poziom 1 + 2 wykonany w całości: **AUC ok. 0.77–0.78, log-loss ok. 0.325–0.33.**
Największy pojedynczy wkład stawiam na P1 (faktoryzacja) i P3 (regularyzacja), bo obie
uderzają w zdiagnozowaną przyczynę, a nie w objaw.

Sufit dla podejścia jednoklatkowego szacuję na **~0.78–0.80 AUC** — tyle osiąga literatura
na czystych danych eventowych, a my mamy nad nią przewagę (widzimy scenę) i wadę
(nie widzimy części ciała ani prędkości piłki). Powyżej tego poziomu dominuje nieredukowalna
losowość: wykonanie strzelca i refleks bramkarza nie są zapisane w klatce sprzed strzału.

Trzy rzeczy, które **odradzam** mimo że wyglądają kusząco: dalsze powiększanie backbone'u
(przeuczenie), dalsze przebieranie w składzie ensembla (poniżej progu pomiaru), TTA
(zmierzone, istotnie szkodzi).
