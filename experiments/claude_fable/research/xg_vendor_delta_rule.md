# Reguła wyboru marginesu δ dla H2 (xg-vendor, ticket 08, decyzja D5)

Status: ustalona 2026-10-11, przed pierwszym uruchomieniem symulacji mocy. Skrypt `scripts/vendor_power.py` odmawia pracy, gdy sha256 tej notatki różni się od stałej `RULE_SHA256` w `xgpoc/vendor/power.py`, i zapisuje ten hash w każdym wyniku. Zmiana reguły po zobaczeniu wyników wymaga nowej notatki i nowego hasha, a obie wersje trafiają do raportu.

## Test, którego moc liczymy

- H2: różnica log-loss modelu klatka + pozycje i modelu samych pozycji (szczebel freeze-frame komparatora pozycyjnego). Wartość ujemna oznacza, że klatka pomaga.
- Oba modele są rekalibrowane tak samo (decyzja D6). Pozycje dostają Platta, a stos to regresja logistyczna na logitach klatki i pozycji. Oba są dopasowane w cross-fittingu na pięciu foldach grupowanych po meczach. Foldy są wspólne i ustalone seedem 0. Żaden strzał nie dostaje predykcji z modelu dopasowanego na jego meczu.
- Wnioskowanie: bootstrap meczowy, 5 000 losowań (seed 0), te same losowania dla obu modeli. TOST na poziomie 5% na stronę: równoważność przy δ, gdy 90-procentowy przedział percentylowy różnicy leży w (−δ, δ).
- Kod testu jest tym samym kodem, który liczy H2 w analizie konfirmacyjnej (`xgpoc/vendor/ladder.py`).

## Projekt symulacji

- Podstawa: tabela deweloperska (`results/xgv_eval_dev`), populacja pierwotna, 2 708 strzałów w 135 meczach, etykieta gola SoccerNet.
- Prawda generująca etykiety ma dwa warianty:
  - deweloperska: regresja logistyczna gola na logicie pozycji i logicie klatki, dopasowana bez regularyzacji na całej tabeli deweloperskiej (klatka dodaje tyle, ile pokazują dane deweloperskie);
  - zerowa: ta sama regresja bez logitu klatki (klatka nie dodaje nic ponad pozycje).
- Jedno symulowane badanie:
  1. Losuje 135 meczów ze zwracaniem z meczów deweloperskich. Każda kopia meczu jest osobnym meczem.
  2. Bierze predykcje pozycji i logity klatki ich strzałów bez zmian.
  3. Losuje nowe etykiety gola z prawdy, niezależnie dla każdego strzału.
  4. Liczy H2 jak wyżej.
- Wariant klatki:
  - archiwalna: logity klatki z tabeli;
  - słabsza: do logitu klatki dodany szum gaussowski N(0, σ²), świeży w każdym badaniu. σ dobieramy bisekcją tak, by oczekiwany log-loss samej klatki po rekalibracji Platta, liczony względem prawdy na całej populacji deweloperskiej, wzrósł o 0,002. Tak wycenia się mniejszą pulę treningową retrenowanego G-combo (D7).
- Populacja:
  - pierwotna: jak wyżej;
  - łączona (projekcja): do populacji pierwotnej dochodzi około 50 odzyskanych goli i około 650 strzałów tylko ze StatsBomb (środek przedziału 500–800), razem 700 strzałów, czyli 25,85% populacji pierwotnej. Strzały wtórne leżą w tych samych 135 meczach. Dlatego projekcja nie dodaje meczów: w każdym symulowanym meczu dolosowuje round(0,2585 · n_meczu) strzałów ze zwracaniem spośród strzałów pierwotnych tego meczu, z etykietami z prawdy. Założenie: strzały wtórne mają ten sam rozkład (pozycje, klatka, prawda) co pierwotne strzały ich meczu. W rzeczywistości strzały zablokowane i niecelne mają niskie xG, a odzyskane gole są golami z definicji, więc projekcja jest raczej optymistyczna.
- 1 000 symulowanych badań na scenariusz, generator `numpy.random.default_rng([0, numer scenariusza])`. Błąd Monte Carlo mocy 0,8 wynosi około 0,013.
- Moc przy δ to odsetek badań, w których TOST stwierdza równoważność przy δ. Raportujemy też moc jednostronnego testu przewagi (klatka + pozycje lepsze niż pozycje, 5%).

## Reguła

1. Zbiór marginesów: {0,002; 0,003; 0,005} log-loss.
2. Scenariusz decyzyjny: populacja pierwotna, prawda deweloperska. Moc decyzyjna przy δ to minimum mocy z dwóch wariantów klatki, archiwalnej i słabszej o 0,002.
3. δ to najmniejszy margines ze zbioru, dla którego moc decyzyjna wynosi co najmniej 0,80. Moc porównujemy wprost z 0,80, bez poprawki na błąd Monte Carlo.
4. Jeśli żaden margines nie osiąga 0,80, reguła nie wybiera δ. Wtedy właściciel decyduje przed pre-rejestracją, a raport podaje tabelę mocy i najmniejszy margines równoważności.
5. Pozostałe scenariusze (prawda zerowa, populacja łączona) są raportowane i nie wpływają na wybór.
6. Właściciel zatwierdza δ wybrane regułą (D5), zanim trafi ono do pre-rejestracji (ticket 14).

## Granice projektu

- Etykiety są losowane niezależnie przy danych predykcjach. Nie ma dodatkowej korelacji etykiet w meczu poza tą, którą niosą predykcje. Wynik symulacji podaje średnie bootstrapowe odchylenie standardowe różnicy H2 obok tego samego odchylenia na prawdziwych etykietach deweloperskich. Gdy symulacja daje wyraźnie węższe przedziały niż dane, moc jest zawyżona.
- Prawda deweloperska pochodzi z modeli, które widziały większość tych meczów w treningu. To liczby deweloperskie, nie wynik.
