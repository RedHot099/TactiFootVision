# Projekt rundy v1.4: kontrakt czasu i wspólnie dostrajane G z lokalnymi widokami

Doprecyzowanie po rundzie, 2026-09-09: zerowe projekcje zapewniają identyczność samej fuzji, lecz nie równoważność całego modelu z G. Dodatkowe tokeny nadal uczestniczą w końcowym poolingu. Kontrola pełnej architektury i jej ograniczenia: [raport diagnostyczny v15](v15_results_2026-09-09.md). Kod oraz wyniki v14 pozostają bez zmian.

Runda odpowiada na dwa punkty handoffu z 2026-09-09: jawny kontrakt kontaktu i czasu wejść oraz pierwszy eksperyment architektury, w którym lokalne widoki są uczone razem z ostatnimi blokami DINOv2, a nie doklejane jako zamrożone wektory. Czas i geometria pozostają poza tą rundą.

## Kontrakt kontaktu i czasu (`xgpoc/v13/contact.py`)

Czas liczony w milisekundach względem kotwicy t0, czyli `position_ms` z etykiet, która w każdym 8-sekundowym klipie leży w 4000 ms. Typy:

- `ContactInterval(lower_ms, upper_ms, source, confidence)`: przedział kontaktu, oba końce albo żaden. Źródło `none | model_review | human`; przedział bez źródła jest błędem. Status `uncertain`, brak granic lub niespójność czasu w przeglądzie oznaczają przedział nieznany.
- `DecodedFrame(path, nominal_ms, measured_ms)`: klatka wejściowa z nominalnym offsetem seeku i, gdy zmierzono, rzeczywistym czasem zdekodowanej klatki. Nominalny offset dodatni jest odrzucany już w konstruktorze.
- `admit(frames, contact) -> Admission` ze statusem:
  - `safe`: przedział znany, wybrana najpóźniejsza klatka o czasie ściśle przed początkiem przedziału; `gap_ms` to odstęp klatki od początku przedziału,
  - `nominal`: przedział nieznany, zostaje nominalna klatka t0 dokładnie jak w baseline, rekord jest oflagowany,
  - `unsafe`: przedział znany, żadna klatka nie leży przed nim; klatka nie jest wybierana, rekord zostaje w tabeli.

Rzeczywisty czas klatki mierzy `locate_frames_in_clip`: sekwencyjne dekodowanie klipu i dopasowanie obrazu do zapisanych JPG. Pomiar 300 zaudytowanych i 300 losowych strzałów (`cache/v13/contact_timing/`): 2849 z 2857 zmierzonych klatek leży dokładnie na nominalnym czasie, jedna była spóźniona o 80 ms, 143 nie dało się jednoznacznie dopasować w statycznych scenach. Kotwica przeglądu kontaktu i kotwica klatek treningowych to ten sam moment. Reguła przenosi 131 ze 136 strzałów o znanym przedziale na klatkę −400 ms lub −800 ms; pozostałe 5 (kontakt po t0) zachowuje t0.

Reguła obowiązuje jednakowo wszystkie warianty tej rundy. Dla 164 niepewnych kontaktów i wszystkich strzałów bez przeglądu status to `nominal`. Przeglądy pochodzą z modelu językowego na siatce 200 ms, nie od człowieka; runner zapisuje sumę kontrolną pliku admisji w tożsamości każdego modelu.

## Model wielowidokowy (`xgpoc/v14/`)

Widoki: pełna klatka 448×252 przygotowana tak jak w G, plus dwa lokalne widoki 224×126 (16×9 tokenów) wycięte z klatki t0 na podstawie detekcji z `cache/full_v1/positional/tokens.npz`:

- `shooter`: kwadrat w współrzędnych znormalizowanych obejmujący piłkę i najbliższego zawodnika z pola, minimalny bok 0,25, margines 0,05. Zawodnik jest strzelcem tylko przy odległości środków do 0,15; heurystyka nie została zweryfikowana wobec prawdy, a arkusz `cache/v14/view_sheet.jpg` pokazuje jedynie, że wycinki obejmują piłkę i sąsiednich zawodników.
- `keeper`: kwadrat wokół wykrytego bramkarza najbliższego piłce. Repo nie ma źródła położenia bramki w obrazie, więc widok „bramkarz–bramka” nie istnieje; jest widok bramkarza.

Pokrycie w puli rozwojowej (8801 strzałów): widok strzelca 61%, bramkarza 19%, oba 11%. Brak piłki, nierozstrzygnięty strzelec lub brak bramkarza dają widok zamaskowany, nigdy pełną klatkę jako zamiennik. Gdy dopuszczona klatka nie jest klatką t0, na której powstały detekcje, oba widoki są zamaskowane.

Architektura `MultiViewViTXG`: jeden współdzielony DINOv2-L (ostatnie 4 bloki trenowane, jak w G) koduje pełną klatkę i obecne wycinki; tokeny lokalne pozostają tokenami. Każdy widok ma uczony wektor identyfikujący; brakujący widok wnosi jeden uczony token, a reszta jego pozycji jest maskowana. Jedna warstwa attention (TransformerEncoderLayer, 8 głowic, norm-first) łączy widoki; jej projekcje wyjściowe startują od zera, więc sama fuzja jest początkowo identycznością. Na końcu istniejąca głowica attention-pooling z maską paddingu, wyjście gol + pomocnicza celność. Głowica uwzględnia niezamaskowane tokeny lokalne, dlatego początkowa predykcja może różnić się od G.

Kontrola `g_views_control`: identyczna architektura i identyczne flagi obecności, ale wycinki o tej samej powierzchni i rozdzielczości z przeciwnego rogu klatki.

## Protokół rundy (`scripts/run_v14.py`, `scripts/evaluate_v14.py`)

Role fit/select/calibrate/evaluate z `xgpoc/v13/protocol.py` (seed 1300), ta sama kohorta co v1.3. Część evaluate była już raz otwarta, więc wynik tej rundy jest wynikiem rozwojowym: pozwala odrzucić kierunek albo uzasadnić test na nowych meczach, nie potwierdza przyrostu.

Warianty i seedy zadeklarowane przed treningiem: `g_base`, `g_views`, `g_views_control`, seedy 0, 1, 2. Wspólne ustawienia: DINOv2-L, 4 bloki, 5 epok, lr 3e-4, mnożnik lr enkodera 0,02, batch 8, EMA 0,998, waga celności 0,5, wybór epoki po skalibrowanym log-loss na select, Platt na calibrate, wspólna stała karnych z fit. Runner zapisuje koszt: minuty, liczbę aktualizacji, szczyt pamięci GPU i liczbę trenowanych parametrów.

Porównania sparowane, bootstrap 2000 losowań po grupach gospodarzy: `g_views` vs `g_base`, `g_views_control` vs `g_base`, `g_views` vs `g_views_control`. Kryterium sukcesu rozwojowego: przedział różnicy log-loss `g_views` względem obu odniesień nie obejmuje zera przy niegorszym AUC. Każde inne ustawienie to wynik nierozstrzygnięty i g_base pozostaje rekomendacją.

Artefakty: `bundle.pt` z konstruktorem, wagami, Plattem, stałą karnych, tożsamością (protokół, kod, piksele, admisje, plik detekcji) i commitem; `scripts/verify_v14_bundle.py` odtwarza predykcje w nowym procesie. Kolejność: smoke (2 epoki, 64 strzały na rolę, pomiar pamięci, odtworzenie paczki), potem pełna runda w nowym katalogu.
