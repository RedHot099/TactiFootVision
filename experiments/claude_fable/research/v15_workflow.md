# Przygotuj następną rundę xG na podstawie diagnozy

Aktualizacja 2026-09-10: na decyzję użytkownika pracujemy teraz na 20 anotacjach AI, bez oczekiwania na człowieka. Importer zachowuje źródło `model_review` i pełny eksport z dowodami. Osobny skrypt porównuje wejścia do zamrożonego G z prowizorycznymi przedziałami kontaktu. Wyniki i ograniczenia opisuje [raport porównania AI](v15_ai_contact_results_2026-09-10.md).

[Instrukcja anotatora AI](v15_ai_annotator.md) pozostaje w wersji użytej podczas anotacji, ponieważ eksport zapisuje jej SHA-256. Jej opis braku obsługi importu AI jest historyczny. Poniższa procedura dwóch recenzentów pozostaje dostępna dla przyszłego niezależnego przeglądu; obecne wyniki AI nie są zgodnością dwóch osób.

Wszystkie polecenia wykonuj z `experiments/claude_fable` w worktree `claude-fable/xg-poc`.
Nie używaj katalogów `results/v14` ani `results/v13_final` do nowych wyników.
Stan wykonania opisuje [raport diagnostyczny](v15_results_2026-09-09.md).

## Bieżący etap: porównanie na 20 anotacjach AI

```bash
./run.sh python scripts/diagnostic_review.py import \
  --export cache/v15/ai_reviews/20260909T135428Z_codex/annotations.json \
  --out cache/v15/ai_imports/review20.csv
./run.sh python scripts/compare_ai_contact.py \
  --annotations cache/v15/ai_reviews/20260909T135428Z_codex/annotations.json \
  --out cache/v15/ai_contact_comparison
```

Powtórzenie importu sprawdza zgodność CSV i kopii `.source.json`. Powtórzenie porównania sprawdza sumy kontrolne wejść i wyników, bez ponownej inferencji. Po zmianie eksportu albo kodu wybierz nowy katalog wyników. Jeśli proces przerwano, jego nieukończone pliki pozostają w katalogu z końcówką `.building`; nie są ukończonym wynikiem.

Porównanie obejmuje `original_t0` z historycznego cache, `coarse_precontact` z jego pięciu offsetów, `clip_t0` z oryginalnego klipu oraz `dense_precontact` z tego samego dekodera. Klatka dense musi mieć czas ≤ 0 ms i ściśle mniejszy od dolnej granicy przedziału AI. Reguła czasu jest ustalona przed inferencją; predykcje nie uczestniczą w wyborze klatki. `clip_t0` pozwala oddzielić zmianę czasu od zmiany sposobu ekstrakcji obrazu.

Skrypt sprawdza dowody obrazowe, czasy i sumy kontrolne, następnie stosuje zamrożone trzy modele, kalibrację i obsługę karnych. Nie weryfikuje automatycznie trafności ocen wizualnych AI. Niepewny kontakt i deklarowana wcześniejsza znajomość anotacji wykluczają dobór klatki precontact, ale nie usuwają próby z pokrycia.

Odczytaj `summary.json`, `predictions.csv`, `paired_summary.csv`, `paired_details.csv` i `largest_changes.jpg` w katalogu wyniku. Różnice log-loss używają niezmienionych etykiet źródłowych. Wyniki rozdzielono według fit/select i próbki losowej/celowanej. Nie służą do wyboru nowego modelu ani kalibracji.

## Jednolita ekstrakcja: dane `canonical_clip_v1`, audyt i raport jakości

Kontrakt opisuje [projekt danych](v15_canonical_data_design.md), wynik [raport z 2026-09-10](v15_canonical_data_results_2026-09-10.md). Trzy kroki, każdy z własnym katalogiem wyniku i `complete.json`; powtórzenie sprawdza sumy kontrolne i nic nie liczy ponownie. Po zmianie kodu lub wejść wybierz nowy katalog.

```bash
./run.sh python scripts/prepare_canonical_data.py \
  --annotations cache/v15/ai_reviews/20260909T135428Z_codex/annotations.json \
  --out cache/v15/canonical_data_v1
./run.sh python scripts/audit_canonical_data.py --out cache/v15/canonical_extraction_audit_v1
./run.sh python scripts/report_data_quality.py --out cache/v15/canonical_quality_v1
```

Dane: `shots.csv` ma jeden rekord na próbę z pochodzeniem etykiety źródłowej, oceną AI wyniku i przypisania gola oraz statusem ekstrakcji. `frames.csv` ma rekordy `t0` i `precontact` z indeksem klatki, PTS, `time_base`, czasem względem kotwicy, odstępem od dolnej granicy kontaktu, sumami pliku, pikseli i tensora. Reguła: czas ≤ 0 i ściśle przed dolną granicą przedziału AI; kontakt niepewny zostaje jawnym brakiem. `source_recovery.csv` wymienia brakujące nagrania połówek z oczekiwanymi sumami.

Audyt: dekoduje tę samą klatkę systemowym FFmpeg, koduje kanoniczny PNG do JPEG q3 i liczy zamrożony zespół G na wszystkich wariantach. Odtwarza też 72 historyczne wejścia w ich pierwotnym składzie batcha; ta kontrola dowodzi niezmienności modelu i potoku. Skład batcha sam zmienia prawdopodobieństwa do około 1e-4, więc wspólne wejścia w innym składzie są porównywane z progiem szumu 1e-3, nie 1e-5.

Raport jakości: `quality_flags.csv` rozdziela flagi techniczne (błąd dekodowania, prawie czarny obraz, dokładny duplikat, podejrzane cięcie w oknie wejścia, sąsiednia próba w innej partycji) od flag przeglądowych (niepewny kontakt, nieznany wynik, niewidoczna piłka lub bramkarz, szeroki przedział, sąsiednia próba w tej samej połowie). Flagi opisują, nie usuwają. `neighbours.csv` wymienia inne strzały z indeksu w tej samej połowie w oknie ±15 s. `post_shot_changes.csv` służy tylko kalibracji miary cięcia na materiale po strzale.

## 1. Sprawdź interpretację inicjalizacji v14

Odczytaj `v15_diagnostics/initialization.json`.
Skrypt sprawdza pełną architekturę DINOv2-L na syntetycznych obrazach, bez pobierania wag i bez treningu.

```bash
./run.sh python scripts/audit_v14_initialization.py --out research/v15_diagnostics/initialization_repeat.json
```

Fuzja startuje jako identyczność. Cały model nie startuje jako G, ponieważ pooling otrzymuje dodatkowe tokeny.
Po zamaskowaniu lokalnych tokenów głowica odtwarza wynik bazowy do błędu numerycznego.
Ta kontrola nie dowodzi, że ten mechanizm spowodował pogorszenie v14.
Nie poprawiaj zamrożonego modelu wstecz.

## 2. Otwórz pakiet dla recenzenta

Pakiet `cache/v15/human_review` zawiera 300 prób: 200 losowych i 100 celowanych.
Próbka celowana obejmuje 50 wcześniej niepewnych i 50 wcześniej uznanych przez model za pewne.
Żadna próba nie pochodzi z ról calibrate ani evaluate.
Grupy i etykiety źródłowe znajdują się poza katalogiem udostępnianym recenzentowi.

Sprawdź istniejący pakiet bez regeneracji:

```bash
./run.sh python scripts/diagnostic_review.py verify
```

Uruchom serwer, jeśli nie działa już na porcie 8765:

```bash
./run.sh python scripts/serve_review.py
```

Otwórz <http://127.0.0.1:8765>. Udostępniaj tylko `public/`.
Ten serwer obsługuje zakresy bajtów potrzebne do przewijania wideo. Nie zastępuj go prostym `http.server`.
Pakiet zawiera klatki po strzale i pełne krótkie klipy. Nie używaj go jako zbioru wejść modelu.

Najpierw niech dwie osoby niezależnie ocenią te same pierwsze 20 prób.
Każda osoba używa własnego aliasu. Nie pokazuj drugiej osobie pierwszych ocen ani predykcji G.
Sprawdź zgodność przed oceną pozostałych prób. Jeśli konwencja przedziału jest niejasna, wyjaśnij ją i zachowaj pierwotne eksporty.

Recenzent wykonuje kolejno:

1. Wskazuje ocenianą próbę przy kotwicy klipu. Rozróżnia podanie, strzał i dobitkę.
2. Ustawia początek przedziału na ostatniej klatce na pewno przed kontaktem, a koniec na pierwszej na pewno po kontakcie.
3. Przy zasłonięciu zachowuje status niepewny. Przy kontakcie poza pokazanym zakresem wybiera osobny status.
4. Wskazuje środek strzelca kliknięciem w obraz. Pozycja jest powiązana z czasem klatki.
5. Ocenia część ciała, sytuację i widoczność. Wynik oraz przypisanie gola sprawdza osobno w pełnym klipie.
6. Przy braku wystarczającego kontekstu zostawia wynik nieznany. Koniec ośmiosekundowego klipu nie dowodzi braku gola.
7. Naciska „Oznacz próbę jako ocenioną”, potem pobiera ukończone adnotacje.

Formularz zachowuje dane w przeglądarce oddzielnie dla pakietu i aliasu.
Eksportuj je regularnie. Zmiana ukończonej oceny przywraca jej status roboczy.

## 3. Zweryfikuj eksporty

Zapisz dwa niezależne eksporty pod różnymi nazwami, np. `cache/v15/human_exports/a.json` i `b.json`.
Nie oznaczaj ocen modelu jako `human`.

```bash
./run.sh python scripts/diagnostic_review.py import \
  --export cache/v15/human_exports/a.json --out cache/v15/human_imports/a.csv
./run.sh python scripts/diagnostic_review.py import \
  --export cache/v15/human_exports/b.json --out cache/v15/human_imports/b.csv
```

Importer sprawdza tożsamość pakietu, identyfikatory, rzeczywiste czasy klatek, granice przedziału i spójność wyniku.
Nie zmienia indeksu treningowego, wyników modeli ani poprzednich adnotacji.
W przypadku poprawionego eksportu użyj nowej nazwy docelowej.

## 4. Odczytaj wrażliwość G i dołącz oceny człowieka

Zamrożony zespół trzech G liczy pięć istniejących klatek na próbę, od −1600 do 0 ms.
Kalibracja pozostaje zamrożona. Odtworzenie predykcji na niezmienionych wejściach select jest częścią kontroli inferencji.
Nominalny offset nie jest dowodem, że klatka poprzedza kontakt.

Gotowe predykcje znajdują się w `cache/v15/g_sensitivity_verified`. Nowe porównanie AI sprawdza ich manifest, sumy kontrolne i odtworzenie t0 dla ocenionych prób. Zmiana importera zmieniła sumę kontrolną źródła zapisaną w historycznym manifeście; ponowne `diagnose_g.py infer` wymaga teraz nowego katalogu. Kopie skryptów sprzed integracji AI zachowano w `cache/v15/sources_before_ai_integration`. Nie regeneruj istniejącego pakietu ani nie nadpisuj historycznych predykcji.

Gdy potrzebna będzie nowa inferencja wszystkich 300 prób:

```bash
./run.sh python scripts/diagnose_g.py infer --out cache/v15/g_sensitivity_current
```

Gotowy raport bez adnotacji znajduje się w `v15_diagnostics/baseline_verified`.
Po otrzymaniu eksportów utwórz kolejny raport:

```bash
./run.sh python scripts/diagnose_g.py report \
  --predictions cache/v15/g_sensitivity_verified \
  --review-exports cache/v15/human_exports/a.json cache/v15/human_exports/b.json \
  --out research/v15_diagnostics/human_review_01
```

Sprawdź najpierw `coverage_by_sample.csv` oraz `human_coverage.csv`.
Raport oznacza zgodny kontakt dopiero dla co najmniej dwóch różnych aliasów z zachodzącymi na siebie przedziałami.
Używa ich konserwatywnej sumy przedziałów. Rozbieżności i pojedyncze oceny nie tworzą pewnej kotwicy.
Różne aliasy nie są technicznym dowodem niezależności osób; niezależność trzeba zachować w organizacji przeglądu.

Odczytaj `contact_comparison.csv`, `paired_safe_comparison.csv` i `situation_comparison.csv`.
Raport dopuszcza klatkę tylko przy zmierzonym czasie ściśle przed początkiem uzgodnionego przedziału.
Zachowuje nieocenione próby w zestawieniu pokrycia.
Etykiety wyniku człowieka i źródłowe są raportowane osobno.

Analizuj osobno próbkę losową i celowaną oraz fit i select.
Wyniki fit są wewnątrz próby treningowej. Select służył do wyboru epoki, więc też nie jest niezależnym testem.
Nie traktuj tabel podgrup jako dowodu poprawy generalizacji.

## 5. Przygotuj zewnętrzny pilotaż

Inwentaryzacja `v15_diagnostics/external_pilot_verified` zawiera 8 kompletnych meczów, 16 połówek i 412 wierszy kandydatów.
Żaden mecz nie jest jeszcze dopuszczony do potwierdzającej oceny.

Skopiuj `half_review.csv` i `attempt_review.csv` do katalogu adnotacji.
Nie edytuj oryginalnych szablonów.
Sprawdź całe nagrania, nie tylko okolice kandydatów. Łączna długość to około 12,5 godziny bez czasu na adnotacje.

W `half_review.csv` zapisz zakres przeglądu, zgodność zegara, sprawdzenie wszystkich prób i liczbę goli.
Sprawdź też tożsamość spotkań względem wszystkich dotychczas użytych danych.
Różne nazwy plików nie dowodzą rozłączności. Wpisz wynik i dowód w kolumnach `overlap_status` i `overlap_evidence`.

W `attempt_review.csv` oceń, czy kandydat jest strzałem, a potem kontakt i wynik.
Wiersze GOAL bez kotwicy nie są automatycznie strzałami. Połącz je z właściwą próbą podczas przeglądu.
Dodaj niewykryte próby z unikalnym `review_id`. Czasy kontaktu podawaj względem początku pliku danej połowy, nie zegara meczu.

```bash
./run.sh python scripts/prepare_external_review.py check \
  --halves cache/v15/external_human/half_review.csv \
  --attempts cache/v15/external_human/attempt_review.csv \
  --out research/v15_diagnostics/external_readiness_01.json
```

Kontrola sprawdza deklaracje recenzenta, kompletność pól i rozliczenie goli. Nie potwierdza samodzielnie prawdziwości adnotacji.
Nawet po przejściu kontroli ustal protokół i zamroź zbiór przed oceną.
Osiem meczów traktuj początkowo jako pilotaż. Dobierz większy test do minimalnej interesującej różnicy log-loss.

## 6. Wybierz jedną zmianę do treningu

Najpierw rozstrzygnij, czy G korzysta z niewłaściwego momentu lub błędnej etykiety.
Jeśli tak, przygotuj nową wersję danych i nowy baseline z tymi samymi regułami czasu dla wszystkich wariantów.
Nie przenoś ręcznie wybranej poprawy kilku klatek na globalne przesunięcie całego zbioru.

Jeśli dane są wiarygodne, wybierz jeden z dwóch eksperymentów:

- Przy błędach związanych z drobnymi szczegółami porównaj pełną klatkę 448×252 i 672×378. Oba wymiary większego wariantu są wielokrotnościami patcha 14. Najpierw sprawdź pamięć GPU.
- Przy utrzymującym się przeuczeniu porównaj 4, 2 i 1 dostrajany blok. Zachowaj rozdzielczość 448×252.

Przed treningiem zapisz warianty, seedy 0–2, źródła wejść, rolę każdej kohorty oraz porównania.
Zachowaj te same cele pomocnicze, reguły wyboru epoki, kalibracji i karnych.
Kontroluj liczbę aktualizacji, efektywny batch oraz koszt, gdy większa rozdzielczość wymaga akumulacji gradientu.
Nie używaj ponownie otwartego evaluate ani historycznego testu jako świeżego potwierdzenia.

Ten etap jest przygotowany jako procedura, nie uruchomiony trening.
Obecne kotwice AI wystarczają do prowizorycznej diagnozy wejść. Trening nowej wersji i ocena generalizacji wymagają szerszych danych oraz ustalonego protokołu; zewnętrzny zbiór nie jest jeszcze dopuszczony.
Nie uruchamiaj wszystkich kierunków tylko dlatego, że GPU jest wolne.

## 7. Sprawdź narzędzia i zamrożone wyniki

```bash
./run.sh python -m pytest -q
./run.sh ruff check .
./run.sh python scripts/verify_frozen_analysis.py
```

Kontrola przeglądarki wymaga uruchomionego lokalnego serwera:

```bash
PYTHONPATH=. uv run --with playwright --project /home/kuba/projects/tactifoot_vision \
  python scripts/check_review_browser.py
```

Test używa izolowanej przeglądarki, usuwa syntetyczny eksport i nie tworzy ludzkich adnotacji.
