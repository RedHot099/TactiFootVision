# Porównanie wejść G na 20 prowizorycznych anotacjach AI

Wykonano import anotacji AI, ekstrakcję klatek bliżej kontaktu i inferencję zamrożonego zespołu G. Gęstsza ekstrakcja zwiększyła pokrycie z 15 do 17 prób i zmniejszyła medianę odstępu od dolnej granicy przedziału kontaktu z 360 do 40 ms. Zmiana sposobu ekstrakcji przy tym samym czasie wpływa na predykcje w podobnej skali jak zmiana czasu. Trzeba uwzględnić oba czynniki w następnej wersji danych.

Nie stwierdzono poprawy jakości modelu. Wszystkie 20 etykiet źródłowych oznaczają brak gola, a próbka obejmuje 18 przykładów fit i 2 select. Niższy log-loss na takiej próbce wynika z obniżenia xG i nie pokazuje, czy model lepiej rozróżnia gole.

## Zakres i pochodzenie danych

Pracę wykonano w worktree `claude-fable/xg-poc`, przy HEAD `1343f5b`, z istniejącymi zmianami lokalnymi. Kod inferencji, wagi i kalibracja v13/v14 pozostały niezmienione. Nowy kod i dane diagnostyczne są poza zamrożonymi katalogami wyników.

- Eksport: `cache/v15/ai_reviews/20260909T135428Z_codex/annotations.json`; SHA-256 `a6cbeae003457a36709b7a1dcf56601596c5e4d12a69aff9796f0e3de45ebf8c`.
- Źródło: `model_review`; recenzent `ai-codex-20260909T135428Z`; dokładna wersja modelu nie została podana przez anotatora (`model=null`).
- Zakres: `r0000`–`r0019`, w tym 12 prób z grupy losowej i 8 z celowanej. To pierwsze 20 ukończonych anotacji, nie nowy reprezentatywny test.
- Kontakt: 17 przedziałów, 3 nierozstrzygnięte próby (`r0012`, `r0014`, `r0019`). Brak zadeklarowanej wcześniejszej znajomości anotacji.
- Wynik według AI: 16 `no_goal`, 4 `unknown`. Etykiet źródłowych nie zastąpiono ocenami AI.
- Sprawdzono sumy kontrolne i czasy 868 obejrzanych klatek, w tym 455 dodatkowych klatek z pełnych klipów. Kontrola potwierdza strukturę i zgodność zapisanych dowodów, nie prawdziwość oceny wizualnej.

Importer zapisuje [CSV](../cache/v15/ai_imports/review20.csv) i kopię oryginalnego eksportu `.source.json`, zachowując dane wykonania, dowody i deklarację wcześniejszej ekspozycji. Powtórzenie tego samego importu nie zmienia plików; inne dane lub dowody wymagają nowej nazwy wyniku. Istniejące uzgadnianie ocen ludzkich nadal odrzuca źródło AI, więc te anotacje nie stają się automatycznie oceną dwóch osób.

## Reguły porównania

Każda z 20 prób ma cztery wiersze w [predictions.csv](../cache/v15/ai_contact_comparison/predictions.csv), również przy braku dostępnej klatki.

| Wariant | Pochodzenie | Dostępne próby |
| --- | --- | ---: |
| `original_t0` | Historyczna klatka nominalnego t0 | 20 |
| `coarse_precontact` | Najpóźniejsza dopuszczalna z pięciu klatek historycznego cache | 15 |
| `clip_t0` | Najpóźniejsza klatka oryginalnego klipu z czasem ≤ 0 ms | 20 |
| `dense_precontact` | Najpóźniejsza klatka klipu z czasem ≤ 0 ms i ściśle przed dolną granicą AI | 17 |

Nowe klatki dekodowano sekwencyjnie przez OpenCV, z czasem `CAP_PROP_POS_MSEC − 4000 ms`. Zachowano oryginalne wymiary i zapisano PNG. Oba nowe warianty przechodzą przez ten sam dekoder, zapis i zamrożony `ImageDataset`: rozmiar 448×252, maska nakładki i dotychczasowa normalizacja. Dowody anotatora używające `clip_pts` zweryfikowano osobno względem znaczników ffprobe; nie pomylono tej osi czasu z czasem względem kotwicy.

Przy niepewnym kontakcie lub zadeklarowanej wcześniejszej ekspozycji skrypt nie wybiera klatki precontact. Predykcje nie uczestniczą w wyborze czasu. Trzy nierozstrzygnięte kontakty pozostają brakami, bez zastępowania predykcji zerem. Dla `r0006` i `r0010` wcześniejsze klatki cache nie miały wystarczająco pewnego dopasowania czasu; pełne dekodowanie dostarczyło wejścia dense.

Zamrożone trzy modele `g_base` wykonały po 72 predykcje. Uśredniono logity i zastosowano istniejącą kalibrację Platta oraz regułę karnych. Odtworzenie 20 historycznych wejść t0 dało maksymalny błąd prawdopodobieństwa **8,33 × 10⁻¹⁷** względem poprzedniej inferencji.

## Wyniki

Dla 17 prób z przedziałem AI historyczne t0 znajdowało się przed dolną granicą w 1 przypadku, wewnątrz przedziału w 13 i po górnej granicy w 3. Są to relacje względem prowizorycznych przedziałów AI, nie niezależnie potwierdzone momenty kontaktu.

Klatki dense wybrano przy offsetach od −200 do 0 ms. W 16 próbach odstęp od dolnej granicy wynosi 40 ms. W `r0002` kontakt zaczyna się według AI przy +80 ms, dlatego ograniczenie wejścia do t ≤ 0 daje odstęp 80 ms. Dla coarse mediana odstępu wynosi 360 ms, a zakres 80–800 ms.

Poniższe agregaty opisują mieszaną próbkę diagnostyczną. Δ oznacza pierwszy wariant minus drugi; zmiany prawdopodobieństwa podano w punktach procentowych. Mniejsza strata na tych przykładach bez goli nie dowodzi lepszej generalizacji.

| Porównanie | Pary | Średnia ΔxG, p.p. | Średnia \|ΔxG\|, p.p. | Δ log-loss |
| --- | ---: | ---: | ---: | ---: |
| dense − original t0 | 17 | −0,952 | 1,545 | −0,012189 |
| dense − coarse | 15 | +1,169 | 2,962 | +0,016805 |
| dense − clip t0 | 17 | −0,482 | 0,982 | −0,006235 |
| clip t0 − original t0, te same 17 prób | 17 | −0,470 | 1,026 | −0,005954 |

Na wszystkich 20 próbach sama zmiana ekstrakcji przy t0 daje średnią bezwzględną zmianę xG 0,901 p.p. Do porównania jej skali z przesunięciem czasu służą dwa ostatnie wiersze tabeli, obejmujące identyczne 17 prób. Kontrola t0 oddziela efekt czasu od łącznego efektu sposobu ekstrakcji i zapisu; nie identyfikuje osobno wpływu kodeka, skalowania ani zaokrągleń dekodera.

W podziale na role i sposób doboru próby wynik dense − clip t0 wygląda następująco:

| Rola i grupa | Pary / wszystkie próby | Średnia ΔxG, p.p. | Średnia \|ΔxG\|, p.p. | Δ log-loss |
| --- | ---: | ---: | ---: | ---: |
| fit, losowa | 7 / 10 | −0,382 | 0,631 | −0,004521 |
| fit, celowana | 8 / 8 | −0,907 | 1,248 | −0,011793 |
| select, losowa | 2 / 2 | +0,863 | 1,145 | +0,009998 |

Kierunek zmiany straty różni się między fit i select. Dwa przykłady select nie pozwalają ocenić stabilności tego zjawiska. Nie liczono AUC, przedziałów potwierdzających poprawę ani rankingu modeli.

Pełne podziały są w [paired_summary.csv](../cache/v15/ai_contact_comparison/paired_summary.csv), a wartości dla pojedynczych prób w [paired_details.csv](../cache/v15/ai_contact_comparison/paired_details.csv).

## Przegląd największych zmian

Obejrzano [zestawienie klatek pięciu największych zmian dense − clip t0](../cache/v15/ai_contact_comparison/largest_changes.jpg): `r0017`, `r0013`, `r0005`, `r0007`, `r0009`. Tabela [largest_changes.csv](../cache/v15/ai_contact_comparison/largest_changes.csv) wiąże je z wynikami. To przegląd po inferencji, nie dodatkowa niezależna anotacja.

Największa zmiana występuje dla `r0017`: xG wynosi 0,378290 na historycznym t0, 0,318262 na świeżym t0 i 0,277927 na klatce −80 ms. Z łącznej różnicy −10,04 p.p. około −6,00 p.p. wiąże się ze zmianą ekstrakcji, a −4,03 p.p. ze zmianą czasu. Przypisanie całej różnicy dokładniejszej kotwicy kontaktu byłoby błędne.

Kontrola `r0002` dodatkowo potwierdza rozdzielenie czynników: oba nowe warianty wybierają dokładnie ten sam plik obrazu przy 0 ms i dają identyczną predykcję 0,102152. Historyczny obraz przy tym czasie daje 0,115962.

## Decyzja i kolejność dalszych prac

Zachowujemy zamrożony G jako punkt odniesienia. Wynik uzasadnia ujednolicenie danych wejściowych przed następną zmianą architektury.

1. W nowej wersji danych ustalić jeden sposób dekodowania, zapisu i skalowania dla treningu oraz inferencji. Zachować kontrolę świeżego t0 i dopasowanie znaczników czasu. Nie podmieniać wejść historycznych wyników v13/v14.
2. Zachować te 20 prób jako zestaw kontroli technicznej: pochodzenie anotacji, wybór czasu, braki i odtworzenie predykcji. Nie wybierać na nich hiperparametrów ani globalnego przesunięcia czasu pod najniższą stratę.
3. Przed oceną jakości rozszerzyć anotacje AI według ustalonego doboru, obejmując również gole, oraz rozliczyć brakujące kontakty. Wybór nie może zależeć od tego, gdzie nowy wariant już poprawił stratę. Anotator nadal dostaje pakiet bez etykiet źródłowych i predykcji.
4. Dopiero po ustaleniu reguły wejść i protokołu wytrenować nowy baseline, a potem porównać jedną zmianę architektury na identycznych danych. Obecne 20 prób nie rozstrzyga między większą rozdzielczością a mniejszą liczbą dostrajanych bloków. Potwierdzenie poprawy nadal wymaga osobnej kohorty.

Punkty 1–4 opisują dalszą rundę. W tym etapie wykonano import i porównanie na istniejących anotacjach; nie uruchamiano treningu ani anotacji pozostałych 280 prób.

## Odtworzenie i weryfikacja

Polecenia oraz zasady ponowienia opisuje [procedura](v15_workflow.md#bieżący-etap-porównanie-na-20-anotacjach-ai). Główny wynik znajduje się w `cache/v15/ai_contact_comparison`. Jego `complete.json` zapisuje sumy kontrolne kodu, modeli, klipów, źródeł i wszystkich plików wynikowych. Nieukończony katalog `.building` nie jest uznawany za wynik.

- **131 testów przeszło**, w tym 17 nowych testów importu AI, granic czasu, braków, dowodów i porównań; Ruff bez błędów. Siedem ostrzeżeń pochodzi z istniejących testów i zależności.
- Ponowienie importu zachowało identyczne sumy CSV i oryginalnego eksportu. Ponowienie porównania zweryfikowało wynik bez ekstrakcji i inferencji.
- Walidator eksportu anotatora przeszedł kontrolę instrukcji, pakietu, zapisanych dowodów i znaczników czasu.
- `diagnostic_review.py verify` sprawdził istniejący pakiet 300 prób bez regeneracji.
- `verify_frozen_analysis.py` potwierdził brak zmian w 21 plikach v14 i 50 plikach v13_final, bez zapisu do zamrożonych katalogów.
- `git diff --check` oraz kontrola białych znaków nowych plików nie wykazały błędów.
