# Kontrakt pierwszej wersji danych z jednolitą ekstrakcją

Nowe dane powstają w `cache/v15/canonical_data_v1` z 20 istniejących anotacji AI. Historyczne pakiety, źródła inferencji i wyniki pozostają zamrożone. Celem jest odtwarzalność wejść i rozliczenie ich jakości, bez treningu ani zmiany etykiet.

## Źródło i czas

`collect_soccernet.py` zapisuje historyczne JPEG-i bezpośrednio z połowy meczu, ale klipy koduje ponownie przez `libx264`, `veryfast`, CRF 26. Pełne połowy dla obecnej próbki nie są dostępne lokalnie. PNG z takiego klipu nie odzyskuje informacji utraconej podczas wcześniejszej kompresji.

Wersja `canonical_clip_v1` używa wyłącznie istniejących klipów, jawnie oznaczonych jako `lossy_clip`. Nie miesza automatycznie klatek z klipu i nagrania połowy. Manifest podaje SHA-256 klipu, tożsamość połowy, oczekiwany SHA-256 nagrania źródłowego i jego dostępność. Przywrócenie nagrań wymaga osobnej wersji danych oraz potwierdzenia mapowania czasu anotacji.

Moduł `xgpoc/data/canonical_frames.py` udostępnia:

- `VideoTimeline`: format czasu, rzeczywiste znaczniki PTS, rozmiar obrazu i format źródła.
- `DecodedFrame`: indeks klatki, PTS, czas względem kotwicy i natywne piksele BGR.
- `extract_inputs(clip, lower_ms, anchor_ms=4000)`: kontrolę t0 i klatkę precontact. Sprawdza zgodność czasów ffprobe z OpenCV i kompletność dekodowania.
- `write_png`: zapis bez zmiany rozmiaru i kontrolę identyczności pikseli po odczycie.
- `image_quality`: pomiary jasności, ostrości i zmian między sąsiednimi klatkami. Pomiary służą przeglądowi, nie automatycznemu usuwaniu trudnych prób.

Czas względem kotwicy wynika z PTS względem pierwszej klatki, pomniejszonego o 4000 ms. Ta konwencja musi zgadzać się z pakietem anotacji. Klipy zdarzeń przed czwartą sekundą połowy wymagają nowego pakietu z prawidłową kotwicą; obecna wersja nie przenosi na nie domyślnego założenia.

## Wyjście i braki

`shots.csv` ma jeden rekord na próbę, a `frames.csv` dwa rekordy: `t0` i `precontact`. Kontakt niepewny lub wcześniejsza ekspozycja na anotacje pozostają jawnym brakiem precontact. Błąd dekodowania jest statusem technicznym. Nieznany wynik AI pozostaje osobnym polem obok niezmienionej etykiety źródłowej. Żaden status nie oznacza automatycznie niezależnie potwierdzonej prawdy.

Wybór precontact wymaga czasu ≤ 0 i ściśle przed dolną granicą przedziału AI. Zapis obejmuje szerokość przedziału i odstęp wejścia od niego. Reguła nie korzysta z predykcji G.

Manifest obejmuje kod, wersje bibliotek i narzędzi, dane wejściowe, obrazy oraz tensor po istniejącym przetwarzaniu G. Powtórzenie ma sprawdzić gotowy wynik bez inferencji. Budowa odbywa się w osobnym katalogu tymczasowym i kończy atomową zmianą nazwy; nieukończona budowa nie blokuje nowej próby ani nie staje się wynikiem.

## Kontrolowane porównanie

Na tym samym indeksie klatki porównujemy dekodowanie OpenCV i systemowego FFmpeg. Osobno kodujemy natywny PNG przez FFmpeg do JPEG z jakością 3 i mierzymy zmianę po odczycie. Historyczny JPEG jest zewnętrznym punktem odniesienia; różnicy względem niego nie przypisujemy w całości CRF 26 bez oryginalnego nagrania.

Osobny audyt uruchamia zamrożony G na tych obrazach, stosując istniejący `ImageDataset`, kalibrację i regułę karnych. Sprawdza odtworzenie wcześniejszych predykcji i rozdziela zmianę czasu, dekodera oraz zapisu. Nie ocenia generalizacji na 20 przykładach bez goli.

Odtworzenie ma dwa poziomy. Historyczne 72 wejścia liczone w pierwotnym składzie batcha muszą zgadzać się do 1e-5; to dowód niezmienności modelu i potoku. Te same obrazy w innym składzie batcha różnią się do około 1e-4 prawdopodobieństwa, bo cuDNN dobiera kernele według kształtu batcha (zmierzone 2026-09-10: do 5e-4 w logitach). Ten szum jest zapisywany i porównywany z progiem 1e-3; efekty, które mierzymy, są rzędu 1e-2.

## Raport jakości

`scripts/report_data_quality.py` czyta ukończoną wersję danych i indeks. Rozdziela błędy techniczne od trudnych, poprawnych przykładów; żadna flaga nie usuwa próby. Dodatkowo liczy zmianę między kolejnymi klatkami w oknie od −1600 do 0 ms jako nieuwiarygodniony wskaźnik cięcia, kalibrowany na materiale po strzale z tych samych klipów, oraz wyszukuje w indeksie inne strzały z tej samej połowy w oknie ±15 s. Sąsiednie próby to kandydaci na tę samą akcję, dobitkę lub powtórkę; raport sprawdza, czy pozostają w jednej partycji danych.

## Warunki zakończenia

Testy obejmują granice czasu, niezerowy początek PTS, uszkodzone dane, niepewne anotacje, tożsamość pikseli i tensorów oraz wykrycie zmienionych plików. Rzeczywiste wykonanie musi rozliczyć wszystkie 20 prób, wygenerować listę brakujących nagrań, sprawdzić duplikaty w tej próbce i przejść ponowną kontrolę wyników. Zamrożone v13/v14 muszą zachować sumy kontrolne.
