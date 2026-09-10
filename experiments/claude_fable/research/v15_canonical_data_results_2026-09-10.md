# Jednolita ekstrakcja na 20 anotacjach AI: dane, audyt, raport jakości

Zbudowano wersję danych `canonical_clip_v1` z jednym dekoderem, natywnymi pikselami w PNG i manifestem wiążącym każdą klatkę z indeksem, PTS i przedziałem kontaktu. Audyt na zamrożonym G rozdzielił dekoder, dodatkową kompresję JPEG, historyczne wycięcie i czas. Raport jakości nie znalazł błędów technicznych; 12 z 20 prób ma flagi przeglądowe, które opisują trudność, nie wykluczają.

Dekoder nie ma wpływu: OpenCV i systemowy FFmpeg dają identyczne piksele i tensory dla wszystkich 20 klatek. Ponowna kompresja JPEG zmienia xG średnio o 0,15 p.p., różnica względem historycznego JPEG wynosi 0,90 p.p., a przesunięcie czasu do klatki przed kontaktem 0,98 p.p. Większości różnicy między historycznym a nowym wejściem nie da się przypisać samemu JPEG; pozostałą część tworzy wcześniejsze transkodowanie klipu lub wyrównanie, którego bez nagrań połówek nie rozdzielimy.

To nadal 20 prób bez gola, z ról fit i select. Wyniki służą kontroli technicznej procesu, nie ocenie jakości modelu.

## Zakres

Praca w worktree `claude-fable/xg-poc`, HEAD `1343f5b` z lokalnymi zmianami. Kod inferencji, wagi i kalibracja pozostały niezmienione; `verify_frozen_analysis.py` potwierdził 21 plików v14 i 50 plików v13_final.

| Krok | Skrypt | Wynik |
| --- | --- | --- |
| Dane | `scripts/prepare_canonical_data.py` | `cache/v15/canonical_data_v1` |
| Audyt ekstrakcji | `scripts/audit_canonical_data.py` | `cache/v15/canonical_extraction_audit_v1` |
| Raport jakości | `scripts/report_data_quality.py` | `cache/v15/canonical_quality_v1` |

Każdy katalog ma `complete.json` z sumami kodu, wejść i plików wynikowych. Powtórzenie sprawdza sumy i nie liczy nic ponownie. Zmiana kodu ekstraktora unieważnia weryfikację v1, więc następna zmiana wymaga nowego katalogu wersji.

## Dane i czas

Wejście: 20 anotacji `model_review` z eksportu `20260909T135428Z_codex`, ten sam pakiet i te same klipy co w [porównaniu AI](v15_ai_contact_results_2026-09-10.md). Klipy są jedynym dostępnym źródłem; manifest oznacza je jako `lossy_clip` i wymienia 20 brakujących nagrań połówek z oczekiwanymi sumami SHA-256 i rozmiarem 20,3 GB.

Każda klatka w `frames.csv` ma indeks w klipie, PTS, `time_base`, czas w klipie, czas względem kotwicy oraz odstęp od dolnej granicy przedziału AI. Czas względem kotwicy to PTS pomniejszony o PTS pierwszej klatki i 4000 ms; ffprobe i OpenCV zgadzają się w granicach 0,5 ms na wszystkich klatkach. Czasy w pakiecie anotacyjnym pokrywają się z tak znormalizowanymi czasami, więc dowody anotatora odnoszą się do tych samych klatek.

Reguła wyboru: czas ≤ 0 i ściśle przed dolną granicą. Pokrycie:

| Wariant | Dostępne | Brak |
| --- | ---: | --- |
| `t0` | 20 | |
| `precontact` | 17 | 3 niepewne kontakty (`r0012`, `r0014`, `r0019`) |

W 16 próbach odstęp od dolnej granicy wynosi 40 ms. W `r0002` kontakt zaczyna się według AI przy +80 ms, więc klatka precontact jest tą samą klatką co t0. W 16 z 17 przedziałów historyczne t0 leży na początku przedziału lub po nim; ta obserwacja dotyczy prowizorycznych przedziałów AI.

Etykiety: `shots.csv` zachowuje etykietę źródłową i osobno ocenę AI wyniku oraz przypisania gola. Cztery próby mają nieznany wynik AI; żadna nie ma niezgodności z etykietą źródłową. Zgodność nie jest niezależnym potwierdzeniem.

## Audyt ekstrakcji

Zamrożony zespół trzech `g_base` policzył 97 wejść: 20 kanonicznych t0, 17 precontact, 20 klatek zdekodowanych przez FFmpeg, 20 kontroli JPEG q3 i 20 historycznych JPEG. Wszystkie tensory zgadzają się z sumami z manifestu.

| Porównanie | Pary | Średnia ΔxG, p.p. | Średnia \|ΔxG\|, p.p. | Maks. \|ΔxG\|, p.p. | MAE pikseli | Identyczne tensory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFmpeg − kanoniczne t0 | 20 | 0,000 | 0,000 | 0,000 | 0,00 | 20 |
| JPEG q3 − kanoniczne t0 | 20 | −0,029 | 0,146 | 0,397 | 1,69 | 0 |
| kanoniczne t0 − historyczne t0 | 20 | −0,381 | 0,901 | 6,003 | 3,22 | 0 |
| precontact − kanoniczne t0 | 17 | −0,482 | 0,982 | 4,033 | 11,07 | 1 |

Największe zmiany dotyczą `r0017` w obu ostatnich wierszach, zgodnie z poprzednim porównaniem. Kontrola JPEG koduje kanoniczny PNG do MJPEG q3 przez FFmpeg; historyczny JPEG powstał z nagrania połówki innym torem, więc ostatni wiersz nie jest czystym pomiarem kodeka.

Odtworzenie: 72 historyczne wejścia policzone w pierwotnym składzie batcha dały maksymalny błąd 9,7e-17. Te same obrazy w składzie audytu różnią się do 1,4e-5. Osobna sonda pokazała, że zmiana rozmiaru batcha lub sąsiadów przesuwa logity zamrożonego G do 5,4e-4, czyli do około 1,2e-4 prawdopodobieństwa; cuDNN dobiera kernele według kształtu batcha przy włączonym TF32. Wcześniejsze odtworzenia rzędu 1e-16 zachodziły tylko przy identycznym składzie. Audyt zapisuje oba poziomy i stosuje próg szumu 1e-3; efekty w tabeli są rzędu 1e-2.

## Raport jakości

Flagi techniczne: 0 prób. Flagi przeglądowe: 12 prób.

| Flaga | Kategoria | Próby |
| --- | --- | ---: |
| niepewny kontakt, brak precontact | przegląd | 3 |
| nieznany wynik AI | przegląd | 4 |
| niewidoczna piłka | przegląd | 3 |
| niewidoczny bramkarz | przegląd | 4 |
| przedział kontaktu szerszy niż 80 ms | przegląd | 3 |
| odstęp precontact większy niż 40 ms | przegląd | 1 |
| sąsiednia próba w tej samej połowie | przegląd | 1 |
| historyczne t0 na początku przedziału lub po nim | informacja | 16 |
| precontact identyczne z t0 | informacja | 1 |

Miara cięcia: średnia bezwzględna zmiana między kolejnymi miniaturami 64×36. W oknie od −1600 do 0 ms maksimum wynosi 0,026 (mediana 0,012). Na materiale po strzale z tych samych klipów mediana wynosi 0,072, maksimum 0,207, a 10 z 20 klipów przekracza próg 0,08. Trzy najwyższe piki obejrzano: `r0011` (+3480 ms), `r0009` (+3280 ms) i `r0002` (+3880 ms) to cięcia z ujęcia szerokiego do zbliżenia. Miara odróżnia te cięcia od ruchu w oknie wejścia na tej próbce; nie jest zweryfikowanym detektorem powtórek.

Sąsiednie próby: `r0002` ma w indeksie drugi strzał 1489 ms później (`Shots off target`, ta sama partycja `fit`). To kandydat na dobitkę albo tę samą akcję pod dwoma identyfikatorami; anotator AI ocenił `r0002` jako `ordinary`, więc wymaga rozstrzygnięcia przy przeglądzie przypisania gola. Podział ról według gospodarza trzyma wszystkie strzały z jednego meczu w jednej partycji, więc konflikt partycji jest tu wykluczony konstrukcyjnie; kontrola pozostaje na wypadek zmiany protokołu. Kontrola meczu pod dwiema nazwami i bliskich duplikatów między splitami istnieje w `scripts/audit_leakage.py` i nie była powtarzana.

## Ograniczenia

- Próbka nie zawiera goli, główek ani rozstrzygniętych dobić; nie mierzy jakości modelu.
- Przedziały kontaktu są prowizoryczne, z jednego anotatora AI. Drugie przejście z innym aliasem i bez wcześniejszych ocen wykryje rozbieżności, ale zgodność nie potwierdza prawdziwości.
- Brak nagrań połówek uniemożliwia bezstratne źródło i rozdzielenie CRF 26 od wyrównania.
- Progi raportu (0,08 dla cięcia, 80 ms dla szerokości, ±15 s dla sąsiadów) są parametrami opisu, nie wynikiem walidacji.

## Następny krok

Rozszerzyć anotacje o gole, główki i dobitki według ustalonego doboru, z osobnym raportowaniem próbki losowej i celowanej, i zbudować `canonical_clip_v2` na większym zbiorze tym samym ekstraktorem. Dopiero na tej wersji trenować nowy baseline z tą samą regułą czasu dla wszystkich wariantów.

## Weryfikacja

- 155 testów przeszło, w tym 5 nowych dla raportu jakości; Ruff bez błędów; `git diff --check` czysty.
- Powtórzenie każdego z trzech skryptów sprawdza sumy bez ponownego liczenia.
- `verify_frozen_analysis.py`: v14 i v13_final niezmienione.
