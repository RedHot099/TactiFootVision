# Wyniki v1.3, 2026-09-09

Zakończono 31 treningów i jednorazową ocenę zamrożonych zespołów. Nie wykazano przyrostu jakości wynikającego z proponowanych rozszerzeń. Rekomendacją tej rundy pozostaje g_base; nie promujemy ROI, czasu, geometrii ani globalnego przesunięcia -400 ms. Nie jest to dowód, że te rodzaje informacji nigdy nie pomogą.

## Zakres i wyniki

Podział obejmuje 4515 strzałów fit, 1309 select, 1042 calibrate i 1935 evaluate. Ocena rozwojowa zawiera 90 meczów i 22 grupy gospodarzy. Grupy gospodarzy nie oznaczają pełnego rozdzielenia wszystkich klubów, ponieważ kluby występują też jako goście. Historyczny test nie służył do wyboru tej rundy. Istniejące dane były wcześniej używane badawczo; nowy podział nie zastępuje niezależnego testu na nowych meczach.

Wyniki dla wszystkich strzałów, w tym 26 karnych obsługiwanych wspólną stałą wyliczoną na fit:

| Wariant | Liczba seedów | Log-loss ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
| g_base | 2 | 0,385748 | 0,715008 |
| g_pre400 | 2 | 0,392403 | 0,704835 |
| full | 3 | 0,387946 | 0,700107 |
| roi | 3 | 0,387274 | 0,701801 |
| control | 3 | 0,387328 | 0,703936 |
| time_mean | 3 | 0,386919 | 0,702468 |
| time_gru | 3 | 0,386776 | 0,703347 |
| geometry | 3 | 0,388428 | 0,699311 |
| time_mean_t0 | 3 | 0,387227 | 0,701515 |
| time_gru_t0 | 3 | 0,386779 | 0,703227 |
| geometry_zero | 3 | 0,386362 | 0,703523 |

Wszystkie 11 zaplanowanych porównań log-loss ma 95% przedział bootstrapu obejmujący zero. Przykłady redukcji straty, dodatnia wartość sprzyja pierwszemu wariantowi:

- ROI względem kontrolnego wycinka: mediana 0,000046; przedział [-0,002012; 0,001925].
- GRU względem tej samej architektury z powtarzaną klatką t0: 0,000003; [-0,000262; 0,000280].
- Geometria względem wyzerowanej geometrii: -0,002096; [-0,004382; 0,001771].
- G -400 ms względem G t0: -0,006720; [-0,013198; 0,003352].

Przedziały pochodzą z 2000 losowań całych grup gospodarzy. Są nominalne i eksploracyjne, bez korekty wielokrotnych porównań. Raportowana mediana bootstrapu nie jest dokładnie różnicą punktowych strat. G pozostaje rekomendacją wybraną na select i ma najlepszy punktowy log-loss evaluate; nie ogłaszamy jego statystycznie dowiedzionej przewagi nad każdym wariantem. Starsze AUC około 0,793 dotyczyło innej kohorty i procedury, więc nie należy interpretować tej różnicy jako regresji modelu.

## Kontrola wykonania

31 plików complete.json opisuje 4 treningi G i 27 treningów głowic. Freeze sprawdził sumy kontrolne artefaktów, zgodność źródeł, protokołu i pełny zestaw seedów. Kalibrację dopasowano do średniej logitów zespołu na osobnym calibrate. Ocena i analiza są zamrożone w results/v13_final; ponowne wywołanie evaluate potwierdziło istniejące sumy kontrolne zamiast ponownego zapisu wyników.

Odtworzenie g_base/seed0 i roi/seed0 w nowych procesach na 32 próbkach dało maksymalny błąd prawdopodobieństwa odpowiednio 8,33e-17 i 9,02e-17. To kontrola dwóch reprezentatywnych artefaktów, nie replay wszystkich 31 modeli. 71 testów przeszło ponownie; po zmianie domyślnego pliku adnotacji przeszły też 2 testy podsumowania i Ruff.

Źródła wyniku: ../results/v13_final/evaluation.csv, paired_evaluation.csv, frozen.json, analysis_complete.json. Wyniki wcześniejszych nieprawidłowych prób nie zostały włączone.

## Ponowny audyt kontaktu

Zakończono ponowny przegląd wszystkich pierwszych 100 arkuszy: 18 przez drugiego recenzenta i 82 w głównym wątku. Surowy status różni się od pierwszego przeglądu w 75 przypadkach. Ta liczba obejmuje niezgodność konwencji przedziałów oraz błędne przypisanie wcześniejszego kontaktu; nie jest estymatą błędów źródłowych etykiet. Pierwotne adnotacje zachowano. Kontrola arkusza 150 podtrzymała niepewność, a w 250 zawężono przedział do [0; 200] ms w osobnym pliku.

Łącznie 300 arkuszy ze 186 meczów ma dokładnie jedną aktualną ocenę bez niespójności statusu z przedziałem: 164 niepewne, 130 przedziałów zawierających t0, 5 kontaktów po t0 i 1 przed t0. Jest to przegląd modeli na siatce co 200 ms, nie ludzki wzorzec kontaktu. Nie zmieniono etykiet goli ani klatek treningowych. Podsumowanie można odtworzyć skryptem summarize_contact_audit.py; pliki first_pass_* zachowują wcześniejszy raport.

## Dalszy kierunek

1. Uzupełnić kontakt w niepewnych próbkach z ciągłego wideo i uzyskać niezależną walidację człowieka. Przy obecnej niezgodności recenzentów automatyczne przesuwanie klatek nie ma uzasadnienia.
2. Pozyskać nowe mecze z pełnymi etykietami strzałów i goli. Lokalny SoccerTrack ma 8 kompletnych nagrań, ale 0 meczów spełnia kontrolę kompletności zaufanych kotwic goli. SoccerReplay wymaga zatwierdzenia dostępu, a BAS hasła do archiwów.
3. Kolejną zmianę architektury trenować i wybierać wyłącznie na fit/select/calibrate. Część evaluate tej rundy jest już otwarta i nie może ponownie potwierdzać poprawy wybranej na podstawie jej wyników. Rozważyć integrację ROI z dostrajanym G dopiero po poprawie kotwic; obecne ablacje kontekstu używały zamrożonych reprezentacji.

Zakończono rundę porównawczą i poprawki audytu. Cel wykazania przyrostu jakości pozostaje nieosiągnięty; dalsze eksperymenty wymagają lepszych kotwic i świeżej oceny, a nie promowania nieistotnych różnic.
