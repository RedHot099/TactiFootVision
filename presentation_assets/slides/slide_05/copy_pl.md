# Slajd 5

## Tytuł
Walidacja i liczby end-to-end

## Tekst na slajdzie
Walidacja względem StatsBomb360:

- 18 meczów
- 3314 eventów
- około 97% coverage
- medianowy błąd lokalizacji około 7.2-7.4 m

Pipeline daje stabilny i mierzalny wynik końcowy na danych meczowych.

## Układ
- Główny wykres: `statsbomb360_mean_dist_and_coverage.png`
- Drugi wykres mniejszy: `positional_error_ridgeplot.png`
- Tekst ograniczyć do 4 bulletów i jednego zdania wniosku.

## Co powiedzieć ustnie
Po demonstracji jakościowej pokazujemy walidację liczbową. Najważniejsze jest to, że porównujemy końcowe pozycje z referencją StatsBomb360, więc oceniamy cały pipeline end to end, a nie tylko pojedynczy komponent, taki jak detektor.
