# Slajd 2

## Tytuł
Pipeline na konkretnym fragmencie meczu

## Tekst na slajdzie
Wejście: 60 sekund meczu.

1. Detekcja obiektów
2. Detekcja punktów boiska
3. Estymacja homografii
4. Tracking
5. Projekcja do układu boiska
6. Eksport do analizy

Ten sam klip zasila wszystkie dalsze wyniki prezentowane w decku.

## Układ
- Po lewej `process_imgs.png`.
- Po prawej albo w dolnej części `intro_split_screen.png`.
- Tekst ograniczyć do listy 6 etapów i jednego zdania na dole.

## Co powiedzieć ustnie
Tutaj pokazujemy pełny przepływ danych. Z jednego wejściowego klipu przechodzimy przez detekcję obiektów, lokalizację geometrii boiska, wyznaczenie homografii, tracking oraz projekcję do współrzędnych boiskowych. Dzięki temu wynik końcowy nie jest tylko overlayem na obrazie, ale uporządkowaną reprezentacją przestrzenną.
