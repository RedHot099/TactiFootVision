# Homografia: zdiagnozowaliśmy problem i skalę możliwej poprawy

**Kontekst:** SoccerNet-GSR `valid`, 58 sekwencji, 43,500 klatek. Porównanie izoluje samą homografię: używa GT footpointów w obrazie i mierzy błąd projekcji na boisko.

## Główne liczby

| Metryka | Obecny baseline `current_yolopose_7pt` | Kontrola oracle | Zmiana |
| --- | ---: | ---: | ---: |
| Mediana błędu | 93.7 m | 0.096 m | 973x mniej |
| Success@2m | 0.041% | 99.35% | +99.31 pp |
| Dostępność homografii | 23.3% | 98.1% | +74.8 pp |

## Przekaz

Obecny baseline używa ludzkich keypointów YOLO-pose jako punktów boiska, więc nie nadaje się do produkcji. Ewaluacja i oracle-control pokazują, że ścieżka image-to-pitch działa poprawnie, a kolejnym krokiem jest podpięcie realnego backendu kalibracji (`pnlcalib` / `sportlight`) zamiast YOLO-pose.

## Presenter note

Nie sprzedawać oracle jako backendu produkcyjnego. To upper-bound i walidacja metryk. Produkcyjna poprawa wymaga teraz artefaktów zewnętrznych metod kalibracji.

