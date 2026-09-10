# Dokładniejszy przegląd kontaktu

Arkusze co 200 ms pozostawiły 164 niepewne przypadki. Narzędzie prepare_contact_audit.py potrafi teraz przygotować wyłącznie ten podzbiór z krokiem 40 ms. Każdy arkusz obejmuje 41 żądanych pozycji od -800 do +800 ms względem dotychczasowej kotwicy w 4. sekundzie klipu. Rozdzielczość czasowa nagrania i zachowanie dekodera nadal ograniczają dokładność; krok 40 ms nie jest gwarancją błędu kontaktu mniejszego niż 40 ms.

Uruchomienie z katalogu experiments/claude_fable:

```bash
./run.sh python scripts/prepare_contact_audit.py \
  --uncertain-review cache/v13/contact_review/combined_review.csv \
  --step-ms 40 --out cache/v13/contact_review_dense
```

Nie używać starego katalogu arkuszy. Stare obrazy nie mają dowodu zgodności ze źródłem i narzędzie odrzuca ich ponowne użycie. Nowe arkusze mają pliki JSON z identyfikatorem strzału, sumą kontrolną nagrania, obrazu i kodu generatora oraz ustawieniami czasu. Zmiana źródła lub uszkodzenie obrazu powoduje błąd zamiast cichego użycia niewłaściwego materiału. Ponowienie niekompletnego generowania odtwarza brakujące obrazy. Narzędzie zakłada jednego zapisującego do katalogu.

Niepewne próbki muszą należeć do oficjalnego train, a ścieżka klipu musi zgadzać się z indeksem. Zachowują identyfikatory wcześniejszego przeglądu. Nowy sample.csv i sampling.json opisują podzbiór oraz pochodzenie poprzedniej oceny.

Arkusze zawierają klatki po strzale i służą wyłącznie do audytu. Nie włączać ich do wejścia xG. Dalsza ocena powinna rozdzielać strzał końcowy, podanie i dobitkę; przy zasłoniętym kontakcie pozostawić niepewność. Same materiały nie są nowymi etykietami ani potwierdzoną poprawą modelu.

## Stan po ponowieniu (2026-09-09)

Generowanie zakończone: 164 arkusze, 127 meczów, komplet plików JSON z dowodem pochodzenia. Ponowne uruchomienie tego samego polecenia zakończyło się kodem 0 bez regeneracji: sumy SHA256 wszystkich 331 plików katalogu są identyczne przed i po. Ocena 164 niepewnych kontaktów przez człowieka nadal nie została wykonana; do tego czasu te strzały mają w kontrakcie czasu status „przedział nieznany”.

Pomiar rzeczywistego czasu klatek wejściowych względem kotwicy klipu (skrypt `scripts/contact_timing_report.py`, wynik w `cache/v13/contact_timing/`) wykazał, że nominalne offsety −1600…0 ms zgadzają się z klatką zdekodowaną z klipu w 2849 z 2857 zmierzonych przypadków; jedna klatka była spóźniona o 80 ms, a 143 klatek nie dało się jednoznacznie dopasować w statycznych scenach. Kotwica arkuszy i kotwica klatek treningowych są więc tym samym momentem.
