# Przegląd wdrożenia v1.3, 2026-09-08

Aktualizacja 2026-09-09: treningi, zamrożenie i ocena zostały zakończone. Ponowny przegląd pierwszych 100 arkuszy również jest kompletny. Poniższy tekst zachowuje stan z 8 września; aktualne wyniki i ograniczenia zawiera [raport końcowy rundy](quality_v13_results_2026-09-09.md).

Nie potwierdzono jeszcze przyrostu jakości. Trwa porównanie rozwojowe; historyczny test nie uczestniczy w wyborze modeli. Nowy podział starych danych nie zastępuje testu na nowych meczach.

## Zaakceptowane uwagi niezależnego recenzenta

Przegląd skillem interrogate wykonał gpt-5.6-sol z wysiłkiem max. Pozostali dostawcy nie byli dostępni. Wszystkie sześć uwag przyjęto po sprawdzeniu kodu:

1. Analiza po otwarciu etykiet była zmienna. Freeze zapisuje teraz kod analizy, pary porównań, liczbę replik bootstrapu, ziarno i jednostkę losowania. Zakończona analiza ma sumy kontrolne raportów; ponowienie z innymi ustawieniami jest odrzucane.
2. Warianty mogły pochodzić z różnych wersji kodu. Suite przypina wspólny snapshot źródeł, sprawdzany przez trening i ocenę.
3. Tożsamość modeli G nie obejmowała obrazów. Dodano sumy kontrolne treści klatek i maski nakładek, kontrolowane przed i po treningu.
4. Karne wpływały na kalibrację zwykłych strzałów, mimo późniejszego zastępowania ich predykcji stałą. Wybór epoki korzysta teraz ze strzałów innych niż karne; Platt również dopasowuje się tylko do nich. Raportowany cel wyboru uwzględnia stałą częstość karnych wyliczoną wyłącznie z fit.
5. Bootstrap losował mecze, choć podział wykonywano po grupach gospodarzy. Porównania losują teraz całe grupy gospodarzy. Ocena obejmuje tylko 22 takie grupy, więc niepewność może być duża.
6. Dodatkowa pojemność modelu mogła tłumaczyć efekty czasu i geometrii. Dodano time_mean_t0 i time_gru_t0 z powtarzaną klatką t0 oraz geometry_zero. Mają te same architektury co odpowiadające im warianty informacyjne.

Nie było uwag odrzuconych. Poprawki wymagają jeszcze potwierdzenia na pełnych artefaktach i końcowym porównaniu.

## Weryfikacja i wykonanie

- 71 testów CPU przeszło; Ruff przeszedł dla nowego kodu.
- Przygotowano pełny cache kontekstów dla 8801 strzałów: ROI, kontrola tej samej powierzchni, pięć klatek od -1600 do 0 ms i geometria z głównej klatki.
- Osiem rzeczywistych próbek przeszło przygotowanie i sześć forwardów modeli. Podobieństwo cosinusowe nowej reprezentacji t0 do starego cache'u wynosiło co najmniej 0,9999989 w tej małej kontroli.
- results/v13 pozostaje wykluczone z porównania z powodu błędnego zapisu pochodzenia kodu.
- Pierwsza próba results/v13_verified została zatrzymana przez zabezpieczenie przed zmianą źródeł podczas treningu. Nie powstał ukończony model.
- Aktualna kolejka zapisuje results/v13_final; log: /tmp/xg-v13-final.log. Kolejka kończy się wyborem na select i kontrolą replay. Nie otwiera automatycznie etykiet evaluate.

## Dane i adnotacje

Wszystkie 300 arkuszy ma pierwszą ocenę wizualną. Surowe statusy zawierają 143 oceny niepewne, 71 „przed”, 58 „kontakt” i 28 „po”. Nie są to wiarygodne estymaty częstości błędów datasetu: recenzenci różnie klasyfikowali przedziały dotykające t0, a kontrola części starszej paczki wykazała możliwe pomylenie podania z finalnym strzałem. Raport zawiera też status wyprowadzony jednolicie z przedziałów oraz rozkład według źródła oceny. Cztery rekordy wymagają rozstrzygnięcia brakujących przedziałów.

Ponowna ocena pierwszych 100 arkuszy została przerwana przez limit użycia agentów po zapisaniu 18 rekordów. Nie zastosowano automatycznych korekt czasu ani etykiet wyniku.

SoccerTrack ma lokalnie osiem kompletnych nagrań meczów, lecz żaden mecz nie ma kompletu zaufanych kotwic goli według obecnego parsera. Raport external_eligibility.json zapisuje te braki. SoccerReplay wymaga zatwierdzonego dostępu, a archiwa SoccerNet BAS hasła. Nie utworzono z tych danych pozornie kompletnego testu xG.

## Pozostała praca

Dokończyć treningi i odtworzyć modele z zapisanych wag; sprawdzić raport wyboru bez otwierania evaluate; zamrozić analizę i wykonać ocenę ze sparowanymi przedziałami ufności; ocenić wyniki kontroli pojemności; rozstrzygnąć sporne adnotacje i dostęp do nowych danych. Wybór najlepszego wariantu spośród wielu porównań wymaga ostrożności wobec wielokrotnego testowania. Sam dodatni wynik pojedynczej ablacji nie uzasadnia deklaracji potwierdzonego przyrostu jakości.
