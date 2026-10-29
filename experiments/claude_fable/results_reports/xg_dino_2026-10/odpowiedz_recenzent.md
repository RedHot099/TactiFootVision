# Odpowiedź na uwagi recenzenta 1–3 (DINOv2 w modelu xG)

2026-10-29. Dotyczy uwag z 27 października 2026, punkty 1–3: ablacja DINOv2, rozszerzone sondy liniowe, zgodność DINOv2 z detektorem. Pełne raporty leżą obok: `results.md` (faza xg-dino, commit `7025b34`) i `info.md` (faza xg-dino-info, commit `cbc0a21`). Każda liczba poniżej pochodzi z tych raportów i jest tam sprawdzona automatycznie ze źródłem.

**Jak czytać wyniki.** To wyniki rozwojowe, nie niezależny test. Wszystkie modele liczyliśmy out of fold na 328 meczach treningowych (7 062 strzały, 902 gole), na tych samych foldach i z tą samą kalibracją, z przedziałami 95% z bootstrapu par po meczach. 172 mecze odłożone zostały już zużyte na wybór detektora, więc odczyty na nich są tylko opisowe. Dane StatsBomb służą wyłącznie do walidacji. Nie wchodzą do żadnego modelu ani nie są celem uczenia. Poniżej występują tylko w postaci zagregowanych liczb.

## Punkt 1. Ablacja DINOv2

Dziękujemy za tę propozycję. Najpierw jedna korekta. Wynik −0.0004 nie dotyczył obrazu dodanego do automatycznej geometrii. To test H2 z badania xg-vendor: model z obrazem i pozycjami StatsBomb (freeze frame) wobec modelu na samych pozycjach StatsBomb, Δ = −0.0004 [−0.0026; +0.0018], równoważne przy δ = 0.005 (najmniejszy margines 0.0022).

Ablację A/B/C policzyliśmy teraz out of fold na 328 meczach treningowych (7 062 strzały), z tymi samymi foldami i kalibracją dla każdego wariantu i z bootstrapem par po meczach. „Identyczny zbiór testowy” oznacza tu więc out of fold, a wynik jest rozwojowy.

| Wariant | Wejście | log-loss |
|---|---|---|
| A | sam DINOv2 (G-combo) | 0.3431 |
| B | sama automatyczna geometria | 0.3422 |
| C | DINOv2 i geometria | 0.3325 |

- C−B = −0.0097 [−0.0131; −0.0064]. DINOv2 wnosi mierzalną wartość ponad geometrię.
- B−A jest nierozstrzygnięte.
- Kontrole wycieku detektora (mecze bez obrazów SoccerNet-v3, strzały, których detektor nie widział) wskazują ten sam kierunek.
- B liczy się 5.3× dłużej niż A, głównie przez homografię.
- B nie jest równoważne C, więc nie rekomendujemy usunięcia DINOv2.
- Wcześniej, opisowo na 172 zużytych meczach, sweep dał dla C −0.0091 [−0.0142; −0.0042] względem samego obrazu.

## Punkt 2. Rozszerzone sondy liniowe

Rozszerzyliśmy sondy o kąt, odległość najbliższego rywala, liczbę rywali w promieniu 3 m (teraz w metrach), głębokość bramkarza i jego odsunięcie od środka bramki. Sondy policzyliśmy na zamrożonym DINOv2 i na G-combo, w każdym z 24 bloków. Cele pochodzą z automatycznej geometrii, więc sondy mówią, co DINOv2 koduje z tego, co mierzy detektor.

- Wcześniej (H3) pokazaliśmy odległość R² 0.711 i obrońcę w stożku AUC 0.761.
- Nowe: odczytuje się też kąt i głębokość bramkarza.
- Liczbę obrońców w stożku, odległość najbliższego rywala i odsunięcie bramkarza embedding koduje ponad dystans, choć sam nie przekracza progu odczytu.
- Liczby rywali w promieniu 3 m nie koduje, a obecność rywala w 3 m tylko słabo.
- Większość geometrii pojawia się w blokach 22–23. Dostrojony G-combo czyta ją tam lepiej niż zamrożony DINOv2, a do bloku 20 oba są praktycznie równe.

## Punkt 3. Zgodność DINOv2 z detektorem

Na 2 721 strzałach z geometrią referencyjną StatsBomb (opisowo, na zużytych meczach):

- Detektor jest bliżej referencji niż sondy DINOv2 na każdej cesze poza odległością najbliższego rywala, gdzie wynik jest remisowy.
- Przypadków „detektor się myli, DINOv2 ma rację” nie ma więcej, niż wynika z konstrukcji testu.
- Proste połączenie obu źródeł nie poprawia detektora.
- Detektor ma systematyczne przesunięcie (gubi zasłoniętych obrońców blisko strzelca), a sondy je dziedziczą.
- Detektor w klatce t0 jest wyraźnie bliżej referencji niż w klatce −400 ms.

Wniosek: DINOv2 nie koryguje błędów geometrii. Jego wkład w xG to inna informacja. Sprawdziliśmy to osobno, w uzupełnieniu poniżej.

## Uzupełnienie: skąd bierze się wkład DINOv2

Po pierwszej odpowiedzi zostało otwarte pytanie, jaka informacja niesie zysk C−B. Sprawdziliśmy to z protokołem przypiętym (sha256) przed pierwszym dopasowaniem. Dane są te same: 7 062 strzały z 328 meczów treningowych, out of fold, bootstrap par po meczach, wyniki rozwojowe.

1. **Co kodują embeddingi.** Sondy liniowe pokazują, że embeddingi dobrze kodują tłok w polu karnym, stały fragment, kadrowanie i wielkość strzelca.
2. **Czy te atrybuty zastępują DINOv2.** Nie. Te same atrybuty policzone automatycznie z detektora, dodane do geometrii, nie zmniejszają wykrywalnie wkładu DINOv2. Ponad geometrię z tymi atrybutami DINOv2 nadal daje −0.0097 [−0.0136; −0.0057] log-loss. Część wkładu pokrywają tylko cechy jakości obrazu i detekcji, ale ten kanał może być skorelowany z wynikiem strzału, więc traktujemy go opisowo. Nasza metoda może potwierdzić taką redundancję, ale nie może jej wykluczyć. Wynik dla atrybutów treści jest więc nierozstrzygnięty, a nie negatywny.
3. **Podgrupy.** Wkład nie różni się wykrywalnie między główkami i strzałami nogą ani między stałymi fragmentami i grą z akcji.
4. **Maskowanie.** Zamaskowaliśmy fragmenty klatki i przeliczyliśmy je przez G-combo.
   - Po zamaskowaniu jednego obiektu (strzelca, piłki, bramkarza, bramki) zostaje od 0.58 do 0.94 wkładu.
   - Po zamaskowaniu wszystkich zawodników zostaje 0.14.
   - Gdy na klatce zostaje sam strzelec z piłką, zostaje 0.09.
   - Wkład jest więc rozłożony na układ zawodników na scenie, a nie zamknięty w jednym obiekcie.
   - Wdrożony model jest najbardziej wrażliwy na zamaskowanie bramkarza i okolicy bramki.

Wszystkie te wyniki mówią o redundancji predykcyjnej i wrażliwości na interwencję, nie o przyczynowej identyfikacji tego, czego DINOv2 używa.

## Rekomendacja

DINOv2 zostaje w modelu razem z geometrią (wariant C). Model na samej automatycznej geometrii nie jest równoważny wariantowi C, a żaden zestaw automatycznych atrybutów nie zastąpił DINOv2.
