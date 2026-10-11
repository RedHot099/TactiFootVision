# Reguła testu źródła czasu (xg-vendor, ticket 04)

Status: ustalona 2026-10-11, zanim porównano jakiekolwiek zdarzenie inne niż strzał. Skrypt `scripts/vendor_timing_source.py` odmawia pracy, gdy sha256 tej notatki różni się od stałej `RULE_SHA256` w `xgpoc/vendor/timing_source.py`. Hash trafia do rekordu werdyktu. Zmiana reguły po zobaczeniu wyników wymaga nowej notatki i nowego hasha, a obie wersje trafiają do raportu.

## Pytanie

Czy kotwica strzału SoccerNet i znacznik czasu StatsBomb pochodzą ze wspólnego zegara (jeden strumień zdarzeń zsynchronizowany z wideo), czy z dwóch niezależnych, precyzyjnych adnotacji tej samej transmisji? Dopóki tego nie rozstrzygniemy, czas StatsBomb nie jest dowodem momentu kontaktu.

## Co było znane przed tą notatką

- Rezydua czasu sparowanych strzałów (prototyp, odtworzone w tickecie 02): mediana 0 ms, IQR od −24 do +23,5 ms, 97,2% w ±200 ms. Z offsetem leave-one-out: 96,5% w ±200 ms.
- Rzuty rożne z prototypu (tylko podania kluczowe z rogu, okno 5 s): 220/244 dopasowanych, mediana +3 ms, 89% w ±200 ms.
- Przed napisaniem tej notatki oglądaliśmy tylko schematy tabel i liczności. W 135 sparowanych meczach jest 749 etykiet Substitution w SoccerNet (165 „not shown”) i 753 zmiany w StatsBomb. 60 zmian StatsBomb z drugiej połowy ma czas nie późniejszy niż pierwsze rozpoczęcie drugiej połowy (56 z nich ma czas 0). Żadnego rezyduum zmian, wrzutów z autu, wyjść piłki, wszystkich rogów, fauli, spalonych ani rozpoczęć nie liczyliśmy.

## Dane

- Zdarzenia StatsBomb: wersja danych `statsbomb_open_v1` (commit open-data 4b73468f), tabele `timing`, `shots`, `matches`. Surowe zdarzenia meczów służą tylko tabeli powodów (niżej). Każdy plik jest użyty dopiero po zgodności z hashem w rekordzie ukończenia (`complete.json`).
- Parowanie: zamrożone wyniki ticketu 02 (`results/xgv_pairing`, commit 177d65e), sprawdzone z ich rekordem ukończenia: `half_offsets.csv`, `match_map.csv`, `pairs.parquet`, `unpaired_sn.parquet`.
- Etykiety SoccerNet Labels-v2 135 sparowanych meczów. Skrót plików musi być równy temu z rekordu parowania.

## Zegar i rezyduum

- Offsety połówek są zamrożone z ticketu 02 i nie są dopasowywane od nowa. Połowa bez offsetu (`offset_source = none`) wypada; w danych takiej nie ma.
- Połowa h SoccerNet odpowiada okresowi h StatsBomb (h = 1, 2).
- Rezyduum etykiety: r = (pozycja SoccerNet − offset połowy) − czas zdarzenia StatsBomb, w ms. Dodatnie r: SoccerNet później.

## Dopasowanie etykiet do zdarzeń

Dla każdej trójki (mecz, połowa, typ etykiety):

1. Etykiety SoccerNet danego typu, z każdą widocznością („visible” i „not shown”).
2. Kandydaci: zdarzenia StatsBomb tego samego meczu i okresu, o typie przypisanym do etykiety (tabela niżej).
3. Przypisanie jeden do jednego, tym samym algorytmem co parowanie strzałów (`shot_pairing.assign`). Minimalizujemy sumę |r|. Każda etykieta ma opcję „bez pary” o koszcie równym oknu, więc algorytm nie wymienia dobrych rezyduów na jedną parę więcej. Zostają pary z |r| ≤ okno.
4. Drużyna nie wchodzi do dopasowania. Zgodność strony (gospodarz/gość) raportujemy osobno.

| Etykieta SoccerNet | Zdarzenia StatsBomb (kolumny tabeli `timing`) | Okno |
|---|---|---|
| Substitution | `type` = Substitution | 120 s |
| Throw-in | podanie z `detail` = Throw-in | 10 s |
| Ball out of play | `ball_out` = True (dowolny typ, także wrzut lub róg, którego podanie wyszło), oraz strzały z `out` = true w JSON zdarzenia albo z wynikiem Off T | 10 s |
| Corner | podanie z `detail` = Corner | 10 s |
| Kick-off | podanie z `detail` = Kick Off | 10 s |
| Foul | `type` = Foul Committed | 10 s |
| Offside | `soccernet_label` = Offside (zdarzenie Offside oraz podanie z wynikiem Pass Offside) | 10 s |

Okno zmian jest szerokie, bo przy niezależnych adnotacjach SoccerNet może zaznaczać zmianę dopiero, gdy pokaże ją transmisja. Zmian jest kilka na połowę, więc szerokie okno nie miesza par.

Wyjątek, zmiany w przerwie: zmianę StatsBomb z okresu 2 o czasie nie późniejszym niż pierwsze podanie Kick Off tego okresu (albo 0 ms, gdy takiego podania brak) nazywamy zmianą w przerwie. StatsBomb przypina ją do początku połowy, a nie do chwili zmiany. Para jest dopasowana normalnie, ale nie wchodzi do statystyk reguły, gdy zmiana StatsBomb jest zmianą w przerwie albo gdy etykieta SoccerNet leży przed początkiem okresu StatsBomb (pozycja − offset < 0). Takie pary raportujemy osobno.

## Strzały

Sparowane strzały ticketu 02 dają rezydua czasu, ale offset połowy jest medianą ich własnych różnic. Dlatego do reguły bierzemy rezyduum leave-one-out: offset strzału i to mediana surowych różnic (pozycja SoccerNet − czas StatsBomb) pozostałych sparowanych strzałów jego połowy. Rezydua z zamrożonym offsetem raportujemy obok.

## Statystyki (dla każdego typu, także osobno dla „visible” i „not shown”)

Liczba etykiet, liczba kandydatów, liczba par, odsetek etykiet z parą, mediana r, kwartyle, percentyle 10 i 90, mediana |r|, odsetek |r| ≤ 200, ≤ 500 i ≤ 1000 ms, odsetek r = 0, zgodność strony. Granice są włącznie.

## Reguła werdyktu

Oznaczenia:

- Z: pary zmian po wyłączeniu zmian w przerwie; n_Z to ich liczba;
- p_Z: odsetek Z z |r| ≤ 200 ms;
- m_Z: mediana |r| w Z;
- q_Z: odsetek etykiet Substitution w połowach z offsetem, które mają parę (wszystkie pary, także z przerwy);
- p_S: odsetek sparowanych strzałów z |r_LOO| ≤ 200 ms;
- p_R, n_R: odsetek rogów z |r| ≤ 200 ms i liczba par rogów.

Kroki, w tej kolejności:

1. Jeśli n_Z < 50 albo q_Z < 0,50, werdykt brzmi **„unresolved”**: za mało danych.
2. Jeśli p_Z ≥ 0,90, werdykt brzmi **„shared”**: wspólne źródło.
3. Jeśli m_Z ≥ 1000 ms, a kopnięcia się zgadzają (p_S ≥ 0,80, n_R ≥ 50 i p_R ≥ 0,80), werdykt brzmi **„independent”**: niezależne adnotacje.
4. W pozostałych przypadkach werdykt brzmi **„unresolved”**.

Kroki 2 i 3 się wykluczają, bo p_Z ≥ 0,90 daje m_Z ≤ 200 ms. Progi porównujemy wprost, bez przedziałów ufności.

## Skutki

- Rekord werdyktu ma schemat `xgv-timing-verdict-1` (`xgpoc/vendor/timing_verdict.py`). Ramię H4 oparte na rezyduum czasu (ticket 09) rusza tylko przy werdykcie „independent”.
- „shared”: czas StatsBomb nie jest dowodem kontaktu. Można go nadal używać do wyrównania zegarów, parowania i odzyskiwania goli. Kotwica populacji wtórnej (czas StatsBomb + offset) jest wtedy bezpieczna.
- „independent”: rezyduum czasu może być dowodem. Raport musi jednak podać, że klatki populacji wtórnej mogą leżeć kilkaset ms od kontaktu.
- „unresolved”: blokuje tak samo jak „shared”.

## Opis bez wpływu na werdykt

- Wrzuty z autu, wyjścia piłki, faule, spalone i rozpoczęcia raportujemy tymi samymi statystykami. Wyjście piłki ma lukę z definicji: StatsBomb stempluje zagranie, po którym piłka wyszła, a SoccerNet chwilę, w której wyszła. Dodatnie r jest tu oczekiwane.
- Wrażliwość statystyk zmian: okna 30, 60 i 300 s oraz same etykiety „visible”.
- Tabela powodów dla każdego niesparowanego strzału SoccerNet (285 w zamrożonym parowaniu, w tym 249 bez strzału StatsBomb w 10 s). Podaje najbliższe zdarzenie StatsBomb dowolnego typu z surowych zdarzeń meczu: typ, szczegół i odległość ze znakiem (zdarzenie − strzał na zegarze StatsBomb). Powód to pierwszy pasujący warunek:
  1. `sb_shot_within_10s`: strzał StatsBomb w 10 s;
  2. `dead_ball`: ostatnie zdarzenie przerywające grę przed strzałem leży najwyżej 10 s przed nim, a między nim a strzałem nie ma wznowienia. Zdarzenia przerywające: Offside, podanie Pass Offside, Foul Committed, zdarzenie z wyjściem piłki, Injury Stoppage, Half End. Wznowienia: podanie typu Throw-in, Corner, Free Kick, Goal Kick, Kick Off, Referee Ball-Drop, Half Start. Powód podaje typ zdarzenia przerywającego;
  3. `outside_half`: strzał leży przed Half Start albo po Half End okresu StatsBomb;
  4. `other_event_within_2_5s`: jakieś zdarzenie StatsBomb leży w 2,5 s;
  5. `no_event_within_2_5s`: w pozostałych przypadkach.

## Granice

- Progi 200 ms, 1000 ms, 0,80 i 0,90 to wybór ze specyfikacji i z tej notatki. Nie wynikają z rachunku mocy.
- Etykiety „not shown” SoccerNet zaznacza bez obrazu zdarzenia. Wchodzą do reguły, a raport pokazuje je osobno.
- Werdykt „shared” nie odróżnia kopii strumienia od dwóch adnotacji, które opierają się na tym samym zegarze transmisji. Mówi tylko, że czasu StatsBomb nie można traktować jako niezależnego pomiaru.
