# ball-detection: raport fazy 3

2026-10-27. Branch `research/ball-detection`, tickety 17–23 (`.scratch/ball-detection/issues/`), odpowiedź na punkty 4–7
uwag zewnętrznego recenzenta (`.scratch/reviewer-2026-10-27/notes.md`). Ten raport nie liczy niczego nowego: wszystkie
liczby pochodzą z raportów [17](17/results.md), [18](18/results.md), [19](19/results.md), [20](20/results.md),
[21](21/results.md) i [23](23/results.md) oraz z wcześniejszych [REPORT.md](REPORT.md) (faza 1) i [PHASE2.md](PHASE2.md)
(faza 2). Ticket 22 (odczyt poza domeną) nie został uruchomiony, bo warunek D5 nie jest spełniony (punkt 7 niżej).
Duże wyniki leżą pod `/mnt/storage/tactifoot_research/detector_data/ball/NN/`.

Odniesieniem jest kandydat fazy 1: dd2c 1280 + filtr 45 px + `track_ball` z ustawieniem z 04. Bez zmian względem fazy 1
i 2 nie powstał żaden nowy detektor ani nowy tracker.

## Podsumowanie: odpowiedzi na punkty 4–7

- **Pkt 4 (czy 5 m wystarcza).** Kandydat fazy 1 ma na grach 2+3 GSR valid: ≤ 1 m 37,3% [32,5; 42,4], ≤ 2 m 58,7%,
  ≤ 5 m 71,2% [64,6; 77,5], p50 1,13 m, p90 9,03 m, p95 14,87 m, 17,0% klatek bez pozycji (wśród wszystkich klatek
  mianownika). Progi 0,5 i 1 m leżą w szumie referencji: sama referencja GSR rzutowana naszą homografią ma medianę
  różnicy 0,94 m (środek ramki), więc to zgodność z GSR, nie dokładność fizyczna. Dla strzałów w GSR jest tylko 42
  klatek (orientacja); na 172 meczach xG piłka t0 ma pozycję w 62,8% strzałów i jest dalej od miejsca StatsBomb niż
  stopy strzelca (p50 1,78 m wobec 1,16 m).
- **Pkt 5 (interpolacja i wiarygodność).** Interpolowane pozycje są głównym źródłem błędów: 25,7% > 5 m wobec 9,8% dla
  zmierzonych. Wynik niepewności (regresja logistyczna) porządkuje pozycje lepiej niż sama długość luki (−1,5 pp
  [−2,2; −0,9] przy 10% odrzuconych), ale bramka D2 nie przechodzi: reguła z jednej gry nie przenosi się na drugą (47,9%
  odrzuconych złych, strata 7,3 pp). `track_ball` zostaje bez zmian; wynik niepewności proponuję jako pole pozycji.
- **Pkt 6 (piłka a xG).** Na 7 062 strzałach z 328 meczów rozwojowych (OOF, zamrożony stos) żadne ramię piłki nie
  poprawia log-loss wobec stóp strzelca `feet_t0`: C +0,0016 [+0,0001; +0,0029], S2 +0,0010 [−0,0014; +0,0035], piłka
  śledzona T −0,0000 [−0,0026; +0,0025]. Wszystkie trzy to „nierozstrzygnięte” (D3), z T najbliżej równoważności. Stopy
  zostają; piłka nie wnosi wykrywalnej wartości do log-loss.
- **Pkt 7 (uogólnienie i filtr).** Filtr 45 px jest odporny na rozdzielczość (1080p, 720p, 540p: 0,0% prawdziwych
  piłek usuniętych, > 5 m −6,1…−6,3 pp), wariant względny do wysokości ludzi nie jest lepszy. Nietkniętego zbioru
  z etykietami piłki nie ma: SoccerTrack v2 nie ma pól piłki w obrazie, reszta była w treningu, doborze albo odczytach.
  Wyników poza domeną nie ma, więc siła uogólnienia pozostaje ograniczona tak, jak pisał recenzent.

## Punkt 4: dokładność lokalizacji piłki

**Było wcześniej.** Mediana błędu i odsetek > 5 m wśród umieszczonych w karcie 02 i `04/results.md`; klatki zmierzone,
interpolowane i bez piłki w 04, 09, 12; test: zmierzone 89,1% dobrze, interpolowane 69,6% i 25,5% > 5 m (09); sufit bez
homografii (09); proxy „w powietrzu” i „pokrycie przez ludzi” dla top-1 (01); „blisko bramki” (08). Brakowało progów
ostrzejszych niż 5 m, percentyli i wyniku osobno dla momentów strzałów.

**Nowe (ticket 17, commit 1065370, [17/results.md](17/results.md)).** Status: opisowe, bez doboru i bez testów; zamrożone
predykcje fazy 1, GSR valid gry 2+3 jako główny zbiór, cały valid i GSR test (D4) tylko opisowo.

Zgodność referencji (rzut ramki prawdy piłki naszą homografią vs `bbox_pitch`, klatka po klatce, bez odejmowania od
błędów predykcji), gry 2+3, wśród klatek z homografią:

| punkt rzutowania | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m | p95 m |
|---|---|---|---|---|---|---|---|
| środek ramki | 25,1 [19,9; 31,7] | 53,4 [47,6; 59,9] | 80,8 | 93,6 [90,2; 96,2] | 0,94 | 3,31 | 6,84 |
| dół ramki | 35,7 [29,8; 42,5] | 61,5 [56,2; 67,0] | 80,5 | 93,2 | 0,71 | 3,44 | 7,14 |

Test: mediana 0,86 m (środek) i 0,61 m (dół). Dla piłki w powietrzu referencja ma p50 2,20 m, więc `bbox_pitch` nie jest
tam rzutem dołu ramki. Około 6% klatek z homografią leży dalej niż 5 m od referencji.

Predykcje, gry 2+3 (39 klipów, 4 250 klatek, 347 bez homografii); odsetki wśród wszystkich klatek, percentyle wśród
klatek z pozycją:

| wiersz | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m | p95 m | bez pozycji |
|---|---|---|---|---|---|---|---|---|
| top-1 dd2c | 12,1 | 27,0 [22,7; 31,9] | 43,6 | 51,6 [44,5; 59,3] | 1,19 | 15,22 | 23,30 | 36,7 |
| kandydat fazy 1 | 16,4 | 37,3 [32,5; 42,4] | 58,7 | 71,2 [64,6; 77,5] | 1,13 [0,99; 1,28] | 9,03 [4,84; 12,03] | 14,87 | 17,0 |
| 13, pojedyncza klatka | 15,3 | 33,4 | 52,7 | 62,4 [56,7; 68,3] | 1,07 | 7,44 | 15,06 | 29,0 |
| 13, śledzony | 16,5 | 37,9 | 59,9 | 72,7 [66,9; 78,2] | 1,14 | 9,07 | 15,46 | 15,0 |

Na GSR test (drugi odczyt, opisowo) kandydat fazy 1 ma ≤ 1 m 44,5% [39,4; 49,5], ≤ 5 m 77,1% [71,5; 81,7], p50 0,94 m,
p90 4,56 m, 15,1% bez pozycji (zgodnie z REPORT.md: 77,1% wobec 69,0% dla top-1).

Rozbicia kandydata fazy 1 (gry 2+3, wartości punktowe):

| cecha | poziom | ≤ 5 m | > 5 m wśród z pozycją | uwaga |
|---|---|---|---|---|
| status | zmierzona | 87,7% | 9,8% | p50 1,07 m |
| status | interpolowana | 70,8% | 25,7% | p50 1,53 m; 25,8% (valid) i 26,8% (test) tak samo |
| w powietrzu (proxy) | w powietrzu | 72,7% | 20,1% | ≤ 1 m tylko 14,9%, p50 2,72 m |
| w powietrzu (proxy) | ziemia | 71,5% | 13,6% | p50 1,01 m |
| wielkość | < 8 px | 35,3% | 36,4% | 44,5% bez pozycji |
| pokrycie przez ludzi | ≥ 0,5 | 63,6% | 14,7% | głównie odbiera pozycję (25,3% bez pozycji wobec 12,9%) |
| strefa | pole karne / reszta | 71,6% / 71,1% | 15,6% / 13,7% | na teście 92,1% / 73,0% |
| moment | strzał ±1 s (42 klatki) | 92,9% | 7,1% | ≤ 1 m 47,6% |

Wiersz „moment strzału” na 172 meczach (D4, opisowo, 2 721 strzałów bez karnych ze 135 meczów sparowanych ze
StatsBomb, StatsBomb tylko jako cel i druga referencja z własnym szumem): piłka t0 (top-1 dd2c): ≤ 1 m 16,0%, ≤ 5 m 49,8%
[47,6; 52,1], p50 1,78 m [1,69; 1,91], bez pozycji 37,2% [34,5; 39,7]. Miejsce `feet_t0`: ≤ 1 m 45,0%, ≤ 5 m 84,8%,
p50 1,16 m [1,05; 1,25], p90 6,79 m. Piłka P1 (po filtrze 45 px) wyszła praktycznie bez zmiany (≤ 5 m 49,1%, miejsce
strzału zmienia się w 0,3% strzałów), S2 wzięła pozycję w 82,7% strzałów (bez pozycji 17,3%), ale z gorszym p50
1,99 m (20: [20/results.md](20/results.md)).

**Ograniczenia.** Zgodność referencji GSR jest rzędu 1 m, więc progi 0,5 i 1 m nie mierzą dokładności fizycznej.
Piłka w powietrzu nie ma prawdy 3D, wyniki idą po proxy. GSR nie ma atrybutu zasłonięcia, więc „zasłonięte” to
proxy (pokrycie przez ludzi). GSR nie podaje meczu, więc tylko akcja środkowa klipu jest oznaczona (stąd 42 klatki
strzałów na grach 2+3, 66 na całym valid, 70 na teście). Drugi stopień 13 jest liczony tylko na GSR valid.

## Punkt 5: interpolacja i wiarygodność

**Było wcześniej.** `max_gap` 2 s wybrany w siatce strojenia 04; interpolacja w obrazie, korytarz, granice restartów
(12: nic nie wygrało; granice restartów zamieniają pokrycie na precyzję: interpolowane 70,8% / 24,4% → 75,4% / 20,2%);
flaga `interpolated` w pakiecie (REPORT.md). W 17 widać ten sam obraz co wyżej: interpolowane pozycje to główne
źródło błędów > 5 m.

**Nowe (ticket 18, commit c8a5247, [18/results.md](18/results.md)).** Status: rozwojowe (gry 2+3, ocena naprzemienna
gra 2 → 3 i 3 → 2, próg wybierany poza grą oceny).

`max_gap` bez odrzucania (gry 2+3):

| `max_gap` | pokrycie | > 5 m wśród przekazanych | > 5 m na wszystkie klatki | dobrze umieszczone | strata vs faza 1 |
|---|---|---|---|---|---|
| 0,25 s | 67,0 | 9,5 | 6,4 | 60,6 | 10,7 pp |
| 0,5 s | 72,1 | 11,0 | 8,0 | 64,2 | 7,1 pp |
| 1 s | 78,7 | 13,3 | 10,5 | 68,3 | 3,0 pp |
| 2 s (faza 1) | 83,0 | 14,2 | 11,8 | 71,2 | 0 |

Krótsza luka usuwa złe pozycje, ale dobre odpada szybciej: przejście z 2 s na 0,5 s usuwa 3,8 pp złych i 7,1 pp
dobrze umieszczonych (na wszystkie klatki). Gry różnią się mocno: w grze 2 interpolacja prawie nic nie zmienia
(dobrze umieszczone 77,3 → 79,4% przy 0,25 → 2 s), w grze 3 daje 17 pp (47,7 → 65,0%).

Wynik niepewności (regresja logistyczna na cechach dostępnych przy inferencji: status, długość luki, czas do pomiaru,
prędkość implikowana, pewność, ruch kamery, wielkość) vs linia bazowa „sama długość luki”, tory 2 s, ocena poza grą,
odrzucone 10% pozycji: > 5 m wśród przekazanych model 10,6% [6,0; 16,1], długość luki 12,1% [7,6; 17,7], różnica
**−1,5 pp [−2,2; −0,9]**. Krzywa modelu leży pod krzywą samego `max_gap` w całym zakresie (przy pokryciu 72,1%: 9,4%
wobec 11,0%). Kalibracja między grami jest słaba powyżej P 0,6 (wartości P ≥ 0,7 odpowiadają ok. 43–56% złych pozycji).
Przy stracie ≤ 2 pp mieści się tylko odrzucenie ok. 5% pozycji (14,2% → 12,3% > 5 m wśród przekazanych, −13% względnie).

Bramka D2 (poza grą, łącznie 2+3): pokrycie 70,4% [63,5; 77,3], > 5 m wśród przekazanych 9,3% [5,2; 14,2], dobrze
umieszczone 63,9% [56,7; 71,1], różnica **−7,3 pp [−10,1; −5,0]**, odrzucone złe **47,9% [35,9; 59,0]**. Warunek (a)
spełniony, (b) i (c) niespełnione, więc **bramka niespełniona i punktu pracy nie ma**. Gra 2 z regułą z gry 3 spełnia
wszystko (strata 0,8 pp), gra 3 z regułą z gry 2 nie (reguła wybiera `max_gap` 1 s, strata 12,4 pp). Zgodnie z D1 gra 4
nie została odczytana (`chosen.json` nie powstał).

Kontrole: `max_gap` 2 s bez odrzucania daje tory identyczne z `ball/09/track_valid`, bez filtra z `ball/04/track`;
odczyt odtwarza 17 (71,2% / 14,2%).

**Ograniczenia.** Tylko dwie gry do uczenia i oceny (współczynniki modelu trzeba uczyć na większej liczbie gier).
Model ruchu ze zmianą kierunku wypadł z zakresu po radzie. Piłka w powietrzu tylko jako proxy, bez osobnego modelu.
Gra 4 nieczytana (brak punktu pracy), gra 5 nieczytana.

## Punkt 6: lokalizacja piłki a xG

**Było wcześniej.** Stopy strzelca vs rzut piłki: błąd miejsca strzału 2,62 → 1,16 m wobec StatsBomb (172, opisowo);
dev 328 meczów: stos ALL vs R_stos −0,0056 → −0,0116 (raport xG efd7908, 99d48d2). Drugi stopień 13 zmienia
miejsce strzału w 16% strzałów, wpływ na xG nieznany (16). Piłka śledzona nie była w xG testowana.

**Nowe (tickety 19 i 20, commity 392ee00, f2d362b, 94281bd; [19/results.md](19/results.md), [20/results.md](20/results.md)).**
Status: rozwojowe (OOF na 7 062 strzałach z 328 meczów rozwojowych, procedura C6 zamrożona, każde ramię trenowane w
foldach, bootstrap par po meczach 2 000 / seed 0); wiersze na 172 meczach i wariant dwustronny opisowe.

Główny odczyt (ramię stosu ALL, ΔLL wobec F = `feet_t0`; plus oznacza gorzej niż F), Holm wspólny dla C, S2, T:

| ramię | LL | AUC | ΔLL vs F [95%] | p poprawy | p TOST ±0,002 | werdykt D3 |
|---|---|---|---|---|---|---|
| F (`feet_t0`) | 0,3314 | 0,760 | – | | | |
| R (piłka bazowa, odniesienie) | 0,3375 | 0,747 | +0,0061 [+0,0029; +0,0090] | | | bez werdyktu |
| C (F + 11 cech relacji piłka–stopy) | 0,3330 | 0,756 | +0,0016 [+0,0001; +0,0029] | 0,98 | 0,25 | nierozstrzygnięte |
| S2 (strzelec z piłki drugiego stopnia 13) | 0,3324 | 0,762 | +0,0010 [−0,0014; +0,0035] | 0,77 | 0,23 | nierozstrzygnięte |
| T (piłka śledzona przyczynowo na klipie, 2 s do t0) | 0,3314 | 0,761 | −0,0000 [−0,0026; +0,0025] | 0,49 | 0,062 | nierozstrzygnięte |

Wnioski:
- Żadne ramię piłki nie wnosi wartości: poprawy nie ma przy żadnym p_T, równoważności w ±0,002 też nie (T przekracza
  granicę o 0,0002). T jest najbliżej równoważności; C jest raczej gorsze od F.
- R − F = +0,0061 [+0,0029; +0,0090] to ten sam efekt co w efd7908 (−0,0056 → −0,0116), tylko liczony parami. Zysk
  przychodzi więc z przejścia na stopy, nie z lepszej piłki.
- Rozbicia po odległości, proxy wysokości piłki (głowy) i źródle piłki nie dają warstwy z przedziałem w całości poniżej
  zera dla żadnego ramienia (opisowo). Na 328 meczach nie ma etykiety części ciała, więc „głowy” to proxy `bf_height`.
- Tracking ma znaczenie dla miejsca strzału tam, gdzie F nie miało piłki: T daje piłkę t0 w 692 strzałach bez piłki
  F i zmienia miejsce strzału w 12,0% strzałów (> 5 m w odległości: 3,9%), a log-loss się nie zmienia.
- Wariant dwustronny ±2 s (offline, opisowo, poza rodziną, korzysta z klatek po strzale): −0,0010 [−0,0040; +0,0019].
  Nawet piłka zrekonstruowana z klatek po strzale nie daje wykrywalnej poprawy.
- Wiersze na 172 meczach (tylko 135 sparowanych ze StatsBomb, opisowo, bez testu parami): miejsce strzału przy piłce S2
  ma medianę błędu 1,09 m [1,02; 1,20] wobec 1,16 m przy F, ≤ 2 m 69,0% wobec 66,9%. Przedziały mocno się nakładają.

Kontrole (19 i 20): zamrożony E3 odtwarza `results/xgs_stack/real/summary.json`, R odtwarza `xgs_stack_dd2c`
(−0,0056), F odtwarza `xgs_stack_feet_t0_flags` (−0,0116), prawdopodobieństwa F w 20 bit w bit takie jak w 19.
Wykonalność T: 98,4% klipów ma 25 fps, 115 klipów (1,6%, głównie Serie A 2016/17) ma nieregularny fps i zostaje z F;
cięcia montażowe w [−2 s; 0] ma 7,2% strzałów, więc tor zaczyna się od ostatniego cięcia.

**Ograniczenia.** F było dobierane na tych samych 328 meczach (efd7908), więc porównanie jest przechylone na korzyść F;
mimo to ramiona piłki nie wychodzą poza ±0,0035. 172 mecze są zużyte, odczyt na nich jest tylko opisowy, a pozycje
StatsBomb tylko celem ewaluacji. Etykiety treningowe xG bez zmian (czas strzału i gol). Wewnętrzne CV wybiera dla C i T
silniejszą karę niż dla F, więc część różnic to szum procedury. Odpowiedź dotyczy miejsca strzału i log-loss, nie
innych zastosowań piłki w xG (np. przebieg akcji).

## Punkt 7: uogólnienie i filtr 45 px

**Było wcześniej.** Filtr 45 px zdefiniowany w pikselach 1080p i przeliczany z rozdzielczością (16: 30 px przy 720p);
przy inferencji 1600 działa tak samo (15); dowód na filtr głównie z gry 3 (REPORT.md); GSR test zużyty na wybór dd2c;
gry 4 i 5 nieprzeczytane w fazie 2 (PHASE2.md).

**Nowe.**

*Odporność filtra (ticket 23, commit 0602d6a, [23/results.md](23/results.md)).* Status: rozwojowe na grach 2+3 (k wariantu
względnego dobrane w próbie, więc tego wariantu nie traktować jako potwierdzenia). Top-1 dd2c bez śledzenia, ≤ 5 m:

| skala | bez filtra | filtr 45 px | abs45 − none, pp | > 5 m wśród wszystkich, pp |
|---|---|---|---|---|
| 1080p | 51,6 [44,5; 59,3] | 52,2 [45,2; 59,8] | +0,6 [+0,1; +1,2] | −6,1 [−9,1; −3,4] |
| 720p | 51,7 [44,6; 59,4] | 52,4 [45,4; 59,9] | +0,7 [+0,1; +1,2] | −6,1 [−9,1; −3,4] |
| 540p | 52,1 [45,0; 59,7] | 52,7 [45,8; 60,1] | +0,6 [0,0; +1,1] | −6,3 [−9,4; −3,5] |

Prawdziwe piłki usunięte przez filtr: 0,0% (górna granica 0,1%) w każdej skali; duże fałszywe usunięte 97,9% / 98,1% /
97,0%. Przy 540p dolna granica zysku ≤ 5 m wynosi formalnie 0,0, czyli wg zarejestrowanej reguły „nie działa”, choć efekt
i spadek odsetka > 5 m są te same. Filtr względny (próg = k × mediana wysokości ludzi, k = 0,474 to dokładnie
odpowiednik 45 px) nie jest lepszy: rel − abs45 ≤ +0,02 pp w każdej skali. Realna różnorodność skali w grach 2+3
(tercyle wg wysokości ludzi): w szerokich ujęciach filtr pomaga (> 5 m −12,9 pp [−18,6; −7,5]), w zbliżeniach nic nie
zmienia (usuwa 0,1% prawdziwych piłek, wariant względny 0,0%). Na klatkach strzałów xG (bez prawdy, mediana wysokości
ludzi 115 px wobec 95 px w GSR) abs45 usuwa 2,2% kandydatów ≥ 0,15, rel 1,1%, ale bez prawdy nie wiadomo, które
usunięcia były trafne. Ekstraklasa dev (opisowo, 56 obrazów): abs45 usuwa 0 z 53 kandydatów.
Wniosek: filtr 45 px zostaje, nie ma oznak spadku skuteczności przy niższej rozdzielczości.

*Inwentarz (ticket 21, commit 0098905, [21/results.md](21/results.md)).* Status: opisowy spis. Żadne źródło z etykietami
piłki nie jest nietknięte: GSR train, valid i test, SoccerNet-v3, v3D, TeamTrack, ISSIA, Soccana, Roboflow i
football_yolo są w treningu dd2c (wszystkie splity), w doborze albo w odczytach; Ekstraklasa dev (46 piłek) była
strażnikiem w fazach 1–2. SNMOT to te same klipy co GSR (bajt w bajt), nie drugie źródło. GSR challenge nie ma etykiet,
SoccerDB nie ma danych lokalnie. SoccerTrack v2 jest nietknięty, ale nie ma pól piłki w obrazie (tor piłki w `ball/`
to rekonstrukcja ze zdarzeń z interpolacją liniową i leży za bramką HF, konto nie ma dostępu).
Wniosek orkiestratora: D5 niespełnione, ticket 22 nie został uruchomiony.

**Ograniczenia.** Wyników poza domeną nie ma. Filtr mierzono na top-1 bez śledzenia (w pipeline działa przed
`track_ball`), jedną metodą przeskalowania (INTER_AREA) i bez symulowanego zoomu (usunięty po radzie). Zbliżenia (ludzie
większi niż w GSR) pozostają bez prawdy piłki. Gry 4 i 5 nie są nietknięte (niżej).

## Uwagi uczciwości

- **Zgodność referencji.** Mediana różnicy między naszą homografią a `bbox_pitch` wynosi 0,94 m (środek ramki; 0,71 m
  dół), ≤ 1 m tylko 53–62% klatek. Progi 0,5 i 1 m opisują zgodność z referencją GSR, nie dokładność w metrach.
  Wynik nie jest odejmowany od błędów predykcji. Dla strzałów drugą referencją jest StatsBomb z własnym szumem
  kodowania i czasu.
- **Brak nietkniętych meczów transmisyjnych z etykietami piłki** (ticket 21). Uogólnienie na inne rozgrywki, kamery i
  oświetlenie nie zostało sprawdzone i punkt 7 recenzenta w tej części pozostaje otwarty.
- **Gra 4 nieczytana w fazie 3.** Brak punktu pracy w 18 (D1), więc nie było odczytu ograniczonej walidacji. Gra 4 nie
  jest przy tym nietknięta: widział ją dd2c (run 2c) i model punktów boiska (ticket 21); odczytana była tylko w 10
  (jakość homografii). **Gra 5 nieczytana**; była czytana w fazie 1 (04, 08) i wchodzi do całego GSR valid.
- **Zbiory zużyte.** GSR test (wybór dd2c) i 172 mecze xG tylko opisowo, na zamrożonych predykcjach, bez doboru (D4).
  17 liczy tu wyłącznie opis.
- **F dobierane na tych samych 328 meczach** co ramiona piłki w 19 i 20, więc porównanie sprzyja F.
- **StatsBomb tylko jako cel ewaluacji**, nigdy wejście modelu ani etykieta treningowa dla pozycji.
- **Dobór w próbie.** Dobór k filtra względnego (23) i wybór gier w 18 (tylko 2 i 3) są na tych samych danych, na
  których mierzono. Tylko 18 dopuszcza ocenę naprzemienną.
- **Wyniki negatywne i niewykonalne.** Brak punktu pracy w 18; wszystkie ramiona piłki w 19/20 nierozstrzygnięte;
  filtr względny bez przewagi; ticket 22 niewykonany; P1/S2 na 172 wykonane dopiero w 20 (GPU).

## Rekomendacje

**Pakiet (`ball-tracking/02`, bez zmian w `src/` w tej fazie).**
- `track_ball`: ustawienia bez zmian, `max_gap` 2 s, bez odrzucania (18).
- Wynik niepewności P(> 5 m) udostępnić jako pole pozycji, nie jako domyślny filtr. Porządkuje pozycje lepiej niż
  długość luki, ale kalibracja między grami jest słaba, a bramka D2 nie przeszła. Użytkownik, który woli mniej złych
  pozycji kosztem pokrycia, może progować to pole sam. Współczynniki trzeba uczyć na więcej niż dwóch grach; potrzebne
  cechy: długość luki, czas do pomiaru, prędkość implikowana z końców luki, pewność końców, ruch kamery, średnica.
- Filtr 45 px zostaje (23). Bez filtra względnego.
- Przy opisie jakości w pakiecie podawać progi 0,5–1 m wyłącznie razem ze zgodnością referencji, a pozycje
  interpolowane oznaczać (flaga `interpolated` już jest).

**xG.**
- Stopy `feet_t0` zostają jako miejsce strzału (19, 20).
- Piłka bazowa (R), z drugiego stopnia (S2) i śledzona (T) nie wnoszą wykrywalnej wartości do log-loss wobec F; cechy
  relacji piłka–stopy (C) nie pomagają. Nie ma uzasadnienia dla dalszych kosztów GPU na piłkę w xG z tego powodu.
- Piłka może mieć znaczenie dla zastosowań innych niż miejsce strzału (np. przebieg akcji), ale to nie było testowane.

## Zużycie GPU fazy 3

Razem ok. 3,6 GPU-h z 8 GPU-h limitu dla ramienia T:
- ticket 20: ok. 3,4 GPU-h (zrzut T 1,56 h, detekcja 172 0,04 h, przerwany start, test szybkości i profil 0,16 h, razem
  ok. 1,8 h do odczytu T; wariant dwustronny 1,61 h);
- ticket 23: ok. 0,2 GPU-h (ok. 12 min);
- tickety 17, 18, 19, 21 i 24: CPU (19 użył CPU: odczyt xG ok. 4,5 min, klipy ok. 6 min); ticket 22 nie uruchomiony.
