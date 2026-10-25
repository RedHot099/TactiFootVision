# ball-detection 03: piłka bez treningu (rozdzielczość, kafle, TTA, post-filtr)

2026-10-24/25. Wagi dd2c bez zmian (`models/detector_yolo11l_1280_dd2c.pt`, sha256 05d706cf…). Rejestracja przed
odczytem: komentarz ticketu 03 z 2026-10-24 21:27 (reguła z ticketu 02, komentarz 21:02). Wszystkie liczby na splicie
wyboru (GSR valid, gry 2+3: 39 klipów, 4 250 klatek mianownika, 5 501 piłek), parami z dd2c 1280 (zrzut
`ball/02/dd2c/dump`), bootstrap po klipach 2 000 / seed 0. Gry 5 nikt nie czytał (brak zwycięzcy). GPU: RTX 4070 Ti
SUPER, blokada `agent-state gpu` na czas zrzutów.

Kod: `scripts/ball_inference.py` (`dump`, `postfilter`, `table`), test `tests/test_ball_inference.py`; karta
`scripts/ball_scorecard.py card --dump`.

## Wynik porównania głównego

**1920 natywnie vs dd2c 1280: −0,21 pp [−2,87; +2,10]. Reguła nie spełniona (dolna granica ≤ 0), brak zwycięzcy
ticketu, brak odczytu potwierdzającego.** Strażnicy względni spełnieni (ludzie mAP50-95 +0,033 [+0,016; +0,049],
Ekstraklasa mAP50-95 +0,017), ale bez sukcesu głównego nie mają znaczenia. 1920 znajduje więcej piłek (recall środka
0,532 vs 0,516), ale też dużo więcej fałszywych (0,36 vs 0,21 na klatkę). Umieszcza piłkę na 72% klatek zamiast 58%,
tylko że 28,8% umieszczonych leży dalej niż 5 m (dd2c 18,5%). AP50 piłki spada o 0,028.

## Tabela wariantów (gry 2+3; Δ = wariant − dd2c 1280)

Wszystkie wiersze poza 1920 są opisowe. Wariant opisowy nie wygrywa ticketu (reguła 02, pkt 2).

| wariant | AP50 piłki (Δ) | dobrze umieszczona | Δ pp [95%] | > 5 m wśród umieszczonych (Δ pp) | recall IoU / środek < 8 px | recall IoU / środek 8–12 px | fałszywe / klatkę | Ekstraklasa AP50 piłki [95%] | Ekstraklasa mAP50-95 (Δ) | ludzie mAP50-95 (Δ) | ms/klatkę |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dd2c 1280 (odniesienie) | 0,296 | 51,60% | 0 | 18,45% | 0,003 / 0,071 | 0,266 / 0,385 | 0,208 | 0,279 [0,25; 0,38] | 0,424 | 0,514 | 19,6 |
| **1920 (główne)** | 0,268 (−0,028) | 51,39% | **−0,21 [−2,87; +2,10]** | 28,77% (+10,32) | 0,016 / 0,049 | 0,274 / 0,398 | 0,364 | 0,318 [0,22; 0,48] | 0,441 (+0,017) | 0,547 (+0,033) | 34,8 |
| 1600 | 0,313 (+0,017) | 52,56% | +0,96 [−0,49; +2,67] | 20,86% (+2,42) | 0,016 / 0,063 | 0,281 / 0,397 | 0,244 | 0,348 [0,24; 0,46] | 0,437 (+0,013) | 0,547 (+0,033) | 28,5 |
| kafle 2×2 + 1280 | 0,214 (−0,082) | 50,87% | −0,73 [−2,89; +1,39] | 37,39% (+18,94) | 0,014 / 0,123 | 0,315 / 0,476 | 0,596 | 0,242 [0,12; 0,50] | 0,415 (−0,009) | 0,514 (0) | 74,8 |
| kafle 3×2 + 1280 | 0,199 (−0,097) | 49,81% | −1,79 [−4,55; +1,03] | 39,06% (+20,62) | 0,014 / 0,115 | 0,309 / 0,466 | 0,605 | 0,248 [0,12; 0,50] | 0,416 (−0,008) | 0,514 (0) | 80,8 |
| 1600 + TTA | 0,302 (+0,006) | 56,49% | +4,89 [+3,23; +6,76] | 22,82% (+4,38) | 0,027 / 0,132 | 0,321 / 0,481 | 0,403 | 0,331 [0,27; 0,41] | 0,441 (+0,017) | 0,547 (+0,033) | 55,5 |
| dd2c 1280 + pf45 | 0,337 (+0,041) | 52,24% | +0,64 [+0,12; +1,18] | 9,61% (−8,84) | 0,003 / 0,071 | 0,266 / 0,385 | 0,090 | 0,279 [0,25; 0,38] | 0,424 (0) | 0,514 (0) | 19,6 |
| 1920 + pf45 | 0,339 (+0,044) | 52,40% | +0,80 [−1,70; +2,96] | 10,45% (−7,99) | 0,016 / 0,049 | 0,274 / 0,398 | 0,118 | 0,318 [0,22; 0,48] | 0,441 (+0,017) | 0,547 (+0,033) | 34,8 |
| 1600 + pf45 | 0,356 (+0,060) | 53,18% | +1,58 [+0,25; +3,05] | 9,53% (−8,92) | 0,016 / 0,063 | 0,281 / 0,397 | 0,092 | 0,349 [0,24; 0,46] | 0,437 (+0,013) | 0,547 (+0,033) | 28,5 |
| 1600 + TTA + pf45 | 0,361 (+0,065) | **58,05%** | **+6,45 [+4,46; +8,82]** | 10,42% (−8,02) | 0,027 / 0,132 | 0,321 / 0,481 | 0,198 | 0,332 [0,27; 0,41] | 0,441 (+0,017) | 0,547 (+0,033) | 55,5 |
| kafle 2×2 + pf45 (poza rejestracją) | 0,346 (+0,050) | **58,00%** | **+6,40 [+4,91; +8,06]** | 12,68% (−5,76) | 0,014 / 0,123 | 0,315 / 0,476 | 0,152 | 0,242 [0,12; 0,50] | 0,415 (−0,009) | 0,514 (0) | 74,8 |
| kafle 3×2 + pf45 (poza rejestracją) | 0,340 (+0,044) | 57,72% | +6,12 [+4,60; +7,87] | 13,66% (−4,79) | 0,014 / 0,115 | 0,309 / 0,466 | 0,158 | 0,249 [0,12; 0,50] | 0,416 (−0,008) | 0,514 (0) | 80,8 |

ms/klatkę to czas wywołań modelu (dla kafli suma kafli + pełnej klatki + scalania), batch 1, kolejno, bez odczytu
obrazu. Ekstraklasa dev ma klatki 720p, więc 1600 / 1920 to tam powiększenie, a nie natywna rozdzielczość. Na 3 klipach
i 46 piłkach przedział Ekstraklasy obejmuje wszystkie różnice. „pf45” = post-filtr: odrzucenie kandydatów piłki o
dłuższym boku > 45 px przy 1080p (próg z prawdy, ticket 02). Pełne karty: `ball/03/<wariant>/scorecard.md`.

Kafle: 1920×1080 → 2×2 kafle 1024×608 (zakładka 128 × 136 px), 3×2 kafle 704×608 (96 × 136 px), każdy przy imgsz
równym dłuższemu bokowi kafla, więc bez skalowania. Scalanie tylko piłek (odrzut piłek dotykających wewnętrznej
krawędzi kafla, potem zachłanny NMS po IoS ≥ 0,5 z pełną klatką 1280); ludzie z pełnej klatki, dlatego mAP ludzi bez
zmian.

## Co z tego wynika

1. **Sama wyższa rozdzielczość jednego wywołania nie pomaga.** 1600 i 1920 dają po kilkadziesiąt dodatkowych trafień
   (głównie piłki 12–16 px), ale też więcej fałszywych piłek z dużą pewnością. Top-1 częściej trafia na fałszywą piłkę,
   więc odsetek dobrze umieszczonych stoi w miejscu. Zysk na ludziach (+0,033) to efekt uboczny, nie cel tego ticketu.
2. **Kafle znajdują najwięcej piłek** (recall środka 0,59 vs 0,52; < 8 px 0,12 vs 0,07; 8–12 px 0,48 vs 0,39), bo
   piłki kafli dochodzą do piłek pełnej klatki 1280. Bez filtra cały zysk zjadają fałszywe piłki (0,60 na klatkę,
   > 5 m 37–39%): na wycinkach model bierze ludzi za piłkę. W 2×2 jest 2 594 kandydatów piłki ≥ 0,15 dłuższych niż
   45 px (dd2c 1280: 690), z czego 2 160 (83%) leży na ramce człowieka z pełnej klatki (IoU ≥ 0,5).
3. **Post-filtr 45 px działa wszędzie i nic nie kosztuje.** Na samym dd2c 1280 daje +0,64 pp [+0,12; +1,18], > 5 m
   18,5% → 9,6% i AP50 piłki +0,041, a ludzie i Ekstraklasa się nie zmieniają. Z kaflami daje +6,4 pp [+4,9; +8,1]
   (2×2). To największy zysk w tej linii, ale jest opisowy: wariantu „kafle + pf45” nie było w rejestracji (dopisałem
   go po odczycie kafli i filtrów, jest więc wybrany po danych). Wg reguły 02 pkt 2 może być tylko porównaniem
   głównym późniejszego ticketu, zapisanym przed odczytem, z odczytem potwierdzającym na grze 5.
4. Koszt: kafle 2×2 to 74,8 ms/klatkę (3,8× dd2c 1280); 3×2 jest wolniejsze i nie lepsze.

5. **TTA (opisowo).** Najlepszy punktowo z wariantów 1920 / 1600 / kafle to 1600, więc TTA (Ultralytics `augment`:
   skale 1 / 0,83 / 0,67, odbicie przy 0,83) poszło na 1600. Wynik: +4,89 pp [+3,23; +6,76], 55,5 ms/klatkę (2,8×
   dd2c). Recall rośnie jak przy kaflach (środek 0,585; < 8 px 0,13; 8–12 px 0,48), fałszywych piłek jest 0,40 na
   klatkę, a > 5 m to 22,8%. Zgodnie z rejestracją post-filtr idzie na najlepszy z wariantów 1–3, czyli teraz na
   1600 + TTA: **+6,45 pp [+4,46; +8,82]**, > 5 m 10,4%, AP50 piłki +0,065, ludzie +0,033, Ekstraklasa mAP50-95
   +0,017. Tyle samo co kafle 2×2 + pf45 przy 74% ich czasu, a ludzie są lepsi. To wariant zarejestrowany, ale
   opisowy (wybrany jako najlepszy na tym samym splicie), więc też tylko kandydat na porównanie główne późniejszego
   ticketu. Zrzut TTA powstał 25.10 17:39–17:45 (blokada GPU wzięta i zwolniona), po końcu treningu 05.

## Decyzja 9: czy zysk skupia się w piłkach < 12 px

Według rejestracji liczy się wiersz główny: **1920 nie ma zysku** (netto −9 klatek), więc „skupiony” = nie. Z
porównania głównego ticketu 03 nie wynika rekomendacja treningu 1600 px. Klatki z piłką < 12 px to 40,0% mianownika,
a piłki < 12 px to 40,3% piłek.

Opisowo, dla kafli liczonych wobec dd2c 1280 + pf45 (ten sam filtr po obu stronach, więc różnica to tylko kafle):
2×2 + pf45 daje netto +245 klatek, z czego 51% w klatkach z piłką < 12 px (zysk na klatkę 0,074 vs 0,047 przy
≥ 12 px). Trafień środka jest +409, z tego 46% w < 12 px. Dla 3×2 jest to 42% i 42%. Zysk z natywnej rozdzielczości
jest więc w małych piłkach tylko trochę silniejszy niż proporcjonalnie, rozłożony na 8–16 px, a piłek < 8 px nadal
prawie nikt nie wykrywa (recall IoU 0,014). Tabela: `ball/03/table/size_gain_vs_dd2c_pf45.md`.

1600 + TTA wobec dd2c 1280: netto +208 klatek, z tego 49,5% w < 12 px (zysk na klatkę 0,061 vs 0,041); trafień
środka +380, z tego 52% w < 12 px. 1600 + TTA + pf45 wobec dd2c 1280 + pf45: +247 klatek, 53% w < 12 px. Obraz jest
ten sam co przy kaflach: zysk w małych piłkach jest nieco ponadproporcjonalny, ale nie skupiony.

## Propozycja dla pipeline'u (tylko propozycja, pakiet bez zmian)

- Opcja detektora `ball_max_side_px` (domyślnie 45 przy 1080p, skalowana wysokością klatki): odrzut kandydatów piłki
  dłuższych niż próg, zanim zobaczy je `track_ball`. Zero kosztu, zysk na samym dd2c 1280.
- TTA na piłkę (`ball_tta=True` przy imgsz 1600) albo kafle (`ball_tiles=(2, 2)`) dopiero po potwierdzeniu w
  osobnym tickecie (porównanie główne zapisane przed odczytem, odczyt gry 5). TTA kosztuje 2,8×, kafle 3,8× czasu
  detekcji; TTA jest tańsze i daje to samo.

## Ścieżki (poza gitem)

`/mnt/storage/tactifoot_research/detector_data/ball/03/`: `<wariant>/dump/` (zrzuty; 1920 obejmuje cały GSR valid,
reszta gry 2+3 i Ekstraklasę), `<wariant>/scorecard.{json,md}`, `<wariant>/placement_selection_*.parquet`,
`table/variants.{md,csv,json}`, `table/size_gain_vs_dd2c_pf45.md`, `logs/`, `run_dumps.sh`, `run_cards.sh`.
