# ball-detection 08: kafle 2×2 + filtr 45 px, odczyt potwierdzający na grze 5

2026-10-25. Wagi dd2c bez zmian (`models/detector_yolo11l_1280_dd2c.pt`, sha256 05d706cf…). Rejestracja w treści
ticketu 08, potwierdzona komentarzem 17:51 przed zrzutem i odczytem gry 5. Wariant to dokładnie ten z ticketu 03
(`scripts/ball_inference.py`): kafle 1024×608 przy imgsz 1024, zakładka 128 × 136 px, odrzut piłek ≤ 2 px od
wewnętrznej krawędzi kafla, NMS po IoS ≥ 0,5 z piłkami pełnej klatki 1280, ludzie z pełnej klatki, a potem
post-filtr: odrzut kandydatów piłki dłuższych niż 45 px przy 1080p. Przedziały 95%: bootstrap po klipach,
2 000 losowań, seed 0. GPU: RTX 4070 Ti SUPER, blokada `agent-state gpu` na czas obu zrzutów.

Kontrola zgodności: na grach 2+3 nowy zrzut (cały GSR valid) jest identyczny ze zrzutem `ball/03/tiles2x2`
(258 653 ramki, różnica ramek i pewności 0,0). Na klatkach co 5. zrzut 25 fps jest identyczny ze zrzutem co 5.
przy conf ≥ 0,01 (227 036 ramek, różnica 0,0).

## Porównanie główne (zarejestrowane)

**Gra 5 (19 klipów, 2 154 klatki): kafle 2×2 + filtr wobec dd2c 1280 bez filtra: +5,29 pp [+3,15; +8,02].
Dolna granica > 0, więc wynik jest potwierdzony.**

| split | dd2c 1280 | kafle 2×2 + filtr | zysk parami, pp [95%] | > 5 m wśród umieszczonych (Δ pp) | AP50 piłki (Δ) |
|---|---|---|---|---|---|
| gra 5 (potwierdzenie) | 77,11% | 82,40% [78,6; 86,4] | **+5,29 [+3,15; +8,02]** | 8,69% (+1,43 [+0,48; +2,58]) | 0,503 (+0,013 [−0,006; +0,033]) |
| gry 2+3 (już znane z 03, nie dowód) | 51,60% | 58,00% | +6,40 [+4,91; +8,06] | 12,68% (−5,76) | 0,346 (+0,050) |
| cały GSR valid (6 404 klatki) | 60,18% | 66,21% [60,2; 71,6] | +6,03 [+4,71; +7,39] | 11,06% (−2,92) | 0,397 (+0,041 [+0,017; +0,067]) |

Na grze 5 kafle umieszczają piłkę częściej (umieszczona +7,1 pp), ale trochę częściej się mylą: > 5 m rośnie o
1,4 pp, a fałszywych piłek ≥ 0,15 jest 0,22 na klatkę (dd2c 0,14). Blisko bramki zysk wynosi +8,0 pp [+4,6; +12,6].

**Strażnicy i bramki:**
- mAP50-95 ludzi, cały GSR valid: −0,000 (ludzie są z tej samej pełnej klatki). Spełniony (≥ −0,01).
- Ekstraklasa dev mAP50-95: −0,009 [−0,015; −0,002]. Spełniony (≥ −0,02). AP50 piłki Ekstraklasy 0,242 (dd2c
  0,279), poniżej kierunku 0,40. Przy 3 klipach przedział [0,12; 0,50] obejmuje wszystko.
- **Bramka 1280 na całym GSR valid: 66,21%, poniżej 68,18% (i poniżej ambitnych 73,18%). Nie spełniona.**
- **AP50 piłki na całym GSR valid: 0,397, poniżej 0,45. Nie spełniony.**

## Wiersze opisowe

**(a) Ile dają same kafle (oba ramiona z filtrem 45 px), wobec dd2c 1280 + filtr:**

| split | dd2c 1280 + filtr | kafle 2×2 + filtr | zysk parami, pp [95%] | > 5 m (Δ pp) |
|---|---|---|---|---|
| gra 5 | 77,11% | 82,40% | +5,29 [+3,15; +8,02] | +1,43 [+0,48; +2,58] |
| gry 2+3 | 52,24% | 58,00% | +5,76 [+4,38; +7,34] | +3,07 [+0,97; +5,91] |
| cały GSR valid | 60,60% | 66,21% | +5,61 [+4,34; +6,98] | +2,44 [+1,11; +4,15] |

Na grze 5 filtr niczego w dd2c nie zmienia. dd2c nie ma tam ani jednego kandydata piłki ≥ 0,15 dłuższego niż
45 px (na grze 3 ma ich 687, na grze 2 trzy), więc wiersz (a) na grze 5 jest równy porównaniu głównemu. Z +6,03 pp
na całości 0,4 pp daje filtr, a 5,6 pp kafle. Filtr jest potrzebny po to, żeby kafle w ogóle działały: bez niego
kafle na grach 2+3 dają −0,73 pp (ticket 03).

**(b) Kafle + filtr + śledzenie** wobec śledzenia na dd2c 1280 bez filtra (ticket 04). Ustawienie z komentarza
21:50 w 04 bez zmian (`ball/04/tune/chosen.json`), ruch kamery z `ball/04/motion`, ludzie z tej samej pełnej
klatki 1280, a zrzut kafli 25 fps (43 500 klatek, conf ≥ 0,01, filtr 45 px):

| split | śledzenie na dd2c | śledzenie na kaflach + filtr | zysk parami, pp [95%] | > 5 m (Δ pp) |
|---|---|---|---|---|
| gra 5 | 86,26% | 85,00% [80,7; 89,3] | **−1,25 [−2,79; +0,24]** | 10,73% (+2,48 [+1,15; +4,08]) |
| gry 2+3 | 68,92% | 73,69% [67,3; 79,5] | +4,78 [+1,55; +8,39] | 17,16% (−1,47 [−4,95; +1,70]) |
| cały GSR valid | 74,75% | 77,50% [72,6; 81,7] | +2,75 [+0,42; +5,51] | 14,90% (0,00) |

Ze śledzeniem kafle pomagają na trudnych grach 2+3, a na grze 5 nie (punkt ujemny, więcej pozycji > 5 m). Tracker
sam odzyskuje część piłek, które kafle dodają (słabe kandydaty, luki), za to idzie za dodatkowymi fałszywymi
piłkami z kafli. Cały GSR valid ze śledzeniem na kaflach: 77,50%, czyli powyżej bramki 68,18% i ambitnej 73,18%.
To jednak wynik trackera na wierszu opisowym, a nie odczyt zarejestrowany. AP50 piłki 0,389.

**(c) Koszt (RTX 4070 Ti SUPER, batch 1, kolejno):**

| pipeline piłki | ms/klatkę | względem dd2c 1280 |
|---|---|---|
| dd2c 1280 (+ filtr: 0 ms) | 19,6 | 1× |
| kafle 2×2 + pełna klatka 1280 + scalanie (+ filtr) | 75,0 (zrzut co 5.); 87,2 przy 25 fps, z równoległą kartą na CPU | 3,8× |
| śledzenie (`track_ball`) | + 0,2 ms CPU | |
| ruch kamery dla bramki w metrach (ticket 04) | + ok. 10 ms CPU, jeden rdzeń, połowa rozdzielczości | |

Zrzut 25 fps zajął 60 min GPU (43 500 klatek).

## Rekomendacja dla pipeline'u

1. **Filtr 45 px zawsze.** Nic nie kosztuje, nie szkodzi ludziom ani Ekstraklasie, a bez niego kafle są
   bezużyteczne. Opcja detektora `ball_max_side_px=45` (przy 1080p, skalowana wysokością klatki), stosowana przed
   `track_ball`.
2. **Śledzenie (ustawienie z 04) to domyślna droga dla wideo.** Daje najwięcej (+14,6 pp na całości, potwierdzone na
   grze 5 w 04) za ok. 10 ms CPU na klatkę. Kafle dokładane do śledzenia nie są potwierdzone na grze 5 (−1,25 pp
   [−2,79; +0,24]) i kosztują 3,8× czasu detekcji (ok. 55 ms więcej na klatkę), więc w pipeline wideo **bez kafli**.
3. **Kafle 2×2 + filtr jako tryb dla pojedynczych klatek**, czyli tam, gdzie nie ma sąsiednich klatek dla trackera
   (np. klatka t0 strzału w xG, wybór strzelca). Tam wynik zarejestrowany jest potwierdzony: +5,3 pp na grze 5,
   +6,0 pp na całości. Koszt na kilka klatek strzału jest pomijalny. Opcja `ball_tiles=(2, 2)` domyślnie wyłączona.
4. Żaden wariant bez śledzenia nie spełnia bramki planu (68,18%) ani AP50 piłki ≥ 0,45. Cel planu osiąga dopiero
   tracker. Detektorowi dalej brakuje piłek < 8 px (recall IoU 0,06 na całości, także z kaflami).

Uwaga: 1600 + TTA + filtr z ticketu 03 (+6,45 pp na grach 2+3, 55,5 ms) jest tańszym odpowiednikiem kafli, ale
nie był rejestrowany ani czytany na grze 5. Gra 5 była już czytana dla 04 i 08, więc kolejny odczyt potwierdzający
na niej słabiej chroni przed doborem po danych.

## Pliki

Kod (bez zmian w `src/`): `scripts/ball_inference.py` (opcja `--ticket` rekordu zrzutu), `scripts/ball_temporal.py`
(`dump --variant tiles2x2 --post-filter`, `track --dump`, kodowanie z jedynego wiersza zrzutu odniesienia, nazwa
gry czytana raz na klip), `scripts/ball_scorecard.py` (wiersz bez odpowiednika w odniesieniu łączy się w parę z
jedynym wierszem jednowierszowego odniesienia). Testy: `tests/test_ball_temporal.py`, `tests/test_ball_scorecard.py`.

Poza gitem, `/mnt/storage/tactifoot_research/detector_data/ball/08/`: `tiles2x2/dump`, `tiles2x2+pf45/dump`
(cały GSR valid + Ekstraklasa), `card_primary/` (porównanie główne, `--confirm`), `card_vs_pf45/` (wiersz a),
`dump25_tiles2x2+pf45/` (25 fps), `track/` (tory i zrzuty kart), `card_track/` (wiersz b), `table.md`, `logs/`,
skrypty `run_*.sh`, `after_dump25.sh`.
