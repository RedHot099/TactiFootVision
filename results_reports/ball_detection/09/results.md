# ball-detection 09: drugi odczyt GSR test kandydata wideo

2026-10-25. Rejestracja w treści ticketu 09, potwierdzona komentarzem 19:33 przed zrzutem i odczytem. Odczyt
GSR test o 20:15, jeden przebieg, nic nie dobierane na teście. **To „drugi odczyt GSR test”:** dd2c był już wybrany
na GSR test w detector-data, więc test nie jest świeży i wynik jest opisowy.

- **Kandydat:** dd2c 1280 (`models/detector_yolo11l_1280_dd2c.pt`, sha256 05d706cf…), kandydaci wszystkich klas
  ≥ 0,01 przy 25 fps, post-filtr 45 px (kandydaci piłki o dłuższym boku > 45 px przy 1080p odrzucani przed
  trackerem), pakietowy `track_ball` z `ball/04/tune/chosen.json` bez zmian, ludzie ≥ 0,3, homografie co klatkę z
  zamrożonych kotwic co 5. klatkę przeniesionych ruchem kamery (ustawienia ticketu 04).
- **Odniesienie:** top-1 dd2c 1280 ≥ 0,15 bez filtra, z tego samego zrzutu, na tych samych klatkach.
- **Miara:** odsetek dobrze umieszczonych (≤ 5 m) na mianowniku reguły 02 (klatki co 5. z jedną piłką na boisku,
  bez homografii = pudło), zysk parami, bootstrap po klipach 2 000 / seed 0.

## Klatki GSR test przy 25 fps

Lokalny GSR test ma tylko klatki co 5. Pełne klipy wziąłem z SoccerNet Tracking test
(`data/soccernet/tracking/extracted/test/SNMOT-NNN`, te same klipy co SNGS-NNN). Sprawdzone przed zrzutem i zapisane
w `dump.json`: 49 sekwencji po 750 klatek, wszystkie 7 350 lokalnych klatek GSR test są bajt w bajt identyczne z
klatkami SNMOT. Etykiety, gry (7, 8, 11) i homografie są z GSR test. Ruch kamery: 0 przerw na 36 701 parach klatek;
kotwica przeniesiona o 5 klatek różni się od następnej zamrożonej o 0,24 m (mediana klipów; bez kompensacji 0,66 m).

## Wiersz na GSR valid (przed testem, opisowo)

Zrzut 25 fps z ticketu 04 + filtr + śledzenie; gra 5 tylko w całym valid.

| split | top-1 dd2c | kandydat [95%] | zysk parami, pp [95%] | wśród z homografią | > 5 m wśród umieszczonych |
|---|---|---|---|---|---|
| gry 2+3 (4 250 klatek, 347 bez H) | 51,60% | 71,25% [64,6; 77,5] | **+19,65 [+13,81; +25,60]** | 56,19% → 77,58% | 18,45% → 14,17% (−4,27 [−9,20; +0,76]) |
| cały GSR valid (6 404, 426 bez H) | 60,18% | 76,30% [71,0; 80,9] | **+16,11 [+11,35; +20,52]** | 64,47% → 81,73% | 13,97% → 12,01% (−1,96 [−5,27; +1,31]) |

Ile daje sam filtr na śledzeniu (kandydat vs śledzenie z 04 bez filtra): gry 2+3 +2,33 pp [+0,72; +4,16], cały
valid +1,55 pp [+0,49; +2,81]; > 5 m −4,47 pp [−7,26; −2,05] / −2,89 pp [−4,75; −1,36]. Kontrola kodu: `read` na
torach z 04 odtwarza karty 04 co do cyfry (68,92% / 74,75%, +17,32 [+11,86; +22,82]), a ponowne śledzenie bez filtra
daje tory identyczne z 04 na 43 500 klatkach.

## Drugi odczyt GSR test (49 klipów, 5 277 klatek mianownika, 688 bez homografii, sufit 87,0%)

| | top-1 dd2c 1280 | kandydat | różnica parami, pp [95%] |
|---|---|---|---|
| **dobrze umieszczona (miara główna)** | 68,96% [63,7; 73,7] | **77,07%** [71,5; 81,7] | **+8,11 [+5,73; +10,63]** |
| dobrze umieszczona wśród klatek z homografią | 79,30% | 88,62% | +9,33 [+6,58; +12,21] |
| umieszczona | 74,72% | 84,86% | +10,14 |
| > 5 m wśród umieszczonych | 7,71% | 9,18% | +1,47 [+0,46; +2,73] |

Dolna granica przedziału jest powyżej 0. Opisowo: 77,07% leży powyżej obu bramek z GSR valid (68,18% / 73,18%),
które tu służą tylko do orientacji (inny zbiór, inny sufit: 87,0% wobec 93,3%).

Klatki według statusu śledzenia:

| status | klatki | kandydat dobrze | kandydat > 5 m | top-1 dobrze | zysk, pp całości |
|---|---|---|---|---|---|
| zmierzone | 4 143 (78,5%) | 89,1% | 6,6% | 85,9% | +2,46 |
| interpolowane | 542 (10,3%) | 69,6% | 25,5% | 14,4% | +5,67 |
| bez piłki | 592 (11,2%) | 0% | 0% | 0,2% | −0,02 |

(> 5 m w tej tabeli liczone wśród wszystkich klatek statusu; klatki bez homografii nie mają pozycji.)

## Wnioski

1. Kierunek z GSR valid się powtarza: śledzenie z filtrem daje na teście +8,1 pp [+5,7; +10,6] nad top-1. Zysk jest
   mniejszy niż na valid (+16,1 pp), bo test jest dla dd2c łatwiejszy (top-1 69,0% wobec 60,2%), podobnie jak gra 5
   w 04 (+9,2 pp).
2. Jak na valid, większość zysku to wypełnione luki (+5,7 z 8,1 pp), a 25,5% interpolowanych pozycji jest dalej niż
   5 m. Na teście odsetek > 5 m wśród umieszczonych lekko rośnie (+1,5 pp), na valid malał. Flaga `interpolated`
   jest potrzebna.
3. Sufit testu to 87,0% (688 klatek bez homografii); wśród klatek z homografią kandydat umieszcza dobrze 88,6%.

## Koszt i GPU

Zrzut 25 fps: 36 750 klatek, 20,85 ms/klatkę (RTX 4070 Ti SUPER, batch 1), GPU 19:57–20:10 (ok. 0,22 GPU-h; blokada
`agent-state gpu` ustawiona przed i zwolniona po zrzucie). Ruch kamery i śledzenie na CPU (śledzenie 0,22 ms/klatkę).

## Kod i odtworzenie

`scripts/ball_temporal.py` (commit f2089c6): `--split test` w `dump`, `motion`, `track` (klatki 25 fps z
`FULL_FRAMES`, kontrola `check_frames`), `track --post-filter`, `read` (umieszczenie z
`detector_bench.placement_rows`, losowania i przedziały z `ball_scorecard`, `--reference` dla pary z innym torem).
Testy w `tests/test_ball_temporal.py`. Bez zmian w `src/`.

```bash
B=/mnt/storage/tactifoot_research/detector_data/ball
uv run python scripts/ball_temporal.py track --games all --post-filter --no-card-dumps \
  --setting $B/04/tune/chosen.json --out $B/09/track_valid
uv run python scripts/ball_temporal.py read --tracks $B/09/track_valid --out $B/09/read_valid
$B/09/run_test.sh   # dump --split test (GPU), motion --split test, track --split test --post-filter
uv run python scripts/ball_temporal.py read --split test --tracks $B/09/track_test --out $B/09/read_test
```

Poza gitem, `/mnt/storage/tactifoot_research/detector_data/ball/09/`: `dump25_test/` (zrzut, `dump.json`),
`motion_test/`, `track_test/`, `read_test/` (`read.md`, `read.json`, klatki umieszczenia), `track_valid/`,
`read_valid/`, `read_valid_vs_track04/`, kontrole `check_04_read/`, `check_04_retrack/`, `run_test.sh`, `logs/`.
