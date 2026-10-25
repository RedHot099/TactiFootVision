# ball-detection 05: dotrenowanie nastawione na piłkę, z ramieniem kontrolnym

2026-10-25. Oba ramiona startują z dd2c (`runs/run2c/CANDIDATE_epoch1.pt`, sha256 05d706cf…) na restartowanym
przepisie run 2c (yolo11l, 1280, batch 4, SGD, cos_lr od lr0 0,002, scale 0,3, maskowany loss klas, seed 0,
close_mosaic 1), 2 epoki po 28 550 iteracji. Różni je tylko lista treningowa. Odczyty tylko na splicie wyboru
(GSR valid, gry 2+3). Gry 5 nie czytano, bo zwycięzcy nie ma. GSR testu też nie czytano.

**Wynik: brak efektu nadpróbkowania.** Porównanie główne (reguła 02): B − A w odsetku dobrze umieszczonych przy
1280, obie po ostatniej epoce: **+0,26 pp [−0,80; +1,35]**. Przedział obejmuje 0, więc zwycięzcy, odczytu
potwierdzającego ani kandydata nie ma.

## Widoki

Kod: `scripts/ball_view.py build`, `RUNS["bd05_A" | "bd05_B"]` w `scripts/detector_train.py`, testy
`tests/test_ball_view.py`. Rekordy: `research/detector_bd05_A_view.json`, `research/detector_bd05_B_view.json`.

- **A (kontrola):** lista run 2c bajt w bajt (sha256 9619660044c7…): 100 214 obrazów, 114 198 linii na epokę.
- **B (piłka):** obraz trafia do koszyka porażek z ticketu 01, gdy któraś piłka prawdy jest mała (średnica
  < 12 px w skali 1080p wejścia 1280, czyli średnica × 1920 / dłuższy bok) albo rozmyta (tylko GSR train:
  prędkość z `track_id` ≥ 1 średnica/klatkę). Źródła bez torów binowane tylko po rozmiarze. Taki obraz jest
  listowany ×2 względem A. Długość epoki wyrównana usunięciem 18 151 nieparzystych klatek GSR train (seed 0,
  bez patrzenia na koszyk; każda usunięta klatka ma obu sąsiadów w widoku). Wynik: 82 063 obrazy, 114 198 linii.

| linie epoki | A | B |
|---|---|---|
| mała (< 12 px) | 21 518 (18,8%) | 31 412 (27,5%) |
| rozmyta (GSR) | 5 390 (4,7%) | 6 178 (5,4%) |
| mała lub rozmyta | 25 761 (22,6%) | 36 302 (31,8%) |
| GSR train razem | 42 750 | 34 969 |

Obrazy w koszyku na źródło (B: obrazy / linie): GSR 10 370 / 20 740, TeamTrack 4 017 / 8 034, v3 660 / 1 320,
ISSIA 591 / 1 182, Roboflow 573 / 1 146, football_yolo 97 / 3 880.

## Karta na epokę (gry 2+3, wiersz 1280)

| ramię, epoka | dobrze umieszczona | > 5 m wśród umieszczonych | AP50 piłki | mAP50-95 ludzi | Eks. AP50 piłki | Eks. mAP50-95 |
|---|---|---|---|---|---|---|
| dd2c | 51,60% | 18,4% | 0,296 | 0,514 | 0,279 | 0,424 |
| A e0 | 50,85% | 19,3% | 0,286 | 0,513 | 0,298 | 0,412 |
| A e1 | 43,08% | 41,1% | 0,134 | 0,478 | 0,288 | 0,403 |
| B e0 | 53,67% | 16,0% | 0,310 | 0,527 | 0,290 | 0,403 |
| B e1 | 43,34% | 41,0% | 0,150 | 0,480 | 0,287 | 0,399 |

Różnice parami (pp albo AP; 95% bootstrap po klipach, 2 000 losowań, seed 0):

| | dobrze umieszczona | AP50 piłki | mAP50-95 ludzi | Eks. mAP50-95 |
|---|---|---|---|---|
| **B − A, e1 (główne)** | **+0,26 [−0,80; +1,35]** | +0,016 [+0,009; +0,025] | +0,002 | −0,005 |
| B − A, e0 | +2,82 [+1,71; +4,00] | +0,023 [+0,012; +0,035] | +0,014 | −0,009 |
| B − dd2c, e1 | −8,26 [−11,68; −5,25] | −0,146 | −0,034 | −0,025 |
| B − dd2c, e0 | +2,07 [+1,15; +3,04] | +0,014 | +0,013 | −0,021 |
| A − dd2c, e1 | −8,52 [−11,93; −5,56] | −0,162 | −0,036 | −0,021 |
| A − dd2c, e0 | −0,75 [−1,59; +0,14] | −0,009 | −0,001 | −0,012 |

Strażnicy B e1: wobec A spełnieni (ludzie +0,002, Ekstraklasa −0,005). Wobec dd2c niespełnieni: ludzie −0,034
(próg −0,01), Ekstraklasa −0,025 (próg −0,02). Ekstraklasa AP50 piłki ≥ 0,40: nie (0,287). Pełna tabela z
przedziałami i wierszem e3 (B − A e1 przy e3: +2,05 [+0,68; +3,56], opisowo):
`/mnt/storage/tactifoot_research/detector_data/ball/05/table.md`.

## Co się stało w epoce 1

Epoka 1 (indeks 1, bez mozaiki, bo close_mosaic 1) psuje oba ramiona tak samo. Fałszywe piłki ≥ 0,15 rosną z
0,19–0,22 do 0,52 na klatkę, precyzja piłki spada z ok. 0,49 do 0,31, a odsetek > 5 m wśród umieszczonych z
16–19% do 41%. Recall piłki prawie się nie zmienia (0,35–0,37 → 0,31–0,32). Val Ultralytics (mAP50 0,754 → 0,729
w A, 0,764 → 0,724 w B) pokazuje to samo. Run 2c miał ten sam wzór w swojej epoce bez mozaiki (AP50 piłki 0,356
→ 0,304). Harmonogram jest restartowany, więc całe 2 epoki leżą w rozgrzewce Ultralytics (3 epoki): lr biasu
spada z 0,1 tylko do 0,034, a reszta parametrów nie dochodzi do lr0. Po epoce 0 B wyprzedza A o +2,8 pp i dd2c o
+2,1 pp. Z tego, co mierzy karta, epoka bez mozaiki zaciera ten zysk, a nadpróbkowania nie da się od niej oddzielić.
Wg reguły 02 wariant opisowy (B e0) nie może wygrać tego ticketu. Może być najwyżej porównaniem głównym
późniejszego ticketu, zapisanym przed jego odczytem.

Recall IoU w koszykach (0,15): 8–12 px A/B e0 0,255 / 0,275, e1 0,215 / 0,225; rozmyta e0 0,180 / 0,187,
e1 0,154 / 0,157; < 8 px wszędzie ≈ 0.

## Koszt i GPU

ms/klatkę przy 1280 bez zmian (19,7–23,7, ten sam yolo11l). GPU (RTX 4070 Ti SUPER, blokada `agent-state gpu`
przy każdym kroku): A 21:44–01:13 (3,49 h); B 08:39–11:00 (proces zabity bez śladu przy 9 150 / 28 550 iteracji
epoki 1, ok. 32 min straconych), wznowienie z `last.pt` 15:10–16:54 (`--resume`, w jednostce `systemd-run
--user`); zrzuty 4 checkpointów (wiersze 1280 i e3) 16:56–17:39. Razem **ok. 8,3 GPU-h**. Logi:
`runs/bd05_{A,B}/logs/gpu.log`, `ball/05/logs/gpu.log`.

## Wnioski na następny szczebel

1. Nadpróbkowanie koszyków ×2 przy ostatniej epoce: brak efektu (+0,26 pp [−0,80; +1,35]).
2. Większym problemem jest epoka bez mozaiki na restartowanym harmonogramie: −8,5 pp wobec dd2c w samym ramieniu
   kontrolnym. Kolejny szczebel dotrenowania powinien zapisać z góry: close_mosaic 0 albo 1 epoka, albo
   warmup_epochs dopasowane do długości przebiegu, i porównanie B − A po epoce 0 jako główne.
3. Twarde negatywy (sędziowie i liniowi w czerni jako „piłka”, ticket 01) świadomie pominięte, żeby nie
   pomieszać B − A. To kandydat na osobny szczebel, zwłaszcza że epoka 1 podwaja fałszywe piłki.

## Ścieżki (poza gitem)

`/mnt/storage/tactifoot_research/detector_data/runs/bd05_A/`, `…/bd05_B/` (`RUN.md`, `train.txt`, `bins.parquet`
w B, `yolo11l_1280/weights/epoch{0,1}.pt`, `last.pt`; w B także `last_epoch0_before_resume.pt`).
`/mnt/storage/tactifoot_research/detector_data/ball/05/` (`{A,B}_e{0,1}/` z kartą i zrzutem,
`B_e{0,1}_vs_A/`, `table.md`, `table.csv`, `verdict.json`, `run_dumps.sh`, `logs/`).

Odtworzenie: `uv run python scripts/ball_view.py build`; `scripts/detector_train.py train --run bd05_A|bd05_B`;
`ball/05/run_dumps.sh A_e0 A_e1 B_e0 B_e1`; `scripts/ball_scorecard.py card --dump …/X/dump --out …/X
[--reference …/A_eN/dump]`; `scripts/ball_finetune.py table`.
