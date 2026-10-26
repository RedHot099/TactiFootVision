# ball-detection 10: gra 4 GSR train jako holdout odczytu

2026-10-26. Rejestracja komentarzem w tickecie (09:45) przed jakimkolwiek przebiegiem na grze 4. **Ten ticket nie
czyta wyniku żadnego detektora ani kandydata na grze 4:** nie ma tu odsetków dobrze umieszczonych, kart ani torów
na grze 4. Są homografie, ich jakość, zrzuty i kod odczytu.

## Klipy

Gra 4 GSR train to 18 klipów SNGS-060…077, po 750 klatek (`info.game_id` = 4). Lista jest w
`research/ball_holdout_game4.json` (`detector_bench.py holdout`): klipy, sha256 etykiet, bliźniaki SoccerNet
Tracking (SNMOT-060…077 w `data/soccernet/tracking/extracted/train`) i liczby klatek co 5. (2 700 klatek, 2 519 z
piłką; ramki: 2 524 piłki, 1 374 bramkarzy, 42 617 graczy, 4 075 sędziów). Sprawdzone: w rozpakowanym SoccerNet
Tracking (train i test) tylko te 18 sekwencji ma `gameID=4`. Ten sam plik jest listą wykluczeń dla ticketów 13–15:
żaden z tych klipów, ich bliźniaków SNMOT ani obrazów z nich pochodzących nie może trafić do widoku treningowego.

## Homografie `kp03_r2_refined`

- **Kod:** `experiments/claude_fable/scripts/pitchkp_gsr_eval.py` z worktree `research-xg-detector` przy commicie
  c246e8d. To checkout pojedynczego odczytu GSR test pitch-keypoints 03; `code.digest` 9e4b8541… jest ten sam co w
  przebiegu valid (c18fbfd). Uruchomiony z czystego `git worktree --detach` tego commitu, z argumentami przebiegu
  testu: model punktów `models/pitch_yolo11m_pose_sn960.pt` (sha256 5be9954a…), `--input square --imgsz 960`,
  `--methods-file research/pitchkp_methods/chosen.json` (tylko `kp03_r2_refined`: próg 0,3, RANSAC 2, dopasowanie
  do linii z maską detektora pakietu, sha256 7183d342…). Biblioteki te same co w rekordzie pitch-keypoints.
- **Jedyna zmiana:** skrypt zna tylko split valid/test/calib, więc opakowanie (`ball/10/pk_run/run_split.py`)
  dopisuje `train` do listy dozwolonych splitów, a `--data` wskazuje katalog z 18 klipami gry 4. Bez
  `--reference-run` (dla train nie ma przebiegu bazowego; dotyczy to tylko `paired.csv`, nie homografii).
- **Kontrola odtworzenia przed użyciem:** to samo uruchomienie na 3 klipach GSR valid (SNGS-021, 039, 078, po
  jednym z gier 2, 3, 5) dało homografie **bit w bit równe** zamrożonym `gsr_valid_homographies.parquet` (441 z 441
  klatek, max |ΔH| = 0, te same klatki bez homografii, te same błędy na osobach).
- **Zamrożenie:** `bench/gsr_holdout_homographies.parquet` (sha256 cd413832…, 2 320 klatek z homografią z 2 700),
  rekord pod kluczem `placement_homographies_holdout` w `research/detector_bench.json`, poza zamrożonym manifestem
  (jak test w 05). Rekord zawiera sha256 źródła, `complete.json` przebiegu (git c246e8d, czysty), model i metodę.

## Jakość homografii (bez predykcji piłki)

Mianownik reguły 02 (klatki co 5. z dokładnie jedną piłką na boisku), klatki bez homografii liczone jako pudło.
Błąd w metrach to miara pitch-keypoints: mediana błędu stóp osób GSR na klatce (`error_m` z `per_frame.parquet`),
liczona na klatkach mianownika z homografią. Ta miara nie dotyczy piłki.

| zbiór | klipy | klatki mianownika | bez homografii | sufit [95%] | najgorszy klip | mediana błędu osób | p90 | klatki < 1 m | klatki > 5 m | zgodność linii (mediana) | z dopasowaniem do linii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GSR valid gry 2+3 | 39 | 4 250 | 347 (8,2%) | 91,8% [86,9; 95,9] | 18,8% | 0,59 m | 2,18 m | 71,3% | 2,8% | 0,95 | 78,5% |
| GSR valid cały | 58 | 6 404 | 426 (6,7%) | 93,3% [90,0; 96,1] | 18,8% | 0,62 m | 2,48 m | 69,1% | 2,8% | 0,87 | 68,7% |
| **gra 4 (holdout)** | 18 | **2 062** | **221 (10,7%)** | **89,3% [83,6; 94,6]** | 70,1% | 0,76 m | 2,15 m | 63,7% | 3,1% | 0,42 | 25,1% |

Te same liczby z rekordu pitch-keypoints (wszystkie klatki z osobą na boisku, nie tylko mianownik piłki):
dostępność 87,5% [80,4; 94,3] na grze 4 wobec 93,0% [89,2; 95,9] na valid; mediana 0,72 m [0,61; 0,86] wobec
0,62 m [0,56; 0,71]; p90 2,01 m wobec 2,35 m. Błąd punktów boiska w pikselach (720p, względem etykiet ticketu 01
pitch-keypoints): mediana 6,4 px [6,1; 6,9] na grze 4 wobec 7,8 px [7,4; 8,3] na valid; p90 15,9 px wobec 20,7 px.

Ruch kamery (`motion --split holdout`, ustawienia ticketu 04): 0 przerw na 13 482 parach klatek. Kotwica
przeniesiona o 5 klatek różni się od następnej zamrożonej o 0,36 m (mediana klipów; bez kompensacji 0,73 m). Na
valid w 04 było to 0,34 m (0,80 m).

**Ocena:** homografie gry 4 da się porównać z valid. Sufit jest o 2,6 pp niższy niż na grach 2+3, a przedziały się
nakładają. Błąd w metrach jest nieco wyższy (mediana 0,76 m wobec 0,59 m), ogon (p90) podobny. Główna różnica to
linie boiska: na grze 4 dopasowanie do linii zadziałało tylko na 25% klatek (na valid na 78%), a zgodność linii jest
niska (0,42 wobec 0,95). Za to błąd punktów w pikselach jest mniejszy niż na valid.

**Uwaga do interpretacji (ważna dla 16):** model punktów boiska (5be9954a…) był trenowany na 2 474 klatkach gry 4
(`pitch_keypoints_sn/images/train/gsr_train_SNGS-060…077_*`, czyli dokładnie te klatki co 5., na których leży
mianownik). Homografie gry 4 są więc „in-sample” dla modelu punktów; widać to po mniejszym błędzie w pikselach. W
metrach przewagi nie ma, bo słabiej działa dopasowanie do linii. Porównanie parami w 16 korzysta z tych samych
homografii po obu stronach, więc różnica wpływa na poziom odsetków i sufit, nie na samo porównanie. Do tego
dochodzi znane już zastrzeżenie, że dd2c widział grę 4 w widoku run 2c.

## Zrzuty (gotowe dla 16, nieczytane)

- **25 fps dd2c 1280:** `ball/10/dump25_holdout` (`ball_temporal.py dump --split holdout`): 18 klipów, 13 500
  klatek, 371 043 kandydatów wszystkich klas z conf ≥ 0,01, bez filtra (filtr 45 px nakłada `track --post-filter`,
  jak w 09), 17,67 ms/klatkę. Klatki z SoccerNet Tracking train, bo lokalny GSR train ma tylko co 5. klatkę
  (`fetch_log.jsonl`). Wszystkie 2 700 lokalnych klatek GSR train są bajt w bajt identyczne z klatkami SNMOT
  (sprawdzone przy zrzucie, zapisane w `dump.json`).
- **Ruch kamery:** `ball/10/motion_holdout` (CPU, kotwice z zamrożonych homografii gry 4).
- **Zrzut karty dd2c:** `ball/10/card_dump_dd2c_holdout` (`detector_bench.py dump --split holdout`, wiersze 1280 i
  e3, conf ≥ 0,001, 2 700 obrazów, sama gra 4 bez Ekstraklasy). To odniesienie karty dla wariantów detektora w 16.
  Nie był kartowany.

## Kod

- `detector_bench.py`: split odczytu `holdout` (`READ_SPLITS`, `HOLDOUT_FILE`, `gsr_folder`, `split_label_files`),
  polecenie `holdout` (lista klipów, kontrola bliźniaków SNMOT, niezmienna po zapisie), `homographies --split
  holdout` (zamrożenie z rekordem przebiegu pitch-keypoints), `dump --split holdout`, `placement_rows` i
  `frozen_homographies` dla holdoutu (`placement_rows` odrzuca klip spoza gry 4).
- `ball_temporal.py`: `--split holdout` w `dump`, `motion`, `track`, `read`; klatki 25 fps z SoccerNet Tracking train.
  `read` ma nową tabelę bariery fazy 2: odsetek klatek mianownika z piłką > 5 m, wartość, różnica parami z
  przedziałem i czy wzrost ≤ 1 pp (`ball_scorecard.barrier`, `BARRIER_PP`).
- `ball_scorecard.py`: `card --holdout` (gra 4 sama, z dumpów `dump --split holdout`, bez Ekstraklasy, kontrola
  splitu dumpu), sekcja `barrier` w każdej karcie GSR, w regule `holdout` (dolna granica > 0 i bariera) oraz
  `barrier` dla selekcji.
- `ball_errors.gsr_balls`: parametr `split` (folder GSR), domyślnie valid, bez zmiany zachowania.
- Testy: `tests/test_ball_holdout.py` (5): lista klipów i jej niezmienność, klatki 25 fps z kontrolą bajtów,
  zamrożenie homografii, odczyt na małym syntetycznym zrzucie (odsetek, przedział, bariera > 5 m, tabela) i
  plumbing karty.

Polecenia dla 16 (nie uruchamiane tutaj na prawdziwych torach):

    uv run python scripts/ball_temporal.py track --split holdout --games all --post-filter \
        --setting ball/04/tune/chosen.json --dump ball/10/dump25_holdout --motion ball/10/motion_holdout --out OUT
    uv run python scripts/ball_temporal.py read --split holdout --tracks FINAL --reference PHASE1/tracks.parquet --out OUT
    uv run python scripts/ball_scorecard.py card --holdout --weights W --reference ball/10/card_dump_dd2c_holdout --out OUT

## Kontrole

- **Regresja:** `read` nowym kodem na `ball/09/track_valid` odtwarza `ball/09/read_valid` co do cyfry: 319 z 319
  liczb `read.json` identycznych, wszystkie cztery `placement_*.parquet` identyczne, `read.md` identyczny poza
  dopisaną tabelą bariery (`ball/10/regress_read_valid`).
- **Plumbing na grze 4 bez odczytu:** karta `--holdout` i `track` + `read --split holdout` na sztucznych dumpach bez
  piłek (`ball/10/smoke_empty`; wszystkie odsetki 0). Zgadzają się mianownik 2 062 klatek, 221 bez homografii i 18
  klipów; tabela prawdy gry 4 (tylko etykiety, 2 524 piłki) trafiła do cache `ball/02/truth`.
- `ruff` czysty; testy ball_* 66 passed (z `test_detector_train.py` 76); szybki zestaw 1 023 passed, 8 skipped.

## Odstępstwa i uwagi

1. Opakowanie dopisujące split `train` do skryptu pitch-keypoints, bez zmian parametrów (kontrola odtworzenia
   powyżej). Bez `--reference-run`.
2. Dodatkowy zrzut karty dd2c na grze 4 (ok. 2 min GPU), żeby karta w 16 była jednym poleceniem.
3. Bariera jest nowym wyjściem `read` i karty (dopisanym, bez zmiany istniejących liczb). Powtórka odczytu 09 na
   całym valid (z grą 5, jak w 09) wypisała przy okazji barierę kandydata fazy 1 względem top-1 na całym valid. Nic
   na tej podstawie nie jest dobierane i liczby nie cytuję.
4. `ball_errors.py` dostał parametr `split` w `gsr_balls` (poza trzema plikami ticketu, zmiana wsteczna zgodna).

## GPU

`agent-state gpu` sprawdzone przed (wolne). Busy 09:45–09:51 (homografie: 3 klipy valid i gra 4, ok. 1,5 min GPU) i
09:59–10:07 (zrzut 25 fps i zrzut karty), potem `free`. Razem ok. 0,25 GPU-h blokady.
