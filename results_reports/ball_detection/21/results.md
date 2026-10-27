# Ticket 21: inwentarz danych z etykietami piłki (ball-detection, faza 3)

Tylko CPU, bez GPU i bez pobierania wideo. Liczby z `pool_index.parquet` (kolumna `n_ball`, obrazy po normalizacji i
cieńczeniu), `sets/*/manifest.json`, `scripts/detector_train.py` (`RUN1_SETS`, `RUNS`), `research/detector_*`,
metadanych modelu punktów (`models/pitch_yolo11m_pose_sn960.pt.json`), raportów ticketów 10, 14, 15, 16 i metadanych
HF (API + README repo `atomscott/soccertrack-v2`).

## Co dotyka każdego źródła: reguły

* **dd2c** = epoka 1 przebiegu run 2c = run 1 (`RUN1_SETS`) + `football_yolo` x20, bez splitu `test` football_yolo
  (klipy Ekstraklasa dev) i bez gier GSR valid/test (2, 3, 5, 7, 8, 11) oraz ich near-duplikatów. Zbiory wchodzą
  **wszystkimi splitami** (train/valid/test źródła), bo widok tnie tylko GSR i holdout run 2c. Wynik: SoccerNet-v3
  (także jego `test`), TeamTrack `test`, Soccana `test` itd. są w treningu dd2c.
* **bd14/bd15:** startują z dd2c, lista run 2c bez klatek gry 4 (A) + twarde negatywy sędziów (B); bd15 = B w 1600.
  Walidacja: gry 2+3 + Ekstraklasa dev.
* **Model punktów boiska** (`pitch_yolo11m_pose_sn960`): trening na `pitch_keypoints_sn/images/train` = 8 082 klatek
  GSR train (wszystkie 57 klipów SNGS-060…170, czyli gry 4, 6, 9) + 7 585 klatek Calibration-2023 train; wybór epoki i
  metody na GSR valid (8 243 klatek) + Calibration valid; GSR test i Calibration test odczytane jednorazowo w
  pitch-keypoints 03. Nie widział piłek, ale jego homografie niosą mianownik odczytów piłki.
* **Dobór:** na dd2c/bd14 używano GSR valid gry 2+3 (walidacja, wybór ustawień, odczyty ticketów 02–08, 11–16), gra 5
  czytana w 04 i 08, Ekstraklasa dev jako strażnik (mAP50-95) i walidacja.
* **Odczyty:** GSR test (gry 7, 8, 11) odczytany w 09 (faza 1) i opisowo w fazie 2; gra 4 raz w 10 (tylko jakość
  homografii; bez predykcji piłki), w 18 jeden odczyt wg D1.

## Tabela inwentarza

Liczba piłek = pola `ball` w etykietach puli (`n_ball`, po cieńczeniu); „poz.” = pozycja piłki na boisku w metrach.

| źródło | domena / kamera | piłek z etykietą (obrazów) | poz. na boisku | dd2c | model punktów | dobór | odczyty | nietknięte? |
|---|---|---|---|---|---|---|---|---|
| GSR train (gry 4, 6, 9) | broadcast SoccerNet, 57 klipów | 40 931 (42 750 klatek co 5.) | tak (GSR: piłka z homografią) | tak (run 1, 2c, bd14: bez gry 4) | tak: 8 082 klatek | tak (zbiór treningowy) | gra 4: ticket 10 (tylko homografie), 18 (D1) | **nie**: trening; gra 4 dodatkowo widziana przez dd2c i model punktów |
| GSR valid (gry 2, 3, 5) | broadcast | 40 900 (43 500) | tak | nie (wykluczone) | wybór epoki i metody (8 243 klatek) | tak: gry 2+3 walidacja i wybór; 5 czytana w 04, 08 | tak, wiele | **nie** |
| GSR test (gry 7, 8, 11) | broadcast | 34 129 (36 750) | tak | nie (wykluczone) | jeden odczyt końcowy | nie (zasada) | 09 (faza 1), faza 2 opisowo, 17 opisowo | **nie**: odczytane, wolno już tylko opisowo na zamrożonych predykcjach |
| GSR challenge | broadcast | brak lokalnie, etykiet nie ma w repo | - | nie | nie | nie | nie | nie do użycia (brak etykiet) |
| SoccerNet Tracking 2022/23 (SNMOT) | te same klipy co GSR | 0 własnych (boksy osób; bez piłki) | - | pośrednio (klatki GSR) | - | - | użyte tylko jako źródło klatek 25 fps | **ten sam materiał co GSR**: SNMOT train 060…170 = SNGS train, SNMOT test 116…200 = SNGS test, 2 700 + 7 350 klatek co 5. bajt w bajt (09, 10); nie jest drugim źródłem |
| SoccerNet-v3 | broadcast, 400 gier (297+48+55), akcje i powtórki | 26 376 (26 376 obrazów) | nie (są pola obrazu, kalibracja tylko w v3D) | tak, **wszystkie splity** | nie | w treningu | porównania źródeł (07) | **nie**; część gier także w puli xG (`xg_role`: training/evaluation/gate) |
| SoccerNet-v3D | te same powtórki co v3, 379 gier | 4 051 (wszystkie z piłką; 3 241 train / 810 test) | **tak, 3D piłki i kalibracja kamery** (z optymalizacji reprojekcji, nie pomiar) | tak: jako dopracowane pola (3 015 z 4 051 uznanych) w `soccernet_v3` | nie | nie | nie | **nie**: obrazy i pola już w treningu; pozycje na boisku nie były używane, ale obrazy tak |
| TeamTrack | stałe kamery z góry/drona, 2 sporty, 7 gier | 4 065 (4 140 klatek: train 2 837, valid 608, test 620) | nie | tak (wszystkie splity) | nie | w treningu | 07 | **nie** |
| ISSIA | 6 stałych kamer, 1 mecz | 834 (3 126 klatek) | nie | tak | nie | w treningu | 07 | **nie** |
| Soccana | mieszane (SoccerNet, rendery, publiczne), 11 434 klatek | 4 978 (3 461 klatek z piłką) | nie | tak (train i test) | nie | w treningu | 07 | **nie** |
| Roboflow (DFL: ball + players) | klipy Bundesliga, 14 klipów | 970 (656 obrazów ball-only + 314 w 359 obrazach players) | nie | tak | nie | w treningu | 07 | **nie** |
| football_yolo train/valid | Lech Poznań (Ekstraklasa/UEFA), kamery LiveTag i szeroka, 6 gier | 675 (736 obrazów) | nie | tak, x20 | nie | w treningu | - | **nie** |
| Ekstraklasa dev (football_yolo `test`) | j.w. 2 gry, 56 obrazów | 46 | nie | nie (holdout) | nie | **tak**: strażnik mAP50-95 i walidacja (run 2, bd14/15) | strażnik w 05–15 | **nie**: zbiór kontrolny faz 1–2, tylko 46 piłek |
| SportsMOT (football) | broadcast | 0 (brak piłki) | - | tak (tylko ludzie) | - | - | - | brak etykiet piłki |
| Soccer Player Detection | 2 gry | 0 | - | tak (ludzie) | - | - | - | brak etykiet piłki |
| SoccerSynth | syntetyczne | 17 126 (13 855 + 3 271 obrazów) | nie (syntetyczne, pozycje silnika nie wczytane) | **nie** (zbiór po run 1; run 2b odrzucony) | nie | sprawdzany w 06 | 06 | nie dotyczy: syntetyczne, poza domeną; ticket 06 go już użył |
| SoccerDB | transmisje, chiński mundial 2014 | 0 lokalnie (tylko CSV kluczy; dane na Baidu) | - | nie | nie | nie | nie | brak danych do użycia |
| SoccerTrack v2 | panorama 3840x1504, stała kamera, 10 meczów (934 min) | **0 pól piłki w obrazie** (patrz niżej) | pozycje piłki tylko w gated `ball/*.npz`: rekonstrukcja z interpolacją liniową między zdarzeniami, nie pomiar | nie | nie | nie | nie | nietknięte, ale **bez użytecznych etykiet** |

Przybliżenia: liczby TeamTrack i Soccana z `n_ball` po cieńczeniu duplikatów; Roboflow: liczby z `n_ball`
(dokładne sumy w `pool_index.parquet`).

## SoccerTrack v2: szczegóły

* **Lokalnie:** `data/soccertrack_v2/videos` ma tylko panoramy (7 plików, 4 mecze: 118577, 118578, 128057, 128058,
  plus trzy niedokończone `.crdownload`); `football_data/xg_shots_full/soccertrack_v2` ma wideo 10 meczów (po połowach),
  `gsr/*.json.zst`, `bas/` (zdarzenia), `challenge_tracks/*.txt` (MOT) i `challenge_alignment.json`.
* **README HF** (dostępny z naszym tokenem; reszta repo: 403 "GatedRepo", konto nie jest na liście): `gsr/` = osoby
  (pozycje na boisku, numery, role); `bas/` = 21 428 zdarzeń akcji piłki (`frame` zsynchronizowany z wideo, wersja v1.2
  naprawia synchronizację meczu 132877); `ball/` = tor piłki per klatka, ale "reconstruction rather than a
  per-frame measurement", zakotwiczony w zdarzeniach i interpolowany odcinkami o stałej prędkości, w meczach
  132831/132877 przycięty do prostokąta boiska; `mot/` = ramki ograniczające dla SoccerTrack Challenge; `raw/` = 65
  ręcznych punktów boiska na mecz, intrinsics, homografia.
* **Sprawdzone lokalnie:** w `gsr/117092_1st.json.zst` kategorie: 1 (gracz) 1 347 468, 2 (bramkarz) 134 750, 5
  (boisko) 67 375; brak kategorii piłki. `challenge_tracks/117092.txt`: 22 id, każde na 6 000 klatkach: same osoby,
  brak toru piłki.
* **Wniosek:** nie ma pól piłki na obrazie (ani w `gsr`, ani w `mot`), a jedyny tor piłki jest rekonstrukcją z
  interpolacją liniową (błąd między zdarzeniami nieznany, ponad 1 s luk przy dwóch zdarzeniach na minutę) i leży za
  bramką HF. Odczyt piłki wymagałby nowych etykiet (AI-labelling wykluczony, StatsBomb nie dotyczy). Synchronizacja
  wideo z danymi jest obiecana w v1.2 dla zdarzeń i osób, ale to nie daje etykiet piłki do odczytu detektora.
  Dodatkowo domena (panorama 3840x1504, piłka rzędu kilku px) leży daleko od dd2c w 1280 px.

## Rekomendacja dla ticketu 22

**Brak sensownego źródła.** Warunek D5 (etykiety piłki z pozycją na boisku i synchronizacją) nie jest spełniony przez
żadne nietknięte źródło:

* nietknięte: SoccerTrack v2 (brak etykiet piłki w obrazie), SoccerDB (brak danych), GSR challenge (brak etykiet);
* z etykietami: wszystkie pozostałe są w treningu dd2c (wszystkie splity), w doborze albo w odczytach.

Ticket 22 powinien się nie wykonywać (warunkowy, D5). W raporcie fazy 3 należy zapisać: wyniki poza domeną nie istnieją,
a jedyny odczyt z innej domeny to Ekstraklasa dev (46 piłek, zbiór kontrolny faz 1–2), opisowo.
Jeśli właściciel chce mimo to wyjść poza domenę: (a) poprosić o dostęp do HF (konto redhot3) i sprawdzić, czy
`ball/` ma choćby kotwice zdarzeń (piłka znana tylko w momencie zdarzeń, ok. 21 tys. punktów), co dałoby ocenę tylko w
klatkach zdarzeń, bez pól w obrazie, więc z pozycją na boisku zamiast pola (porównanie przez projekcję); to jest
decyzja zakresu, nie moja; (b) nowe etykiety ręczne (poza zasadami).

## Uczciwość: gry 4 i 5

Gra 4: widziana przez dd2c (run 2c), model punktów (2 474 klatek mianownika), bd14/15 ją wykluczają. Gra 5: czytana w
04 i 08. Żadnej nie nazywam nietkniętą. SNMOT nie jest niezależnym źródłem względem GSR.
