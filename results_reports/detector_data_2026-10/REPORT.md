# Nowy detektor ludzi i piłki z otwartych danych: raport fazy (detector-data, ticket 06)

Jakub Eichner, 2026-10-22. Gałąź `research/detector-data`, faza detector-data, tickety 01–05.

Raport zamyka fazę. Cel właściciela z 2026-10-14: „detektor jest po prostu słaby, chciałbym, żebyś zebrał jak najwięcej danych open access, które pozwolą nam dotrenować go do jak najlepszej jakości”.

W tekście źródłowym po każdej liczbie stoi niewidoczny znacznik `<!--n:…-->`. Plik `claims.toml` obok mówi, z którego pliku wyników pochodzi liczba. Sprawdzenie: `uv run python scripts/check_report.py results_reports/detector_data_2026-10/REPORT.md`. Skrypt odczytuje każdą liczbę ze źródła i kończy się błędem przy niezgodności. Pilnuje też rozdziału liczb: wybór na danych rozwojowych stoi tylko pod nagłówkami z „wyniki rozwojowe”, a jedyny odczyt GSR test tylko pod nagłówkami z „wynik testu”. Wyniki leżą na `/mnt/storage/tactifoot_research/detector_data/` (poza gitem), więc sprawdzenie działa tylko na tej maszynie.

## 1. W skrócie: wynik testu

**Model.** Nowy detektor to YOLO11l z wejściem 1280 px i tymi samymi 4 klasami co dotąd (piłka, bramkarz, zawodnik, sędzia). Nazwa: `dd2c`. Wagi: `models/detector_yolo11l_1280_dd2c.pt`, obok karta modelu. Od ticketu 05 to domyślny detektor pliku przebiegu (`configs/pipeline.yaml`).

**Test.** Kandydata wybrano tylko na danych rozwojowych. Potem raz oceniono go na GSR test: 7 350<!--n:bench.test.frames--> klatek, 49<!--n:bench.test.clips--> klipów z 3 meczów. Wynik starego detektora na tym teście, zapisany w tickecie 02, przeczytano razem z wynikiem kandydata.

| GSR test, wejście 1280 | stary detektor | dd2c |
|---|---|---|
| mAP50 | 0.229<!--n:t.b.1280.map50--> | **0.747<!--n:t.c.1280.map50-->** |
| mAP50-95 | 0.095<!--n:t.b.1280.map5095--> | **0.451<!--n:t.c.1280.map5095-->** |
| AP50 piłki | 0.183<!--n:t.b.1280.ball--> | **0.464<!--n:t.c.1280.ball-->** |
| AP50 bramkarza | 0.000<!--n:t.b.1280.gk--> | **0.666<!--n:t.c.1280.gk-->** |
| AP50 zawodnika | 0.733<!--n:t.b.1280.player--> | **0.953<!--n:t.c.1280.player-->** |
| AP50 sędziego | 0.000<!--n:t.b.1280.ref--> | **0.905<!--n:t.c.1280.ref-->** |
| piłka umieszczona na boisku | 63.6%<!--n:tp.b.placed.s--> | **77.9%<!--n:tp.c.placed.s-->** |
| umieszczona piłka dalej niż 5 m od prawdy | 19.9%<!--n:tp.b.far.s--> | **7.4%<!--n:tp.c.far.s-->** |

**Co dał każdy krok.**
- Run 1 (duży trening na prawdziwych obrazach) nauczył model bramkarzy i sędziów i podniósł piłkę.
- Run 2c dołożył obrazy z Ekstraklasy i naprawił tę domenę.
- SoccerSynth (dane syntetyczne) zaszkodził i nie wszedł do modelu.

**Co ogranicza dalej.** Piłka na Ekstraklasie i bramkarz na teście (sekcja 8). Dalsza praca nad piłką przeszła do `.scratch/ball-detection/` (sekcja 9).

## 2. Punkt wyjścia: stary detektor (wyniki rozwojowe)

**Stary detektor.** YOLO11m z wejściem 640<!--n:base.imgsz--> px. Uczony na zbiorze Roboflow `football_yolo`: 1 259<!--n:base.fy.train--> obrazów treningowych i 193<!--n:base.fy.valid--> walidacyjnych. Klatki pochodzą z meczów Lecha Poznań i dwóch innych transmisji.

**Benchmark (ticket 02).** Zamrożony przed treningiem. SoccerNet GSR (liga szwajcarska), co piąta klatka:
- dev = GSR valid: 8 700<!--n:bench.valid.frames--> klatek, 58<!--n:bench.valid.clips--> klipów, 3 mecze;
- test = GSR test: 7 350<!--n:bench.test.frames--> klatek, 3 inne mecze, czytany raz.

Miary:
- AP w stylu COCO dla każdej klasy;
- precyzja i recall w punkcie pracy pipeline'u (próg 0.3<!--n:op.conf-->, dla piłki 0.15<!--n:op.ball-->);
- umieszczenie piłki na boisku: środek ramki piłki o najwyższej pewności, rzutowany zamrożoną homografią z modelu punktów boiska; błąd w metrach wobec pozycji piłki z GSR.

Wiersz „1280” to wejście 1280 px. Umieszczenie piłki liczy się przy ustawieniu pipeline'u e3 (klatka 720p, imgsz 1920), jak w raporcie pitch-keypoints.

**Stary detektor na GSR valid:**

| wejście | piłka AP50 | bramkarz AP50 | zawodnik AP50 | sędzia AP50 | mAP50 / mAP50-95 | piłka P / R |
|---|---|---|---|---|---|---|
| 640 | 0.070<!--n:v.b.640.ball--> | 0.000<!--n:v.b.640.gk--> | 0.776<!--n:v.b.640.player--> | 0.015<!--n:v.b.640.ref--> | 0.215 / 0.094<!--n:v.b.640.map--> | 0.40 / 0.12<!--n:v.b.640.ballpr--> |
| 1280 | 0.141<!--n:v.b.1280.ball--> | 0.000<!--n:v.b.1280.gk--> | 0.757<!--n:v.b.1280.player--> | 0.012<!--n:v.b.1280.ref--> | 0.228 / 0.100<!--n:v.b.1280.map--> | 0.38 / 0.22<!--n:v.b.1280.ballpr--> |

Piłka umieszczona na 50.2%<!--n:vp.b.placed--> z 6 404<!--n:vp.frames--> klatek. Mediana błędu to 1.42 m<!--n:vp.b.med-->. Dalej niż 5 m leży 25.1%<!--n:vp.b.far--> umieszczonych piłek. Te liczby raport pitch-keypoints wskazał jako nowe ograniczenie.

**Dlaczego był słaby.**
1. **Nie odróżniał bramkarzy i sędziów.** Recall bramkarza 0.000<!--n:v.b.1280.gkr-->, sędziego 0.002<!--n:v.b.1280.refr--> (wejście 1280). Na próbce z ticketu 02 większość bramkarzy i sędziów GSR dostała ramkę „zawodnik”.
2. **Mały i wąski zbiór.** Nieco ponad tysiąc obrazów z kilku transmisji to za mało, żeby przenieść się na inne ligi i kamery.
3. **Za małe wejście dla piłki.** Piłka ma na klatce kilkanaście pikseli, a przy 640 px jeszcze mniej. Ten sam stary model przy 1280 ma AP50 piłki 0.141<!--n:v.b.1280.ball-->, przy 640 tylko 0.070<!--n:v.b.640.ball-->.
4. **Wynik na własnym zbiorze był zawyżony.** Na `football_yolo` valid stary model ma mAP50 0.785<!--n:v.b.fy.640.map50-->, ale walidacja dzieli klipy z treningiem.

## 3. Zebrane dane (tickety 01 i 03)

**Zasada właściciela (2026-10-14):** zebrać wszystko, oznaczyć pochodzenie każdego obrazu, a podział zrobić później. Każdy obraz ma znaczniki: źródło, mecz, klip, klatka, podział GSR, rola meczu w xG, duplikat benchmarku, licencja i oznaczone klasy. Przed run 1 pula miała 179 728<!--n:pool.run1--> obrazów. Po dodaniu SoccerSynth i `football_yolo` miała 231 397<!--n:pool.run2--> obrazów.

**Źródła użyte w treningu** (obrazy i ramki piłki w widoku treningowym; liczby z karty modelu):

| źródło | charakter | obrazy | ramki piłki | oznaczone klasy | licencja |
|---|---|---|---|---|---|
| SoccerNet GSR (3 mecze treningowe) | transmisje TV, liga szwajcarska | 42 750<!--n:src.gsr.img--> | 40 931<!--n:src.gsr.ball--> | wszystkie 4 | NDA SoccerNet, tylko badania |
| SoccerNet-v3 (piłka poprawiona z v3D) | transmisje TV, mecze lig europejskich | 33 986<!--n:src.v3.img--> | 26 376<!--n:src.v3.ball--> | wszystkie 4 | NDA SoccerNet, tylko badania |
| Soccana | mieszanka z Hugging Face (kafle, rendery, augmentacje) | 11 434<!--n:src.soccana.img--> | 4 978<!--n:src.soccana.ball--> | piłka, zawodnik, sędzia | brak licencji |
| TeamTrack | panorama 8K i dron | 4 140<!--n:src.tt.img--> | 4 065<!--n:src.tt.ball--> | piłka, osoba | MIT (słaba podstawa) |
| ISSIA-CNR | 6 stałych kamer, 1 mecz | 3 126<!--n:src.issia.img--> | 834<!--n:src.issia.ball--> | piłka, osoba | brak licencji |
| SportsMOT (piłka nożna) | klipy TV do śledzenia | 1 952<!--n:src.smot.img--> | 0 | osoba | CC BY-NC 4.0 |
| SoccerPlayerDetection (UBC) | 2 transmisje z YouTube | 1 075<!--n:src.spd.img--> | 0 | osoba | tylko badania |
| Roboflow (piłka + zawodnicy) | klipy DFL Bundesliga | 1 015<!--n:src.rf.img--> | 970<!--n:src.rf.ball--> | wszystkie 4 (zbiór piłki: tylko piłka) | CC BY 4.0, materiał źródłowy niejasny |
| `football_yolo` (tylko run 2c) | Lech Poznań, UEFA, 2 inne transmisje | 736<!--n:src.fy.img--> | 675<!--n:src.fy.ball--> | wszystkie 4 | nieznana (eksport Roboflow) |

„Osoba” znaczy: źródło oznacza każdą osobę jako zawodnika, bez ról.

**Zebrane, ale nie w modelu:** SoccerSynth-Detection, 50 877<!--n:src.synth.img--> renderów i 17 126<!--n:src.synth.ball--> ramek piłki. Oznaczone są tylko piłka i osoba. Rozmycie ruchu miało 18 441<!--n:src.synth.blur--> z 54 248<!--n:src.synth.raw--> obrazów przed przerzedzeniem. Licencja danych nie jest podana. Etap 2b pokazał, że szkodzi (sekcja 4).

**Zablokowane (decyzja właściciela):** SoccerDB (Google Drive zwraca 404, zostaje Baidu Pan) i SoccerTrack v2 (bramka na Hugging Face dla konta `redhot3`).

**Maskowanie klas bez oznaczeń.** Nie każde źródło oznacza bramkarzy, sędziów i piłkę. Dla każdego obrazu klasy spoza jego listy oznaczonych klas nie dostają lossu klasyfikacji, ani dodatniego, ani ujemnego. Loss ramki zostaje bez zmian. Partnerzy w mozaice mają tę samą maskę. W widoku run 1 maskowaną klasę ma 22.5%<!--n:mask.any--> obrazów:

| klasa | bramkarz | sędzia | piłka | zawodnik |
|---|---|---|---|---|
| obrazy z maską | 22.5%<!--n:mask.gk--> | 11.0%<!--n:mask.ref--> | 3.0%<!--n:mask.ball--> | 0.7%<!--n:mask.player--> |

Bez maski model uczyłby się, że bramkarz z TeamTrack czy SportsMOT to tło.

**Higiena.** Mecze benchmarku (GSR valid i test) są poza każdym widokiem treningowym. Kontrola duplikatów (dHash i korelacja miniatur) wobec GSR valid, GSR test i `football_yolo` valid nie dała trafień. W widoku są natomiast mecze ewaluacyjne i bramkowe xG z SoccerNet-v3. Skutek opisuje sekcja 7.

## 4. Drabina treningów (wyniki rozwojowe, ticket 04)

Wybór szedł tylko na danych rozwojowych: GSR valid, Ekstraklasa dev (sekcja 6) i umieszczenie piłki na GSR valid. Ultralytics zapisuje `epochN.pt` po epoce N+1.

| szczebel | start | dane | epoki | czas | czego nauczył |
|---|---|---|---|---|---|
| run 1 | COCO | 99 478<!--n:v1.images--> obrazów z 8 źródeł | 20<!--n:r1.epochs--> (wczesny stop, najlepsza 13<!--n:r1.best-->) | 29.6 h<!--n:r1.hours--> | Duże dane i 1280 px dają role i piłkę. Ekstraklasy nie ma w danych. |
| 2a | run 1 | tylko `football_yolo`, 736<!--n:v2a.images--> obrazów | 35<!--n:r2a.epochs--> | 36 min<!--n:r2a.min--> | Ekstraklasa dogania w kilka epok, ale model zapomina GSR. |
| 2b | 2a, epoka 2 | tylko SoccerSynth, 50 877<!--n:src.synth.img--> obrazów | 5<!--n:r2b.epochs--> | 3.8 h<!--n:r2b.hours--> | Same rendery niszczą role i piłkę. |
| 2c | run 1 | dane run 1 + `football_yolo` ×20<!--n:v2c.over-->, 114 198<!--n:v2c.per_epoch--> obrazów na epokę | 3<!--n:r2c.epochs--> | 5.1 h<!--n:r2c.hours--> | Mieszanka trzyma GSR i Ekstraklasę. |

W run 2c `football_yolo` to 14 720<!--n:v2c.fy.per_epoch--> obrazów każdej epoki.

**Walidacja w trakcie treningu** (co czwarta klatka GSR valid; w etapach 2 także Ekstraklasa dev), mAP50-95:
- run 1: od 0.470<!--n:r1.val.first--> po epoce 1 do najwyżej 0.487<!--n:r1.val.max-->. Wynik płaski od pierwszej epoki, więc ograniczają dane, nie długość treningu.
- 2a: 0.394<!--n:r2a.val.first--> po epoce 1, najwyżej 0.398<!--n:r2a.val.max-->, na końcu 0.314<!--n:r2a.val.last-->. Zapominanie GSR od pierwszej epoki.
- 2b: od 0.336<!--n:r2b.val.first--> do 0.264<!--n:r2b.val.last-->, spadek z każdą epoką.
- 2c: 0.476<!--n:r2c.val.e1-->, 0.472<!--n:r2c.val.e2-->, 0.465<!--n:r2c.val.e3--> po kolejnych epokach.

**Porównanie na pełnym GSR valid, Ekstraklasa dev i umieszczeniu piłki** (wejście 1280):

| model | GSR mAP50 / 50-95 | GSR piłka AP50 | GSR bramkarz AP50 | Ekstraklasa mAP50 / 50-95 | Ekstraklasa piłka AP50 | piłka umieszczona | mediana błędu | > 5 m |
|---|---|---|---|---|---|---|---|---|
| stary detektor | 0.228 / 0.100<!--n:v.b.1280.map--> | 0.141<!--n:v.b.1280.ball--> | 0.000<!--n:v.b.1280.gk--> | 0.756 / 0.331<!--n:e.b.map--> (zawyżone) | 0.502<!--n:e.b.ball--> | 50.2%<!--n:vp.b.placed--> | 1.42 m<!--n:vp.b.med--> | 25.1%<!--n:vp.b.far--> |
| run 1, najlepsza epoka | **0.760 / 0.465<!--n:v.r1.map-->** | **0.380<!--n:v.r1.ball-->** | 0.884<!--n:v.r1.gk--> | 0.412 / 0.145<!--n:e.r1.map--> | 0.078<!--n:e.r1.ball--> | 69.0%<!--n:vp.r1.placed--> | 1.29 m<!--n:vp.r1.med--> | 21.5%<!--n:vp.r1.far--> |
| 2a `epoch1.pt` | 0.543 / 0.312<!--n:v.2a1.map--> | 0.367<!--n:v.2a1.ball--> | 0.584<!--n:v.2a1.gk--> | 0.670 / 0.350<!--n:e.2a1.map--> | 0.161<!--n:e.2a1.ball--> | | | |
| 2a `epoch11.pt` | 0.389 / 0.201<!--n:v.2a11.map--> | 0.326<!--n:v.2a11.ball--> | 0.178<!--n:v.2a11.gk--> | 0.739 / 0.454<!--n:e.2a11.map--> | 0.346<!--n:e.2a11.ball--> | 57.9%<!--n:vp.2a11.placed--> | 1.08 m<!--n:vp.2a11.med--> | 7.6%<!--n:vp.2a11.far--> |
| 2c `epoch0.pt` | 0.741 / 0.450<!--n:v.2c0.map--> | 0.376<!--n:v.2c0.ball--> | 0.841<!--n:v.2c0.gk--> | 0.722 / 0.429<!--n:e.2c0.map--> | 0.245<!--n:e.2c0.ball--> | 72.2%<!--n:vp.2c0.placed--> | 1.13 m<!--n:vp.2c0.med--> | 14.1%<!--n:vp.2c0.far--> |
| **2c `epoch1.pt` = dd2c** | 0.743 / 0.450<!--n:v.2c1.map--> | 0.356<!--n:v.2c1.ball--> | 0.869<!--n:v.2c1.gk--> | **0.721 / 0.424<!--n:e.2c1.map-->** | 0.279<!--n:e.2c1.ball--> | 72.5%<!--n:vp.2c1.placed--> | 1.12 m<!--n:vp.2c1.med--> | 14.6%<!--n:vp.2c1.far--> |
| 2c `epoch2.pt` | 0.722 / 0.440<!--n:v.2c2.map--> | 0.304<!--n:v.2c2.ball--> | 0.870<!--n:v.2c2.gk--> | 0.712 / 0.411<!--n:e.2c2.map--> | 0.260<!--n:e.2c2.ball--> | 77.5%<!--n:vp.2c2.placed--> | 1.28 m<!--n:vp.2c2.med--> | 22.3%<!--n:vp.2c2.far--> |

Stary detektor uczył się na klipach Ekstraklasa dev, więc jego wynik tam jest zawyżony i nieporównywalny.

**Run 1: duże dane i 1280 px.**
- Bramkarz z 0.000<!--n:v.b.1280.gk--> do 0.884<!--n:v.r1.gk--> AP50, sędzia z 0.012<!--n:v.b.1280.ref--> do 0.815<!--n:v.r1.ref-->.
- Piłka z 0.141<!--n:v.b.1280.ball--> do 0.380<!--n:v.r1.ball-->. Rozdzielczość jest warunkiem: ten sam model przy 640 ma piłkę 0.112<!--n:v.r1.640.ball-->.
- Na Ekstraklasie tylko 0.412<!--n:e.r1.map50--> mAP50, a piłka 0.078<!--n:e.r1.ball-->. Tej domeny nie było w danych.

**2a: samo `football_yolo`.** Ekstraklasa rośnie szybko, ale GSR spada. Już `epoch1.pt` ma na GSR sędziego 0.285<!--n:v.2a1.ref--> (run 1: 0.815<!--n:v.r1.ref-->). Samo dotrenowanie na małym zbiorze odpada.

**2b: sam SoccerSynth.** Start: najlepsze wagi 2a (`epoch1.pt`). Ekstraklasa dev, AP50 przy 1280:

| klasa | start (2a) | 2b po epoce 1 | 2b po epoce 5 |
|---|---|---|---|
| piłka | 0.161<!--n:e.2a1.ball--> | 0.004<!--n:e.2b0.ball--> | 0.0003<!--n:e.2b4.ball--> |
| bramkarz | 0.864<!--n:e.2a1.gk--> | 0.317<!--n:e.2b0.gk--> | 0.005<!--n:e.2b4.gk--> |
| sędzia | 0.778<!--n:e.2a1.ref--> | 0.207<!--n:e.2b0.ref--> | 0.001<!--n:e.2b4.ref--> |
| zawodnik | 0.878<!--n:e.2a1.player--> | 0.765<!--n:e.2b0.player--> | 0.554<!--n:e.2b4.player--> |
| mAP50 | 0.670<!--n:e.2a1.map50--> | 0.323<!--n:e.2b0.map50--> | 0.140<!--n:e.2b4.map50--> |

Kanały bramkarza i sędziego są w SoccerSynth maskowane, więc nie dostają gradientu. Zawodnik i piłka uczą się na renderowanych ludziach. Na prawdziwych klatkach piłka, bramkarz i sędzia spadają prawie do zera. Nie sprawdzono SoccerSynth jako małego dodatku do prawdziwych danych.

**2c: mieszanka.** Dane run 1 plus `football_yolo` powtórzone 20<!--n:v2c.over--> razy, start z run 1.
- GSR zostaje blisko run 1: 0.743<!--n:v.2c1.map50--> wobec 0.760<!--n:v.r1.map50--> mAP50.
- Ekstraklasa jak przy 2a: 0.721<!--n:e.2c1.map50--> mAP50.
- Piłka na GSR spada z każdą epoką: 0.376<!--n:v.2c0.ball-->, 0.356<!--n:v.2c1.ball-->, 0.304<!--n:v.2c2.ball-->.

**Wybór.** Kandydat to `epoch1.pt` z run 2c. Ma najlepszy profil błędu umieszczenia: 72.5%<!--n:vp.2c1.placed--> umieszczonych i 14.6%<!--n:vp.2c1.far--> dalej niż 5 m. `epoch2.pt` umieszcza więcej (77.5%<!--n:vp.2c2.placed-->), ale 22.3%<!--n:vp.2c2.far--> z tego to pomyłki. Wybór zapisano w tickecie 05 przed odczytem testu.

## 5. Wynik testu na GSR test (ticket 05)

Kandydat i stary detektor, każdy raz. Żaden inny model ani ustawienie nie dotknęło GSR test. Plik starego detektora zapisano w tickecie 02 i przeczytano dopiero tu.

### 5.1 Detekcja (wynik testu)

| wejście | model | piłka | bramkarz | zawodnik | sędzia | mAP50 / 50-95 | piłka P / R |
|---|---|---|---|---|---|---|---|
| 640 | stary | 0.111<!--n:t.b.640.ball--> | 0.000<!--n:t.b.640.gk--> | 0.774<!--n:t.b.640.player--> | 0.000<!--n:t.b.640.ref--> | 0.221 / 0.098<!--n:t.b.640.map--> | 0.38 / 0.17<!--n:t.b.640.ballpr--> |
| 640 | dd2c | 0.215<!--n:t.c.640.ball--> | 0.636<!--n:t.c.640.gk--> | 0.939<!--n:t.c.640.player--> | 0.778<!--n:t.c.640.ref--> | 0.642 / 0.378<!--n:t.c.640.map--> | 0.55 / 0.24<!--n:t.c.640.ballpr--> |
| 1280 | stary | 0.183<!--n:t.b.1280.ball--> | 0.000<!--n:t.b.1280.gk--> | 0.733<!--n:t.b.1280.player--> | 0.000<!--n:t.b.1280.ref--> | 0.229 / 0.095<!--n:t.b.1280.map--> | 0.32 / 0.31<!--n:t.b.1280.ballpr--> |
| 1280 | **dd2c** | **0.464<!--n:t.c.1280.ball-->** | **0.666<!--n:t.c.1280.gk-->** | **0.953<!--n:t.c.1280.player-->** | **0.905<!--n:t.c.1280.ref-->** | **0.747 / 0.451<!--n:t.c.1280.map-->** | 0.58 / 0.52<!--n:t.c.1280.ballpr--> |
| e3 | stary | 0.164<!--n:t.b.e3.ball--> | 0.000<!--n:t.b.e3.gk--> | 0.636<!--n:t.b.e3.player--> | 0.004<!--n:t.b.e3.ref--> | 0.201 / 0.079<!--n:t.b.e3.map--> | 0.28 / 0.29<!--n:t.b.e3.ballpr--> |
| e3 | dd2c | 0.493<!--n:t.c.e3.ball--> | 0.687<!--n:t.c.e3.gk--> | 0.947<!--n:t.c.e3.player--> | 0.926<!--n:t.c.e3.ref--> | 0.763 / 0.450<!--n:t.c.e3.map--> | 0.61 / 0.55<!--n:t.c.e3.ballpr--> |

Kolumny klas to AP50. P / R to precyzja i recall w punkcie pracy.

**Bramkarz.** Przy 1280 recall bramkarza to 0.64<!--n:t.c.1280.gkr-->, a precyzja 0.94<!--n:t.c.1280.gkp-->. Model rzadko wskazuje złego bramkarza, ale co trzeciego nie znajduje jako bramkarza.

### 5.2 Umieszczenie piłki na boisku (wynik testu)

Homografie testu pochodzą z jedynego odczytu GSR test modelu punktów boiska z fazy pitch-keypoints. Zamrożono je jako tabelę 6 209<!--n:tp.homographies--> klatek. Ocenianych klatek z piłką na boisku jest 5 277<!--n:tp.frames-->. Przedziały 95%: bootstrap po klipach.

| | umieszczona | mediana błędu | < 1 m | < 2 m | > 5 m |
|---|---|---|---|---|---|
| stary detektor | 63.6% [57.4; 69.0]<!--n:tp.b.placed--> | 1.02 m [0.90; 1.14]<!--n:tp.b.med--> | 49.2%<!--n:tp.b.lt1--> | 69.4%<!--n:tp.b.lt2--> | 19.9% [15.8; 24.6]<!--n:tp.b.far--> |
| **dd2c** | **77.9% [72.7; 82.4]<!--n:tp.c.placed-->** | **0.88 m [0.78; 0.99]<!--n:tp.c.med-->** | 55.3%<!--n:tp.c.lt1--> | 80.9%<!--n:tp.c.lt2--> | **7.4% [5.8; 9.2]<!--n:tp.c.far-->** |
| różnica sparowana, dd2c − stary | +14.4 pp [+10.3; +18.5]<!--n:tp.d.placed--> | −0.14 m [−0.23; −0.06]<!--n:tp.d.med--> | | | −12.5 pp [−16.2; −9.2]<!--n:tp.d.far--> |

Wszystkie trzy przedziały różnic omijają zero.

**Przy bramce** (2 714<!--n:tp.ng.frames--> klatek, osoby w medianie najwyżej 30 m od linii bramkowej): piłka umieszczona na 66.5%<!--n:tp.b.ng.placed--> → 83.1%<!--n:tp.c.ng.placed--> klatek, a dalej niż 5 m leży 20.1%<!--n:tp.b.ng.far--> → 5.9%<!--n:tp.c.ng.far--> umieszczonych.

Licząc od wszystkich ocenianych klatek, trafna pozycja piłki (do 5 m) jest teraz w około trzech czwartych klatek. Stary detektor miał ją w około połowie.

## 6. Zbiory kontrolne (wyniki rozwojowe)

**Ekstraklasa dev.** Trzy całe klipy LiveTag z meczów Lecha, wyjęte z `football_yolo` przed run 2: 56<!--n:ek.images--> obrazów, 46<!--n:ek.balls--> piłek. Dla dd2c to dane nieznane, dla starego detektora dane treningowe.

| wejście 1280, AP50 | piłka | bramkarz | zawodnik | sędzia | mAP50 / 50-95 |
|---|---|---|---|---|---|
| stary (zawyżony) | 0.502<!--n:e.b.ball--> | 0.882<!--n:e.b.gk--> | 0.925<!--n:e.b.player--> | 0.717<!--n:e.b.ref--> | 0.756 / 0.331<!--n:e.b.map--> |
| dd2c | 0.279<!--n:e.2c1.ball--> | 0.845<!--n:e.2c1.gk--> | 0.909<!--n:e.2c1.player--> | 0.853<!--n:e.2c1.ref--> | 0.721 / 0.424<!--n:e.2c1.map--> |

dd2c ma wyższe mAP50-95, ale piłkę wyraźnie gorszą niż stary model na jego własnych klipach.

**`football_yolo` valid** (193<!--n:fy.valid.images--> obrazów, dla obu modeli w dużej części dane treningowe). Eksport Roboflow rozciąga klatki 16:9 do kwadratu. Wiersz 1280 podaje kwadrat, wiersz e3 klatkę przywróconą do 16:9.

| | stary | dd2c |
|---|---|---|
| wejście 1280, kwadrat: mAP50 / 50-95 | 0.752 / 0.402<!--n:fy.b.1280--> | 0.324 / 0.188<!--n:fy.c.1280--> |
| e3, 16:9: mAP50 / 50-95 | 0.654 / 0.251<!--n:fy.b.e3--> | **0.798 / 0.431<!--n:fy.c.e3-->** |

dd2c nauczył się klatek 16:9. Wideo w pipeline nigdy nie jest rozciągane, więc liczy się wiersz e3.

## 7. Skutek dalej: miejsce strzału w xG (wyniki rozwojowe; Data: StatsBomb.)

Faza xg-detector powtórzyła audyt z dd2c (audyt 4 na gałęzi `research/xg-detector`, 2026-10-18). Konfiguracja E3 jest ta sama co w audycie 3: model punktów boiska z fazy pitch-keypoints i piłka z klatki −400 ms. Zmienił się tylko detektor. Populacja: 2 721<!--n:a4.shots--> sparowanych strzałów bez karnych z 135<!--n:a4.matches--> meczów. Prawda to pozycje StatsBomb. Data: StatsBomb.

| | audyt 3 (stary detektor) | audyt 4 (dd2c) |
|---|---|---|
| pokrycie miejsca strzału | 99.5%<!--n:a3.cov--> | 99.5%<!--n:a4.cov--> |
| mediana błędu miejsca strzału | 5.18 m [4.68; 5.85]<!--n:a3.med--> | **2.62 m [2.45; 2.78]<!--n:a4.med-->** |
| strzały z błędem poniżej 5 m | 49.3%<!--n:a3.lt5--> | 68.2%<!--n:a4.lt5--> |
| bramkarz wykryty jako klasa | 18.3%<!--n:a3.gkdet--> | 90.6%<!--n:a4.gkdet--> |
| mediana błędu pozycji bramkarza | 6.50 m<!--n:a3.gkmed--> | 0.77 m<!--n:a4.gkmed--> |

Data: StatsBomb.

- **Bramka audytu przechodzi.** Wymaga mediany najwyżej 3.0 m<!--n:a4.gate-->. Audyt 3 dał STOP, audyt 4 PASS.
- **Dalszy krok poza detektorem.** Reguła `feet_t0` bierze miejsce strzału ze stóp strzelca, nie z rzutu piłki. Obniżyła medianę do 1.16 m<!--n:sr.feet.med-->. Opisuje ją raport `results_reports/sweep_2026_10/REPORT.md` na `research/xg-detector`, razem ze skutkiem dla log-loss xG. Data: StatsBomb.
- **To nie jest niezależny test.** dd2c uczył się na klatkach SoccerNet-v3 z meczów xG, czasem z samych strzałów. Na 172 meczach odłożonych klatkę w treningu ma 66.8%<!--n:sr.seen.goals--> goli i 40.9%<!--n:sr.seen.other--> pozostałych strzałów. Kontrola na strzałach niewidzianych jest w `results/xgs_stack_dd2c_leak` na tamtej gałęzi.

## 8. Ograniczenia

### 8.1 Piłka na Ekstraklasie (wyniki rozwojowe)

AP50 piłki dd2c na Ekstraklasa dev to 0.279<!--n:e.2c1.ball-->. To najsłabszy punkt modelu w domenie produktu. Pomiar opiera się na 46<!--n:ek.balls--> piłkach z 3 klipów, więc jest bardzo niepewny. Stary detektor ma tu 0.502<!--n:e.b.ball-->, ale na własnych danych treningowych.

### 8.2 Bramkarz na teście (wynik testu)

Na GSR test AP50 bramkarza to 0.666<!--n:t.c.1280.gk-->, a recall 0.64<!--n:t.c.1280.gkr-->. Na GSR valid bramkarz wypadł wyraźnie lepiej (sekcja 4). Test ma tylko 3 mecze. Przyczyny nie badano, żeby nie czytać testu drugi raz.

### 8.3 Licencje: wagi tylko do badań

W treningu są dane pod NDA SoccerNet (GSR, v3), niekomercyjne (SportsMOT, CC BY-NC) i bez licencji albo z nieznaną (Soccana, ISSIA, `football_yolo`). Wagi `dd2c` nadają się więc tylko do badań. Wariant z czystą licencją wymaga nowego treningu bez tych źródeł. Pula ma licencję przy każdym obrazie, więc taki widok da się zbudować filtrem.

### 8.4 Brak ablacji źródeł

Nie wiadomo, które źródło dało najwięcej. W run 1 zmieniły się naraz dane, rozdzielczość wejścia i model (11m → 11l). Najbardziej prawdopodobne źródła zysku to klatki GSR train (ta sama domena co benchmark) i SoccerNet-v3, ale to hipoteza. SoccerSynth sprawdzono tylko sam, nie jako mały dodatek.

### 8.5 Inne

- GSR to jedna liga z 2019 roku i po 3 mecze w valid i test.
- Bramki `pytest -m model` i `examples/06_cli.sh` nie działały w tym worktree (brak `data/` i części `models/`). Notebooki i przykłady `03`, `04` i `06` celowo używają jeszcze starych wag.
- Kod fazy jest tylko na `research/detector-data`, nie na `master`.

## 9. Co przeszło do ball-detection

Szczebel „piłka” z ticketu 04 jest od 2026-10-22 osobną linią: `.scratch/ball-detection/` (gałąź `research/ball-detection`). GSR test zostaje zużyty dla dd2c. Nowe warianty porównuje się na GSR valid, Ekstraklasa dev i umieszczeniu piłki z zamrożonymi homografiami. Tickety:
1. analiza błędów piłki dd2c na GSR valid;
2. karta wyników piłki i reguła wyboru, zamrożone przed pierwszym wariantem;
3. piłka bez treningu: rozdzielczość wejścia, kafle, TTA;
4. ile pudeł odzyskuje śledzenie piłki w czasie (pomiar offline);
5. dotrenowanie nastawione na piłkę, z ramieniem kontrolnym;
6. SoccerSynth jako mały udział w mieszanym widoku;
7. ablacja źródeł dla piłki.

## 10. Otwarte dla właściciela

1. **Ticket 04** ma nadal status „in-progress”. Wszystkie szczeble poza piłką są zrobione, a piłka przeszła do `ball-detection`. Czy zamknąć ticket?
2. **Zablokowane źródła.** SoccerDB (Baidu Pan) i SoccerTrack v2 (bramka Hugging Face) czekają na działanie właściciela.
3. **Merge do `master`.** Kod fazy (maskowany loss, benchmark, konwertery) i przełączenie domyślnego detektora czekają na decyzję.
4. **Wariant z czystą licencją:** czy jest potrzebny teraz.
5. **Triage `ball-detection`:** spec ma status „needs-triage”.

## 11. Rozbieżności między notatkami a plikami

### 11.1 Ticket 05 (wynik testu)

- Tabela ticketu 05 podaje dla dd2c przy 1280 piłkę P / R `0.59 / 0.53`. Plik daje 0.58 / 0.52<!--n:t.c.1280.ballpr-->. Tę samą parę z ticketu skopiował spec `ball-detection`. Notatki z wątku mają wartość z pliku.

### 11.2 Ticket 04 (wyniki rozwojowe)

- W tabeli porównawczej ticketu 04 kolumny Ekstraklasy dla 2a są przesunięte o jedną epokę. Przy `epoch1.pt` stoi wynik `epoch2.pt`, a przy `epoch11.pt` wynik `epoch12.pt`. Poprawne wartości to 0.670 / 0.350<!--n:e.2a1.map--> i 0.739 / 0.454<!--n:e.2a11.map--> (tabela w sekcji 4). Kolumny GSR i umieszczenia są dobre.
- Czas run 2c: ticket podaje `5 h 08`, a `results.csv` daje 5.1 h<!--n:r2c.hours-->. Różnica to zapewne czas poza epokami.

### 11.3 Liczby bez pliku wyników

- Próbka pomyłek ról starego detektora z ticketu 02 (`103 z 135` bramkarzy i `240 z 319` sędziów z ramką „player”) nie ma pliku wyników. Raport podaje ją bez liczb.
- Mediana rozmiaru piłki (`14 px` przy 1080p) z notatek ticketu 04 też nie ma pliku wyników. Raport podaje ją bez liczby.

## Źródła i sumy kontrolne

| plik | SHA-256 | zawartość |
|---|---|---|
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/sweep_report/location_error.csv` | `e6944e8c40e47ea97badd6dd85d813e64af600da65be70d70f6da15d8add7213` | błąd miejsca strzału dd2c i `feet_t0` (xg-detector; Data: StatsBomb.) |
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/sweep_report/summary.json` | `6b1cab3013c1a45dce5566d091f2f3cb1812989abb084bb7fe830130fd4b287d` | udział strzałów z klatką w treningu dd2c (xg-detector) |
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/xgd_audit3/metrics.csv` | `38d2d5b4d67761d4a80487b76e8b5fa9ff0a2ac15cafcdb50ef2523f8ace6d93` | audyt 3, stary detektor (xg-detector; Data: StatsBomb.) |
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/xgd_audit4/choice.json` | `db8436d8cd67d7b57236496b993e1f6f04e9f64f4fc02c641747d6cd1ef29d17` | audyt 4: bramka 3 m (xg-detector) |
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/xgd_audit4/complete.json` | `0a3597e069f2b8275e66341cb8025339b048de1c0a469be02c26d18f687110b8` | audyt 4: populacja strzałów (xg-detector) |
| `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable/results/xgd_audit4/metrics.csv` | `bf24144babc80896f6dff8197c74e44649bae5ef4f16f605bf5e098ead65e91c` | audyt 4, dd2c (xg-detector; Data: StatsBomb.) |
| `research/detector_bench.json` | `114b0dc68585b723795b111bcf920d082f6992504827071d743f3253a4d288e1` | zamrożony benchmark, punkt pracy, homografie testu (tickety 02, 05) |
| `research/detector_run1_view.json` | `7a39d4b1e466f2572017a457c8fe40088504261a265179080e4ff5b0636678d3` | widok run 1: obrazy, maski klas, pula (ticket 04) |
| `research/detector_run2a_view.json` | `1ebdcecec5c08d617245619c8c5c2408e76baab9b5fafc55c93110c6f6fd7171` | widok 2a (ticket 04) |
| `research/detector_run2b_view.json` | `d303923c21a1a28b6f9527f38c1d72018993635e79182a6350af65c34c60a347` | widok 2b, SoccerSynth (ticket 04) |
| `research/detector_run2c_view.json` | `05fd89bbbd34b620d992c0f7d59d12913df6f6aa30402120bd69726c5ea22c82` | widok 2c, nadpróbkowanie `football_yolo` (ticket 04) |
| `research/ekstraklasa_dev_split.json` | `95dbd5549892ac6c10aca21e56b490efab70f95f170386d02a01e72a0338cbb5` | Ekstraklasa dev: obrazy i piłki (ticket 04) |
| `/home/kuba/projects/tactifoot_vision/data/datasets/football_yolo/train/images` | `b6e6a94daf2f91cfaf45d0f33f9520b59ca0cb08ec8f4d90c086ec52a2fc44f9` | obrazy treningowe starego detektora (katalog: skrót listy nazw) |
| `/home/kuba/projects/tactifoot_vision/data/datasets/football_yolo/valid/images` | `59c1858a5abbe13be860c324b613388363cc79cdd18ffc30d8e9e32c88a90318` | obrazy walidacyjne starego detektora (katalog: skrót listy nazw) |
| `/home/kuba/projects/tactifoot_vision/models/detector_yolo11l_1280_dd2c.pt.json` | `46e0d4532b605916ded59ed9f4f67911e3746f2445c6cfa372736eccf6e9d959` | karta modelu dd2c: źródła, obrazy, ramki, licencje, przepis (ticket 05) |
| `/home/kuba/projects/tactifoot_vision/results/project/raw/yolo11m/training/detect_yolo11m/args.yaml` | `22f60bcd840aef93ef9519dd8d90d76105ba53476d65c752ef7dd027f6657473` | ustawienia treningu starego detektora |
| `/mnt/storage/tactifoot_research/detector_data/baseline/detection_valid.json` | `78180694403c47fc99bc72fd5685356ab083727abc01b9857659640172cbb95e` | stary detektor, GSR valid i `football_yolo` valid (ticket 02) |
| `/mnt/storage/tactifoot_research/detector_data/baseline/ekstraklasa_dev/detection_valid.json` | `762a6ba9e6d6bbb2026a5d05b570f61d14f16e148f93cfe111f95f7c6b75b6db` | stary detektor, Ekstraklasa dev (ticket 04) |
| `/mnt/storage/tactifoot_research/detector_data/baseline/placement/placement.json` | `a8d848f7a7317c6e9f257ec6a6aa384253d714263c036a81a24cb63833107907` | stary detektor, umieszczenie piłki na GSR valid (ticket 02) |
| `/mnt/storage/tactifoot_research/detector_data/baseline/test/detection_test.json` | `3fb45dee876c8d7e0a898940d91787eb01c7754c1498213155327eb5c882ab1f` | stary detektor, GSR test (zapisany w tickecie 02, przeczytany w tickecie 05) |
| `/mnt/storage/tactifoot_research/detector_data/final/baseline_placement_test/placement.json` | `ca8d2bf5d45f9850b16a22c1f00560c282b86413ebc6a946826a60eba3e395bc` | stary detektor, umieszczenie piłki na GSR test (ticket 05) |
| `/mnt/storage/tactifoot_research/detector_data/final/candidate_placement_test/placement.json` | `ecabfd176d73e511e3317b0b53d032749540b4b55940b302bc59639e3ec934db` | dd2c, umieszczenie piłki na GSR test (ticket 05) |
| `/mnt/storage/tactifoot_research/detector_data/final/candidate_test/detection_test.json` | `c603803591a33d79d0565dda0cec58936b5da1a6443d32ebf9725b9b0c0d3ff6` | dd2c, GSR test (ticket 05) |
| `/mnt/storage/tactifoot_research/detector_data/final/candidate_valid/detection_valid.json` | `12739b78d71552d46d04b0d7574c2cc4fb2dc64abb5677b169a12fdf119a3069` | dd2c, GSR valid, `football_yolo` valid, Ekstraklasa dev (ticket 05) |
| `/mnt/storage/tactifoot_research/detector_data/final/placement_test_paired.txt` | `93ad7239d047d814dfaa3b09d9d68a1551436f7f55a5627d0e5e396ae0f02a37` | różnice sparowane umieszczenia na GSR test (ticket 05) |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/placement_run2a_epoch11/placement.json` | `47b1eafb21e4d8cf81e97522d71afb839245015c4d140ae49e489ddc93597d51` | 2a `epoch11.pt`, umieszczenie na GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/placement_run2c_epoch0/placement.json` | `c1314119a7f6c5a050cf1033eee23771ec38481592ba95b38b63cceb654cbc8e` | 2c `epoch0.pt`, umieszczenie na GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/placement_run2c_epoch1/placement.json` | `89c9a5bf915cbaa49c850918f2e8f2216e4c79c4ad227080c3f7388707b3e7a3` | 2c `epoch1.pt` (dd2c), umieszczenie na GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/placement_run2c_epoch2/placement.json` | `92482098a6959122efa11cceef4e4332ad23eb0cc362a4e8feb41b0ae2aa1e4c` | 2c `epoch2.pt`, umieszczenie na GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/run2a_epoch1/detection_valid.json` | `d01604e156990831893822bce1ce93ec10d4889a1eb3e70fda89966b89671783` | 2a `epoch1.pt`, GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/compare/run2a_epoch11/detection_valid.json` | `58ef6756bab3f0db1b7bfaf6057224b81bed62ed6dbb8378704c17849f522d47` | 2a `epoch11.pt`, GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/run1/eval/detection_valid.json` | `9959dda21a3d41f9503b6001c55d8b1a2b975460b7c2201689641cced1f04e96` | run 1, GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/run1/eval/ekstraklasa_dev/detection_valid.json` | `9286a1406d09ad95acfef2a163022b7cb9cf7cc8070e51ca7acc920ba051dacd` | run 1, Ekstraklasa dev |
| `/mnt/storage/tactifoot_research/detector_data/runs/run1/eval/placement/placement.json` | `a93acd608978367c79ee5b273e22e6532604a3fb7324d9ac0ba4a0a6d5514b0f` | run 1, umieszczenie na GSR valid |
| `/mnt/storage/tactifoot_research/detector_data/runs/run1/yolo11l_1280/results.csv` | `c3dfb78ccd2ac3e4ec1202344ad6ea940d137c481fe58f058398ef0634656dda` | run 1: epoki, czas, walidacja w trakcie treningu |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2a/eval_epochs_ekstraklasa/epochs.csv` | `a700c83a584daf860a985a7788f689e583316b16866c32264c6e8a7b3e32893e` | 2a: Ekstraklasa dev dla każdej epoki |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2a/yolo11l_1280/results.csv` | `fd9ed85c5b544d589bd41b9a40b3d5fee5774dd69429cc3fb0a616c937a4e7fd` | 2a: epoki, czas, walidacja |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2b/eval_epochs_ekstraklasa/epochs.csv` | `02907a1e3b5c5f04b2b425a7d067145ccfa80a0f4fcbf15f1a47685163fa82a0` | 2b: Ekstraklasa dev dla każdej epoki |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2b/yolo11l_1280/results.csv` | `3558e1c5974dc130bfae5cdb218ed8d2bde6e8002c732231a98f92713c1b343b` | 2b: epoki, czas, walidacja |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2c/eval_epochs/epochs.csv` | `6ff454e83315e556d162e0c0a1384268e2802938eba4059530a1a0d4aa623793` | 2c: GSR valid i Ekstraklasa dev dla każdej epoki |
| `/mnt/storage/tactifoot_research/detector_data/runs/run2c/yolo11l_1280/results.csv` | `a2931c7d8ded6b614f80bff8504764b0d00f9ecb2e268007604bc6d5120c978a` | 2c: epoki, czas, walidacja |
| `/mnt/storage/tactifoot_research/detector_data/sets/soccersynth/manifest.json` | `7381bfd9118f08d49addb4d7593ab159ce9660682615b12e59391a01bababe2b` | SoccerSynth: obrazy przed przerzedzeniem, rozmycie ruchu |
