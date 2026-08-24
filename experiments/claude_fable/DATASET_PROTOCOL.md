# Protokół budowy pełnego datasetu xG z obrazu

Wersja: 1.0 (2026-08-24). Poprzedzają go: `research/dataset_plan.md` (inwentarz,
uzasadnienia) i `README.md` (kontrakt indexu). Wszystkie komendy z
`experiments/claude_fable/` w worktree `claude-fable-xg-poc`.

## 0. Dostępy (zweryfikowane 2026-08-24)

- SoccerNet NDA: hasło w `.env` (`SOCCERNET_PASSWORD`), poza gitem. Działa dla 224p
  i 720p (test: pobrane 195 MB / 975 MB połówki).
- HF token w `.env` (`HF_TOKEN`). Uwaga: `atomscott/soccertrack-v2` NIE istnieje
  jeszcze na HF (404 także w wyszukiwarce) — jedyna droga to mirror Google Drive
  z manifestem na branchu codex (`fetch_soccertrack.py`, folder `1N2Qx...`).
- StatsBomb open data: bez klucza (raw.githubusercontent, commit przypięty).

## 1. Definicja rekordu (zamrożona)

Target: `goal ∈ {0,1}` dla pojedynczej próby strzału; wejście: klatka z chwili
oddania strzału. Kotwica klatki pochodzi wyłącznie z anotacji strzału
(`Shots on/off target`, BAS `SHOT`/`HEADER`/`FREE KICK`). `Goal` służy tylko jako
dowód wyniku (parowanie: ta sama drużyna, ta sama połowa, okno 6 s — na 500 meczach
89.6% goli ma strzał ≤15 s wcześniej, mediana odstępu 0.6 s, p90 1.1 s). Zakazane:
syntetyczna kotwica `goal − offset`, klatki po wyniku jako wejście, `not shown`.
Karne z flagą `is_penalty` (raport główny i bez karnych). Każdy event źródłowy jest
rozliczony: trafia do indexu (`status=ok`) albo do `attempts_all.parquet` z reason
code. Zmiana tej definicji = nowa wersja datasetu.

## 2. Faza 1: SoccerNet 720p (główny wolumen)

Narzędzie: `scripts/collect_soccernet.py` (przetestowane pilotem na 2 meczach).
Działanie na mecz: pobierz połówkę 720p → wytnij JPEG w offsetach
(-1600,-1200,-800,-400,0,+400,+800) ms dla każdego zaufanego strzału → (opcjonalnie
`--clips`) klip 8 s x264 crf26 → zapisz sha256/ffprobe do `raw_assets.jsonl` →
skasuj mkv (chyba że `--keep-video`). Wznawialne per mecz (shard parquet).

Uruchomienie (pełny zbiór, klatki + klipy):

```bash
SOCCERNET_PASSWORD=... nohup ./run.sh python scripts/collect_soccernet.py \
    --out <MIEJSCE_NA_DANE>/soccernet_720p --clips > collect.log 2>&1 &
```

Szacunki (pomiar z pilota: ~11.7 MB/s, 975 MB/połówka):
- transfer: 500 meczów × 2 × ~1 GB ≈ **1 TB pobrania, ~24-40 h** (dysk zajęty
  chwilowo ~2 GB, bo mkv są kasowane po wycięciu);
- wynik: ~11 000 strzałów, ~1 500 goli (13-14%); klatki ≈ **5-8 GB**;
  klipy ≈ **50-70 GB** (opcjonalne, ale rekomendowane — umożliwiają ablacje
  klipowe i ponowne wycinanie klatek bez ponownego 1 TB transferu);
- na koniec: `index.parquet` (tylko `status=ok`) + `attempts_all.parquet` (pełne
  rozliczenie) + `raw_assets.jsonl`.

Bramki odbioru fazy 1:
1. `check_index_quality.py`: 0 meczów wspólnych między splitami (split oficjalny
   SoccerNet: 300/100/100), goal rate 10-16%, `missing frames` = 0.
2. Retencja goli vs nie-goli raportowana per liga/sezon (kontrola selection bias);
   odsetek goli w `goal_without_shot_annotation` ≈ 10% (zmierzone 10.4%).
3. Na cache pozycyjnym (detektor `imgsz=1920`, prog piłki 0.15): `homography_ok`
   ≥ 60%, `has_ball` na t0 ≥ 40% (pilot: 68% / 44%); goal rate rosnący przy malejącym
   `dist_to_goal`.
4. Duplikaty perceptual-hash między splitami: 0 (do dopisania w check-skrypcie).

## 3. Faza 2: SoccerTrack v2 (kohorta z pozycjami GT)

1. Dokończyć pobrania z mirrora Drive (brakuje 11/20 połówek wideo, 2/20 GSR):
   wznowić `scripts/fetch_soccertrack.py` + `download_daemon.py` z brancha codex
   (gdown z backoffem na HTTP 429; pliki lądują w
   `..._codex_xg_video/experiments/codex_xg_video/data/soccertrack_v2/`).
   Monitorować pojawienie się datasetu na HF (`atomscott/soccertrack-v2`) — jeśli
   wyjdzie z gated dostępem, przepiąć się na `snapshot_download` z przypiętą rewizją.
2. Negatywy: jak w sample v0 (`build_sample_soccertrack.py`) — kotwice SHOT/HEADER/FK.
3. Pozytywy wymagają **lokalizatora kontaktu** (BAS nie anotuje strzału przy golu):
   okno [ostatnia akcja drużyny, GOAL] ≤ 6 s → tracking piłki (detektor dostrojony
   do panoramy lub ROI wokół GSR-piłki) → kandydat kontaktu z minimum dystansu
   piłka–stopy + zmiana prędkości + trajektoria do bramki; poziomy `event_exact` /
   `auto_high` / `auto_weak`; do testu tylko dwa pierwsze (protokół za przewodnikiem
   codexa, krok 4). To osobny task (~2-3 dni pracy) — bez niego SoccerTrack
   dostarcza tylko negatywy i `gt_positions` z GSR.
4. GSR czytać strumieniowo (pliki ~2.7 GB); zapis pozycji do `gt_positions` JSON
   per strzał (schemat FreezeFrame).

## 4. Faza 3: StatsBomb 360 (pretraining pozycyjny, bez obrazu)

30 meczów już lokalnie (723 strzały, 88 goli); można rozszerzyć do pełnego open-data
(~3.4k meczów z eventami, częściowo 360). Rola: pretraining / sanity podejść A i B
oraz kalibracja cech; nigdy w jednej tabeli wyników z modelami obrazowymi.

## 5. Struktura docelowa

```
<MIEJSCE_NA_DANE>/
  soccernet_720p/{frames/,clips/,shards/,index.parquet,attempts_all.parquet,raw_assets.jsonl}
  soccertrack_v2/{frames/,gt_positions/,index.parquet,attempts_weak.csv}
  statsbomb/{events/,three-sixty/}
  index.parquet            # merge (scripts/merge_sample_indexes.py na podkatalogach)
  dataset_card.md          # wersja, definicja, liczności, licencje
```

Licencje: SoccerNet — tylko badania (NDA); SoccerTrack — CC BY 4.0; StatsBomb —
niekomercyjne, bez redystrybucji. Surowe wideo i klatki nie wchodzą do gita.

## 6. Kolejność uruchomienia (runbook)

1. `./run.sh pytest -q` (10 testów zielonych).
2. Faza 1 w tle (nohup jak wyżej); postęp: `tail -f collect.log`, wznowienie po
   przerwaniu = ta sama komenda.
3. Po fazie 1: `check_index_quality.py <out>/index.parquet` + cache pozycyjny na
   próbce 500 strzałów + przegląd 30 losowych klatek okiem.
4. Faza 2 pkt 1 (Drive) równolegle z fazą 1 (inne łącze-limity).
5. Merge indexów, `dataset_card.md`, commit metadanych (bez danych).
6. Handoff do treningu: `HANDOFF.md` krok 1 na pełnym indexie.

## 7. Decyzje (potwierdzone 2026-08-24)

1. `<MIEJSCE_NA_DANE>` = `/home/kuba/projects/football_data/xg_shots_full/`.
   Przeniesione tam: `soccernet_labels/` (dawne `data/soccernet_as`, symlink
   zostawiony), `soccernet_224p/`, `samples/` (indeksy z przepisanymi ścieżkami),
   `soccertrack_v2/` i `statsbomb/` (z worktree codexa).
2. `--clips` włączone. Nieużywane dane skompresowane: GSR 43 GB -> 5.5 GB (zstd -8,
   pliki `.json.zst`; czytnik fazy 2 musi dekompresować strumieniowo). `soccernet_224p`
   zostaje bez kompresji (H264 się nie kompresuje); do usunięcia, gdy 720p pokryje te mecze.
3. W każdej chwili >= 5 GB wolnego dysku: kolektor ma strażnika (czeka przy < 8 GB,
   przerywa po godzinie), fetch SoccerTrack i kompresja zatrzymują się przy < 8 GB.
4. Lokalizator kontaktu SoccerTrack: po pierwszym pełnym treningu na SoccerNet.
   Fetch brakujących plików z Drive uruchomiony równolegle.
```
