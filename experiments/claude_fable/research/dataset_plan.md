# Plan danych xG z obrazu (przejęcie po agencie codex)

Data: 2026-08-24. Status: sample v0 zbudowany, skalowanie zablokowane na dostępach.

## Co przejąłem z brancha `codex/xg-video-poc`

Praca codexa była częściowo bardzo wartościowa; przejmuję dane i wnioski z audytu,
odrzucam jego benchmark (sam go unieważnił: pozytywy kotwiczone w timestampie GOAL,
czyli klatka może pokazywać piłkę w bramce).

Lokalny inwentarz (poza gitem):

| Zasób | Ścieżka | Stan |
|---|---|---|
| SoccerNet Labels-v2 | `data/soccernet_as/` | 500 meczów, komplet |
| SoccerNet 224p wideo | `data/soccernet_xg_smoke_224p/` | 19 połówek (~10 meczów), 2.8 GB |
| SoccerTrack v2 BAS | `..._codex_xg_video/experiments/codex_xg_video/data/soccertrack_v2/bas/` | 10 meczów, komplet |
| SoccerTrack v2 wideo pełne | `.../videos/` | 9/20 połówek, 5.6 GB (1920×954 panorama, 25 fps) |
| SoccerTrack v2 GSR | `.../gsr/` | 18/20 połówek, 43 GB (po ~2.7 GB na połówkę) |
| SoccerTrack v2 challenge clips | `.../challenge_videos/` | 10/10, 3.3 GB |
| StatsBomb 360 (30 meczów) | `.../data/statsbomb/` | 723 strzały, 88 goli, bez obrazu |

## Sample v0 (zbudowany dziś)

`data/xg_shots_sample/claude_fable/index.parquet` — 434 strzały, 35 goli, 15 meczów;
klatki w `frames/` (offsety -800..+800 ms). Buildery:
`scripts/build_sample_soccernet.py`, `scripts/build_sample_soccertrack.py`,
`scripts/merge_sample_indexes.py`.

Reguły jakości (za audytem codexa):
- kotwica klatki wyłącznie z anotacji strzału (`Shots on/off target`, BAS `SHOT`/
  `HEADER`/`FREE KICK`); GOAL służy tylko jako dowód wyniku;
- żadnego syntetycznego `goal - 1s` (przeciek targetu); gole bez anotacji strzału idą
  do `attempts_weak.csv` z reason code;
- `not shown` wykluczone; split po meczach.

Wyniki kontroli jakości sample'a:

| Źródło | strzały | gole | homografia OK | piłka wykryta |
|---|---|---|---|---|
| soccernet 224p | 224 | 34 (15.2%) | 72% | 4.5% |
| soccertrack panorama | 210 | 1 (0.5%) | 31% | 0% |

Wnioski:
1. **224p nie wystarcza dla podejść pozycyjnych** (piłka ma ~2 px). Podejścia obrazowe
   (C/D/E) mogą trenować na 224p; A/B wymagają 720p.
2. **SoccerTrack BAS nie anotuje strzału przy golach** (annotator znakuje DRIVE/CROSS,
   potem GOAL; część goli nie ma żadnego eventu tej samej drużyny w 12 s). Pozytywy
   z tego źródła wymagają lokalizatora kontaktu z piłką z wideo (plan poniżej) albo
   pozycji z GSR. Na razie źródło daje wiarygodne negatywy i 1 pozytyw.
3. Model keypointów (broadcast) nie działa na panoramie SoccerTrack; dla tego źródła
   pozycje bierzemy docelowo z GSR (oracle), nie z homografii.

## Skalowanie do pełnego zbioru: co jest potrzebne

1. **Hasło NDA SoccerNet** (formularz na https://www.soccer-net.org/). Wideo — także
   224p — wymaga hasła (sprawdzone: `password required`). Skąd pochodzi lokalne 2.8 GB,
   nie ustaliłem; w środowisku hasła nie ma. Z hasłem: pobranie 720p dla wszystkich
   meczów z co najmniej jednym strzałem; szacunek ~1.4 GB/połówka → **~500 meczów ≈
   0.7-1.0 TB** przed kompresją. Opcja oszczędna: pobierać połówkę → wycinać okna
   ±5 s wokół strzałów → usuwać mkv (redukcja do ~2-3 GB na całość, kosztem ponownego
   pobrania przy zmianie protokołu). Rekomendacja: wycinanie okien od razu.
2. **Dostęp do HF `atomscott/soccertrack-v2`** (gated; lokalny token dostaje 404 —
   trzeba kliknąć "request access" na stronie datasetu z konta właściciela tokena).
   Alternatywa: mirror Google Drive (codex miał 429 rate limit; brakuje 11 połówek
   wideo i 2 GSR).
3. **Miejsce na dane** — przygotowywane przez użytkownika; docelowa struktura:
   `raw/` (mkv/mp4, read-only) → `frames/` (JPEG okna wokół strzałów) →
   `index.parquet` wg kontraktu z README.

## Kolejne kroki budowy datasetu (po odblokowaniu dostępów)

1. SoccerNet: pobieranie 720p partiami (mecz po meczu), ekstrakcja okien strzałów,
   filtr `grass_ratio` + liczba keypointów, rejestr `raw_assets` z sha256; docelowo
   ~10k strzałów, ~1.2k goli.
2. SoccerTrack: dokończyć 11 połówek wideo; lokalizator kontaktu dla goli (tracking
   piłki w oknie [ostatnia akcja drużyny, GOAL], minimum dystansu piłka-stopy, kontrola
   trajektorii do bramki — protokół z przewodnika codexa, krok 4); dołączenie pozycji
   GSR jako `gt_positions`.
3. StatsBomb 360: bez obrazu; użycie jako pretraining podejść pozycyjnych (A/B) i
   sanity check cech.
4. Wersjonowanie: `dataset_card.md` + wersja przy każdej zmianie definicji; retencja
   goli vs nie-goli raportowana per mecz (kontrola selection bias).

## Ryzyka

- Brak hasła NDA zatrzymuje 95% wolumenu (SoccerNet).
- Lokalizator kontaktu dla SoccerTrack to osobny mini-projekt (detektor piłki na
  panoramie); do czasu jego walidacji SoccerTrack nie wnosi pozytywów.
- 385 GB wolnego dysku < pełne 720p bez wycinania okien.
