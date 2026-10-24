# xg-teams 00: merge `master` do `research/xg-detector`

Data: 2026-10-25. Commit merge: `fdaddd2` (rodzice `a15ad2c` i `master` `19a5712`). Tylko CPU, bez GPU, bez odczytu etykiet, bez nowych wyników.

## Co jest teraz na gałęzi

- Szwy z `master`: `tv.evaluation.evaluate_teams` / `evaluate_teams_clips`, czytnik SoccerNet GSR z drużynami, rolami i pozycjami (`tv.data.load_labelled_clip`, `data/soccernet.py`, `data/labelled_clip.py`), `assign_roles`, `tv.analysis.attack_direction`, post-processy, `FrameResult.ball_candidates`, `tv.enums`.
- Tryb jednej klatki z gałęzi: `FrameAnalyzer`, `assign_frame_teams`, `HomographyEstimator.fit`, `fill_ball`, `class_conf`, `resize`.

## Decyzje przy konfliktach (szczegóły w komunikacie `fdaddd2`)

- **Dwie poprawki homografii na liniach.** Obie strony napisały ją niezależnie (`pitch/lines.py`, add/add). `pitch/lines.py` to teraz wersja z `master` (`LineRefiner`); używa jej `update`, czyli tryb wideo i `Pipeline`. Wersja z gałęzi (xg-detector 08) przeszła bez zmian do `pitch/frame_lines.py`; używa jej `fit`, czyli `FrameAnalyzer`. Z nią zbudowano cache badawcze, więc ekstraktory badawcze działają jak przed merge. Którą wersję zostawi pakiet, rozstrzygnie bilet 10a.
- **`class_conf`:** mechanizm z `master` (wspólny dla modeli, te same progi); `resize` z gałęzi zostaje.
- **Podział piłka/ludzie:** jedna funkcja `_detect_ball_and_people` w stylu `master` (wszyscy kandydaci na piłkę, najlepszy pierwszy), wspólna dla `Pipeline` i `FrameAnalyzer`. Wybór piłki taki sam jak wcześniej.
- **Kod badawczy:** zmiany tylko tam, gdzie wymusił je merge: `ball_candidates` w `xgpoc/detector/extract.py` i `tests/xgd_fixtures.py`, import `pitch.frame_lines` w `tests/test_pitchkp_gsr_eval.py`.

## Sprawdzenia

| sprawdzenie | przed merge (`a15ad2c`) | po merge |
|---|---|---|
| testy pakietu (szybkie) | 442 passed | 940 passed, 8 skipped (`master`: 889 passed) |
| testy badawcze (`experiments/claude_fable`) | 1526 passed | 1526 passed |
| ruff `src tests examples notebooks/build` (check + format) | czysto | czysto |
| ruff badawczy (śledzone pliki `experiments/claude_fable`) | 97 błędów | 97 błędów (te same, stary kod v13/v14) |
| `scripts/location_check.py` | — | all passed (`results/feet_location_check_xgt00`) |

`location_check`: `xgd_full3` i `xgd_full4` dają te same wiersze cech, tabele E3 i populacje; cache `cache/xgd_full4_feet_t0/E3` (features sha256 `93b4b580...`) odtwarza się bez różnic; skróty tabel `feet_t0` `db05127b...` / `f3c22b93...` są takie, jak zapisano wcześniej.

Szwy importują się przez `./run.sh --no-sync python` z pakietu tej gałęzi.

## Uwagi

- Skróty kodu zapisane w rekordach zużytych testów (`research/xg_detector_test_hashes.json`, `research/xgs_stack_protocol*.json`) nie pasują już do zmienionych plików pakietu i `extract.py`. Te jednorazowe skrypty testowe odmówią ponownego uruchomienia, i tak ma być, bo testy są zużyte.
- Pliki uruchomieniowe i README z `refine_with_lines: true` dla trybu wideo używają na tej gałęzi `LineRefiner` z `master`.
- Testy `-m model` (wagi, GPU) nie były uruchamiane.
