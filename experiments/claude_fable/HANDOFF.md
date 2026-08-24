# Handoff: krok 1, pierwsze uruchomienie na prawdziwych danych

Stan na 2026-08-23. Kontekst projektu: `README.md`, research: `research/literature_and_datasets.md`.
Wszystkie komendy z katalogu `experiments/claude_fable` w worktree
`.claude/worktrees/claude-fable-xg-poc` (branch `claude-fable/xg-poc`).

## Warunek wejścia

Agent danych dostarcza `index.parquet` zgodny z kontraktem z README
(`shot_id, source, game, split, goal, frame_path` + opcjonalne `is_penalty`,
`visibility`, `gt_positions`). Umówiona lokalizacja: zapytaj agenta danych; dalej
oznaczana `$INDEX`. Nazwa runu `$RUN` (np. `soccernet_v1`).

Nic nie pobieraj i nie wycinaj klatek samodzielnie; to po stronie agenta danych.
Jeśli dostaniesz tylko etykiety + mkv, parser i ekstraktor są w
`xgpoc/data/soccernet_labels.py` i `xgpoc/data/frames.py`, ale ich uruchomienie
uzgodnij z użytkownikiem.

## Kroki

1. Kontrola indexu (bez GPU, sekundy):
   ```bash
   ./run.sh pytest -q
   ./run.sh python scripts/check_index_quality.py $INDEX
   ```
   Oczekiwania: 0 meczów wspólnych train/test; goal rate 8-15% (SoccerNet: gole
   to osobna etykieta, parser je dokleja; jeśli >20% albo <5%, etykiety są źle
   złączone, wróć do agenta danych); `missing frames` bliskie 0.

2. Cache pozycji (GPU, ~30-50 klatek/s, czyli ~5 min na 10k strzałów):
   ```bash
   ./run.sh python scripts/build_positional_cache.py $INDEX cache/$RUN/positional
   ./run.sh python scripts/check_index_quality.py $INDEX cache/$RUN/positional
   ```
   Zapisz do raportu: odsetek `homography_ok`, `has_ball`, `team_known`, `usable for A/B`.
   Kontrola merytoryczna: goal rate musi rosnąć wraz z malejącym `dist_to_goal`
   (bucket 0-8 m wyraźnie wyżej niż 25-40 m). Jeśli nie rośnie, homografia albo
   wybór bramki atakowanej są złe; obejrzyj kilkanaście klatek z nałożonymi
   pozycjami (`FreezeFrameExtractor` zwraca też `boxes_xyxy`, rysowanie do dopisania).

3. Krótki przebieg wszystkich PoC (limit kilku minut na model, zgodnie z ustaleniem
   z użytkownikiem: na razie tylko drobne próby):
   ```bash
   ./run.sh python scripts/run_poc.py --index $INDEX --cache cache/$RUN \
       --out results/$RUN --poc a b d e c --epochs 10 --max-minutes 4
   ./run.sh python scripts/compare.py --index $INDEX --results results/$RUN
   ```
   Kolejność a b d e c jest celowa: C (fine-tuning ConvNeXt) jest najdroższe;
   cache DINOv2 (`cache/$RUN/features_dinov2_s.npz`) liczy się raz przy pierwszym D.
   Przy >20k klatek cache DINOv2 ma ~20k × 577 × 384 × 2 B ≈ 9 GB; jeśli RAM nie
   starczy, ogranicz index (`df.sample`) albo dopisz ładowanie z `mmap_mode="r"`.

4. Zapisz wynik w `results/$RUN/REPORT.md` (results/ jest w .gitignore; skopiuj
   tabelę do README w sekcji "Stan"): tabela z `comparison.csv` (Brier, BSS,
   log-loss, AUC z CI), odsetki z kroku 2, czas treningu i liczba parametrów
   z `*_metrics.json`. Commit na branchu `claude-fable/xg-poc`.

## Kryteria "krok 1 zaliczony"

- Wszystkie 5 PoC mają metryki na tym samym zbiorze testowym (ta sama liczba `n`
  dla C/D; A/B/E mogą mieć mniej, bo odpadają klatki bez cache, zapisz ile).
- Co najmniej A_gbm ma BSS > 0 i AUC > 0.65 na teście. Jeśli nie, to problem danych
  lub pozycji, nie architektur; nie idź dalej w trening.
- Różnice między PoC podane z CI po meczach, bez wniosków "X wygrywa", jeśli CI się pokrywają.

## Czego nie robić

- Nie zmieniać kontraktu indexu bez uzgodnienia z agentem danych.
- Nie odpalać długiego treningu; to osobny krok na sygnał użytkownika.
- Nie modyfikować plików poza `experiments/claude_fable/`.

## Znane ryzyka do obserwacji

- Piłka: na losowych klatkach detektor widział ją w 23%; w momencie strzału
  powinno być więcej, ale jeśli `has_ball` < 50%, podejścia A/B stracą główną cechę.
  Opcja awaryjna: użyć klatek z offsetem -400 ms (agent danych może je dostarczyć
  jako dodatkową kolumnę), gdzie piłka jest jeszcze przy nodze.
- Detektor prawie nie zwraca klasy goalkeeper; cecha `gk_*` może być stale zerowa.
- Klatki bez boiska (zbliżenia, powtórki): `homography_ok` poniżej 60% oznacza,
  że filtr klatek po stronie danych jest za słaby.
- Split: musi być po meczach. `split_by_game` w `xgpoc/data/index.py` daje
  deterministyczny podział, jeśli agent danych nie dostarczył własnego.
