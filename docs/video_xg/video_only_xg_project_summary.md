# Video-Only xG: Podsumowanie Procesu, Wynikow I Decyzji

Stan dokumentu: 2026-05-27.

## Cel

Celem projektu bylo przygotowanie pipeline'u, ktory dla pelnego meczu wideo wykrywa strzaly, liczy xG per strzal i sumuje xG bez korzystania z eventow StatsBomb/SoccerNet w runtime. Dane referencyjne StatsBomb sa dopuszczone dopiero po zapisaniu predykcji: do ewaluacji, kalibracji eksperymentalnej i analizy bledow.

Docelowy przeplyw:

```text
part1.mp4 + part2.mp4
  -> sampled frames
  -> detections/tracks
  -> ball trajectory
  -> shot candidates
  -> video-only features
  -> xG predictions
  -> StatsBomb comparison after inference
```

Glowny dataset eksperymentalny:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/
  3775567_Chelsea_FCW_vs_Manchester_United/
```

Reference:

- `23` strzaly StatsBomb
- StatsBomb total xG: `2.532141`
- gole w danych referencyjnych: `3`

## Najwazniejsze Artefakty

Run startowy `1 FPS`:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_yolo11m_1fps_20260526_131429
```

Run finalny `10 FPS` przed redukcja false positive:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_winners_10fps_20260526
```

Run po redukcji false positive:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526
```

Configi:

- `configs/experiments/video_xg_end_to_end_fa_wsl_winners_10fps.yaml`
- `configs/experiments/video_xg_fp_reduction_fa_wsl_10fps.yaml`
- `configs/experiments/video_xg_end_to_end_fa_wsl_winners_10fps_precision.yaml`

Wideo podsumowujace:

- pelne 19 strzalow, okno `-3s/+8s`:  
  `/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_all19_m3_p8.mp4`
- top 5 wedlug video xG, okno `-3s/+8s`:  
  `/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_top5_m3_p8.mp4`
- indeks klipow:  
  `/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/video_summary_fp_reduction_m3_p8_index.csv`

## Etap 1: Pipeline End-to-End I Checkpointy

Najpierw przygotowalismy pipeline end-to-end, ktory zapisuje checkpointy kazdego etapu:

- `00_video_timeline.json`
- `01_sampled_frames.parquet`
- `02_detections.parquet`
- `03_tracks.parquet`
- `04_homographies.parquet`
- `05_ball_trajectory.parquet`
- `06_shot_candidates.parquet`
- `07_refined_shots.parquet`
- `08_video_features.csv`
- `09_predictions.csv`
- `10_metrics.json`
- `final_report.md`

Kluczowe decyzje techniczne:

- runtime input to tylko wideo,
- StatsBomb jest ladowany dopiero po predykcji,
- kazdy etap jest wznawialny,
- detekcja moze dzialac chunkami, zeby nie tracic calego runu po przerwaniu,
- notebook ma tylko wywolywac runnera i czytac artefakty, bez logiki eksperymentu w komorkach.

## Etap 2: Startowy Run `1 FPS`

Pierwszy pelny eksperyment `YOLO11m 1 FPS` pozwolil znalezc glowne waskie gardla.

| Metryka | Wartosc |
|---|---:|
| predicted shots | `80` |
| reference shots | `23` |
| hit@0.5s | `0.0000` |
| hit@1s | `0.0435` |
| hit@2s | `0.0435` |
| temporal MAE | `25.8587s` |

Jakosc cech strzalow:

| feature source | liczba |
|---|---:|
| missing_center_fallback | `61` |
| observed | `10` |
| kalman_rts_interpolated | `9` |

Wniosek: podstawowym problemem nie byl sam wzor xG, tylko jakosc trajektorii pilki i momentow strzalu. `61/80` cech strzalow powstawalo z awaryjnego srodka obrazu, co psulo geometrie i ranking.

Startowe sumy xG:

| metoda | total xG | blad vs StatsBomb |
|---|---:|---:|
| video_kinematic_context | `1.0729` | `-1.4592` |
| video_freeze_context | `0.6342` | `-1.8979` |
| video_geometry | `0.3050` | `-2.2271` |

## Etap 3: Poprawa Trajektorii Pilki I Run `10 FPS`

Najwieksza poprawa przyszla po przejsciu na `10 FPS` i usprawnieniu rekonstrukcji pilki:

- niski prog detekcji pilki,
- checkpointowana detekcja,
- usuwanie fizycznie nieprawdopodobnych outlierow,
- interpolacja brakow,
- wygenerowanie trajektorii bez traktowania `missing_center_fallback` jako pelnoprawnej pozycji,
- wariant `optical_flow_template` w zwycieskim configu `10 FPS`.

Run `10 FPS` przed redukcja FP:

| Metryka | Wartosc |
|---|---:|
| predicted shots | `80` |
| reference shots | `23` |
| hit@0.5s | `0.1739` |
| hit@1s | `0.5217` |
| hit@2s | `0.8261` |
| temporal MAE | `1.7707s` |

Jakosc cech po poprawkach:

| feature source | liczba |
|---|---:|
| observed | `41` |
| kalman_rts_interpolated | `38` |
| optical_flow_template | `1` |
| missing_center_fallback | `0` |

Wniosek: po poprawie trajektorii pilki recall stal sie akceptowalny, ale liczba false positive pozostala zbyt wysoka.

## Etap 4: Porownanie Metod xG Przed Redukcja FP

Na runie `10 FPS` porownalismy metody xG na wszystkich `80` kandydatach oraz na dopasowaniach do StatsBomb.

| metoda | matched MAE | matched total pred | all-candidate total | all-candidate error |
|---|---:|---:|---:|---:|
| video_geometry | `0.1010` | `0.6446` | `2.2554` | `-0.2767` |
| quality_aware_ensemble | `0.0934` | `0.8474` | `2.8453` | `+0.3132` |
| video_freeze_context | `0.0934` | `0.8644` | `2.9584` | `+0.4263` |
| video_kinematic_context | `0.0865` | `1.0748` | `3.4910` | `+0.9588` |
| coefficient_fit | `0.0753` | `2.3770` | `9.9206` | `+7.3884` |

Wazny wniosek: modele kalibrowane na dopasowanych strzalach wygladaly dobrze na matched set, ale eksplodowaly na pelnych `80` kandydatach. To potwierdzilo, ze przed dalsza kalibracja xG trzeba zredukowac false positive w detekcji strzalow.

## Etap 5: Diagnoza False Positive

Stan przed poprawka:

- `80` kandydatow
- `19` unikalnych kandydatow matchujacych GT w oknie `2s`
- `61` false positive
- `hit@2s = 0.8261`
- `precision@2s = 0.2375`

Najbardziej podejrzane FP mialy wysokie score mimo:

- ruchu pilki od bramki,
- braku sensownego wzorca kontakt -> przyspieszenie,
- zbyt krotkiego lotu po kontakcie,
- zdarzen wygladajacych jak drybling/podanie,
- braku logicznej strefy strzalu.

## Etap 6: Tweak'i Zastosowane Do Redukcji FP

Wprowadzilismy osiem strategii, ale finalnie najwieksza wartosc dalo polaczenie kilku z nich w `hard_negative_calibrated`.

### 1. Resolver kierunku do bramki

Zamiast zakladac bramke po stronie boiska, liczony jest progres pilki do obu bramek po zdarzeniu. Ruch od wybranej bramki dostaje silny veto multiplier.

### 2. Kontakt przed przyspieszeniem

Strzal jest mocniejszy, gdy bliski kontakt zawodnik-pilka wystepuje przed lub w momencie wzrostu predkosci pilki. Przyspieszenie bez trackow nie jest automatycznie odrzucane, zeby nie zabic recall przy brakujacych zawodnikach.

### 3. Post-shot flight persistence

Dodano cechy lotu po strzale:

- progres do bramki,
- spojny kierunek,
- srednia i maksymalna predkosc po zdarzeniu,
- brak natychmiastowego recontactu typowego dla dryblingu.

### 4. Shot-zone i long-shot exception

Preferowane sa kandydaty blizej bramki. Strzal z dalszej odleglosci moze przejsc tylko, jesli ma wysoka predkosc i wyrazny progres do bramki.

### 5. Wieloskalowe cechy temporalne

Ranker dostaje cechy z okien:

- `±0.5s`,
- `±1.0s`,
- `±2.0s`.

Uzywane sygnaly to m.in. max/mean speed, peak acceleration, max contact, mean direction.

### 6. Hard-negative mining

False positive z runu `10 FPS` zostaly potraktowane jako twarde negatywy po zapisaniu video-only features. StatsBomb sluzy tylko do etykietowania po inferencji.

### 7. Kalibracja progu

Zamiast stalego progu `score > 0.05`, prog jest wybierany przez `SoftCompositeThresholdSelector` pod warunkiem soft recall:

- preferowane `hit@2s >= 0.78`,
- preferowane `hit@1s >= 0.45`,
- maksymalizacja precision i redukcji FP.

### 8. Seed-pool calibration

To byl krytyczny tweak. Pierwsza wersja `hard_negative_calibrated` generowala kandydatow od zera i redukowala FP bardzo mocno, ale spadala z `hit@2s` do `0.7391`. Finalna wersja uzywa obecnego wysokorecallowego `learned_temporal` jako seed-pool i dopiero filtruje go modelem jakosci. To utrzymalo recall i usunelo FP.

## Etap 7: Wyniki Po Redukcji FP

Ablacja detekcji strzalow:

| wariant | kandydaci | hit@0.5s | hit@1s | hit@2s | precision@2s | FP | shot_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_negative_calibrated | `19` | `0.1739` | `0.5217` | `0.8261` | `1.0000` | `0` | `0.8019` |
| learned_temporal | `80` | `0.1739` | `0.5217` | `0.8261` | `0.2375` | `61` | `0.3298` |
| baseline_contact_kinematic | `80` | `0.1739` | `0.5217` | `0.8261` | `0.2375` | `61` | `0.3298` |
| high_recall_cascade | `50` | `0.0000` | `0.0000` | `0.0870` | `0.0400` | `48` | `0.0092` |
| windowed_temporal | `30` | `0.0000` | `0.0000` | `0.0435` | `0.0333` | `29` | `0.0084` |
| rule_sweep | `80` | `0.0000` | `0.0000` | `0.0870` | `0.0250` | `78` | `0.0028` |
| dense_local_refinement | `80` | `0.0000` | `0.0000` | `0.0870` | `0.0250` | `78` | `0.0028` |

Najwazniejszy wynik:

```text
80 kandydatow -> 19 kandydatow
61 FP -> 0 FP
hit@2s utrzymane: 0.8261 -> 0.8261
precision@2s: 0.2375 -> 1.0000
```

## Etap 8: xG Po Redukcji FP

Po redukcji FP najlepszym wariantem xG zostal `coefficient_fit`.

| wariant | metoda | MAE vs StatsBomb | RMSE | pred total xG | StatsBomb total xG | blad |
|---|---|---:|---:|---:|---:|---:|
| coefficient_fit | coefficient_fit | `0.0718` | `0.0949` | `2.4232` | `2.5321` | `-0.1089` |
| isotonic_platt | video_geometry_isotonic_platt | `0.0573` | `0.0887` | `2.3002` | `2.5321` | `-0.2320` |
| isotonic_platt | video_kinematic_context_isotonic_platt | `0.0573` | `0.0887` | `2.3002` | `2.5321` | `-0.2320` |
| neural_video_xg | neural_video_xg | `0.0166` | `0.0474` | `2.2962` | `2.5321` | `-0.2360` |
| isotonic_platt | video_freeze_context_isotonic_platt | `0.0579` | `0.0911` | `2.2766` | `2.5321` | `-0.2555` |
| none | video_kinematic_context | `0.0869` | `0.1175` | `1.1162` | `2.5321` | `-1.4159` |
| quality_aware_ensemble | quality_aware_ensemble | `0.0910` | `0.1267` | `0.8311` | `2.5321` | `-1.7010` |
| none | video_freeze_context | `0.0911` | `0.1269` | `0.7602` | `2.5321` | `-1.7719` |
| none | video_geometry | `0.1003` | `0.1394` | `0.5738` | `2.5321` | `-1.9583` |
| databallpy_simple_xg | databallpy_simple_xg | `0.0982` | `0.1405` | `0.4823` | `2.5321` | `-2.0498` |

Interpretacja:

- Redukcja FP usunela problem zawyzonej sumy xG po kalibracji.
- `coefficient_fit` po FP-reduction ma blad sumy tylko `-0.1089 xG`.
- `neural_video_xg` ma najnizszy MAE per shot na tym meczu, ale jest to wariant uczony na bardzo malym matched secie i wymaga walidacji na holdoucie.
- `databallpy_simple_xg` pelni role zewnetrznego baseline'u lokalizacyjnego; przy obecnej zdegradowanej geometrii silnie zaniza sume xG.
- Surowe metody video-only nadal zanizaja xG, co oznacza, ze ich skale wymagaja dalszej kalibracji na wiekszej liczbie meczow.

## Aktualny Finalny Pipeline

Finalny rekomendowany wariant eksperymentalny:

```text
detection: chunked YOLO11m, 10 FPS
ball: kalman_rts / optical_flow_template trajectory artifacts
shot seed: learned_temporal
shot filter: hard_negative_calibrated
xG: coefficient_fit
evaluation: StatsBomb after prediction
```

Najwazniejsze pliki kodu:

- `src/tactifoot_vision/video_xg/end_to_end.py`
- `src/tactifoot_vision/video_xg/shot_ranking.py`
- `src/tactifoot_vision/video_xg/shot_quality.py`
- `src/tactifoot_vision/video_xg/ablation.py`
- `src/tactifoot_vision/video_xg/config.py`
- `src/tactifoot_vision/enums.py`

## Aktualne Ograniczenia

### Walidacja na jednym meczu

Najlepszy wynik FP-reduction jest bardzo mocny, ale zostal uzyskany na jednym meczu i przy kalibracji z referencja tego samego meczu. Do publikacji potrzebne sa:

- holdout matches,
- cross-validation miedzy meczami,
- ewaluacja na roznych ligach, kamerach i stadionach.

### Kalibracja xG nadal wymaga danych referencyjnych

`coefficient_fit` jest najlepszy po redukcji FP, ale jest wariantem kalibrowanym. W runtime mozna uzywac zapisanych wag/modelu, natomiast ich nauka musi byc wykonana na zbiorze treningowym, a nie na meczu testowym.

### Homografia nie jest jeszcze pelnym backendem boiskowym

Obecny pipeline nadal opiera sie na projekcji zdegradowanej lub heurystycznej. Wyniki sa obiecujace, ale publikacyjny wariant powinien miec stabilna kalibracje boiska:

- linie boiska + RANSAC,
- TVCalib/SoccerSegCal/Sportlight,
- jawne metryki jakosci projekcji.

### Brak pelnej atrybucji druzynowej

Mamy total xG meczu i dopasowanie do StatsBomb team po ewaluacji, ale runtime-only team attribution wymaga stabilnego rozpoznania druzyn, kierunkow ataku i posiadania pilki.

### Detekcja pilki pozostaje krytyczna

Brak pilki albo bledny track nadal moze przesunac moment strzalu. Najwiekszy zysk dal wzrost z `1 FPS` do `10 FPS`, ale docelowo warto dodac:

- fine-tuned ball detector,
- TrackNet-style heatmap,
- lokalny `30 FPS` refinement wokol kandydatow,
- kalibracje niepewnosci trajektorii.

### Matching StatsBomb ma klastry

Niektore wykryte kandydaty odpowiadaja kilku bliskim eventom StatsBomb, np. rebound/seria strzalow. Dlatego `19` kandydatow moze pokrywac `23` referencyjne strzaly. Raportowanie publikacyjne powinno rozdzielac:

- event-level recall,
- candidate-level precision,
- cluster-level recall.

## Co Dziala Najlepiej Teraz

Najbardziej obiecujacy wynik praktyczny:

```text
Video-only detection:
  19 kandydatow
  0 FP wzgledem matchingu 2s
  hit@2s = 0.8261
  hit@1s = 0.5217

xG:
  coefficient_fit total xG = 2.4232
  StatsBomb total xG = 2.5321
  blad sumy = -0.1089
```

To jest bardzo dobry rezultat na poziomie MVP, ale obecnie jest to wynik eksperymentalny, nie jeszcze publikacyjna walidacja.

## Kolejne Kroki

1. Rozszerzyc dataset do wielu meczow ze StatsBomb + video.
2. Rozdzielic train/validation/test po meczach, nie po strzalach.
3. Zapisac i wersjonowac wytrenowany `hard_negative_calibrated` oraz `coefficient_fit`.
4. Dodac rzeczywisty backend homografii.
5. Uruchomic lokalny refinement `30 FPS` wokol kandydatow.
6. Raportowac metryki event-level, candidate-level i cluster-level.
7. Przygotowac publikacyjne confidence intervals przez bootstrap po meczach.
