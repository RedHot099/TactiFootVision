# Plan Prezentacji Badawczej: Video-Only xG

Cel prezentacji: opowiedziec o projekcie jako o eksperymencie badawczym nad liczeniem xG z samego wideo. Prezentacja ma pokazywac kolejne problemy percepcyjne, zastosowane architektury, wyniki ablation i trade-offy.

Proponowany czas: `18-22 minuty` + `5-8 minut Q&A`.

## Glowna Linia Narracji

Video-only xG nie jest jednym modelem. To lancuch zaleznosci:

```text
homografia -> detekcja pilki -> tracking pilki -> moment strzalu -> cechy -> model xG
```

Najwieksze zyski przyszly z poprawy etapow percepcyjnych i redukcji false positive. Dopiero po tych poprawkach porownanie modeli xG zaczelo byc sensowne.

## Kontrakt Metodologiczny

Runtime korzysta wylacznie z nagran wideo. StatsBomb jest uzywany dopiero po zapisaniu predykcji: do matchingu, ewaluacji, kalibracji eksperymentalnej i analizy bledow.

Badanie jest obecnie single-match study:

```text
Chelsea FCW vs Manchester United
FA WSL 2020/2021
reference: 23 StatsBomb shots
StatsBomb total xG: 2.532141
```

Wyniki sa dobre jako proof-of-concept i diagnoza architektury, ale nie sa jeszcze walidacja publikacyjna.

## Materialy Do Uzycia

Pelne demo 19 strzalow, okno `-3s/+8s`:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_all19_m3_p8.mp4
```

Krotsze demo top 5:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_top5_m3_p8.mp4
```

Starsze demo roznic xG:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_winners_10fps_20260526/previews/xg_delta_shot_compilation_wide.mp4
```

Top 5 wedlug StatsBomb:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_winners_10fps_20260526/previews/top5_statsbomb_xg_compilation.mp4
```

Glowne artefakty tabelaryczne:

- `docs/video_xg/video_only_xg_project_summary.md`
- `02_ball_ablation.csv`
- `03_shot_ablation.csv`
- `04_projection_ablation.csv`
- `05_xg_ablation.csv`
- `13_shot_fp_ablation.csv`
- `final_variant_ranking.csv`
- `final_shot_detection_report.md`

## Struktura Slajdow

### Slajd 1: Tytulowy

Tytul:

```text
Video-Only xG: estymacja jakosci strzalow z broadcast video
```

Podtytul:

```text
Od homografii i trajektorii pilki do porownania modeli xG
```

Na slajdzie:

- nazwa projektu,
- dataset: `Chelsea FCW vs Manchester United, FA WSL 2020/2021`,
- informacja: `runtime input = only video`,
- autorzy / afiliacja / data.

Wizual:

- kadr z meczu z lekko zaznaczona pilka i bramka,
- bez claimow wynikowych na pierwszym slajdzie.

### Slajd 2: Spis Tresci

Proponowany spis:

1. Homografia i geometria boiska.
2. Detekcja pilki i czestotliwosc probkowania.
3. Rekonstrukcja trajektorii pilki.
4. Wykrywanie momentu strzalu.
5. Redukcja false positive.
6. Modele xG i ich roznice.
7. Wyniki na przykladowym meczu.
8. Przyklady wideo.
9. Ograniczenia, rozwoj i literatura.

Przekaz:

- prezentacja idzie po kolei przez zaleznosci pipeline'u,
- kazdy etap ma osobne metryki i osobne ograniczenia.

### Slajd 3: Poprawa Jakosci Homografii

Ten slajd zostaje celowo pusty do pozniejszego uzupelnienia.

Sugerowana struktura, gdy bedziemy go dopisywac:

- problem: broadcast video nie daje stabilnego pitch coordinate system,
- porownane warianty: `degraded_image_normalized`, `line_box_heuristic`, `last_stable_homography`,
- metryki: homography coverage, projection confidence, fallback ratio,
- wniosek: bez mocnej homografii distance/angle sa tylko przyblizeniem.

Placeholder wizualny:

```text
[TU DODAC: boisko, linie, punkty kontrolne, przyklad projekcji]
```

### Slajd 4: Czestsze Wykrywanie Pilki

Pytanie badawcze:

```text
Czy przejscie z rzadkiego skanowania do 10 FPS poprawia wykrywanie strzalow?
```

Tabela:

| Metryka | 1 FPS | 10 FPS |
|---|---:|---:|
| candidates | `80` | `80` |
| hit@2s | `0.0435` | `0.8261` |
| hit@1s | `0.0435` | `0.5217` |
| temporal MAE | `25.86s` | `1.77s` |
| missing-center fallback | `61/80` | `0/80` |

Wniosek:

- pilka jest zbyt mala i zbyt szybka, zeby `1 FPS` wystarczal do shot spottingu,
- `10 FPS` radykalnie poprawia recall i jakosc cech,
- koszt obliczeniowy rosnie, wiec potrzebne byly checkpointy i chunked detection.

Trade-off:

- `1 FPS`: szybki feedback loop, slaba jakosc temporalna,
- `10 FPS`: duzo lepszy sygnal, wiekszy koszt detekcji i storage.

### Slajd 5: Poprawa Sledzenia Pilki

Pytanie badawcze:

```text
Ktora metoda rekonstrukcji trajektorii pilki daje najlepszy kompromis coverage/stabilnosc?
```

Porownane podejscia:

- `baseline_kalman`: prosty smoother z interpolacja,
- `optical_flow_template`: lokalne sledzenie template pilki,
- `viterbi_dp`: globalnie spojna sciezka przez detekcje,
- `kalman_rts_v2`: ostrozniejszy smoother z gatingiem.

Tabela:

| wariant | coverage | recall | trajectory F1 proxy | GT-window coverage | speed outliers |
|---|---:|---:|---:|---:|---:|
| Baseline Kalman Smoother | `0.926` | `1.000` | `1.000` | `1.000` | `0.000` |
| Optical Flow Template Refinement | `0.926` | `1.000` | `1.000` | `1.000` | `0.000` |
| Viterbi Dynamic Programming Ball Path Reconstruction | `0.705` | `0.870` | `0.839` | `0.680` | `0.189` |
| Kalman RTS Conservative Smoother v2 | `0.659` | `0.870` | `0.930` | `0.678` | `0.000` |

`trajectory F1 proxy` laczy recall z kara za speed outliers. To nie jest klasyczne F1 z pelnego GT pozycji pilki, tylko czytelny skrot do slajdu: wysoka wartosc oznacza, ze metoda pokrywa okna strzalow i rzadko generuje fizycznie podejrzane skoki.

Wniosek:

- wygral prostszy wariant Kalman, bo utrzymal coverage wokol strzalow,
- Viterbi/DP byl ciekawy architektonicznie, ale w tym runie tracil obserwacje i generowal outliery,
- ostrozniejsze filtrowanie nie zawsze jest lepsze, gdy downstream wymaga wysokiego recall.

Trade-off:

- wiecej interpolacji poprawia coverage, ale moze wygladzac nietypowe loty,
- mocniejszy gating poprawia wiarygodnosc, ale moze usunac realne akcje.

### Slajd 6: Wykrywanie Momentu Strzalu

Pytanie badawcze:

```text
Czy same cechy kinematyczne i kontakt pilka-zawodnik wystarcza do znalezienia strzalu?
```

Baseline po poprawie pilki:

| Metryka | 10 FPS przed redukcja FP |
|---|---:|
| candidates | `80` |
| hit@2s | `0.8261` |
| hit@1s | `0.5217` |
| precision@2s | `0.2375` |
| false positives | `61` |
| temporal MAE | `1.7707s` |

Interpretacja:

- kinematyka pilki daje wysoki recall,
- problemem sa zdarzenia podobne do strzalu: podania, wybicia, dosrodkowania, dryblingi,
- dobry spotting czasowy nie wystarcza, jezeli kandydatow jest zbyt duzo.

Trade-off:

- obnizenie progu poprawia recall,
- ale kazdy dodatkowy FP dodaje sztuczne xG do sumy meczu.

### Slajd 7: Poprawa Jakosci Wykrywania Strzalu / Redukcja False Positive

Pytanie badawcze:

```text
Czy da sie usunac FP bez istotnej utraty hit@2s?
```

Porownane warianty:

- `baseline_contact_kinematic`,
- `learned_temporal`,
- `hard_negative_calibrated`,
- `high_recall_cascade`,
- `windowed_temporal`,
- `rule_sweep`,
- `dense_local_refinement`.

Tabela:

| wariant | cand. | hit@2s | precision@2s | FP | MAE |
|---|---:|---:|---:|---:|---:|
| Hard-Negative Calibrated Temporal Ranker | `19` | `0.826` | `1.000` | `0` | `3.11s` |
| Learned Temporal Ranker | `80` | `0.826` | `0.238` | `61` | `1.77s` |
| Baseline Contact-Kinematic Detector | `80` | `0.826` | `0.238` | `61` | `1.77s` |
| High-Recall Cascade Ranker | `50` | `0.087` | `0.040` | `48` | `37.31s` |
| Windowed Temporal Ranker | `30` | `0.043` | `0.033` | `29` | `50.69s` |
| Rule-Sweep Shot Ranker | `80` | `0.087` | `0.025` | `78` | `22.36s` |
| Dense Local Contact Refinement | `80` | `0.087` | `0.025` | `78` | `22.33s` |

Architektura zwyciezcy:

```text
high-recall seed candidates
  -> window features
  -> direction/contact/flight features
  -> hard-negative filter
  -> threshold selected under recall constraint
```

Wniosek:

- `hard_negative_calibrated` utrzymal `hit@2s = 0.8261` i zredukowal FP z `61` do `0`,
- koszt: temporal MAE wzroslo z `1.77s` do `3.11s`,
- dla sumy xG to dobry trade-off, dla precyzyjnego spottingu wymaga dalszej pracy.

### Slajd 8: Rozne Podejscia Do Wyliczania xG

Pytanie badawcze:

```text
Ktore rodziny modeli xG sa sensowne, gdy cechy pochodza z video?
```

Podejscia:

| podejscie | wejscia | intuicja |
|---|---|---|
| DataBallPy Simple Location xG | lokalizacja strzalu + parametry DataBallPy | zewnetrzny prosty baseline lokalizacyjny |
| Video Geometry xG | odleglosc, kat, centralnosc | klasyczny prosty baseline |
| Video Freeze-Frame Context xG | geometria + bramkarka/obroncy | przyblizenie freeze-frame context |
| Video Kinematic Context xG | geometria + ruch pilki | dodaje dynamike przed/po strzale |
| Quality-Aware xG Ensemble | outputy modeli + jakosc cech | obniza zaufanie przy slabych cechach |
| Formula Coefficient Fit xG | video-derived features + fit wag | najlepsze dopasowanie do meczu |
| Isotonic/Platt xG Calibration | kalibracja outputow | poprawa kalibracji per shot |
| Neural Video xG MLP | mala MLP na video-derived features | nieliniowe odwzorowanie StatsBomb xG |

Przekaz:

- proste modele sa stabilne, ale niedoszacowuja trudniejsze sytuacje,
- kontekst i kinematyka dodaja informacje, ale sa wrazliwe na jakosc percepcji,
- kalibracja poprawia liczby, lecz moze overfitowac przy jednym meczu.

### Slajd 9: Najwazniejsze Roznice Pomiedzy Podejsciami Do xG

Tabela trade-offow:

| metoda | mocna strona | slaba strona |
|---|---|---|
| DataBallPy Simple Location xG | porownanie z biblioteka zewnetrzna | opiera sie glownie na lokalizacji strzalu |
| Video Geometry xG | prosta, interpretowalna | ignoruje presje, bramkarke i ruch |
| Video Freeze-Frame Context xG | dodaje uklad zawodnikow | zalezy od tracking/projection quality |
| Video Kinematic Context xG | uwzglednia predkosc i kierunek pilki | wrazliwa na blad trajektorii |
| Quality-Aware xG Ensemble | jawnie uzywa jakosci cech | konserwatywna w tym meczu |
| Formula Coefficient Fit xG | najlepszy total xG | najwieksze ryzyko overfitu |
| Isotonic/Platt xG Calibration | niski MAE per shot | gorszy total error niz coefficient fit |
| Neural Video xG MLP | lapie nieliniowe zaleznosci cech video | wymaga wiekszego train setu i holdoutu |

Najwazniejszy wniosek:

- model najlepszy per shot nie musi byc najlepszy jako suma meczu,
- total xG jest bardzo wrazliwe na FP,
- bez redukcji FP bardziej zlozony model moze wygladac gorzej niz prosty.

### Slajd 10: Podsumowanie Metryk Dla xG Na Przykladowym Meczu

Tabela po redukcji FP:

| podejscie | oceniana metoda | MAE vs SB xG | total pred | total error |
|---|---|---:|---:|---:|
| DataBallPy Simple Location xG | DataBallPy Simple Location xG | `0.0982` | `0.4823` | `-2.0498` |
| Video Geometry xG | Video Geometry xG | `0.1003` | `0.5738` | `-1.9583` |
| Video Freeze-Frame Context xG | Video Freeze-Frame Context xG | `0.0911` | `0.7602` | `-1.7719` |
| Video Kinematic Context xG | Video Kinematic Context xG | `0.0869` | `1.1162` | `-1.4159` |
| Quality-Aware xG Ensemble | Quality-Aware xG Ensemble | `0.0910` | `0.8311` | `-1.7010` |
| Formula Coefficient Fit xG | Formula Coefficient Fit xG | `0.0718` | `2.4232` | `-0.1089` |
| Isotonic/Platt xG Calibration | Video Geometry xG + Isotonic/Platt Calibration | `0.0573` | `2.3002` | `-0.2320` |
| Isotonic/Platt xG Calibration | Video Freeze-Frame Context xG + Isotonic/Platt Calibration | `0.0579` | `2.2766` | `-0.2555` |
| Isotonic/Platt xG Calibration | Video Kinematic Context xG + Isotonic/Platt Calibration | `0.0573` | `2.3002` | `-0.2320` |
| Neural Video xG MLP | Neural Video xG MLP | `0.0166` | `2.2962` | `-0.2360` |

Wynik finalny:

| metryka | wartosc |
|---|---:|
| hit@2s | `0.8261` |
| hit@1s | `0.5217` |
| total xG | `2.4232` |
| StatsBomb total xG | `2.5321` |
| total xG error | `-0.1089` |


### Slajd 11: Przyklad Video Z Wyliczonego xG

Material:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_top5_m3_p8.mp4
```

Cel slajdu:

- pokazac 5 najwyzej ocenionych strzalow,
- pokazac overlay `video xG`, `StatsBomb xG`, delta i outcome,
- zatrzymac sie na jednym strzale z malym bledem i jednym z wiekszym bledem.

Komentarz:

- to jest dowod jakosciowy, nie glowna metryka,
- nalezy zwrocic uwage na timing eventu i stabilnosc pilki.

### Slajd 12: Przyklad Video Z Wyliczonego xG v2

Material:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_xg_fp_reduction_10fps_20260526/previews/fp_reduction_xg_video_summary_all19_m3_p8.mp4
```

Alternatywnie, gdy chcemy pokazac starsze porownanie roznic:

```text
/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/3775567_Chelsea_FCW_vs_Manchester_United/experiments/video_only_xg_end_to_end_winners_10fps_20260526/previews/xg_delta_shot_compilation_wide.mp4
```

Cel slajdu:

- pokazac pelniejszy rozklad przypadkow, nie tylko top 5,
- pokazac, gdzie model jest stabilny, a gdzie ma problem,
- omowic reboundy, klastry strzalow i przypadki z opoznieniem timestampu.

Komentarz:

- v2 powinien sluzyc do dyskusji o failure modes,
- nie przeciagac demo; wybrac 2-3 konkretne momenty.

### Slajd 13: Obecne Ograniczenia / Mozliwosci Rozwoju

Ograniczenia:

- walidacja na jednym meczu,
- kalibracja i ewaluacja sa jeszcze zbyt blisko siebie,
- homografia jest obecnie najslabszym etapem dla generalizacji,
- brakuje solidnej runtime team attribution,
- timestampy video i eventow moga miec przesuniecia,
- reboundy i klastry wymagaja osobnej metryki,
- precyzyjne `hit@0.5s` nadal jest niskie.

Mozliwosci rozwoju:

1. multi-match train/validation/test split po meczach,
2. osobny benchmark homografii i projekcji cech,
3. lokalne `30 FPS` refinement wokol kandydatow,
4. model temporalny inspirowany SoccerNet Ball Action Spotting,
5. kalibracja xG na wielu meczach z confidence intervals,
6. osobna ewaluacja event-level, cluster-level i match-level,
7. porownanie z event-only StatsBomb baseline.

### Slajd 14: Literatura

Proponowane zrodla:

- SoccerNet Action Spotting: https://www.soccer-net.org/tasks/action-spotting
- SoccerNet data and tasks: https://www.soccer-net.org/data
- SoccerNet `sn-spotting`: https://github.com/SoccerNet/sn-spotting
- Ball Action Spotting winning repo: https://github.com/lRomul/ball-action-spotting
- SoccerNet 2024 challenge results: https://www.soccer-net.org/challenges/2024
- TrackNet: https://arxiv.org/abs/1907.03698
- TrackNetV4: https://tracknetv4.github.io/
- ByteTrack: https://arxiv.org/abs/2110.06864
- KU Leuven `soccer_xg`: https://github.com/ML-KULeuven/soccer_xg
- StatsBomb xG explainer: https://www.hudl.com/blog/expected-goals-xg-explained
- StatsBomb freeze frames: https://blogarchive.statsbomb.com/news/statsbomb-data-case-studies-freeze-frames-and-defender-locations/
- DataBallPy simple xG: https://databallpy.readthedocs.io/en/v0.5.0/features/simple_xG_models.html

Na slajdzie pokazac tylko 6-8 pozycji. Reszte przeniesc do backupu albo notatek.

### Slajd 15: Slajd Koncowy

Trzy zdania finalne:

1. Najwiekszy zysk przyszedl z poprawy percepcji pilki i kontroli false positive, nie z bardziej zlozonego wzoru xG.
2. Najlepszy wariant osiagnal `hit@2s = 0.8261`, `precision@2s = 1.0` i total xG `2.4232` wobec StatsBomb `2.5321`.
3. Glowny kierunek dalszych badan to multi-match validation, mocniejsza homografia i test generalizacji poza meczem kalibracyjnym.

Ostatni slajd:

```text
Questions?
```

Opcjonalnie:

- QR/link do repo lub dokumentacji,
- link do pelnego raportu,
- jedna klatka z overlayem finalnego pipeline'u.

## Proponowany Timing

| Slajdy | Sekcja | Czas |
|---|---|---:|
| 1-2 | tytul i spis tresci | `1.5 min` |
| 3 | homografia | `1.5 min` |
| 4-5 | detekcja i tracking pilki | `4 min` |
| 6-7 | shot spotting i FP reduction | `5 min` |
| 8-10 | modele xG i metryki | `5 min` |
| 11-12 | przyklady wideo | `3 min` |
| 13 | ograniczenia i rozwoj | `2 min` |
| 14-15 | literatura i zakonczenie | `1 min` |
| Q&A | pytania | `5-8 min` |

## Backup Slides

1. Pelna tabela `13_shot_fp_ablation.csv`.
2. Pelna tabela `05_xg_ablation.csv`.
3. Pelna tabela `02_ball_ablation.csv`.
4. Pelna tabela `04_projection_ablation.csv`.
5. Szczegoly scoringu `hard_negative_calibrated`.
6. Przyklady matched shots z najwiekszym bledem xG.
7. Przyklady false positive usunietych przez filtr.
8. Przyklady klastrow StatsBomb/reboundow.
9. Szczegoly checkpointow pipeline'u.
10. Plan eksperymentu multi-match.
