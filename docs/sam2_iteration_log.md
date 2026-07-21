# SAM2 iteration log (SNMOT-116)

Cel: doprowadzić SAM2 do możliwie najlepszych wyników względem ByteTrack/BoT-SORT, iterując na małym podzbiorze jednej sekwencji.

## Użyte checkpointy RF-DETR (detektor)
- RF-DETR Base: `results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking/models/rfdetr_base_soccernet2023.pth`

## Zmiany w algorytmie i środowisku
- Dodany tryb reseedingu `reanchor`: przy re-promptowaniu używamy bbox-ów z detektora (po IoU match), a nie bbox-ów z masek.
- Dodane opcje post-processingu masek: `mask_threshold`, `mask_open`, `mask_close`.
- SAM2 działa teraz na CPU, jeśli CUDA jest niedostępne (auto wybór urządzenia).
- Poprawiono backend SAM2, aby nie wymuszał `.cuda()` przy pracy na CPU.
- TrackEval może teraz oceniać tylko pierwsze `N` klatek (`--max-frames`), co pozwala na szybkie iteracje.
- Dodano tryb wyjścia `detector_strict`: bbox-y z detektora + odrzucanie tracków bez matcha (kontrola FP).
- Dodano tryb wyjścia `detector_blend`: snap do detekcji + mieszanie bbox-ów detektor/maska.
- Dodano opcję per-frame `unmatched_drop_after`: drop tracków bez matcha do detekcji przez N klatek.

Uwaga: SAM2 (nawet wariant small) na CPU jest bardzo wolny. Próby z checkpointem `large` nie kończą się w rozsądnym czasie bez GPU.

## Iteracje (50 klatek, SNMOT-116)
Wszystkie iteracje: `RF-DETR Base`, `max_side=512`, `max_objects=40`, `drop_after=3`, `reseed_mode=reanchor`.

### Iteracja 1: `sam2_iter_1_50f_small`
- Config: `small`, `reseed_interval=10`, `mask_threshold=0.0`, `mask_open=0`, `mask_close=0`.
- Player: `HOTA=0.6044`, `IDF1=0.7640`, `MOTA=0.6024`, `ID-switch=0`.
- Makro/ważone: `macro_HOTA=0.3240`, `weighted_HOTA=0.5661`, `weighted_IDF1=0.7249`.
- Komentarz: najlepszy balans jak dotąd; bardzo niski `ID-switch`, FP/FN niższe niż ByteTrack/BoT-SORT.

### Iteracja 2: `sam2_iter_2_50f_small_thr02_open3_close5`
- Config: `small`, `reseed_interval=5`, `mask_threshold=0.2`, `mask_open=3`, `mask_close=5`.
- Player: `HOTA=0.5745`, `IDF1=0.7553`, `MOTA=0.6106`, `ID-switch=5`.
- Makro/ważone: `macro_HOTA=0.3127`, `weighted_HOTA=0.5395`.
- Komentarz: post-processing i wyższy próg obniżają HOTA; ustawienia odrzucone.

### Iteracja 3: `sam2_iter_3_50f_small_reseed5`
- Config: `small`, `reseed_interval=5`, `mask_threshold=0.0`, brak morfologii.
- Player: `HOTA=0.5865`, `IDF1=0.7592`, `MOTA=0.6153`, `ID-switch=5`.
- Makro/ważone: `macro_HOTA=0.3232`, `weighted_HOTA=0.5520`.
- Komentarz: częstszy reseed nie poprawił HOTA.

### Iteracja 4: `sam2_iter_4_50f_small_thr01`
- Config: `small`, `reseed_interval=10`, `mask_threshold=0.1`, brak morfologii.
- Player: `HOTA=0.6027`, `IDF1=0.7640`, `MOTA=0.6024`, `ID-switch=0`.
- Makro/ważone: `macro_HOTA=0.3232`, `weighted_HOTA=0.5646`.
- Komentarz: prawie identycznie jak iteracja 1, bez wyraźnego zysku.

### Iteracja 5: `sam2_iter_gpu_retry_50f_large`
- Config: `large`, `reseed_interval=10`, `mask_threshold=0.0`, brak morfologii.
- Player: `HOTA=0.6579`, `IDF1=0.7653`, `MOTA=0.6000`, `ID-switch=0`.
- Makro/ważone: `macro_HOTA=0.3589`, `weighted_HOTA=0.6181`.
- Komentarz: najlepszy wynik do tej pory; SAM2 wygrywa HOTA/IDF1/MOTA vs ByteTrack i BoT-SORT na 50 klatkach.

## Iteracje (750 klatek, SNMOT-116)
Wszystkie iteracje: `RF-DETR Base`, `max_objects=40`, `reseed_mode=reanchor`, GPU.

### Iteracja 6: `sam2_iter_gpu_best_750f_large`
- Config: `large`, `reseed_interval=10`, `mask_threshold=0.0`, brak morfologii, 750 klatek.
- Player: `HOTA=0.4307`, `IDF1=0.5280`, `MOTA=0.5576`, `ID-switch=55`.
- Makro/ważone: `macro_HOTA=0.2177`, `weighted_HOTA=0.3969`.
- Komentarz: na pełnych 750 klatkach SAM2 przegrywa z ByteTrack/BoT-SORT (dużo FP, gorszy HOTA).

### Iteracja 7: `sam2_iter_gpu_best_750f_large_detbox`
- Config: `large`, `reseed_interval=10`, `output_mode=detector`, `output_iou=0.3`.
- Player: `HOTA=0.4444`, `IDF1=0.5450`, `MOTA=0.5909`, `ID-switch=99`.
- Makro/ważone: `macro_HOTA=0.2337`, `weighted_HOTA=0.4124`.
- Komentarz: prawie doganiamy ByteTrack; FP nadal wyższe, ale HOTA bardzo blisko.

### Iteracja 8: `sam2_iter_gpu_750f_large_detbox_iou05`
- Config: `large`, `reseed_interval=10`, `output_mode=detector`, `output_iou=0.5`, `reseed_iou=0.5`.
- Player: `HOTA=0.4247`, `IDF1=0.5069`, `MOTA=0.5360`, `ID-switch=85`.
- Makro/ważone: `macro_HOTA=0.2254`, `weighted_HOTA=0.3951`.
- Komentarz: wyższe IoU pogarsza wynik; odrzucone.

### Iteracja 9: `sam2_iter_gpu_best_750f_large_player_only`
- Config: `large`, `reseed_interval=10`, `allowed_classes=player`.
- Player: `HOTA=0.4306`, `IDF1=0.5282`, `MOTA=0.5578`, `ID-switch=55`.
- Makro/ważone: `macro_HOTA=0.1076`, `weighted_HOTA=0.3668`.
- Komentarz: tylko `player` nie poprawia wyniku całościowego (inne klasy mają zera).

### Iteracja 10: `sam2_iter_gpu_750f_large_detstrict`
- Config: `large`, `reseed_interval=10`, `output_mode=detector_strict`, `output_iou=0.3`.
- Player: `HOTA=0.4221`, `IDF1=0.5062`, `MOTA=0.6634`, `ID-switch=122`.
- Makro/ważone: `macro_HOTA=0.2253`, `weighted_HOTA=0.3941`.
- Komentarz: `detector_strict` mocno redukuje FP, ale zwiększa FN/ID-switch i obniża HOTA.

### Iteracja 11: `sam2_iter_gpu_750f_large_detstrict_tuned`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=15`, `reseed_iou=0.4`, `output_mode=detector_strict`, `output_iou=0.4`.
- Player: `HOTA=0.4344`, `IDF1=0.5366`, `MOTA=0.6647`, `ID-switch=106`.
- Makro/ważone: `macro_HOTA=0.2290`, `weighted_HOTA=0.4047`.
- Komentarz: lekkie odbicie, ale nadal poniżej ByteTrack/BoT-SORT.

### Iteracja 12: `sam2_iter_gpu_750f_large_detbox_tuned`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `output_mode=detector`, `output_iou=0.3`.
- Player: `HOTA=0.4433`, `IDF1=0.5357`, `MOTA=0.5960`, `ID-switch=98`.
- Makro/ważone: `macro_HOTA=0.2342`, `weighted_HOTA=0.4117`.
- Komentarz: bardzo blisko ByteTrack (różnice setne); FP wciąż wysokie.

### Iteracja 13: `sam2_iter_gpu_750f_large_detbox_tuned_drop2`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `drop_after=2`, `output_mode=detector`, `output_iou=0.3`.
- Player: `HOTA=0.4447`, `IDF1=0.5288`, `MOTA=0.6330`, `ID-switch=87`.
- Makro/ważone: `macro_HOTA=0.2350`, `weighted_HOTA=0.4137`.
- Komentarz: **najlepszy wynik na 750 klatkach** – SAM2 minimalnie przebija ByteTrack/BoT-SORT (ważone HOTA).

### Iteracja 14: `sam2_iter_gpu_750f_large_detbox_tuned_drop1`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `drop_after=1`, `output_mode=detector`, `output_iou=0.3`.
- Player: `HOTA=0.4219`, `IDF1=0.4983`, `MOTA=0.6099`, `ID-switch=108`.
- Makro/ważone: `macro_HOTA=0.2243`, `weighted_HOTA=0.3934`.
- Komentarz: zbyt agresywne kasowanie tracków – HOTA wyraźnie spada; odrzucone.

### Iteracja 15: `sam2_iter_gpu_750f_large_detbox_tuned_drop2_reseed5`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=5`, `drop_after=2`, `output_mode=detector`, `output_iou=0.3`.
- Player: `HOTA=0.4349`, `IDF1=0.5106`, `MOTA=0.6188`, `ID-switch=115`.
- Makro/ważone: `macro_HOTA=0.2329`, `weighted_HOTA=0.4057`.
- Komentarz: częstszy reseed obniża HOTA/IDF1; odrzucone.

### Iteracja 16: `sam2_iter_gpu_750f_large_detbox_tuned_drop2_thr02`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `drop_after=2`, `output_mode=detector`, `output_iou=0.3`, `detector_thr=0.2`.
- Player: `HOTA=0.4402`, `IDF1=0.5205`, `MOTA=0.6262`, `ID-switch=100`.
- Makro/ważone: `macro_HOTA=0.2271`, `weighted_HOTA=0.4075`.
- Komentarz: niższy próg detekcji poprawia FN, ale HOTA nadal < 0.5; odrzucone.

### Iteracja 17: `sam2_iter_gpu_750f_large_mask_thr02_open3_close5`
- Config: `large`, `max_side=1280`, `mask_threshold=0.2`, `mask_open=3`, `mask_close=5`, `mask_filter_distance=150`, `reseed_interval=10`, `drop_after=2`, `output_mode=mask`.
- Player: `HOTA=0.4314`, `IDF1=0.5089`, `MOTA=0.5885`, `ID-switch=66`.
- Makro/ważone: `macro_HOTA=0.2204`, `weighted_HOTA=0.3988`.
- Komentarz: maskowe bbox-y z morfologią pogarszają HOTA; odrzucone.

### Iteracja 18: `sam2_iter_gpu_750f_large_detbox_drop2_iou02`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector`, `output_iou=0.2`.
- Player: `HOTA=0.4550`, `IDF1=0.5474`, `MOTA=0.6320`, `ID-switch=96`.
- Makro/ważone: `macro_HOTA=0.2396`, `weighted_HOTA=0.4230`.
- Komentarz: niższe IoU daje najlepszy wynik z dotychczasowych (ale nadal < 0.5).

### Iteracja 19: `sam2_iter_gpu_750f_large_detbox_drop2_iou01`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.1`, `drop_after=2`, `output_mode=detector`, `output_iou=0.1`.
- Player: `HOTA=0.4506`, `IDF1=0.5529`, `MOTA=0.6303`, `ID-switch=115`.
- Makro/ważone: `macro_HOTA=0.2384`, `weighted_HOTA=0.4193`.
- Komentarz: niższe IoU nie poprawia HOTA; gorszy wynik niż iteracja 18.

### Iteracja 20: `sam2_iter_gpu_750f_large_detstrict_iou02`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_strict`, `output_iou=0.2`.
- Player: `HOTA=0.4341`, `IDF1=0.5160`, `MOTA=0.6640`, `ID-switch=129`.
- Makro/ważone: `macro_HOTA=0.2306`, `weighted_HOTA=0.4050`.
- Komentarz: tryb `detector_strict` pogarsza HOTA; odrzucone.

### Iteracja 21: `sam2_iter_gpu_750f_large_detbox_drop2_iou02_reseed1`
- Config: `large`, `max_side=1280`, `mask_filter_distance=150`, `reseed_interval=1`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector`, `output_iou=0.2`.
- Player: `HOTA=0.4205`, `IDF1=0.4845`, `MOTA=0.6723`, `ID-switch=169`.
- Makro/ważone: `macro_HOTA=0.2296`, `weighted_HOTA=0.3940`.
- Komentarz: reseed co klatkę degraduje stabilność ID i HOTA; odrzucone.

### Iteracja 22: `sam2_iter_gpu_750f_large_detbox_drop2_iou02_side1536`
- Config: `large`, `max_side=1536`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector`, `output_iou=0.2`.
- Player: `HOTA=0.4584`, `IDF1=0.5485`, `MOTA=0.6331`, `ID-switch=98`.
- Makro/ważone: `macro_HOTA=0.2408`, `weighted_HOTA=0.4259`.
- Komentarz: większa rozdzielczość minimalnie poprawia HOTA, ale nadal < 0.5.

### Iteracja 23: `sam2_iter_gpu_750f_large_detblend_iou02_alpha07_side1536`
- Config: `large`, `max_side=1536`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.7`.
- Player: `HOTA=0.4677`, `IDF1=0.5465`, `MOTA=0.6412`, `ID-switch=79`.
- Makro/ważone: `macro_HOTA=0.2451`, `weighted_HOTA=0.4344`.
- Komentarz: blend poprawia HOTA względem czystego `detector`.

### Iteracja 24: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536`
- Config: `large`, `max_side=1536`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4697`, `IDF1=0.5447`, `MOTA=0.6386`, `ID-switch=67`.
- Makro/ważone: `macro_HOTA=0.2449`, `weighted_HOTA=0.4358`.
- Komentarz: najlepszy wynik dotychczas (ale wciąż < 0.5).

### Iteracja 25: `sam2_iter_gpu_750f_large_detblend_iou02_alpha03_side1536`
- Config: `large`, `max_side=1536`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.3`.
- Player: `HOTA=0.4668`, `IDF1=0.5407`, `MOTA=0.6271`, `ID-switch=59`.
- Makro/ważone: `macro_HOTA=0.2414`, `weighted_HOTA=0.4324`.
- Komentarz: niższa waga detektora pogarsza HOTA.

### Iteracja 26: `sam2_iter_gpu_750f_large_detblend_iou015_alpha05_side1536`
- Config: `large`, `max_side=1536`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.15`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.15`, `alpha=0.5`.
- Player: `HOTA=0.4689`, `IDF1=0.5540`, `MOTA=0.6377`, `ID-switch=68`.
- Makro/ważone: `macro_HOTA=0.2445`, `weighted_HOTA=0.4350`.
- Komentarz: niższe IoU nie daje wyraźnego zysku względem iteracji 24.

### Iteracja 27: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536_mfd100`
- Config: `large`, `max_side=1536`, `mask_filter_distance=100`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4696`, `IDF1=0.5446`, `MOTA=0.6377`, `ID-switch=67`.
- Makro/ważone: `macro_HOTA=0.2448`, `weighted_HOTA=0.4357`.
- Komentarz: mask_filter_distance=100 nie poprawia HOTA.

### Iteracja 28: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536_thr01`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4698`, `IDF1=0.5451`, `MOTA=0.6405`, `ID-switch=65`.
- Makro/ważone: `macro_HOTA=0.2448`, `weighted_HOTA=0.4358`.
- Komentarz: lekko lepsza MOTA, HOTA nadal < 0.5.

### Iteracja 29: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536_um2_thr01`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `unmatched_drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4431`, `IDF1=0.5079`, `MOTA=0.6728`, `ID-switch=104`.
- Makro/ważone: `macro_HOTA=0.2308`, `weighted_HOTA=0.4147`.
- Komentarz: per-frame drop jest zbyt agresywny – HOTA spada.

### Iteracja 30: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1792`
- Config: `large`, `max_side=1792`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4582`, `IDF1=0.5332`, `MOTA=0.6416`, `ID-switch=69`.
- Makro/ważone: `macro_HOTA=0.2408`, `weighted_HOTA=0.4261`.
- Komentarz: większy `max_side` nie daje poprawy HOTA.

### Iteracja 31: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536_playeronly`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `allowed_classes=player`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4699`, `IDF1=0.5453`, `MOTA=0.6403`, `ID-switch=65`.
- Makro/ważone: `macro_HOTA=0.1175`, `weighted_HOTA=0.4003`.
- Komentarz: minimalnie lepszy `player` HOTA, ale inne klasy spadają do zera (niska ważona HOTA).

### Iteracja 32: `sam2_iter_gpu_750f_large_detblend_iou02_alpha06_side1536`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.6`.
- Player: `HOTA=0.4697`, `IDF1=0.5465`, `MOTA=0.6421`, `ID-switch=74`.
- Makro/ważone: `macro_HOTA=0.2449`, `weighted_HOTA=0.4359`.
- Komentarz: brak zysku względem alpha=0.5.

### Iteracja 33: `sam2_iter_gpu_750f_large_detblend_iou02_alpha05_side1536_reseed5`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=5`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.5`.
- Player: `HOTA=0.4464`, `IDF1=0.5183`, `MOTA=0.6690`, `ID-switch=88`.
- Makro/ważone: `macro_HOTA=0.2333`, `weighted_HOTA=0.4163`.
- Komentarz: częstszy reseed degraduje HOTA (większy chaos ID).

### Iteracja 34: `sam2_iter_gpu_750f_large_detblend_iou01_alpha05_side1536`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.1`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.1`, `alpha=0.5`.
- Player: `HOTA=0.4656`, `IDF1=0.5492`, `MOTA=0.6396`, `ID-switch=68`.
- Makro/ważone: `macro_HOTA=0.2433`, `weighted_HOTA=0.4323`.
- Komentarz: niższe IoU nie poprawia HOTA.

### Iteracja 35: `sam2_iter_gpu_750f_large_detblend_iou02_alpha04_side1536`
- Config: `large`, `max_side=1536`, `mask_threshold=0.1`, `mask_filter_distance=150`, `reseed_interval=10`, `reseed_iou=0.2`, `drop_after=2`, `output_mode=detector_blend`, `output_iou=0.2`, `alpha=0.4`.
- Player: `HOTA=0.4693`, `IDF1=0.5437`, `MOTA=0.6369`, `ID-switch=60`.
- Makro/ważone: `macro_HOTA=0.2446`, `weighted_HOTA=0.4351`.
- Komentarz: brak poprawy względem alpha=0.5.

## Wnioski (stan obecny)
- Najlepsza konfiguracja na 750 klatkach to iteracja 24/28: `large + max_side=1536 + output_mode=detector_blend + alpha=0.5 + drop_after=2 + reseed_iou=0.2`.
- Najwyższy `player` HOTA to iteracja 31 (0.4699), ale tylko po ograniczeniu do klasy `player` (spada ważona HOTA).
- SAM2 ma wyższą ważoną HOTA od ByteTrack/BoT-SORT, ale **player HOTA nadal < 0.5** (ok. 0.47).
- Największy zysk dało mieszanie bbox-ów detektora i maski oraz większa rozdzielczość wejścia.

## Następne kroki (jeśli dostępna GPU)
1) Sprawdzić stabilność wyniku z iteracji 24 na innej sekwencji (np. 1–2 dodatkowe SNMOT).
2) Dalsze strojenie: `output_iou ∈ {0.25}`, `mask_filter_distance ∈ {200}`, `drop_after ∈ {3}`.
3) Jeśli HOTA nadal < 0.5, rozważyć lepszy detektor (fine-tuning RF-DETR na SNMOT) lub hybrydę ByteTrack IDs + SAM2 mask refinement.

## Proxy baseline (200 klatek, GPU) – punkt wyjścia do dalszych iteracji
Sekwencja: `SNMOT-116`, `max_frames=200`. SAM2: `large`, `max_side=1536`, `detector_blend` (`alpha=0.5`).

### Base (RF-DETR Base, pełny checkpoint)
- Wynik SAM2 (player): `HOTA=0.5440`, `IDF1=0.6588`, `MOTA=0.5897`, `ID-switch=26`, `FP=186`, `FN=1167`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5271`, `weighted_MOTA=0.5611`.

### Seg (RF-DETR Seg, **tiny checkpoint**)
- Wynik SAM2 (player): `HOTA=0.4788`, `IDF1=0.5723`, `MOTA=0.5126`, `ID-switch=31`, `FP=598`, `FN=1009`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4300`, `weighted_MOTA=-0.0263`.
- Uwaga: pełny checkpoint Seg nie był dostępny; użyto `results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg/models/rfdetr_seg_soccernet2023.pth`.

## Iteracja: per-class thresholds (Idea 1)
Ustawienia: `player=0.25`, `goalkeeper=0.25`, `referee=0.3`, `ball=0.15` (reszta = global).

### Base (RF-DETR Base)
- Wynik SAM2 (player): `HOTA=0.5608`, `IDF1=0.6953`, `MOTA=0.5986`, `ID-switch=16`, `FP=136`, `FN=1197`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5414`, `weighted_MOTA=0.5687`.
- Wniosek: wyraźna poprawa HOTA/MOTA i spadek FP/ID-switch względem baseline.

### Seg (RF-DETR Seg tiny)
- Wynik SAM2 (player): `HOTA=0.4948`, `IDF1=0.5937`, `MOTA=0.5370`, `ID-switch=33`, `FP=456`, `FN=1067`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4438`, `weighted_MOTA=0.0019`.
- Wniosek: poprawa HOTA/MOTA i wyjście z ujemnej ważonej MOTA.

## Iteracja: NMS=0.4 (Idea 2)
Ustawienia: `detector_nms_threshold=0.4` + per-class thresholds z Idei 1.

### Base (RF-DETR Base)
- Wynik SAM2 (player): `HOTA=0.5572`, `IDF1=0.6781`, `MOTA=0.5948`, `ID-switch=17`, `FP=137`, `FN=1208`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5383`, `weighted_MOTA=0.5654`.
- Wniosek: HOTA spada vs Idea 1, NMS=0.4 odrzucony.

### Seg (RF-DETR Seg tiny)
- Wynik SAM2 (player): `HOTA=0.4828`, `IDF1=0.5883`, `MOTA=0.5364`, `ID-switch=29`, `FP=409`, `FN=1120`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4336`, `weighted_MOTA=0.0014`.
- Wniosek: brak poprawy; NMS=0.4 odrzucony.

## Iteracja: reseed skip IoU=0.6 (Idea 3)
Ustawienia: `sam2_reseed_skip_iou=0.6` + per-class thresholds z Idei 1.

### Base (RF-DETR Base)
- Wynik SAM2 (player): `HOTA=0.5541`, `IDF1=0.6819`, `MOTA=0.5980`, `ID-switch=16`, `FP=137`, `FN=1198`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5357`, `weighted_MOTA=0.5720`.
- Wniosek: minimalny spadek HOTA (base), zysk w MOTA – nie trzymamy.

### Seg (RF-DETR Seg tiny)
- Wynik SAM2 (player): `HOTA=0.4989`, `IDF1=0.5992`, `MOTA=0.5409`, `ID-switch=34`, `FP=449`, `FN=1060`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4473`, `weighted_MOTA=0.0052`.
- Wniosek: lekka poprawa względem Idei 1 (seg), ale efekt mały.

## Iteracja: per-class blend alpha (Idea 4)
Ustawienia: `alpha_by_class={player:0.5,goalkeeper:0.6,referee:0.8,ball:0.9}` + per-class thresholds z Idei 1.

### Base (RF-DETR Base)
- Wynik SAM2 (player): `HOTA=0.5608`, `IDF1=0.6953`, `MOTA=0.5986`, `ID-switch=16`, `FP=136`, `FN=1197`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5409`, `weighted_MOTA=0.5687`.
- Wniosek: praktycznie bez zmian vs Idea 1; odrzucone.

### Seg (RF-DETR Seg tiny)
- Wynik SAM2 (player): `HOTA=0.4948`, `IDF1=0.5937`, `MOTA=0.5370`, `ID-switch=33`, `FP=456`, `FN=1067`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4422`, `weighted_MOTA=0.0019`.
- Wniosek: brak poprawy.

## Iteracja: filtr geometrii bbox (Idea 5)
Ustawienia: per-class `min_area_ratio`/`max_area_ratio`/`max_aspect`:
`player=0.0004/0.2/8`, `goalkeeper=0.0004/0.2/8`, `referee=0.0004/0.2/8`, `ball=0.00002/0.01/3`.

### Base (RF-DETR Base)
- Wynik SAM2 (player): `HOTA=0.5660`, `IDF1=0.6979`, `MOTA=0.6049`, `ID-switch=16`, `FP=115`, `FN=1197`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.5458`, `weighted_MOTA=0.5740`.
- Wniosek: najlepszy wynik na proxy 200f – spadek FP i wzrost HOTA/MOTA.

### Seg (RF-DETR Seg tiny)
- Wynik SAM2 (player): `HOTA=0.5004`, `IDF1=0.5968`, `MOTA=0.5466`, `ID-switch=33`, `FP=424`, `FN=1067`.
- Wynik SAM2 (ważone): `weighted_HOTA=0.4487`, `weighted_MOTA=0.0156`.
- Wniosek: wyraźna poprawa względem Idei 1 i 3; redukcja FP.

## Odkrycia i dalsze kierunki (podsumowanie)
### Co zadziałało
- `output_mode=detector_blend` podnosi HOTA względem `detector` i `mask` (największy pojedynczy zysk).
- Wyższa rozdzielczość wejścia do SAM2 (`max_side=1536`) daje mały, ale stabilny wzrost HOTA.
- Umiarkowane IoU (`output_iou=0.2`) jest najlepszym kompromisem; zbyt niskie i zbyt wysokie wartości pogarszają HOTA.
- Per-class thresholds poprawiają FP/ID-switch (Idea 1).
- Filtr geometrii bbox (min/max area + max aspect) poprawia HOTA/MOTA i redukuje FP (Idea 5).

### Co nie zadziałało
- Agresywny per-frame drop (`unmatched_drop_after=2`) obniża HOTA (za duża utrata tracków).
- Częstszy reseed (`reseed_interval=5`) pogarsza stabilność ID i HOTA.
- Dalsze zwiększenie rozdzielczości (`max_side=1792`) nie daje zysku względem 1536.
- Ograniczenie `allowed_classes=player` poprawia jedynie HOTA dla `player`, ale znacząco obniża metryki ważone.
- Niższy NMS (`nms=0.4`) obniża HOTA (Idea 2).
- Per-class blend alpha nie daje zysku (Idea 4).
- Reseed skip `iou=0.6` pogarsza base (Idea 3).

### Bottleneck i hipotezy
- SAM2 traci stabilność ID w dłuższych sekwencjach przez brak jawnego modelu motion/association (rola detektora i reseedingu jest kluczowa).
- FP/FN są bardziej związane z jakością detekcji niż z samą segmentacją; dalsze mikro‑strojenia SAM2 mają ograniczoną dźwignię.
- Problemem jest długookresowa spójność tracków, nie krótkoterminowa jakość masek.

### Potencjalne dalsze kroki (wysoki wpływ)
1) Fine-tuning detektora na SNMOT (RF‑DETR), potem ponowne strojenie SAM2.
2) Hybryda: ByteTrack IDs + SAM2 mask refinement (utrzymać stabilne ID z ByteTrack, poprawić bbox/mask SAM2).
3) Dodatkowe filtrowanie tracków po stronie SAM2:
   - filtr min/max area per klasa,
   - filtr aspect ratio,
   - filtr krótkich tracków (np. min długość 3–5 klatek).
4) Reseed adaptacyjny:
   - reseed tylko gdy IoU spada poniżej progu,
   - wydłużony `reseed_interval` przy stabilnych trackach.
5) Test na kolejnych sekwencjach SNMOT:
   - sprawdzić, czy plateau wynika ze specyfiki SNMOT-116.

### Kolejność rekomendowanych prób
1) Detektor: fine‑tuning lub inny checkpoint (największa szansa na skok HOTA).
2) Hybryda ByteTrack + SAM2 (ID stabilność + lepsze maski/bbox).
3) Filtry geometryczne i minimalna długość tracków.

## GPU status (bieżące środowisko)
GPU dostępna i używana (`RTX 4070 Ti SUPER`). SAM2 działa na GPU, ale natywne rozszerzenie nadal się nie ładuje (`_C.so`), więc post-processing masek jest wolniejszy (Python CC).

Gdy GPU będzie widoczne, uruchom:
```bash
XDG_CONFIG_HOME=$PWD/.config \
ULTRALYTICS_CONFIG_DIR=$PWD/.ultralytics \
.venv/bin/python scripts/run_soccernet_train2seq_infer1seq.py \
  --skip-training \
  --detectors base \
  --infer-root data/soccernet/tracking/extracted/test \
  --infer-sequence-index 0 \
  --max-frames 50 \
  --preview-frames 4 \
  --results-dir results/detection_tracking/raw/sam2_iter_gpu_50f_large \
  --base-checkpoint results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking/models/rfdetr_base_soccernet2023.pth \
  --sam2-checkpoint external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-max-side 1024 \
  --sam2-max-objects 40 \
  --sam2-reseed-interval 10 \
  --sam2-reseed-mode reanchor \
  --sam2-drop-after 3 \
  --sam2-mask-threshold 0.0 \
  --sam2-mask-open 0 \
  --sam2-mask-close 0
```
I ewaluacja + wykresy:
```bash
.venv/bin/python scripts/evaluate_existing_soccernet_inference.py \
  --results-dir results/detection_tracking/raw/sam2_iter_gpu_50f_large \
  --extracted-root data/soccernet/tracking/extracted/test \
  --sequence SNMOT-116 \
  --split-name test \
  --skip-timing \
  --max-frames 50
```
