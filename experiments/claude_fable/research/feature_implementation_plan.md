# Plan implementacji cech (pakiet P0+P1 z burzy mózgów)

Data: 2026-08-31. Baza ablacji: D DINOv2-L @448 (ens AUC 0.741) + GBM pozycyjny
(0.615). Zasada nadrzędna: każda cecha wchodzi osobno, z flagą braków, i zostaje
tylko jeśli nie pogarsza bazy (3 seedy, CI po meczach).

## Decyzje projektowe (wspólne)

- **Zero ponownej ekstrakcji GPU dla P0**: wszystkie cechy P0 liczymy z już
  zbudowanego cache'a (`cache/full_v2_fallback/positional/`): tokeny (pozycje
  kanoniczne + drużyny + GK) i boxy obrazowe. Nowy moduł `xgpoc/features/aux.py`
  buduje tabelę `aux.parquet` (shot_id + cechy + flagi *_valid).
- **Światło bramki w przestrzeni boiska**, nie obrazu: 32 keypointy nie zawierają
  słupków (bezimienne przecięcia linii), więc wariant obrazowy wymagałby nowego
  detektora. Wersja gaussowska wg StatsBomb 2022: każdy zasłaniający rzutuje na
  linię bramki (7.32 m) cień o szerokości kątowej ciała 0.8 m (GK 1.6 m), rozmyty
  Gaussem; occlusion = 1 − iloczyn (1 − pokrycie). Flaga ważności = piłka+homografia
  +drużyny znane.
- **Naprawa bugu obrońców** nie rusza `tabular.py` (zachowujemy porównywalność
  starych wyników); poprawna logika żyje w `aux.py`: obrońca tylko przy znanej
  drużynie, osobna flaga `teams_known`.
- **Wejście do modelu D**: `AttentionPoolHead` dostaje opcjonalny wektor aux
  konkatenowany do zbiorczej reprezentacji przed MLP (LayerNorm na aux). Braki
  zerowane + flaga. Do GBM: merge aux po shot_id z istniejącymi 20 cechami.

## P0.1 `is_penalty` (plan)

1. Źródło: kolumna `is_penalty` z indexu v1.1 (146 karnych; 100% pokrycia).
2. Wejście: skalar 0/1 do aux; do GBM jako 21. kolumna.
3. Walidacja: log-loss globalny + średnie xG karnych (oczekiwane ~0.6-0.76 zamiast
   0.42-0.46); AUC bez zmian lub ↑.
4. Ryzyko wdrożeniowe (odnotowane w karcie): w produkcji flaga musi pochodzić z
   detekcji sytuacji lub zewnętrznego feedu.

## P0.2 Światło bramki (plan)

1. Wejścia z tokenów (kanonicznych): piłka (x,y), przeciwnicy `is_defending=1`,
   GK z klasy detektora.
2. Geometria: dla zasłaniającego P między piłką B a linią bramki (x_P > x_B,
   rzut kątowy z B): środek cienia = przecięcie promienia B→P z x=105; szerokość
   cienia = w_body × (105 − x_B)/(x_P − x_B) (skala perspektywiczna), w_body=0.8 m
   (1.6 m GK). Pokrycie linii [GOAL_Y±3.66] sumowane po siatce 1 cm z profilem
   gaussowskim (σ = szer/2), łączone 1−Π(1−c).
3. Cechy: `occlusion` (0-1), `occlusion_gk` (sam GK), `n_shadow_players`,
   `occl_valid`.
4. Testy jednostkowe: obrońca dokładnie na linii strzału blisko piłki → occlusion
   wysoka; nikt między → 0; obrońca za piłką → 0.
5. Walidacja: ΔAUC/Δlog-loss GBM i D; sanity: korelacja ujemna occlusion↔goal na
   podzbiorze valid z occl_valid=1.

## P0.3 GK-proxy + poprawne cechy obrońców (plan)

1. GK-proxy = gracz drużyny broniącej o max x (najgłębszy); jeśli klasa GK z
   detektora obecna — ona ma priorytet.
2. Cechy: `gkp_depth` (105 − x), `gkp_lateral` (odległość od prostej piłka-środek
   bramki), `gkp_dist_ball`, `gkp_valid`; poprawione `def_in_cone_fixed`,
   `nearest_def_fixed` (tylko przy znanych drużynach) + `teams_known`.
3. Walidacja jak wyżej; oczekiwanie: pokrycie cech GK 20%→~55%.

## P1.1 Piłka w powietrzu / proxy główki (plan)

1. Z cache'owanych boxów obrazowych: box piłki (cls=0) i gracze w promieniu
   2× wysokości gracza; `ball_in_air` = środek piłki powyżej górnej ćwiartki
   najbliższego boxu gracza; `ball_air_valid` = piłka w boxach.
2. Podwójna rola: cecha + flaga jakości `ball_xy` (piłka w locie łamie założenie
   płaszczyzny boiska).
3. Test: syntetyczne boxy.

## P1.2 Prędkość ataku (plan — osobny etap, wymaga GPU ~30 min)

1. Ekstrakcja pozycji (detektor+homografia) dla klatek −1600 i −800 ms → lekki
   cache centroidów drużyn i piłki.
2. Cecha: przesunięcie centroidu graczy (m/s) w osi x kanonicznej między −1600 a
   −800; wyłącznie pary bez t0 (bramka przeciw przeciekowi wyniku przy ~1 s szumie
   anotacji); `speed_valid` = homografia OK na obu klatkach + brak cięcia kamery
   (kontrola: przesunięcie < 15 m/s).
3. Wchodzi do ablacji dopiero po przejściu P0.

## Harmonogram ablacji

Konfiguracje: base / +pen / +occl / +gk / +air / +all(P0+P1.1) — każda 3 seedy
D(DINOv2-L, głowa 40 s) i GBM. Skrypt `scripts/run_feature_ablation.py` → tabela
z ens AUC/BSS/log-loss + CI i xG karnych. Czas łączny ~45 min GPU.

Kryteria akceptacji: cecha zostaje w finalnym zestawie tylko gdy Δlog-loss ≤ 0
(nie pogarsza) i poprawia metrykę celu (globalną lub kieszeń, dla której powstała).
