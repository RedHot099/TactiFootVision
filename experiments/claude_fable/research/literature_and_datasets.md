# xG z obrazu: literatura i dane (research do projektu PoC)

Data: 2026-08-23. Autor: Claude Fable (worktree `claude-fable/xg-poc`).

## Streszczenie (PL)

1. Klasyczne xG z danych eventowych (dystans, kąt, część ciała, typ asysty) daje na
   strzałach otwartej gry log-loss rzędu 0.27-0.30, Brier ok. 0.08 i AUC 0.75-0.80.
   Dodanie freeze-frame'u (pozycje obrońców i bramkarza) poprawia log-loss o ok.
   0.01-0.02. Bazowa częstość gola to ok. 10-11%, więc Brier "stałej predykcji" to ok. 0.09.
   To jest skala, w której poruszają się wszystkie modele; różnice między
   architekturami są małe i trzeba je mierzyć ostrożnie (split po meczach, CI bootstrap).
2. Modele głębokie na pozycjach (GNN, Set Transformer, rastry "SoccerMap") wygrywają z
   tabelarycznymi tylko przy dużych zbiorach (dziesiątki tysięcy strzałów) i głównie
   dzięki lepszemu opisowi obrony. Przy kilku tysiącach strzałów gradient boosting
   na cechach inżynierskich jest trudny do pobicia.
3. Nie znalazłem recenzowanej pracy, która przewiduje xG wprost z klatki broadcastu.
   Najbliższe: predykcja *wystąpienia* strzału z wideo (grafy dwudzielne graczy,
   PMC10181557), action spotting SoccerNet (klasy Shots on/off target, Goal), oraz
   w koszykówce predykcja trafienia z obrazu trajektorii. Podejście end-to-end z obrazu
   jest więc otwartym pytaniem badawczym, a nie ugruntowaną metodą.
4. W SoccerNet frozen backbone'y (ResNet/ DINOv2/ VideoMAE-klasy) plus lekka głowa
   dominują w action spotting, bo danych jest za mało na pełny fine-tuning.
   To sugeruje, że wariant "zamrożony foundation model + mała głowa" powinien być jednym z PoC.
5. Dane: SoccerNet v2 (500 meczów, etykiety Shots on/off target + Goal, 720p za hasłem
   NDA) jest jedynym dużym źródłem "klatka + wynik strzału". SoccerTrack v2 (10 meczów,
   panoramiczne 4K, etykiety BAS z klasami Shot i Goal oraz GSR z pozycjami 2D na boisku)
   daje dodatkowo pozycje ground-truth, co pozwala odseparować błąd detekcji/homografii
   od błędu modelu xG. SoccerReplay-1988 (1988 meczów, klasy goal/shot off target/saved)
   jest większy, ale dostęp po NDA trwa tygodnie.

## 1. Classical xG from event / freeze-frame data

Feature sets and reported quality:

- StatsBomb's xG uses shot location, body part, shot type, and freeze-frame features:
  goalkeeper position, defenders between ball and goal, "pressure". Every shot has a
  freeze frame attached. Source: https://blogarchive.statsbomb.com/news/statsbomb-data-case-studies-freeze-frames-and-defender-locations/
- "Expected goals in football: Improving model performance and demonstrating value"
  (Mead, O'Hare, McMenemy; PLOS One 2023) compares logistic regression, random forest,
  gradient boosting and NNs on event-level features; tree ensembles give the best
  Brier/log-loss and the gains from richer features exceed gains from model class.
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0282295
- "Explainable expected goal models" (Cavus & Biecek 2022) reports log-loss ~0.28-0.30
  and AUC ~0.78-0.80 for gradient boosting on Wyscout-style features.
  https://arxiv.org/pdf/2206.07212
- "Biases in Expected Goals Models Confound Finishing Ability" (Davis et al. 2024)
  warns that models overfit to shooter identity / league; per-match splits needed.
  https://arxiv.org/pdf/2401.09940

Takeaway for us: the tabular approach is the reference point; any image-based model
must at least match gradient boosting on distance/angle/defender features.

## 2. Deep models on positions (graph / set / raster)

- SoccerMap (Fernández & Bornn, ECML 2020): fully convolutional net on a pitch raster
  (l x h x c channels: player positions, velocities, distances to goal) that outputs
  probability surfaces; trained on tracking data at 10 fps. Architecture transfers to
  xG by reading the surface at the ball location. https://arxiv.org/abs/2010.10202
- USSF GNN (MIT SSAC 2023): player/ball nodes with position, velocity, distance/angle
  to goal and ball, team flag; Delaunay / dense edges; Spektral GCN; 20,863 event
  graphs. Evaluated with log-loss, ROC-AUC and calibration.
  https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn
- unravelsports implements the Bekkers & Sahasrabudhe (2023) graph conversion for
  tracking frames and StatsBomb 360 freeze frames, used at Hudl/StatsBomb conferences
  for xG and xPass 360. https://github.com/UnravelSports/UnravelSports,
  https://www.hudl.com/blog/xpass-360-upgrading-expected-pass-xpass-models
- Hierarchical Bayesian counterfactual xG (2025) uses StatsBomb 360 freeze frames;
  confirms freeze-frame defender geometry is the main signal beyond location.
  https://arxiv.org/pdf/2511.23072

Takeaway: a permutation-invariant model over player tokens (Set Transformer or GNN)
with ball-relative and goal-relative features is the standard deep design. A pitch
raster + small CNN (SoccerMap style) is a cheap, interpretable alternative.

## 3. End-to-end from video

- Shooting *event* prediction from broadcast video via complete bipartite graphs of
  players' spatio-temporal relations (PMC10181557, 2023). Predicts that a shot will
  happen, not its outcome. https://pmc.ncbi.nlm.nih.gov/articles/PMC10181557/
- Action spotting survey (Giancola et al. 2024): frame-level features from frozen
  backbones plus temporal heads (NetVLAD++, E2E-Spot) are the SoccerNet baseline.
  https://arxiv.org/html/2410.01304v1
- Basketball: CNN over multi-agent trajectory images for shot make/miss (error 39%),
  https://arxiv.org/pdf/1609.04849; video-based CNNs for shooting accuracy
  (ScienceDirect 2025). These show image-encoded context can predict outcomes but all
  use clean tracking or fixed cameras.

Takeaway: nobody has published "broadcast frame -> xG". Expect the end-to-end model
to be data-hungry and sensitive to camera cuts / replays at the shot moment.

## 4. Vision foundation models in football

- SoccerNet challenge results (2024-2026) consistently use frozen DINOv2 / CLIP /
  VideoMAE features with light heads; fine-tuning full backbones on ~500 games
  rarely helps. https://arxiv.org/pdf/2607.07320
- MatchVision / SoccerReplay-1988 (arXiv 2412.01820) trains a soccer-specific video
  encoder on 1988 matches with event classes including goal, shot off target,
  saved by goalkeeper; weights are released and could be a drop-in backbone.
  https://arxiv.org/abs/2412.01820

## 5. Datasets with shot frames + outcome

| Dataset | Video | Shot/outcome labels | Access |
|---|---|---|---|
| SoccerNet v2 Action Spotting | 500 games, 25 fps, 224p/720p mkv | `Shots on target`, `Shots off target`, `Goal`, `Penalty` with ms timestamp, team, visibility | labels free (`pip install SoccerNet`), videos need NDA password. https://www.soccer-net.org/tasks/action-spotting |
| SoccerNet Ball Action Spotting 2025 | 7+2 EPL/EFL games, 720p | 12 classes incl. `Shot`, `Goal`, dense | NDA password, HF SN-BAS-2025 |
| SoccerTrack v2 | 10 university matches, panoramic 4K full pitch | BAS 12 classes incl. `Shot`, `Goal`; GSR: 2D pitch coords, IDs, roles, teams | HF `atomscott/soccertrack-v2` (gated), GitHub AtomScott/SoccerTrack-v2, CC BY 4.0 paper. https://arxiv.org/abs/2508.01802 |
| SoccerReplay-1988 | 1988 matches, 360-720p | goal, shot off target, saved by GK, penalty, penalty missed | NDA form, weeks of review |
| SoccerNet-v3 | 34k still frames at v2 action timestamps | linkable to v2 shot/goal label | ungated MIT mirror on HF (Voxel51/SoccerNet-V3, 7 GB subset) |
| StatsBomb open data | none | shots with `shot_statsbomb_xg`, 360 freeze frames | free; use as synthetic-render / pretraining source |

Not usable (no shot outcome): DFL Bundesliga Shootout, SportsMOT, SoccerNet GSR,
SoccerNet Caption / Echoes / MatchTime.

Known pitfalls:

- SoccerNet label precision is ~1 s; at the shot moment broadcast may cut to a
  close-up or replay. Expect 20-40% of frames to be unusable; need a "pitch visible"
  filter (grass ratio, keypoint count) and multi-offset extraction.
- `Goal` is a separate label from the shot; a goal's shot annotation may be absent.
  Our parser joins them (same team, <= 6 s) and synthesises a shot event otherwise.
- Class imbalance ~10% goals; use log-loss/Brier, not accuracy; calibrate on valid.
- Split by game, never by shot; same-game frames share camera, kit and lighting.

## 6. Evaluation protocol adopted

Brier, log-loss, AUC, ECE on a per-game test split; Brier skill score vs constant
base rate; bootstrap CI over games; calibration via temperature scaling on valid.

## 7. Implications for the PoC set

1. Tabular xG on extracted positions (reference; must be beaten).
2. Set Transformer / GNN on player tokens from the same positions.
3. End-to-end CNN/ViT fine-tuned on the raw frame.
4. Frozen foundation model (DINOv2 / SigLIP) patch tokens + attention-pooling head.
5. Hybrid: detected-object tokens (crop embedding + box + pitch coords) fused with
   global image tokens in one small transformer.

Variants worth running as ablations, not separate PoCs: SoccerMap-style raster CNN
(inside 2), short clip instead of single frame (inside 3/4), ground-truth positions
from SoccerTrack v2 GSR instead of detector output (inside 1/2) to isolate the
detection/homography error.
