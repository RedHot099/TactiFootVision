# detector_data: open detection sources for the 4-class detector (detector-data 03)

Owner rule (2026-10-14): collect everything, tag, split later. Converters never drop an image for leak reasons; they
tag it. A training split is a filter on `pool_index.parquet`.

Data root: `/mnt/storage/tactifoot_research/detector_data/`

- `raw/<source>/`: downloads (untrusted; run readers with `python -I`);
- `work/<source>/`: re-encoded images (PNG -> JPEG, ignore regions greyed);
- `sets/<set>/`: YOLO export (`data.yaml`, `<split>/images|labels`), `manifest.json`, `leak_report.json`,
  `label_sha256.txt`;
- `leak_refs/`: cached fingerprints of the benchmark frames (all GSR valid + test frames on disk, football_yolo valid);
- `pool_index.parquet`: one row per image of every set.

Run a converter: `uv run python -I -c "import sys; sys.path.insert(0, '<repo>'); from scripts.detector_data import <name> as m; m.main()"`
(the `-I` form keeps the data folders off `sys.path`). Then `... pool_index as m; m.main()` rebuilds the index.

## Classes

`ball, goalkeeper, player, referee` (ids 0-3). Every set records `annotated_classes`: the classes its source labels
exhaustively. The others are **not annotated** for that source and must be masked out of the loss, never treated as
background.

## Pool-index schema (contract for every converter, the GSR/Tracking ones included)

A converter calls `common.export_set(dataset, sets/<set>, manifest, annotated, tags, report)`:

- `manifest` needs `source` (family id), `licence_tag` (short), `licence` and `access_route` (long text);
- `tags` maps each source image path to values for the tag columns (`game`, `clip`, `frame`, `image_type`,
  `xg_role`, `gsr_split`, `near_dup_*`); missing ones are null; any other key is refused;
- the exporter fills `set`, `source`, `split`, `image`, `source_image`, size, licence, synthetic,
  `annotated_classes` and the per-class counts.

The manifest's `per_image` list holds one row per image with exactly these keys; `pool_index.py` refuses any row
that misses or adds one.

| column | meaning |
|---|---|
| `set` | set folder name under sets/ (one per source or source part) |
| `source` | source family, e.g. soccernet_v3, roboflow, soccersynth, sportsmot, gsr |
| `split` | split inside the set's YOLO export (the source's own split, not a training decision) |
| `image` | absolute path of the exported image (sets/<set>/<split>/images/...); its label is the YOLO txt beside |
| `source_image` | path of the image the export links to (raw or work copy) |
| `width` | image width, px |
| `height` | image height, px |
| `game` | game identity: SoccerNet game path, GSR game_id as 'gsr:<id>', or null when unknown |
| `clip` | clip / video id, or null |
| `frame` | frame index inside the clip, or null |
| `image_type` | source-specific kind (v3: action / replay; soccersynth: clear / motion_blur), or null |
| `xg_role` | plan.csv role of the game (training / evaluation / gate), null if not a plan game |
| `gsr_split` | GSR split of the game or of the matched benchmark frame (train / valid / test / challenge), or null |
| `near_dup_of` | benchmark frame this picture duplicates ('gsr:<split>/<clip>/<frame>' or 'football_yolo_valid:<file>'), or null |
| `near_dup_set` | gsr_valid / gsr_test / football_yolo_valid (benchmark), or soccernet_v3 (sources of unknown provenance are also matched against every v3 frame, so v3 game and xg_role carry over), or null |
| `near_dup_corr` | thumbnail correlation of the best near-dup hit, or null |
| `near_dup_hits` | number of confirmed benchmark near-dup pairs |
| `licence` | short licence tag of the source |
| `synthetic` | rendered rather than real footage |
| `annotated_classes` | comma-separated classes the source labels exhaustively; the others are not annotated and must be masked out of the loss (never treated as background) |
| `n_ball` | ball boxes in the label |
| `n_goalkeeper` | goalkeeper boxes |
| `n_player` | player boxes |
| `n_referee` | referee boxes |

GSR/Tracking rows come from the other converter's `pool_index_gsr.parquet` (research/detector-data 2ad8f07), mapped
in `pool_index.py` (`GSR_COLUMNS`): `game_id` -> `game`, `image_target` -> `source_image`, `image_source`
(gsr / tracking) -> `image_type`, `split` = `gsr_split`; its extra columns (label, stride_group, n_other_dropped)
stay in that file.

Conventions: GSR rows use `game = "gsr:<game_id>"`, `clip = "SNGS-NNN"`, `frame` = frame number and
`gsr_split` = the clip's split. SoccerNet-v2 games use the game path (`league/season/date - teams`), which is the
key of the xG plan (`xg_role`).

## Tags

- `xg_role`: plan.csv role of the game (training / evaluation / gate; the 172 evaluation + gate games are the xG
  held-out set).
- `near_dup_of`: dHash <= 4 bits **and** 64x36 grey-thumbnail Pearson >= 0.95 against every GSR valid/test frame on
  disk and every football_yolo valid image; the best hit is recorded (`gsr:<split>/<clip>/<frame>` or
  `football_yolo_valid:<file>`), with `gsr_split` for a GSR hit.
- Thinning: only consecutive near-duplicate video frames inside one set may be removed (counted in the manifest as
  `near_duplicate_frames_thinned`); `--no-thin` keeps them.

## Default training view

```python
pool = pd.read_parquet(".../pool_index.parquet")
view = pool[~pool.xg_role.isin(["evaluation", "gate"]) & ~pool.gsr_split.isin(["valid", "test"]) & pool.near_dup_of.isna()]
```

## Sources (one set each)

soccernet_v3 (+ v3D ball refinement), roboflow_players, roboflow_ball, soccer_player_detection, issia, teamtrack,
sportsmot_football, soccana, soccersynth (synthetic), football_yolo (the baseline's Roboflow set: Ekstraklasa LiveTag
clips un-stretched to 16:9 + three other families; its `test` split is the Ekstraklasa dev holdout, whole clips, record
`research/ekstraklasa_dev_split.json`, YOLO copy `bench/ekstraklasa_dev`; views must leave it out), plus soccernet_gsr
from the GSR converter. Blocked (owner action):
SoccerTrack v2 (Hugging Face gate), SoccerDB (Google Drive links dead, Baidu only).
