# SynLoc Author Baseline Sources

- Official baseline repo: `https://github.com/Spiideo/mmpose.git`
- Official baseline branch: `spiideo_scenes`
- Pinned baseline commit: `bf1b4401a0f12b1f7d2a2e007d287e0f26ca789e`
- Official devkit repo: `https://github.com/Spiideo/sskit.git`
- Pinned devkit commit: `9e28ad1bdc9b5a79deb82c337eeaa19f481b415e`
- Target baseline config: `configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py`
- Public init checkpoint: `https://download.openmmlab.com/mmpose/v1/pretrained_models/yolox_m_8x8_300e_coco_20230829.pth`
- Final SynLoc checkpoint: available behind `research.spiideo.com` login, not as a public direct URL
- Training fallback: train from `configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py` and use `work_dirs/yoloxpose_m_4xb64-300e_960/epoch_300.pth`
- Paper reference: `mAP-LocSim 79.3`
- Leaderboard-adjusted reference after the sskit bug fix: `mAP-LocSim 76.17`
