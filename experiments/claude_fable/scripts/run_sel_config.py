"""Run only base + selected-features config of the ablation into a separate out dir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_feature_ablation as rfa

rfa.CONFIGS = {"base": [], "sel": rfa.CONFIGS["sel"]}
sys.argv = ["run_sel_config.py", "--out", "results/feature_ablation_sel"]
rfa.main()
