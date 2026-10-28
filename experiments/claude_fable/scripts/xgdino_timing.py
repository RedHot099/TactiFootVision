"""Inference-time benchmark of A (G-combo), B (automatic geometry) and C (both) with real weights (xg-dino ticket 03,
council C1 "Timing"). The paths and their stages: `xgpoc/dino/timing.py`.

    ./run.sh --no-sync python scripts/xgdino_timing.py --check-only        # CPU: sample, designs, feature-path check
    ./run.sh --no-sync python scripts/xgdino_timing.py --budget 420        # GPU, resumable: run until done
    ./run.sh --no-sync python scripts/xgdino_timing.py --finish            # CPU: timing.json, figure, complete.json

Sample: `--shots` (100) training shots of the 7,062-shot population, drawn with `--seed` (0), in shot-id order; the
batch-16 configurations use the first 96 (six full batches). Configurations (`CONFIGS`): A at fp32 / fp16 (autocast),
batch 1 / 16, 1 tail (fold 0 seed 0) / 10 tails (the deployed G-combo); B at fp32 (the frozen configuration: YOLO in
fp32, SigLIP in its built-in fp16 autocast) and fp16 (YOLO detector and keypoint model with `half=True`; not the frozen
configuration, its features can differ); B runs one frame at a time (the configured detector batch); C at fp32 / fp16,
batch 1 / 16, 10 tails, measured directly: one decode per shot shared by A and B.

Timing: a unit is one repetition of one configuration over the whole sample; units run repetition by repetition, every
configuration inside a repetition, so slow load phases spread over all configurations. Before its first unit in a
process a configuration is warmed up on 3 shots (1 batch). Every stage is CUDA-synchronised. Per unit: the end-to-end
time of every shot (batch 1) or batch (batch 16, divided by 16), the per-shot mean of every stage, the load average.
Frames are read from the page cache after the first repetition (warm decode).

Resume: units go to <out>/parts/units.jsonl as they finish; a run stops starting units after `--budget` seconds and a
re-run continues; parts/identity.json pins the sample, configurations and code. Every session records the GPU state
before the models load (idle check), uptime before and after. Checks (first session): the live B features of the fp32
analyser against the frozen feet_t0 table, the own-fold z (2 members, fp32, batch 1) against `z_g_combo` of the
out-of-fold table; on CPU (`--check-only` too) the feature path replayed from the cached frames against the table.
`--finish` writes timing.json, timing.png and complete.json, last; it accepts parts written by another version of this
script (only the sample, configurations and `timing.py` must match; both script hashes are recorded). Label-free: no outcome and no StatsBomb data is read.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from run_xgc_g import OVERLAY
from score_offset_frames import XGT_G_COMBO, verify_bundles
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.detector import location as loc
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.dino import timing as tm
from xgpoc.models import g_bundle
from xgpoc.paths import INDEX
from xgpoc.probe.scoring import SharedScorer
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack_location import inputs as sl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_timing"
FEET = ab.FROZEN.feet
FULL4 = ROOT / "cache/xgd_full4/E3"                      # the frozen frames run (dd2c, E3 pass)
MODELS = Path("/home/kuba/projects/tactifoot_vision/models")
VERSION = "xgdino_timing_v1"
BATCH = 16
WARMUP_SHOTS = 3
LOW_LOAD = 8.0                       # 1-min load average bound of the low-load repeat of the headline
CHECK_TOLERANCE = 1e-4               # |Δ| of a geometry column, live analyser vs the frozen table, counted as equal
ONE_TAIL = (0, 0)                    # (fold, seed) of the single member


def config(name: str, arm: str, precision: str, batch: int = 1, tails: int = 10) -> dict:
    return {"name": name, "arm": arm, "precision": precision, "batch": batch, "tails": tails}


CONFIGS = [config(f"A_{p}_b{b}_t{t}", "A", p, b, t) for p in ("fp32", "fp16") for b in (1, BATCH) for t in (1, 10)]
CONFIGS += [config(f"B_{p}", "B", p) for p in ("fp32", "fp16")]
CONFIGS += [config(f"C_{p}_b{b}", "C", p, b) for p in ("fp32", "fp16") for b in (1, BATCH)]


# ------------------------------------------------------------------ environment
def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def gpu_state() -> dict:
    """GPU utilisation, memory and compute processes (nvidia-smi), and the machine load, right now."""
    query = run_text(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,"
                      "temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader"])
    apps = run_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return {"time": datetime.now(UTC).isoformat(timespec="seconds"), "gpu": query, "compute_apps": apps.splitlines(),
            "loadavg": list(os.getloadavg()), "uptime": run_text(["uptime"])}


def environment() -> dict:
    from importlib.metadata import version

    cpu = next((line.split(":", 1)[1].strip() for line in run_text(["lscpu"]).splitlines()
                if line.startswith("Model name")), platform.processor())
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": run_text(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "ultralytics": version("ultralytics"), "transformers": version("transformers"), "timm": version("timm"),
            "opencv": cv2.__version__, "python": platform.python_version(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}


# ------------------------------------------------------------------ inputs (CPU, label-free)
def population() -> pd.DataFrame:
    with no_statsbomb_reads():
        return sl.build_population(loc.FEET_T0, INDEX, si.OOF, e3=FEET, plan=a1.verify_plan(INDEX)).table


def sample_ids(table: pd.DataFrame, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return sorted(table.index[rng.choice(len(table), n, replace=False)])


def logistics(table: pd.DataFrame) -> dict[str, tm.Logistic]:
    """B and C at deployment: the C1 arms on the population (zero-variance rule), the preprocessor fitted on every
    population shot (label-free), surrogate coefficients of the arm's width (`timing.surrogate`)."""
    arms, _ = ab.arm_set(table, ("B", "C"))
    return {name: tm.surrogate(sf.Preprocessor.fit(arm, table), seed) for seed, (name, arm) in enumerate(arms.items())}


def geometry_gap(rows: dict[str, dict], table: pd.DataFrame) -> dict:
    """Live (or replayed) geometry rows against the frozen table on the C1 allowlist: NaN pattern, max |Δ|, shots
    equal within `CHECK_TOLERANCE`."""
    live = pd.DataFrame.from_dict(rows, orient="index")[list(ab.GEOMETRY)].astype(float)
    frozen = table.loc[live.index, list(ab.GEOMETRY)].astype(float)
    nan_differs = live.isna() != frozen.isna()
    diff = (live - frozen).abs().fillna(0.0)
    per_shot = (diff > CHECK_TOLERANCE).any(axis=1) | nan_differs.any(axis=1)
    worst = diff.max().sort_values(ascending=False)
    return {"shots": len(live), "shots_equal": int((~per_shot).sum()), "max_abs": float(diff.to_numpy().max()),
            "nan_pattern_differs": int(nan_differs.any(axis=1).sum()), "tolerance": CHECK_TOLERANCE,
            "worst_columns": {k: float(v) for k, v in worst.head(5).items()},
            "columns_differing": {c: int(n) for c, n in ((diff > CHECK_TOLERANCE) | nan_differs).sum().items() if n}}


def replay_check(ids: list[str], table: pd.DataFrame) -> dict:
    """`timing.geometry_row` on the cached frames of the sample against the frozen table (CPU)."""
    frames = pd.read_parquet(FEET / "frames.parquet", filters=[("shot_id", "in", ids)])
    rows = {}
    for shot_id, group in frames.groupby("shot_id", sort=True):
        cached = {int(r["offset_ms"]): ex.frame_from_row(r) for r in group.to_dict("records")}
        rows[shot_id] = tm.geometry_row(cached)
    return geometry_gap(rows, table)


# ------------------------------------------------------------------ models
def frozen_weights(models_dir: Path) -> tuple[Path, Path]:
    """The detector and keypoint checkpoints in `models_dir`, refused unless their sha256 are those the frozen
    cache/xgd_full4/E3 run recorded (this worktree has no models/ folder; `extract3` points there)."""
    recorded = json.loads((FULL4 / "complete.json").read_text())["identity"]["models"]
    out = []
    for kind in ("detector", "keypoints"):
        path = models_dir / Path(recorded[kind]["path"]).name
        if sha256_file(path) != recorded[kind]["sha256"]:
            raise SystemExit(f"{path} is not the {kind} checkpoint of the frozen run")
        out.append(path)
    return out[0], out[1]


def build_b(device: str, half: bool, models_dir: Path):
    """(analyser with teams, ball-only analyser) of the frozen feet_t0 run: `extract3.build_analyzers` with the dd2c
    detector at 1280 and the settings of `extract3.COMMON`, the checkpoints from `models_dir`. `half` sets the YOLO
    models' `half` option before their first prediction (the predictor is built then)."""
    import tactifoot_vision as tv
    from tactifoot_vision.pipeline import FrameAnalyzer

    detector_weights, keypoint_weights = frozen_weights(models_dir)
    common = x3.COMMON
    detector = tv.load_model("yolo", detector_weights, conf=common["detector_conf"], iou=common["detector_iou"],
                             imgsz=x3.DETECTORS["dd2c_1280"].imgsz, class_conf={"ball": common["ball_conf"]}, device=device)
    keypoints = tv.load_model("yolo_pose", keypoint_weights, conf=common["keypoint_conf"], imgsz=common["keypoint_imgsz"],
                              resize=common["keypoint_resize"], device=device)
    for model in (detector, keypoints):
        model._yolo.overrides["half"] = half
    homography = tv.pitch.HomographyEstimator(min_confidence=common["homography_min_confidence"],
                                              ransac_threshold=common["homography_ransac_threshold"],
                                              refine_with_lines=common["refine_with_lines"])
    ball_only = FrameAnalyzer(detector, keypoints, None, homography=homography)
    embedder = tv.teams.SigLIPEmbedder(model_name=ex.SIGLIP, device=device)
    teams = tv.teams.TeamClassifier(embedder, n_teams=common["n_teams"], reducer=common["team_reducer"])
    analyzer = FrameAnalyzer(detector, keypoints, teams, homography=homography, crop_scale=common["crop_scale"])
    return analyzer, ball_only


def yolo_half(analyzer) -> dict[str, bool]:
    return {name: bool(getattr(m._yolo.predictor.model, "fp16", False))
            for name, m in (("detect", analyzer.detector), ("keypoints", analyzer.keypoint_model))}


def build_a(device: str, clock: tm.StageClock) -> tuple[tm.APath, list[dict], dict]:
    models = verify_bundles(XGT_G_COMBO, "g_combo", INDEX, a1.FOLDS, ROOT / OVERLAY)
    if len(models) != 10:
        raise ValueError(f"expected the 10 G-combo fold models, found {len(models)}")
    spec = models[0]["spec"]
    pretrained, _ = g_bundle.build_g(spec)
    scorer = SharedScorer(pretrained, spec["model"]["n_blocks"], device)
    members = []
    for m in sorted(models, key=lambda m: (m["fold"], m["seed"]) != ONE_TAIL):          # the single member first
        bundle = g_bundle.load_g(Path(m["path"]))
        if sha256_file(Path(m["path"])) != m["sha256"]:
            raise ValueError(f"{m['path']} is not the verified bundle")
        members.append(scorer.add("g_combo", m["fold"], m["seed"], m["temperature"], bundle.model))
        del bundle
    calibration = json.loads((XGT_G_COMBO / "ensemble.json").read_text())["calibration"]
    g_input = tm.GInput(tuple(spec["input"]["size_wh"]), spec["input"]["mask"])
    path = tm.APath(scorer.vit, spec["model"]["n_blocks"], members, g_input, calibration, clock, device)
    described = [{k: m[k] for k in ("fold", "seed", "sha256", "temperature")} for m in models]
    return path, described, spec


# ------------------------------------------------------------------ one unit
class Bench:
    def __init__(self, a: tm.APath, b: dict[str, tm.BPath], logistic: dict[str, tm.Logistic], clock: tm.StageClock,
                 shots: list[dict]):
        self.a, self.b, self.logistic, self.clock, self.shots = a, b, logistic, clock, shots

    def _a(self, cfg: dict, shots: list[dict]) -> None:
        with self.clock.stage("decode"):
            images = [cv2.imread(s["frame_path"]) for s in shots]
        self.a(images, cfg["tails"], cfg["precision"] == "fp16")

    def _b(self, cfg: dict, shot: dict) -> dict[str, float]:
        b = self.b[cfg["precision"]]
        row = b.features(b.analyse(shot, b.decode(shot)))
        with self.clock.stage("b_logistic"):
            self.logistic["B"](pd.DataFrame([row]))
        return row

    def _c(self, cfg: dict, shots: list[dict]) -> None:
        b = self.b[cfg["precision"]]
        images = [b.decode(s) for s in shots]
        z, _ = self.a([i[0] for i in images], cfg["tails"], cfg["precision"] == "fp16")
        rows = [b.features(b.analyse(s, i)) for s, i in zip(shots, images, strict=True)]
        with self.clock.stage("c_logistic"):
            self.logistic["C"](pd.DataFrame(rows).assign(**{sf.Z: z}))

    def call(self, cfg: dict, shots: list[dict]) -> dict | None:
        if cfg["arm"] == "A":
            self._a(cfg, shots)
        elif cfg["arm"] == "B":
            return self._b(cfg, shots[0])
        else:
            self._c(cfg, shots)
        return None

    def batches(self, cfg: dict, shots: list[dict]) -> list[list[dict]]:
        size = cfg["batch"]
        return [shots[i:i + size] for i in range(0, len(shots) - size + 1, size)]

    def warmup(self, cfg: dict) -> None:
        for batch in self.batches(cfg, self.shots)[: 1 if cfg["batch"] > 1 else WARMUP_SHOTS]:
            self.call(cfg, batch)
        self.clock.take()

    def unit(self, cfg: dict) -> tuple[dict, dict[str, dict]]:
        """One repetition over the sample: per-call totals (ms per shot), per-shot stage means, -800 ms reads."""
        totals, stages, rows = [], {}, {}
        batches = self.batches(cfg, self.shots)
        load = os.getloadavg()[0]
        for batch in batches:
            tm.cuda_sync()
            start = time.perf_counter()
            row = self.call(cfg, batch)
            tm.cuda_sync()
            totals.append((time.perf_counter() - start) * 1000 / len(batch))
            for k, v in self.clock.take().items():
                stages[k] = stages.get(k, 0.0) + v
            if row is not None:
                rows[batch[0]["shot_id"]] = row
        n = sum(len(b) for b in batches)
        stages = {k: v / n for k, v in stages.items()}
        if "analyse" in stages:
            stages["analyse_rest"] = tm.analyse_rest(stages)
        return {"totals_ms": totals, "stages_ms": stages, "shots": n, "load1_start": load, "load1_end": os.getloadavg()[0]}, rows


# ------------------------------------------------------------------ run
def identity(ids: list[str]) -> dict:
    return {"version": VERSION, "sample": {"count": len(ids), "digest": canonical_digest(ids)},
            "configs": CONFIGS, "batch": BATCH, "warmup_shots": WARMUP_SHOTS,
            "code": {"timing.py": sha256_file(ROOT / "xgpoc/dino/timing.py"), "script": sha256_file(Path(__file__))}}


def read_units(parts: Path) -> list[dict]:
    path = parts / "units.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def a_check(a: tm.APath, shots: list[dict], folds: pd.Series, oof_z: pd.Series) -> dict:
    """Own-fold z (the two members of the shot's held-out fold, fp32, batch 1) against `z_g_combo`."""
    own = []
    for shot in shots:
        image = cv2.imread(shot["frame_path"])
        members = [m for m in a.members if m.fold == folds[shot["shot_id"]]]
        path = tm.APath(a.vit, a.n_trained, members, a.g_input, a.calibration, tm.StageClock(), a.device)
        own.append(float(path([image], len(members), False)[0][0]))
    ids = [s["shot_id"] for s in shots]
    gap = np.abs(np.array(own) - oof_z.loc[ids].to_numpy())
    return {"shots": len(ids), "members_per_shot": 2, "mean_abs": float(gap.mean()), "max_abs": float(gap.max()),
            "r": float(np.corrcoef(own, oof_z.loc[ids].to_numpy())[0, 1])}


def run(args, table: pd.DataFrame, ids: list[str]) -> None:
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    ident = identity(ids)
    if (parts / "identity.json").exists():
        if json.loads((parts / "identity.json").read_text()) != ident:
            raise SystemExit(f"{parts} holds units of another sample, configuration set or code; use a new --out")
    else:
        (parts / "identity.json").write_text(json.dumps(ident, indent=1) + "\n")
    done = {(u["config"], u["rep"]) for u in read_units(parts)}
    todo = [(rep, cfg) for rep in range(args.reps) for cfg in CONFIGS if (cfg["name"], rep) not in done]
    if not todo:
        print("every unit is done; run --finish")
        return
    session = {"started": datetime.now(UTC).isoformat(timespec="seconds"), "before": gpu_state(),
               "git": records.git_state(), "environment": environment()}
    clock = tm.StageClock(tm.cuda_sync)
    loading = time.perf_counter()
    a, members, spec = build_a(args.device, clock)
    b_models = {p: build_b(args.device, p == "fp16", args.models) for p in ("fp32", "fp16")}
    b = {p: tm.BPath(an, bo, clock) for p, (an, bo) in b_models.items()}
    shots = ex.frame_table(INDEX, ids).to_dict("records")
    bench = Bench(a, b, logistics(table), clock, shots)
    session["load_seconds"] = round(time.perf_counter() - loading, 1)
    if not (parts / "checks.json").exists():
        bench.warmup(CONFIGS[0])
        checks = {"a_own_fold_z": a_check(a, shots, table.fold, table[sf.Z])}
        _, rows = bench.unit(next(c for c in CONFIGS if c["name"] == "B_fp32"))
        checks["b_live_vs_frozen"] = geometry_gap(rows, table)
        _, rows16 = bench.unit(next(c for c in CONFIGS if c["name"] == "B_fp16"))
        checks["b_fp16_vs_frozen"] = geometry_gap(rows16, table)
        checks["yolo_half"] = {p: yolo_half(an) for p, (an, _) in b_models.items()}
        checks["members"], checks["spec"] = members, spec
        (parts / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
        print(json.dumps({k: checks[k] for k in ("a_own_fold_z", "b_live_vs_frozen", "yolo_half")}), flush=True)
    warmed: set[str] = set()
    started, count = time.perf_counter(), 0
    for rep, cfg in todo:
        if time.perf_counter() - started > args.budget:
            break
        if cfg["name"] not in warmed:
            bench.warmup(cfg)
            warmed.add(cfg["name"])
        result, _ = bench.unit(cfg)
        record = {"config": cfg["name"], "rep": rep, "session": session["started"]} | result
        with (parts / "units.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        count += 1
        print(f"{cfg['name']} rep {rep}: {np.mean(result['totals_ms']):.1f} ms/shot, load {result['load1_end']:.1f}",
              flush=True)
    session |= {"finished": datetime.now(UTC).isoformat(timespec="seconds"), "units": count, "after": gpu_state(),
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}
    with (parts / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(session) + "\n")
    left = len(todo) - count
    print(f"session: {count} units, {left} left")


# ------------------------------------------------------------------ finish
STAGE_ORDER = ("decode", "g_input", "g_trunk", "g_tails", "g_calibrate", "detect", "keypoints", "homography", "teams",
               "analyse_rest", "features", "b_logistic", "c_logistic")


def low_load(unit: dict) -> bool:
    return max(unit["load1_start"], unit["load1_end"]) <= LOW_LOAD


def aggregate(units: list[dict]) -> dict[str, dict]:
    """Per configuration: the per-repetition mean ms per shot (median, IQR over repetitions), every call's ms per shot
    (median, IQR over calls x repetitions), and each stage's per-repetition mean, with the load during its units."""
    out = {}
    for cfg in CONFIGS:
        mine = sorted((u for u in units if u["config"] == cfg["name"]), key=lambda u: u["rep"])
        if not mine:
            continue
        stages = {s: tm.summary([u["stages_ms"].get(s, 0.0) for u in mine]) for s in STAGE_ORDER
                  if any(s in u["stages_ms"] for u in mine)}
        out[cfg["name"]] = cfg | {
            "reps": len(mine), "shots_per_rep": mine[0]["shots"],
            "ms_per_shot": tm.summary([np.mean(u["totals_ms"]) for u in mine]),
            "ms_per_call_shot": tm.summary(np.concatenate([u["totals_ms"] for u in mine])),
            "stages_ms_per_shot": stages,
            "load1": tm.summary([(u["load1_start"] + u["load1_end"]) / 2 for u in mine])}
    return out


def headline(table: dict[str, dict]) -> dict:
    """A (1 / 10 tails), B and C in ms per shot (medians), C's measured value beside A + B and max(A, B) (the parallel
    bound, an estimate from the separate medians, not measured)."""
    med = {k: v["ms_per_shot"]["median"] for k, v in table.items()}
    out = {}
    for p in ("fp32", "fp16"):
        for b in (1, BATCH):
            a10, a1_, bb, c = med.get(f"A_{p}_b{b}_t10"), med.get(f"A_{p}_b{b}_t1"), med.get(f"B_{p}"), med.get(f"C_{p}_b{b}")
            if None in (a10, a1_, bb, c):
                continue
            out[f"{p}_b{b}"] = {"A_1_tail": a1_, "A_10_tails": a10, "B": bb, "C_measured": c, "A_plus_B": a10 + bb,
                                "max_A_B_parallel_estimate": max(a10, bb)}
    return out


def figure(table: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c["name"] for c in CONFIGS if c["name"] in table and (c["arm"] != "A" or c["tails"] == 10 or c["batch"] == 1)]
    groups = {"decode": ("decode",), "G-combo (input, trunk, tails, calibration)": ("g_input", "g_trunk", "g_tails", "g_calibrate"),
              "detector (dd2c)": ("detect",), "pitch keypoints": ("keypoints",), "homography + lines": ("homography",),
              "teams (SigLIP + k-means)": ("teams",), "features + logistic + rest": ("analyse_rest", "features", "b_logistic", "c_logistic")}
    colours = ["#9aa5b1", "#3b6fb6", "#d1495b", "#edae49", "#66a182", "#8d6cab", "#555555"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for (label, keys), colour in zip(groups.items(), colours, strict=True):
        width = np.array([sum(table[n]["stages_ms_per_shot"].get(k, {}).get("median", 0.0) for k in keys) for n in names])
        ax.barh(y, width, left=left, color=colour, label=label, height=0.7)
        left += width
    for i, n in enumerate(names):
        s = table[n]["ms_per_shot"]
        ax.errorbar(s["median"], i, xerr=[[s["median"] - s["q25"]], [s["q75"] - s["median"]]], fmt="k|", ms=10, lw=1)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ms per shot (bars: stage medians; tick: end-to-end median, IQR)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("xg-dino 03: inference time per shot, RTX 4070 Ti SUPER, real weights", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def finish(args, ids: list[str]) -> None:
    parts = args.out / "parts"
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out}/complete.json exists")
    pinned, current = json.loads((parts / "identity.json").read_text()), identity(ids)
    if pinned | {"code": pinned["code"] | {"script": None}} != current | {"code": current["code"] | {"script": None}}:
        raise SystemExit("the parts are of another sample, configuration set or timing code")
    units = read_units(parts)
    missing = [(c["name"], r) for r in range(args.reps) for c in CONFIGS
               if not any(u["config"] == c["name"] and u["rep"] == r for u in units)]
    if missing:
        raise SystemExit(f"{len(missing)} units missing, e.g. {missing[0]}")
    sessions = [json.loads(line) for line in (parts / "sessions.jsonl").read_text().splitlines()]
    checks = json.loads((parts / "checks.json").read_text())
    table = aggregate(units)
    quiet = aggregate([u for u in units if low_load(u)])
    loads = [s["before"]["loadavg"][0] for s in sessions] + [s["after"]["loadavg"][0] for s in sessions]
    result = {
        "what": "inference time per shot of A (G-combo), B (automatic geometry, feet_t0 E3) and C (both), real weights; "
                "deployment cost of one new shot, not the development (out-of-fold) predictions of ticket 02",
        "environment": sessions[0]["environment"], "sample": identity(ids)["sample"] | {"seed": args.seed},
        "reps": args.reps, "batch": BATCH, "warmup_shots": WARMUP_SHOTS,
        "load": {"loadavg_1min_session_bounds": tm.summary(loads), "above_8": bool(max(loads) > 8),
                 "note": "load average above 8 (other agents' CPU jobs) during the run; B's CPU stages are the most "
                         "affected" if max(loads) > 8 else "load average at most 8"},
        "idle_gpu_before": [s["before"] for s in sessions], "after": [s["after"] for s in sessions],
        "headline_ms_per_shot": headline(table),
        "headline_ms_per_shot_low_load": {"rule": f"units whose 1-min load average was <= {LOW_LOAD} at start and end",
                                          "reps": {k: v["reps"] for k, v in quiet.items()}} | headline(quiet),
        "configs": table, "configs_low_load": quiet, "checks": checks,
        "deviations": [
            "B and C logistic: the fitted coefficients of ticket 02 were not stored; a logistic of the same design width "
            "with fixed random coefficients is timed on the real preprocessor (imputation, spline, standardisation) "
            "fitted label-free on the 7,062 population shots",
            "the E3 ball-source columns that feet_t0 takes from E3's order at t0 are not computed: they are outside the "
            "C1 geometry allowlist of B and C",
            "B fp16 sets YOLO half=True for the detector and keypoint model; it is not the frozen configuration (SigLIP "
            "runs in its built-in fp16 autocast in both B rows)",
            "C is measured sequentially (one stream); max(A, B) is an estimate of a parallel run from separate medians",
            "frames are JPEGs from the index, read from the page cache after the first repetition"],
        "sessions": [{k: s[k] for k in ("started", "finished", "units", "load_seconds", "peak_gpu_mb", "git")} for s in sessions]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing.json").write_text(json.dumps(result, indent=1) + "\n")
    figure(table, args.out / "timing.png")
    record = {"version": VERSION, "identity": pinned, "script_at_finish": current["code"]["script"], "code": records.code_hashes(Path(__file__))
              | {"timing.py": sha256_file(ROOT / "xgpoc/dino/timing.py")}, "git": records.git_state(),
              "files": {p.name: sha256_file(p) for p in sorted(args.out.glob("*")) if p.is_file()}
              | {f"parts/{p.name}": sha256_file(p) for p in sorted(parts.glob("*")) if p.is_file()}}
    (args.out / "complete.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("headline_ms_per_shot", "headline_ms_per_shot_low_load", "load")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--budget", type=float, default=420.0, help="seconds after which no new unit starts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", type=Path, default=MODELS, help="folder of the dd2c and pitch keypoint checkpoints")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--finish", action="store_true")
    args = ap.parse_args()
    table = population()
    ids = sample_ids(table, args.shots, args.seed)
    if args.finish:
        return finish(args, ids)
    if args.check_only:
        widths = {k: v.width for k, v in logistics(table).items()}
        print(json.dumps({"sample": identity(ids)["sample"], "widths": widths, "replay": replay_check(ids, table)}, indent=1))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    replay = replay_check(ids, table)
    if replay["shots_equal"] != replay["shots"]:
        raise SystemExit(f"the feature path does not replay the frozen table: {replay}")
    with no_statsbomb_reads():
        run(args, table, ids)


if __name__ == "__main__":
    sys.exit(main())
