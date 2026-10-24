"""Safe-margin check of o_max = -600 ms (xg-temporal ticket 02). Rule: `research/xgt_safe_margin_rule.md`.

    ./run.sh --no-sync python scripts/xgt_safe_margin.py sample
    ./run.sh --no-sync python scripts/xgt_safe_margin.py sheets [--out outputs/xgt_safe_margin]
    ./run.sh --no-sync python scripts/xgt_safe_margin.py zoom SHEET_ID --center X Y [--width 320] [--offsets ...] --out Z.png
    ./run.sh --no-sync python scripts/xgt_safe_margin.py evaluate --run-a A.csv [...] --run-b B.csv [...]
                                                              [--out results/xgt_safe_margin]

`sample` draws the frozen sample: 120 non-penalty shots of the xgv `training` role whose -800, -600, -400 and 0 ms
frames exist in `cache/xgc/offset_frames_v1` and whose -600 ms frame has no camera cut before the anchor. It writes
`research/xgt_safe_margin_sample.csv` (sheet_id, batch, shot_id, match); the rule pins its sha256.

`sheets` cuts the labelling material from the 720p clips, decoding each clip only up to its anchor frame (no frame
after 0 ms is ever decoded). Per shot, under an anonymous sheet id: one contact sheet of 24 tiles (448x252, the size of
the cached frames) and the same 24 frames at native 720p for zooming. Frames on the cache grid are checked against
`frames.csv` (frame index, cut flag, digest of the native decoded pixels). Writes `batches.csv` (4 batches of 30) and
`sheets_manifest.json`. Nothing here reads a shot outcome. `zoom` is the labellers' magnifier: the same crop of the
native frames of one sheet at several offsets (<= 0), each scaled to 640x360.

`evaluate` reads the two independent answer runs (sheet_id, struck_at_m800, struck_at_m600, struck_at_m400 in
{yes, no, unknown}, optional note), applies the worst-case rule, the one-sided 95% Clopper-Pearson bound and the
agreement statistics, adds the descriptive evidence table (GSR kicks, v1.5 contacts: time columns only) and the D6
spread check (the aggregated share in `timing_verdict.json`, no per-shot residual), and writes `<out>/answers.csv`,
`evidence.csv`, `result.json` and `complete.json` (input and output hashes), last.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import beta
from xgpoc.bundle import sha256_file
from xgpoc.data.canonical_frames import array_digest, probe_video
from xgpoc.data.offset_frames import (
    ANCHOR_MS,
    CUT_THRESHOLD,
    TIMESTAMP_AGREEMENT_MS,
    anchor_frame_index,
    anchor_selection,
    check_frame_rate,
    grey_thumbnail,
    max_change_between,
    select_offsets,
)
from xgpoc.detector import records

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/xgv_split_plan/plan.csv"
PLAN_SHA256 = "a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a"
OFFSET_FRAMES = ROOT / "cache/xgc/offset_frames_v1"
SAMPLE = ROOT / "research/xgt_safe_margin_sample.csv"
SAMPLE_SHA256 = "1f9d38cd9b3ebd8cf9f87f9eb75815a9d2ed683db74cc7e548a7761402c20e5e"  # pinned in the rule too
RULE = ROOT / "research/xgt_safe_margin_rule.md"
RULE_SHA256 = ROOT / "research/xgt_safe_margin_rule.md.sha256"   # `sha256sum` line, committed with the rule
GSR_TIMING = ROOT / "results/pk_ball_evidence/timing.csv"
V15_SHOTS = ROOT / "cache/v15/canonical_data_v1/shots.csv"
TIMING_VERDICT = ROOT / "results/xgv_timing_source/timing_verdict.json"
SHEETS_OUT = ROOT / "outputs/xgt_safe_margin"
RESULTS_OUT = ROOT / "results/xgt_safe_margin"

SEED = 20261025
N_SHOTS = 120
N_BATCHES = 4
ELIGIBLE_OK = (-800, -600, -400, 0)       # frames that must exist (status ok) in offset_frames_v1
O_MAX = -600                              # the checked bound (decision D1); its frame must have no cut before 0
SHEET_OFFSETS = (-1600, -1200, -1000, *range(-800, 1, 40))
QUESTIONS = (-800, -600, -400)            # answer columns, earliest first
ANSWERS = ("yes", "no", "unknown")
MAX_RATE, CONFIDENCE = 0.05, 0.95
TILE = (448, 252)
COLUMNS = 4
LABEL_H, HEADER_H = 30, 44
GSR_POSSIBLE_PASS_MS = -600               # GSR kicks at or before this may be a preceding pass (spec, unchecked)
OUTCOME_WORDS = ("goal", "label", "outcome", "statsbomb", "sb_", "on_target", "ai_outcome")


def column(offset: int) -> str:
    return f"struck_at_m{-offset}"


# ---------------------------------------------------------------------------------------------------------- rule maths

def cp_upper(violations: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided upper Clopper-Pearson bound of a binomial share."""
    if not 0 <= violations <= n or n <= 0:
        raise ValueError("need 0 <= violations <= n and n > 0")
    return 1.0 if violations == n else float(beta.ppf(confidence, violations + 1, n - violations))


def passes(violations: int, n: int) -> bool:
    return cp_upper(violations, n) <= MAX_RATE


def struck_worst(answers: dict[int, str], offset: int) -> bool:
    """Worst case of one run at `offset`: struck unless the run says `no` there and no earlier offset says `yes`.

    Being struck is monotone in time: `yes` at an earlier offset implies struck at every later one, so a later `no`
    contradicting it counts as struck (the earliest strike)."""
    if answers[offset] != "no":
        return True
    return any(answers[o] == "yes" for o in QUESTIONS if o < offset)


def inconsistent(answers: dict[int, str]) -> bool:
    """A `no` after a `yes` in time order (the ball cannot be un-struck)."""
    seen_yes = False
    for offset in QUESTIONS:
        seen_yes |= answers[offset] == "yes"
        if seen_yes and answers[offset] == "no":
            return True
    return False


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa over the answer categories; None when chance agreement is 1 (both runs constant and equal)."""
    if len(a) != len(b) or not a:
        raise ValueError("two equally long non-empty answer lists needed")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in ANSWERS)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


# ------------------------------------------------------------------------------------------------------------- sample

def check_hash(path: Path, expected: str) -> str:
    digest = sha256_file(path)
    if digest != expected:
        raise SystemExit(f"{path} sha256 {digest} != expected {expected}")
    return digest


def offset_frames_hash(name: str) -> str:
    return json.loads((OFFSET_FRAMES / "complete.json").read_text())["files"][name]


def eligible_shots() -> pd.DataFrame:
    """Non-penalty `training` shots with the frames of the check; outcome-free columns only."""
    check_hash(PLAN, PLAN_SHA256)
    for name in ("frames.csv", "shots.csv"):
        check_hash(OFFSET_FRAMES / name, offset_frames_hash(name))
    plan = pd.read_csv(PLAN, usecols=["game", "role"])
    shots = pd.read_csv(OFFSET_FRAMES / "shots.csv", usecols=["shot_id", "game", "is_penalty"])
    shots = shots.merge(plan, on="game", how="left", validate="many_to_one")
    shots = shots[(shots.role == "training") & ~shots.is_penalty.astype(bool)]
    frames = pd.read_csv(OFFSET_FRAMES / "frames.csv", usecols=["shot_id", "offset_ms", "status", "cut_between_anchor"],
                         dtype={"cut_between_anchor": "string"})
    frames = frames[frames.offset_ms.isin(ELIGIBLE_OK) & frames.shot_id.isin(shots.shot_id)]
    status = frames.pivot(index="shot_id", columns="offset_ms", values="status")
    cut = frames.pivot(index="shot_id", columns="offset_ms", values="cut_between_anchor")
    ok = (status[list(ELIGIBLE_OK)] == "ok").all(axis=1) & (cut[O_MAX] == "False")
    keep = set(ok[ok].index)
    return shots[shots.shot_id.isin(keep)].sort_values("shot_id").reset_index(drop=True)


def draw_sample(shot_ids: list[str], games: list[str], seed: int = SEED, n: int = N_SHOTS,
                batches: int = N_BATCHES) -> pd.DataFrame:
    """Seeded draw without replacement from the sorted ids; the draw order is the (anonymous) sheet order."""
    order = np.argsort(np.asarray(shot_ids), kind="stable")
    ids, matches = np.asarray(shot_ids)[order], np.asarray(games)[order]
    picked = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    per_batch = -(-n // batches)
    return pd.DataFrame({"sheet_id": [f"s{i + 1:03d}" for i in range(n)],
                         "batch": [i // per_batch + 1 for i in range(n)],
                         "shot_id": ids[picked], "match": matches[picked]})


def cmd_sample(args: argparse.Namespace) -> None:
    shots = eligible_shots()
    sample = draw_sample(shots.shot_id.tolist(), shots.game.tolist())
    sample.to_csv(args.sample, index=False)
    print(json.dumps({"eligible_shots": len(shots), "eligible_matches": int(shots.game.nunique()),
                      "sample_shots": len(sample), "sample_matches": int(sample["match"].nunique()),
                      "seed": SEED, "sample": str(args.sample), "sha256": sha256_file(args.sample)}, indent=1))


# ------------------------------------------------------------------------------------------------------------- sheets

def check_rule() -> str:
    """The frozen rule must match the sha256 committed next to it."""
    return check_hash(RULE, RULE_SHA256.read_text().split()[0])


def load_sample(path: Path = SAMPLE) -> pd.DataFrame:
    if not SAMPLE_SHA256:
        raise SystemExit("SAMPLE_SHA256 is not set: run `sample`, then pin its hash in the script and the rule")
    check_hash(path, SAMPLE_SHA256)
    return pd.read_csv(path)


def decode_until_anchor(path: Path, offsets: tuple[int, ...]) -> dict[int, dict]:
    """Frames at `offsets` (all <= 0) by the offset-frames selection rule, decoding only up to the anchor frame.

    Returns per offset: status, frame index, decoded ms, BGR (or None) and the cut flag over the transitions after
    the frame up to and including the anchor frame (the rule of `xgpoc.data.offset_frames`)."""
    if any(o > 0 for o in offsets):
        raise ValueError("only offsets <= 0 may be decoded")
    timeline = probe_video(path)
    relative = np.array([timeline.relative_ms(i, ANCHOR_MS) for i in range(len(timeline.pts))])
    check_frame_rate(relative)
    anchor = anchor_frame_index(relative)
    selections = select_offsets(relative, tuple(o for o in offsets if o != 0))
    if 0 in offsets:
        selections.append(anchor_selection(relative))
    chosen = {s.frame_index for s in selections if s.frame_index is not None}
    if any(i > anchor for i in chosen):
        raise ValueError("a selected frame lies after the anchor frame")
    first = min(chosen)
    kept, thumbnails = {}, {}
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    cv_first = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path}")
        for index in range(anchor + 1):  # never past the anchor frame
            ok, bgr = capture.read()
            if not ok or bgr.shape != (timeline.height, timeline.width, 3):
                raise ValueError("Decoder shape or frame count differs from ffprobe")
            cv_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            cv_first = cv_ms if cv_first is None else cv_first
            if not np.isfinite(cv_ms) or abs((cv_ms - cv_first - ANCHOR_MS) - relative[index]) > TIMESTAMP_AGREEMENT_MS:
                raise ValueError("OpenCV and ffprobe frame timestamps disagree")
            if index >= first:
                thumbnails[index] = grey_thumbnail(bgr)
            if index in chosen:
                kept[index] = bgr
    finally:
        capture.release()
    changes = {k: float(np.abs(thumbnails[k] - thumbnails[k - 1]).mean() / 255) for k in range(first + 1, anchor + 1)}
    out = {}
    for s in selections:
        if s.frame_index is None:
            out[s.offset_ms] = {"status": s.status, "index": None, "decoded_ms": None, "bgr": None, "cut": None}
        else:
            cut = max_change_between(changes, anchor, s.frame_index) >= CUT_THRESHOLD
            out[s.offset_ms] = {"status": s.status, "index": s.frame_index, "decoded_ms": s.decoded_ms,
                                "bgr": kept[s.frame_index], "cut": bool(cut)}
    return out


def tile(frame: dict, offset: int) -> np.ndarray:
    """One 448x252 tile with a label strip: offset, `Q` on the three question offsets, `CUT` / `MISSING`."""
    width, height = TILE
    image = np.full((height, width, 3), 60, np.uint8)
    if frame["bgr"] is not None:
        image = cv2.resize(frame["bgr"], TILE, interpolation=cv2.INTER_AREA)
    strip = np.full((LABEL_H, width, 3), 255, np.uint8)
    text = f"{offset:+d} ms" if offset else "0 ms (anchor)"
    if offset in QUESTIONS:
        strip[:] = (200, 235, 255)
        text += "   <- question"
    cv2.putText(strip, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    flag = "MISSING" if frame["bgr"] is None else ("CUT" if frame["cut"] else "")
    if flag:
        cv2.putText(strip, flag, (width - 120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
        if flag == "CUT":
            cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 220), 4)
    return np.vstack([strip, image])


def contact_sheet(sheet_id: str, frames: dict[int, dict]) -> np.ndarray:
    tiles = [tile(frames[o], o) for o in SHEET_OFFSETS]
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % COLUMNS)
    rows = [np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)]
    header = np.full((HEADER_H, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{sheet_id}   frames up to the shot anchor (0 ms); no later frame is shown",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([header, *rows])


def offset_name(offset: int) -> str:
    return f"m{-offset:04d}" if offset < 0 else "p0000"


def cmd_sheets(args: argparse.Namespace) -> None:
    rule_sha = check_rule()
    sample = load_sample().head(args.limit)
    shots = pd.read_csv(OFFSET_FRAMES / "shots.csv", usecols=["shot_id", "clip_path"]).set_index("shot_id")
    cached = pd.read_csv(OFFSET_FRAMES / "frames.csv", usecols=["shot_id", "offset_ms", "status", "source_frame_index",
                                                                  "cut_between_anchor", "pixel_sha256"],
                         dtype={"cut_between_anchor": "string"})
    cached = cached[cached.shot_id.isin(sample.shot_id) & (cached.offset_ms <= 0)].set_index(["shot_id", "offset_ms"])
    sheets, native = args.out / "sheets", args.out / "frames"
    sheets.mkdir(parents=True, exist_ok=True)
    checked, mismatches, cut_tiles, missing_tiles = 0, [], 0, 0
    for row in sample.itertuples():
        frames = decode_until_anchor(Path(shots.loc[row.shot_id, "clip_path"]), SHEET_OFFSETS)
        for offset, frame in frames.items():
            cut_tiles += bool(frame["cut"])
            missing_tiles += frame["bgr"] is None
            if (row.shot_id, offset) not in cached.index:
                continue
            ref = cached.loc[(row.shot_id, offset)]
            checked += 1
            if ref.status != frame["status"] or (frame["bgr"] is not None and (
                    int(ref.source_frame_index) != frame["index"] or ref.cut_between_anchor != str(frame["cut"])
                    or ref.pixel_sha256 != array_digest(frame["bgr"]))):
                mismatches.append({"sheet_id": row.sheet_id, "offset_ms": offset})
        cv2.imwrite(str(sheets / f"{row.sheet_id}.png"), contact_sheet(row.sheet_id, frames))
        folder = native / row.sheet_id
        folder.mkdir(parents=True, exist_ok=True)
        for offset, frame in frames.items():
            if frame["bgr"] is not None:
                cv2.imwrite(str(folder / f"{offset_name(offset)}.jpg"), frame["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if mismatches:
        raise SystemExit(f"decoded frames differ from offset_frames_v1: {mismatches[:10]}")
    batches = sample[["sheet_id", "batch"]].assign(
        sheet=[str(sheets / f"{s}.png") for s in sample.sheet_id],
        native_frames=[str(native / s) for s in sample.sheet_id])
    batches.to_csv(args.out / "batches.csv", index=False)
    manifest = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": records.git_state(),
                "sample": {"path": str(SAMPLE.relative_to(ROOT)), "sha256": SAMPLE_SHA256},
                "rule": {"path": str(RULE.relative_to(ROOT)), "sha256": rule_sha},
                "sheet_offsets_ms": list(SHEET_OFFSETS), "tile": list(TILE), "columns": COLUMNS,
                "native_frames": "720p JPEG quality 95, <sheet_id>/m<ms>.jpg, p0000 = anchor frame",
                "decode": "clip decoded only up to its anchor frame; offset-frames selection and cut rule",
                "checked_against_offset_frames_v1": checked, "mismatches": 0,
                "cut_tiles": cut_tiles, "missing_tiles": missing_tiles,
                "batches.csv": sha256_file(args.out / "batches.csv"),
                "sheets": {f"{s}.png": sha256_file(sheets / f"{s}.png") for s in sample.sheet_id}}
    (args.out / "sheets_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("checked_against_offset_frames_v1", "cut_tiles", "missing_tiles")}))


def zoom_sheet(frames_dir: Path, sheet_id: str, offsets: list[int], center: tuple[int, int], width: int) -> np.ndarray:
    """2-column sheet of the same native-frame crop (16:9, `width` px wide) at each offset, each scaled to 640 px."""
    if any(o > 0 for o in offsets):
        raise SystemExit("only offsets <= 0 exist")
    height = round(width * 9 / 16)
    tiles = []
    for offset in offsets:
        path = frames_dir / sheet_id / f"{offset_name(offset)}.jpg"
        if not path.exists():
            raise SystemExit(f"no native frame {path} (sheet offsets: {list(SHEET_OFFSETS)})")
        bgr = cv2.imread(str(path))
        x0 = int(np.clip(center[0] - width // 2, 0, bgr.shape[1] - width))
        y0 = int(np.clip(center[1] - height // 2, 0, bgr.shape[0] - height))
        crop = cv2.resize(bgr[y0:y0 + height, x0:x0 + width], (640, 360), interpolation=cv2.INTER_CUBIC)
        strip = np.full((LABEL_H, 640, 3), 255, np.uint8)
        cv2.putText(strip, f"{sheet_id} {offset:+d} ms  crop x{x0}-{x0 + width} y{y0}-{y0 + height}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([strip, crop]))
    tiles += [np.full_like(tiles[0], 255)] * (len(tiles) % 2)
    return np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])


def cmd_zoom(args: argparse.Namespace) -> None:
    image = zoom_sheet(args.frames, args.sheet_id, args.offsets, tuple(args.center), args.width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    print(args.out)


# ----------------------------------------------------------------------------------------------------------- evaluate

def read_answers(paths: list[Path], sheet_ids: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    answers = pd.concat(frames, ignore_index=True)
    needed = ["sheet_id", *(column(o) for o in QUESTIONS)]
    if missing := [c for c in needed if c not in answers.columns]:
        raise SystemExit(f"answer columns missing: {missing}")
    answers = answers[needed + (["note"] if "note" in answers.columns else [])]
    for c in needed[1:]:
        answers[c] = answers[c].str.strip().str.lower()
        if bad := sorted(set(answers[c]) - set(ANSWERS)):
            raise SystemExit(f"{c}: answers outside {ANSWERS}: {bad}")
    if answers.sheet_id.duplicated().any():
        raise SystemExit(f"duplicate sheet ids: {sorted(answers.sheet_id[answers.sheet_id.duplicated()])}")
    if set(answers.sheet_id) != set(sheet_ids):
        raise SystemExit(f"answers cover {len(set(answers.sheet_id) & set(sheet_ids))} of {len(sheet_ids)} sheets; "
                         f"unknown ids {sorted(set(answers.sheet_id) - set(sheet_ids))}")
    return answers.set_index("sheet_id").loc[sheet_ids]


def evaluate_runs(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-sheet table and the rule result. `a`, `b`: answers indexed by sheet id, same order."""
    table = pd.DataFrame(index=a.index)
    for run, frame in (("a", a), ("b", b)):
        rows = [{o: r[column(o)] for o in QUESTIONS} for _, r in frame.iterrows()]
        for o in QUESTIONS:
            table[f"{run}_{column(o)}"] = frame[column(o)]
            table[f"{run}_worst_m{-o}"] = [struck_worst(r, o) for r in rows]
        table[f"{run}_inconsistent"] = [inconsistent(r) for r in rows]
    for o in QUESTIONS:
        table[f"agree_m{-o}"] = a[column(o)] == b[column(o)]
        table[f"violation_m{-o}"] = table[f"a_worst_m{-o}"] | table[f"b_worst_m{-o}"]
    n = len(table)
    per_offset = {}
    for o in QUESTIONS:
        k = int(table[f"violation_m{-o}"].sum())
        per_offset[str(o)] = {
            "role": "rule" if o == O_MAX else "descriptive",
            "worst_case_struck": k, "n": n, "share": k / n, "cp_upper_95": cp_upper(k, n),
            "both_yes": int(((a[column(o)] == "yes") & (b[column(o)] == "yes")).sum()),
            "any_unknown": int(((a[column(o)] == "unknown") | (b[column(o)] == "unknown")).sum()),
            "disagreements": int((~table[f"agree_m{-o}"]).sum()),
            "agreement": float(table[f"agree_m{-o}"].mean()),
            "cohen_kappa": cohen_kappa(a[column(o)].tolist(), b[column(o)].tolist()),
            "answers_a": {c: int((a[column(o)] == c).sum()) for c in ANSWERS},
            "answers_b": {c: int((b[column(o)] == c).sum()) for c in ANSWERS}}
    k = per_offset[str(O_MAX)]["worst_case_struck"]
    verdict = "pass" if passes(k, n) else "stop"
    result = {"o_max_ms": O_MAX, "verdict": verdict, "violations": k, "n": n, "cp_upper_95": cp_upper(k, n),
              "max_rate": MAX_RATE, "per_offset": per_offset,
              "inconsistent_runs": {"a": int(table.a_inconsistent.sum()), "b": int(table.b_inconsistent.sum())},
              "rule": "violation at -600 = a run answers yes/unknown at -600 or yes at -800, in either run; "
                      "pass iff one-sided 95% Clopper-Pearson upper bound <= 0.05"}
    if verdict == "stop":
        result["next"] = "phase stops; ticket 03 blocked; question to the owner (o_max may only move earlier)"
    return table.reset_index(), result


def evidence_table() -> pd.DataFrame:
    """Existing contact evidence, descriptive: GSR kick times and v1.5 AI contacts (time columns only)."""
    rows = []
    gsr = pd.read_csv(GSR_TIMING, usecols=["split", "clip", "kick_ms"]).dropna(subset=["kick_ms"])
    for r in gsr.itertuples():
        rows.append({"source": "gsr_kick_rule", "item": f"{r.split}/{r.clip}", "contact_lower_ms": r.kick_ms,
                     "contact_upper_ms": r.kick_ms, "before_o_max": r.kick_ms <= O_MAX,
                     "note": ("may be a preceding pass: kick.py takes the first kick in its window (unchecked)"
                              if r.kick_ms <= GSR_POSSIBLE_PASS_MS else ""),
                     "provenance": f"{GSR_TIMING.relative_to(ROOT)} sha256 {sha256_file(GSR_TIMING)[:12]}"})
    v15 = pd.read_csv(V15_SHOTS, usecols=["shot_id", "game", "contact_status", "contact_quality",
                                          "contact_lower_ms", "contact_upper_ms"])
    plan = pd.read_csv(PLAN, usecols=["game", "role"])
    v15 = v15[v15.contact_status == "bounded"].merge(plan, on="game", how="left")
    for r in v15.itertuples():
        rows.append({"source": "v15_ai_contact", "item": r.shot_id, "contact_lower_ms": r.contact_lower_ms,
                     "contact_upper_ms": r.contact_upper_ms, "before_o_max": r.contact_upper_ms <= O_MAX,
                     "note": f"{r.contact_quality}; xgv role {r.role}",
                     "provenance": f"{V15_SHOTS.relative_to(ROOT)} sha256 {sha256_file(V15_SHOTS)[:12]}"})
    table = pd.DataFrame(rows)
    if bad := [c for c in table.columns if any(w in c.lower() for w in OUTCOME_WORDS)]:
        raise AssertionError(f"outcome-like columns in the evidence table: {bad}")
    return table


def spread_check() -> dict:
    """D6: only the committed aggregate of timing_verdict.json, no per-shot residual."""
    stats = json.loads(TIMING_VERDICT.read_text())["statistics"]
    return {"source": str(TIMING_VERDICT.relative_to(ROOT)), "sha256": sha256_file(TIMING_VERDICT),
            "shot_share_within_200ms": stats["shot_loo_share_within_200ms"], "n_shots": stats["n_shots"],
            "reading": "spread of SoccerNet vs vendor shot times after the half offset; bounds spread, not contact "
                       "position; descriptive, does not change the rule"}


def cmd_evaluate(args: argparse.Namespace) -> None:
    rule_sha = check_rule()
    sample = load_sample()
    sheet_ids = sample.sheet_id.tolist()
    a, b = read_answers(args.run_a, sheet_ids), read_answers(args.run_b, sheet_ids)
    table, result = evaluate_runs(a, b)
    evidence = evidence_table()
    result["evidence"] = {"rows": len(evidence),
                          "gsr_kicks": int((evidence.source == "gsr_kick_rule").sum()),
                          "gsr_possible_pass": int(evidence.note.str.startswith("may be").sum()),
                          "v15_bounded": int((evidence.source == "v15_ai_contact").sum()),
                          "v15_before_o_max": int(evidence[evidence.source == "v15_ai_contact"].before_o_max.sum()),
                          "note": "descriptive only; 0 of 28 would give a 10.1% bound, so it is not part of the rule",
                          "cp_upper_0_of_28": cp_upper(0, 28)}
    result["spread_check_d6"] = spread_check()
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "answers.csv", index=False)
    evidence.to_csv(args.out / "evidence.csv", index=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    inputs = {str(p): sha256_file(p) for p in [*args.run_a, *args.run_b, SAMPLE, RULE, GSR_TIMING, V15_SHOTS,
                                                 TIMING_VERDICT, PLAN]}
    complete = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": records.git_state(),
                "rule": {"path": str(RULE.relative_to(ROOT)), "sha256": rule_sha},
                "runs": {"a": [str(p) for p in args.run_a], "b": [str(p) for p in args.run_b]},
                "inputs": inputs,
                "code": {str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve())},
                "outputs": {n: sha256_file(args.out / n) for n in ("answers.csv", "evidence.csv", "result.json")}}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("verdict", "violations", "n", "cp_upper_95")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--sample", type=Path, default=SAMPLE)
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("sheets")
    p.add_argument("--out", type=Path, default=SHEETS_OUT)
    p.add_argument("--limit", type=int, default=None, help="first N sheets only (a readability trial)")
    p.set_defaults(func=cmd_sheets)
    p = sub.add_parser("zoom", help="labeller aid: the same crop of the native frames at several offsets, magnified")
    p.add_argument("sheet_id")
    p.add_argument("--center", type=int, nargs=2, required=True, metavar=("X", "Y"), help="crop centre, 1280x720 px")
    p.add_argument("--width", type=int, default=320, help="crop width in native px (16:9); 320 = 2x magnification")
    p.add_argument("--offsets", type=int, nargs="+", default=list(QUESTIONS))
    p.add_argument("--frames", type=Path, default=SHEETS_OUT / "frames")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_zoom)
    p = sub.add_parser("evaluate")
    p.add_argument("--run-a", type=Path, nargs="+", required=True)
    p.add_argument("--run-b", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=RESULTS_OUT)
    p.set_defaults(func=cmd_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
