"""The xg-vendor pre-registration as data (ticket 14): its parameter block, the hash list and the records.

The Polish document `research/xg_vendor_preregistration.md` is the authority. It holds one JSON block between the
markers `PARAMETERS_BEGIN` / `PARAMETERS_END`: the analysis parameters (margin, draws, seeds, offset logits, H3
thresholds, confirmatory targets and control-gate threshold, the bootstrap of the recalibrated rungs, the frame exclusions, the dry-run
permutation seed and margin) and the pins (path and sha256 of every frozen input). `read_parameters` parses and checks
it; a `null` margin or a pin without a hash is a placeholder. The owner's choices before the label join are parameters
of this block, so a changed choice changes the document and, through the dry-run check below, forces new dry runs and
a new hash list.

The hash list (`hash_list`) is a deterministic JSON object, without timestamps, so a rerun on unchanged files gives the
same bytes:
    parameters   the parsed block
    pins         every pinned file with its recorded and its current sha256 (they must agree)
    code         sha256 of every source file of the confirmatory run: the entry scripts and every module of the
                 `xgpoc` package they import, transitively (`code_closure`), package `__init__` files included
    evidence     completion records of the dry runs; a missing one is a placeholder, and so is a stale one: a dry run
                 is evidence only when it ran the current code closure, the current parameters (pins aside) and the
                 table its pin names (`evidence_mismatch`)
    placeholders what still has to be filled; `status` is "complete" only when the list is empty
The document prints the hash list's own sha256 and its pins and code hashes between `HASHES_BEGIN` / `HASHES_END`
(`render_hash_block`), so the document's sha256 pins everything transitively. The pre-registration record
(`evaltable.PreRegistration`) is the document's sha256, its commit time and the parameters; `confirmatory_record`
builds it from a committed document, `draft_record` from a draft one (the dry run only).
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from xgpoc.bundle import sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor.leakage import OFFSET_LOGITS
from xgpoc.vendor.probe_targets import TARGET_COLUMNS

SCHEMA = "xgv-prereg-hashes-1"
PARAMETERS_BEGIN, PARAMETERS_END = "<!-- prereg-parameters:begin -->", "<!-- prereg-parameters:end -->"
HASHES_BEGIN, HASHES_END = "<!-- prereg-hashes:begin -->", "<!-- prereg-hashes:end -->"
PLACEHOLDER_TOKEN = "DO UZUPEŁNIENIA"
PARAMETER_FIELDS = ("margin", "n_boot", "fold_seed", "boot_seed", "offset_logits", "h3_thresholds", "h3_confirmatory_targets",
                    "h3_control_max_r2", "recalibrated_bootstrap", "frame_exclusions", "permutation_seed", "dry_run_margin", "pins")
REQUIRED_PINS = ("primary_label_free_table", "combined_label_free_table", "split_plan", "comparator_predictions", "timing_verdict",
                 "timing_rule", "delta_rule")
TABLE_PINS = ("primary_label_free_table", "combined_label_free_table")
ENTRY_SCRIPTS = ("scripts/vendor_confirm.py", "scripts/vendor_ladder.py", "scripts/vendor_probes.py", "scripts/vendor_leakage.py",
                 "scripts/vendor_dryrun.py", "scripts/vendor_prereg_hashes.py")
EVIDENCE = {"dry_run_primary": "results/xgv_dryrun_primary/complete.json", "dry_run_combined": "results/xgv_dryrun_combined/complete.json"}
EVIDENCE_TABLE_PINS = {"dry_run_primary": "primary_label_free_table", "dry_run_combined": "combined_label_free_table"}


class PreRegistrationError(ValueError):
    """The document, its parameter block or a pinned file breaks the pre-registration contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreRegistrationError(message)


def _block(text: str, begin: str, end: str) -> tuple[int, int]:
    """Start and end offsets of the text strictly between the two markers (each must occur exactly once)."""
    _require(text.count(begin) == 1 and text.count(end) == 1, f"the document needs exactly one {begin} and one {end}")
    start, stop = text.index(begin) + len(begin), text.index(end)
    _require(start <= stop, f"{end} comes before {begin}")
    return start, stop


# ------------------------------------------------------------------ parameters
def _is_margin(value) -> bool:
    return isinstance(value, float) and 0 < value <= 0.01


def read_parameters(text: str) -> dict:
    """The parameter block of the document text, checked (module docstring)."""
    start, stop = _block(text, PARAMETERS_BEGIN, PARAMETERS_END)
    fenced = re.fullmatch(r"\s*```json\n(.*)\n```\s*", text[start:stop], re.DOTALL)
    _require(fenced is not None, "the parameter block must be one ```json fence")
    params = json.loads(fenced.group(1))
    _require(set(params) == set(PARAMETER_FIELDS), f"the parameter block needs exactly the fields {PARAMETER_FIELDS}")
    _require(params["margin"] is None or _is_margin(params["margin"]), "margin must be a log-loss margin in (0, 0.01] or null")
    _require(_is_margin(params["dry_run_margin"]), "dry_run_margin must be a log-loss margin in (0, 0.01]")
    for name in ("n_boot", "fold_seed", "boot_seed", "permutation_seed"):
        _require(isinstance(params[name], int) and not isinstance(params[name], bool) and params[name] >= 0, f"{name} must be an integer >= 0")
    _require(params["n_boot"] >= 5000, "the spec fixes at least 5,000 match draws")
    _require(params["offset_logits"] in OFFSET_LOGITS, f"offset_logits must be one of {sorted(OFFSET_LOGITS)}")
    _require(set(params["h3_thresholds"]) == set(TARGET_COLUMNS), f"h3_thresholds needs exactly the targets {TARGET_COLUMNS}")
    _require(set(params["h3_confirmatory_targets"]) <= set(TARGET_COLUMNS), "unknown confirmatory H3 target")
    gate = params["h3_control_max_r2"]
    _require(isinstance(gate, int | float) and not isinstance(gate, bool) and 0 <= gate <= et.MAX_CONTROL_R2_LIMIT,
             f"h3_control_max_r2 must be an R^2 in [0, {et.MAX_CONTROL_R2_LIMIT}]")
    _require(params["recalibrated_bootstrap"] in et.RECALIBRATED_BOOTSTRAPS, f"recalibrated_bootstrap must be one of {et.RECALIBRATED_BOOTSTRAPS}")
    exclusions = params["frame_exclusions"]
    _require(isinstance(exclusions, dict) and set(exclusions.values()) <= set(et.FRAME_EXCLUSIONS),
             f"frame_exclusions must map row keys to reason codes of {sorted(et.FRAME_EXCLUSIONS)}")
    _require(set(params["pins"]) >= set(REQUIRED_PINS), f"pins must include {REQUIRED_PINS}")
    for name, pin in params["pins"].items():
        _require(set(pin) == {"path", "sha256"}, f"pin {name} needs exactly path and sha256")
        _require((pin["path"] is None) == (pin["sha256"] is None), f"pin {name}: path and sha256 are both set or both null")
        _require(pin["sha256"] is None or et.is_sha256(pin["sha256"]), f"pin {name}: not a sha256 hex digest")
    return params


def placeholders(text: str, params: dict) -> list[str]:
    """What the document still leaves open: a null margin, pins without a hash and placeholder marks in the prose (the
    generated hash block is not searched)."""
    start, stop = _block(text, HASHES_BEGIN, HASHES_END)
    prose = text[:start] + text[stop:]
    open_items = ["margin"] if params["margin"] is None else []
    open_items += [f"pin {name}" for name, pin in params["pins"].items() if pin["sha256"] is None]
    marks = prose.count(PLACEHOLDER_TOKEN)
    return open_items + ([f"{marks} '{PLACEHOLDER_TOKEN}' marks in the document"] if marks else [])


# ------------------------------------------------------------------ code closure
def _module_file(root: Path, dotted: str) -> Path | None:
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(path: Path, root: Path) -> set[Path]:
    """Files of the `xgpoc` modules a source file imports (a `from package import name` counts the submodule `name`
    when it is one), and the `__init__` of every package on the way."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        dotted = []
        if isinstance(node, ast.Import):
            dotted = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            dotted = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        for name in dotted:
            if name.split(".")[0] != "xgpoc":
                continue
            parts = name.split(".")
            for k in range(1, len(parts) + 1):
                module = _module_file(root, ".".join(parts[:k]))
                if module is not None:
                    found.add(module)
    return found


def code_closure(root: Path, entries: tuple[str, ...] = ENTRY_SCRIPTS) -> list[Path]:
    """The entry scripts and every `xgpoc` source file they import, transitively, sorted."""
    root = Path(root)
    todo = [root / e for e in entries]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        _require(path.is_file(), f"{path} does not exist")
        seen.add(path)
        todo += sorted(_imports(path, root) - seen)
    return sorted(seen)


# ------------------------------------------------------------------ hash list
def _resolve(root: Path, path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else Path(root) / path


def check_frame_exclusions(root: Path, params: dict) -> None:
    """Every key of `frame_exclusions` must be a secondary row with a scored frame of the pinned combined label-free
    table (only its key, population and frame status are read)."""
    if not params["frame_exclusions"]:
        return
    pin = params["pins"]["combined_label_free_table"]
    _require(pin["sha256"] is not None, "frame_exclusions need the pinned combined label-free table")
    rows = pd.read_parquet(_resolve(root, pin["path"]), columns=["key", "population", "frame_status"]).set_index("key")
    _require(rows.index.is_unique, "the pinned combined label-free table has duplicate keys")
    for key in params["frame_exclusions"]:
        _require(key in rows.index, f"frame exclusion {key} is not a row of the combined label-free table")
        _require(rows.population[key] != "primary" and rows.frame_status[key] == et.OK, f"frame exclusion {key} must be a secondary row with a scored frame")


def evidence_mismatch(record: object, code: dict, params: dict, table_sha256: str | None) -> list[str]:
    """Why a dry run's completion record (parsed JSON, or None) is not evidence for the current document and code: it
    must have run exactly the code closure `code`, the parameters of the block (pins aside) and the label-free table
    `table_sha256`. Empty when it is evidence."""
    identity = record.get("identity") if isinstance(record, dict) else None
    if not isinstance(identity, dict):
        return ["not a dry-run completion record"]
    ran = {k: v for k, v in (identity.get("parameters") or {}).items() if k != "margin_used"}
    reasons = [] if identity.get("sources") == code else ["analysis code differs"]
    reasons += [] if ran == {k: v for k, v in params.items() if k != "pins"} else ["parameters differ"]
    reasons += [] if (identity.get("label_free_table") or {}).get("sha256") == table_sha256 else ["table differs from its pin"]
    return reasons


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def hash_list(root: Path, document: Path, entries: tuple[str, ...] = ENTRY_SCRIPTS) -> dict:
    """The hash list of the document (module docstring); `entries` are the entry scripts of the code closure. Raises when
    a pinned file has another hash than its pin or a frame exclusion is not a scored secondary row of the pinned table."""
    root, text = Path(root), Path(document).read_text()
    params = read_parameters(text)
    pins = {}
    for name, pin in sorted(params["pins"].items()):
        if pin["sha256"] is None:
            pins[name] = {"path": None, "sha256": None}
            continue
        actual = sha256_file(_resolve(root, pin["path"]))
        _require(actual == pin["sha256"], f"pin {name}: {pin['path']} has sha256 {actual}, the document pins {pin['sha256']}")
        pins[name] = dict(pin)
    check_frame_exclusions(root, params)
    code = {str(p.relative_to(root)): sha256_file(p) for p in code_closure(root, entries)}
    evidence, open_evidence = {}, []
    for name, path in EVIDENCE.items():
        found = (root / path).is_file()
        evidence[name] = {"path": path, "sha256": sha256_file(root / path) if found else None}
        stale = evidence_mismatch(_read_json(root / path), code, params, params["pins"][EVIDENCE_TABLE_PINS[name]]["sha256"]) if found else []
        if not found:
            open_evidence.append(f"evidence {name}")
        elif stale:
            open_evidence.append(f"evidence {name} is stale ({', '.join(stale)}); rerun the dry run")
    open_items = placeholders(text, params) + open_evidence
    return {"schema": SCHEMA, "status": "draft" if open_items else "complete", "document": str(Path(document).resolve().relative_to(root)),
            "parameters": params, "pins": pins, "code": code, "evidence": evidence, "placeholders": open_items}


def hash_list_bytes(hashes: dict) -> bytes:
    """The one serialisation of the hash list (its sha256 is what the document prints)."""
    return (json.dumps(hashes, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def render_hash_block(hashes: dict, hashes_path: str, hashes_sha256: str) -> str:
    """The generated section of the document between `HASHES_BEGIN` and `HASHES_END`."""
    status = "KOMPLETNA" if hashes["status"] == "complete" else "SZKIC"
    lines = ["", f"Lista hashy: `{hashes_path}`, sha256 `{hashes_sha256}`, status: **{status}**.", "",
             "| Wejście | Plik | sha256 |", "|---|---|---|"]
    lines += [f"| {name} | {pin['path'] or '—'} | {('`' + pin['sha256'] + '`') if pin['sha256'] else PLACEHOLDER_TOKEN.lower()} |"
              for name, pin in hashes["pins"].items()]
    lines += [f"| dowód: {name} | {e['path']} | {('`' + e['sha256'] + '`') if e['sha256'] else 'brak (do wykonania)'} |"
              for name, e in hashes["evidence"].items()]
    lines += ["", f"Kod analizy ({len(hashes['code'])} plików):", "", "| Plik | sha256 |", "|---|---|"]
    lines += [f"| {path} | `{sha}` |" for path, sha in hashes["code"].items()]
    if hashes["placeholders"]:
        lines += ["", "Otwarte pozycje: " + "; ".join(hashes["placeholders"]) + "."]
    return "\n".join(lines) + "\n"


def replace_hash_block(text: str, block: str) -> str:
    start, stop = _block(text, HASHES_BEGIN, HASHES_END)
    return text[:start] + block + text[stop:]


# ------------------------------------------------------------------ records
def _record(document_sha256: str, recorded_at: datetime, table_sha256: str, margin: float, params: dict, draft: bool,
            commit: str | None) -> et.PreRegistration:
    timing_rule = params["pins"]["timing_rule"]["sha256"]
    _require(timing_rule is not None, "the pin timing_rule is still a placeholder")
    return et.PreRegistration(sha256=document_sha256, recorded_at=recorded_at, label_free_table_sha256=table_sha256, margin=margin,
                              n_boot=params["n_boot"], fold_seed=params["fold_seed"], boot_seed=params["boot_seed"],
                              offset_logits=params["offset_logits"], h3_thresholds=dict(params["h3_thresholds"]),
                              h3_confirmatory_targets=tuple(params["h3_confirmatory_targets"]),
                              h3_control_max_r2=float(params["h3_control_max_r2"]), timing_rule_sha256=timing_rule,
                              recalibrated_bootstrap=params["recalibrated_bootstrap"], frame_exclusions=dict(params["frame_exclusions"]),
                              draft=draft, commit=commit)


def draft_record(document: Path, table_pin: str, now: datetime) -> et.PreRegistration:
    """A draft record of the current document for a dry run: the pinned table `table_pin`, the margin or, while it is a
    placeholder, the dry-run margin. It never admits a confirmatory join."""
    text = Path(document).read_text()
    params = read_parameters(text)
    _require(table_pin in TABLE_PINS, f"unknown table pin {table_pin}")
    table_sha = params["pins"][table_pin]["sha256"]
    _require(table_sha is not None, f"the pin {table_pin} is still a placeholder")
    margin = params["margin"] if params["margin"] is not None else params["dry_run_margin"]
    return _record(sha256_file(document), now, table_sha, margin, params, True, None)


def confirmatory_record(document: Path, hashes: dict, committed_at: datetime, commit: str) -> et.PreRegistration:
    """The record of a committed, complete document: its sha256, the commit time, the combined label-free table it pins
    and its parameters. The caller checks that the committed bytes are the working file's."""
    _require(hashes["status"] == "complete", f"the hash list is a draft: {hashes['placeholders']}")
    params = hashes["parameters"]
    return _record(sha256_file(document), committed_at, params["pins"]["combined_label_free_table"]["sha256"], params["margin"], params,
                   False, commit)
