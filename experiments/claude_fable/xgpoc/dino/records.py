"""Code identity of the xg-dino runs: the import closure the protocol pins.

`package_hashes` follows every `import xgpoc...` / `from xgpoc... import ...` statement (anywhere in a file, so lazy
imports count) from the ablation's modules (`ROOTS`, not the whole of `xgpoc/dino`: other tickets add modules there)
and the run script, statically, so the digest does not depend on what a process
happened to import. Each package's `__init__.py` on the way is included. The run script is hashed with its one
`PROTOCOL_SHA256` line normalised: pinning the protocol after the freeze leaves the digest unchanged, any other edit
changes it.
"""

import ast
import hashlib
import re
from pathlib import Path

from xgpoc.bundle import canonical_digest, sha256_file

EXPERIMENT = Path(__file__).resolve().parents[2]          # experiments/claude_fable
RUN_SCRIPT = "scripts/xgdino_ablation.py"
ROOTS = ("xgpoc/dino/__init__.py", "xgpoc/dino/ablation.py", "xgpoc/dino/controls.py", "xgpoc/dino/records.py")
PIN = re.compile(r"^PROTOCOL_SHA256: str \| None = .*$", re.MULTILINE)
PIN_NORMALISED = "PROTOCOL_SHA256: str | None = <pin>"


def module_files(module: str) -> list[Path]:
    """The files that importing `module` (an xgpoc dotted name) executes: each package __init__ and the module."""
    parts = module.split(".")
    out = []
    for i in range(1, len(parts) + 1):
        base = EXPERIMENT.joinpath(*parts[:i])
        if (base / "__init__.py").exists():
            out.append(base / "__init__.py")
        elif base.with_suffix(".py").exists():
            out.append(base.with_suffix(".py"))
    return out


def imported_modules(path: Path) -> set[str]:
    """The xgpoc modules `path` imports (`from xgpoc.a import b` counts xgpoc.a and, if it is a module, xgpoc.a.b)."""
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text(), str(path))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "xgpoc"}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] == "xgpoc":
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def import_closure(roots: list[Path]) -> list[Path]:
    """Every xgpoc source file reachable from `roots` by import statements, sorted (the roots included)."""
    seen: set[Path] = set()
    stack = [Path(r).resolve() for r in roots]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in imported_modules(path):
            stack += [f.resolve() for f in module_files(module) if f.resolve() not in seen]
    return sorted(seen)


def run_script_sha256(path: Path) -> str:
    text, n = PIN.subn(PIN_NORMALISED, Path(path).read_text())
    if n != 1:
        raise ValueError(f"{path} must assign PROTOCOL_SHA256 exactly once on its own line, found {n}")
    return hashlib.sha256(text.encode()).hexdigest()


def package_hashes(script: Path = EXPERIMENT / RUN_SCRIPT) -> dict:
    """sha256 of every file of the import closure of `ROOTS` and the run script (the script pin-normalised), and their
    digest."""
    script = Path(script).resolve()
    roots = [*(EXPERIMENT / name for name in ROOTS), script]
    hashes = {str(p.relative_to(EXPERIMENT)): sha256_file(p) for p in import_closure(roots) if p != script}
    hashes[f"{script.relative_to(EXPERIMENT)} (pin normalised)"] = run_script_sha256(script)
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}
