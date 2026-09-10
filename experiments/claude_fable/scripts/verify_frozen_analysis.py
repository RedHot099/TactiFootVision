"""Verify frozen analysis checksums without writing to its result directory."""

import argparse
import json
from pathlib import Path

from diagnostic_review import json_text
from xgpoc.v13.protocol import digest


def verify(root):
    frozen = json.loads((root / "frozen.json").read_text())
    evaluated = json.loads((root / "evaluated.json").read_text())
    complete = json.loads((root / "analysis_complete.json").read_text())
    checks = {root / "frozen.json": evaluated["frozen_sha256"], root / "protocol.json": frozen["protocol_sha256"],
              root / "evaluated.json": complete["evaluated_sha256"]}
    checks.update({Path(p): h for p, h in frozen["analysis"]["source_sha256"].items()})
    checks.update({root / p: h for p, h in complete["files"].items()})
    for variant in frozen["variants"].values():
        checks[root / variant["prediction_file"]] = variant["prediction_sha256"]
        checks.update({Path(p): h for p, h in variant["members"].items()})
    for path, checksum in checks.items():
        if digest(path) != checksum:
            raise ValueError(f"Frozen evidence changed: {path}")
    return {"root": str(root), "verified_files": len(checks), "status": "unchanged", "writes_to_frozen_directory": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("results/v14"), Path("results/v13_final")])
    parser.add_argument("--out", type=Path, default=Path("research/v15_diagnostics/frozen_integrity.json"))
    args = parser.parse_args()
    result = [verify(root) for root in args.roots]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json_text(result))
    print(json_text(result))


if __name__ == "__main__":
    main()
