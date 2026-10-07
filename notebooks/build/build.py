"""Generate the showcase notebooks and optionally execute them.

    uv run python notebooks/build/build.py [--execute] [names...]

Names select notebooks by prefix (``01``, ``01_data``, ``data`` all pick
``01_data``); none means all. The cells are formatted with ruff. With
``--execute`` each notebook runs with ``nbclient`` (kernel working directory
``notebooks/``) and is saved with its outputs, also when a cell fails, times
out or kills the kernel; the remaining notebooks still run.
"""

import argparse
import subprocess
import sys
import time

import nb00_overview
import nb01_data
import nb02_augmentation
import nb03_training_and_evaluation
import nb04_inference
import nb05_visualisation
import nbclient
import nbformat
from common import NOTEBOOK_DIR

GENERATORS = (
    nb00_overview,
    nb01_data,
    nb02_augmentation,
    nb03_training_and_evaluation,
    nb04_inference,
    nb05_visualisation,
)
CELL_TIMEOUT = 1800  # seconds; the longest cell is a demo training run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="notebooks to build (default: all)")
    parser.add_argument(
        "--execute", action="store_true", help="run the notebooks and save outputs"
    )
    args = parser.parse_args()

    notebooks = [generator.notebook() for generator in GENERATORS]
    selected = [nb for nb in notebooks if _selected(nb.name, args.names)]
    unknown = [
        name
        for name in args.names
        if not any(_selected(nb.name, [name]) for nb in notebooks)
    ]
    if unknown:
        parser.error(
            f"unknown notebook(s) {unknown}; choose from {[nb.name for nb in notebooks]}"
        )

    failed = 0
    for notebook in selected:
        nbformat.write(notebook.to_node(), notebook.path)
        # Cells follow the repository's code style, like every other Python file.
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--quiet", str(notebook.path)],
            check=True,
        )
        if not args.execute:
            print(f"wrote {notebook.path.relative_to(NOTEBOOK_DIR.parent)}")
            continue
        node = nbformat.read(notebook.path, as_version=4)
        ok, seconds, error = _execute(node)
        nbformat.write(node, notebook.path)
        failed += not ok
        size = notebook.path.stat().st_size / 1e6
        status = "ok" if ok else "FAILED"
        print(f"{notebook.name:32s} {status:6s} {seconds:6.0f} s {size:5.2f} MB")
        if error:
            print(error, file=sys.stderr)
    return 1 if failed else 0


def _selected(name: str, wanted: list[str]) -> bool:
    if not wanted:
        return True
    short = name.split("_", 1)[1]
    return any(name.startswith(w) or short.startswith(w) for w in wanted)


def _execute(node: nbformat.NotebookNode) -> tuple[bool, float, str]:
    client = nbclient.NotebookClient(
        node,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    start = time.perf_counter()
    try:
        client.execute()
    # CellControlSignal covers a failing cell (CellExecutionError) and a cell
    # timeout; a dead kernel (e.g. out of memory) is a DeadKernelError.
    except (
        nbclient.exceptions.CellControlSignal,
        nbclient.exceptions.DeadKernelError,
    ) as error:
        message = f"{type(error).__name__}: {error}"[-3000:]
        return False, time.perf_counter() - start, message
    return True, time.perf_counter() - start, ""


if __name__ == "__main__":
    sys.exit(main())
