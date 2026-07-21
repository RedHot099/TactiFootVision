from pathlib import Path

import pandas as pd


def write_accuracy_plot_data(rows: list[dict[str, object]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
