"""Table and figure helpers shared by run_research.py and the notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import performance_metrics
from src.utils.config import DATA_CONFIG

ANN = DATA_CONFIG.annualization_factor

METRIC_ORDER = [
    "sharpe_ratio", "sortino_ratio", "cagr", "volatility", "max_drawdown",
    "calmar_ratio", "var_95", "expected_shortfall_95",
]


def metrics_table(named_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Performance metrics for several return streams side by side (rows = names)."""

    table = pd.DataFrame({name: performance_metrics(r.dropna()) for name, r in named_returns.items()}).T
    table["start"] = pd.Series({name: str(r.dropna().index.min().date()) for name, r in named_returns.items()})
    table["years"] = pd.Series({name: round(len(r.dropna()) / ANN, 1) for name, r in named_returns.items()})
    return table[METRIC_ORDER + ["start", "years"]]


def latex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def save_table(
    frame: pd.DataFrame,
    name: str,
    results_dir: Path,
    tables_dir: Path,
    float_fmt: str = "%.3f",
    column_formats: dict[str, str] | None = None,
) -> None:
    """Write a table as CSV (results/) and as a LaTeX tabular body (docs/tables/)."""

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / f"{name}.csv")

    column_formats = column_formats or {}

    def fmt(value: object, column: str) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        if isinstance(value, (int, np.integer)) and column not in column_formats:
            return str(value)
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return "--"
            return column_formats.get(column, float_fmt) % value
        return latex_escape(value)

    index_names = [n for n in (frame.index.names or []) if n]
    n_index = frame.index.nlevels
    columns = [latex_escape(c) for c in frame.columns]
    header = [latex_escape(n) for n in index_names] if index_names else [""] * n_index
    lines = [
        r"\begin{tabular}{" + "l" * n_index + "r" * len(columns) + "}",
        r"\toprule",
        " & ".join(header + columns) + r" \\",
        r"\midrule",
    ]
    for index_value, row in frame.iterrows():
        index_cells = list(index_value) if isinstance(index_value, tuple) else [index_value]
        cells = [latex_escape(v) for v in index_cells] + [fmt(v, c) for c, v in zip(frame.columns, row)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axis(ax) -> None:
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
