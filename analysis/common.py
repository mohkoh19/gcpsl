"""Shared paths, labels and plotting style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REAL = RESULTS / "real_execution"
SCHEDULE = RESULTS / "schedule_accounted"

# Paul Tol's muted, color-blind-safe palette.
RANDOM, SIZE, LABEL, GRAY = "#999933", "#44AA99", "#882255", "#5F6368"
GRID = "#E6E6E6"

REAL_LABELS = {
    "gcpsl_random": "GCPSL, random",
    "gcpsl_size_balanced": "GCPSL, size-balanced",
    "gcpsl_label_aware": "GCPSL, label-aware",
    "gcpsl_label_aware_sequential": "GCPSL, label-aware, sequential",
    "gpsl_single_worker": "GPSL, single-worker",
    "gpsl_larger_batch": "GPSL, larger-batch",
    "gpsl_sharded_sync": "GPSL, sharded synchronous",
    "gpsl_data_parallel": "GPSL, data-parallel",
    "gpsl_repartitioned": "GPSL, repartitioned periodic",
}


def real_runs() -> pd.DataFrame:
    return pd.read_csv(REAL / "runs.csv")


def schedule_runs() -> pd.DataFrame:
    return pd.read_csv(SCHEDULE / "cifar10_runs.csv")


def mean_std(values, digits: int = 2, scale: float = 1.0) -> str:
    """Mean and sample standard deviation.  Missing values are unreached targets."""
    values = pd.Series(list(values), dtype=float) * scale
    reached = values.dropna()
    if reached.empty:
        return "--"
    if len(reached) == 1:
        text = f"{reached.iloc[0]:.{digits}f}"
    else:
        text = f"{reached.mean():.{digits}f} ± {reached.std(ddof=1):.{digits}f}"
    return text if len(reached) == len(values) else f"{text} ({len(reached)}/{len(values)} runs)"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
        }
    )


def style_axis(ax, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
