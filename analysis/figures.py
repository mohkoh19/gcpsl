"""Recreate the result figures of the paper from the released per-run data.

    python analysis/figures.py --output figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    GRAY,
    LABEL,
    RANDOM,
    SCHEDULE,
    SIZE,
    real_runs,
    save,
    schedule_runs,
    setup_style,
    style_axis,
)
from matplotlib.patches import Patch


def operating_points(output: Path) -> None:
    """Time to 85% validation accuracy against direct data participation."""
    runs = real_runs()
    points = [
        ("gpsl_single_worker", "GPSL, 1 GPU", GRAY, "D"),
        ("gpsl_sharded_sync", "GPSL, sharded", GRAY, "s"),
        ("gpsl_data_parallel", "GPSL, data-parallel", GRAY, "^"),
        ("gcpsl_random", "GCPSL, random", RANDOM, "o"),
        ("gcpsl_size_balanced", "GCPSL, size-balanced", SIZE, "o"),
        ("gcpsl_label_aware", "GCPSL, label-aware", LABEL, "o"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for config, label, color, marker in points:
        g = runs[runs.config == config]
        ax.scatter(
            100 * g.direct_participation.mean(), g.t85_seconds.mean() / 60,
            s=90 if color == GRAY else 40, facecolors="none" if color == GRAY else color,
            edgecolors=color, marker=marker, linewidths=1.3, label=label, zorder=3,
        )
    ax.set_xlabel("direct data participation (%)")
    ax.set_ylabel("time to 85% (min)")
    ax.set_ylim(4.5, 12)
    style_axis(ax, "both")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, output, "operating_points")


def concurrency(output: Path) -> None:
    runs = real_runs()
    configs = ["gcpsl_label_aware", "gcpsl_label_aware_sequential"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    panels = [("t85_seconds", "time to 85% (min)"), ("t85_gpu_seconds", "GPU-worker time (min)")]
    for ax, (column, label) in zip(axes, panels, strict=True):
        values = [runs[runs.config == c][column].mean() / 60 for c in configs]
        bars = ax.bar(
            ["Concurrent", "Sequential"], values, color=[LABEL, "white"], edgecolor=LABEL,
            hatch=["", "///"], width=0.58, zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylabel(label)
        ax.margins(y=0.15)
        style_axis(ax)
    save(fig, output, "concurrency")


def learning_curves(output: Path) -> None:
    epochs = pd.read_csv(SCHEDULE / "cifar10_epochs.csv")
    styles = [("GPSL", "gpsl", GRAY, "-"), ("Size-balanced", "size_balanced_n{n}", SIZE, "--"),
              ("Label-aware", "label_aware_n{n}", LABEL, "-.")]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.9), sharey=True)
    for row, n in zip(axes, (4, 8), strict=True):
        axis_labels = {"cumulative_rounds": "schedule rounds", "processed_samples": "processed samples"}
        for ax, (x, label) in zip(row, axis_labels.items(), strict=True):
            for name, pattern, color, line in styles:
                g = epochs[epochs.config == "cifar10_b64_" + pattern.format(n=n)]
                curve = g.groupby("epoch").agg(
                    x=(x, "mean"), y=("test_accuracy", "mean"), s=("test_accuracy", "std")
                )
                ax.plot(curve.x, 100 * curve.y, color=color, linestyle=line, label=name, zorder=3)
                low, high = 100 * (curve.y - curve.s), 100 * (curve.y + curve.s)
                ax.fill_between(curve.x, low, high, color=color, alpha=0.08, linewidth=0)
            ax.set_ylim(80, 91.5)
            ax.set_xlim(0, 25_000 if x == "cumulative_rounds" else 5_000_000)
            ax.set_xlabel(label)
            ax.text(0.02, 0.96, f"$N={n}$", transform=ax.transAxes, va="top", fontweight="bold")
            style_axis(ax)
        row[0].set_ylabel("test accuracy (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output, "learning_curves")


def cluster_count(output: Path) -> None:
    runs = schedule_runs()
    panels = [
        ("rounds_to_85", 1e-3, "rounds to 85% ($10^3$)"),
        ("rounds_to_88", 1e-3, "rounds to 88% ($10^3$)"),
        ("final_test_accuracy", 100, "final accuracy (%)"),
        ("inactivity", 100, "inactivity (%)"),
    ]
    counts = np.array([2, 4, 8, 16])
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, (column, scale, label) in zip(axes, panels, strict=True):
        for offset, rule, color, line, marker, name in [
            (-0.1, "label_aware", LABEL, "-.", "^", "Label-aware"),
            (0.1, "size_balanced", SIZE, "--", "s", "Size-balanced"),
        ]:
            groups = [runs[runs.config == f"cifar10_b64_{rule}_n{n}"][column] * scale for n in counts]
            mean = [g.mean() if g.notna().all() else np.nan for g in groups]
            std = [g.std(ddof=1) for g in groups]
            ax.errorbar(
                np.arange(4) + offset, mean, yerr=std, color=color, linestyle=line, marker=marker,
                markersize=3.5, markerfacecolor="white", capsize=2, label=name,
            )
        ax.set_xticks(range(4), counts)
        ax.set_xlabel("clusters $N$")
        ax.set_ylabel(label)
        style_axis(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, output, "cluster_count")


def capacity(output: Path) -> None:
    runs, memory = schedule_runs(), pd.read_csv(SCHEDULE / "peak_memory.csv").set_index("config")
    bars = [
        ("cifar10_b64_gpsl", "GPSL\n$B=64$", GRAY),
        ("cifar10_b64_label_aware_n4", "Label-aware\n$N=4$", LABEL),
        ("cifar10_b64_size_balanced_n4", "Size-balanced\n$N=4$", SIZE),
        ("cifar10_b256_gpsl", "GPSL\n$B=256$", GRAY),
        ("cifar10_b512_gpsl", "GPSL\n$B=512$", GRAY),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.4), width_ratios=[2.25, 0.85])
    x = np.arange(len(bars))
    colors = [color for *_, color in bars]
    for shift, column, filled in [(-0.17, "rounds_to_85", True), (0.17, "rounds_to_88", False)]:
        values = [runs[runs.config == c][column].mean() / 1e3 for c, *_ in bars]
        left.bar(
            x + shift, values, 0.34, color=colors if filled else "white", edgecolor=colors,
            hatch="" if filled else "///", zorder=3,
        )
    left.set_xticks(x, [label for _, label, _ in bars])
    left.set_ylabel("target rounds ($10^3$)")
    left.legend(
        handles=[Patch(color="#7A7A7A", label="85%"),
                 Patch(facecolor="white", edgecolor="#7A7A7A", hatch="///", label="88%")],
        frameon=False,
    )
    style_axis(left)
    selected = memory.loc[["cifar10_b64_label_aware_n4", "cifar10_b512_gpsl"], "peak_gpu_memory_mib"]
    drawn = right.bar(["Label-aware\n$N=4$, $B=64$", "GPSL\n$B=512$"], selected / 1024, color=[LABEL, GRAY],
                      width=0.56, zorder=3)
    right.bar_label(drawn, [f"{v} MiB" for v in selected])
    right.set_ylabel("peak GPU memory (GiB)")
    right.margins(y=0.15)
    style_axis(right)
    fig.tight_layout()
    save(fig, output, "capacity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    directory = parser.parse_args().output
    setup_style()
    for figure in (operating_points, concurrency, learning_curves, cluster_count, capacity):
        figure(directory)
