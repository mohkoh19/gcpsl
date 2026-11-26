"""Print the result tables of the paper from the released per-run data.

    python analysis/tables.py
"""

from __future__ import annotations

import pandas as pd
from common import REAL_LABELS, SCHEDULE, mean_std, real_runs, schedule_runs

TRAINING_REGIONS = [
    "schedule", "data_wait", "host_to_device", "training_compute", "gradient_sync",
    "optimizer", "barrier_wait", "fusion", "replica_check",
]


def show(title: str, rows: list[dict]) -> None:
    table = pd.DataFrame(rows).astype(str)
    widths = {c: max(len(c), *table[c].str.len()) for c in table}
    print(f"\n{title}\n")
    print("  ".join(c.ljust(widths[c]) for c in table))
    for _, row in table.iterrows():
        print("  ".join(row[c].ljust(widths[c]) for c in table))


def real_execution(runs: pd.DataFrame) -> None:
    rows, targets, timing, sync = [], [], [], []
    for config, label in REAL_LABELS.items():
        g = runs[runs.config == config]
        rows.append(
            {
                "Method": label,
                "GPUs": g.workers.iloc[0],
                "t85 (min)": mean_std(g.t85_seconds, scale=1 / 60),
                "GPU-min": f"{g.t85_gpu_seconds.mean() / 60:.2f}",
                "Acc. (%)": f"{100 * g.final_test_accuracy.mean():.2f}",
                "Best-val (%)": f"{100 * g.best_checkpoint_test_accuracy.mean():.2f}",
                "Part. (%)": f"{100 * g.direct_participation.mean():.2f}",
                "Payload (TB)": f"{g.t85_logical_payload_bytes.mean() / 1e12:.3f}",
            }
        )
        if not config.endswith("sequential"):
            minutes = {f"{q}% (min)": mean_std(g[f"t{q}_seconds"], scale=1 / 60) for q in (80, 85, 88)}
            targets.append({"Method": label, **minutes})
        regions = g[[f"{name}_rank_seconds" for name in TRAINING_REGIONS]].sum(axis=1)
        if config in {"gcpsl_random", "gcpsl_size_balanced", "gcpsl_label_aware", "gpsl_data_parallel"}:
            share = (g.fusion_rank_seconds + g.gradient_sync_rank_seconds) / regions
            timing.append(
                {
                    "Method": label,
                    "Epoch (s)": f"{g.training_seconds.mean() / 100:.2f}",
                    "Barrier (%)": f"{100 * (g.barrier_wait_rank_seconds / regions).mean():.1f}",
                    "Sync. (%)": f"{100 * share.mean():.1f}",
                    "Epochs to 85%": mean_std(g.t85_epoch, 1),
                }
            )
        if g.workers.iloc[0] > 1 and not config.endswith("sequential"):
            names = ("fusion", "gradient_sync", "batchnorm")
            volume = g[[f"{n}_collective_bytes" for n in names]].sum(axis=1)
            seconds = g[[f"{n}_collective_rank_seconds" for n in names]].sum(axis=1) / g.workers
            sync.append(
                {
                    "Method": label,
                    "Volume (GB)": f"{volume.mean() / 1e9:,.1f}",
                    "To 85% (GB)": f"{g.t85_collective_bytes.mean() / 1e9:,.1f}",
                    "Time (s)": f"{seconds.mean():.1f}",
                }
            )
    show("Real execution, CIFAR-10, three runs", rows)
    show("Time to validation targets", targets)
    show("Training time per epoch and overhead", timing)
    show("Cross-GPU synchronization over 100 epochs", sync)
    participation = []
    for config, label in REAL_LABELS.items():
        g = runs[runs.config == config]
        pi, mu = g.direct_participation.mean(), g.model_update_participation.mean()
        participation.append(
            {
                "Method": label,
                "pi (%)": f"{100 * pi:.2f}",
                "mu (%)": f"{100 * mu:.2f}",
                "pi/mu (%)": f"{100 * pi / mu:.2f}",
                "mean gap": f"{g.mean_gap.mean():.2f}",
                "p95 gap": f"{g.p95_gap.mean():.0f}",
                "Gini": f"{g.gini.mean():.3f}",
            }
        )
    show("Direct data participation", participation)


def schedule_accounted(runs: pd.DataFrame) -> None:
    def row(config: str) -> dict:
        g = runs[runs.config == config]
        return {
            "Configuration": config,
            "Acc. (%)": mean_std(g.final_test_accuracy, scale=100),
            "R85 (10^3)": mean_std(g.rounds_to_85, scale=1e-3),
            "R88 (10^3)": mean_std(g.rounds_to_88, scale=1e-3),
            "Inactivity (%)": mean_std(g.inactivity, scale=100),
            "Batch deviation": f"{g.batch_deviation.mean():.2f}",
        }

    show("Schedule-accounted studies, CIFAR-10, three runs", [row(c) for c in sorted(runs.config.unique())])

    payload = runs[runs.config.str.startswith("cifar10_b64_") & ~runs.config.str.contains("n8_")]
    payload = payload.groupby("config", as_index=False).mean(numeric_only=True)
    show(
        "Analytical payload, single-process accounting",
        [
            {
                "Configuration": r.config,
                "GB per round": f"{r.payload_bytes_per_round / 1e9:.3f}",
                "TB to 85%": f"{r.payload_bytes_to_85 / 1e12:.3f}",
                "TB to 88%": f"{r.payload_bytes_to_88 / 1e12:.3f}",
            }
            for r in payload.itertuples()
        ],
    )
    memory = pd.read_csv(SCHEDULE / "peak_memory.csv")
    show("Peak GPU memory", memory.to_dict("records"))


def cifar100() -> None:
    runs = pd.read_csv(SCHEDULE / "cifar100_runs.csv")
    rows = []
    for config, g in runs.groupby("config"):
        rows.append(
            {
                "Configuration": config,
                "Acc. (%)": mean_std(g.final_test_accuracy, scale=100),
                "R50 (10^3)": mean_std(g.rounds_to_50, scale=1e-3),
                "R60 (10^3)": mean_std(g.rounds_to_60, scale=1e-3),
            }
        )
    show("Schedule-accounted studies, CIFAR-100, three runs", rows)


if __name__ == "__main__":
    real_execution(real_runs())
    schedule_accounted(schedule_runs())
    if (SCHEDULE / "cifar100_runs.csv").exists():
        cifar100()
