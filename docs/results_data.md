# Results data

Per-run data behind the paper's tables and figures.  Configuration names match the files in
`configs/`.  Times are in seconds, sizes in bytes, accuracies and rates are fractions.

## `results/real_execution/runs.csv`

27 runs: nine configurations, seeds 40, 72 and 313, on one node with four NVIDIA H100 GPUs.

| Column | Meaning |
|---|---|
| `workers` | GPUs allocated to the run |
| `final_test_accuracy` | test accuracy after epoch 100 |
| `best_validation_accuracy`, `best_validation_epoch` | best validation result |
| `best_checkpoint_test_accuracy` | test accuracy of that checkpoint |
| `direct_participation` | fraction of client–phase slots with direct data participation (π) |
| `model_update_participation` | fraction of slots in which the client's workload is active (μ) |
| `coverage`, `gini`, `mean_gap`, `p95_gap` | participation coverage, inequality and gaps in phases |
| `t80_*`, `t85_*`, `t88_*` | epoch, wall-clock time, allocated GPU time, logical payload and collective tensor bytes up to the validation target |
| `training_seconds`, `total_seconds` | critical-path training and total time |
| `*_rank_seconds` | time in one region, summed over workers |
| `*_collective_bytes`, `*_collective_rank_seconds` | tensors passed to collectives and measured collective time, summed over workers, 100 epochs |
| `peak_gpu_allocated_bytes_total` | peak PyTorch allocation, summed over GPUs |

## `results/schedule_accounted/`

| File | Content |
|---|---|
| `cifar10_runs.csv` | one row per CIFAR-10 run: rounds to 85% and 88%, final accuracy, inactivity, batch deviation, analytical payload |
| `cifar10_epochs.csv` | learning curves of all CIFAR-10 runs |
| `cifar100_runs.csv`, `cifar100_epochs.csv` | the same for CIFAR-100: rounds to 50% and 60%, final accuracy, inactivity, batch deviation |
| `peak_memory.csv` | peak device memory over ten epochs |

Empty target cells mean that the run did not reach the target.
