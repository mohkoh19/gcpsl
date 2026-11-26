# Reproducing the paper

## 1. Environment

The real-execution runs used PyTorch 2.6.0, torchvision 0.21.0 and CUDA 12.4 in one container,
with FP32 arithmetic, TF32 matrix multiplication and deterministic algorithms.  The `Dockerfile`
uses the same versions.  `requirements.lock` pins the remaining packages (Python 3.11).

```bash
export PYTHONHASHSEED=0      # required for deterministic runs
pytest -q                    # several minutes on CPU; spawns up to four processes
```

## 2. Real-execution study

Nine configurations, three seeds, four NVIDIA H100 GPUs on one node (one GPU for the
single-worker and larger-batch baselines):

```bash
for seed in 40 72 313; do
  for config in gcpsl_random gcpsl_size_balanced gcpsl_label_aware \
                gcpsl_label_aware_sequential gpsl_sharded_sync \
                gpsl_data_parallel gpsl_repartitioned; do
    torchrun --standalone --nproc-per-node 4 train_distributed.py \
      --config-name $config seed=$seed data_dir=/path/to/data
  done
  for config in gpsl_single_worker gpsl_larger_batch; do
    torchrun --standalone --nproc-per-node 1 train_distributed.py \
      --config-name $config seed=$seed data_dir=/path/to/data
  done
done
```

Run times, 100 epochs on H100 GPUs: about 24 minutes for GCPSL and the repartitioned baseline,
31 for data-parallel, 78 for sharded synchronous, 82 for sequential label-aware, 65 for
larger-batch and 99 for single-worker.

The learning rate 0.02 of the larger-batch and data-parallel baselines comes from a sweep over
0.005, 0.01, 0.02, 0.04 and 0.08 with 30 epochs, selected by validation accuracy at epoch 30:

```bash
torchrun --standalone --nproc-per-node 1 train_distributed.py --config-name gpsl_larger_batch \
  seed=40 epochs=30 learning_rate=0.04 output_dir=outputs/sweep/larger_batch_lr0.04 data_dir=/path/to/data
```

A run also prints the test accuracy at its end; the sweep must not use it.

## 3. Schedule-accounted studies

```bash
for seed in 40 72 313; do
  for config in configs/schedule_accounted/cifar*.yaml; do
    python train_schedule_accounted.py --config-name $(basename $config .yaml) \
      seed=$seed data_dir=/path/to/data
  done
done
```

All of them run on a single GPU.  The memory control ran ten epochs (`epochs=10`) of
`cifar10_b64_gpsl`, `cifar10_b64_label_aware_n4` and `cifar10_b512_gpsl` on an RTX A4000 while
sampling the device memory with `nvidia-smi`.  The `peak_gpu_bytes` value of this code is the
memory allocated by PyTorch and therefore a different quantity.

## 4. Tables and figures

```bash
python analysis/tables.py
python analysis/figures.py --output figures
```

Both read [`results/`](../results), which holds the per-run data behind every table and figure
of the paper ([column definitions](results_data.md)).

## What is reproduced

| Part | Status |
|---|---|
| Tables and figures | regenerated from `results/` |
| CIFAR-10 schedule-accounted runs (51) | produced with this code and the configurations in `configs/schedule_accounted` |
| CIFAR-100 schedule-accounted runs (12) | produced with the research implementation of the same algorithm; one full run of this code (label-aware GCPSL, seed 40) gives 68.18% against 68.05% final accuracy and reaches 60% after 4059 rounds in both. These runs reproduce statistically, not bit for bit |
| Real-execution runs (27) | measured on four H100 GPUs with the research implementation. This code reproduces its training state bit for bit on CPU for all nine configurations (three epochs, CIFAR-10 subset) and on one RTX A4000 for the single-GPU baselines |

Full 100-epoch real-execution runs with NCCL on several GPUs were not repeated with this code.
