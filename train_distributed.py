"""Train one method of the real-execution study.

    torchrun --standalone --nproc-per-node 4 train_distributed.py --config-name gcpsl_size_balanced seed=40
"""

from __future__ import annotations

import os

# Required before CUDA is initialized for deterministic execution.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra  # noqa: E402
import torch.distributed as dist  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from gcpsl.distributed.context import init_distributed  # noqa: E402
from gcpsl.distributed.engine import DistributedRun, RunConfig  # noqa: E402


@hydra.main(version_base="1.3", config_path="configs/distributed", config_name="gcpsl_size_balanced")
def main(cfg: DictConfig) -> None:
    options = OmegaConf.to_container(cfg, resolve=True)
    device = options.pop("device")
    world_size = options.pop("world_size")
    context = init_distributed(device)
    if context.world_size != world_size:
        raise SystemExit(
            f"this configuration runs on {world_size} process(es), "
            f"but torchrun started {context.world_size}"
        )
    DistributedRun(RunConfig(**options), context).run()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
