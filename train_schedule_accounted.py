"""Train one configuration of the schedule-accounted studies on a single device.

    python train_schedule_accounted.py --config-name cifar10_b64_size_balanced_n4 seed=40
"""

from __future__ import annotations

import os

# Required before CUDA is initialized for deterministic execution.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from gcpsl.schedule_accounted.trainer import ScheduleAccountedRun, ScheduleConfig  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path="configs/schedule_accounted",
    config_name="cifar10_b64_size_balanced_n4",
)
def main(cfg: DictConfig) -> None:
    ScheduleAccountedRun(ScheduleConfig(**OmegaConf.to_container(cfg, resolve=True))).run()


if __name__ == "__main__":
    main()
