"""Process-group setup and small collective helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def broadcast_object(self, value: Any) -> Any:
        if self.world_size == 1:
            return value
        payload = [value if self.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def all_reduce_max(self, value: float) -> float:
        if self.world_size == 1:
            return value
        tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def all_reduce_sum(self, values: list[float]) -> list[float]:
        if self.world_size == 1:
            return values
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()


def init_distributed(device: str = "auto", timeout_seconds: int = 600) -> DistributedContext:
    """Create the process group from the environment set by ``torchrun``.

    CUDA runs use NCCL with one process per GPU, CPU runs use Gloo.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("the distributed runs need one GPU per process")
        torch.cuda.set_device(local_rank)
        resolved = torch.device("cuda", local_rank)
    else:
        resolved = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if resolved.type == "cuda" else "gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(rank=rank, world_size=world_size, device=resolved)
