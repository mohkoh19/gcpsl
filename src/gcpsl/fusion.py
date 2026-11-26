"""Sample-weighted fusion of split-model replicas.

Parameters and BatchNorm running statistics are averaged with weights proportional to
the number of training examples of each cluster, accumulated in float64.  BatchNorm
batch counters take the maximum.  Optimizer states are not fused.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn

from gcpsl.models.split import combined_state
from gcpsl.normalization import CollectiveCounter
from gcpsl.seeding import state_sha256


def fusion_weights(sample_counts: Sequence[int]) -> list[float]:
    # The order of floating-point operations is fixed, because weights that differ in the
    # last bit change the fused FP32 parameters.
    largest = max(float(count) for count in sample_counts)
    scaled = [float(count) / largest for count in sample_counts]
    total = math.fsum(scaled)
    return [value / total for value in scaled]


def fuse_states(
    states: Sequence[dict[str, torch.Tensor]], sample_counts: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Fuse complete model states held by one process."""
    weights = fusion_weights(sample_counts)
    fused: dict[str, torch.Tensor] = {}
    for name in sorted(states[0]):
        values = [state[name].detach() for state in states]
        if torch.is_floating_point(values[0]):
            accumulator = torch.zeros_like(values[0], dtype=torch.float64)
            for value, weight in zip(values, weights, strict=True):
                accumulator.add_(value.to(torch.float64), alpha=weight)
            fused[name] = accumulator.to(values[0].dtype)
        else:
            fused[name] = torch.stack(list(values)).max(dim=0).values
    return fused


def _sync(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def fuse_across_ranks(
    client: nn.Module, server: nn.Module, sample_counts: Sequence[int], counter: CollectiveCounter
) -> None:
    """Reduce all replicas to rank 0, broadcast the result, and verify that it is identical."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    rank = dist.get_rank()
    weight = fusion_weights(sample_counts)[rank]
    state = combined_state(client, server)

    def timed(action, tensor: torch.Tensor) -> None:
        _sync(tensor)
        start = time.perf_counter()
        action()
        _sync(tensor)
        counter.add(tensor, time.perf_counter() - start)

    with torch.no_grad():
        reduced: dict[str, torch.Tensor] = {}
        for name in sorted(state):
            value = state[name]
            if torch.is_floating_point(value):
                buffer = value.detach().to(torch.float64, copy=True).mul_(weight)
                timed(lambda b=buffer: dist.reduce(b, dst=0, op=dist.ReduceOp.SUM), buffer)
            else:
                buffer = value.detach().to(torch.int64, copy=True)
                timed(lambda b=buffer: dist.reduce(b, dst=0, op=dist.ReduceOp.MAX), buffer)
            if rank == 0:
                reduced[name] = buffer.to(value.dtype)
        for name in sorted(state):
            if rank == 0:
                state[name].copy_(reduced[name])
            timed(lambda t=state[name]: dist.broadcast(t, src=0), state[name])
    assert_identical_replicas(client, server)


def assert_identical_replicas(client: nn.Module, server: nn.Module) -> None:
    """Compare the state hash of all ranks."""
    digest = state_sha256(combined_state(client, server))
    if dist.is_initialized() and dist.get_world_size() > 1:
        digests: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(digests, digest)
        if any(other != digest for other in digests):
            raise RuntimeError(f"model replicas differ across ranks: {digests}")
