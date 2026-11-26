"""Globally synchronized BatchNorm for the sharded and data-parallel baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class CollectiveCounter:
    """Local tensor bytes and duration of the collectives issued by this process."""

    tensor_bytes: int = 0
    seconds: float = 0.0

    def add(self, tensor: Tensor, seconds: float) -> None:
        self.tensor_bytes += tensor.numel() * tensor.element_size()
        self.seconds += seconds


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _sync(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def timed_all_reduce_(tensor: Tensor, op, counter: CollectiveCounter) -> Tensor:
    _sync(tensor)
    start = time.perf_counter()
    dist.all_reduce(tensor, op=op)
    _sync(tensor)
    counter.add(tensor, time.perf_counter() - start)
    return tensor


class _AllReduceSum(torch.autograd.Function):
    """All-reduce SUM whose adjoint is also an all-reduce SUM."""

    @staticmethod
    def forward(ctx, tensor: Tensor, counter: CollectiveCounter) -> Tensor:
        ctx.counter = counter
        return timed_all_reduce_(tensor.contiguous().clone(), dist.ReduceOp.SUM, counter)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return timed_all_reduce_(grad.contiguous().clone(), dist.ReduceOp.SUM, ctx.counter), None


class GlobalBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm over the union of all shards, including uneven and empty ones.

    Means and variances are weighted by the number of elements on each rank.  The
    variance uses a second reduction of centered squares, which is numerically stable in
    FP32.  Parameter gradients must be synchronized by the caller.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = CollectiveCounter()

    def _advance_batch_counter(self) -> None:
        current = self.num_batches_tracked.detach().to(torch.int64)
        extrema = torch.stack((current, -current))
        timed_all_reduce_(extrema, dist.ReduceOp.MAX, self.counter)
        if not torch.equal(extrema[0], -extrema[1]):
            raise RuntimeError("BatchNorm replicas have processed different numbers of batches")
        self.num_batches_tracked.copy_(extrema[0].add(1))

    def _global_statistics(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dims = (0, 2, 3)
        count = x.new_tensor([x.shape[0] * x.shape[2] * x.shape[3]])
        reduced = _AllReduceSum.apply(torch.cat((x.sum(dim=dims), count)), self.counter)
        total = reduced[-1]
        mean = reduced[: self.num_features] / total
        centered = (x - mean.view(1, -1, 1, 1)).square().sum(dim=dims)
        variance = (_AllReduceSum.apply(centered, self.counter) / total).clamp_min(0)
        return mean, variance, total

    def forward(self, x: Tensor) -> Tensor:
        if _world_size() == 1:
            return super().forward(x)
        self._check_input_dim(x)
        if not self.training:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps
            )
        with torch.no_grad():
            self._advance_batch_counter()
        mean, variance, total = self._global_statistics(x)
        with torch.no_grad():
            unbiased = variance.detach() * (total.detach() / (total.detach() - 1))
            self.running_mean.lerp_(mean.detach(), float(self.momentum))
            self.running_var.lerp_(unbiased, float(self.momentum))
        shape = (1, self.num_features, 1, 1)
        output = (x - mean.view(shape)) * torch.rsqrt(variance + self.eps).view(shape)
        return output * self.weight.view(shape) + self.bias.view(shape)


def convert_batchnorm2d(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d in ``module``, reusing its parameters and buffers."""
    if isinstance(module, nn.BatchNorm2d) and not isinstance(module, GlobalBatchNorm2d):
        converted = GlobalBatchNorm2d(module.num_features, eps=module.eps, momentum=module.momentum)
        converted.weight, converted.bias = module.weight, module.bias
        converted.running_mean, converted.running_var = module.running_mean, module.running_var
        converted.num_batches_tracked = module.num_batches_tracked
        converted.training = module.training
        return converted
    for name, child in list(module.named_children()):
        setattr(module, name, convert_batchnorm2d(child))
    return module


def batchnorm_counter(module: nn.Module) -> CollectiveCounter:
    """Sum of the collective counters of all synchronized BatchNorm layers."""
    total = CollectiveCounter()
    for child in module.modules():
        if isinstance(child, GlobalBatchNorm2d):
            total.tensor_bytes += child.counter.tensor_bytes
            total.seconds += child.counter.seconds
    return total
