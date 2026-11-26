"""One GPSL training step, shared by all methods.

The active clients run their forward pass one client block at a time and send the cut
activations.  The server completes forward and backward, and returns the cut gradients.
The client gradients are summed.  Because the loss is a mean over the batch, this sum
weights each client by its number of examples.

All client replicas of one workload apply the same update and therefore stay identical.
One model instance represents them.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StepResult:
    samples: int
    active_clients: tuple[int, ...]
    loss_sum: float
    correct: int
    cut_bytes: int
    label_bytes: int
    gradient_sync_seconds: float = 0.0
    gradient_sync_bytes: int = 0


def make_optimizers(
    client: nn.Module, server: nn.Module, *, lr: float, momentum: float, weight_decay: float, nesterov: bool
) -> tuple[torch.optim.SGD, torch.optim.SGD]:
    """Separate SGD optimizers for the client and server segments."""
    options = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return torch.optim.SGD(client.parameters(), **options), torch.optim.SGD(server.parameters(), **options)


def client_blocks(batch: Sequence[int], owners: torch.Tensor) -> list[tuple[int, int, int]]:
    """Contiguous ``(client, start, stop)`` blocks of a batch, in batch order."""
    sequence = owners[torch.as_tensor(batch, dtype=torch.long)].tolist()
    blocks: list[tuple[int, int, int]] = []
    start = 0
    for position in range(1, len(sequence) + 1):
        if position < len(sequence) and sequence[position] == sequence[start]:
            continue
        blocks.append((sequence[start], start, position))
        start = position
    if len({client for client, _, _ in blocks}) != len(blocks):
        raise ValueError("a client's samples must be contiguous within a batch")
    return blocks


def _client_backward(client: nn.Module, cuts: Sequence[torch.Tensor], cut_gradient: torch.Tensor) -> None:
    """Back-propagate every client block separately and sum the gradients in block order."""
    parameters = list(client.parameters())
    per_client: list[list[torch.Tensor]] = []
    offset = 0
    for cut in cuts:
        for parameter in parameters:
            parameter.grad = None
        stop = offset + int(cut.shape[0])
        cut.backward(cut_gradient[offset:stop])
        per_client.append([parameter.grad.clone() for parameter in parameters])
        offset = stop
    for index, parameter in enumerate(parameters):
        parameter.grad = sum(gradients[index] for gradients in per_client)


def _forward_backward(
    client: nn.Module,
    server: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch: Sequence[int],
    owners: torch.Tensor,
    label_smoothing: float,
    global_samples: int | None,
    batch_clients: bool,
) -> StepResult:
    if len(batch):
        blocks = client_blocks(batch, owners)
        if batch_clients:
            cuts = [client(inputs)]
        else:
            cuts = [client(inputs[start:stop]) for _, start, stop in blocks]
        cut = torch.cat(cuts, dim=0)
        active = tuple(block[0] for block in blocks)
    else:
        # An empty shard still takes part in the synchronized BatchNorm collectives.
        cuts, cut, active = [], client(inputs), ()

    # The server segment adds in place, so it works on a copy of the received activation.
    received = cut.detach().requires_grad_(True)
    logits = server(received.clone())

    if not len(batch):
        loss, loss_sum, correct = logits.sum() * 0.0, 0.0, 0
    else:
        if global_samples is None:
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            loss_sum = float(loss.detach()) * len(batch)
        else:
            total = F.cross_entropy(logits, labels, reduction="sum", label_smoothing=label_smoothing)
            loss = total / global_samples
            loss_sum = float(total.detach())
        if not math.isfinite(loss_sum):
            raise FloatingPointError("non-finite training loss")
        correct = int((logits.argmax(dim=1) == labels).sum())
    loss.backward()
    if cuts:
        _client_backward(client, cuts, received.grad)

    return StepResult(
        samples=len(batch),
        active_clients=active,
        loss_sum=loss_sum,
        correct=correct,
        cut_bytes=cut.numel() * cut.element_size(),
        label_bytes=labels.numel() * labels.element_size(),
    )


def _all_reduce_gradients(modules: Sequence[nn.Module]) -> tuple[float, int]:
    """Sum the gradients of all ranks in one flat collective."""
    gradients = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            gradients.append(parameter.grad)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return 0.0, 0
    flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
    if flat.is_cuda:
        torch.cuda.synchronize(flat.device)
    start = time.perf_counter()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if flat.is_cuda:
        torch.cuda.synchronize(flat.device)
    seconds = time.perf_counter() - start
    offset = 0
    with torch.no_grad():
        for gradient in gradients:
            count = gradient.numel()
            gradient.copy_(flat[offset : offset + count].view_as(gradient))
            offset += count
    return seconds, flat.numel() * flat.element_size()


def train_step(
    client: nn.Module,
    server: nn.Module,
    client_optimizer: torch.optim.Optimizer,
    server_optimizer: torch.optim.Optimizer,
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch: Sequence[int],
    owners: torch.Tensor,
    label_smoothing: float,
    global_samples: int | None = None,
    batch_clients: bool = False,
    region: Callable[[str], AbstractContextManager] = lambda name: nullcontext(),
) -> StepResult:
    """Run one local step, or one globally synchronized step.

    ``global_samples`` is the size of the global batch for the synchronous methods and
    ``None`` for local steps.  ``batch_clients`` runs the client segment once on the whole
    batch instead of once per client.  The result is the same up to rounding, because the
    segment has no operation across samples.  ``region`` is an optional timing hook.
    """
    if global_samples is None and not len(batch):
        raise ValueError("a waiting worker must not take an optimizer step")
    client_optimizer.zero_grad(set_to_none=True)
    server_optimizer.zero_grad(set_to_none=True)
    with region("training_compute"):
        result = _forward_backward(
            client, server, inputs, labels, batch, owners, label_smoothing, global_samples, batch_clients
        )
    if global_samples is not None:
        with region("gradient_sync"):
            seconds, size = _all_reduce_gradients([client, server])
        result = StepResult(
            **{**result.__dict__, "gradient_sync_seconds": seconds, "gradient_sync_bytes": size}
        )
    with region("optimizer"):
        client_optimizer.step()
        server_optimizer.step()
    return result
