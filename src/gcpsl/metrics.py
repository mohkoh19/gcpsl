"""Evaluation, participation and target metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float


def evaluate(
    client: nn.Module,
    server: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    label_smoothing: float,
    device: torch.device,
) -> Evaluation:
    """Mean loss and accuracy (correct / total) over all examples of a split."""
    modes = client.training, server.training
    client.eval()
    server.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = server(client(inputs))
            loss_sum += float(
                F.cross_entropy(logits, labels, reduction="sum", label_smoothing=label_smoothing)
            )
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
    client.train(modes[0])
    server.train(modes[1])
    return Evaluation(loss=loss_sum / count, accuracy=correct / count)


def gini(values: Iterable[float]) -> float:
    """Gini coefficient, including zeros."""
    ordered = sorted(float(v) for v in values)
    total = sum(ordered)
    n = len(ordered)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return 2.0 * weighted / (n * total) - (n + 1) / n


class ParticipationTracker:
    """Direct data participation of every client over system phases.

    A system phase groups local round ``t`` of all workloads.  ``pi`` is the fraction of
    client-phase slots with direct data participation, ``mu`` the fraction in which the
    client's workload is active, and ``pi = mu * pi_within``.
    """

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.phases = 0
        self.direct = [0] * num_clients
        self.updated = [0] * num_clients
        self._last = [-1] * num_clients
        self._gaps: Counter[int] = Counter()

    def add_phase(self, active: Iterable[int], updated: Iterable[int]) -> None:
        """Record the direct participants and all clients that apply a model update."""
        for client in active:
            self.direct[client] += 1
            if self._last[client] >= 0:
                self._gaps[self.phases - self._last[client] - 1] += 1
            self._last[client] = self.phases
        for client in updated:
            self.updated[client] += 1
        self.phases += 1

    def add_epoch(self, populations, batches, owners: torch.Tensor, *, shared: bool) -> None:
        """Record every phase of an epoch schedule.

        ``shared`` is true for methods whose workers all serve the full client population.
        """
        for phase in range(len(batches[0])):
            active: set[int] = set()
            updated: set[int] = set()
            for population, rank_batches in zip(populations, batches, strict=True):
                batch = rank_batches[phase]
                if batch:
                    active.update(owners[torch.as_tensor(batch)].tolist())
                if batch or shared:
                    updated.update(population)
            self.add_phase(active, updated)

    def _gap_percentile(self, q: float) -> float:
        total = sum(self._gaps.values())
        position = (total - 1) * q
        lower, upper = int(position), min(int(position) + 1, total - 1)
        ordered = sorted(self._gaps.items())

        def at(index: int) -> int:
            seen = 0
            for value, count in ordered:
                seen += count
                if index < seen:
                    return value
            raise IndexError(index)

        return at(lower) + (position - lower) * (at(upper) - at(lower))

    def summary(self) -> dict[str, float]:
        slots = self.num_clients * self.phases
        pi = sum(self.direct) / slots
        mu = sum(self.updated) / slots
        gaps = sum(self._gaps.values())
        return {
            "direct_participation": pi,
            "model_update_participation": mu,
            "within_round_participation": pi / mu,
            "coverage": sum(count > 0 for count in self.direct) / self.num_clients,
            "gini": gini(self.direct),
            "mean_gap": sum(g * c for g, c in self._gaps.items()) / gaps if gaps else float("nan"),
            "p95_gap": self._gap_percentile(0.95) if gaps else float("nan"),
        }


@dataclass
class Target:
    """First epoch whose validation accuracy reaches ``accuracy``."""

    accuracy: float
    epoch: int | None = None
    seconds: float | None = None

    def update(self, *, accuracy: float, epoch: int, seconds: float) -> None:
        if self.epoch is None and accuracy >= self.accuracy:
            self.epoch, self.seconds = epoch, seconds


def make_targets(accuracies: Sequence[float]) -> list[Target]:
    return [Target(float(accuracy)) for accuracy in accuracies]
