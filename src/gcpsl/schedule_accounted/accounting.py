"""Schedule rounds, inactivity, batch composition and analytical payload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def schedule_rounds(cluster_steps: Sequence[int], *, sequential: bool) -> int:
    """Rounds of one epoch: the longest cluster, or all steps for sequential execution."""
    return sum(cluster_steps) if sequential else max(cluster_steps)


def cluster_local_inactivity(
    clusters: Sequence[Sequence[int]],
    batches: Sequence[Sequence[Sequence[int]]],
    owners: torch.Tensor,
) -> float:
    """Mean fraction of its cluster's rounds in which a client supplies no example.

    This is the inactivity reported for the schedule-accounted studies.  Rounds in which a
    cluster waits at the fusion barrier are not counted.
    """
    total, clients = 0.0, 0
    for cluster, cluster_batches in zip(clusters, batches, strict=True):
        active = dict.fromkeys(cluster, 0)
        for batch in cluster_batches:
            for client in set(owners[torch.as_tensor(batch)].tolist()):
                active[client] += 1
        total += sum(1 - count / len(cluster_batches) for count in active.values())
        clients += len(cluster)
    return total / clients


def batch_deviation(labels: torch.Tensor, global_distribution: torch.Tensor) -> float:
    """L1 distance between the class distributions of a batch and of the training set."""
    counts = torch.bincount(labels, minlength=len(global_distribution)).double()
    return float((counts / counts.sum() - global_distribution).abs().sum())


@dataclass
class AnalyticalPayload:
    """Payload handled by all endpoints, in bytes.

    Every transfer is counted at its sender and at its receiver.  In each round a cluster
    moves activations up, cut gradients down, and one client model per client in both
    directions.  This accounting differs from the logical payload of the real-execution
    study; the two must not be compared.
    """

    client_bytes: int
    server_bytes: int
    num_clients: int
    total: int = 0

    def add_round(self, *, cut_bytes: int, cluster_size: int) -> None:
        self.total += 2 * (2 * cut_bytes + 2 * self.client_bytes * cluster_size)

    def add_fusion(self, num_clusters: int) -> None:
        self.total += 2 * self.server_bytes * (num_clusters - 1)
        self.total += 2 * self.client_bytes * (self.num_clients + num_clusters)
