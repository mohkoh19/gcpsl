"""The GPSL global sampler and the per-epoch batch schedule of every worker.

At the start of an epoch the sampler lists one token per training example, labeled with
its owner, shuffles the list and splits it into ``ceil(M / B)`` batches whose sizes
differ by at most one.  Token counts say how many examples each client contributes to
a batch; the examples themselves come from a per-client shuffled queue.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from gcpsl.methods import METHODS, Method, global_batch_size
from gcpsl.seeding import SeedComponent, derive_seed

Batch = tuple[int, ...]


def client_queues(
    client_indices: Sequence[Sequence[int]], *, seed: int, epoch: int
) -> list[tuple[int, ...]]:
    """Shuffled sample order of every client in one (zero-based) epoch."""
    queues = []
    for client, indices in enumerate(client_indices):
        generator = torch.Generator().manual_seed(
            derive_seed(seed, SeedComponent.LOCAL_CLIENT_SAMPLER, client=client)
        )
        # The client's sampler persists across epochs: epoch e uses its (e + 1)-th permutation.
        for _ in range(epoch + 1):
            permutation = torch.randperm(len(indices), generator=generator)
        queues.append(tuple(int(indices[int(position)]) for position in permutation))
    return queues


def gpsl_batches(
    population: Sequence[int],
    queues: Sequence[Sequence[int]],
    *,
    batch_size: int,
    sampler_seed: int,
    epoch: int,
) -> list[Batch]:
    """Batches of one GPSL workload over ``population`` in one epoch.

    Within a batch, samples are ordered by client in population order.
    """
    tokens = np.concatenate(
        [np.full(len(queues[client]), client, dtype=np.int64) for client in population]
    )
    rng = np.random.default_rng(sampler_seed)
    # One shuffle at construction, then one per epoch.
    for _ in range(epoch + 2):
        rng.shuffle(tokens)
    cursors = dict.fromkeys(population, 0)
    batches: list[Batch] = []
    for part in np.array_split(tokens, math.ceil(len(tokens) / batch_size)):
        counts = np.bincount(part, minlength=len(queues))
        batch: list[int] = []
        for client in population:
            start, stop = cursors[client], cursors[client] + int(counts[client])
            batch.extend(queues[client][start:stop])
            cursors[client] = stop
        batches.append(tuple(batch))
    return batches


@dataclass(frozen=True)
class EpochSchedule:
    """Batches and client population of every worker in one epoch.

    ``batches[rank][phase]`` is empty when the rank waits at the barrier.
    ``sample_counts[rank]`` is the number of training examples of the rank's population
    and determines its fusion weight.
    """

    populations: tuple[tuple[int, ...], ...]
    sample_counts: tuple[int, ...]
    batches: tuple[tuple[Batch, ...], ...]

    @property
    def num_phases(self) -> int:
        return len(self.batches[0])


def _pad(batches: Sequence[Sequence[Batch]]) -> tuple[tuple[Batch, ...], ...]:
    horizon = max(len(rank) for rank in batches)
    return tuple(tuple(rank) + ((),) * (horizon - len(rank)) for rank in batches)


def _repartition(num_clients: int, world_size: int, *, seed: int, epoch: int) -> list[list[int]]:
    clients = np.arange(num_clients, dtype=np.int64)
    np.random.default_rng(
        derive_seed(seed, SeedComponent.RP_REPARTITION, epoch=epoch, interval=0)
    ).shuffle(clients)
    return [[int(k) for k in clients[rank::world_size]] for rank in range(world_size)]


def build_epoch_schedule(
    method: Method,
    *,
    client_indices: Sequence[Sequence[int]],
    world_size: int,
    batch_size: int,
    seed: int,
    epoch: int,
    clusters: Sequence[Sequence[int]] | None = None,
) -> EpochSchedule:
    """Schedule of one zero-based epoch.  GCPSL methods need their fixed ``clusters``."""
    spec = METHODS[method]
    queues = client_queues(client_indices, seed=seed, epoch=epoch)
    everyone = tuple(range(len(client_indices)))

    if spec.periodic:
        if spec.clusters is not None:
            if clusters is None or len(clusters) != world_size:
                raise ValueError("GCPSL needs one cluster per worker")
            populations = [list(cluster) for cluster in clusters]
        else:
            populations = _repartition(len(client_indices), world_size, seed=seed, epoch=epoch)
        batches = []
        for rank, population in enumerate(populations):
            if spec.clusters is not None and world_size > 1:
                sampler_seed = derive_seed(seed, SeedComponent.CLUSTER_SAMPLER, rank=rank)
                sampler_epoch = epoch
            elif spec.clusters is not None:
                sampler_seed = derive_seed(seed, SeedComponent.GLOBAL_SAMPLER)
                sampler_epoch = epoch
            else:
                # Every repartition starts a new sampler, whose seed already contains the epoch.
                sampler_seed = derive_seed(
                    seed, SeedComponent.CLUSTER_SAMPLER, rank=rank, epoch=epoch, interval=0
                )
                sampler_epoch = 0
            batches.append(
                gpsl_batches(
                    population, queues, batch_size=batch_size, sampler_seed=sampler_seed, epoch=sampler_epoch
                )
            )
        rank_batches = _pad(batches)
    else:
        populations = [list(everyone) for _ in range(world_size)]
        capacity = global_batch_size(method, batch_size=batch_size, world_size=world_size)
        batches = gpsl_batches(
            everyone,
            queues,
            batch_size=capacity,
            sampler_seed=derive_seed(seed, SeedComponent.GLOBAL_SAMPLER),
            epoch=epoch,
        )
        if spec.single_worker:
            rank_batches = (tuple(batches),)
        else:
            rank_batches = tuple(
                tuple(_shard(method, batch, rank, world_size, batch_size) for batch in batches)
                for rank in range(world_size)
            )

    sample_counts = tuple(sum(len(client_indices[k]) for k in population) for population in populations)
    return EpochSchedule(
        populations=tuple(tuple(population) for population in populations),
        sample_counts=sample_counts,
        batches=rank_batches,
    )


def _shard(method: Method, batch: Batch, rank: int, world_size: int, batch_size: int) -> Batch:
    """Contiguous part of a global batch processed by ``rank``."""
    if method is Method.GPSL_SHARDED_SYNC:
        return tuple(int(i) for i in np.array_split(np.asarray(batch, dtype=np.int64), world_size)[rank])
    return batch[rank * batch_size : (rank + 1) * batch_size]
