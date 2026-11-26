"""The methods compared in the paper and how each one uses the available workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Method(str, Enum):
    GCPSL_RANDOM = "gcpsl_random"
    GCPSL_SIZE_BALANCED = "gcpsl_size_balanced"
    GCPSL_LABEL_AWARE = "gcpsl_label_aware"
    GPSL_SINGLE_WORKER = "gpsl_single_worker"
    GPSL_LARGER_BATCH = "gpsl_larger_batch"
    GPSL_SHARDED_SYNC = "gpsl_sharded_sync"
    GPSL_DATA_PARALLEL = "gpsl_data_parallel"
    GPSL_REPARTITIONED = "gpsl_repartitioned"


@dataclass(frozen=True)
class MethodSpec:
    """Execution semantics of one method.

    Attributes:
        clusters: name of the client-assignment rule, for GCPSL methods only.
        synchronous: gradients and server BatchNorm are synchronized at every step.
        periodic: one GPSL workload per rank, fused at epoch boundaries.
        single_worker: the method is defined for one rank only.
    """

    clusters: str | None = None
    synchronous: bool = False
    periodic: bool = False
    single_worker: bool = False


METHODS: dict[Method, MethodSpec] = {
    Method.GCPSL_RANDOM: MethodSpec(clusters="random", periodic=True),
    Method.GCPSL_SIZE_BALANCED: MethodSpec(clusters="size_balanced", periodic=True),
    Method.GCPSL_LABEL_AWARE: MethodSpec(clusters="label_aware", periodic=True),
    Method.GPSL_SINGLE_WORKER: MethodSpec(single_worker=True),
    Method.GPSL_LARGER_BATCH: MethodSpec(single_worker=True),
    Method.GPSL_SHARDED_SYNC: MethodSpec(synchronous=True),
    Method.GPSL_DATA_PARALLEL: MethodSpec(synchronous=True),
    Method.GPSL_REPARTITIONED: MethodSpec(periodic=True),
}


def global_batch_size(method: Method, *, batch_size: int, world_size: int) -> int:
    """Number of examples drawn from the global population per step.

    Only defined for methods that draw one batch from all clients.
    """
    if method is Method.GPSL_DATA_PARALLEL:
        return world_size * batch_size
    return batch_size
