"""Wall-clock accounting by region on one rank."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from gcpsl.distributed.context import DistributedContext

# Further regions: initialization, validation, checkpoint, test.
TRAINING_REGIONS = (
    "schedule",
    "data_wait",
    "host_to_device",
    "training_compute",
    "gradient_sync",
    "optimizer",
    "barrier_wait",
    "fusion",
    "replica_check",
)


class RegionTimer:
    """Accumulates the duration of non-overlapping regions, with CUDA synchronization."""

    def __init__(self, context: DistributedContext):
        self._context = context
        self.seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        self._context.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._context.synchronize()
            self.seconds[name] += time.perf_counter() - start

    @property
    def training_seconds(self) -> float:
        return sum(self.seconds[name] for name in TRAINING_REGIONS)
