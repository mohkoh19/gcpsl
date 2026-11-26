"""Deterministic seed derivation and process-level determinism settings.

Every random stream (partition, clustering, sampling, augmentation, ...) gets its
own seed, derived from the experiment seed with SHA-256.  The schema string and the
component names are part of the derivation: changing them changes all results.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from enum import Enum

import numpy as np
import torch

SEED_SCHEMA = "gcpsl-seed-v1"


class SeedComponent(str, Enum):
    """Independent random streams of one run."""

    CLIENT_PARTITION = "client_partition"
    CLUSTER_ASSIGNMENT = "cluster_assignment"
    MODEL_INITIALIZATION = "model_initialization"
    GLOBAL_SAMPLER = "global_sampler"
    CLUSTER_SAMPLER = "cluster_sampler"
    LOCAL_CLIENT_SAMPLER = "local_client_sampler"
    RANK_RUNTIME = "rank_runtime"
    DATALOADER_GENERATOR = "dataloader_generator"
    DATALOADER_WORKER = "dataloader_worker"
    AUGMENTATION = "augmentation"
    RP_REPARTITION = "rp_repartition"


def derive_seed(
    base_seed: int,
    component: SeedComponent,
    *,
    rank: int | None = None,
    worker: int | None = None,
    client: int | None = None,
    epoch: int | None = None,
    interval: int | None = None,
) -> int:
    """Derive a 63-bit seed for one component and an optional set of coordinates."""
    payload = json.dumps(
        [SEED_SCHEMA, int(base_seed), component.value, rank, worker, client, epoch, interval],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) & ((1 << 63) - 1)


def to_uint32_seed(seed: int) -> int:
    """Reduce a seed to the 32-bit range used by the NumPy partition and clustering RNGs."""
    return int(seed) % (2**32 - 1)


def augmentation_seed(base_seed: int, *, epoch: int, source_index: int) -> int:
    """Seed of one augmentation, keyed by epoch and original dataset index.

    The key does not depend on the rank, worker or owning client, so a sample is
    augmented identically wherever it is processed.
    """
    return derive_seed(base_seed, SeedComponent.AUGMENTATION, client=source_index, epoch=epoch)


def seed_process(seed: int, *, strict: bool = True) -> None:
    """Seed all process RNGs and enable deterministic FP32 execution.

    With ``strict=True`` the environment must already define ``PYTHONHASHSEED`` and
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, because setting them here would be too late.
    """
    if strict:
        if not os.environ.get("PYTHONHASHSEED", "").isdigit():
            raise RuntimeError("set PYTHONHASHSEED to a non-negative integer before starting Python")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    random.seed(seed, version=2)
    np.random.seed(to_uint32_seed(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash a named tensor state (names, dtypes, shapes and raw bytes)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
    return digest.hexdigest()
