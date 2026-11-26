"""Client partitioning and the train/validation split."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.stats import dirichlet

VALIDATION_SPLIT_SEED = 20260715


def extended_dirichlet_partition(
    labels: np.ndarray,
    *,
    num_clients: int,
    classes_per_client: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Extended-Dirichlet partition (https://arxiv.org/abs/2302.01633).

    Each client is first assigned ``classes_per_client`` classes, with a preference for
    classes that few clients hold.  The samples of each class are then distributed over
    the clients holding it with Dirichlet concentration ``alpha``.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    num_classes = np.unique(labels).size

    holds_class = np.zeros((num_clients, num_classes))
    for client in range(num_clients):
        probabilities = np.exp(-holds_class.sum(axis=0) * num_classes)
        probabilities /= probabilities.sum()
        chosen = rng.choice(num_classes, size=classes_per_client, p=probabilities, replace=False)
        holds_class[client, chosen] += 1

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        concentration = holds_class[:, class_id] * alpha + 1e-10
        shares = dirichlet.rvs(concentration, size=1, random_state=rng).flatten()
        samples = np.where(labels == class_id)[0]
        owners = rng.choice(range(num_clients), size=samples.size, p=shares)
        for sample, owner in zip(samples, owners, strict=True):
            client_indices[owner].append(int(sample))
    return client_indices


def stratified_split(
    labels: Sequence[int] | np.ndarray, *, validation_size: int, seed: int = VALIDATION_SPLIT_SEED
) -> tuple[list[int], list[int]]:
    """Split the original training set into train and validation indices.

    Class quotas use the largest-remainder rule.  The split seed is fixed across
    experiment seeds, so all runs share the same validation set.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if not 0 < validation_size < len(labels):
        raise ValueError("validation_size must lie strictly between 0 and the dataset size")
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts.astype(np.float64) * (float(validation_size) / float(len(labels)))
    quotas = np.floor(exact).astype(np.int64)
    missing = int(validation_size - int(quotas.sum()))
    if missing:
        fractional = exact - quotas
        order = sorted(range(len(classes)), key=lambda i: (-float(fractional[i]), int(classes[i])))
        for index in order[:missing]:
            quotas[index] += 1

    rng = np.random.default_rng(int(seed))
    validation: list[int] = []
    for class_id, quota in zip(classes.tolist(), quotas.tolist(), strict=True):
        members = np.flatnonzero(labels == int(class_id))
        validation.extend(int(i) for i in rng.permutation(members)[: int(quota)])
    validation = sorted(validation)
    held_out = set(validation)
    train = [index for index in range(len(labels)) if index not in held_out]
    return train, validation


def owners_from_partition(client_indices: Sequence[Sequence[int]], num_samples: int) -> torch.Tensor:
    """Return the owning client of every training sample."""
    owners = torch.full((num_samples,), -1, dtype=torch.long)
    for client, indices in enumerate(client_indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        if not len(index):
            raise ValueError(f"client {client} owns no samples")
        if bool(torch.any(owners[index] != -1)):
            raise ValueError("a sample is assigned to more than one client")
        owners[index] = client
    if bool(torch.any(owners < 0)):
        raise ValueError("the partition does not cover every training sample")
    return owners
