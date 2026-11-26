"""Stable client clustering (Algorithm 1 of the paper)."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon

Clusters = list[list[int]]


def random_clusters(num_clients: int, num_clusters: int, *, seed: int) -> Clusters:
    """Permute the clients and assign them round-robin.  Uses no client metadata."""
    order = np.arange(num_clients, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    clusters: Clusters = [[] for _ in range(num_clusters)]
    for position, client in enumerate(order.tolist()):
        clusters[position % num_clusters].append(int(client))
    return clusters


def size_balanced_clusters(
    sample_counts: Sequence[int], num_clusters: int, *, max_size_fraction: float = 1.25
) -> Clusters:
    """Place clients by decreasing sample count in the lightest allowed cluster.

    A cluster is allowed if it stays within ``max_size_fraction`` of the mean cluster
    size.  If no cluster is allowed, the lightest one is used.
    """
    limit = max_size_fraction * (sum(sample_counts) / num_clusters)
    order = sorted(range(len(sample_counts)), key=lambda k: (-float(sample_counts[k]), k))
    clusters: Clusters = [[] for _ in range(num_clusters)]
    sizes = [0.0] * num_clusters
    for client in order:
        count = float(sample_counts[client])
        allowed = [n for n, size in enumerate(sizes) if size + count <= limit]
        target = min(allowed or range(num_clusters), key=lambda n: (sizes[n], len(clusters[n]), n))
        clusters[target].append(client)
        sizes[target] += count
    return clusters


def label_aware_objective(
    clusters: Clusters,
    sample_counts: np.ndarray,
    distributions: np.ndarray,
    *,
    size_weight: float = 1.0,
    label_weight: float = 12.0,
) -> float:
    """Objective J: normalized size deviation plus weighted Jensen-Shannon distance."""
    total = int(sample_counts.sum())
    target_size = np.float64(total) / len(clusters)
    global_distribution = np.dot(sample_counts, distributions) / float(total)

    sizes = [sum(sample_counts[k] for k in cluster) for cluster in clusters]
    cluster_distributions = []
    for cluster, size in zip(clusters, sizes, strict=True):
        weighted = [0.0] * distributions.shape[1]
        for k in cluster:
            weighted = [w + sample_counts[k] * p for w, p in zip(weighted, distributions[k], strict=True)]
        cluster_distributions.append([w / size for w in weighted])

    size_cost = np.sum(abs(np.asarray(sizes) - target_size)) / sum(sizes)
    distances = np.array(
        [jensenshannon(p, global_distribution) * s for p, s in zip(cluster_distributions, sizes, strict=True)]
    )
    return float(size_weight * size_cost + label_weight * np.sum(distances / total))


def label_aware_clusters(
    sample_counts: Sequence[int],
    label_histograms: Sequence[Sequence[int]],
    num_clusters: int,
    *,
    seed: int,
    max_size_fraction: float = 1.25,
    size_weight: float = 1.0,
    label_weight: float = 12.0,
    max_moves: int = 3000,
) -> Clusters:
    """Start from a random round-robin assignment and apply the best single-client moves."""
    counts = np.asarray(sample_counts, dtype=np.int64)
    distributions = np.asarray(label_histograms, dtype=np.int64) / counts[:, None]
    limit = max_size_fraction * (float(counts.sum()) / num_clusters)

    def objective(clusters: Clusters) -> float:
        return label_aware_objective(
            clusters, counts, distributions, size_weight=size_weight, label_weight=label_weight
        )

    # The shuffled object must stay a Python list: NumPy shuffles lists and arrays differently.
    order = list(range(len(counts)))
    np.random.default_rng(seed).shuffle(order)
    clusters: Clusters = [order[n::num_clusters] for n in range(num_clusters)]
    cost = objective(clusters)

    for _ in range(max_moves):
        sizes = [sum(counts[k] for k in cluster) for cluster in clusters]
        best_delta, best_clusters = 0.0, None
        for source, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
            for client in cluster:
                for target in range(num_clusters):
                    if target == source or sizes[target] + counts[client] > limit:
                        continue
                    candidate = copy.deepcopy(clusters)
                    candidate[source].remove(client)
                    candidate[target].append(client)
                    delta = objective(candidate) - cost
                    if delta < best_delta:
                        best_delta, best_clusters = delta, candidate
        if best_clusters is None:
            break
        clusters = best_clusters
        cost += best_delta
    return [[int(k) for k in cluster] for cluster in clusters]


def build_clusters(
    rule: str,
    *,
    sample_counts: Sequence[int],
    label_histograms: Sequence[Sequence[int]],
    num_clusters: int,
    seed: int,
    **options,
) -> Clusters:
    """Apply one assignment rule.  Each rule receives only the metadata it is allowed to use."""
    if num_clusters == 1:
        return [list(range(len(sample_counts)))]
    if rule == "random":
        return random_clusters(len(sample_counts), num_clusters, seed=seed)
    if rule == "size_balanced":
        return size_balanced_clusters(
            sample_counts, num_clusters, max_size_fraction=options.get("max_size_fraction", 1.25)
        )
    if rule == "label_aware":
        return label_aware_clusters(sample_counts, label_histograms, num_clusters, seed=seed, **options)
    raise ValueError(f"unknown assignment rule {rule!r}")
