import numpy as np
import pytest

from gcpsl.clustering import (
    build_clusters,
    label_aware_clusters,
    label_aware_objective,
    random_clusters,
    size_balanced_clusters,
)
from gcpsl.data.partition import extended_dirichlet_partition, stratified_split


def example_metadata(num_clients=24, num_classes=6, seed=0):
    rng = np.random.default_rng(seed)
    histograms = np.zeros((num_clients, num_classes), dtype=int)
    for client in range(num_clients):
        for class_id in rng.choice(num_classes, 2, replace=False):
            histograms[client, class_id] = rng.integers(1, 40)
    return histograms.sum(axis=1).tolist(), histograms.tolist()


def test_partition_covers_every_sample_once_with_two_classes():
    labels = np.repeat(np.arange(10), 200)
    parts = extended_dirichlet_partition(labels, num_clients=32, classes_per_client=2, alpha=3, seed=1)
    assert sorted(i for part in parts for i in part) == list(range(len(labels)))
    assert all(len(set(labels[part])) <= 2 for part in parts)


def test_split_is_stratified_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    train, validation = stratified_split(labels, validation_size=100)
    assert set(train).isdisjoint(validation) and len(train) + len(validation) == 1000
    assert np.bincount(labels[validation]).tolist() == [10] * 10


@pytest.mark.parametrize("rule", ["random", "size_balanced", "label_aware"])
@pytest.mark.parametrize("num_clusters", [1, 2, 4])
def test_rules_give_disjoint_covering_nonempty_clusters(rule, num_clusters):
    counts, histograms = example_metadata()
    clusters = build_clusters(
        rule, sample_counts=counts, label_histograms=histograms, num_clusters=num_clusters, seed=5
    )
    assert len(clusters) == num_clusters and all(clusters)
    assert sorted(k for c in clusters for k in c) == list(range(len(counts)))


def test_random_clusters_have_nearly_equal_cardinality():
    sizes = [len(c) for c in random_clusters(10, 4, seed=0)]
    assert max(sizes) - min(sizes) <= 1


def test_size_cap_is_respected_when_feasible():
    counts, histograms = example_metadata()
    limit = 1.25 * sum(counts) / 4
    for clusters in (
        size_balanced_clusters(counts, 4),
        label_aware_clusters(counts, histograms, 4, seed=2),
    ):
        assert all(sum(counts[k] for k in c) <= limit for c in clusters)


def test_size_balanced_falls_back_to_lightest_cluster():
    assert size_balanced_clusters([100, 1, 1], 3) == [[0], [1], [2]]


def test_label_aware_search_lowers_the_objective_to_a_local_optimum():
    counts, histograms = example_metadata()
    counts_array = np.asarray(counts)
    distributions = np.asarray(histograms) / counts_array[:, None]
    order = list(range(len(counts)))
    np.random.default_rng(7).shuffle(order)
    start = [order[n::3] for n in range(3)]
    result = label_aware_clusters(counts, histograms, 3, seed=7)
    before = label_aware_objective(start, counts_array, distributions)
    after = label_aware_objective(result, counts_array, distributions)
    assert after < before
    assert label_aware_clusters(counts, histograms, 3, seed=7, max_moves=10_000) == result
