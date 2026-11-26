import math

import pytest
import torch

from gcpsl.methods import Method
from gcpsl.sampling import build_epoch_schedule, client_queues, gpsl_batches
from gcpsl.step import client_blocks

INDICES = [list(range(start, start + 10 + client)) for client, start in enumerate(range(0, 1600, 100))]
TOTAL = sum(len(i) for i in INDICES)
CLUSTERS = [list(range(rank, 16, 4)) for rank in range(4)]


def owners():
    table = torch.full((1700,), -1)
    for client, indices in enumerate(INDICES):
        table[indices] = client
    return table


def schedule(method, *, world_size=4, batch_size=16, epoch=0, seed=40):
    return build_epoch_schedule(
        Method(method),
        client_indices=INDICES,
        world_size=world_size,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        clusters=CLUSTERS,
    )


PERIODIC = ["gcpsl_random", "gcpsl_size_balanced", "gcpsl_label_aware", "gpsl_repartitioned"]
SYNCHRONOUS = ["gpsl_sharded_sync", "gpsl_data_parallel"]


@pytest.mark.parametrize("method", PERIODIC + SYNCHRONOUS)
@pytest.mark.parametrize("epoch", [0, 1, 5])
def test_every_sample_is_scheduled_exactly_once(method, epoch):
    plan = schedule(method, epoch=epoch)
    samples = [i for rank in plan.batches for batch in rank for i in batch]
    assert sorted(samples) == sorted(i for indices in INDICES for i in indices)


@pytest.mark.parametrize("method", PERIODIC)
def test_periodic_workloads_stay_in_their_population(method):
    plan, table = schedule(method), owners()
    assert sorted(k for p in plan.populations for k in p) == list(range(16))
    for population, batches, count in zip(plan.populations, plan.batches, plan.sample_counts, strict=True):
        samples = [i for batch in batches for i in batch]
        assert set(table[samples].tolist()) <= set(population)
        assert len(samples) == count
        # One update per ceil(M / B), and waiting afterwards.
        rounds = math.ceil(count / 16)
        assert all(batches[:rounds]) and not any(batches[rounds:])


def test_static_clusters_persist_and_repartitioning_changes_them():
    assert schedule("gcpsl_random").populations == schedule("gcpsl_random", epoch=3).populations
    assert schedule("gpsl_repartitioned").populations != schedule("gpsl_repartitioned", epoch=1).populations


def test_synchronous_shards_rebuild_the_global_batch():
    single = schedule("gpsl_single_worker", world_size=1).batches[0]
    sharded = schedule("gpsl_sharded_sync").batches
    assert [tuple(i for rank in sharded for i in rank[phase]) for phase in range(len(single))] == list(single)
    large = schedule("gpsl_larger_batch", world_size=1, batch_size=64).batches[0]
    parallel = schedule("gpsl_data_parallel").batches
    assert [tuple(i for rank in parallel for i in rank[phase]) for phase in range(len(large))] == list(large)
    assert max(len(batch) for rank in parallel for batch in rank) <= 16


def test_batch_sizes_differ_by_at_most_one_and_clients_are_contiguous():
    queues, table = client_queues(INDICES, seed=1, epoch=0), owners()
    batches = gpsl_batches(range(16), queues, batch_size=16, sampler_seed=1, epoch=0)
    sizes = [len(batch) for batch in batches]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == TOTAL
    for batch in batches:
        blocks = client_blocks(batch, table)
        assert [block[0] for block in blocks] == sorted({block[0] for block in blocks})


def test_schedules_depend_on_seed_and_epoch_only():
    assert schedule("gcpsl_random") == schedule("gcpsl_random")
    assert schedule("gcpsl_random") != schedule("gcpsl_random", epoch=1)
    assert schedule("gcpsl_random") != schedule("gcpsl_random", seed=72)
