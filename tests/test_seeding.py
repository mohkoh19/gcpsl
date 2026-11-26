"""Golden values: changing any of these changes every published result."""

from gcpsl.seeding import SeedComponent, augmentation_seed, derive_seed, to_uint32_seed


def test_golden_seeds():
    assert derive_seed(40, SeedComponent.CLIENT_PARTITION) == GOLDEN["partition"]
    assert derive_seed(40, SeedComponent.CLUSTER_SAMPLER, rank=2) == GOLDEN["cluster_sampler"]
    assert derive_seed(72, SeedComponent.RP_REPARTITION, epoch=3, interval=0) == GOLDEN["repartition"]
    assert augmentation_seed(313, epoch=7, source_index=1234) == GOLDEN["augmentation"]


def test_components_and_coordinates_are_independent():
    seeds = {derive_seed(40, component) for component in SeedComponent}
    assert len(seeds) == len(SeedComponent)
    assert derive_seed(40, SeedComponent.AUGMENTATION, client=1, epoch=2) != derive_seed(
        40, SeedComponent.AUGMENTATION, client=2, epoch=1
    )


def test_uint32_range():
    assert to_uint32_seed(2**63 - 1) < 2**32


GOLDEN = {
    "partition": 1696093086873553206,
    "cluster_sampler": 4147038445678496390,
    "repartition": 3314508689996232328,
    "augmentation": 2449250805548202643,
}
