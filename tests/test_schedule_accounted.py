import pytest
import torch

from gcpsl.schedule_accounted.accounting import AnalyticalPayload, batch_deviation, schedule_rounds
from gcpsl.schedule_accounted.trainer import ScheduleAccountedRun, ScheduleConfig, execution_order


def make_run(data, **options):
    defaults = dict(
        epochs=1, batch_size=16, num_clients=16, num_workers=0, device="cpu", targets=(0.0, 0.99)
    )
    return ScheduleAccountedRun(ScheduleConfig(**{**defaults, **options}), data)


def test_execution_order():
    assert list(execution_order([2, 1], sequential=True, seed=0)) == [0, 0, 1]
    order = list(execution_order([30, 20, 10], sequential=False, seed=40))
    assert [order.count(c) for c in range(3)] == [30, 20, 10]
    assert order == list(execution_order([30, 20, 10], sequential=False, seed=40))


def test_schedule_rounds():
    assert schedule_rounds([5, 3, 4], sequential=False) == 5
    assert schedule_rounds([5, 3, 4], sequential=True) == 12


def test_batch_deviation():
    uniform = torch.full((4,), 0.25, dtype=torch.float64)
    assert batch_deviation(torch.arange(4), uniform) == 0
    assert batch_deviation(torch.zeros(4, dtype=torch.long), uniform) == pytest.approx(1.5)


def test_payload_per_round_matches_the_published_accounting():
    """GPSL with B=64 and K=256 handles about 681 MB per round."""
    payload = AnalyticalPayload(client_bytes=149_824 * 4, server_bytes=11_024_138 * 4, num_clients=256)
    payload.add_round(cut_bytes=64 * 64 * 32 * 32 * 4, cluster_size=256)
    assert payload.total == pytest.approx(680.8e6, rel=1e-3)


@pytest.mark.parametrize(
    "options",
    [
        dict(clusters=None, num_clusters=1),
        dict(clusters="random", num_clusters=2),
        dict(clusters="size_balanced", num_clusters=4),
        dict(clusters="label_aware", num_clusters=2, fusion=5),
        dict(clusters="label_aware", num_clusters=2, fusion="none"),
        dict(clusters="label_aware", num_clusters=2, sequential=True),
    ],
)
def test_every_mode_trains(small_data, options):
    summary = make_run(small_data, **options).run()
    record = summary["history"][0]
    assert record["rounds"] == schedule_rounds(
        record["cluster_steps"], sequential=options.get("sequential", False)
    )
    assert 0 <= summary["inactivity"] <= 1
    assert summary["rounds_to_target"] == {"0.0": record["rounds"], "0.99": None}


def test_epoch_fusion_does_not_depend_on_the_execution_order(small_data):
    runs = [
        make_run(small_data, clusters="random", num_clusters=2, sequential=sequential)
        for sequential in (False, True)
    ]
    for run in runs:
        run.train_epoch(0)
    a, b = (run.fused_state() for run in runs)
    assert all(torch.equal(a[name], b[name]) for name in a)


def test_no_fusion_keeps_replicas_apart(small_data):
    run = make_run(small_data, clusters="random", num_clusters=2, fusion="none")
    run.train_epoch(0)
    first, second = (next(w.server.parameters()) for w in run.workloads)
    assert not torch.equal(first, second)
