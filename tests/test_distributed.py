import pytest
import torch
import torch.distributed as dist
from conftest import make_data, run_distributed
from torch import nn

from gcpsl.distributed.context import init_distributed
from gcpsl.distributed.engine import DistributedRun, RunConfig
from gcpsl.fusion import fuse_across_ranks, fuse_states
from gcpsl.models.split import build_split_resnet18, combined_state
from gcpsl.normalization import CollectiveCounter, GlobalBatchNorm2d
from gcpsl.step import make_optimizers, train_step

pytestmark = pytest.mark.distributed
SHARDS = {2: [5, 3], 4: [4, 0, 3, 1]}
SGD = dict(lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)


def _batch():
    generator = torch.Generator().manual_seed(1)
    return torch.randn(8, 3, 32, 32, generator=generator), torch.arange(8) % 10


def _batchnorm(world_size):
    torch.manual_seed(0)
    x = torch.randn(8, 4, 3, 3, dtype=torch.float32) * 3 + 100
    layer = GlobalBatchNorm2d(4)
    local = x.split(SHARDS[world_size])[dist.get_rank()].clone().requires_grad_(True)
    output = layer(local)
    (output.sin().sum()).backward()
    return output.detach(), local.grad, layer.running_mean, layer.running_var, layer.weight.grad


@pytest.mark.parametrize("world_size", [2, 4])
def test_synchronized_batchnorm_matches_one_process(world_size):
    results = run_distributed(_batchnorm, world_size, world_size)
    torch.manual_seed(0)
    x = (torch.randn(8, 4, 3, 3) * 3 + 100).requires_grad_(True)
    reference = nn.BatchNorm2d(4)
    output = reference(x)
    output.sin().sum().backward()
    assert torch.allclose(torch.cat([r[0] for r in results]), output, atol=2e-4)
    assert torch.allclose(torch.cat([r[1] for r in results]), x.grad, atol=2e-4)
    assert torch.allclose(sum(r[4] for r in results), reference.weight.grad, atol=2e-3)
    for rank in results:
        assert torch.allclose(rank[2], reference.running_mean, rtol=1e-5)
        assert torch.allclose(rank[3], reference.running_var, rtol=1e-3)


def _synchronized_step(world_size):
    torch.manual_seed(0)
    client, server = build_split_resnet18(10, synchronized_batchnorm=True)
    inputs, labels = _batch()
    owners = torch.arange(8)
    rank = dist.get_rank()
    shard = torch.arange(8).split(SHARDS[world_size])[rank].tolist()
    train_step(
        client, server, *make_optimizers(client, server, **SGD),
        inputs=inputs[shard], labels=labels[shard], batch=shard, owners=owners,
        label_smoothing=0.1, global_samples=8,
    )
    return {name: tensor.clone() for name, tensor in combined_state(client, server).items()}


@pytest.mark.parametrize("world_size", [2, 4])
def test_synchronized_step_matches_the_global_batch(world_size):
    """Sharding one batch, including empty shards, gives the single-process update."""
    results = run_distributed(_synchronized_step, world_size, world_size)
    torch.manual_seed(0)
    client, server = build_split_resnet18(10)
    inputs, labels = _batch()
    train_step(
        client, server, *make_optimizers(client, server, **SGD),
        inputs=inputs, labels=labels, batch=range(8), owners=torch.arange(8), label_smoothing=0.1,
    )
    for name, tensor in combined_state(client, server).items():
        for rank in results:
            assert torch.equal(rank[name], results[0][name])
            assert torch.allclose(rank[name].double(), tensor.double(), atol=2e-4), name


def _fuse():
    rank = dist.get_rank()
    torch.manual_seed(rank)
    client, server = build_split_resnet18(10)
    before = {name: tensor.clone() for name, tensor in combined_state(client, server).items()}
    counter = CollectiveCounter()
    fuse_across_ranks(client, server, [3, 1, 4, 2], counter)
    return before, {n: t.clone() for n, t in combined_state(client, server).items()}, counter.tensor_bytes


def test_distributed_fusion_matches_local_fusion():
    results = run_distributed(_fuse, 4)
    expected = fuse_states([r[0] for r in results], [3, 1, 4, 2])
    for _, after, size in results:
        assert size > 0
        for name, tensor in expected.items():
            assert torch.allclose(after[name].double(), tensor.double(), rtol=1e-6), name


def _run(method, sequential, world_size):
    data = make_data()
    config = RunConfig(
        method=method, sequential=sequential, epochs=2, batch_size=16, num_clients=16,
        validation_size=64, num_workers=0, targets=(0.0, 0.99), eval_batch_size=32,
    )
    run = DistributedRun(config, init_distributed("cpu"), data)
    summary = run.run()
    return summary, {n: t.clone() for n, t in combined_state(run.client, run.server).items()}


METHODS = [
    ("gcpsl_random", 4), ("gcpsl_size_balanced", 4), ("gcpsl_label_aware", 4),
    ("gpsl_sharded_sync", 4), ("gpsl_data_parallel", 4), ("gpsl_repartitioned", 4),
]


@pytest.mark.parametrize(("method", "world_size"), METHODS)
def test_every_method_trains(method, world_size):
    results = run_distributed(_run, world_size, method, False, world_size)
    summary = results[0][0]
    assert all(r[0] is None for r in results[1:])
    assert len(summary["history"]) == 2
    assert summary["targets"][0]["epoch"] == 1 and summary["targets"][1]["epoch"] is None
    participation = summary["participation"]
    assert participation["coverage"] == 1.0
    assert participation["direct_participation"] == pytest.approx(
        participation["model_update_participation"] * participation["within_round_participation"]
    )
    for other in results[1:]:
        assert all(torch.equal(other[1][n], t) for n, t in results[0][1].items())


@pytest.mark.parametrize("method", ["gpsl_single_worker", "gpsl_larger_batch"])
def test_single_worker_methods_train(method):
    summary, _ = run_distributed(_run, 1, method, False, 1)[0]
    assert len(summary["history"]) == 2


def test_sequential_execution_gives_the_concurrent_result():
    concurrent = run_distributed(_run, 2, "gcpsl_label_aware", False, 2)[0]
    sequential = run_distributed(_run, 2, "gcpsl_label_aware", True, 2)[0]
    assert all(torch.equal(sequential[1][n], t) for n, t in concurrent[1].items())
