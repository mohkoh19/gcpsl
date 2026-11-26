import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gcpsl.fusion import fuse_states, fusion_weights
from gcpsl.models.split import build_split_resnet18, combined_state, load_combined_state
from gcpsl.step import make_optimizers, train_step

SGD = dict(lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)


@pytest.fixture
def segments():
    torch.manual_seed(0)
    return build_split_resnet18(10)


def test_segment_contents(segments):
    client, server = segments
    assert sum(p.numel() for p in client.parameters()) == 149_824
    assert sum(p.numel() for p in server.parameters()) == 11_024_138
    assert not any(isinstance(m, nn.BatchNorm2d) for m in client.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in server.modules())
    assert sum(isinstance(m, nn.BatchNorm2d) for m in server.modules()) == 15
    assert client(torch.randn(2, 3, 32, 32)).shape == (2, 64, 32, 32)


def test_server_scales_the_cut_activation_by_four(segments):
    """Documented property of the split used in the paper."""
    client, server = segments
    seen = {}
    server.layer2.register_forward_pre_hook(lambda module, args: seen.update(x=args[0].clone()))
    cut = client(torch.randn(2, 3, 32, 32)).detach()
    server(cut.clone())
    assert torch.equal(seen["x"], 4 * cut)


def test_state_round_trip(segments):
    client, server = segments
    other = build_split_resnet18(10)
    load_combined_state(*other, combined_state(client, server))
    for name, tensor in combined_state(*other).items():
        assert torch.equal(tensor, combined_state(client, server)[name])


def test_step_equals_per_client_replicas(segments):
    """One shared client model is equivalent to one replica per client."""
    client, server = segments
    owners = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    inputs, labels = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    replicas = [copy.deepcopy(client) for _ in range(4)]  # client 3 supplies no data
    reference_server = copy.deepcopy(server)
    optimizers = [torch.optim.SGD(r.parameters(), **SGD) for r in replicas]
    server_optimizer = torch.optim.SGD(reference_server.parameters(), **SGD)

    cuts = [replicas[k](inputs[owners == k]) for k in range(3)]
    received = torch.cat(cuts).detach().requires_grad_(True)
    F.cross_entropy(reference_server(received.clone()), labels, label_smoothing=0.1).backward()
    for cut, gradient in zip(cuts, received.grad.split([3, 2, 3]), strict=True):
        cut.backward(gradient)
    groups = zip(*(r.parameters() for r in replicas[:3]), strict=True)
    summed = [sum(parameter.grad for parameter in group) for group in groups]
    for replica in replicas:
        for parameter, gradient in zip(replica.parameters(), summed, strict=True):
            parameter.grad = gradient.clone()
    for optimizer in [*optimizers, server_optimizer]:
        optimizer.step()

    result = train_step(
        client, server, *make_optimizers(client, server, **SGD),
        inputs=inputs, labels=labels, batch=range(8), owners=owners, label_smoothing=0.1,
    )
    assert result.active_clients == (0, 1, 2)
    for replica in replicas:
        for a, b in zip(client.parameters(), replica.parameters(), strict=True):
            assert torch.equal(a, b)
    for a, b in zip(server.parameters(), reference_server.parameters(), strict=True):
        assert torch.equal(a, b)


def test_waiting_worker_must_not_step(segments):
    client, server = segments
    with pytest.raises(ValueError, match="waiting worker"):
        train_step(
            client, server, *make_optimizers(client, server, **SGD),
            inputs=torch.empty(0, 3, 32, 32), labels=torch.empty(0, dtype=torch.long),
            batch=(), owners=torch.zeros(1, dtype=torch.long), label_smoothing=0.1,
        )


def test_fusion_weights_parameters_and_counters():
    states = [
        {"w": torch.tensor([0.0, 4.0]), "bn.num_batches_tracked": torch.tensor(3)},
        {"w": torch.tensor([8.0, 0.0]), "bn.num_batches_tracked": torch.tensor(5)},
    ]
    fused = fuse_states(states, [1, 3])
    assert fusion_weights([1, 3]) == [0.25, 0.75]
    assert fused["w"].tolist() == [6.0, 1.0]
    assert int(fused["bn.num_batches_tracked"]) == 5


def test_fusion_of_identical_states_is_the_identity(segments):
    state = combined_state(*segments)
    for name, tensor in fuse_states([state] * 3, [5, 7, 11]).items():
        assert torch.allclose(tensor.double(), state[name].double(), rtol=1e-6, atol=0)


def test_batched_clients_give_the_same_update():
    """One pass over the batch equals one pass per client, in exact arithmetic."""
    states = []
    for batch_clients in (False, True):
        torch.manual_seed(0)
        client, server = (m.double() for m in build_split_resnet18(10))
        optimizers = make_optimizers(client, server, **SGD)
        generator = torch.Generator().manual_seed(1)
        owners = torch.tensor([0] * 3 + [1] + [2] * 5 + [3] * 2 + [4] * 5)
        for _ in range(2):
            result = train_step(
                client, server, *optimizers,
                inputs=torch.randn(16, 3, 32, 32, dtype=torch.double, generator=generator),
                labels=torch.randint(0, 10, (16,), generator=generator),
                batch=range(16), owners=owners, label_smoothing=0.1, batch_clients=batch_clients,
            )
        assert result.active_clients == (0, 1, 2, 3, 4)
        states.append((result.loss_sum, combined_state(client, server)))
    assert states[0][0] == pytest.approx(states[1][0], rel=1e-12)
    for name, tensor in states[0][1].items():
        assert torch.allclose(tensor, states[1][1][name], rtol=1e-9, atol=1e-11), name
