"""Splitting the network into a client and a server segment.

The split follows the implementation used for all experiments in the paper.  It walks
the module tree up to ``cut_layer`` and moves every leaf it meets into a sequential
client segment, leaving ``nn.Identity`` behind.  For ResNet-18 cut before ``layer2``
this has two consequences (see ``docs/implementation_notes.md``):

* the client segment applies the convolutions of ``layer1`` without their skip connections;
* the server segment keeps the two emptied residual blocks.  Each computes ``x + x``, so
  the server multiplies the received cut activation by four.
"""

from __future__ import annotations

from torch import nn

from gcpsl.models.resnet import resnet18_small
from gcpsl.normalization import convert_batchnorm2d

CUT_LAYER = "layer2"
GROUP_NORM_GROUPS = 32


def _move_leaves(module: nn.Module, cut_layer: str, prefix: str, target: nn.Sequential) -> bool:
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name == cut_layer:
            return True
        if list(child.children()):
            branch = nn.Sequential()
            target.add_module(name, branch)
            if _move_leaves(child, cut_layer, full_name, branch):
                return True
        else:
            target.add_module(name, child)
            setattr(module, name, nn.Identity())
    return False


def split_model(model: nn.Module, cut_layer: str = CUT_LAYER) -> tuple[nn.Module, nn.Module]:
    """Split ``model`` in place and return ``(client_segment, server_segment)``."""
    client = nn.Sequential()
    if not _move_leaves(model, cut_layer, "", client):
        raise ValueError(f"cut layer {cut_layer!r} not found")
    return client, model


def replace_batchnorm_with_groupnorm(model: nn.Module, num_groups: int = GROUP_NORM_GROUPS) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.GroupNorm(num_groups, module.num_features))
        else:
            replace_batchnorm_with_groupnorm(module, num_groups)
    return model


def build_split_resnet18(
    num_classes: int, *, synchronized_batchnorm: bool = False
) -> tuple[nn.Module, nn.Module]:
    """Client segment with GroupNorm, server segment with (optionally synchronized) BatchNorm."""
    client, server = split_model(resnet18_small(num_classes))
    client = replace_batchnorm_with_groupnorm(client)
    if synchronized_batchnorm:
        server = convert_batchnorm2d(server)
    return client, server


def combined_state(client: nn.Module, server: nn.Module) -> dict:
    """State of both segments under ``client.*`` and ``server.*`` keys."""
    return {
        **{f"client.{name}": tensor for name, tensor in client.state_dict().items()},
        **{f"server.{name}": tensor for name, tensor in server.state_dict().items()},
    }


def load_combined_state(client: nn.Module, server: nn.Module, state: dict) -> None:
    client.load_state_dict({k[len("client."):]: v for k, v in state.items() if k.startswith("client.")})
    server.load_state_dict({k[len("server."):]: v for k, v in state.items() if k.startswith("server.")})
