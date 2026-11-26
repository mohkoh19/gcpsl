from __future__ import annotations

import io
import os
import socket
import traceback

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

torch.set_num_threads(2)


class FakeCifar(Dataset):
    """Random images with the interface of a torchvision CIFAR dataset."""

    def __init__(self, size: int, num_classes: int, seed: int):
        rng = np.random.default_rng(seed)
        self.data = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = [int(i % num_classes) for i in rng.permutation(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return Image.fromarray(self.data[index]), self.targets[index]


def make_data(*, size: int = 256, num_clients: int = 16, validation_size: int = 64, seed: int = 3):
    from gcpsl.data.cifar import build_client_data

    return build_client_data(
        dataset="cifar10",
        train_dataset=FakeCifar(size + validation_size, 10, seed),
        test_dataset=FakeCifar(64, 10, seed + 1),
        num_clients=num_clients,
        classes_per_client=2,
        alpha=3.0,
        partition_seed=seed,
        augmentation_base_seed=seed,
        validation_size=validation_size,
    )


@pytest.fixture
def small_data():
    return make_data()


def _worker(rank, world_size, port, function, queue, args):
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        buffer = io.BytesIO()
        torch.save((True, function(*args)), buffer)
    except Exception:  # reported by the parent, so that a failure does not hang the test
        buffer = io.BytesIO()
        torch.save((False, traceback.format_exc()), buffer)
    queue.put((rank, buffer.getvalue()))
    dist.destroy_process_group()


def run_distributed(function, world_size: int, *args):
    """Run ``function`` on ``world_size`` CPU processes and return the results by rank."""
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    spawned = mp.spawn(
        _worker, args=(world_size, port, function, queue, args), nprocs=world_size, join=False
    )
    results = {}
    while len(results) < world_size:
        rank, payload = queue.get()
        success, results[rank] = torch.load(io.BytesIO(payload), weights_only=False)
        if not success:
            for process in spawned.processes:
                process.terminate()
            pytest.fail(f"rank {rank} failed:\n{results[rank]}")
    spawned.join()
    return [results[rank] for rank in range(world_size)]
