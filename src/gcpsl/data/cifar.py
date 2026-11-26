"""Index-addressable CIFAR data with identity-keyed augmentation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gcpsl.data.partition import (
    extended_dirichlet_partition,
    owners_from_partition,
    stratified_split,
)
from gcpsl.seeding import augmentation_seed


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    crop_padding: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    torchvision_class: str


DATASETS = {
    "cifar10": DatasetSpec(10, 1, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), "CIFAR10"),
    "cifar100": DatasetSpec(100, 4, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), "CIFAR100"),
}


class CifarTransform:
    """Random crop, horizontal flip and normalization.

    The augmentation of a sample depends only on the run seed, the epoch and the
    original dataset index.
    """

    def __init__(self, spec: DatasetSpec, seed: int, augment: bool):
        self.spec = spec
        self.seed = int(seed)
        self.augment = bool(augment)

    def __call__(self, image, *, epoch: int, source_index: int) -> torch.Tensor:
        from torchvision.transforms import functional as F

        if self.augment:
            generator = torch.Generator().manual_seed(
                augmentation_seed(self.seed, epoch=epoch, source_index=source_index)
            )
            image = F.pad(image, self.spec.crop_padding, fill=0, padding_mode="constant")
            _, height, width = F.get_dimensions(image)
            top = int(torch.randint(0, height - 32 + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - 32 + 1, (1,), generator=generator).item())
            image = F.crop(image, top, left, 32, 32)
            if float(torch.rand((), generator=generator).item()) < 0.5:
                image = F.hflip(image)
        return F.normalize(F.to_tensor(image), self.spec.mean, self.spec.std)


@dataclass
class ClientData:
    """A dataset, its splits, and the partition of the training split over clients.

    ``client_indices`` and ``owners`` index the training split, not the original dataset.
    """

    spec: DatasetSpec
    train_dataset: Dataset
    test_dataset: Dataset
    train_sources: list[int]
    validation_sources: list[int]
    train_labels: torch.Tensor
    validation_labels: torch.Tensor
    test_labels: torch.Tensor
    client_indices: list[list[int]]
    owners: torch.Tensor
    train_transform: CifarTransform
    eval_transform: CifarTransform

    @property
    def num_clients(self) -> int:
        return len(self.client_indices)

    def split(self, name: str) -> tuple[Dataset, Sequence[int], torch.Tensor]:
        if name == "train":
            return self.train_dataset, self.train_sources, self.train_labels
        if name == "validation":
            return self.train_dataset, self.validation_sources, self.validation_labels
        if name == "test":
            return self.test_dataset, range(len(self.test_labels)), self.test_labels
        raise ValueError(f"unknown split {name!r}")

    def sample(self, request: tuple[str, int, int]) -> tuple[torch.Tensor, int]:
        """Resolve one ``(split, epoch, index)`` request."""
        split, epoch, index = request
        dataset, sources, labels = self.split(split)
        source = int(sources[index])
        image, _ = dataset[source]
        transform = self.train_transform if split == "train" else self.eval_transform
        return transform(image, epoch=epoch, source_index=source), int(labels[index])

    def label_histograms(self) -> list[list[int]]:
        return [
            torch.bincount(
                self.train_labels[torch.as_tensor(indices)], minlength=self.spec.num_classes
            ).tolist()
            for indices in self.client_indices
        ]


class _RequestDataset(Dataset):
    def __init__(self, data: ClientData):
        self.data = data

    def __len__(self) -> int:
        return sum(len(self.data.split(name)[2]) for name in ("train", "validation", "test"))

    def __getitem__(self, request: tuple[str, int, int]) -> tuple[torch.Tensor, int]:
        return self.data.sample(request)


def build_client_data(
    *,
    dataset: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    classes_per_client: int,
    alpha: float,
    partition_seed: int,
    augmentation_base_seed: int,
    validation_size: int,
) -> ClientData:
    """Split, partition and wrap raw torchvision datasets.

    ``validation_size=0`` trains on the complete training set, as in the
    schedule-accounted studies.
    """
    spec = DATASETS[dataset]
    all_labels = np.asarray(train_dataset.targets, dtype=np.int64)
    if validation_size:
        train_sources, validation_sources = stratified_split(
            all_labels, validation_size=validation_size
        )
    else:
        train_sources, validation_sources = list(range(len(all_labels))), []
    train_labels = all_labels[np.asarray(train_sources, dtype=np.int64)]
    validation_labels = all_labels[np.asarray(validation_sources, dtype=np.int64)]
    client_indices = extended_dirichlet_partition(
        train_labels,
        num_clients=num_clients,
        classes_per_client=classes_per_client,
        alpha=alpha,
        seed=partition_seed,
    )
    return ClientData(
        spec=spec,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_sources=train_sources,
        validation_sources=validation_sources,
        train_labels=torch.as_tensor(train_labels, dtype=torch.long),
        validation_labels=torch.as_tensor(validation_labels, dtype=torch.long),
        test_labels=torch.as_tensor(np.asarray(test_dataset.targets), dtype=torch.long),
        client_indices=client_indices,
        owners=owners_from_partition(client_indices, len(train_labels)),
        train_transform=CifarTransform(spec, augmentation_base_seed, augment=True),
        eval_transform=CifarTransform(spec, augmentation_base_seed, augment=False),
    )


def load_client_data(*, dataset: str, data_dir: str | Path, download: bool = False, **kwargs) -> ClientData:
    """Load CIFAR-10 or CIFAR-100 with torchvision and partition it over the clients."""
    from torchvision import datasets as tv_datasets

    cls = getattr(tv_datasets, DATASETS[dataset].torchvision_class)
    root = str(Path(data_dir).expanduser())
    return build_client_data(
        dataset=dataset,
        train_dataset=cls(root=root, train=True, download=download),
        test_dataset=cls(root=root, train=False, download=download),
        **kwargs,
    )


class BatchLoader:
    """Loads exactly the batches a schedule asks for, in order.

    Worker processes persist across epochs.  Because every request carries its epoch and
    index, results do not depend on the number of workers.
    """

    def __init__(
        self,
        data: ClientData,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        worker_init_fn=None,
        generator: torch.Generator | None = None,
    ):
        self._batches: list[list[tuple[str, int, int]]] = []
        self._data = data
        self._loader = torch.utils.data.DataLoader(
            _RequestDataset(data),
            batch_sampler=self,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator,
            prefetch_factor=2 if num_workers else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn" if num_workers else None,
        )

    # The loader object doubles as its own batch sampler.
    def __iter__(self) -> Iterator[list[tuple[str, int, int]]]:
        yield from list(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def train_batches(
        self, batches: Sequence[Sequence[int]], *, epoch: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        self._batches = [[("train", epoch, int(i)) for i in batch] for batch in batches if len(batch)]
        yield from self._loader

    def eval_batches(self, split: str, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        total = len(self._data.split(split)[2])
        self._batches = [
            [(split, 0, i) for i in range(start, min(start + batch_size, total))]
            for start in range(0, total, batch_size)
        ]
        yield from self._loader
