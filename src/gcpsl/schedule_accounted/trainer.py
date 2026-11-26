"""Single-process training of all cluster workloads.

Learning follows GCPSL exactly: every cluster has its own replicas, sampler and
optimizers, and fusion couples the clusters.  Time is not measured but counted in
schedule rounds.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gcpsl.clustering import build_clusters
from gcpsl.data.cifar import BatchLoader, ClientData, load_client_data
from gcpsl.fusion import fuse_states
from gcpsl.models.split import build_split_resnet18, combined_state, load_combined_state
from gcpsl.sampling import client_queues, gpsl_batches
from gcpsl.schedule_accounted.accounting import (
    AnalyticalPayload,
    batch_deviation,
    cluster_local_inactivity,
    schedule_rounds,
)
from gcpsl.seeding import SeedComponent, derive_seed, seed_process
from gcpsl.step import make_optimizers, train_step


@dataclass(frozen=True)
class ScheduleConfig:
    """Configuration of one schedule-accounted run.

    ``clusters=None`` runs GPSL, a single workload over all clients.  ``fusion`` is
    ``"epoch"``, ``"none"`` or a number of local steps summed over all clusters; a numeric
    interval also fuses at the end of every epoch.  ``batch_clients`` runs the client segment
    of a batch in one pass, which changes nothing but rounding and speed.
    """

    clusters: str | None = "size_balanced"
    num_clusters: int = 4
    fusion: str | int = "epoch"
    sequential: bool = False
    seed: int = 40
    epochs: int = 100
    dataset: str = "cifar10"
    data_dir: str = "data"
    download: bool = False
    num_clients: int = 256
    classes_per_client: int = 2
    dirichlet_alpha: float = 3.0
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1
    targets: tuple[float, ...] = (0.85, 0.88)
    cluster_options: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    batch_clients: bool = True
    device: str = "auto"
    strict_determinism: bool = True
    output_dir: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(float(t) for t in self.targets))
        if self.clusters is None and self.num_clusters != 1:
            raise ValueError("GPSL has a single workload: set num_clusters=1")
        if isinstance(self.fusion, str) and self.fusion not in {"epoch", "none"}:
            raise ValueError("fusion must be 'epoch', 'none' or a number of local steps")


def load_data(config: ScheduleConfig) -> ClientData:
    """All training examples, partitioned over the clients; no validation split."""
    return load_client_data(
        dataset=config.dataset,
        data_dir=config.data_dir,
        download=config.download,
        num_clients=config.num_clients,
        classes_per_client=config.classes_per_client,
        alpha=config.dirichlet_alpha,
        partition_seed=config.seed,
        augmentation_base_seed=config.seed,
        validation_size=0,
    )


def execution_order(steps: list[int], *, sequential: bool, seed: int) -> Iterator[int]:
    """Cluster executed at every local step of an epoch.

    The order matters only for step-based fusion intervals.  By default clusters are
    interleaved at random; sequential execution runs one cluster after another.
    """
    if sequential:
        for cluster, count in enumerate(steps):
            yield from [cluster] * count
        return
    remaining = list(steps)
    alive = list(range(len(steps)))
    rng = random.Random(seed)
    while alive:
        cluster = rng.choice(alive)
        if remaining[cluster]:
            remaining[cluster] -= 1
            yield cluster
        else:
            alive.remove(cluster)


@dataclass
class Workload:
    """Replicas and optimizers of one cluster."""

    client: nn.Module
    server: nn.Module
    client_optimizer: torch.optim.Optimizer
    server_optimizer: torch.optim.Optimizer


class ScheduleAccountedRun:
    def __init__(self, config: ScheduleConfig, data: ClientData | None = None):
        self.config = config
        name = config.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(name)
        self.data = data if data is not None else load_data(config)
        self.loader = BatchLoader(
            self.data, num_workers=config.num_workers, pin_memory=self.device.type == "cuda"
        )
        if config.clusters is None:
            self.clusters = [list(range(self.data.num_clients))]
        else:
            self.clusters = build_clusters(
                config.clusters,
                sample_counts=[len(indices) for indices in self.data.client_indices],
                label_histograms=self.data.label_histograms(),
                num_clusters=config.num_clusters,
                seed=config.seed,
                **config.cluster_options,
            )
        self.sample_counts = [sum(len(self.data.client_indices[k]) for k in c) for c in self.clusters]
        labels = self.data.train_labels
        self.global_distribution = (
            torch.bincount(labels, minlength=self.data.spec.num_classes).double() / len(labels)
        )
        self._build()

    def _build(self) -> None:
        seed_process(
            derive_seed(self.config.seed, SeedComponent.MODEL_INITIALIZATION),
            strict=self.config.strict_determinism,
        )
        client, server = build_split_resnet18(self.data.spec.num_classes)
        self.workloads = []
        for _ in self.clusters:
            replica = copy.deepcopy(client).to(self.device), copy.deepcopy(server).to(self.device)
            optimizers = make_optimizers(
                *replica,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.nesterov,
            )
            self.workloads.append(Workload(*replica, *optimizers))
        self.payload = AnalyticalPayload(
            client_bytes=sum(p.numel() * p.element_size() for p in client.parameters()),
            server_bytes=sum(p.numel() * p.element_size() for p in server.parameters()),
            num_clients=self.data.num_clients,
        )

    def epoch_batches(self, epoch: int) -> list[list[tuple[int, ...]]]:
        """Batches of every cluster in one zero-based epoch."""
        queues = client_queues(self.data.client_indices, seed=self.config.seed, epoch=epoch)
        single = len(self.clusters) == 1
        return [
            gpsl_batches(
                population,
                queues,
                batch_size=self.config.batch_size,
                sampler_seed=(
                    derive_seed(self.config.seed, SeedComponent.GLOBAL_SAMPLER)
                    if single
                    else derive_seed(self.config.seed, SeedComponent.CLUSTER_SAMPLER, rank=index)
                ),
                epoch=epoch,
            )
            for index, population in enumerate(self.clusters)
        ]

    def fused_state(self) -> dict[str, torch.Tensor]:
        return fuse_states([combined_state(w.client, w.server) for w in self.workloads], self.sample_counts)

    def _fuse(self) -> None:
        state = self.fused_state()
        for workload in self.workloads:
            load_combined_state(workload.client, workload.server, state)
        self.payload.add_fusion(len(self.workloads))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        batches = self.epoch_batches(epoch)
        steps = [len(cluster) for cluster in batches]
        order = list(execution_order(steps, sequential=self.config.sequential, seed=self.config.seed))
        cursors = [0] * len(batches)
        requests = []
        for cluster in order:
            requests.append((cluster, batches[cluster][cursors[cluster]]))
            cursors[cluster] += 1

        loss_sum, samples, deviation = 0.0, 0, 0.0
        loaded = self.loader.train_batches([batch for _, batch in requests], epoch=epoch)
        for step, ((cluster, batch), (inputs, labels)) in enumerate(zip(requests, loaded, strict=True), 1):
            workload = self.workloads[cluster]
            deviation += batch_deviation(labels, self.global_distribution)
            result = train_step(
                workload.client,
                workload.server,
                workload.client_optimizer,
                workload.server_optimizer,
                inputs=inputs.to(self.device, non_blocking=True),
                labels=labels.to(self.device, non_blocking=True),
                batch=batch,
                owners=self.data.owners,
                label_smoothing=self.config.label_smoothing,
                batch_clients=self.config.batch_clients,
            )
            loss_sum += result.loss_sum
            samples += result.samples
            self.payload.add_round(cut_bytes=result.cut_bytes, cluster_size=len(self.clusters[cluster]))
            if len(self.clusters) > 1 and self._fuse_after(step, len(requests)):
                self._fuse()

        return {
            "train_loss": loss_sum / samples,
            "rounds": schedule_rounds(steps, sequential=self.config.sequential),
            "cluster_steps": steps,
            "inactivity": cluster_local_inactivity(self.clusters, batches, self.data.owners),
            "batch_deviation": deviation / len(requests),
        }

    def _fuse_after(self, step: int, steps_per_epoch: int) -> bool:
        if self.config.fusion == "none":
            return False
        if step == steps_per_epoch:
            return True
        return self.config.fusion != "epoch" and step % int(self.config.fusion) == 0

    def evaluate(self) -> dict[str, float]:
        """Test accuracy of the fused model."""
        reference = self.workloads[0]
        client, server = copy.deepcopy(reference.client), copy.deepcopy(reference.server)
        if len(self.workloads) > 1:
            load_combined_state(client, server, self.fused_state())
        client.eval()
        server.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.loader.eval_batches("test", self.config.batch_size):
                predictions = server(client(inputs.to(self.device))).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        return {"test_accuracy": correct / total}

    def _write_epoch(self, record: dict[str, Any]) -> None:
        if self.config.output_dir is None:
            return
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "epochs.jsonl").open("w" if record["epoch"] == 1 else "a") as handle:
            handle.write(json.dumps(record) + "\n")

    def run(self) -> dict[str, Any]:
        history, rounds = [], 0
        reached: dict[float, int | None] = dict.fromkeys(self.config.targets)
        for epoch in range(1, self.config.epochs + 1):
            record = {"epoch": epoch, **self.train_epoch(epoch - 1)}
            record.update(self.evaluate())
            rounds += record["rounds"]
            record["cumulative_rounds"] = rounds
            record["cumulative_payload_bytes"] = self.payload.total
            for target, at in reached.items():
                if at is None and record["test_accuracy"] >= target:
                    reached[target] = rounds
            history.append(record)
            self._write_epoch(record)
            print(
                f"epoch {epoch:3d}  rounds {rounds:6d}  train loss {record['train_loss']:.4f}  "
                f"test acc {100 * record['test_accuracy']:.2f}%",
                flush=True,
            )
        summary = {
            "config": asdict(self.config),
            "clusters": self.clusters,
            "final_test_accuracy": history[-1]["test_accuracy"],
            "rounds_to_target": {str(target): at for target, at in reached.items()},
            "inactivity": sum(r["inactivity"] for r in history) / len(history),
            "batch_deviation": sum(r["batch_deviation"] for r in history) / len(history),
            "peak_gpu_bytes": (
                torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
            ),
            "history": history,
        }
        if self.config.output_dir is not None:
            path = Path(self.config.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
