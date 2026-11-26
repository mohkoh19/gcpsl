"""Training loop of the real-execution study.

One process runs on each worker.  GCPSL runs one cluster per process and fuses the
replicas after every epoch; the synchronous baselines synchronize at every step.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from gcpsl.clustering import build_clusters
from gcpsl.data.cifar import BatchLoader, ClientData, load_client_data
from gcpsl.distributed.context import DistributedContext
from gcpsl.distributed.timing import RegionTimer
from gcpsl.fusion import assert_identical_replicas, fuse_across_ranks
from gcpsl.methods import METHODS, Method
from gcpsl.metrics import ParticipationTracker, evaluate, make_targets
from gcpsl.models.split import build_split_resnet18, combined_state, load_combined_state
from gcpsl.normalization import CollectiveCounter, batchnorm_counter
from gcpsl.sampling import EpochSchedule, build_epoch_schedule
from gcpsl.seeding import SeedComponent, derive_seed, seed_process, to_uint32_seed
from gcpsl.step import StepResult, make_optimizers, train_step


@dataclass(frozen=True)
class RunConfig:
    """Configuration of one real-execution run.  Defaults are those of the paper."""

    method: Method = Method.GCPSL_SIZE_BALANCED
    sequential: bool = False
    seed: int = 40
    epochs: int = 100
    dataset: str = "cifar10"
    data_dir: str = "data"
    download: bool = False
    num_clients: int = 256
    classes_per_client: int = 2
    dirichlet_alpha: float = 3.0
    validation_size: int = 5000
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1
    eval_batch_size: int = 256
    targets: tuple[float, ...] = (0.80, 0.85, 0.88)
    cluster_options: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    strict_determinism: bool = True
    warmup: bool = True
    output_dir: str | None = None
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", Method(self.method))
        object.__setattr__(self, "targets", tuple(float(t) for t in self.targets))


class _WorkerSeeder:
    def __init__(self, seed: int, rank: int):
        self.seed, self.rank = seed, rank

    def __call__(self, worker: int) -> None:
        torch.manual_seed(
            derive_seed(self.seed, SeedComponent.DATALOADER_WORKER, rank=self.rank, worker=worker)
        )


def load_data(config: RunConfig) -> ClientData:
    return load_client_data(
        dataset=config.dataset,
        data_dir=config.data_dir,
        download=config.download,
        num_clients=config.num_clients,
        classes_per_client=config.classes_per_client,
        alpha=config.dirichlet_alpha,
        partition_seed=to_uint32_seed(derive_seed(config.seed, SeedComponent.CLIENT_PARTITION)),
        augmentation_base_seed=config.seed,
        validation_size=config.validation_size,
    )


class DistributedRun:
    """One training run of one method on all ranks."""

    def __init__(self, config: RunConfig, context: DistributedContext, data: ClientData | None = None):
        self.config = config
        self.context = context
        self.spec = METHODS[config.method]
        if self.spec.single_worker and context.world_size != 1:
            raise ValueError(f"{config.method.value} runs on one process")
        if config.sequential and not self.spec.periodic:
            raise ValueError("sequential execution is defined for periodic methods only")
        self.timer = RegionTimer(context)
        startup_begin = time.perf_counter()
        if data is None:
            if config.download:
                # One process downloads while the others wait.
                if context.is_main:
                    load_data(config)
                context.barrier()
            data = load_data(replace(config, download=False))
        self.data = data
        self.loader = BatchLoader(
            self.data,
            num_workers=config.num_workers,
            pin_memory=context.device.type == "cuda",
            worker_init_fn=_WorkerSeeder(config.seed, context.rank),
            generator=torch.Generator().manual_seed(
                derive_seed(config.seed, SeedComponent.DATALOADER_GENERATOR, rank=context.rank)
            ),
        )
        self.startup_seconds = time.perf_counter() - startup_begin
        self.fusion_counter = CollectiveCounter()
        self.gradient_counter = CollectiveCounter()
        self.payload_bytes = 0
        self.participation = ParticipationTracker(self.data.num_clients)

    # ------------------------------------------------------------------ setup

    def _build_models(self) -> None:
        seed_process(
            derive_seed(self.config.seed, SeedComponent.MODEL_INITIALIZATION),
            strict=self.config.strict_determinism,
        )
        client, server = build_split_resnet18(
            self.data.spec.num_classes, synchronized_batchnorm=self.spec.synchronous
        )
        self.client = client.to(self.context.device)
        self.server = server.to(self.context.device)
        if self.context.world_size > 1:
            for tensor in combined_state(self.client, self.server).values():
                dist.broadcast(tensor, src=0)
        self.client_optimizer, self.server_optimizer = make_optimizers(
            self.client,
            self.server,
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.nesterov,
        )

    def _initialize(self) -> None:
        self._build_models()
        assert_identical_replicas(self.client, self.server)
        seed_process(
            derive_seed(self.config.seed, SeedComponent.RANK_RUNTIME, rank=self.context.rank),
            strict=self.config.strict_determinism,
        )
        self.clusters = None
        if self.spec.clusters is not None:
            if self.context.is_main:
                self.clusters = build_clusters(
                    self.spec.clusters,
                    sample_counts=[len(indices) for indices in self.data.client_indices],
                    label_histograms=self.data.label_histograms(),
                    num_clusters=self.context.world_size,
                    seed=to_uint32_seed(derive_seed(self.config.seed, SeedComponent.CLUSTER_ASSIGNMENT)),
                    **self.config.cluster_options,
                )
            self.clusters = self.context.broadcast_object(self.clusters)
        self.client_parameter_bytes = sum(p.numel() * p.element_size() for p in self.client.parameters())
        self.client_state_bytes = sum(t.numel() * t.element_size() for t in self.client.state_dict().values())
        self.state_bytes = sum(
            t.numel() * t.element_size() for t in combined_state(self.client, self.server).values()
        )

    def _warm_up(self) -> None:
        """One disposable update outside the measured time."""
        self._build_models()
        schedule = self._schedule(0, warmup=True)
        batch = schedule.batches[self.context.rank][0]
        inputs, labels = self._load_once(batch)
        self._step(batch, schedule, 0, inputs, labels)
        self.context.barrier()
        del self.client, self.server, self.client_optimizer, self.server_optimizer
        self.gradient_counter = CollectiveCounter()
        self.timer = RegionTimer(self.context)
        if self.context.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_once(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if not batch:
            return self._empty_batch()
        samples = [self.data.sample(("train", 0, int(i))) for i in batch]
        return torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples])

    @staticmethod
    def _empty_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)

    # --------------------------------------------------------------- schedule

    def _schedule(self, epoch: int, *, warmup: bool = False) -> EpochSchedule:
        schedule = None
        if self.context.is_main:
            clusters = getattr(self, "clusters", None)
            if warmup and self.spec.clusters is not None:
                world = self.context.world_size
                clusters = [list(range(rank, self.data.num_clients, world)) for rank in range(world)]
            schedule = build_epoch_schedule(
                self.config.method,
                client_indices=self.data.client_indices,
                world_size=self.context.world_size,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                epoch=epoch,
                clusters=clusters,
            )
        return self.context.broadcast_object(schedule)

    # --------------------------------------------------------------- training

    def _step(self, batch, schedule: EpochSchedule, phase: int, inputs, labels) -> StepResult:
        with self.timer.region("host_to_device"):
            inputs = inputs.to(self.context.device, non_blocking=inputs.is_pinned())
            labels = labels.to(self.context.device, non_blocking=labels.is_pinned())
        global_samples = None
        if self.spec.synchronous:
            global_samples = sum(len(rank[phase]) for rank in schedule.batches)
        result = train_step(
            self.client,
            self.server,
            self.client_optimizer,
            self.server_optimizer,
            inputs=inputs,
            labels=labels,
            batch=batch,
            owners=self.data.owners,
            label_smoothing=self.config.label_smoothing,
            global_samples=global_samples,
            region=self.timer.region,
        )
        if result.gradient_sync_bytes:
            self.gradient_counter.tensor_bytes += result.gradient_sync_bytes
            self.gradient_counter.seconds += result.gradient_sync_seconds
        return result

    def _train_epoch(self, epoch: int) -> tuple[float, float, float]:
        with self.timer.region("schedule"):
            schedule = self._schedule(epoch)
        rank = self.context.rank
        batches = schedule.batches[rank]
        iterator = iter(self.loader.train_batches(batches, epoch=epoch))
        totals = [0.0, 0.0, 0.0]

        def execute(batch, phase: int) -> StepResult:
            with self.timer.region("data_wait"):
                inputs, labels = next(iterator) if batch else self._empty_batch()
            return self._step(batch, schedule, phase, inputs, labels)

        for phase, batch in enumerate(batches):
            result = None
            if self.config.sequential:
                # Workers take turns within every system phase.
                for turn in range(self.context.world_size):
                    if turn == rank and batch:
                        result = execute(batch, phase)
                    with self.timer.region("barrier_wait"):
                        self.context.barrier()
            elif batch or self.spec.synchronous:
                result = execute(batch, phase)

            if result is not None:
                totals[0] += result.loss_sum
                totals[1] += result.correct
                totals[2] += result.samples
                self._count_payload(result, schedule, phase)

        if next(iterator, None) is not None:
            raise RuntimeError("the loader returned more batches than scheduled")
        if self.spec.periodic:
            with self.timer.region("barrier_wait"):
                self.context.barrier()
            with self.timer.region("fusion"):
                fuse_across_ranks(self.client, self.server, schedule.sample_counts, self.fusion_counter)
            if self.context.is_main:
                self.payload_bytes += (
                    2 * self.context.world_size * self.state_bytes
                    + self.data.num_clients * self.client_state_bytes
                )
        with self.timer.region("barrier_wait"):
            self.context.barrier()
        with self.timer.region("replica_check"):
            assert_identical_replicas(self.client, self.server)
        if self.context.is_main:
            self.participation.add_epoch(
                schedule.populations, schedule.batches, self.data.owners, shared=not self.spec.periodic
            )
        totals = self.context.all_reduce_sum(totals)
        return tuple(totals)

    def _count_payload(self, result: StepResult, schedule: EpochSchedule, phase: int) -> None:
        """Logical application payload: every tensor transfer is counted once."""
        self.payload_bytes += 2 * result.cut_bytes + result.label_bytes
        if self.spec.periodic:
            population = schedule.populations[self.context.rank]
            exchanged = len(result.active_clients) + len(population)
            self.payload_bytes += exchanged * self.client_parameter_bytes
        elif self.context.is_main:
            samples = [i for rank_batches in schedule.batches for i in rank_batches[phase]]
            active = set(self.data.owners[torch.as_tensor(samples)].tolist())
            self.payload_bytes += (len(active) + self.data.num_clients) * self.client_parameter_bytes

    def _evaluate(self, client, server, split: str):
        return evaluate(
            client,
            server,
            self.loader.eval_batches(split, self.config.eval_batch_size),
            label_smoothing=self.config.label_smoothing,
            device=self.context.device,
        )

    def _collective_totals(self) -> dict[str, float]:
        batchnorm = batchnorm_counter(self.server)
        names = ("fusion", "gradient_sync", "batchnorm")
        counters = (self.fusion_counter, self.gradient_counter, batchnorm)
        values = [float(c.tensor_bytes) for c in counters] + [c.seconds for c in counters]
        values = self.context.all_reduce_sum(values)
        return {
            **{f"{name}_bytes": value for name, value in zip(names, values[:3], strict=True)},
            **{
                f"{name}_seconds_per_rank": value / self.context.world_size
                for name, value in zip(names, values[3:], strict=True)
            },
        }

    def run(self) -> dict[str, Any] | None:
        """Train, evaluate and return the run summary on rank 0."""
        config, context = self.config, self.context
        if config.warmup:
            self._warm_up()
        context.barrier()
        start = time.perf_counter()
        startup = context.all_reduce_max(self.startup_seconds)
        with self.timer.region("initialization"):
            self._initialize()

        targets = make_targets(config.targets)
        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_state = None
        training_seconds = 0.0
        for epoch in range(1, config.epochs + 1):
            before = self.timer.training_seconds
            loss, correct, samples = self._train_epoch(epoch - 1)
            # Critical path: the slowest rank of every epoch.
            training_seconds += context.all_reduce_max(self.timer.training_seconds - before)
            validation = None
            with self.timer.region("validation"):
                if context.is_main:
                    validation = self._evaluate(self.client, self.server, "validation")
                validation = context.broadcast_object(validation)
            # Elapsed time includes startup (without warm-up), initialization and clustering,
            # training, collectives, barrier waits and all validation passes so far.
            elapsed = startup + context.all_reduce_max(time.perf_counter() - start)
            for target in targets:
                target.update(accuracy=validation.accuracy, epoch=epoch, seconds=elapsed)
            payload = context.all_reduce_sum([float(self.payload_bytes)])[0]
            record = {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "training_seconds": training_seconds,
                "train_loss": loss / samples,
                "train_accuracy": correct / samples,
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
                "logical_payload_bytes": payload,
                **self._collective_totals(),
            }
            history.append(record)
            with self.timer.region("checkpoint"):
                better = best is None or (validation.accuracy, -validation.loss) > (
                    best["validation_accuracy"],
                    -best["validation_loss"],
                )
                if better:
                    best = record
                    if context.is_main:
                        best_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in combined_state(self.client, self.server).items()
                        }
            if context.is_main:
                self._write_epoch(record)
                print(
                    f"epoch {epoch:3d}  {elapsed / 60:7.2f} min  "
                    f"train loss {record['train_loss']:.4f}  val acc {100 * validation.accuracy:.2f}%",
                    flush=True,
                )

        regions = {
            name: context.all_reduce_sum([self.timer.seconds[name]])[0] / context.world_size
            for name in sorted(self.timer.seconds)
        }
        if not context.is_main:
            return None
        with self.timer.region("test"):
            final_test = self._evaluate(self.client, self.server, "test")
            best_client, best_server = copy.deepcopy(self.client), copy.deepcopy(self.server)
            load_combined_state(best_client, best_server, best_state)
            best_test = self._evaluate(best_client, best_server, "test")
        summary = {
            "config": {**asdict(config), "method": config.method.value},
            "world_size": context.world_size,
            "targets": [asdict(target) for target in targets],
            "final_test_accuracy": final_test.accuracy,
            "best_validation_epoch": best["epoch"],
            "best_validation_accuracy": best["validation_accuracy"],
            "best_checkpoint_test_accuracy": best_test.accuracy,
            "participation": self.participation.summary(),
            "region_seconds_per_rank": regions,
            "clusters": self.clusters,
            "history": history,
        }
        self._write_summary(summary, best_state)
        return summary

    # ----------------------------------------------------------------- output

    def _output_dir(self) -> Path | None:
        if self.config.output_dir is None:
            return None
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_epoch(self, record: dict[str, Any]) -> None:
        path = self._output_dir()
        if path is not None:
            with (path / "epochs.jsonl").open("w" if record["epoch"] == 1 else "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _write_summary(self, summary: dict[str, Any], best_state) -> None:
        path = self._output_dir()
        if path is None:
            return
        (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.config.save_checkpoints:
            torch.save(best_state, path / "best.pt")
            torch.save(combined_state(self.client, self.server), path / "final.pt")
