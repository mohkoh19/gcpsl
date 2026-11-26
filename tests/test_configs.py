from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gcpsl.distributed.engine import RunConfig
from gcpsl.schedule_accounted.trainer import ScheduleConfig

ROOT = Path(__file__).resolve().parents[1] / "configs"


def load(group: str, name: str) -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / group)):
        cfg = compose(config_name=name, overrides=["output_dir=null"])
    return OmegaConf.to_container(cfg, resolve=True)


def names(group: str) -> list[str]:
    return sorted(p.stem for p in (ROOT / group).glob("*.yaml") if p.stem != "base")


@pytest.mark.parametrize("name", names("distributed"))
def test_distributed_configs(name):
    options = load("distributed", name)
    options.pop("device")
    world_size = options.pop("world_size")
    config = RunConfig(**options)
    large = name in {"gpsl_larger_batch", "gpsl_data_parallel"}
    assert config.learning_rate == (0.02 if large else 0.01)
    assert world_size == (1 if name in {"gpsl_single_worker", "gpsl_larger_batch"} else 4)
    assert config.sequential == name.endswith("_sequential")


@pytest.mark.parametrize("name", names("schedule_accounted"))
def test_schedule_accounted_configs(name):
    config = ScheduleConfig(**load("schedule_accounted", name))
    assert (config.clusters is None) == name.endswith("gpsl")
    assert f"b{config.batch_size}_" in name and name.startswith(config.dataset)
    assert config.learning_rate == (0.04 if config.batch_size >= 256 else 0.01)


def test_the_paper_matrix_is_complete():
    assert len(names("distributed")) == 9
    assert len(names("schedule_accounted")) == 21
