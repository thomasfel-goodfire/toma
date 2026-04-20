import numpy as np
import pytest
from toma.configs import CONFIGS, SMALL, MEDIUM, LARGE, GIANT, BenchmarkConfig
from toma.dgp import Concept


@pytest.mark.parametrize("cfg", [SMALL, MEDIUM, LARGE, GIANT])
def test_config_dims(cfg):
    assert cfg.d in (128, 256, 512, 1024)


@pytest.mark.parametrize("cfg", [SMALL, MEDIUM, LARGE, GIANT])
def test_concept_fraction(cfg):
    n_concepts = sum(1 for m in cfg.manifolds if isinstance(m, Concept))
    frac = n_concepts / cfg.n_components
    assert 0.25 <= frac <= 0.35


@pytest.mark.parametrize("cfg", [SMALL, MEDIUM, LARGE, GIANT])
def test_build_dgp_and_sample(cfg):
    np.random.seed(0)
    dgp = cfg.build_dgp()
    data = dgp.sample(n=10, l0=cfg.l0)
    assert data["x"].shape == (10, cfg.d)
    assert data["contributions"].shape == (10, cfg.n_components, cfg.d)
    assert data["active_masks"].shape == (10, cfg.n_components)


def test_configs_dict_keys():
    assert set(CONFIGS.keys()) == {"small", "medium", "large", "giant"}


def test_monotone_scale():
    configs = [SMALL, MEDIUM, LARGE, GIANT]
    for a, b in zip(configs, configs[1:]):
        assert b.d > a.d
        assert b.n_components > a.n_components
        assert b.sae_width > a.sae_width
        assert b.l0 >= a.l0
