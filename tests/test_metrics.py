import numpy as np
import pytest
from toma.dgp import DGP, Circle, Sphere, Helix
from toma.metrics import oracle_k_r2, component_f1


@pytest.fixture
def setup():
    np.random.seed(0)
    dgp = DGP([Circle(), Sphere(), Helix()], d=32)
    np.random.seed(1)
    data = dgp.sample(n=500, l0=2)
    return dgp, data


def test_oracle_k_r2_shape(setup):
    dgp, data = setup
    np.random.seed(2)
    codes = np.abs(np.random.rand(500, 64).astype(np.float32))
    dictionary = np.random.randn(64, 32).astype(np.float32)
    r2 = oracle_k_r2(codes, dictionary, data["contributions"], data["active_masks"], dgp.components)
    assert r2.shape == (3,)


def test_oracle_k_r2_at_most_one(setup):
    dgp, data = setup
    np.random.seed(2)
    codes = np.abs(np.random.rand(500, 64).astype(np.float32))
    dictionary = np.random.randn(64, 32).astype(np.float32)
    r2 = oracle_k_r2(codes, dictionary, data["contributions"], data["active_masks"], dgp.components)
    assert np.all(r2[~np.isnan(r2)] <= 1.0 + 1e-6)


def test_component_f1_shape(setup):
    dgp, data = setup
    np.random.seed(3)
    codes = (np.random.rand(500, 64) > 0.9).astype(np.float32)
    f1, best = component_f1(codes, data["active_masks"], dgp.components)
    assert f1.shape == (3,)
    assert best.shape == (3,)


def test_component_f1_in_unit_interval(setup):
    dgp, data = setup
    np.random.seed(3)
    codes = (np.random.rand(500, 64) > 0.9).astype(np.float32)
    f1, _ = component_f1(codes, data["active_masks"], dgp.components)
    assert np.all(f1 >= 0) and np.all(f1 <= 1.0 + 1e-6)


def test_perfect_recovery():
    """oracle_r2 ≈ 1 when the SAE perfectly encodes the manifold subspace."""
    np.random.seed(0)
    dgp = DGP([Circle()], d=16, sigma_bias=0.0)
    np.random.seed(1)
    data = dgp.sample(n=2000, l0=1, noise=0.0)
    comp = dgp.components[0]

    # dictionary = embedding rows U  (shape: ambient_dim × d)
    dictionary = comp.U
    mask = data["active_masks"][:, 0]
    mi = data["contributions"][mask, 0]  # (n_act, 16)

    # since mᵢ = z @ U (no bias), z = mᵢ @ Uᵀ  (U orthonormal → UUᵀ = I)
    z_local = mi @ comp.U.T
    codes = np.zeros((2000, comp.ambient_dim), dtype=np.float32)
    codes[mask] = z_local

    r2 = oracle_k_r2(codes, dictionary, data["contributions"], data["active_masks"], dgp.components)
    assert r2[0] > 0.99
