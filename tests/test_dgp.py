import numpy as np
import pytest
from toma.dgp import (
    DGP, Circle, Sphere, Hypersphere, Torus, Mobius, SwissRoll,
    Helix, FlatDisk, LineSegment, Concept,
)

ALL_MANIFOLDS = [
    (Circle,      2, 1),
    (Sphere,      3, 2),
    (Torus,       3, 2),
    (Mobius,      3, 2),
    (SwissRoll,   3, 2),
    (Helix,       3, 1),
    (FlatDisk,    2, 2),
    (LineSegment, 1, 1),
    (Concept,     1, 1),
]


@pytest.mark.parametrize("Cls,ambient_dim,intrinsic_dim", ALL_MANIFOLDS)
def test_manifold_sample_shapes(Cls, ambient_dim, intrinsic_dim):
    np.random.seed(0)
    z, theta = Cls().sample(100)
    assert z.shape == (100, ambient_dim)
    assert theta.shape == (100, intrinsic_dim)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_hypersphere_sample_shapes(n):
    np.random.seed(0)
    m = Hypersphere(n=n)
    z, theta = m.sample(100)
    assert z.shape == (100, n + 1)
    assert theta.shape == (100, n)


def test_hypersphere_on_surface():
    np.random.seed(0)
    m = Hypersphere(n=3, r=2.0)
    z, _ = m.sample(1000)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 2.0, atol=1e-10)


def test_swiss_roll_arc_length_uniform():
    """verify t is sampled ∝ arc-length (half the samples below arc-length median)."""
    np.random.seed(0)
    m = SwissRoll()
    _, theta = m.sample(10_000)
    t = theta[:, 0]
    t_min = 1.5 * np.pi
    t_med = np.sqrt((t_min ** 2 + m.theta_max ** 2) / 2)
    np.testing.assert_allclose((t < t_med).mean(), 0.5, atol=0.03)


@pytest.fixture
def dgp():
    np.random.seed(42)
    return DGP([Circle(), Sphere(), Torus(), Helix(), Concept()], d=32, sigma_bias=0.01)


def test_sample_shapes(dgp):
    np.random.seed(0)
    data = dgp.sample(n=100, l0=2)
    M = len(dgp.components)
    assert data["x"].shape == (100, 32)
    assert data["contributions"].shape == (100, M, 32)
    assert data["active_masks"].shape == (100, M)
    assert len(data["thetas"]) == M


@pytest.mark.parametrize("l0", [1, 2, 3])
def test_exactly_l0_active(dgp, l0):
    np.random.seed(l0)
    data = dgp.sample(n=200, l0=l0)
    assert (data["active_masks"].sum(1) == l0).all()


def test_superposition_identity(dgp):
    """x equals sum of contributions when noise=0."""
    np.random.seed(0)
    data = dgp.sample(n=200, l0=2, noise=0.0)
    np.testing.assert_allclose(data["x"], data["contributions"].sum(1), atol=1e-5)


def test_inactive_contributions_zero(dgp):
    np.random.seed(0)
    data = dgp.sample(n=200, l0=1)
    for i in range(len(dgp.components)):
        inactive = ~data["active_masks"][:, i]
        assert (data["contributions"][inactive, i] == 0).all()


def test_thetas_length_matches_active(dgp):
    np.random.seed(0)
    data = dgp.sample(n=300, l0=2)
    for i, comp in enumerate(dgp.components):
        n_active = int(data["active_masks"][:, i].sum())
        assert data["thetas"][i].shape == (n_active, comp.intrinsic_dim)


def test_reproducible_with_external_seed(dgp):
    np.random.seed(7)
    d1 = dgp.sample(n=50, l0=2)
    np.random.seed(7)
    d2 = dgp.sample(n=50, l0=2)
    np.testing.assert_array_equal(d1["x"], d2["x"])
    np.testing.assert_array_equal(d1["active_masks"], d2["active_masks"])
