"""Benchmark configurations for the toy model of manifolds.

Four scales — small, medium, large, giant — each increasing the number of
manifold components, ambient dimension, and SAE width.  About 30 % of
components are 1-D line-segment ``Concept`` atoms (classic dictionary
learning); the rest are higher-dimensional manifolds drawn from the full zoo.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .dgp import (
    DGP,
    Circle, Sphere, Torus, Mobius, SwissRoll,
    Helix, FlatDisk, Concept, Manifold,
)


@dataclass
class BenchmarkConfig:
    """Parameters for a single benchmark scale.

    Attributes
    ----------
    name : str
        Human-readable label (``"small"``, ``"medium"``, …).
    d : int
        Ambient dimension of the superposition space.
    manifolds : list of Manifold
        Ordered list of manifold instances (concepts last).
    n_train : int
        Number of training samples.
    n_test : int
        Number of held-out test samples.
    l0 : int
        Number of active components per sample.
    noise : float
        Gaussian noise std added to ``x``.
    sae_width : int
        Number of SAE dictionary atoms.
    sae_k : int
        TopK sparsity of the SAE.
    """

    name: str
    d: int
    manifolds: List[Manifold]
    n_train: int
    n_test: int
    l0: int
    noise: float
    sae_width: int
    sae_k: int

    @property
    def n_components(self) -> int:
        return len(self.manifolds)

    def build_dgp(self, sigma_bias: float = 0.01) -> DGP:
        """Instantiate and return a :class:`~toma.dgp.DGP` for this config.

        Parameters
        ----------
        sigma_bias : float
            Std of per-component bias offset. Default 0.01.

        Returns
        -------
        DGP
        """
        return DGP(self.manifolds, d=self.d, sigma_bias=sigma_bias)


def _manifold_pool(n_manifolds: int, seed: int = 0) -> List[Manifold]:
    """Return a list of *n_manifolds* manifolds, ~70 % zoo + ~30 % Concept.

    Parameters
    ----------
    n_manifolds : int
        Total number of components.
    seed : int
        NumPy seed used only for choosing the manifold order.

    Returns
    -------
    list of Manifold
    """
    zoo = [Circle(), Sphere(), Torus(), Mobius(), SwissRoll(), Helix(), FlatDisk()]
    n_concepts = max(1, round(0.3 * n_manifolds))
    n_zoo = n_manifolds - n_concepts

    rng = np.random.default_rng(seed)
    chosen = [zoo[i % len(zoo)] for i in rng.permutation(n_zoo)]
    concepts = [Concept() for _ in range(n_concepts)]
    return chosen + concepts


SMALL = BenchmarkConfig(
    name="small",
    d=128,
    manifolds=_manifold_pool(10),
    n_train=20_000,
    n_test=5_000,
    l0=2,
    noise=0.01,
    sae_width=256,
    sae_k=4,
)

MEDIUM = BenchmarkConfig(
    name="medium",
    d=256,
    manifolds=_manifold_pool(20),
    n_train=50_000,
    n_test=10_000,
    l0=3,
    noise=0.01,
    sae_width=512,
    sae_k=6,
)

LARGE = BenchmarkConfig(
    name="large",
    d=512,
    manifolds=_manifold_pool(40),
    n_train=100_000,
    n_test=20_000,
    l0=4,
    noise=0.01,
    sae_width=1024,
    sae_k=8,
)

GIANT = BenchmarkConfig(
    name="giant",
    d=1024,
    manifolds=_manifold_pool(80),
    n_train=200_000,
    n_test=40_000,
    l0=5,
    noise=0.01,
    sae_width=2048,
    sae_k=10,
)

CONFIGS = {c.name: c for c in [SMALL, MEDIUM, LARGE, GIANT]}
