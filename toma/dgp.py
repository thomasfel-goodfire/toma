import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


class Manifold:
    """Base class for parametric manifold samplers.

    Subclasses must define ``ambient_dim`` and ``intrinsic_dim`` as class
    (or instance) attributes and implement :meth:`sample`.
    """

    ambient_dim: int
    intrinsic_dim: int

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample *n* points uniformly from the manifold.

        Parameters
        ----------
        n : int
            Number of points.

        Returns
        -------
        z : ndarray of shape (n, ambient_dim)
            Manifold coordinates.
        theta : ndarray of shape (n, intrinsic_dim)
            Intrinsic parameters (useful for coloring).
        """
        raise NotImplementedError

    def _calibrate(self, n: int = 50_000) -> Tuple[np.ndarray, float]:
        """Estimate center and unit-RMS scale by sampling the manifold.

        Parameters
        ----------
        n : int
            Number of samples used for estimation.

        Returns
        -------
        center : ndarray of shape (ambient_dim,)
        scale : float
            Reciprocal of RMS radius after centering.
        """
        z, _ = self.sample(n)
        center = z.mean(0)
        rms = np.sqrt(((z - center) ** 2).sum(1).mean())
        return center, 1.0 / max(rms, 1e-12)


class Circle(Manifold):
    """Circle in ℝ²."""

    ambient_dim = 2
    intrinsic_dim = 1

    def __init__(self, r: float = 1.0):
        self.r = r

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.random.uniform(0, 2 * np.pi, n)
        return np.stack([self.r * np.cos(t), self.r * np.sin(t)], -1), t[:, None]


class Sphere(Manifold):
    """2-sphere in ℝ³."""

    ambient_dim = 3
    intrinsic_dim = 2

    def __init__(self, r: float = 1.0):
        self.r = r

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        phi = np.arccos(np.random.uniform(-1, 1, n))
        t = np.random.uniform(0, 2 * np.pi, n)
        z = np.stack([
            self.r * np.sin(phi) * np.cos(t),
            self.r * np.sin(phi) * np.sin(t),
            self.r * np.cos(phi),
        ], -1)
        return z, np.stack([phi, t], -1)


class Hypersphere(Manifold):
    """n-sphere in ℝⁿ⁺¹ (generic dimension).

    Parameters
    ----------
    n : int
        Sphere dimension (2 = ordinary sphere, 3 = glome, …).
    r : float
        Radius.
    """

    def __init__(self, n: int = 3, r: float = 1.0):
        self.n = n
        self.r = r
        self.ambient_dim = n + 1
        self.intrinsic_dim = n

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        z = np.random.randn(n_samples, self.n + 1)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z *= self.r
        return z, z[:, : self.n]


class Torus(Manifold):
    """Torus in ℝ³."""

    ambient_dim = 3
    intrinsic_dim = 2

    def __init__(self, R: float = 2.0, r: float = 0.5):
        self.R = R
        self.r = r

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.random.uniform(0, 2 * np.pi, n)
        s = np.random.uniform(0, 2 * np.pi, n)
        z = np.stack([
            (self.R + self.r * np.cos(s)) * np.cos(t),
            (self.R + self.r * np.cos(s)) * np.sin(t),
            self.r * np.sin(s),
        ], -1)
        return z, np.stack([t, s], -1)


class Mobius(Manifold):
    """Möbius band in ℝ³."""

    ambient_dim = 3
    intrinsic_dim = 2

    def __init__(self, w: float = 0.5):
        self.w = w

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        phi = np.random.uniform(0, 2 * np.pi, n)
        t = np.random.uniform(-self.w / 2, self.w / 2, n)
        z = np.stack([
            (1 + t * np.cos(phi / 2)) * np.cos(phi),
            (1 + t * np.cos(phi / 2)) * np.sin(phi),
            t * np.sin(phi / 2),
        ], -1)
        return z, np.stack([phi, t], -1)


class SwissRoll(Manifold):
    """Swiss roll in ℝ³.

    Samples *t* proportional to arc length (density ∝ t) via inverse-CDF,
    avoiding the density pile-up near the center that uniform-in-t produces.
    """

    ambient_dim = 3
    intrinsic_dim = 2

    def __init__(self, theta_max: float = 3.0 * np.pi, h_max: float = 3.0):
        self.theta_max = theta_max
        self.h_max = h_max

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t_min = 1.5 * np.pi
        u = np.random.uniform(0, 1, n)
        t = np.sqrt(u * (self.theta_max ** 2 - t_min ** 2) + t_min ** 2)
        h = np.random.uniform(0, self.h_max, n)
        return np.stack([t * np.cos(t), h, t * np.sin(t)], -1), np.stack([t, h], -1)


class Helix(Manifold):
    """Helix in ℝ³."""

    ambient_dim = 3
    intrinsic_dim = 1

    def __init__(self, r: float = 1.0, alpha: float = 0.3, n_turns: int = 3):
        self.r = r
        self.alpha = alpha
        self.n_turns = n_turns

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.random.uniform(0, 2 * np.pi * self.n_turns, n)
        z = np.stack([self.r * np.cos(t), self.r * np.sin(t), self.alpha * t], -1)
        return z, t[:, None]


class FlatDisk(Manifold):
    """Flat disk in ℝ²."""

    ambient_dim = 2
    intrinsic_dim = 2

    def __init__(self, R: float = 1.0):
        self.R = R

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        r = self.R * np.sqrt(np.random.uniform(0, 1, n))
        t = np.random.uniform(0, 2 * np.pi, n)
        return np.stack([r * np.cos(t), r * np.sin(t)], -1), np.stack([r, t], -1)


class LineSegment(Manifold):
    """1D line segment in ℝ¹."""

    ambient_dim = 1
    intrinsic_dim = 1

    def __init__(self, length: float = 1.0):
        self.length = length

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.random.uniform(0, self.length, n)
        return t[:, None], t[:, None]


class Concept(LineSegment):
    """Single-direction concept — classic dictionary-learning atom."""
    pass


@dataclass
class Component:
    """A manifold instance embedded in ℝᵈ via a random orthonormal map.

    Attributes
    ----------
    idx : int
        Position in the parent DGP component list.
    manifold : Manifold
        Underlying manifold.
    U : ndarray of shape (ambient_dim, d)
        Orthonormal embedding rows.
    b : ndarray of shape (d,)
        Random bias in ambient space.
    center : ndarray
        Manifold centroid used for zero-centering.
    scale : float
        Isotropic scale for unit-RMS normalization.
    """

    idx: int
    manifold: Manifold
    U: np.ndarray
    b: np.ndarray
    center: np.ndarray
    scale: float

    @property
    def ambient_dim(self) -> int:
        return self.manifold.ambient_dim

    @property
    def intrinsic_dim(self) -> int:
        return self.manifold.intrinsic_dim

    @property
    def mtype(self) -> str:
        return type(self.manifold).__name__

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample *n* contributions in ambient space.

        Parameters
        ----------
        n : int
            Number of points.

        Returns
        -------
        contribution : ndarray of shape (n, d)
            mᵢ = z_norm @ U + b.
        z_local : ndarray of shape (n, ambient_dim)
            Normalized local coordinates.
        theta : ndarray of shape (n, intrinsic_dim)
            Intrinsic parameters.
        """
        z_raw, theta = self.manifold.sample(n)
        z = (z_raw - self.center) * self.scale
        return (z @ self.U + self.b).astype(np.float32), z.astype(np.float32), theta


class DGP:
    """Sparse mixture of embedded manifolds: x = Σᵢ mᵢ + ε.

    Set ``np.random.seed`` externally before construction and sampling
    for reproducibility.

    Parameters
    ----------
    manifolds : list of Manifold
        Manifold instances to embed in ℝᵈ.
    d : int
        Ambient dimension.
    sigma_bias : float, optional
        Std of random per-component bias offsets. Default 0.0.
    """

    def __init__(
        self,
        manifolds: List[Manifold],
        d: int,
        sigma_bias: float = 0.0,
    ):
        self.d = d
        self.components: List[Component] = self._build(manifolds, d, sigma_bias)

    def _build(self, manifolds, d, sigma_bias):
        components = []
        for i, manifold in enumerate(manifolds):
            k = manifold.ambient_dim
            center, scale = manifold._calibrate()
            G = np.random.randn(d, k)
            U = np.linalg.qr(G)[0].T.astype(np.float32)
            b = (np.random.randn(d) * sigma_bias).astype(np.float32)
            components.append(Component(i, manifold, U, b, center, scale))
        return components

    def sample(
        self,
        n: int,
        l0: int,
        noise: float = 0.01,
    ) -> dict:
        """Generate *n* samples from the sparse mixture.

        Set ``np.random.seed`` externally for reproducibility.

        Parameters
        ----------
        n : int
            Number of samples.
        l0 : int
            Number of active components per sample.
        noise : float, optional
            Gaussian noise std added to x. Default 0.01.

        Returns
        -------
        dict
            x : ndarray of shape (n, d)
                Superposition x = Σᵢ mᵢ + ε.
            contributions : ndarray of shape (n, M, d)
                Per-component mᵢ; zero where inactive.
            active_masks : ndarray of shape (n, M)
                True where component i is active.
            thetas : list of ndarray
                Intrinsic coordinates; ``thetas[i]`` has shape
                ``(n_active_i, intrinsic_dim_i)``.
        """
        M = len(self.components)

        priorities = np.random.random((n, M))
        active_sets = np.argsort(priorities, axis=1)[:, :l0]

        active_masks = np.zeros((n, M), dtype=bool)
        for j in range(l0):
            active_masks[np.arange(n), active_sets[:, j]] = True

        contributions = np.zeros((n, M, self.d), dtype=np.float32)
        x = np.zeros((n, self.d), dtype=np.float32)
        thetas: List = [None] * M

        for comp in self.components:
            mask = active_masks[:, comp.idx]
            if not mask.any():
                thetas[comp.idx] = np.zeros((0, comp.intrinsic_dim), dtype=np.float32)
                continue
            contrib, _, theta = comp.sample(int(mask.sum()))
            contributions[mask, comp.idx] = contrib
            x[mask] += contrib
            thetas[comp.idx] = theta.astype(np.float32)

        if noise > 0:
            x += (np.random.randn(n, self.d) * noise).astype(np.float32)

        return {
            "x": x,
            "contributions": contributions,
            "active_masks": active_masks,
            "thetas": thetas,
        }
