<p align="center">
  <img src="assets/logo.png" width="256" alt="toma logo">
</p>

# Toy model of manifolds (TOMA)

A minimal benchmark for studying how sparse autoencoders recover low-dimensional manifold structure from superposed representations.

**Data generating process**

```
x = Σᵢ mᵢ + ε
```

Each `mᵢ` is a manifold contribution: a point sampled from a low-dimensional shape (circle, sphere, torus, …) and embedded in ℝᵈ via a random orthonormal map. A fraction of components are 1-D `Concept` atoms (classic dictionary-learning). Only `l0` components are active per sample.

---

## install

```bash
git clone https://github.com/goodfire-ai/toma
cd toma
uv sync
```

---

## quick start

```python
import numpy as np
from toma.dgp import DGP, Circle, Sphere, Torus, Helix, Concept

np.random.seed(42)
dgp = DGP([Circle(), Sphere(), Torus(), Helix(), Concept()], d=64, sigma_bias=0.01)

np.random.seed(0)
data = dgp.sample(n=10_000, l0=3, noise=0.01)

x        = data["x"]             # (N, d)
contribs = data["contributions"] # (N, M, d)  — mᵢ per sample
masks    = data["active_masks"]  # (N, M) bool
thetas   = data["thetas"]        # list[M], intrinsic coords per component
```

See `starter.ipynb` for a full walkthrough: manifold zoo visualization, SAE training with [overcomplete](https://github.com/KempnerInstitute/overcomplete), and recovery metric plots.

---

## manifolds

| class | ambient dim | intrinsic dim |
|-------|-------------|---------------|
| `Circle` | 2 | 1 |
| `Sphere` | 3 | 2 |
| `Hypersphere(n)` | n+1 | n |
| `Torus` | 3 | 2 |
| `Mobius` | 3 | 2 |
| `SwissRoll` | 3 | 2 |
| `Helix` | 3 | 1 |
| `FlatDisk` | 2 | 2 |
| `LineSegment` / `Concept` | 1 | 1 |

---

## benchmark configs

Four pre-defined scales in `toma.configs`, each with ~30 % `Concept` atoms and the remainder drawn from the manifold zoo:

| config | d | components | n_train | SAE width | K |
|--------|---|-----------|---------|-----------|---|
| `SMALL` | 128 | 10 | 20k | 256 | 4 |
| `MEDIUM` | 256 | 20 | 50k | 512 | 6 |
| `LARGE` | 512 | 40 | 100k | 1024 | 8 |
| `GIANT` | 1024 | 80 | 200k | 2048 | 10 |

```python
from toma.configs import MEDIUM

np.random.seed(42)
dgp  = MEDIUM.build_dgp()
data = dgp.sample(n=MEDIUM.n_train, l0=MEDIUM.l0)
```

---

## recovery metrics

```python
from toma.metrics import oracle_k_r2, component_f1

# oracle R²: top kᵢ SAE atoms reconstruct mᵢ
r2 = oracle_k_r2(codes, dictionary, contribs, masks, dgp.components)

# best-atom F1: single atom predicting whether mᵢ is active
f1, best_atoms = component_f1(codes, masks, dgp.components)
```

**oracle R²** — for component *i* with known ambient dim *kᵢ*, take the *kᵢ* most active SAE atoms across all samples where *i* is active, reconstruct, and compute R² against the true contribution `mᵢ`.

**component F1** — for each component *i*, find the SAE atom whose binary on/off firing best predicts `active_masks[:, i]` by F1 score.

---

## tests

```bash
uv sync
uv run pytest tests/
```
