import numpy as np
from typing import List, Tuple
from .dgp import Component


def oracle_k_r2(
    codes: np.ndarray,
    dictionary: np.ndarray,
    contributions: np.ndarray,
    active_masks: np.ndarray,
    components: List[Component],
) -> np.ndarray:
    """
    for each active component i with known ambient dim k_i = embedding_dim,
    reconstruct using only its top k_i SAE atoms, measure r² vs true m_i.

    codes:         (N, D_dict)  sae activations (sparse, top-K non-zero)
    dictionary:    (D_dict, d)  decoder weights
    contributions: (N, M, d)   true m_i per component
    active_masks:  (N, M)      bool

    returns r² per component (M,), nan if component never active
    """
    M = len(components)
    r2 = np.full(M, np.nan)
    for comp in components:
        i = comp.idx
        k = comp.ambient_dim
        mask = active_masks[:, i]
        if mask.sum() < 10:
            continue
        c = codes[mask]           # (n_act, D_dict)
        m = contributions[mask, i]  # (n_act, d)
        n_act = c.shape[0]

        top_idx = np.argsort(c, axis=1)[:, -k:]          # (n_act, k)
        c_topk  = c[np.arange(n_act)[:, None], top_idx]  # (n_act, k)
        W_topk  = dictionary[top_idx]                     # (n_act, k, d)
        x_hat   = np.einsum("nk,nkd->nd", c_topk, W_topk)

        ss_res = ((m - x_hat) ** 2).sum()
        ss_tot = ((m - m.mean(0)) ** 2).sum()
        if ss_tot < 1e-10:
            continue
        r2[i] = 1.0 - ss_res / ss_tot

    return r2


def component_f1(
    codes: np.ndarray,
    active_masks: np.ndarray,
    components: List[Component],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    for each component i, find the SAE atom whose firing best predicts when i is active.
    a loose unsupervised recovery signal: no ground truth used at training time.

    codes:        (N, D_dict)  sae activations
    active_masks: (N, M)       bool

    returns (f1_scores (M,), best_atom_idx (M,))
    """
    M = len(components)
    atom_fired = codes > 0  # (N, D_dict)
    f1_scores  = np.zeros(M)
    best_atoms = np.zeros(M, dtype=int)

    for comp in components:
        i = comp.idx
        y = active_masks[:, i]  # (N,) bool ground truth
        if not y.any():
            continue
        tp = (atom_fired &  y[:, None]).sum(0).astype(float)
        fp = (atom_fired & ~y[:, None]).sum(0).astype(float)
        fn = (~atom_fired & y[:, None]).sum(0).astype(float)
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        best = int(f1.argmax())
        f1_scores[i]  = f1[best]
        best_atoms[i] = best

    return f1_scores, best_atoms
