"""Synthetic benchmark distributions (Paper Appendix D.2) & offline memmap readers."""

import pathlib
from typing import Iterator, Optional, Tuple
import numpy as np


def generate_stretched_gaussian(
    n: int,
    d: int,
    exponent: float = 1.0,
    rotate: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Anisotropic stretched Gaussian with power-law decaying eigenvalues (Paper App. D.2):
        lambda_i = d * i^(-a) / sum_{j=1}^d j^(-a),  a in {0.5, 1.0, 2.0}
    Normalized to unit average variance per coordinate: (1/d) sum Var(x_i) = 1.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(1, d + 1, dtype=np.float64)
    raw = idx ** (-exponent)
    lambdas = d * raw / np.sum(raw)
    z = rng.normal(size=(n, d))
    X = z * np.sqrt(lambdas)[None, :]
    if rotate:
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X = X @ q.T
    return X.astype(np.float32), lambdas


def generate_gaussian_mixture(
    n: int,
    d: int,
    num_true_clusters: int = 16,
    zipf: bool = False,
    zipf_power: float = 1.0,
    anisotropic: bool = False,
    rotated: bool = False,
    sigma_mu_sq: float = 2.0 / 3.0,
    sigma_w_sq: float = 1.0 / 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Gaussian Mixture (Uniform or Zipf-imbalanced, isotropic or
    anisotropic/correlated) following Paper Appendix D.2.

    Returns:
        X: (n, d) float32 points
        labels: (n,) int64 ground-truth component IDs
        true_weights: (num_true_clusters,) mixture weights
    """
    rng = np.random.default_rng(seed)
    G = num_true_clusters
    means = rng.normal(scale=np.sqrt(sigma_mu_sq), size=(G, d))

    if zipf:
        ranks = np.arange(1, G + 1, dtype=np.float64)
        weights = ranks ** (-zipf_power)
        weights /= weights.sum()
    else:
        weights = np.full(G, 1.0 / G, dtype=np.float64)

    labels = rng.choice(G, size=n, p=weights)
    z = rng.normal(size=(n, d))

    if not anisotropic:
        X = means[labels] + np.sqrt(sigma_w_sq) * z
    else:
        # Per-cluster stretched diagonal scales (and optional random orthogonal rotation)
        X = np.empty((n, d), dtype=np.float64)
        base_idx = np.arange(1, d + 1, dtype=np.float64)
        for g in range(G):
            mask = labels == g
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            exp_g = 0.6 + 1.4 * (g / max(G - 1, 1))
            lam_g = d * (base_idx ** (-exp_g)) / np.sum(base_idx ** (-exp_g))
            scale_g = np.sqrt(sigma_w_sq * lam_g)
            # Permute coordinate axes across clusters so different clusters elongate differently
            perm = rng.permutation(d)
            local = z[mask] * scale_g[perm][None, :]
            if rotated:
                q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                local = local @ q.T
            X[mask] = means[g][None, :] + local

    return X.astype(np.float32), labels, weights


def generate_isotropic_gaussian(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def generate_uniform_cube(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Uniform(-sqrt(3), sqrt(3))^d normalized to unit variance per coordinate."""
    rng = np.random.default_rng(seed)
    bound = np.sqrt(3.0)
    return rng.uniform(-bound, bound, size=(n, d)).astype(np.float32)


def iter_minibatches(
    X: np.ndarray,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[np.ndarray]:
    """Yield mini-batches from X with optional fresh random permutation."""
    n = X.shape[0]
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n)
    else:
        perm = np.arange(n)

    limit = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, limit, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx]


def load_vector_memmap(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """
    Zero-download local reader for `.npy`, `.fvecs`, or `.bvecs` embedding files.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset file not found: {p}")

    ext = p.suffix.lower()
    if ext == ".npy":
        arr = np.load(p, mmap_mode="r")
        return arr[:max_rows] if max_rows is not None else arr

    if ext in (".fvecs", ".bvecs"):
        # Each record: 4-byte int32 dimension d followed by d float32 (.fvecs) or uint8 (.bvecs)
        dim_probe = np.fromfile(p, dtype=np.int32, count=1)
        if dim_probe.size == 0:
            raise ValueError(f"Empty vector file: {p}")
        d = int(dim_probe[0])
        if ext == ".fvecs":
            rec_bytes = 4 + 4 * d
            raw = np.memmap(p, dtype=np.uint8, mode="r")
            n_total = raw.size // rec_bytes
            n_use = min(n_total, max_rows) if max_rows is not None else n_total
            mat = np.memmap(p, dtype=np.float32, mode="r", shape=(n_use, d + 1))
            return mat[:, 1:]
        else:
            rec_bytes = 4 + d
            raw = np.memmap(p, dtype=np.uint8, mode="r")
            n_total = raw.size // rec_bytes
            n_use = min(n_total, max_rows) if max_rows is not None else n_total
            mat = np.memmap(p, dtype=np.uint8, mode="r", shape=(n_use, rec_bytes))
            return mat[:, 4:]

    raise ValueError(f"Unsupported vector file extension '{ext}'. Expected .npy, .fvecs, or .bvecs.")
