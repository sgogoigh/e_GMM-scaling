"""Synthetic benchmark distributions (Paper Appendix D.2), strict local vector readers, and
the preprocessing used by the experiments (random splits, scale normalization, PCA, TwoNN).

Nothing in this module falls back silently: a missing or malformed file raises, and a requested
dimension larger than the data raises instead of being truncated or padded.
"""

import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
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
    """Back-compat alias of `load_vectors` (now validates headers and sizes)."""
    return load_vectors(path, max_rows=max_rows)


# ---------------------------------------------------------------------------------------------
# Strict local readers
# ---------------------------------------------------------------------------------------------

SUPPORTED_VECTOR_EXTENSIONS = (".npy", ".fvecs", ".bvecs", ".fbin", ".u8bin", ".i8bin")
_BIGANN_BIN_DTYPES = {".fbin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8}


def load_vectors(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """
    Memory-map a local embedding file and return an (n, d) view. Raises on a missing file,
    an unknown extension, or a file whose size does not match its header.

    Formats:
      .npy                    2-D array
      .fvecs / .bvecs         TEXMEX: per record an int32 dimension then d float32 / uint8 values
      .fbin / .u8bin / .i8bin big-ann-benchmarks: uint32 n, uint32 d header, then n*d values
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Local dataset file not found: {p}")
    ext = p.suffix.lower()
    size = p.stat().st_size

    if ext == ".npy":
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected a 2-D array, got shape {arr.shape}.")
        return arr[:max_rows] if max_rows is not None else arr

    if ext in (".fvecs", ".bvecs"):
        dim_probe = np.fromfile(p, dtype=np.int32, count=1)
        if dim_probe.size == 0:
            raise ValueError(f"Empty vector file: {p}")
        d = int(dim_probe[0])
        if d <= 0:
            raise ValueError(f"{p}: invalid record dimension {d}.")
        rec_bytes = 4 + (4 * d if ext == ".fvecs" else d)
        if size % rec_bytes != 0:
            raise ValueError(f"{p}: size {size} is not a multiple of the record size {rec_bytes} (d={d}).")
        n_total = size // rec_bytes
        n_use = min(n_total, max_rows) if max_rows is not None else n_total
        raw = np.memmap(p, dtype=np.uint8, mode="r", shape=(n_use, rec_bytes))
        n_check = min(n_use, 1000)
        heads = np.ascontiguousarray(raw[:n_check, :4]).view(np.int32).ravel()
        if np.any(heads != d):
            raise ValueError(f"{p}: records have inconsistent dimensions (expected {d} in every header).")
        if ext == ".fvecs":
            mat = np.memmap(p, dtype=np.float32, mode="r", shape=(n_use, d + 1))
            return mat[:, 1:]
        return raw[:, 4:]

    if ext in _BIGANN_BIN_DTYPES:
        dtype = np.dtype(_BIGANN_BIN_DTYPES[ext])
        header = np.fromfile(p, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"{p}: missing the 8-byte (n, d) header.")
        n_total, d = int(header[0]), int(header[1])
        expected = 8 + n_total * d * dtype.itemsize
        if size < expected:
            raise ValueError(
                f"{p}: header says {n_total}x{d} {dtype} ({expected} bytes) but the file has {size} bytes."
            )
        n_use = min(n_total, max_rows) if max_rows is not None else n_total
        return np.memmap(p, dtype=dtype, mode="r", offset=8, shape=(n_use, d))

    raise ValueError(f"Unsupported vector file extension '{ext}'. Expected one of {SUPPORTED_VECTOR_EXTENSIONS}.")


def find_vector_file(data_dir: str, stem: str) -> pathlib.Path:
    """Return <data_dir>/<stem><ext> for the first supported extension that exists, else raise."""
    base = pathlib.Path(data_dir)
    for ext in SUPPORTED_VECTOR_EXTENSIONS:
        cand = base / f"{stem}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No file named {stem} with an extension in {SUPPORTED_VECTOR_EXTENSIONS} under {base}. "
        "Real-data runs never fall back to synthetic data."
    )


# ---------------------------------------------------------------------------------------------
# Splits and preprocessing
# ---------------------------------------------------------------------------------------------


def _row_hashes(X: np.ndarray) -> np.ndarray:
    """64-bit FNV-1a-style hash of each row's bytes, vectorized over rows (memory O(n), not O(n*d))."""
    Xc = np.ascontiguousarray(X)
    n = Xc.shape[0]
    b = Xc.view(np.uint8).reshape(n, -1)
    pad = (-b.shape[1]) % 8
    if pad:
        b = np.pad(b, ((0, 0), (0, pad)))
    words = np.ascontiguousarray(b).view(np.uint64)
    h = np.full(n, 14695981039346656037, dtype=np.uint64)
    prime = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for j in range(words.shape[1]):
            h = (h ^ words[:, j]) * prime
    return h


def random_splits(
    arr: np.ndarray,
    sizes: Sequence[int],
    seed: int = 0,
    dedup_heldout: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Draw disjoint random row subsets of the requested sizes (the first is the training split).

    Rows are sampled uniformly without replacement from the whole file, not taken from its head.
    With dedup_heldout, held-out rows that are byte-identical to a training row are replaced by
    fresh rows, so test NLL is never measured on memorized points. Raises if the file has too few
    rows.
    """
    n_total = int(arr.shape[0])
    sizes = [int(s) for s in sizes]
    need = sum(sizes)
    if need > n_total:
        raise ValueError(f"Requested {need} rows ({sizes}) but the data has only {n_total}.")
    rng = np.random.default_rng(seed)
    pool = rng.permutation(n_total)
    cursor = 0
    out: List[np.ndarray] = []
    train_hashes = np.empty(0, dtype=np.uint64)
    dups_replaced = 0
    for si, size in enumerate(sizes):
        chosen: List[int] = []
        while len(chosen) < size:
            take = size - len(chosen)
            if cursor + take > n_total:
                raise ValueError(
                    f"Ran out of rows while replacing held-out duplicates of training rows "
                    f"(replaced {dups_replaced} so far)."
                )
            cand = np.sort(pool[cursor : cursor + take])
            cursor += take
            if si > 0 and dedup_heldout:
                block = np.asarray(arr[cand])
                # A hash collision can only drop a non-duplicate held-out row (it is then replaced).
                dup = np.isin(_row_hashes(block), train_hashes)
                keep = [int(i) for i in cand[~dup]]
                dups_replaced += take - len(keep)
                chosen.extend(keep)
            else:
                chosen.extend(int(i) for i in cand)
        idx = np.sort(np.asarray(chosen, dtype=np.int64))
        raw_part = np.asarray(arr[idx])
        if si == 0 and dedup_heldout and len(sizes) > 1:
            train_hashes = np.unique(_row_hashes(raw_part))
        out.append(raw_part.astype(np.float32))
    return out, {"n_total": float(n_total), "dups_replaced": float(dups_replaced)}


def normalize_scale(X_train: np.ndarray, *others: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Center on the training mean and divide by one global scalar so the mean per-coordinate training
    variance is 1. Relative anisotropy is preserved; only the overall scale changes, which makes an
    absolute covariance floor (lambda_0 = 1e-4) mean the same thing on every dataset.
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    mean = Xt.mean(axis=0)
    scale = float(np.sqrt(np.mean(Xt.var(axis=0))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training data has zero variance; cannot normalize.")
    outs = [((np.asarray(A, dtype=np.float64) - mean) / scale).astype(np.float32) for A in (X_train, *others)]
    return outs, {"scale": scale, "mean_norm": float(np.linalg.norm(mean))}


def pca_project(X_train: np.ndarray, out_dim: int, *others: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Project onto the top out_dim principal axes of the TRAINING data. Varying out_dim on one data
    source varies d without changing the source, which is what an exponent on d needs. Raises if
    out_dim exceeds the data dimension (never pads or truncates silently).
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    n, d = Xt.shape
    if not (1 <= out_dim <= d):
        raise ValueError(f"PCA output dimension {out_dim} must be in [1, {d}] (the data dimension).")
    mean = Xt.mean(axis=0)
    cov = np.cov(Xt - mean, rowvar=False).reshape(d, d)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    W = evecs[:, order[:out_dim]]
    outs = [((np.asarray(A, dtype=np.float64) - mean) @ W).astype(np.float32) for A in (X_train, *others)]
    total = float(evals.sum())
    return outs, {"explained_variance_ratio": float(evals[:out_dim].sum() / total) if total > 0 else float("nan")}


def estimate_intrinsic_dim_twonn(
    X: np.ndarray,
    max_points: int = 2000,
    discard_fraction: float = 0.1,
    seed: int = 0,
) -> float:
    """
    TwoNN intrinsic-dimension estimate (Facco et al., 2017, Sci. Rep. 7:12140).

    For each point mu = r2 / r1 (ratio of 2nd to 1st nearest-neighbour distance). Under local
    uniformity F(mu) = 1 - mu^(-d), so d is the slope of -log(1 - F) against log(mu) through the
    origin, fitted after dropping the largest discard_fraction of mu values.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], size=max_points, replace=False)]
    n = X.shape[0]
    if n < 10:
        raise ValueError("TwoNN needs at least 10 points.")
    sq = np.sum(X ** 2, axis=1)
    r1 = np.empty(n)
    r2 = np.empty(n)
    chunk = 512
    for start in range(0, n, chunk):
        xb = X[start : start + chunk]
        d2 = sq[start : start + chunk, None] + sq[None, :] - 2.0 * (xb @ X.T)
        rows = np.arange(xb.shape[0])
        d2[rows, start + rows] = np.inf
        part = np.partition(d2, 1, axis=1)[:, :2]
        part = np.sort(np.maximum(part, 0.0), axis=1)
        r1[start : start + chunk] = np.sqrt(part[:, 0])
        r2[start : start + chunk] = np.sqrt(part[:, 1])
    valid = (r1 > 0) & np.isfinite(r2)
    mu = np.sort(r2[valid] / r1[valid])
    m = mu.size
    if m < 10:
        raise ValueError("TwoNN: too many duplicate points to estimate a dimension.")
    keep = int(np.floor(m * (1.0 - discard_fraction)))
    F = np.arange(1, m + 1, dtype=np.float64) / m
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise ValueError("TwoNN: degenerate neighbour-distance ratios.")
    return float(np.dot(x, y) / denom)
