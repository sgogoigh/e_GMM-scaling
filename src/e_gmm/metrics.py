"""Exact Gaussian log-likelihood, NLL, BIC/AIC, and GMM health metrics (NumPy)."""

import math
from typing import Dict, Optional, Tuple
import numpy as np


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def logsumexp_1d(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        res = np.squeeze(res, axis=axis)
    return res


def gaussian_log_prob_diag(
    X: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log joint probabilities log(pi_j * N(x_i | mu_j, diag(var_j))) via
    the Flash-GMM 2d-augmented GEMM expansion.
    """
    X64 = np.asarray(X, dtype=np.float64)
    mu64 = np.asarray(mu, dtype=np.float64)
    var64 = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    pi64 = np.maximum(np.asarray(pi, dtype=np.float64), 1e-30)
    d = X64.shape[1]

    inv_var = 1.0 / var64  # (k, d)
    # b_j = log(pi_j) - 0.5 * sum_m (mu_jm^2 / var_jm + log(2*pi*var_jm))
    c_j = np.sum((mu64 ** 2) * inv_var + np.log(2.0 * math.pi * var64), axis=1)
    b_j = np.log(pi64) - 0.5 * c_j  # (k,)

    # Augmented GEMM: [x^2, x] @ [-0.5 / var, mu / var]^T + b_j
    X_aug = np.concatenate([X64 ** 2, X64], axis=1)             # (B, 2d)
    W_aug = np.concatenate([-0.5 * inv_var, mu64 * inv_var], axis=1)  # (k, 2d)
    log_joint = X_aug @ W_aug.T + b_j[None, :]
    lse = logsumexp_1d(log_joint, axis=1, keepdims=False)
    return log_joint, lse


def gaussian_log_prob_low_rank_diag(
    X: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    diag_var: np.ndarray,
    U: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact log joint probabilities for Sigma_j = diag(d_j) + U_j U_j^T using
    the Woodbury matrix identity and Matrix Determinant Lemma.

    Args:
        X: (B, d)
        pi: (k,)
        mu: (k, d)
        diag_var: (k, d) strictly positive diagonal component
        U: (k, d, r) low-rank factors
    """
    X64 = np.asarray(X, dtype=np.float64)
    mu64 = np.asarray(mu, dtype=np.float64)
    d_var = np.maximum(np.asarray(diag_var, dtype=np.float64), 1e-12)
    U64 = np.asarray(U, dtype=np.float64)
    pi64 = np.maximum(np.asarray(pi, dtype=np.float64), 1e-30)

    B, d = X64.shape
    k, _, r = U64.shape
    log_joint = np.empty((B, k), dtype=np.float64)
    const_term = d * math.log(2.0 * math.pi)
    eye_r = np.eye(r, dtype=np.float64)

    for j in range(k):
        inv_d = 1.0 / d_var[j]                          # (d,)
        Uj = U64[j]                                     # (d, r)
        # Capacitance C_j = I_r + U_j^T D_j^{-1} U_j
        Dinv_U = inv_d[:, None] * Uj                    # (d, r)
        C = eye_r + Uj.T @ Dinv_U                       # (r, r)
        C = 0.5 * (C + C.T)
        L_c = np.linalg.cholesky(C)                     # (r, r)
        logdet = np.sum(np.log(d_var[j])) + 2.0 * np.sum(np.log(np.diag(L_c)))

        diff = X64 - mu64[j][None, :]                   # (B, d)
        diag_quad = np.sum((diff ** 2) * inv_d[None, :], axis=1)
        # proj = U_j^T D_j^{-1} diff^T -> (r, B)
        proj = Dinv_U.T @ diff.T                        # (r, B)
        solved = np.linalg.solve(L_c, proj)             # (r, B)
        maha = diag_quad - np.sum(solved ** 2, axis=0)
        log_joint[:, j] = math.log(pi64[j]) - 0.5 * (const_term + logdet + maha)

    lse = logsumexp_1d(log_joint, axis=1, keepdims=False)
    return log_joint, lse


def gaussian_log_prob_full(
    X: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Exact log joint probabilities for full covariance matrices using Cholesky
    factorization with adaptive jitter fallback. Returns (log_joint, lse, chol_fallbacks).
    """
    X64 = np.asarray(X, dtype=np.float64)
    mu64 = np.asarray(mu, dtype=np.float64)
    cov64 = np.asarray(cov, dtype=np.float64)
    pi64 = np.maximum(np.asarray(pi, dtype=np.float64), 1e-30)

    B, d = X64.shape
    k = mu64.shape[0]
    log_joint = np.empty((B, k), dtype=np.float64)
    const_term = d * math.log(2.0 * math.pi)
    eye_d = np.eye(d, dtype=np.float64)
    chol_fallbacks = 0

    for j in range(k):
        Sj = 0.5 * (cov64[j] + cov64[j].T)
        L = None
        for jit in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                scale = max(float(np.trace(Sj)) / d, 1.0)
                L = np.linalg.cholesky(Sj + (jit * scale) * eye_d)
                if jit > 0.0:
                    chol_fallbacks += 1
                break
            except np.linalg.LinAlgError:
                continue
        if L is None:
            evals, evecs = np.linalg.eigh(Sj)
            evals = np.maximum(evals, 1e-6)
            L = np.linalg.cholesky((evecs * evals[None, :]) @ evecs.T)
            chol_fallbacks += 1

        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        diff = X64 - mu64[j][None, :]                   # (B, d)
        solved = np.linalg.solve(L, diff.T)             # (d, B)
        maha = np.sum(solved ** 2, axis=0)
        log_joint[:, j] = math.log(pi64[j]) - 0.5 * (const_term + logdet + maha)

    lse = logsumexp_1d(log_joint, axis=1, keepdims=False)
    return log_joint, lse, chol_fallbacks


def _effective_num_params(k: int, d: int, cov_type: str, rank: int = 0) -> int:
    # Weights (k-1) + means (k*d) + covariance parameters
    base = (k - 1) + k * d
    if cov_type == "diag":
        return base + k * d
    if cov_type == "low_rank_diag":
        return base + k * (d + d * rank - (rank * (rank - 1)) // 2)
    return base + k * (d * (d + 1) // 2)


def evaluate_gmm_nll(
    X: np.ndarray,
    pi,
    mu,
    cov_or_var,
    cov_type: str = "diag",
    U=None,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Evaluate exact Negative Log-Likelihood, total log-likelihood, BIC, and AIC."""
    X_np = _to_numpy(X)
    pi_np = _to_numpy(pi)
    mu_np = _to_numpy(mu)
    cov_np = _to_numpy(cov_or_var)
    U_np = _to_numpy(U) if U is not None else None

    n, d = X_np.shape
    k = mu_np.shape[0]
    total_ll = 0.0
    chol_fallbacks = 0

    for start in range(0, n, batch_size):
        xb = X_np[start : start + batch_size]
        if cov_type == "diag":
            _, lse = gaussian_log_prob_diag(xb, pi_np, mu_np, cov_np)
        elif cov_type == "low_rank_diag":
            if U_np is None:
                raise ValueError("U must be provided for cov_type='low_rank_diag'.")
            _, lse = gaussian_log_prob_low_rank_diag(xb, pi_np, mu_np, cov_np, U_np)
        elif cov_type == "full":
            _, lse, fb = gaussian_log_prob_full(xb, pi_np, mu_np, cov_np)
            chol_fallbacks += fb
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")
        total_ll += float(np.sum(lse))

    mean_ll = total_ll / max(n, 1)
    nll = -mean_ll
    rank = U_np.shape[-1] if U_np is not None else 0
    n_params = _effective_num_params(k, d, cov_type, rank=rank)
    bic = -2.0 * total_ll + n_params * math.log(max(n, 1))
    aic = -2.0 * total_ll + 2.0 * n_params
    return {
        "nll": float(nll),
        "nll_per_dim": float(nll / max(d, 1)),
        "total_log_likelihood": float(total_ll),
        "bic": float(bic),
        "aic": float(aic),
        "n_effective_params": float(n_params),
        "chol_fallbacks": float(chol_fallbacks),
    }


def compute_gmm_health(
    weights_or_N,
    cov_or_var=None,
    cov_type: str = "diag",
    cov_floor: float = 1e-4,
    dead_tau: float = 0.1,
    U=None,
) -> Dict[str, float]:
    """
    Compute mixture utilization, dead-component rate, entropy/Gini balance,
    and covariance conditioning diagnostics.
    """
    counts = np.maximum(_to_numpy(weights_or_N).ravel(), 0.0)
    k = counts.size
    total = float(counts.sum())
    mean_n = total / max(k, 1)

    if total > 1e-30:
        dead_count = int(np.sum(counts < (dead_tau * mean_n)))
        max_to_mean = float(counts.max() / max(mean_n, 1e-30))
        min_to_fair = float(counts.min() / max(mean_n, 1e-30))
        p = counts / total
        nz = p[p > 1e-15]
        norm_entropy = -float(np.sum(nz * np.log(nz))) / float(math.log(max(k, 2))) if k > 1 else 1.0
        sorted_c = np.sort(counts)
        idx = np.arange(1, k + 1, dtype=np.float64)
        gini = float((2.0 * np.sum(idx * sorted_c)) / (k * total) - (k + 1.0) / k) if k > 1 else 0.0
    else:
        dead_count = k
        max_to_mean = 0.0
        min_to_fair = 0.0
        norm_entropy = 0.0
        gini = 1.0

    floor_hit_rate = 0.0
    mean_trace = 0.0
    max_cond = 1.0
    mean_eff_rank = 0.0

    if cov_or_var is not None:
        cov_np = _to_numpy(cov_or_var)
        if cov_type == "diag":
            floor_hit_rate = float(np.mean(cov_np <= cov_floor * (1.0 + 1e-4)))
            mean_trace = float(np.mean(np.sum(cov_np, axis=1)))
            conds = np.max(cov_np, axis=1) / np.maximum(np.min(cov_np, axis=1), 1e-12)
            max_cond = float(np.max(conds))
            # Effective rank (participation ratio of eigenvalues)
            s_norm = cov_np / np.maximum(np.sum(cov_np, axis=1, keepdims=True), 1e-12)
            mean_eff_rank = float(np.mean(1.0 / np.maximum(np.sum(s_norm ** 2, axis=1), 1e-12)))
        elif cov_type == "low_rank_diag":
            U_np = _to_numpy(U) if U is not None else np.zeros((k, cov_np.shape[1], 1))
            diag_full = cov_np + np.sum(U_np ** 2, axis=2)
            floor_hit_rate = float(np.mean(cov_np <= cov_floor * (1.0 + 1e-4)))
            mean_trace = float(np.mean(np.sum(diag_full, axis=1)))
            max_cond = float(np.max(np.max(diag_full, axis=1) / np.maximum(np.min(cov_np, axis=1), 1e-12)))
            mean_eff_rank = float(cov_np.shape[1])
        else:
            evals = np.linalg.eigvalsh(0.5 * (cov_np + np.swapaxes(cov_np, -1, -2)))
            evals = np.maximum(evals, 1e-12)
            floor_hit_rate = float(np.mean(np.min(evals, axis=1) <= cov_floor * (1.0 + 1e-3)))
            mean_trace = float(np.mean(np.sum(evals, axis=1)))
            max_cond = float(np.max(np.max(evals, axis=1) / np.min(evals, axis=1)))
            s_norm = evals / np.maximum(np.sum(evals, axis=1, keepdims=True), 1e-12)
            mean_eff_rank = float(np.mean(1.0 / np.maximum(np.sum(s_norm ** 2, axis=1), 1e-12)))

    return {
        "dead_count": float(dead_count),
        "active_fraction": float(1.0 - dead_count / max(k, 1)),
        "max_to_mean": max_to_mean,
        "min_to_fair": min_to_fair,
        "norm_entropy": norm_entropy,
        "gini": max(0.0, gini),
        "floor_hit_rate": floor_hit_rate,
        "mean_trace": mean_trace,
        "max_condition_number": max_cond,
        "mean_effective_rank": mean_eff_rank,
    }
