"""Dimension-aware OLS power-law fitter and LOSO/LODO/OOD validators for e-GMM."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np


@dataclass
class DimensionScalingObservation:
    """Single empirical sweep optimum for (n, k, d, B, T)."""

    num_train: float        # n
    num_components: float   # k
    dim: float              # d
    batch_size: float       # B
    epochs: float           # T
    alpha_0_opt: float      # empirically best peak cosine step
    nll_opt: float = 0.0
    b_knee_opt: Optional[float] = None


@dataclass
class DimensionAlphaLawFit:
    """Fitted power law: alpha_0 = min(C * k^a_k * B^a_B * n^a_n * T^a_T * d^a_d, 1)."""

    c_alpha: float
    a_k: float
    a_b: float
    a_n: float
    a_t: float
    a_d: float
    r2_log: float
    include_dim: bool = True

    def predict(
        self,
        num_components: float,
        batch_size: float,
        num_train: float,
        epochs: float,
        dim: float,
    ) -> float:
        val = (
            self.c_alpha
            * (num_components ** self.a_k)
            * (batch_size ** self.a_b)
            * (num_train ** self.a_n)
            * (epochs ** self.a_t)
            * (dim ** self.a_d)
        )
        return float(np.clip(val, 1e-6, 1.0))


@dataclass
class DimensionBatchLawFit:
    """Fitted power law: B_knee = C_B * k^b_k * n^b_n * T^b_T * d^b_d."""

    c_b: float
    b_k: float
    b_n: float
    b_t: float
    b_d: float
    r2_log: float
    include_dim: bool = True

    def predict(
        self,
        num_components: float,
        num_train: float,
        epochs: float,
        dim: float,
        round_pow2: bool = True,
    ) -> int:
        val = (
            self.c_b
            * (num_components ** self.b_k)
            * (num_train ** self.b_n)
            * (epochs ** self.b_t)
            * (dim ** self.b_d)
        )
        val = float(np.clip(val, num_components, num_train))
        if round_pow2:
            pow2 = 2 ** round(float(np.log2(max(val, 1.0))))
            return int(np.clip(pow2, 1, int(num_train)))
        return int(round(val))


def fit_dimension_aware_alpha0(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
) -> DimensionAlphaLawFit:
    """Fit log-linear OLS regression predicting alpha_0 from (k, B, n, T[, d])."""
    min_obs = 6 if include_dim else 5
    if len(observations) < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {len(observations)}.")

    rows = []
    for o in observations:
        row = [
            1.0,
            np.log(o.num_components),
            np.log(o.batch_size),
            np.log(o.num_train),
            np.log(o.epochs),
        ]
        if include_dim:
            row.append(np.log(o.dim))
        rows.append(row)

    X = np.array(rows, dtype=np.float64)
    y = np.array([np.log(o.alpha_0_opt) for o in observations], dtype=np.float64)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

    return DimensionAlphaLawFit(
        c_alpha=float(np.exp(beta[0])),
        a_k=float(beta[1]),
        a_b=float(beta[2]),
        a_n=float(beta[3]),
        a_t=float(beta[4]),
        a_d=float(beta[5]) if include_dim else 0.0,
        r2_log=float(r2),
        include_dim=include_dim,
    )


def fit_dimension_aware_batch_knee(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
) -> DimensionBatchLawFit:
    """Fit log-linear OLS regression predicting knee batch size B from (k, n, T[, d])."""
    valid = [o for o in observations if o.b_knee_opt is not None and o.b_knee_opt > 0]
    min_obs = 5 if include_dim else 4
    if len(valid) < min_obs:
        raise ValueError(f"Need at least {min_obs} observations with b_knee_opt, got {len(valid)}.")

    rows = []
    for o in valid:
        row = [1.0, np.log(o.num_components), np.log(o.num_train), np.log(o.epochs)]
        if include_dim:
            row.append(np.log(o.dim))
        rows.append(row)

    X = np.array(rows, dtype=np.float64)
    y = np.array([np.log(float(o.b_knee_opt)) for o in valid], dtype=np.float64)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

    return DimensionBatchLawFit(
        c_b=float(np.exp(beta[0])),
        b_k=float(beta[1]),
        b_n=float(beta[2]),
        b_t=float(beta[3]),
        b_d=float(beta[4]) if include_dim else 0.0,
        r2_log=float(r2),
        include_dim=include_dim,
    )


def _eval_leave_one_attr_out(
    observations: Sequence[DimensionScalingObservation],
    attr_name: str,
    include_dim: bool = True,
) -> Dict[str, float]:
    unique_vals = sorted({getattr(o, attr_name) for o in observations})
    log_sq_errors: List[float] = []
    rel_errors: List[float] = []

    min_train = 6 if include_dim else 5
    for held_val in unique_vals:
        train_obs = [o for o in observations if getattr(o, attr_name) != held_val]
        test_obs = [o for o in observations if getattr(o, attr_name) == held_val]
        if len(train_obs) < min_train or not test_obs:
            continue
        fit = fit_dimension_aware_alpha0(train_obs, include_dim=include_dim)
        for o in test_obs:
            pred = fit.predict(o.num_components, o.batch_size, o.num_train, o.epochs, o.dim)
            log_sq_errors.append(float((np.log(pred) - np.log(o.alpha_0_opt)) ** 2))
            rel_errors.append(float(abs(pred - o.alpha_0_opt) / o.alpha_0_opt))

    return {
        "log_rmse": float(np.sqrt(np.mean(log_sq_errors))) if log_sq_errors else float("nan"),
        "mean_rel_err": float(np.mean(rel_errors)) if rel_errors else float("nan"),
        "n_folds": float(len(unique_vals)),
    }


def evaluate_loso_n(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
) -> Dict[str, float]:
    """Leave-One-Size-Out (LOSO) log-RMSE validation across dataset sizes n."""
    res = _eval_leave_one_attr_out(observations, "num_train", include_dim=include_dim)
    return {
        "loso_log_rmse": res["log_rmse"],
        "loso_mean_rel_err": res["mean_rel_err"],
        "n_folds": res["n_folds"],
    }


def evaluate_lodo_d(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
) -> Dict[str, float]:
    """Leave-One-Dimension-Out (LODO) log-RMSE validation across ambient dimensions d."""
    res = _eval_leave_one_attr_out(observations, "dim", include_dim=include_dim)
    return {
        "lodo_log_rmse": res["log_rmse"],
        "lodo_mean_rel_err": res["mean_rel_err"],
        "n_folds": res["n_folds"],
    }


def compute_nll_regret(
    nll_law: float,
    nll_oracle: float,
    nll_reference: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute both absolute NLL gap (nats) and relative NLL regret.
    If nll_reference (e.g., 1-component Gaussian NLL) is given, normalizes
    by |nll_reference - nll_oracle| so regret stays well-defined even when NLL < 0.
    """
    gap = float(nll_law - nll_oracle)
    if nll_reference is not None:
        denom = max(abs(float(nll_reference) - float(nll_oracle)), 1e-8)
    else:
        denom = max(abs(float(nll_oracle)), 1e-8)
    return {
        "nll_gap_nats": gap,
        "nll_rel_regret": float(gap / denom),
    }
