"""Power-law fitting for e-GMM step-size and batch-size laws, with identifiability and censoring checks.

Model (log-linear OLS, as in e-means Eq. 6/7):
    ln alpha_0 = ln C + a_k ln k + a_B ln B + a_n ln n + a_T ln T [+ a_d ln d]
    ln B_knee  = ln C + b_k ln k + b_n ln n + b_T ln T [+ b_d ln d]

Changes after the 2026-11-25 review (docs/review/2026-11-25-REVIEW.md):
  * A regressor that does not vary in the sweep (e.g. T fixed at 3) is NOT fitted: its exponent
    is reported as None and listed in `unidentified`. Previously the min-norm least-squares
    solution split the intercept and produced a meaningless a_T.
  * Rank-deficient designs (collinear regressors) are rejected.
  * R^2 is None when the target has no variance (it used to report 1.0 from 0/0).
  * Censored observations (best alpha_0 or knee at the edge of the tested grid) are excluded
    from fits and counted.
  * Leave-one-out evaluation reports out-of-sample NLL regret (nats/dim) read off each held-out
    configuration's measured alpha curve, not only the error of the predicted alpha.
  * `dim_feature="intrinsic"` uses an intrinsic-dimension estimate instead of ambient d.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_VAR_TOL = 1e-10


@dataclass
class DimensionScalingObservation:
    """Single empirical sweep optimum for (n, k, d, B, T)."""

    num_train: float        # n
    num_components: float   # k
    dim: float              # ambient d
    batch_size: float       # B
    epochs: float           # T
    alpha_0_opt: float      # empirically best peak cosine step
    nll_opt: float = 0.0
    b_knee_opt: Optional[float] = None
    # Added 2026-11-25
    alpha_censored: bool = False          # best alpha at the edge of the tested ladder
    knee_censored: bool = False           # knee at the edge of the tested batch range
    dim_intrinsic: Optional[float] = None  # e.g. TwoNN estimate on the training split
    alpha_curve: Optional[Tuple[List[float], List[float]]] = None  # (alphas, mean val NLL)
    source: str = ""                      # dataset family / file tag


def _dim_value(o: DimensionScalingObservation, dim_feature: str) -> float:
    if dim_feature == "ambient":
        return float(o.dim)
    if dim_feature == "intrinsic":
        if o.dim_intrinsic is None or not np.isfinite(o.dim_intrinsic) or o.dim_intrinsic <= 0:
            raise ValueError("dim_feature='intrinsic' requires dim_intrinsic on every observation.")
        return float(o.dim_intrinsic)
    raise ValueError(f"Unknown dim_feature '{dim_feature}'.")


@dataclass
class PowerLawFit:
    """Generic log-linear power-law fit with identifiability bookkeeping."""

    intercept: float
    exponents: Dict[str, Optional[float]]
    r2_log: Optional[float]
    n_used: int
    unidentified: List[str] = field(default_factory=list)


def fit_power_law(features: Dict[str, Sequence[float]], target: Sequence[float]) -> PowerLawFit:
    """
    OLS of ln(target) on ln(features). Features with no variance are dropped and reported as
    unidentified; a rank-deficient remaining design raises ValueError.
    """
    y = np.log(np.asarray(target, dtype=np.float64))
    n_obs = y.size
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations to fit a power law, got {n_obs}.")
    cols: List[np.ndarray] = [np.ones(n_obs)]
    names: List[str] = []
    unidentified: List[str] = []
    for name, vals in features.items():
        v = np.log(np.asarray(vals, dtype=np.float64))
        if v.size != n_obs:
            raise ValueError(f"Feature '{name}' has {v.size} values, expected {n_obs}.")
        if float(np.std(v)) < _VAR_TOL:
            unidentified.append(name)
            continue
        cols.append(v)
        names.append(name)
    X = np.stack(cols, axis=1)
    if n_obs < X.shape[1] + 1:
        raise ValueError(f"Need at least {X.shape[1] + 1} observations for {X.shape[1]} parameters, got {n_obs}.")
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise ValueError(f"Rank-deficient design (rank {rank} < {X.shape[1]}): regressors {names} are collinear.")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = None if ss_tot < _VAR_TOL else 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot
    exps: Dict[str, Optional[float]] = {name: None for name in features}
    for name, b in zip(names, beta[1:]):
        exps[name] = float(b)
    return PowerLawFit(float(beta[0]), exps, r2, n_obs, unidentified)


@dataclass
class DimensionAlphaLawFit:
    """alpha_0 = min(C * k^a_k * B^a_B * n^a_n * T^a_T * d^a_d, 1). None = not identified."""

    c_alpha: float
    a_k: Optional[float]
    a_b: Optional[float]
    a_n: Optional[float]
    a_t: Optional[float]
    a_d: Optional[float]
    r2_log: Optional[float]
    include_dim: bool = True
    dim_feature: str = "ambient"
    unidentified: List[str] = field(default_factory=list)
    n_used: int = 0
    n_censored_excluded: int = 0

    def predict(self, num_components: float, batch_size: float, num_train: float, epochs: float, dim: float) -> float:
        """`dim` must be on the same scale as `dim_feature` (ambient d or intrinsic estimate)."""
        val = self.c_alpha
        for v, e in ((num_components, self.a_k), (batch_size, self.a_b), (num_train, self.a_n), (epochs, self.a_t), (dim, self.a_d)):
            if e is not None:
                val *= float(v) ** e
        return float(np.clip(val, 1e-6, 1.0))


@dataclass
class DimensionBatchLawFit:
    """B_knee = C_B * k^b_k * n^b_n * T^b_T * d^b_d. None = not identified."""

    c_b: float
    b_k: Optional[float]
    b_n: Optional[float]
    b_t: Optional[float]
    b_d: Optional[float]
    r2_log: Optional[float]
    include_dim: bool = True
    dim_feature: str = "ambient"
    unidentified: List[str] = field(default_factory=list)
    n_used: int = 0
    n_censored_excluded: int = 0

    def predict(self, num_components: float, num_train: float, epochs: float, dim: float, round_pow2: bool = True) -> int:
        val = self.c_b
        for v, e in ((num_components, self.b_k), (num_train, self.b_n), (epochs, self.b_t), (dim, self.b_d)):
            if e is not None:
                val *= float(v) ** e
        val = float(np.clip(val, 1.0, num_train))
        if round_pow2:
            pow2 = 2 ** round(float(np.log2(max(val, 1.0))))
            return int(np.clip(pow2, 1, int(num_train)))
        return int(round(val))


def fit_dimension_aware_alpha0(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
    dim_feature: str = "ambient",
) -> DimensionAlphaLawFit:
    """Fit ln alpha_0 on ln(k, B, n, T[, d]) over uncensored observations."""
    used = [o for o in observations if not o.alpha_censored]
    feats: Dict[str, List[float]] = {
        "k": [o.num_components for o in used],
        "B": [o.batch_size for o in used],
        "n": [o.num_train for o in used],
        "T": [o.epochs for o in used],
    }
    if include_dim:
        feats["d"] = [_dim_value(o, dim_feature) for o in used]
    fit = fit_power_law(feats, [o.alpha_0_opt for o in used])
    e = fit.exponents
    return DimensionAlphaLawFit(
        c_alpha=float(np.exp(fit.intercept)),
        a_k=e["k"], a_b=e["B"], a_n=e["n"], a_t=e["T"], a_d=e.get("d") if include_dim else None,
        r2_log=fit.r2_log, include_dim=include_dim, dim_feature=dim_feature,
        unidentified=fit.unidentified, n_used=fit.n_used,
        n_censored_excluded=len(observations) - len(used),
    )


def fit_dimension_aware_batch_knee(
    observations: Sequence[DimensionScalingObservation],
    include_dim: bool = True,
    dim_feature: str = "ambient",
) -> DimensionBatchLawFit:
    """Fit ln B_knee on ln(k, n, T[, d]) over observations with an uncensored knee."""
    with_knee = [o for o in observations if o.b_knee_opt is not None and o.b_knee_opt > 0]
    used = [o for o in with_knee if not o.knee_censored]
    feats: Dict[str, List[float]] = {
        "k": [o.num_components for o in used],
        "n": [o.num_train for o in used],
        "T": [o.epochs for o in used],
    }
    if include_dim:
        feats["d"] = [_dim_value(o, dim_feature) for o in used]
    fit = fit_power_law(feats, [float(o.b_knee_opt) for o in used])  # type: ignore[arg-type]
    e = fit.exponents
    return DimensionBatchLawFit(
        c_b=float(np.exp(fit.intercept)),
        b_k=e["k"], b_n=e["n"], b_t=e["T"], b_d=e.get("d") if include_dim else None,
        r2_log=fit.r2_log, include_dim=include_dim, dim_feature=dim_feature,
        unidentified=fit.unidentified, n_used=fit.n_used,
        n_censored_excluded=len(with_knee) - len(used),
    )


def nll_at_alpha(alpha_curve: Tuple[Sequence[float], Sequence[float]], alpha: float) -> Tuple[float, bool]:
    """Linear interpolation of a measured NLL curve in log(alpha). Returns (nll, extrapolated_flag)."""
    alphas = np.asarray(alpha_curve[0], dtype=np.float64)
    nlls = np.asarray(alpha_curve[1], dtype=np.float64)
    order = np.argsort(alphas)
    la, ln_ = np.log(alphas[order]), nlls[order]
    x = float(np.log(alpha))
    extrap = x < la[0] - 1e-12 or x > la[-1] + 1e-12
    return float(np.interp(np.clip(x, la[0], la[-1]), la, ln_)), bool(extrap)


def evaluate_leave_one_out(
    observations: Sequence[DimensionScalingObservation],
    attr_name: str,
    include_dim: bool = True,
    dim_feature: str = "ambient",
    return_points: bool = False,
) -> Dict[str, Any]:
    """
    Leave one value of `attr_name` out, refit on the rest, predict alpha_0 for the held-out
    configurations. Reports log-RMSE / relative error of alpha and, where the held-out
    observation carries its measured curve, the NLL regret in nats per dimension.
    """
    usable = [(i, o) for i, o in enumerate(observations) if not o.alpha_censored]
    unique_vals = sorted({getattr(o, attr_name) for _, o in usable})
    log_sq, rel, regret_pd, n_extrap, folds = [], [], [], 0, 0
    points: List[Dict[str, float]] = []
    for held in unique_vals:
        train = [o for _, o in usable if getattr(o, attr_name) != held]
        test = [(i, o) for i, o in usable if getattr(o, attr_name) == held]
        try:
            fit = fit_dimension_aware_alpha0(train, include_dim=include_dim, dim_feature=dim_feature)
        except ValueError:
            continue
        folds += 1
        for i, o in test:
            dval = _dim_value(o, dim_feature) if include_dim else o.dim
            pred = fit.predict(o.num_components, o.batch_size, o.num_train, o.epochs, dval)
            log_sq.append(float((np.log(pred) - np.log(o.alpha_0_opt)) ** 2))
            rel.append(float(abs(pred - o.alpha_0_opt) / o.alpha_0_opt))
            point = {"index": float(i), "alpha_pred": float(pred), "log_err": float(np.log(pred) - np.log(o.alpha_0_opt))}
            if o.alpha_curve is not None:
                nll_pred, ex = nll_at_alpha(o.alpha_curve, pred)
                n_extrap += int(ex)
                r = (nll_pred - min(o.alpha_curve[1])) / max(o.dim, 1.0)
                regret_pd.append(r)
                point["regret_nats_per_dim"] = float(r)
            points.append(point)
    nan = float("nan")
    return {
        "log_rmse": float(np.sqrt(np.mean(log_sq))) if log_sq else nan,
        "mean_rel_err": float(np.mean(rel)) if rel else nan,
        "regret_nats_per_dim_mean": float(np.mean(regret_pd)) if regret_pd else nan,
        "regret_nats_per_dim_median": float(np.median(regret_pd)) if regret_pd else nan,
        "n_regret_extrapolated": float(n_extrap),
        "n_folds": float(folds),
        "n_heldout_points": float(len(log_sq)),
        **({"points": points} if return_points else {}),
    }


def evaluate_loso_n(observations, include_dim: bool = True, dim_feature: str = "ambient") -> Dict[str, float]:
    """Leave-One-Size-Out (LOSO) validation across dataset sizes n."""
    res = evaluate_leave_one_out(observations, "num_train", include_dim, dim_feature)
    out = {f"loso_{k}": v for k, v in res.items() if k not in ("n_folds",)}
    out["n_folds"] = res["n_folds"]
    return out


def evaluate_lodo_d(observations, include_dim: bool = True, dim_feature: str = "ambient") -> Dict[str, float]:
    """Leave-One-Dimension-Out (LODO) validation across ambient dimensions d."""
    res = evaluate_leave_one_out(observations, "dim", include_dim, dim_feature)
    out = {f"lodo_{k}": v for k, v in res.items() if k not in ("n_folds",)}
    out["n_folds"] = res["n_folds"]
    return out


def compute_nll_regret(
    nll_law: float,
    nll_oracle: float,
    nll_reference: Optional[float] = None,
) -> Dict[str, float]:
    """
    Absolute NLL gap (nats) and relative regret. If nll_reference (e.g. a 1-component Gaussian
    NLL) is given, normalize by |nll_reference - nll_oracle| so the ratio stays defined when NLL < 0.
    """
    gap = float(nll_law - nll_oracle)
    if nll_reference is not None:
        denom = max(abs(float(nll_reference) - float(nll_oracle)), 1e-8)
    else:
        denom = max(abs(float(nll_oracle)), 1e-8)
    return {"nll_gap_nats": gap, "nll_rel_regret": float(gap / denom)}
