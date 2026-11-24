"""Scaling-law sweep designed so the exponents are identifiable (added 2026-11-25, see docs/review/).

What changed relative to the original PoC-3 / Exp-01 sweeps:
  - T (epochs) is varied, so a_T is not collinear with the intercept.
  - The best alpha_0 is found on an adaptive ladder that is extended until the minimum is interior;
    if it still sits on an edge (1e-3 or 1.0) the observation is flagged as censored and excluded
    from the fit instead of being fitted as if it were the optimum.
  - Batch knees are searched up to n/4 with a tolerance in nats per dimension, using the fitted
    alpha law at every batch size; a knee at the top of the range is flagged as censored.
  - The data generator is a 32-cluster anisotropic rotated mixture (not a single Gaussian), and
    for real data d is varied by PCA projection of ONE source, so d is not confounded with the
    dataset.
  - The intrinsic dimension (TwoNN) of every training split is recorded, and the law is fitted with
    no d term, with ambient d and with intrinsic d. The d term is judged by held-out NLL regret
    (leave-one-d-out and out-of-distribution configurations), not by R^2.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .benchmark import RunContext, RunFn, egmm_runner
from .datasets import (
    estimate_intrinsic_dim_twonn,
    generate_gaussian_mixture,
    normalize_scale,
    pca_project,
    random_splits,
)
from .metrics import evaluate_gmm_nll
from .scaling_law import (
    DimensionScalingObservation,
    evaluate_lodo_d,
    evaluate_loso_n,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
    nll_at_alpha,
)
from .schedule import predict_d_aware_alpha0
from .stats import paired_comparison

DEFAULT_ALPHA_LADDER = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
ALPHA_MIN, ALPHA_MAX = 1e-3, 1.0

DataFn = Callable[[int, int, int, int], Tuple[np.ndarray, np.ndarray]]  # (d, n_train, n_val, seed) -> (train, val)


# ---------------------------------------------------------------------------------------------
# One-dimensional searches
# ---------------------------------------------------------------------------------------------


def _parabolic_vertex(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x0, x1, x2 = x
    y0, y1, y2 = y
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if a <= 0:
        return None
    return float(np.clip(-b / (2 * a), x0, x2))


def alpha_ladder_search(
    evaluate: Callable[[float], float],
    base: Sequence[float] = DEFAULT_ALPHA_LADDER,
    lo: float = ALPHA_MIN,
    hi: float = ALPHA_MAX,
    max_extensions: int = 4,
) -> Dict[str, Any]:
    """
    Evaluate `evaluate(alpha)` (lower is better) on a geometric ladder, extending it by factors of 2
    while the minimum sits on an edge and the bound [lo, hi] allows. `censored` is True when the
    final minimum is still on an edge of the tested ladder. `alpha_star` is the vertex of a parabola
    in log(alpha) through the minimum and its neighbours (the argmin itself when censored).
    """
    cache: Dict[float, float] = {}

    def ev(a: float) -> None:
        a = float(a)
        if a not in cache:
            cache[a] = float(evaluate(a))

    for a in base:
        ev(min(max(a, lo), hi))
    extensions = 0
    while extensions < max_extensions:
        alphas = sorted(cache)
        i = int(np.argmin([cache[a] for a in alphas]))
        if i == 0 and alphas[0] > lo * (1 + 1e-9):
            ev(max(alphas[0] / 2.0, lo))
        elif i == len(alphas) - 1 and alphas[-1] < hi * (1 - 1e-9):
            ev(min(alphas[-1] * 2.0, hi))
        else:
            break
        extensions += 1
    alphas = sorted(cache)
    nlls = [cache[a] for a in alphas]
    i = int(np.argmin(nlls))
    censored = i == 0 or i == len(alphas) - 1
    alpha_star = alphas[i]
    if not censored:
        v = _parabolic_vertex(np.log(alphas[i - 1 : i + 2]), nlls[i - 1 : i + 2])
        if v is not None:
            alpha_star = float(np.exp(v))
    return {
        "alpha_star": float(alpha_star),
        "alpha_argmin": float(alphas[i]),
        "nll_min": float(nlls[i]),
        "censored": bool(censored),
        "censored_side": ("low" if i == 0 else "high") if censored else None,
        "alphas": [float(a) for a in alphas],
        "nlls": [float(v) for v in nlls],
        "extensions": extensions,
    }


def power_of_two_batches(n: int, b_min: int = 16, max_fraction: float = 0.25) -> List[int]:
    b_max = int(n * max_fraction)
    if b_max < b_min:
        raise ValueError(f"n={n} is too small for a knee search starting at B={b_min} (max B = {b_max}).")
    out, b = [], b_min
    while b <= b_max:
        out.append(b)
        b *= 2
    return out


def batch_knee_search(
    evaluate: Callable[[int], float],
    batch_sizes: Sequence[int],
    dim: int,
    tol_nats_per_dim: float = 0.002,
) -> Dict[str, Any]:
    """
    Knee = the largest batch size reached by walking up from the best batch size while the NLL stays
    within `tol_nats_per_dim * dim` of the best. Censored when that walk reaches the largest tested B.
    """
    bs = sorted(int(b) for b in batch_sizes)
    nlls = [float(evaluate(b)) for b in bs]
    i_best = int(np.argmin(nlls))
    limit = nlls[i_best] + tol_nats_per_dim * dim
    j = i_best
    while j + 1 < len(bs) and nlls[j + 1] <= limit:
        j += 1
    return {
        "knee": int(bs[j]),
        "censored": bool(j == len(bs) - 1),
        "best_batch": int(bs[i_best]),
        "batch_sizes": bs,
        "nlls": nlls,
        "tol_nats_per_dim": tol_nats_per_dim,
    }


# ---------------------------------------------------------------------------------------------
# Data providers
# ---------------------------------------------------------------------------------------------


def synthetic_data_fn(num_true_clusters: int = 32, zipf: bool = True, zipf_power: float = 1.0) -> DataFn:
    """32-cluster anisotropic, rotated mixture; for fixed (d, seed) the mixture is the same for every n."""

    def make(d: int, n_train: int, n_val: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        X, _, _ = generate_gaussian_mixture(
            n=n_train + n_val, d=d, num_true_clusters=num_true_clusters, zipf=zipf, zipf_power=zipf_power,
            anisotropic=True, rotated=True, seed=seed,
        )
        (tr, va), _ = normalize_scale(X[:n_train], X[n_train:])
        return tr, va

    return make


def real_data_fn(arr: np.ndarray) -> DataFn:
    """Random rows of one source, PCA-projected to d (fitted on the training split), scale-normalized."""
    d_src = int(arr.shape[1])

    def make(d: int, n_train: int, n_val: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        if d > d_src:
            raise ValueError(f"Requested d={d} but the source has only {d_src} dimensions.")
        (tr, va), _ = random_splits(arr, [n_train, n_val], seed=seed)
        if d < d_src:
            (tr, va), _ = pca_project(tr, d, va)
        (tr, va), _ = normalize_scale(tr, va)
        return tr, va

    return make


# ---------------------------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------------------------


@dataclass
class SweepGrid:
    dims: List[int]
    ks: List[int]
    ns: List[int]
    batch_sizes: List[int]
    epochs: List[int]
    seeds: List[int]
    n_val: int = 1000
    cov_type: str = "diag"
    cov_floor: float = 1e-4
    knee_b_min: int = 16
    knee_tol_nats_per_dim: float = 0.002
    ood: List[Dict[str, int]] = field(default_factory=list)  # each {"d","k","n","B","T"}
    alpha_ladder: Tuple[float, ...] = DEFAULT_ALPHA_LADDER


def _val_nll(X_tr: np.ndarray, X_val: np.ndarray, k: int, B: int, T: int, alpha: float, seed: int,
             grid: SweepGrid, overrides: Dict[str, Any], runner: Optional[RunFn] = None) -> float:
    ctx = RunContext(k, grid.cov_type, grid.cov_floor, B, T, seed, overrides)
    fit = (runner or egmm_runner("spectral_v1"))(X_tr, ctx, {"alpha_0": float(alpha)})
    return float(evaluate_gmm_nll(X_val, fit["pi"], fit["mu"], fit["cov"], cov_type=grid.cov_type, U=fit.get("U"))["nll"])


def _law_dict(fit: Any) -> Dict[str, Any]:
    return asdict(fit)


def run_sweep(
    data_fn: DataFn,
    grid: SweepGrid,
    source: str = "synthetic",
    egmm_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    runner: Optional[RunFn] = None,
) -> Dict[str, Any]:
    """
    runner: fits one e-GMM given (X_train, RunContext, {"alpha_0": a}) and returns pi/mu/cov[/U];
    defaults to the NumPy spectral-revival e-GMM (benchmark.egmm_runner("spectral_v1")).
    """
    overrides = dict(egmm_overrides or {})
    t_start = time.perf_counter()
    data_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def data(d: int, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (d, n, seed)
        if key not in data_cache:
            data_cache[key] = data_fn(d, n, grid.n_val, seed)
        return data_cache[key]

    intrinsic: Dict[Tuple[int, int], float] = {}

    def d_int(d: int, n: int) -> float:
        if (d, n) not in intrinsic:
            intrinsic[(d, n)] = estimate_intrinsic_dim_twonn(data(d, n, grid.seeds[0])[0], seed=0)
        return intrinsic[(d, n)]

    def ladder(d: int, k: int, n: int, B: int, T: int) -> Dict[str, Any]:
        def evaluate(a: float) -> float:
            return float(np.mean([_val_nll(*data(d, n, s), k, B, T, a, s, grid, overrides, runner)
                                  for s in grid.seeds]))

        return alpha_ladder_search(evaluate, base=grid.alpha_ladder)

    # Phase 1: alpha ladders --------------------------------------------------------------------
    observations: List[DimensionScalingObservation] = []
    skipped_configs: List[Dict[str, Any]] = []
    for d in grid.dims:
        for n in grid.ns:
            for k in grid.ks:
                for T in grid.epochs:
                    for B in grid.batch_sizes:
                        if B > n // 2 or n < 10 * k:
                            skipped_configs.append({"d": d, "k": k, "n": n, "B": B, "T": T,
                                                    "reason": "B > n/2 or n < 10k"})
                            continue
                        res = ladder(d, k, n, B, T)
                        observations.append(DimensionScalingObservation(
                            num_train=n, num_components=k, dim=d, batch_size=B, epochs=T,
                            alpha_0_opt=res["alpha_star"], nll_opt=res["nll_min"],
                            alpha_censored=res["censored"], dim_intrinsic=d_int(d, n),
                            alpha_curve=(res["alphas"], res["nlls"]), source=source,
                        ))
                        if verbose:
                            flag = " CENSORED-" + str(res["censored_side"]) if res["censored"] else ""
                            print(f"  d={d:4d} k={k:3d} n={n:6d} B={B:5d} T={T}: alpha*={res['alpha_star']:.4f}{flag}")

    # Phase 2: alpha laws ------------------------------------------------------------------------
    variants = {"no_d": (False, "ambient"), "ambient_d": (True, "ambient"), "intrinsic_d": (True, "intrinsic")}
    alpha_laws: Dict[str, Any] = {}
    fitted: Dict[str, Any] = {}
    for name, (inc, feat) in variants.items():
        try:
            law = fit_dimension_aware_alpha0(observations, include_dim=inc, dim_feature=feat)
            fitted[name] = law
            alpha_laws[name] = {
                "fit": _law_dict(law),
                "loso_n": evaluate_loso_n(observations, include_dim=inc, dim_feature=feat),
                "lodo_d": evaluate_lodo_d(observations, include_dim=inc, dim_feature=feat),
            }
        except ValueError as exc:
            alpha_laws[name] = {"error": str(exc)}

    lodo_paired = _paired_lodo(observations, variants)

    # Phase 3: knees at the law-predicted alpha --------------------------------------------------
    knee_law_source = "ambient_d" if "ambient_d" in fitted else ("no_d" if "no_d" in fitted else None)
    knee_observations: List[DimensionScalingObservation] = []
    knee_records: List[Dict[str, Any]] = []
    knee_skip_reason = None
    if knee_law_source is None:
        knee_skip_reason = "no alpha law could be fitted (see alpha_laws errors)"
    else:
        law = fitted[knee_law_source]
        for d in grid.dims:
            for n in grid.ns:
                try:
                    bs = power_of_two_batches(n, grid.knee_b_min)
                except ValueError as exc:
                    knee_records.append({"d": d, "n": n, "error": str(exc)})
                    continue
                for k in grid.ks:
                    if n < 10 * k:
                        continue
                    for T in grid.epochs:
                        def a_for(B: int, k=k, n=n, T=T, d=d) -> float:
                            dval = d_int(d, n) if law.dim_feature == "intrinsic" else d
                            return float(np.clip(law.predict(k, B, n, T, dval), ALPHA_MIN, ALPHA_MAX))

                        def evaluate_b(B: int, k=k, n=n, T=T, d=d) -> float:
                            return float(np.mean([_val_nll(*data(d, n, s), k, B, T, a_for(B), s, grid, overrides, runner)
                                                  for s in grid.seeds]))

                        res = batch_knee_search(evaluate_b, bs, dim=d, tol_nats_per_dim=grid.knee_tol_nats_per_dim)
                        res.update({"d": d, "k": k, "n": n, "T": T, "alphas_used": [a_for(b) for b in bs]})
                        knee_records.append(res)
                        knee_observations.append(DimensionScalingObservation(
                            num_train=n, num_components=k, dim=d, batch_size=res["knee"], epochs=T,
                            alpha_0_opt=a_for(res["knee"]), b_knee_opt=float(res["knee"]),
                            knee_censored=res["censored"], dim_intrinsic=d_int(d, n), source=source,
                        ))
                        if verbose:
                            flag = " CENSORED" if res["censored"] else ""
                            print(f"  knee d={d:4d} k={k:3d} n={n:6d} T={T}: B_knee={res['knee']}{flag}")

    knee_laws: Dict[str, Any] = {}
    for name, (inc, feat) in variants.items():
        try:
            knee_laws[name] = {"fit": _law_dict(fit_dimension_aware_batch_knee(knee_observations, include_dim=inc, dim_feature=feat))}
        except ValueError as exc:
            knee_laws[name] = {"error": str(exc)}

    # Phase 4: out-of-distribution configurations ------------------------------------------------
    ood = _ood_eval(grid, data, d_int, ladder, fitted)

    n_obs = len(observations)
    n_knee = len(knee_observations)
    return {
        "source": source,
        "grid": asdict(grid),
        "egmm_overrides": overrides,
        "observations": [asdict(o) for o in observations],
        "skipped_configs": skipped_configs,
        "censoring": {
            "alpha_censored": int(sum(o.alpha_censored for o in observations)),
            "alpha_total": n_obs,
            "knee_censored": int(sum(o.knee_censored for o in knee_observations)),
            "knee_total": n_knee,
        },
        "intrinsic_dims": {f"d={d},n={n}": v for (d, n), v in intrinsic.items()},
        "alpha_laws": alpha_laws,
        "lodo_paired_regret": lodo_paired,
        "knee_law_alpha_source": knee_law_source,
        "knee_skip_reason": knee_skip_reason,
        "knees": knee_records,
        "knee_laws": knee_laws,
        "ood": ood,
        "seconds": time.perf_counter() - t_start,
    }


def _paired_lodo(observations: Sequence[DimensionScalingObservation], variants: Dict[str, Tuple[bool, str]]) -> Dict[str, Any]:
    """Paired leave-one-d-out regret: each d-variant against the no-d law on the same held-out points."""
    from .scaling_law import evaluate_leave_one_out

    pts: Dict[str, Dict[int, float]] = {}
    for name, (inc, feat) in variants.items():
        res = evaluate_leave_one_out(observations, "dim", include_dim=inc, dim_feature=feat, return_points=True)
        pts[name] = {int(p["index"]): p["regret_nats_per_dim"] for p in res.get("points", []) if "regret_nats_per_dim" in p}
    out: Dict[str, Any] = {}
    for name in ("ambient_d", "intrinsic_d"):
        common = sorted(set(pts.get(name, {})) & set(pts.get("no_d", {})))
        if len(common) < 2:
            out[f"{name}_vs_no_d"] = {"n": float(len(common)), "note": "too few common held-out points"}
            continue
        out[f"{name}_vs_no_d"] = paired_comparison([pts[name][i] for i in common], [pts["no_d"][i] for i in common])
    return out


def _ood_eval(grid: SweepGrid, data, d_int, ladder, fitted: Dict[str, Any]) -> Dict[str, Any]:
    if not grid.ood:
        return {"configs": [], "note": "no OOD configurations requested"}
    rows = []
    for cfg in grid.ood:
        d, k, n, B, T = (int(cfg[key]) for key in ("d", "k", "n", "B", "T"))
        res = ladder(d, k, n, B, T)
        curve = (res["alphas"], res["nlls"])
        row: Dict[str, Any] = {"config": cfg, "alpha_star": res["alpha_star"], "censored": res["censored"],
                               "nll_min": res["nll_min"], "regret_nats_per_dim": {}, "alpha_pred": {}}
        preds = {"paper_prior_no_d": predict_d_aware_alpha0(k, B, n, T, d)}
        for name, law in fitted.items():
            dval = d_int(d, n) if law.dim_feature == "intrinsic" else d
            preds[name] = float(np.clip(law.predict(k, B, n, T, dval), ALPHA_MIN, ALPHA_MAX))
        for name, a in preds.items():
            nll, ex = nll_at_alpha(curve, a)
            row["alpha_pred"][name] = a
            row["regret_nats_per_dim"][name] = (nll - res["nll_min"]) / d
            row.setdefault("extrapolated", {})[name] = ex
        rows.append(row)
    paired: Dict[str, Any] = {}
    for name in ("ambient_d", "intrinsic_d", "paper_prior_no_d"):
        a = [r["regret_nats_per_dim"].get(name) for r in rows]
        b = [r["regret_nats_per_dim"].get("no_d") for r in rows]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) >= 2:
            xs, ys = zip(*pairs)
            paired[f"{name}_vs_no_d"] = paired_comparison(list(xs), list(ys))
    return {"configs": rows, "paired_vs_no_d": paired}
