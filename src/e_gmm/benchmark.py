"""Matched, tuned baseline benchmark for e-GMM (added 2026-11-25, see docs/review/).

Every method gets the same data split, covariance type, covariance floor (added to the readout),
mini-batch size and epoch budget. Each method's step/decay hyper-parameters are tuned on a
validation split drawn with a dedicated tuning seed, then the chosen setting is evaluated on
held-out test data for every evaluation seed. Results are paired by seed and compared against
the reference method (tuned e-GMM with spectral revival) with `e_gmm.stats.paired_comparison`.

Units: test NLL is reported in nats per dimension (nll / d), the unit of the 0.01 practical
threshold used by the report compilers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import EGMMConfig
from .metrics import evaluate_gmm_nll
from .numpy_egmm import (
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyEMeansThenEM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
)
from .stats import paired_comparison

Split = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (train, val, test)
RunFn = Callable[[np.ndarray, "RunContext", Dict[str, Any]], Dict[str, Any]]

DEFAULT_EGMM_ALPHAS = (0.05, 0.10, 0.20, 0.35)
DEFAULT_ONLINE_BETAS = (0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_ONLINE_T0S = (1.0, 10.0, 100.0)
DEFAULT_EMEANS_ALPHAS = (0.10, 0.20, 0.35)


@dataclass
class RunContext:
    k: int
    cov_type: str
    cov_floor: float
    batch_size: int
    epochs: int
    seed: int
    egmm_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodSpec:
    name: str
    grid: List[Dict[str, Any]]
    run: RunFn
    description: str = ""
    # param -> (factor, lo, hi): if the tuned value lands on the edge of the grid, tuning tries value*factor
    # (upper edge) or value/factor (lower edge), clipped to [lo, hi], until the optimum is interior.
    extend: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


ALPHA_EXTEND = {"alpha_0": (2.0, 1e-3, 1.0)}
MAX_GRID_EXTENSIONS = 4


def egmm_config(dim: int, ctx: RunContext, params: Dict[str, Any], split_strategy: str = "spectral_v1") -> EGMMConfig:
    """The single e-GMM configuration used by the benchmark and the sweep (NumPy and torch)."""
    cfg_kwargs: Dict[str, Any] = dict(
        num_components=ctx.k, dim=dim, cov_type=ctx.cov_type, cov_floor=ctx.cov_floor,
        init_prior_mass=1.0, init_sigma0=1.0, split_interval=5, split_tau=0.10, split_delta=0.50,
        split_strategy=split_strategy, donor_weight_by="variance_mass", donor_min_mult=1.25,
        cooldown_steps=15, finalization="last_epoch", seed=ctx.seed,
    )
    cfg_kwargs.update(ctx.egmm_overrides)
    cfg_kwargs.update(params)
    return EGMMConfig(**cfg_kwargs)


def egmm_runner(split_strategy: str) -> RunFn:
    def run(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = egmm_config(X.shape[1], ctx, params, split_strategy)
        return NumPyEGMM(cfg).fit(X, batch_size=ctx.batch_size, epochs=ctx.epochs)

    return run


def torch_egmm_runner(split_strategy: str = "spectral_v1", device: Optional[str] = None) -> RunFn:
    """Same configuration on TorchEGMM. NOT executed on the review machine (torch not installed)."""
    import torch

    from .torch_egmm import TorchEGMM

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = egmm_config(X.shape[1], ctx, params, split_strategy)
        res = TorchEGMM(cfg).to(dev).fit(
            torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev),
            batch_size=ctx.batch_size, epochs=ctx.epochs,
        )
        out = {}
        for key in ("pi", "mu", "cov", "U", "health"):
            val = res.get(key)
            out[key] = val.detach().cpu().numpy() if hasattr(val, "detach") else val
        return out

    return run


def _online_em(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
    return NumPyOnlineEMCappeMoulines(
        ctx.k, X.shape[1], cov_type=ctx.cov_type, beta=params["beta"], t0=params["t0"],
        cov_floor=ctx.cov_floor, init_sigma0=1.0, seed=ctx.seed,
    ).fit(X, batch_size=ctx.batch_size, epochs=ctx.epochs)


def _emeans_posthoc(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
    return NumPyEMeansPostHocGMM(
        ctx.k, X.shape[1], cov_type=ctx.cov_type, cov_floor=ctx.cov_floor, alpha_0=params["alpha_0"], seed=ctx.seed,
    ).fit(X, batch_size=ctx.batch_size, epochs=ctx.epochs)


def _emeans_then_em(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
    return NumPyEMeansThenEM(
        ctx.k, X.shape[1], cov_type=ctx.cov_type, cov_floor=ctx.cov_floor, alpha_0=params["alpha_0"],
        em_iters=params.get("em_iters", 3), seed=ctx.seed,
    ).fit(X, batch_size=ctx.batch_size, epochs=ctx.epochs)


def _fullbatch(init: str) -> RunFn:
    def run(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
        return NumPyFullBatchEM(
            ctx.k, X.shape[1], cov_type=ctx.cov_type, cov_floor=ctx.cov_floor, seed=ctx.seed, init=init,
            emeans_batch_size=ctx.batch_size,
        ).fit(X, max_iters=params.get("max_iters", 25), tol=params.get("tol", 1e-4))

    return run


def _sklearn_gmm(X: np.ndarray, ctx: RunContext, params: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.mixture import GaussianMixture  # optional dependency

    gm = GaussianMixture(
        n_components=ctx.k, covariance_type=ctx.cov_type, reg_covar=ctx.cov_floor,
        max_iter=params.get("max_iter", 25), random_state=ctx.seed,
    ).fit(np.asarray(X, dtype=np.float64))
    return {"pi": gm.weights_, "mu": gm.means_, "cov": gm.covariances_}


def default_methods(
    egmm_alphas: Sequence[float] = DEFAULT_EGMM_ALPHAS,
    online_betas: Sequence[float] = DEFAULT_ONLINE_BETAS,
    online_t0s: Sequence[float] = DEFAULT_ONLINE_T0S,
    emeans_alphas: Sequence[float] = DEFAULT_EMEANS_ALPHAS,
    include_sklearn: bool = True,
) -> Tuple[List[MethodSpec], Dict[str, str]]:
    """Return (methods, skipped) where skipped maps a method name to the reason it is absent."""
    methods = [
        MethodSpec("egmm_spectral", [{"alpha_0": a} for a in egmm_alphas], egmm_runner("spectral_v1"),
                   "e-GMM, spectral v1 revival, alpha_0 tuned", ALPHA_EXTEND),
        MethodSpec("egmm_no_revive", [{"alpha_0": a} for a in egmm_alphas], egmm_runner("none"),
                   "e-GMM without revival, alpha_0 tuned (the H0 control)", ALPHA_EXTEND),
        MethodSpec("egmm_faiss_alt", [{"alpha_0": a} for a in egmm_alphas], egmm_runner("faiss_alt"),
                   "e-GMM, Faiss-style +/-delta revival, alpha_0 tuned", ALPHA_EXTEND),
        MethodSpec("online_em", [{"beta": b, "t0": t} for b in online_betas for t in online_t0s], _online_em,
                   "Cappe-Moulines online EM, (beta, t0) tuned, matched floor", {"t0": (10.0, 0.1, 1e4)}),
        MethodSpec("emeans_posthoc", [{"alpha_0": a} for a in emeans_alphas], _emeans_posthoc,
                   "e-means + per-cell covariance, alpha_0 tuned", ALPHA_EXTEND),
        MethodSpec("emeans_then_em3", [{"alpha_0": a, "em_iters": 3} for a in emeans_alphas], _emeans_then_em,
                   "e-means init + 3 full-batch EM iterations, alpha_0 tuned", ALPHA_EXTEND),
        MethodSpec("fullbatch_em_random", [{"max_iters": 25}], _fullbatch("random"),
                   "full-batch EM, random-point init, 25 iterations"),
        MethodSpec("fullbatch_em_emeans", [{"max_iters": 25}], _fullbatch("emeans"),
                   "full-batch EM, e-means init, 25 iterations"),
    ]
    skipped: Dict[str, str] = {}
    if include_sklearn:
        try:
            import sklearn.mixture  # noqa: F401

            methods.append(MethodSpec("sklearn_gmm", [{"max_iter": 25}], _sklearn_gmm,
                                      "scikit-learn GaussianMixture (k-means init), 25 iterations"))
        except ImportError:
            skipped["sklearn_gmm"] = "scikit-learn is not installed"
    return methods, skipped


def _nll_per_dim(X: np.ndarray, fit: Dict[str, Any], cov_type: str) -> float:
    res = evaluate_gmm_nll(X, fit["pi"], fit["mu"], fit["cov"], cov_type=cov_type, U=fit.get("U"))
    return float(res["nll_per_dim"])


def _grid_edge(grid: Sequence[Dict[str, Any]], best: Dict[str, Any]) -> List[str]:
    """Names of tuned parameters whose chosen value is the min or max of a multi-valued grid."""
    edges = []
    for key, val in best.items():
        values = sorted({g[key] for g in grid if key in g and isinstance(g[key], (int, float))})
        if len(values) > 1 and val in (values[0], values[-1]):
            edges.append(key)
    return edges


def _extension_points(spec: MethodSpec, grid: Sequence[Dict[str, Any]], best: Dict[str, Any]) -> List[Dict[str, Any]]:
    """New grid points one geometric step past each extendable edge the best setting sits on."""
    new = []
    for key in _grid_edge(grid, best):
        if key not in spec.extend:
            continue
        factor, lo, hi = spec.extend[key]
        values = sorted({g[key] for g in grid if key in g})
        val = best[key]
        cand = min(val * factor, hi) if val == values[-1] else max(val / factor, lo)
        point = {**best, key: float(cand)}
        if all(point != g for g in grid) and point not in new:
            new.append(point)
    return new


def tune_method(spec: MethodSpec, X_train: np.ndarray, X_val: np.ndarray, ctx: RunContext,
                max_extensions: int = MAX_GRID_EXTENSIONS) -> Dict[str, Any]:
    """
    Pick the setting with the lowest validation NLL per dimension. When the best setting is on the edge
    of the grid for a parameter listed in `spec.extend`, the grid is extended one geometric step past that
    edge and re-scored, up to `max_extensions` times, so a method is not compared while under-tuned.
    `at_grid_edge` lists parameters still on the edge of the final grid (the optimum may lie outside it).
    """
    table: List[Dict[str, Any]] = []
    grid: List[Dict[str, Any]] = []

    def score(points: Sequence[Dict[str, Any]]) -> None:
        for params in points:
            grid.append(params)
            try:
                fit = spec.run(X_train, ctx, params)
                val = _nll_per_dim(X_val, fit, ctx.cov_type)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                table.append({"params": params, "val_nll_per_dim": None, "error": repr(exc)})
                continue
            table.append({"params": params, "val_nll_per_dim": val if np.isfinite(val) else None})

    score(spec.grid)
    extensions = 0
    while True:
        ok = [row for row in table if row["val_nll_per_dim"] is not None]
        if not ok:
            raise RuntimeError(f"{spec.name}: every grid point failed during tuning: {table}")
        best = min(ok, key=lambda row: row["val_nll_per_dim"])
        if extensions >= max_extensions:
            break
        new = _extension_points(spec, grid, best["params"])
        if not new:
            break
        score(new)
        extensions += 1
    return {"best_params": best["params"], "table": table, "extensions": extensions,
            "at_grid_edge": _grid_edge(grid, best["params"])}


def run_benchmark(
    make_split: Callable[[int], Split],
    k: int,
    cov_type: str,
    batch_size: int,
    epochs: int,
    seeds: Sequence[int],
    tuning_seed: int = 10_000,
    cov_floor: float = 1e-4,
    methods: Optional[List[MethodSpec]] = None,
    skipped: Optional[Dict[str, str]] = None,
    reference: str = "egmm_spectral",
    egmm_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Tune every method on the validation split of `make_split(tuning_seed)`, then fit each method with
    its tuned setting on the training split of `make_split(seed)` for every evaluation seed and score
    test NLL per dimension. Returns a JSON-serializable dict.
    """
    if cov_type not in ("diag", "full"):
        raise ValueError("The matched benchmark supports cov_type 'diag' or 'full' (the baselines do not do low-rank).")
    if tuning_seed in seeds:
        raise ValueError("tuning_seed must differ from every evaluation seed.")
    if methods is None:
        methods, skipped = default_methods()
    skipped = dict(skipped or {})
    names = [m.name for m in methods]
    if reference not in names:
        raise ValueError(f"Reference method {reference!r} is not in {names}.")
    overrides = dict(egmm_overrides or {})

    X_tr, X_val, _ = make_split(tuning_seed)
    tuning: Dict[str, Any] = {}
    for spec in methods:
        t0 = time.perf_counter()
        ctx = RunContext(k, cov_type, cov_floor, batch_size, epochs, tuning_seed, overrides)
        tuning[spec.name] = tune_method(spec, X_tr, X_val, ctx)
        tuning[spec.name]["seconds"] = time.perf_counter() - t0
        if verbose:
            print(f"  tuned {spec.name:22s} -> {tuning[spec.name]['best_params']}")

    per_seed: Dict[str, Dict[str, List[Optional[float]]]] = {
        m.name: {"test_nll_per_dim": [], "dead_count": [], "seconds": []} for m in methods
    }
    failures: Dict[str, List[str]] = {}
    for seed in seeds:
        X_tr, X_val, X_te = make_split(seed)
        for spec in methods:
            ctx = RunContext(k, cov_type, cov_floor, batch_size, epochs, seed, overrides)
            t0 = time.perf_counter()
            try:
                fit = spec.run(X_tr, ctx, tuning[spec.name]["best_params"])
                nll = _nll_per_dim(X_te, fit, cov_type)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                failures.setdefault(spec.name, []).append(f"seed {seed}: {exc!r}")
                nll, fit = float("nan"), {}
            rec = per_seed[spec.name]
            rec["test_nll_per_dim"].append(nll if np.isfinite(nll) else None)
            health = fit.get("health") or {}
            rec["dead_count"].append(float(health["dead_count"]) if "dead_count" in health else None)
            rec["seconds"].append(time.perf_counter() - t0)
        if verbose:
            print(f"  seed {seed} done")

    ref = per_seed[reference]["test_nll_per_dim"]
    comparisons: Dict[str, Any] = {}
    for spec in methods:
        if spec.name == reference:
            continue
        vals = per_seed[spec.name]["test_nll_per_dim"]
        pairs = [(a, b) for a, b in zip(vals, ref) if a is not None and b is not None]
        if len(pairs) < 2:
            comparisons[spec.name] = {"n": float(len(pairs)), "note": "too few finite paired results"}
            continue
        a, b = zip(*pairs)
        comparisons[spec.name] = paired_comparison(list(a), list(b))

    summary = {}
    for name, rec in per_seed.items():
        vals = np.array([v for v in rec["test_nll_per_dim"] if v is not None], dtype=np.float64)
        dead = [v for v in rec["dead_count"] if v is not None]
        summary[name] = {
            "test_nll_per_dim_mean": float(vals.mean()) if vals.size else None,
            "test_nll_per_dim_std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "n_finite": int(vals.size),
            "dead_count_mean": float(np.mean(dead)) if dead else None,
            "seconds_mean": float(np.mean(rec["seconds"])),
        }

    return {
        "setup": {
            "k": k, "cov_type": cov_type, "cov_floor": cov_floor, "batch_size": batch_size, "epochs": epochs,
            "seeds": list(seeds), "tuning_seed": tuning_seed, "reference": reference,
            "egmm_overrides": overrides, "unit": "nats per dimension (test NLL / d)",
        },
        "methods": {m.name: {"description": m.description, "grid_size": len(m.grid)} for m in methods},
        "skipped_methods": skipped,
        "tuning": tuning,
        "per_seed": per_seed,
        "summary": summary,
        "paired_vs_reference": comparisons,
        "failures": failures,
    }
