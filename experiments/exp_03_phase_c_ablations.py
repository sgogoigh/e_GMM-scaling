"""
Experiment 03 (Phase C): paired ablations of e-GMM's components
================================================================

Rewritten 2026-11-25 (docs/review/2026-11-25-REVIEW.md). Changes from the original:
  - Revival is crossed with alpha_0. The 2026-11-25 re-check found that the "no revival collapses"
    result came from alpha_0 = 0.35; at alpha_0 = 0.10 no-revival was within noise of spectral splitting.
    Each strategy's alpha_0 is tuned on a validation split before the paired test comparison.
  - Every comparison is paired by seed (default 10 seeds) with an exact sign-flip test; the report
    compiler turns the numbers into verdicts. Nothing here states a conclusion.
  - The floor-dominated rate uses the fixed metric (raw variance below 2 x lambda_0).
  - Data are scale-normalized on the training split, so lambda_0 is comparable across regimes.

HYPOTHESES (experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md):
  H0: with alpha_0 tuned, e-GMM without revival matches e-GMM with spectral revival.
  H5: spectral v1 revival beats the other revival strategies at tuned alpha_0.
  H6: low-rank+diag gives the best BIC on rotated data; top-m keeps dense NLL; lambda_0 trade-off.
  H7: slow-EMA finalization generalizes better than last-epoch finalization.

HOW TO RUN:
  python experiments/exp_03_phase_c_ablations.py --dry-run
  python experiments/exp_03_phase_c_ablations.py --mode smoke --seeds 10
  python experiments/exp_03_phase_c_ablations.py --mode smoke --seeds 10 --only revival

SAVED ARTIFACTS:
  - Results : experiments/results/exp_03_phase_c_ablations.json
  - Log     : experiments/logs/exp_03_phase_c/latest.log
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture, generate_stretched_gaussian, normalize_scale  # noqa: E402
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from e_gmm.stats import paired_comparison  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20, 0.35)
ALPHA_LO = 1e-3            # lower bound when the alpha_0 grid is extended past its lower edge
MAX_GRID_EXTENSIONS = 4
REVIVAL = [
    ("none", "count"), ("batch_resample", "count"), ("faiss_alt", "count"),
    ("random_orth", "count"), ("spectral_v1", "count"), ("spectral_v1", "variance_mass"),
]
SplitFn = Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 03 (Phase C): paired ablations")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--only", type=str, default=None, help="Comma-separated subset: revival,floor,structure,topm,finalization")
    p.add_argument("--backend", choices=["numpy"], default="numpy",
                   help="Only numpy: the torch path has not been run on the review machine.")
    p.add_argument("--out", type=str, default=str(RESULTS_ROOT / "exp_03_phase_c_ablations.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def mixture_split(n_tr, n_val, n_te, d, G, **kw) -> SplitFn:
    def make(seed):
        X, _, _ = generate_gaussian_mixture(n=n_tr + n_val + n_te, d=d, num_true_clusters=G, seed=seed, **kw)
        (tr, va, te), _ = normalize_scale(X[:n_tr], X[n_tr : n_tr + n_val], X[n_tr + n_val :])
        return tr, va, te

    return make


def stretched_split(n_tr, n_val, n_te, d, exponent) -> SplitFn:
    def make(seed):
        X, _ = generate_stretched_gaussian(n=n_tr + n_val + n_te, d=d, exponent=exponent, rotate=True, seed=seed)
        (tr, va, te), _ = normalize_scale(X[:n_tr], X[n_tr : n_tr + n_val], X[n_tr + n_val :])
        return tr, va, te

    return make


def fit_eval(X_tr, X_eval: Dict[str, np.ndarray], cfg: EGMMConfig, B: int, epochs: int) -> Dict[str, Any]:
    res = NumPyEGMM(cfg).fit(X_tr, batch_size=B, epochs=epochs)
    out: Dict[str, Any] = {"health": res["health"], "splits": res["splits"]}
    for name, X in X_eval.items():
        ev = evaluate_gmm_nll(X, res["pi"], res["mu"], res["cov"], cov_type=cfg.cov_type, U=res.get("U"))
        out[f"{name}_nll_per_dim"] = ev["nll_per_dim"]
        out[f"{name}_bic"] = ev["bic"]
        out["n_params"] = ev["n_effective_params"]
    return out


def run_variants(variants: Dict[str, Dict[str, Any]], base: Dict[str, Any], split: SplitFn, seeds: List[int],
                 B: int, epochs: int, reference: str, metric: str = "test_nll_per_dim",
                 extra: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Fit every variant on every seed's split; pair each variant with `reference` on `metric`."""
    per: Dict[str, Dict[str, List[float]]] = {v: {} for v in variants}
    for s in seeds:
        tr, va, te = split(s)
        for vname, over in variants.items():
            cfg = EGMMConfig(**{**base, **over, "dim": tr.shape[1], "seed": s})
            r = fit_eval(tr, {"train": tr, "test": te}, cfg, B, epochs)
            row = {"test_nll_per_dim": r["test_nll_per_dim"], "train_nll_per_dim": r["train_nll_per_dim"],
                   "train_bic": r["train_bic"], "test_bic": r["test_bic"], "dead_count": r["health"]["dead_count"],
                   "floor_hit_rate": r["health"]["floor_hit_rate"], "n_params": r["n_params"], "splits": r["splits"]}
            for key in extra:
                row[key] = r["health"].get(key, float("nan"))
            for key, val in row.items():
                per[vname].setdefault(key, []).append(float(val))
    summary = {v: {k: float(np.mean(vals)) for k, vals in rec.items()} for v, rec in per.items()}
    paired = {v: paired_comparison(per[v][metric], per[reference][metric]) for v in variants if v != reference}
    return {"variants": variants, "reference": reference, "metric": metric, "per_seed": per,
            "summary": summary, "paired_vs_reference": paired}


def tune_alpha(over: Dict[str, Any], base: Dict[str, Any], split: SplitFn, tuning_seeds: List[int], B: int, epochs: int):
    """
    alpha_0 with the lowest mean validation NLL over the tuning seeds (disjoint from evaluation seeds).
    If the best value is on the edge of the grid, the grid is extended one factor-2 step past it (within
    [ALPHA_LO, 1.0], at most MAX_GRID_EXTENSIONS times), as in e_gmm.benchmark.tune_method.
    """
    scores: Dict[float, float] = {}

    def score(a: float) -> None:
        vals = []
        for s in tuning_seeds:
            tr, va, _ = split(s)
            cfg = EGMMConfig(**{**base, **over, "alpha_0": a, "dim": tr.shape[1], "seed": s})
            vals.append(fit_eval(tr, {"val": va}, cfg, B, epochs)["val_nll_per_dim"])
        scores[a] = float(np.mean(vals))

    for a in ALPHAS:
        score(a)
    for _ in range(MAX_GRID_EXTENSIONS):
        best, grid = min(scores, key=scores.get), sorted(scores)
        nxt = max(best / 2, ALPHA_LO) if best == grid[0] else min(best * 2, 1.0) if best == grid[-1] else None
        if nxt is None or nxt in scores:
            break
        score(nxt)
    best = min(scores, key=scores.get)
    return best, scores


def main():
    args = parse_args()
    full = args.mode == "full"
    seeds = list(range(1, args.seeds + 1))
    tuning_seeds = [1001, 1002, 1003]
    wanted = set(args.only.split(",")) if args.only else {"revival", "floor", "structure", "topm", "finalization"}
    print(f"Exp 03 | mode={args.mode} seeds={seeds} ablations={sorted(wanted)}")
    if args.dry_run:
        print("Dry run: nothing fitted.")
        return

    base = dict(num_components=16, cov_type="full", cov_floor=1e-4, init_prior_mass=1.0, init_sigma0=1.0,
                split_interval=5, split_tau=0.10, split_delta=0.50, donor_min_mult=1.25, cooldown_steps=15,
                finalization="last_epoch", split_strategy="spectral_v1", donor_weight_by="variance_mass", alpha_0=0.10)
    out: Dict[str, Any] = {"experiment_id": "exp_03_phase_c_ablations", "mode": args.mode, "backend": "numpy",
                           "seeds": seeds, "tuning_seeds": tuning_seeds, "unit": "nats per dimension"}
    with tee_experiment_logs("exp_03_phase_c", run_tag=args.mode):
        t_start = time.perf_counter()

        if "revival" in wanted:
            n = (40_000, 5000, 5000) if full else (2400, 800, 800)
            k = 64 if full else 16
            B, ep = (1024, 6) if full else (64, 5)
            split = mixture_split(*n, d=64 if full else 16, G=k, zipf=True, zipf_power=1.2, anisotropic=True, rotated=True)
            rb = {**base, "num_components": k}
            grid_rows = {}
            for a in ALPHAS:
                variants = {f"{st}/{dw}": {"split_strategy": st, "donor_weight_by": dw, "alpha_0": a} for st, dw in REVIVAL}
                grid_rows[str(a)] = run_variants(variants, rb, split, seeds, B, ep, reference="spectral_v1/variance_mass")
                s = grid_rows[str(a)]["summary"]
                print(f"  revival alpha_0={a}: " + ", ".join(f"{v}={s[v]['test_nll_per_dim']:.4f} (dead {s[v]['dead_count']:.1f})" for v in s))
            tuned_alpha, tuning_scores = {}, {}
            for st, dw in REVIVAL:
                name = f"{st}/{dw}"
                tuned_alpha[name], tuning_scores[name] = tune_alpha({"split_strategy": st, "donor_weight_by": dw}, rb, split, tuning_seeds, B, ep)
            tuned = run_variants({f"{st}/{dw}": {"split_strategy": st, "donor_weight_by": dw, "alpha_0": tuned_alpha[f"{st}/{dw}"]}
                                  for st, dw in REVIVAL}, rb, split, seeds, B, ep, reference="spectral_v1/variance_mass")
            print(f"  revival tuned alphas: {tuned_alpha}")
            out["revival"] = {"setup": {"n": n, "k": k, "B": B, "epochs": ep, "data": "Zipf-1.2 rotated anisotropic mixture"},
                              "by_alpha": grid_rows, "tuned_alpha": tuned_alpha, "tuning_val_nll_per_dim": tuning_scores,
                              "tuned": tuned}

        if "floor" in wanted:
            n = (20_000, 2000, 4000) if full else (1200, 400, 800)
            split = stretched_split(*n, d=64 if full else 24, exponent=2.0)
            fb = {**base, "num_components": 64 if full else 8, "cov_type": "diag"}
            variants = {f"{fl:g}": {"cov_floor": fl} for fl in (1e-6, 1e-4, 1e-2, 1e-1)}
            out["floor"] = run_variants(variants, fb, split, seeds, 512 if full else 64, 4, reference="0.0001",
                                        extra=("max_condition_number",))
            print("  floor: " + ", ".join(f"{v}={r['test_nll_per_dim']:.4f} (floor-dominated {r['floor_hit_rate']:.2f})"
                                          for v, r in out["floor"]["summary"].items()))

        if "structure" in wanted:
            d = 64 if full else 16
            n = (30_000, 3000, 7500) if full else (1800, 600, 600)
            split = mixture_split(*n, d=d, G=32 if full else 8, anisotropic=True, rotated=True)
            sb = {**base, "num_components": 32 if full else 8}
            variants = {"diag": {"cov_type": "diag"}, "full": {"cov_type": "full"}}
            for r in ((2, 4, 8, 12) if full else (2, 4, 8)):
                variants[f"low_rank_r{r}"] = {"cov_type": "low_rank_diag", "low_rank": r}
            out["structure"] = run_variants(variants, sb, split, seeds, 512 if full else 128, 4, reference="diag")
            out["structure"]["note"] = ("low_rank_diag uses an Oja-style subspace heuristic, not a principled MFA M-step; "
                                        "train_bic (used for H6a) is computed on the training split, test_bic on the test split")
            print("  structure: " + ", ".join(f"{v}: NLL/d={r['test_nll_per_dim']:.4f} train BIC={r['train_bic']:.1f}"
                                              for v, r in out["structure"]["summary"].items()))

        if "topm" in wanted:
            n = (30_000, 3000, 7500) if full else (1800, 600, 600)
            k = 128 if full else 32
            split = mixture_split(*n, d=64 if full else 20, G=k, anisotropic=True)
            tb = {**base, "num_components": k, "cov_type": "diag"}
            variants = {"dense": {"top_m": None}, **{f"m{m}": {"top_m": m} for m in (2, 4, 8, 16)}}
            out["topm"] = run_variants(variants, tb, split, seeds, 512 if full else 128, 4, reference="dense")
            out["topm"]["m_step_work_ratio"] = {v: (1.0 if o["top_m"] is None else o["top_m"] / k) for v, o in variants.items()}
            print("  top-m: " + ", ".join(f"{v}={r['test_nll_per_dim']:.4f}" for v, r in out["topm"]["summary"].items()))

        if "finalization" in wanted:
            out["finalization"] = {}
            for label, n_tr in (("small_n_over_k", 4000 if full else 800), ("medium_n_over_k", 40_000 if full else 3200)):
                split = mixture_split(n_tr, max(400, n_tr // 4), max(400, n_tr // 4), d=16, G=16, zipf=True, anisotropic=True, rotated=True)
                variants = {f: {"finalization": f} for f in ("last_epoch", "slow_ema", "main_ema")}
                out["finalization"][label] = run_variants(variants, base, split, seeds, 64, 5, reference="last_epoch")
                print(f"  finalization [{label}]: " + ", ".join(
                    f"{v}: train {r['train_nll_per_dim']:.4f} test {r['test_nll_per_dim']:.4f}"
                    for v, r in out["finalization"][label]["summary"].items()))

        out["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"Saved -> {path} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
