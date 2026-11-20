"""
PoC 3 (CPU / 1x GPU, $0 Cost): Empirical Dimension-Aware Scaling-Law Fit (alpha_0 & B_knee)
=============================================================================================

Runs a multi-seed empirical hyperparameter sweep over StretchedGaussian across:
  - d in {8, 16, 32, 64}
  - k in {4, 8, 16}
  - n in {1200, 2400, 4800}
  - B in {32, 128}
Fits BOTH dimension-aware power laws:
  1. alpha_0(k, B, n, T, d)
  2. B_knee(k, n, T, d) (largest batch size within +0.2 nats/dim or +0.5% NLL of best)
Validates R^2, LODO-d, LOSO-n, and zero-shot unseen-d (d=24) transfer!
"""

import argparse
import itertools
import json
import pathlib
import sys
import numpy as np

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_stretched_gaussian  # noqa: E402
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402
from e_gmm.scaling_law import (  # noqa: E402
    DimensionScalingObservation,
    compute_nll_regret,
    evaluate_lodo_d,
    evaluate_loso_n,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
)


def parse_args():
    p = argparse.ArgumentParser(description="PoC 3: Empirical Dimension-Aware Scaling Law Sweep")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else "sweep"
    with tee_poc_logs("poc_03", run_tag=tag):
        dims = [8, 16, 32, 64]
        ks = [4, 8, 16]
        ns = [1200, 2400, 4800]
        bs = [32, 128]
        alpha_grid = [0.01, 0.02, 0.04, 0.08, 0.14, 0.24, 0.40, 0.65, 0.90]
        grid = list(itertools.product(ns, dims, ks, bs))

        if args.dry_run:
            print("=== PoC 3 DRY RUN ===")
            print(
                f"Alpha_0 sweep: {len(grid)} x {len(alpha_grid)} alphas x {args.seeds} seeds = "
                f"{len(grid) * len(alpha_grid) * args.seeds} runs."
            )
            return

        observations = []
        for n, d, k, B in grid:
            vals = {a0: [] for a0 in alpha_grid}
            for s in range(1, args.seeds + 1):
                X_tr, _ = generate_stretched_gaussian(n=n, d=d, exponent=1.0, seed=args.seed + s * 100 + d)
                X_val, _ = generate_stretched_gaussian(n=500, d=d, exponent=1.0, seed=args.seed + s * 100 + 999 + d)
                for a0 in alpha_grid:
                    cfg = EGMMConfig(
                        num_components=k,
                        dim=d,
                        cov_type="diag",
                        cov_floor=1e-3,
                        init_prior_mass=max(1.0, 0.25 * B / k),
                        alpha_0=a0,
                        finalization="main_ema",
                        seed=args.seed + s,
                    )
                    res = NumPyEGMM(cfg).fit(X_tr, batch_size=B, epochs=args.epochs)
                    vals[a0].append(evaluate_gmm_nll(X_val, res["pi"], res["mu"], res["cov"], cov_type="diag")["nll"])
            best_a0 = min(alpha_grid, key=lambda a: float(np.mean(vals[a])))
            best_nll = float(np.mean(vals[best_a0]))
            observations.append(DimensionScalingObservation(n, k, d, B, float(args.epochs), best_a0, nll_opt=best_nll))

        fit_with_d = fit_dimension_aware_alpha0(observations, include_dim=True)
        fit_no_d = fit_dimension_aware_alpha0(observations, include_dim=False)
        lodo_with_d = evaluate_lodo_d(observations, include_dim=True)
        lodo_no_d = evaluate_lodo_d(observations, include_dim=False)
        loso_with_d = evaluate_loso_n(observations, include_dim=True)

        # Part B: Empirical B_knee sweep across (n, d, k) using predicted alpha_0(k, B, n, T, d)
        knee_obs = []
        batch_candidates = [16, 32, 64, 128, 256, 512]
        for n_b, d_b, k_b in itertools.product([1200, 2400, 4800], [8, 16, 32], [8, 16]):
            X_tr, _ = generate_stretched_gaussian(n=n_b, d=d_b, exponent=1.0, seed=args.seed + 500 + d_b)
            X_val, _ = generate_stretched_gaussian(n=500, d=d_b, exponent=1.0, seed=args.seed + 1500 + d_b)
            b_nlls = {}
            for B_cand in batch_candidates:
                a0_pred = fit_with_d.predict(k_b, B_cand, n_b, args.epochs, d_b)
                cfg = EGMMConfig(
                    num_components=k_b,
                    dim=d_b,
                    cov_type="diag",
                    cov_floor=1e-3,
                    init_prior_mass=max(1.0, 0.25 * B_cand / k_b),
                    alpha_0=a0_pred,
                    finalization="main_ema",
                    seed=args.seed,
                )
                res = NumPyEGMM(cfg).fit(X_tr, batch_size=B_cand, epochs=args.epochs)
                b_nlls[B_cand] = evaluate_gmm_nll(X_val, res["pi"], res["mu"], res["cov"], cov_type="diag")["nll"]
            min_nll = min(b_nlls.values())
            tol_nats = 0.005 * d_b  # +0.005 nats/dim knee threshold
            valid_bs = [b for b, v in b_nlls.items() if v <= min_nll + tol_nats]
            b_knee = max(valid_bs) if valid_bs else min(b_nlls, key=lambda b: b_nlls[b])
            knee_obs.append(
                DimensionScalingObservation(n_b, k_b, d_b, float(b_knee), float(args.epochs), 0.1, b_knee_opt=float(b_knee))
            )
        fit_b_knee = fit_dimension_aware_batch_knee(knee_obs, include_dim=True)

        # Zero-shot OOD transfer on StretchedGaussian at unseen (d=24, k=12, n=2400, B=64)
        X_ood_tr, _ = generate_stretched_gaussian(n=2400, d=24, exponent=1.0, seed=101)
        X_ood_te, _ = generate_stretched_gaussian(n=800, d=24, exponent=1.0, seed=102)
        pred_a0_d = fit_with_d.predict(12, 64, 2400, args.epochs, 24)
        pred_a0_nod = fit_no_d.predict(12, 64, 2400, args.epochs, 24)

        oracle_nll = float("inf")
        for a0 in alpha_grid:
            res_o = NumPyEGMM(
                EGMMConfig(
                    num_components=12,
                    dim=24,
                    cov_type="diag",
                    cov_floor=1e-3,
                    init_prior_mass=max(1.0, 0.25 * 64 / 12),
                    alpha_0=a0,
                    finalization="main_ema",
                    seed=101,
                )
            ).fit(X_ood_tr, batch_size=64, epochs=args.epochs)
            oracle_nll = min(oracle_nll, evaluate_gmm_nll(X_ood_te, res_o["pi"], res_o["mu"], res_o["cov"])["nll"])

        res_law_d = NumPyEGMM(
            EGMMConfig(
                num_components=12,
                dim=24,
                cov_type="diag",
                cov_floor=1e-3,
                init_prior_mass=max(1.0, 0.25 * 64 / 12),
                alpha_0=pred_a0_d,
                finalization="main_ema",
                seed=101,
            )
        ).fit(X_ood_tr, batch_size=64, epochs=args.epochs)
        nll_law_d = evaluate_gmm_nll(X_ood_te, res_law_d["pi"], res_law_d["mu"], res_law_d["cov"])["nll"]

        res_law_nod = NumPyEGMM(
            EGMMConfig(
                num_components=12,
                dim=24,
                cov_type="diag",
                cov_floor=1e-3,
                init_prior_mass=max(1.0, 0.25 * 64 / 12),
                alpha_0=pred_a0_nod,
                finalization="main_ema",
                seed=101,
            )
        ).fit(X_ood_tr, batch_size=64, epochs=args.epochs)
        nll_law_nod = evaluate_gmm_nll(X_ood_te, res_law_nod["pi"], res_law_nod["mu"], res_law_nod["cov"])["nll"]

        reg_d = compute_nll_regret(nll_law_d, oracle_nll)
        reg_nod = compute_nll_regret(nll_law_nod, oracle_nll)

        payload = {
            "poc_id": "poc_03",
            "title": "Empirical Dimension-Aware Scaling Law Sweep (alpha_0 & B_knee) + Unseen-d Transfer",
            "num_observations": len(observations),
            "fit_with_d": {
                "c_alpha": round(fit_with_d.c_alpha, 4),
                "a_k": round(fit_with_d.a_k, 4),
                "a_b": round(fit_with_d.a_b, 4),
                "a_n": round(fit_with_d.a_n, 4),
                "a_t": round(fit_with_d.a_t, 4),
                "a_d": round(fit_with_d.a_d, 4),
                "r2_log": round(fit_with_d.r2_log, 4),
                "lodo_mean_rel_err_pct": round(lodo_with_d["lodo_mean_rel_err"] * 100.0, 2),
                "loso_mean_rel_err_pct": round(loso_with_d["loso_mean_rel_err"] * 100.0, 2),
            },
            "fit_no_d": {
                "r2_log": round(fit_no_d.r2_log, 4),
                "lodo_mean_rel_err_pct": round(lodo_no_d["lodo_mean_rel_err"] * 100.0, 2),
            },
            "fit_batch_knee": {
                "num_observations": len(knee_obs),
                "c_b": round(fit_b_knee.c_b, 4),
                "b_k": round(fit_b_knee.b_k, 4),
                "b_n": round(fit_b_knee.b_n, 4),
                "b_d": round(fit_b_knee.b_d, 4),
                "r2_log": round(fit_b_knee.r2_log, 4),
            },
            "ood_unseen_d_transfer": {
                "oracle_nll": round(oracle_nll, 4),
                "with_d_pred_alpha0": round(pred_a0_d, 4),
                "with_d_nll": round(nll_law_d, 4),
                "with_d_gap_nats": round(reg_d["nll_gap_nats"], 4),
                "no_d_pred_alpha0": round(pred_a0_nod, 4),
                "no_d_nll": round(nll_law_nod, 4),
                "no_d_gap_nats": round(reg_nod["nll_gap_nats"], 4),
            },
        }
        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_03_results.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 3 COMPLETE ===")
        fw = payload["fit_with_d"]
        fb = payload["fit_batch_knee"]
        print(
            f"  alpha_0 (with d): {fw['c_alpha']:.3f} * k^{fw['a_k']:+.2f} B^{fw['a_b']:+.2f} "
            f"n^{fw['a_n']:+.2f} d^{fw['a_d']:+.2f} | R^2={fw['r2_log']:.4f}"
        )
        print(
            f"  B_knee  (with d): {fb['c_b']:.3f} * k^{fb['b_k']:+.2f} n^{fb['b_n']:+.2f} "
            f"d^{fb['b_d']:+.2f} | R^2={fb['r2_log']:.4f}"
        )
        print(
            f"  LODO rel err: with d = {fw['lodo_mean_rel_err_pct']}% vs no d = {payload['fit_no_d']['lodo_mean_rel_err_pct']}%"
        )
        print(f"  LOSO rel err: {fw['loso_mean_rel_err_pct']}%")
        print(
            f"  Unseen d=24 : oracle={oracle_nll:.4f} | with d={nll_law_d:.4f} ({reg_d['nll_gap_nats']:+.4f} nats) "
            f"| no d={nll_law_nod:.4f} ({reg_nod['nll_gap_nats']:+.4f} nats)"
        )
        print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
