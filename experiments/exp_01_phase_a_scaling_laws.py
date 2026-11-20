"""
Experiment 01 (Phase A): Dimension-Aware Scaling Laws (alpha_0 & B_knee) + OOD Transfer
========================================================================================

RESEARCH HYPOTHESES TESTED:
  - H1: Including ambient dimension d^{a_d} in log-linear OLS scaling laws for alpha_0
        yields a_d < 0, raises R^2 over the d-agnostic law, and reduces LODO-d error >= 1.5x.
  - H2: Laws fitted on proxy distributions (`DEEP` / `StretchedGaussian` + `AnisotropicGMM`)
        transfer zero-shot to unseen (d, n) and OOD datasets (`SIFT`, `SPACEV`, `Zipf-GMM`)
        with minimal NLL regret relative to exhaustive grid-search oracle.

COMPUTE & HARDWARE REQUIREMENTS:
  - `--mode smoke` : Any CPU, ~10-25 seconds, $0 (NumPy).
  - `--mode full`  : 1x NVIDIA A100/H100 (80GB), ~80 GPU-hours across d in {32, 96, 256, 768},
                     k in {64, 512, 4096, 16384}, n in {1e5, 1e6, 1e7, 1e8}.
                     Requires `--data-dir` with `deep_train.fvecs`, `sift_train.fvecs`,
                     `spacev_train.bvecs`, or falls back to scaled StretchedGaussian proxies.

HOW TO RUN:
  python experiments/exp_01_phase_a_scaling_laws.py --dry-run
  python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --seeds 2
  python experiments/exp_01_phase_a_scaling_laws.py --mode full --backend torch --data-dir /data/egmm --seeds 5

SAVED ARTIFACTS:
  - Telemetry : `experiments/results/exp_01_phase_a_scaling_laws.json`
  - Log       : `experiments/logs/exp_01_phase_a/latest.log`
"""

import argparse
import itertools
import json
import pathlib
import sys
import time
import numpy as np

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import (  # noqa: E402
    generate_gaussian_mixture,
    generate_stretched_gaussian,
    load_vector_memmap,
)
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from e_gmm.scaling_law import (  # noqa: E402
    DimensionScalingObservation,
    compute_nll_regret,
    evaluate_lodo_d,
    evaluate_loso_n,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
)
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 01 (Phase A): Dimension-Aware Scaling-Law Sweeps")
    p.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--data-dir", type=str, default=None, help="Optional directory containing DEEP/SIFT/SPACEV memmaps.")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _fit_and_eval_nll(X_tr: np.ndarray, X_val: np.ndarray, cfg: EGMMConfig, B: int, epochs: int, backend: str) -> float:
    if backend == "torch":
        import torch
        from e_gmm.torch_egmm import TorchEGMM

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = TorchEGMM(cfg).to(dev)
        res = m.fit(torch.from_numpy(X_tr).to(dev), batch_size=B, epochs=epochs)
        return evaluate_gmm_nll(X_val, res["pi"], res["mu"], res["cov"], cov_type=cfg.cov_type)["nll"]
    res = NumPyEGMM(cfg).fit(X_tr, batch_size=B, epochs=epochs)
    return evaluate_gmm_nll(X_val, res["pi"], res["mu"], res["cov"], cov_type=cfg.cov_type)["nll"]


def _load_or_generate_split(data_dir: str | None, name: str, n: int, d: int, seed: int):
    if data_dir is not None:
        p = pathlib.Path(data_dir) / name
        if p.exists():
            arr = np.asarray(load_vector_memmap(str(p), max_rows=n + 5000), dtype=np.float32)
            if arr.shape[1] >= d and arr.shape[0] > 5000:
                return arr[:n, :d], arr[n : n + 5000, :d]
    X_tr, _ = generate_stretched_gaussian(n=n, d=d, exponent=1.0, seed=seed)
    X_va, _ = generate_stretched_gaussian(n=min(1000, max(500, n // 4)), d=d, exponent=1.0, seed=seed + 999)
    return X_tr, X_va


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else f"{args.mode}_{args.backend}"
    with tee_experiment_logs("exp_01_phase_a", run_tag=tag):
        if args.mode == "smoke":
            dims = [8, 16, 32, 64]
            ks = [4, 8, 16]
            ns = [1200, 2400, 4800]
            bs = [32, 128]
            alpha_grid = [0.01, 0.02, 0.04, 0.08, 0.14, 0.24, 0.40, 0.65, 0.90]
            knee_ns = [1200, 2400, 4800]
            knee_ds = [8, 16, 32]
            knee_ks = [8, 16]
            knee_bs = [16, 32, 64, 128, 256, 512]
        else:
            dims = [32, 96, 256, 768]
            ks = [64, 512, 4096, 16384]
            ns = [100_000, 1_000_000, 10_000_000]
            bs = [4096, 16384, 65536]
            alpha_grid = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.90]
            knee_ns = [100_000, 1_000_000, 10_000_000]
            knee_ds = [32, 96, 256]
            knee_ks = [64, 512, 4096]
            knee_bs = [1024, 4096, 16384, 65536, 131072]

        grid = [
            (n, d, k, B)
            for (n, d, k, B) in itertools.product(ns, dims, ks, bs)
            if B <= n and (n // k) >= 8
        ]

        if args.dry_run:
            print("=== EXP 01 (PHASE A) DRY RUN ===")
            print(f"Mode: {args.mode} | Backend: {args.backend} | Seeds: {args.seeds}")
            print(f"Alpha_0 Sweep Grid: {len(grid)} configs x {len(alpha_grid)} alphas x {args.seeds} seeds = {len(grid) * len(alpha_grid) * args.seeds} runs")
            print(f"B_knee Sweep Grid : {len(knee_ns) * len(knee_ds) * len(knee_ks)} configs x {len(knee_bs)} batches")
            return

        t_start = time.perf_counter()
        observations = []
        for n, d, k, B in grid:
            vals = {a0: [] for a0 in alpha_grid}
            for s in range(1, args.seeds + 1):
                X_tr, X_val = _load_or_generate_split(args.data_dir, "deep_train.fvecs", n, d, args.seed + s * 100 + d)
                for a0 in alpha_grid:
                    cfg = EGMMConfig(
                        num_components=k,
                        dim=d,
                        cov_type="diag",
                        cov_floor=1e-3,
                        init_prior_mass=max(1.0, 0.25 * B / k),
                        alpha_0=a0,
                        top_m=16 if k >= 64 else None,
                        finalization="main_ema",
                        seed=args.seed + s,
                    )
                    vals[a0].append(_fit_and_eval_nll(X_tr, X_val, cfg, B, args.epochs, args.backend))
            best_a0 = min(alpha_grid, key=lambda a: float(np.mean(vals[a])))
            best_nll = float(np.mean(vals[best_a0]))
            observations.append(DimensionScalingObservation(n, k, d, B, float(args.epochs), best_a0, nll_opt=best_nll))

        fit_with_d = fit_dimension_aware_alpha0(observations, include_dim=True)
        fit_no_d = fit_dimension_aware_alpha0(observations, include_dim=False)
        lodo_with_d = evaluate_lodo_d(observations, include_dim=True)
        lodo_no_d = evaluate_lodo_d(observations, include_dim=False)
        loso_with_d = evaluate_loso_n(observations, include_dim=True)

        # Empirical B_knee sweep
        knee_obs = []
        for n_b, d_b, k_b in itertools.product(knee_ns, knee_ds, knee_ks):
            X_tr, X_val = _load_or_generate_split(args.data_dir, "deep_train.fvecs", n_b, d_b, args.seed + 500 + d_b)
            b_nlls = {}
            for B_cand in knee_bs:
                if B_cand > n_b:
                    continue
                a0_pred = fit_with_d.predict(k_b, B_cand, n_b, args.epochs, d_b)
                cfg = EGMMConfig(
                    num_components=k_b,
                    dim=d_b,
                    cov_type="diag",
                    cov_floor=1e-3,
                    init_prior_mass=max(1.0, 0.25 * B_cand / k_b),
                    alpha_0=a0_pred,
                    top_m=16 if k_b >= 64 else None,
                    finalization="main_ema",
                    seed=args.seed,
                )
                b_nlls[B_cand] = _fit_and_eval_nll(X_tr, X_val, cfg, B_cand, args.epochs, args.backend)
            min_nll = min(b_nlls.values())
            tol_nats = 0.005 * d_b
            valid_bs = [b for b, v in b_nlls.items() if v <= min_nll + tol_nats]
            b_knee = max(valid_bs) if valid_bs else min(b_nlls, key=lambda b: b_nlls[b])
            knee_obs.append(DimensionScalingObservation(n_b, k_b, d_b, float(b_knee), float(args.epochs), 0.1, b_knee_opt=float(b_knee)))

        fit_b = fit_dimension_aware_batch_knee(knee_obs, include_dim=True)

        # Zero-shot OOD transfer on unseen (d=24, Zipf-Anisotropic GMM / SIFT fallback)
        ood_d, ood_k, ood_n, ood_B = (24, 12, 2400, 64) if args.mode == "smoke" else (128, 1024, 500_000, 16384)
        X_ood_all, _, _ = generate_gaussian_mixture(
            n=ood_n + 800, d=ood_d, num_true_clusters=ood_k, zipf=True, anisotropic=True, seed=909
        )
        X_ood_tr, X_ood_te = X_ood_all[:ood_n], X_ood_all[ood_n:]
        pred_d = fit_with_d.predict(ood_k, ood_B, ood_n, args.epochs, ood_d)
        pred_nod = fit_no_d.predict(ood_k, ood_B, ood_n, args.epochs, ood_d)

        oracle_nll = float("inf")
        for a0 in alpha_grid:
            cfg_o = EGMMConfig(num_components=ood_k, dim=ood_d, cov_type="diag", cov_floor=1e-3, init_prior_mass=max(1.0, 0.25 * ood_B / ood_k), alpha_0=a0, finalization="main_ema", seed=909)
            oracle_nll = min(oracle_nll, _fit_and_eval_nll(X_ood_tr, X_ood_te, cfg_o, ood_B, args.epochs, args.backend))

        nll_d = _fit_and_eval_nll(
            X_ood_tr,
            X_ood_te,
            EGMMConfig(num_components=ood_k, dim=ood_d, cov_type="diag", cov_floor=1e-3, init_prior_mass=max(1.0, 0.25 * ood_B / ood_k), alpha_0=pred_d, finalization="main_ema", seed=909),
            ood_B,
            args.epochs,
            args.backend,
        )
        nll_nod = _fit_and_eval_nll(
            X_ood_tr,
            X_ood_te,
            EGMMConfig(num_components=ood_k, dim=ood_d, cov_type="diag", cov_floor=1e-3, init_prior_mass=max(1.0, 0.25 * ood_B / ood_k), alpha_0=pred_nod, finalization="main_ema", seed=909),
            ood_B,
            args.epochs,
            args.backend,
        )

        reg_d = compute_nll_regret(nll_d, oracle_nll)
        reg_nod = compute_nll_regret(nll_nod, oracle_nll)
        elapsed_s = round(time.perf_counter() - t_start, 2)

        payload = {
            "experiment_id": "exp_01_phase_a_scaling_laws",
            "mode": args.mode,
            "backend": args.backend,
            "elapsed_sec": elapsed_s,
            "num_alpha_observations": len(observations),
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
                "c_b": round(fit_b.c_b, 4),
                "b_k": round(fit_b.b_k, 4),
                "b_n": round(fit_b.b_n, 4),
                "b_d": round(fit_b.b_d, 4),
                "r2_log": round(fit_b.r2_log, 4),
            },
            "ood_transfer": {
                "dim": ood_d,
                "k": ood_k,
                "oracle_nll": round(oracle_nll, 4),
                "with_d_pred_alpha0": round(pred_d, 4),
                "with_d_nll": round(nll_d, 4),
                "with_d_gap_nats": round(reg_d["nll_gap_nats"], 4),
                "no_d_pred_alpha0": round(pred_nod, 4),
                "no_d_nll": round(nll_nod, 4),
                "no_d_gap_nats": round(reg_nod["nll_gap_nats"], 4),
            },
        }
        out_path = RESULTS_ROOT / "exp_01_phase_a_scaling_laws.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== EXP 01 (PHASE A) COMPLETE ===")
        fw = payload["fit_with_d"]
        print(f"  d-Aware alpha_0 : {fw['c_alpha']} * k^{fw['a_k']:+.2f} B^{fw['a_b']:+.2f} n^{fw['a_n']:+.2f} d^{fw['a_d']:+.2f} | R^2={fw['r2_log']} (no-d: {payload['fit_no_d']['r2_log']})")
        print(f"  LODO-d Rel Err  : {fw['lodo_mean_rel_err_pct']}% (with d) vs {payload['fit_no_d']['lodo_mean_rel_err_pct']}% (no d)")
        print(f"  OOD Zipf-GMM    : oracle={oracle_nll:.4f} | with-d={nll_d:.4f} ({reg_d['nll_gap_nats']:+.4f} nats)")
        print(f"Saved telemetry -> {out_path} ({elapsed_s}s)")


if __name__ == "__main__":
    main()
