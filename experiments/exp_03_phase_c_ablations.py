"""
Experiment 03 (Phase C): Comprehensive Ablations (H5, H6, H7)
==============================================================

RESEARCH HYPOTHESES TESTED:
  - H5: Spectral v1 donor splitting + variance_mass donor weighting outperforms
        no-revival, batch resampling, Faiss alternating +/- delta, and random orthogonal splits.
  - H6: Covariance floor lambda_0 balances conditioning vs. NLL bias; Low-Rank+Diag (Woodbury)
        captures rotated subspace correlations across rank r in {2, 4, 8, 12} with top BIC;
        Top-m sparse truncation preserves soft-EM NLL at m/k M-step cost.
  - H7: `last_epoch` exact EM wins Train NLL while passive `slow_ema` wins Test NLL
        across small-to-medium n/k regimes.

COMPUTE & HARDWARE REQUIREMENTS:
  - `--mode smoke` : Any CPU, ~10 seconds, $0.
  - `--mode full`  : 1x A100/H100 (80GB), ~50 GPU-hours across 5 seeds.

HOW TO RUN:
  python experiments/exp_03_phase_c_ablations.py --dry-run
  python experiments/exp_03_phase_c_ablations.py --mode smoke --seeds 3
  python experiments/exp_03_phase_c_ablations.py --mode full --backend torch --seeds 5

SAVED ARTIFACTS:
  - Telemetry : `experiments/results/exp_03_phase_c_ablations.json`
  - Log       : `experiments/logs/exp_03_phase_c/latest.log`
"""

import argparse
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
from e_gmm.datasets import generate_gaussian_mixture, generate_stretched_gaussian  # noqa: E402
from e_gmm.metrics import compute_gmm_health, evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 03 (Phase C): Systematic Ablations")
    p.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run_egmm(cfg: EGMMConfig, X_tr: np.ndarray, batch_size: int, epochs: int, backend: str) -> dict:
    if backend == "torch":
        import torch
        from e_gmm.torch_egmm import TorchEGMM

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = TorchEGMM(cfg).to(dev)
        res = m.fit(torch.from_numpy(X_tr).to(dev), batch_size=batch_size, epochs=epochs)
        health = compute_gmm_health(
            res["pi"], res["cov"], cov_type=cfg.cov_type, U=res["U"], cov_floor=cfg.cov_floor
        )
        return {
            "model": m,
            "pi": res["pi"],
            "mu": res["mu"],
            "cov": res["cov"],
            "U": res["U"],
            "health": health,
            "splits": res["splits"],
        }
    m = NumPyEGMM(cfg)
    res = m.fit(X_tr, batch_size=batch_size, epochs=epochs)
    return {
        "model": m,
        "pi": res["pi"],
        "mu": res["mu"],
        "cov": res["cov"],
        "U": res["U"],
        "health": res["health"],
        "splits": res["splits"],
    }


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else f"{args.mode}_{args.backend}"
    is_full = args.mode == "full"
    n_rev = 50_000 if is_full else 2400
    d_rev = 64 if is_full else 16
    k_rev = 64 if is_full else 16
    B_rev = 1024 if is_full else 64
    ep_rev = 6 if is_full else 5
    with tee_experiment_logs("exp_03_phase_c", run_tag=tag):
        if args.dry_run:
            print("=== EXP 03 (PHASE C) DRY RUN ===")
            print(
                f"Mode: {args.mode} | Backend: {args.backend} | "
                f"Ablations: Revival (6), CovFloor (4), LowRank r (5), Top-m (5), Finalization (3) x {args.seeds} seeds"
            )
            return

        t_start = time.perf_counter()

        # Ablation 1: Revival policy & donor weighting (H5) on Zipf-1.2 Rotated Full-Cov
        rev_configs = [
            ("none", "count"),
            ("batch_resample", "count"),
            ("faiss_alt", "count"),
            ("random_orth", "count"),
            ("spectral_v1", "count"),
            ("spectral_v1", "variance_mass"),
        ]
        rev_rows = []
        n_rev_tr = int(0.75 * n_rev)
        for strat, dweight in rev_configs:
            nlls, deads, splits_l = [], [], []
            for s in range(1, args.seeds + 1):
                X_all, _, _ = generate_gaussian_mixture(
                    n=n_rev, d=d_rev, num_true_clusters=k_rev, zipf=True, zipf_power=1.2, anisotropic=True, rotated=True, seed=s
                )
                cfg = EGMMConfig(
                    num_components=k_rev,
                    dim=d_rev,
                    cov_type="full",
                    cov_floor=1e-4,
                    alpha_0=0.35,
                    split_strategy=strat,
                    donor_weight_by=dweight,
                    finalization="last_epoch",
                    seed=s,
                )
                res = _run_egmm(cfg, X_all[:n_rev_tr], batch_size=B_rev, epochs=ep_rev, backend=args.backend)
                nlls.append(evaluate_gmm_nll(X_all[n_rev_tr:], res["pi"], res["mu"], res["cov"], cov_type="full")["nll"])
                deads.append(res["health"]["dead_count"])
                splits_l.append(res["splits"])
            rev_rows.append(
                {
                    "split_strategy": strat,
                    "donor_weight_by": dweight,
                    "test_nll_mean": round(float(np.mean(nlls)), 4),
                    "test_nll_std": round(float(np.std(nlls)), 4),
                    "dead_mean": round(float(np.mean(deads)), 2),
                    "splits_mean": round(float(np.mean(splits_l)), 1),
                }
            )

        # Ablation 2: Covariance floor lambda_0 sensitivity (H6)
        floor_rows = []
        n_fl = 20_000 if is_full else 1200
        d_fl = 64 if is_full else 24
        k_fl = 64 if is_full else 8
        B_fl = 512 if is_full else 64
        for fl in (1e-6, 1e-4, 1e-2, 1e-1):
            nlls, conds, fhits = [], [], []
            for s in range(1, args.seeds + 1):
                X_tr, _ = generate_stretched_gaussian(n=n_fl, d=d_fl, exponent=2.0, seed=s + 10)
                X_te, _ = generate_stretched_gaussian(n=min(4000, n_fl // 2), d=d_fl, exponent=2.0, seed=s + 910)
                res = _run_egmm(
                    EGMMConfig(num_components=k_fl, dim=d_fl, cov_type="diag", cov_floor=fl, alpha_0=0.3, seed=s),
                    X_tr,
                    batch_size=B_fl,
                    epochs=4,
                    backend=args.backend,
                )
                nlls.append(evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"])["nll"])
                conds.append(res["health"]["max_condition_number"])
                fhits.append(res["health"]["floor_hit_rate"])
            floor_rows.append(
                {
                    "cov_floor": fl,
                    "test_nll_mean": round(float(np.mean(nlls)), 4),
                    "max_cond_mean": round(float(np.mean(conds)), 1),
                    "floor_hit_pct": round(float(np.mean(fhits)) * 100.0, 2),
                }
            )

        # Ablation 3: Covariance structure & rank r sweep (diag vs low_rank_diag r in {2,4,8} vs full) (H6)
        cov_rows = []
        n_cr = 30_000 if is_full else 2400
        d_cr = 64 if is_full else 16
        k_cr = 32 if is_full else 8
        n_cr_tr = int(0.75 * n_cr)
        ranks = [("diag", "diag", 0), ("low_rank_r2", "low_rank_diag", 2), ("low_rank_r4", "low_rank_diag", 4), ("low_rank_r8", "low_rank_diag", 8)]
        if is_full:
            ranks.append(("low_rank_r12", "low_rank_diag", 12))
        ranks.append(("full", "full", d_cr))
        for label, ctype, r_rank in ranks:
            nlls, bics, params_l = [], [], []
            for s in range(1, args.seeds + 1):
                X_all, _, _ = generate_gaussian_mixture(
                    n=n_cr, d=d_cr, num_true_clusters=k_cr, anisotropic=True, rotated=True, seed=s + 20
                )
                cfg = EGMMConfig(num_components=k_cr, dim=d_cr, cov_type=ctype, low_rank=max(r_rank, 1), alpha_0=0.35, seed=s)
                res = _run_egmm(cfg, X_all[:n_cr_tr], batch_size=512 if is_full else 128, epochs=4, backend=args.backend)
                ev = evaluate_gmm_nll(X_all[n_cr_tr:], res["pi"], res["mu"], res["cov"], cov_type=ctype, U=res["U"])
                nlls.append(ev["nll"])
                bics.append(ev["bic"])
                params_l.append(ev["n_effective_params"])
            cov_rows.append(
                {
                    "label": label,
                    "cov_type": ctype,
                    "rank": r_rank,
                    "n_params": int(params_l[0]),
                    "test_nll_mean": round(float(np.mean(nlls)), 4),
                    "test_bic_mean": round(float(np.mean(bics)), 1),
                }
            )

        # Ablation 4: Top-m sparse E-step truncation (H6)
        topm_rows = []
        n_tm = 30_000 if is_full else 2400
        d_tm = 64 if is_full else 20
        k_tm = 128 if is_full else 32
        n_tm_tr = int(0.75 * n_tm)
        for tm in (2, 4, 8, 16, None):
            nlls = []
            for s in range(1, args.seeds + 1):
                X_all, _, _ = generate_gaussian_mixture(n=n_tm, d=d_tm, num_true_clusters=k_tm, anisotropic=True, seed=s + 30)
                res = _run_egmm(
                    EGMMConfig(num_components=k_tm, dim=d_tm, cov_type="diag", top_m=tm, alpha_0=0.4, seed=s),
                    X_all[:n_tm_tr],
                    batch_size=512 if is_full else 128,
                    epochs=4,
                    backend=args.backend,
                )
                nlls.append(evaluate_gmm_nll(X_all[n_tm_tr:], res["pi"], res["mu"], res["cov"])["nll"])
            topm_rows.append(
                {
                    "top_m": "dense" if tm is None else tm,
                    "m_step_work_ratio": 1.0 if tm is None else round(tm / float(k_tm), 4),
                    "test_nll_mean": round(float(np.mean(nlls)), 4),
                }
            )

        # Ablation 5: Finalization strategy across n/k regimes (H7)
        fin_rows = []
        n_fn = 20_000 if is_full else 1000
        d_fn = 64 if is_full else 20
        k_fn = 256 if is_full else 24
        for fin_mode in ("main_ema", "last_epoch", "slow_ema"):
            tr_nlls, te_nlls = [], []
            for s in range(1, args.seeds + 1):
                X_tr, _ = generate_stretched_gaussian(n=n_fn, d=d_fn, exponent=1.0, seed=s + 40)
                X_te, _ = generate_stretched_gaussian(n=min(4000, n_fn), d=d_fn, exponent=1.0, seed=s + 1040)
                res = _run_egmm(
                    EGMMConfig(num_components=k_fn, dim=d_fn, cov_type="diag", alpha_0=0.45, slow_ema_alpha=0.05, seed=s),
                    X_tr,
                    batch_size=512 if is_full else 64,
                    epochs=4,
                    backend=args.backend,
                )
                pi, mu, cov = res["model"].readout(fin_mode)[:3]
                tr_nlls.append(evaluate_gmm_nll(X_tr, pi, mu, cov)["nll"])
                te_nlls.append(evaluate_gmm_nll(X_te, pi, mu, cov)["nll"])
            fin_rows.append(
                {
                    "finalizer": fin_mode,
                    "train_nll_mean": round(float(np.mean(tr_nlls)), 4),
                    "test_nll_mean": round(float(np.mean(te_nlls)), 4),
                }
            )

        elapsed_s = round(time.perf_counter() - t_start, 2)
        payload = {
            "experiment_id": "exp_03_phase_c_ablations",
            "mode": args.mode,
            "seeds": args.seeds,
            "elapsed_sec": elapsed_s,
            "revival_ablation": rev_rows,
            "cov_floor_ablation": floor_rows,
            "cov_rank_ablation": cov_rows,
            "top_m_ablation": topm_rows,
            "finalization_ablation": fin_rows,
        }
        out_path = RESULTS_ROOT / "exp_03_phase_c_ablations.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== EXP 03 (PHASE C) COMPLETE ===")
        print("  Revival      :", [(f"{r['split_strategy']}:{r['donor_weight_by']}", r["test_nll_mean"], r["dead_mean"]) for r in rev_rows])
        print("  Cov/Rank     :", [(r["label"], r["test_nll_mean"], r["test_bic_mean"]) for r in cov_rows])
        print("  Top-m        :", [(r["top_m"], r["test_nll_mean"]) for r in topm_rows])
        print("  Finalization :", [(r["finalizer"], r["train_nll_mean"], r["test_nll_mean"]) for r in fin_rows])
        print(f"Saved telemetry -> {out_path} ({elapsed_s}s)")


if __name__ == "__main__":
    main()
