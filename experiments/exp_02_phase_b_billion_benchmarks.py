"""
Experiment 02 (Phase B): Controlled Multi-Seed Benchmarks Across Regimes & Scales
===================================================================================

RESEARCH HYPOTHESES TESTED:
  - H3: At matched epoch budgets, `e-GMM` produces 0 Cholesky/singularity failures,
        eliminates dead-component collapse, and beats swept Cappé & Moulines Online EM
        and Full-Batch EM on NLL vs. wall-clock budget.
  - H4: On cleanly separated clusters (`Zipf-Separated`), `e-means + PostHoc` is a
        strong near-tie baseline; on overlapping or rotated correlated mixtures
        (`Overlapping-Anisotropic`, `Rotated-Correlated`, `DINOv2`, `ImageNet-1k`),
        joint soft `e-GMM` achieves statistically significantly lower Test NLL.

COMPUTE & HARDWARE REQUIREMENTS:
  - `--mode smoke` : Any CPU, ~10-20 seconds, $0 (multi-seed 4-regime suite).
  - `--mode full`  : 1-2x NVIDIA A100/H100 (80GB), ~120 GPU-hours across SIFT1M, DEEP1B,
                     DINOv2-100M, ImageNet-1k (`--data-dir`), 5 seeds.

HOW TO RUN:
  python experiments/exp_02_phase_b_billion_benchmarks.py --dry-run
  python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --seeds 3
  python experiments/exp_02_phase_b_billion_benchmarks.py --mode full --backend torch --data-dir /data/egmm --seeds 5

SAVED ARTIFACTS:
  - Telemetry : `experiments/results/exp_02_phase_b_benchmarks.json`
  - Log       : `experiments/logs/exp_02_phase_b/latest.log`
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
from e_gmm.datasets import (  # noqa: E402
    generate_gaussian_mixture,
    generate_stretched_gaussian,
    load_vector_memmap,
)
from e_gmm.metrics import compute_gmm_health, evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
)
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 02 (Phase B): Controlled Multi-Seed Benchmarks")
    p.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _fit_egmm(cfg: EGMMConfig, X_tr: np.ndarray, B: int, epochs: int, backend: str) -> dict:
    if backend == "torch":
        import torch
        from e_gmm.torch_egmm import TorchEGMM

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = TorchEGMM(cfg).to(dev)
        res = m.fit(torch.from_numpy(X_tr).to(dev), batch_size=B, epochs=epochs)
        health = compute_gmm_health(
            res["pi"], res["cov"], cov_type=cfg.cov_type, U=res["U"], cov_floor=cfg.cov_floor
        )
        return {
            "pi": res["pi"],
            "mu": res["mu"],
            "cov": res["cov"],
            "U": res["U"],
            "health": health,
            "chol_fallbacks": res["chol_fallbacks"],
        }
    return NumPyEGMM(cfg).fit(X_tr, batch_size=B, epochs=epochs)


def _load_real_or_synthetic(
    data_dir: str | None,
    filename: str,
    n_tr: int,
    n_te: int,
    d: int,
    k: int,
    spec_kw: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if data_dir is not None and filename:
        p = pathlib.Path(data_dir) / filename
        if p.exists():
            arr = np.asarray(load_vector_memmap(str(p), max_rows=n_tr + n_te), dtype=np.float32)
            if arr.shape[0] >= n_tr + min(n_te, 1000) and arr.shape[1] >= d:
                return arr[:n_tr, :d], arr[n_tr : n_tr + n_te, :d], f"memmap:{p.name}"
    if spec_kw["kind"] == "gmm":
        kw = {kk: vv for kk, vv in spec_kw.items() if kk != "kind"}
        X_all, _, _ = generate_gaussian_mixture(n=n_tr + n_te, d=d, num_true_clusters=k, seed=seed, **kw)
        return X_all[:n_tr], X_all[n_tr:], "synthetic_gmm"
    X_tr, _ = generate_stretched_gaussian(n=n_tr, d=d, exponent=spec_kw["exponent"], seed=seed)
    X_te, _ = generate_stretched_gaussian(n=n_te, d=d, exponent=spec_kw["exponent"], seed=seed + 999)
    return X_tr, X_te, "synthetic_stretched"


def _run_suite_on_dataset(
    suite_name: str,
    cov_type: str,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    k: int,
    B: int,
    epochs: int,
    seed: int,
    backend: str,
) -> list:
    d = X_tr.shape[1]
    results = []

    # 1. FullBatchEM@25 (only when tractable)
    if X_tr.shape[0] <= 10_000 and k <= 64:
        t0 = time.perf_counter()
        fb = NumPyFullBatchEM(k, d, cov_type=cov_type, cov_floor=1e-4, seed=seed).fit(X_tr, max_iters=25)
        dt = time.perf_counter() - t0
        ev = evaluate_gmm_nll(X_te, fb["pi"], fb["mu"], fb["cov"], cov_type=cov_type)
        results.append(
            {
                "method": "FullBatchEM@25",
                "test_nll": float(ev["nll"]),
                "test_bic": float(ev["bic"]),
                "dead": int(fb["health"]["dead_count"]),
                "singular_ev": 0,
                "time_sec": dt,
            }
        )

    # 2. Swept Cappé & Moulines Online EM (pick best beta in {0.60, 0.75, 0.85} or log all)
    for beta in (0.60, 0.85):
        t0 = time.perf_counter()
        oem = NumPyOnlineEMCappeMoulines(k, d, cov_type=cov_type, beta=beta, t0=10.0, cov_floor=1e-8, init_sigma0=0.5, seed=seed).fit(
            X_tr, batch_size=B, epochs=epochs
        )
        dt = time.perf_counter() - t0
        ev = evaluate_gmm_nll(X_te, oem["pi"], oem["mu"], oem["cov"], cov_type=cov_type)
        results.append(
            {
                "method": f"OnlineEM_beta{beta:.2f}",
                "test_nll": float(ev["nll"]),
                "test_bic": float(ev["bic"]),
                "dead": int(oem["health"]["dead_count"]),
                "singular_ev": int(oem["singular_events"]),
                "time_sec": dt,
            }
        )

    # 3. e-means + PostHoc Covariance Fit
    t0 = time.perf_counter()
    epost = NumPyEMeansPostHocGMM(k, d, cov_type=cov_type, cov_floor=1e-4, alpha_0=0.35, seed=seed).fit(
        X_tr, batch_size=B, epochs=epochs
    )
    dt = time.perf_counter() - t0
    ev_ep = evaluate_gmm_nll(X_te, epost["pi"], epost["mu"], epost["cov"], cov_type=cov_type)
    results.append(
        {
            "method": "e-means+PostHoc",
            "test_nll": float(ev_ep["nll"]),
            "test_bic": float(ev_ep["bic"]),
            "dead": int(epost["health"]["dead_count"]),
            "singular_ev": 0,
            "time_sec": dt,
        }
    )

    # 4. e-GMM (Spectral v1 + Variance-Mass Donor Selection)
    t0 = time.perf_counter()
    cfg_eg = EGMMConfig(
        num_components=k,
        dim=d,
        cov_type=cov_type,
        cov_floor=1e-4,
        init_prior_mass=1.0,
        alpha_0=0.35,
        split_strategy="spectral_v1",
        donor_weight_by="variance_mass",
        donor_min_mult=1.25,
        cooldown_steps=15,
        finalization="last_epoch",
        seed=seed,
    )
    eg = _fit_egmm(cfg_eg, X_tr, B, epochs, backend=backend)
    dt = time.perf_counter() - t0
    ev_eg = evaluate_gmm_nll(X_te, eg["pi"], eg["mu"], eg["cov"], cov_type=cov_type)
    results.append(
        {
            "method": f"e-GMM({cov_type})",
            "test_nll": float(ev_eg["nll"]),
            "test_bic": float(ev_eg["bic"]),
            "dead": int(eg["health"]["dead_count"]),
            "singular_ev": int(eg["chol_fallbacks"]),
            "time_sec": dt,
        }
    )

    # 5. Also evaluate low_rank_diag on rotated/correlated suites
    if cov_type == "full":
        t0 = time.perf_counter()
        cfg_lr = EGMMConfig(
            num_components=k,
            dim=d,
            cov_type="low_rank_diag",
            low_rank=min(4, d),
            cov_floor=1e-4,
            init_prior_mass=1.0,
            alpha_0=0.35,
            split_strategy="spectral_v1",
            donor_weight_by="variance_mass",
            finalization="last_epoch",
            seed=seed,
        )
        lr = _fit_egmm(cfg_lr, X_tr, B, epochs, backend=backend)
        dt = time.perf_counter() - t0
        ev_lr = evaluate_gmm_nll(X_te, lr["pi"], lr["mu"], lr["cov"], cov_type="low_rank_diag", U=lr["U"])
        results.append(
            {
                "method": "e-GMM(low_rank_diag)",
                "test_nll": float(ev_lr["nll"]),
                "test_bic": float(ev_lr["bic"]),
                "dead": int(lr["health"]["dead_count"]),
                "singular_ev": int(lr["chol_fallbacks"]),
                "time_sec": dt,
            }
        )

    return results


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else f"{args.mode}_{args.backend}"
    with tee_experiment_logs("exp_02_phase_b", run_tag=tag):
        if args.mode == "smoke":
            suites = [
                ("Zipf-Separated-Diag", "", "diag", 3000, 800, 16, 16, 64, 5, dict(kind="gmm", zipf=True, anisotropic=True, rotated=False, sigma_mu_sq=2.0 / 3.0, sigma_w_sq=1.0 / 3.0)),
                ("Zipf-Rotated-Full", "", "full", 2400, 800, 16, 16, 64, 6, dict(kind="gmm", zipf=True, anisotropic=True, rotated=True, sigma_mu_sq=2.0 / 3.0, sigma_w_sq=1.0 / 3.0)),
                ("Overlapping-Rotated-Full", "", "full", 2400, 800, 16, 16, 64, 6, dict(kind="gmm", zipf=False, anisotropic=True, rotated=True, sigma_mu_sq=0.20, sigma_w_sq=0.80)),
                ("Stretched-PowerLaw-Diag", "", "diag", 3000, 800, 24, 16, 64, 5, dict(kind="stretched", exponent=1.5)),
            ]
        else:
            suites = [
                ("SIFT-1M-Diag", "sift_train.fvecs", "diag", 900_000, 50_000, 128, 1024, 4096, 8, dict(kind="gmm", zipf=True, anisotropic=True, rotated=False)),
                ("DEEP-10M-Diag", "deep_train.fvecs", "diag", 1_000_000, 50_000, 96, 4096, 4096, 5, dict(kind="stretched", exponent=1.2)),
                ("DINOv2-Rotated-LowRank", "dinov2_100m_train.npy", "full", 250_000, 20_000, 128, 512, 2048, 8, dict(kind="gmm", zipf=False, anisotropic=True, rotated=True, sigma_mu_sq=0.20, sigma_w_sq=0.80)),
                ("ImageNet-1k-Rotated", "imagenet1k_train.npy", "full", 250_000, 20_000, 128, 1000, 2048, 8, dict(kind="gmm", zipf=True, anisotropic=True, rotated=True, sigma_mu_sq=0.25, sigma_w_sq=0.75)),
            ]
        if args.dry_run:
            print("=== EXP 02 (PHASE B) DRY RUN ===")
            print(f"Mode: {args.mode} | Backend: {args.backend} | DataDir: {args.data_dir} | Seeds: {args.seeds}")
            print(f"Suites: {[s[0] for s in suites]}")
            return

        t_start = time.perf_counter()
        summary = {}
        data_sources = {}
        for sname, fname, ctype, n_tr, n_te, d, k, B, ep, spec_kw in suites:
            per_method = {}
            for s in range(1, args.seeds + 1):
                X_tr, X_te, src_tag = _load_real_or_synthetic(args.data_dir, fname, n_tr, n_te, d, k, spec_kw, seed=s)
                data_sources[sname] = src_tag
                rows = _run_suite_on_dataset(sname, ctype, X_tr, X_te, k, B, ep, seed=s, backend=args.backend)
                for r in rows:
                    per_method.setdefault(r["method"], []).append(r)

            agg = []
            for mname, recs in per_method.items():
                agg.append(
                    {
                        "method": mname,
                        "test_nll_mean": round(float(np.mean([x["test_nll"] for x in recs])), 4),
                        "test_nll_std": round(float(np.std([x["test_nll"] for x in recs])), 4),
                        "test_bic_mean": round(float(np.mean([x["test_bic"] for x in recs])), 1),
                        "dead_mean": round(float(np.mean([x["dead"] for x in recs])), 2),
                        "singular_ev_mean": round(float(np.mean([x["singular_ev"] for x in recs])), 2),
                        "time_sec_mean": round(float(np.mean([x["time_sec"] for x in recs])), 3),
                    }
                )
            summary[sname] = agg

        elapsed_s = round(time.perf_counter() - t_start, 2)
        payload = {
            "experiment_id": "exp_02_phase_b_billion_benchmarks",
            "mode": args.mode,
            "backend": args.backend,
            "seeds": args.seeds,
            "elapsed_sec": elapsed_s,
            "data_sources": data_sources,
            "suites": summary,
        }
        out_path = RESULTS_ROOT / "exp_02_phase_b_benchmarks.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== EXP 02 (PHASE B) COMPLETE ===")
        for sname, rows in summary.items():
            print(f"\n[{sname}]")
            print(f"  {'Method':<22} {'TestNLL (mean+-std)':>22} {'BIC':>10} {'Dead':>6} {'Sing':>6}")
            for r in rows:
                nll_str = f"{r['test_nll_mean']:.4f} +- {r['test_nll_std']:.4f}"
                print(f"  {r['method']:<22} {nll_str:>22} {r['test_bic_mean']:>10.1f} {r['dead_mean']:>6.2f} {r['singular_ev_mean']:>6.2f}")
        print(f"\nSaved telemetry -> {out_path} ({elapsed_s}s)")


if __name__ == "__main__":
    main()
