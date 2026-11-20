"""
PoC 2 (CPU / 1x GPU, $0 Cost): Covariance Collapse, Spectral Splitting & Soft-EM Advantage
==========================================================================================

Runs 3-seed benchmarks across TWO complementary regimes (`cov_type="full"`, d=16, k=16):
  Regime A — Zipf-1.2 Rotated Anisotropic GMM (extreme cluster imbalance & starvation):
    Compares 8 methods including Cappé-Moulines Online EM, e-means+PostHoc, and
    e-GMM revival ablations (none, batch_resample, faiss_alt, random_orth, spectral_v1
    with variance-mass donor selection).
  Regime B — Overlapping Rotated Anisotropic GMM (sigma_mu^2=0.2, sigma_w^2=0.8):
    Proves that joint soft EM in e-GMM strictly outperforms hard k-means + post-hoc
    covariance (`emeans_posthoc`) when Gaussian tails overlap anisotropically.
"""

import argparse
import json
import pathlib
import sys
import numpy as np

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture  # noqa: E402
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyOnlineEMCappeMoulines,
)
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 2: Collapse & Spectral Component Splitting Benchmark")
    p.add_argument("--n-train", type=int, default=2400)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--num-components", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--cov-type", type=str, default="full", choices=["diag", "full"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _eval_single(label: str, args, X_tr: np.ndarray, X_te: np.ndarray, seed: int) -> dict:
    k, d, B, ep = args.num_components, args.dim, args.batch_size, args.epochs
    ctype = args.cov_type

    if label.startswith("online_em_beta"):
        beta = float(label.split("beta")[1])
        oem = NumPyOnlineEMCappeMoulines(
            k, d, cov_type=ctype, beta=beta, t0=10.0, cov_floor=1e-8, init_sigma0=0.5, seed=seed
        )
        res = oem.fit(X_tr, batch_size=B, epochs=ep)
        splits, sing_fb = 0, int(res["singular_events"])
        pi, mu, cov, health = res["pi"], res["mu"], res["cov"], res["health"]
    elif label == "emeans_posthoc":
        epost = NumPyEMeansPostHocGMM(k, d, cov_type=ctype, cov_floor=1e-4, alpha_0=0.35, seed=seed)
        res = epost.fit(X_tr, batch_size=B, epochs=ep)
        splits, sing_fb = int(res["splits"]), 0
        pi, mu, cov, health = res["pi"], res["mu"], res["cov"], res["health"]
    else:
        strat_map = {
            "egmm_no_revive": "none",
            "egmm_batch_resample": "batch_resample",
            "egmm_faiss_alt": "faiss_alt",
            "egmm_random_orth": "random_orth",
            "egmm_spectral_v1": "spectral_v1",
        }
        cfg = EGMMConfig(
            num_components=k,
            dim=d,
            cov_type=ctype,
            cov_floor=1e-4,
            init_prior_mass=1.0,
            alpha_0=0.35,
            split_interval=5,
            split_tau=0.10,
            split_delta=0.50,
            split_strategy=strat_map[label],
            donor_weight_by="variance_mass",
            donor_min_mult=1.25,
            cooldown_steps=15,
            finalization="last_epoch",
            seed=seed,
        )
        m = NumPyEGMM(cfg)
        res = m.fit(X_tr, batch_size=B, epochs=ep)
        splits, sing_fb = int(res["splits"]), int(res["chol_fallbacks"])
        pi, mu, cov, health = res["pi"], res["mu"], res["cov"], res["health"]

    ev_tr = evaluate_gmm_nll(X_tr, pi, mu, cov, cov_type=ctype)
    ev_te = evaluate_gmm_nll(X_te, pi, mu, cov, cov_type=ctype)
    return {
        "splits": splits,
        "dead": int(health["dead_count"]),
        "max_to_mean": float(health["max_to_mean"]),
        "norm_entropy": float(health["norm_entropy"]),
        "sing_fb": sing_fb,
        "train_nll": float(ev_tr["nll"]),
        "test_nll": float(ev_te["nll"]),
        "test_bic": float(ev_te["bic"]),
    }


def _run_regime(methods: list, args, zipf: bool, sigma_mu_sq: float, sigma_w_sq: float) -> list:
    per_method = {m: [] for m in methods}
    for s in range(1, args.seeds + 1):
        X_all, _, _ = generate_gaussian_mixture(
            n=args.n_train + args.n_test,
            d=args.dim,
            num_true_clusters=args.num_components,
            zipf=zipf,
            zipf_power=1.20,
            anisotropic=True,
            rotated=(args.cov_type == "full"),
            sigma_mu_sq=sigma_mu_sq,
            sigma_w_sq=sigma_w_sq,
            seed=s,
        )
        X_tr, X_te = X_all[: args.n_train], X_all[args.n_train :]
        for mname in methods:
            per_method[mname].append(_eval_single(mname, args, X_tr, X_te, seed=s))

    runs = []
    for mname in methods:
        recs = per_method[mname]
        runs.append(
            {
                "method": mname,
                "splits": round(float(np.mean([r["splits"] for r in recs])), 1),
                "dead_components": round(float(np.mean([r["dead"] for r in recs])), 1),
                "max_to_mean": round(float(np.mean([r["max_to_mean"] for r in recs])), 2),
                "norm_entropy": round(float(np.mean([r["norm_entropy"] for r in recs])), 4),
                "singular_or_floor_events": round(float(np.mean([r["sing_fb"] for r in recs])), 1),
                "train_nll": round(float(np.mean([r["train_nll"] for r in recs])), 4),
                "test_nll": round(float(np.mean([r["test_nll"] for r in recs])), 4),
                "test_nll_std": round(float(np.std([r["test_nll"] for r in recs])), 4),
                "test_bic": round(float(np.mean([r["test_bic"] for r in recs])), 1),
            }
        )
    return runs


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else args.cov_type
    with tee_poc_logs("poc_02", run_tag=tag):
        if args.dry_run:
            print("=== PoC 2 DRY RUN ===")
            print(
                f"Config: n_train={args.n_train}, n_test={args.n_test}, d={args.dim}, "
                f"k={args.num_components}, cov={args.cov_type}, seeds={args.seeds}"
            )
            return

        methods_zipf = [
            "online_em_beta0.60",
            "online_em_beta0.85",
            "emeans_posthoc",
            "egmm_no_revive",
            "egmm_batch_resample",
            "egmm_faiss_alt",
            "egmm_random_orth",
            "egmm_spectral_v1",
        ]
        runs_zipf = _run_regime(methods_zipf, args, zipf=True, sigma_mu_sq=2.0 / 3.0, sigma_w_sq=1.0 / 3.0)

        methods_overlap = ["online_em_beta0.85", "emeans_posthoc", "egmm_no_revive", "egmm_spectral_v1"]
        runs_overlap = _run_regime(methods_overlap, args, zipf=False, sigma_mu_sq=0.20, sigma_w_sq=0.80)

        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_02_results.json"
        payload = {
            "poc_id": "poc_02",
            "title": "Rotated Anisotropic Zipf & Overlapping GMM Collapse + Spectral Splitting Ablation",
            "config": vars(args),
            "runs": runs_zipf,
            "overlapping_runs": runs_overlap,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 2 COMPLETE ===")
        print("\n[Regime A: Zipf-1.2 Rotated Anisotropic GMM]")
        print(f"{'Method':<22} {'Splits':>6} {'Dead':>6} {'Max/Mean':>9} {'Entropy':>8} {'TestNLL':>10}")
        print("-" * 70)
        for r in runs_zipf:
            print(
                f"{r['method']:<22} {r['splits']:>6.1f} {r['dead_components']:>6.1f} "
                f"{r['max_to_mean']:>8.2f}x {r['norm_entropy']:>8.4f} {r['test_nll']:>10.4f}"
            )
        print("\n[Regime B: Overlapping Rotated Anisotropic GMM (sigma_mu^2=0.2, sigma_w^2=0.8)]")
        print(f"{'Method':<22} {'Splits':>6} {'Dead':>6} {'Max/Mean':>9} {'Entropy':>8} {'TestNLL':>10}")
        print("-" * 70)
        for r in runs_overlap:
            print(
                f"{r['method']:<22} {r['splits']:>6.1f} {r['dead_components']:>6.1f} "
                f"{r['max_to_mean']:>8.2f}x {r['norm_entropy']:>8.4f} {r['test_nll']:>10.4f}"
            )
        print(f"\nSaved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
