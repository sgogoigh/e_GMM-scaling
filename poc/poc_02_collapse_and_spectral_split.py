"""
PoC 2 (CPU / 1x GPU, $0 Cost): Covariance Collapse & Spectral Splitting Ablation
=================================================================================

Compares 8 methods on an Anisotropic Zipf-Imbalanced Gaussian Mixture (heavy cluster
imbalance + starved slots):
  1. online_em_beta0.60  (Cappe & Moulines, low floor -> singular instability)
  2. online_em_beta0.85  (Cappe & Moulines, aggressive decay -> early stall)
  3. emeans_posthoc      (Hard e-means + 1 post-hoc covariance M-step)
  4. egmm_no_revive      (e-GMM ablation without dead-component revival)
  5. egmm_batch_resample (VQ-VAE style random batch resampling)
  6. egmm_faiss_alt      (Faiss alternating +/- delta hyperplane split)
  7. egmm_random_orth    (Random direction split)
  8. egmm_spectral_v1    (Ours: Spectral donor split along principal eigenvector v_1)
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
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=1200)
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--num-components", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cov-type", type=str, default="diag", choices=["diag", "full"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else args.cov_type
    with tee_poc_logs("poc_02", run_tag=tag):
        if args.dry_run:
            print("=== PoC 2 DRY RUN ===")
            print(f"Config: n_train={args.n_train}, n_test={args.n_test}, d={args.dim}, k={args.num_components}, cov={args.cov_type}")
            return

        X_all, _, _ = generate_gaussian_mixture(
            n=args.n_train + args.n_test,
            d=args.dim,
            num_true_clusters=args.num_components,
            zipf=True,
            zipf_power=1.25,
            anisotropic=True,
            seed=args.seed,
        )
        X_tr, X_te = X_all[: args.n_train], X_all[args.n_train :]
        k, d, B, ep = args.num_components, args.dim, args.batch_size, args.epochs

        runs = []

        for label, beta, floor in [("online_em_beta0.60", 0.60, 1e-9), ("online_em_beta0.85", 0.85, 1e-9)]:
            oem = NumPyOnlineEMCappeMoulines(k, d, cov_type=args.cov_type, beta=beta, t0=10.0, cov_floor=floor, seed=args.seed)
            res = oem.fit(X_tr, batch_size=B, epochs=ep)
            ev_tr = evaluate_gmm_nll(X_tr, res["pi"], res["mu"], res["cov"], cov_type=args.cov_type)
            ev_te = evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"], cov_type=args.cov_type)
            runs.append(
                {
                    "method": label,
                    "splits": 0,
                    "dead_components": int(res["health"]["dead_count"]),
                    "max_to_mean": round(res["health"]["max_to_mean"], 3),
                    "norm_entropy": round(res["health"]["norm_entropy"], 4),
                    "singular_or_floor_events": int(res["singular_events"]),
                    "train_nll": round(ev_tr["nll"], 4),
                    "test_nll": round(ev_te["nll"], 4),
                    "test_bic": round(ev_te["bic"], 1),
                }
            )

        epost = NumPyEMeansPostHocGMM(k, d, cov_type=args.cov_type, cov_floor=1e-4, alpha_0=0.45, seed=args.seed)
        res_ep = epost.fit(X_tr, batch_size=B, epochs=ep)
        ev_tr = evaluate_gmm_nll(X_tr, res_ep["pi"], res_ep["mu"], res_ep["cov"], cov_type=args.cov_type)
        ev_te = evaluate_gmm_nll(X_te, res_ep["pi"], res_ep["mu"], res_ep["cov"], cov_type=args.cov_type)
        runs.append(
            {
                "method": "emeans_posthoc",
                "splits": int(res_ep["splits"]),
                "dead_components": int(res_ep["health"]["dead_count"]),
                "max_to_mean": round(res_ep["health"]["max_to_mean"], 3),
                "norm_entropy": round(res_ep["health"]["norm_entropy"], 4),
                "singular_or_floor_events": 0,
                "train_nll": round(ev_tr["nll"], 4),
                "test_nll": round(ev_te["nll"], 4),
                "test_bic": round(ev_te["bic"], 1),
            }
        )

        for label, strat in [
            ("egmm_no_revive", "none"),
            ("egmm_batch_resample", "batch_resample"),
            ("egmm_faiss_alt", "faiss_alt"),
            ("egmm_random_orth", "random_orth"),
            ("egmm_spectral_v1", "spectral_v1"),
        ]:
            cfg = EGMMConfig(
                num_components=k,
                dim=d,
                cov_type=args.cov_type,
                cov_floor=1e-4,
                alpha_0=0.45,
                split_interval=5,
                split_tau=0.10,
                split_delta=0.50,
                split_strategy=strat,
                finalization="last_epoch",
                seed=args.seed,
            )
            m = NumPyEGMM(cfg)
            res_m = m.fit(X_tr, batch_size=B, epochs=ep)
            ev_tr = evaluate_gmm_nll(X_tr, res_m["pi"], res_m["mu"], res_m["cov"], cov_type=args.cov_type)
            ev_te = evaluate_gmm_nll(X_te, res_m["pi"], res_m["mu"], res_m["cov"], cov_type=args.cov_type)
            runs.append(
                {
                    "method": label,
                    "splits": int(res_m["splits"]),
                    "dead_components": int(res_m["health"]["dead_count"]),
                    "max_to_mean": round(res_m["health"]["max_to_mean"], 3),
                    "norm_entropy": round(res_m["health"]["norm_entropy"], 4),
                    "singular_or_floor_events": int(res_m["chol_fallbacks"]),
                    "train_nll": round(ev_tr["nll"], 4),
                    "test_nll": round(ev_te["nll"], 4),
                    "test_bic": round(ev_te["bic"], 1),
                }
            )

        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_02_results.json"
        payload = {
            "poc_id": "poc_02",
            "title": "Anisotropic Zipf-GMM Collapse & Spectral Splitting Ablation",
            "config": vars(args),
            "runs": runs,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 2 COMPLETE ===")
        print(f"{'Method':<22} {'Splits':>6} {'Dead':>5} {'Max/Mean':>9} {'Entropy':>8} {'Sing/FB':>8} {'TestNLL':>10}")
        print("-" * 78)
        for r in runs:
            print(
                f"{r['method']:<22} {r['splits']:>6d} {r['dead_components']:>5d} "
                f"{r['max_to_mean']:>8.2f}x {r['norm_entropy']:>8.4f} {r['singular_or_floor_events']:>8d} {r['test_nll']:>10.4f}"
            )
        print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
