"""
PoC 2 (CPU / 1x GPU, $0 Cost): Covariance Collapse & Spectral Splitting Ablation
=================================================================================

Runs a 3-seed benchmark on Rotated Anisotropic Zipf-Imbalanced Gaussian Mixtures
comparing 8 methods:
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

        methods = [
            "online_em_beta0.60",
            "online_em_beta0.85",
            "emeans_posthoc",
            "egmm_no_revive",
            "egmm_batch_resample",
            "egmm_faiss_alt",
            "egmm_random_orth",
            "egmm_spectral_v1",
        ]
        per_method = {m: [] for m in methods}

        for s in range(1, args.seeds + 1):
            X_all, _, _ = generate_gaussian_mixture(
                n=args.n_train + args.n_test,
                d=args.dim,
                num_true_clusters=args.num_components,
                zipf=True,
                zipf_power=1.20,
                anisotropic=True,
                rotated=(args.cov_type == "full"),
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

        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_02_results.json"
        payload = {
            "poc_id": "poc_02",
            "title": "Rotated Anisotropic Zipf-GMM Collapse & Spectral Splitting Ablation",
            "config": vars(args),
            "runs": runs,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 2 COMPLETE ===")
        print(f"{'Method':<22} {'Splits':>6} {'Dead':>6} {'Max/Mean':>9} {'Entropy':>8} {'Sing/FB':>8} {'TestNLL':>10}")
        print("-" * 80)
        for r in runs:
            print(
                f"{r['method']:<22} {r['splits']:>6.1f} {r['dead_components']:>6.1f} "
                f"{r['max_to_mean']:>8.2f}x {r['norm_entropy']:>8.4f} {r['singular_or_floor_events']:>8.1f} {r['test_nll']:>10.4f}"
            )
        print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
