"""
PoC 4 (CPU NumPy or 1x GPU PyTorch): Top-m Sparsity, Low-Rank Woodbury & Finalization
======================================================================================

Evaluates three Phase-C architectural ablations:
  1. Top-m sparse responsibility truncation (m in {2, 4, 8, dense})
  2. Rotated correlated embeddings: Diagonal vs Low-Rank+Diag (Woodbury, r=4) vs Full
  3. Finalization trade-off at small n/k: Main-EMA vs Last-Epoch EM vs Slow-EMA shadow
"""

import argparse
import json
import pathlib
import sys
import time

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture, generate_stretched_gaussian  # noqa: E402
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 4: Top-m Sparsity, Low-Rank Woodbury & Finalization Ablations")
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else args.backend
    with tee_poc_logs("poc_04", run_tag=tag):
        if args.dry_run:
            print("=== PoC 4 DRY RUN ===")
            print(f"Planned ablations: Top-m {{2,4,8,full}}, CovModes {{diag, low_rank_diag, full}}, Finalizers {{main, last, slow}}")
            return

        # 1. Top-m Sparse E-step ablation (k=32, d=20)
        X_tm, _, _ = generate_gaussian_mixture(n=3000, d=20, num_true_clusters=32, zipf=False, anisotropic=True, seed=args.seed)
        X_tm_tr, X_tm_te = X_tm[:2200], X_tm[2200:]
        topm_rows = []
        for top_m in (2, 4, 8, None):
            t0 = time.perf_counter()
            m = NumPyEGMM(EGMMConfig(num_components=32, dim=20, cov_type="diag", top_m=top_m, alpha_0=0.4, seed=args.seed))
            res = m.fit(X_tm_tr, batch_size=128, epochs=4)
            dt = (time.perf_counter() - t0) * 1000.0
            ev = evaluate_gmm_nll(X_tm_te, res["pi"], res["mu"], res["cov"], cov_type="diag")
            topm_rows.append(
                {
                    "top_m": "dense" if top_m is None else top_m,
                    "m_step_work_ratio": 1.0 if top_m is None else round(top_m / 32.0, 4),
                    "test_nll": round(ev["nll"], 4),
                    "fit_ms": round(dt, 1),
                }
            )

        # 2. Rotated Correlated GMM: diag vs low_rank_diag (r=4) vs full
        X_rot, _, _ = generate_gaussian_mixture(
            n=2600, d=16, num_true_clusters=8, zipf=False, anisotropic=True, rotated=True, seed=args.seed + 5
        )
        X_r_tr, X_r_te = X_rot[:1800], X_rot[1800:]
        cov_rows = []
        for ctype in ("diag", "low_rank_diag", "full"):
            cfg = EGMMConfig(num_components=8, dim=16, cov_type=ctype, low_rank=4, alpha_0=0.35, seed=args.seed)
            m = NumPyEGMM(cfg)
            res = m.fit(X_r_tr, batch_size=128, epochs=4)
            ev = evaluate_gmm_nll(X_r_te, res["pi"], res["mu"], res["cov"], cov_type=ctype, U=res["U"])
            cov_rows.append(
                {
                    "cov_type": ctype,
                    "test_nll": round(ev["nll"], 4),
                    "test_bic": round(ev["bic"], 1),
                    "n_params": int(ev["n_effective_params"]),
                }
            )

        # 3. Finalization: main_ema vs last_epoch vs slow_ema in small n/k overfitting regime
        X_fin_tr, _ = generate_stretched_gaussian(n=1200, d=20, exponent=1.0, seed=args.seed + 11)
        X_fin_te, _ = generate_stretched_gaussian(n=1200, d=20, exponent=1.0, seed=args.seed + 12)
        m_fin = NumPyEGMM(
            EGMMConfig(
                num_components=24,
                dim=20,
                cov_type="diag",
                alpha_0=0.45,
                slow_ema_alpha=0.05,
                finalization="slow_ema",
                seed=args.seed,
            )
        )
        m_fin.fit(X_fin_tr, batch_size=64, epochs=4)
        fin_rows = []
        for mode in ("main_ema", "last_epoch", "slow_ema"):
            pi, mu, cov, _ = m_fin.readout(mode)
            tr_nll = evaluate_gmm_nll(X_fin_tr, pi, mu, cov, cov_type="diag")["nll"]
            te_nll = evaluate_gmm_nll(X_fin_te, pi, mu, cov, cov_type="diag")["nll"]
            fin_rows.append({"finalizer": mode, "train_nll": round(tr_nll, 4), "test_nll": round(te_nll, 4)})

        torch_parity = None
        if args.backend == "torch":
            try:
                import torch
                from e_gmm.triton_kernels import flash_gmm_diag_log_probs

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                Xt = torch.from_numpy(X_tm_tr[:256]).to(dev)
                pi_t = torch.full((32,), 1.0 / 32.0, device=dev)
                mu_t = torch.randn(32, 20, device=dev)
                var_t = torch.rand(32, 20, device=dev) + 0.2
                r_f, _, _, lse_f = flash_gmm_diag_log_probs(Xt, pi_t, mu_t, var_t, chunk_k=32)
                r_c, _, _, lse_c = flash_gmm_diag_log_probs(Xt, pi_t, mu_t, var_t, chunk_k=8)
                torch_parity = {
                    "device": dev,
                    "chunked_lse_max_diff": float((lse_f - lse_c).abs().max()),
                    "chunked_r_max_diff": float((r_f - r_c).abs().max()),
                }
            except ImportError:
                torch_parity = {"error": "torch not installed"}

        payload = {
            "poc_id": "poc_04",
            "title": "Top-m Sparsity, Woodbury Low-Rank+Diag & Dual Finalization Ablations",
            "top_m_ablation": topm_rows,
            "covariance_mode_ablation": cov_rows,
            "finalization_ablation": fin_rows,
            "torch_parity": torch_parity,
        }
        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "poc_04_results.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 4 COMPLETE ===")
        print("  Top-m Sparsity:", [(r["top_m"], r["test_nll"]) for r in topm_rows])
        print("  Cov Modes     :", [(r["cov_type"], r["test_nll"]) for r in cov_rows])
        print("  Finalization  :", [(r["finalizer"], r["train_nll"], r["test_nll"]) for r in fin_rows])
        print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
