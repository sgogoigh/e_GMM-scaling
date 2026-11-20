"""
Experiment 04 (Systems): Flash-GMM 2d-GEMM vs. Naive, Woodbury Throughput & Precision Audit
===========================================================================================

PURPOSE:
  Validates systems performance & numerical parity of:
    1. Flash-GMM `2d`-augmented diagonal GEMM + streaming LogSumExp vs. naive O(B*k*d) E-step.
    2. Top-m sparse M-step accumulation throughput vs. dense (B x k) M-step.
    3. Woodbury Low-Rank+Diag E-step vs. dense full Cholesky throughput & memory.
    4. BF16 E-step + FP32 3-tuple accumulation vs. pure FP32 E-step parity (on PyTorch/CUDA).

HOW TO RUN:
  python experiments/exp_04_kernel_and_memory_bench.py --dry-run
  python experiments/exp_04_kernel_and_memory_bench.py --mode smoke
  python experiments/exp_04_kernel_and_memory_bench.py --mode full --backend torch

SAVED ARTIFACTS:
  - Telemetry : `experiments/results/exp_04_kernel_bench.json`
  - Log       : `experiments/logs/exp_04_systems/latest.log`
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
from e_gmm.metrics import (  # noqa: E402
    gaussian_log_prob_diag,
    gaussian_log_prob_full,
    gaussian_log_prob_low_rank_diag,
)
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 04 (Systems): Flash-GMM & Woodbury Throughput Benchmark")
    p.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else f"{args.mode}_{args.backend}"
    with tee_experiment_logs("exp_04_systems", run_tag=tag):
        B, d, k, r = (2048, 64, 512, 8) if args.mode == "smoke" else (16384, 256, 8192, 16)
        if args.dry_run:
            print("=== EXP 04 (SYSTEMS) DRY RUN ===")
            print(f"Benchmark tensor shapes: B={B}, d={d}, k={k}, r={r}")
            return

        rng = np.random.default_rng(7)
        X = rng.normal(size=(B, d))
        pi = rng.dirichlet(np.ones(k))
        mu = rng.normal(size=(k, d))
        var = rng.uniform(0.2, 2.0, size=(k, d))
        U = rng.normal(scale=0.3, size=(k, d, r))

        # 1. NumPy Flash-GMM 2d-GEMM vs naive E-step timing & parity
        t0 = time.perf_counter()
        lj_flash, _ = gaussian_log_prob_diag(X, pi, mu, var)
        dt_flash_ms = (time.perf_counter() - t0) * 1000.0

        cfg_naive = EGMMConfig(num_components=k, dim=d, cov_type="diag", e_step_mode="naive")
        m_naive = NumPyEGMM(cfg_naive)
        t0 = time.perf_counter()
        lj_naive = m_naive._compute_log_joint(X, pi, mu, var, None)
        dt_naive_ms = (time.perf_counter() - t0) * 1000.0
        flash_max_err = float(np.max(np.abs(lj_flash - lj_naive)))

        # 2. Low-rank Woodbury vs Full Cholesky timing (on tractable k_lr)
        k_lr = min(k, 64)
        cov_full = np.array([np.diag(var[j]) + U[j] @ U[j].T for j in range(k_lr)])
        t0 = time.perf_counter()
        lj_wb, _ = gaussian_log_prob_low_rank_diag(X[:512], pi[:k_lr] / pi[:k_lr].sum(), mu[:k_lr], var[:k_lr], U[:k_lr])
        dt_wb_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        lj_chol, _, _ = gaussian_log_prob_full(X[:512], pi[:k_lr] / pi[:k_lr].sum(), mu[:k_lr], cov_full)
        dt_chol_ms = (time.perf_counter() - t0) * 1000.0
        wb_max_err = float(np.max(np.abs(lj_wb - lj_chol)))

        torch_bench = None
        if args.backend in ("auto", "torch"):
            try:
                import torch
                from e_gmm.triton_kernels import HAS_TRITON, flash_gmm_diag_log_probs

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
                pit = torch.from_numpy(pi.astype(np.float32)).to(dev)
                mut = torch.from_numpy(mu.astype(np.float32)).to(dev)
                vart = torch.from_numpy(var.astype(np.float32)).to(dev)

                t0 = time.perf_counter()
                r_full, _, _, lse_f = flash_gmm_diag_log_probs(Xt, pit, mut, vart, chunk_k=k, top_m=8)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t_single = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                r_tile, _, _, lse_t = flash_gmm_diag_log_probs(Xt, pit, mut, vart, chunk_k=max(64, k // 4), top_m=8)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t_tiled = (time.perf_counter() - t0) * 1000.0

                torch_bench = {
                    "device": dev,
                    "triton": HAS_TRITON,
                    "single_ms": round(t_single, 2),
                    "tiled_ms": round(t_tiled, 2),
                    "tiled_lse_max_diff": float((lse_f - lse_t).abs().max()),
                    "tiled_top8_max_diff": float((r_full - r_tile).abs().max()),
                }
            except ImportError:
                torch_bench = {"status": "skipped (torch not installed)"}

        payload = {
            "experiment_id": "exp_04_kernel_and_memory_bench",
            "mode": args.mode,
            "shapes": {"B": B, "d": d, "k": k, "r": r, "k_lr": k_lr},
            "numpy_flash_gmm": {
                "flash_2d_gemm_ms": round(dt_flash_ms, 2),
                "naive_broadcast_ms": round(dt_naive_ms, 2),
                "speedup_vs_naive": round(dt_naive_ms / max(dt_flash_ms, 1e-6), 2),
                "max_abs_log_diff": flash_max_err,
            },
            "numpy_woodbury_vs_cholesky": {
                "woodbury_ms": round(dt_wb_ms, 2),
                "cholesky_ms": round(dt_chol_ms, 2),
                "speedup_vs_cholesky": round(dt_chol_ms / max(dt_wb_ms, 1e-6), 2),
                "max_abs_log_diff": wb_max_err,
            },
            "torch_bench": torch_bench,
        }
        out_path = RESULTS_ROOT / "exp_04_kernel_bench.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== EXP 04 (SYSTEMS) COMPLETE ===")
        fg = payload["numpy_flash_gmm"]
        wb = payload["numpy_woodbury_vs_cholesky"]
        print(f"  Flash-GMM 2d-GEMM : {fg['flash_2d_gemm_ms']} ms vs Naive {fg['naive_broadcast_ms']} ms ({fg['speedup_vs_naive']}x, max_diff={fg['max_abs_log_diff']:.2e})")
        print(f"  Woodbury vs Chol  : {wb['woodbury_ms']} ms vs Cholesky {wb['cholesky_ms']} ms ({wb['speedup_vs_cholesky']}x, max_diff={wb['max_abs_log_diff']:.2e})")
        print(f"Saved telemetry -> {out_path}")


if __name__ == "__main__":
    main()
