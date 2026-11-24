"""
Experiment 04 (Systems): CPU timing and numerical parity of the E-step formulations
===================================================================================

Relabelled 2026-11-25 (docs/review/2026-11-25-REVIEW.md). What this measures, and what it does not:
  1. NumPy diagonal E-step via the [x^2, x] GEMM expansion vs a naive broadcast E-step (CPU timing,
     median of repeats, plus max |difference|). This is a reformulation, not a fused "Flash" kernel.
  2. Float32 precision of the expansion with and without centering, on data whose mean is far from
     the origin (the cancellation the 2026-11-25 centering fix addresses), against a float64 reference.
  3. NumPy Woodbury low-rank+diag log-density vs dense Cholesky (CPU timing and parity).
  4. The torch tiled path, only if torch is installed. It has NOT been run on the review machine, and
     the Triton kernel in triton_kernels.py is not wired into it. Nothing here measures GPU memory or
     a fused kernel.

HOW TO RUN:
  python experiments/exp_04_kernel_and_memory_bench.py --dry-run
  python experiments/exp_04_kernel_and_memory_bench.py --mode smoke

SAVED ARTIFACTS:
  - Results : experiments/results/exp_04_kernel_bench.json
  - Log     : experiments/logs/exp_04_systems/latest.log
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
    p = argparse.ArgumentParser(description="Exp 04 (Systems): CPU E-step timing and precision")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--out", type=str, default=str(RESULTS_ROOT / "exp_04_kernel_bench.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def median_ms(fn, repeats):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), out


def precision_check(rng, B=2048, d=64, k=64):
    """float32 log-joint error vs float64, with and without centering, as the data mean moves away from 0."""
    rows = []
    mu0 = rng.normal(size=(k, d))
    var = rng.uniform(0.2, 2.0, size=(k, d))
    pi = rng.dirichlet(np.ones(k))
    X0 = mu0[rng.integers(0, k, size=B)] + rng.normal(size=(B, d)) * np.sqrt(var[0])
    for offset in (0.0, 10.0, 100.0, 1000.0):
        X, mu = X0 + offset, mu0 + offset
        ref, _ = gaussian_log_prob_diag(X, pi, mu, var, center=True, compute_dtype=np.float64)
        c32, _ = gaussian_log_prob_diag(X, pi, mu, var, center=True, compute_dtype=np.float32)
        u32, _ = gaussian_log_prob_diag(X, pi, mu, var, center=False, compute_dtype=np.float32)
        rows.append({
            "mean_offset": offset,
            "float32_centered_max_abs_err": float(np.max(np.abs(c32 - ref))),
            "float32_uncentered_max_abs_err": float(np.max(np.abs(u32 - ref))),
        })
    return rows


def main():
    args = parse_args()
    B, d, k, r = (2048, 64, 512, 8) if args.mode == "smoke" else (16384, 256, 8192, 16)
    print(f"Exp 04 | mode={args.mode} B={B} d={d} k={k} r={r} repeats={args.repeats} (CPU NumPy timings)")
    if args.dry_run:
        print("Dry run: nothing measured.")
        return

    with tee_experiment_logs("exp_04_systems", run_tag=f"{args.mode}_{args.backend}"):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(B, d))
        pi = rng.dirichlet(np.ones(k))
        mu = rng.normal(size=(k, d))
        var = rng.uniform(0.2, 2.0, size=(k, d))
        U = rng.normal(scale=0.3, size=(k, d, r))

        t_gemm, (lj_gemm, _) = median_ms(lambda: gaussian_log_prob_diag(X, pi, mu, var), args.repeats)
        naive = NumPyEGMM(EGMMConfig(num_components=k, dim=d, cov_type="diag", e_step_mode="naive"))
        t_naive, lj_naive = median_ms(lambda: naive._compute_log_joint(X, pi, mu, var, None), args.repeats)

        k_lr = min(k, 64)
        cov_full = np.array([np.diag(var[j]) + U[j] @ U[j].T for j in range(k_lr)])
        p_lr = pi[:k_lr] / pi[:k_lr].sum()
        t_wb, (lj_wb, _) = median_ms(lambda: gaussian_log_prob_low_rank_diag(X[:512], p_lr, mu[:k_lr], var[:k_lr], U[:k_lr]), args.repeats)
        t_ch, (lj_ch, _, _) = median_ms(lambda: gaussian_log_prob_full(X[:512], p_lr, mu[:k_lr], cov_full), args.repeats)

        precision = precision_check(np.random.default_rng(11))

        torch_bench = {"status": "not requested"}
        if args.backend in ("auto", "torch"):
            try:
                import torch

                from e_gmm.triton_kernels import flash_gmm_diag_log_probs

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                to = lambda a: torch.from_numpy(a.astype(np.float32)).to(dev)  # noqa: E731
                Xt, pit, mut, vart = to(X), to(pi), to(mu), to(var)
                r_one, _, _, lse_one = flash_gmm_diag_log_probs(Xt, pit, mut, vart, chunk_k=k, top_m=8)
                r_tile, _, _, lse_tile = flash_gmm_diag_log_probs(Xt, pit, mut, vart, chunk_k=max(64, k // 4), top_m=8)
                torch_bench = {
                    "device": dev,
                    "triton_kernel_used": False,
                    "tiled_lse_max_diff": float((lse_one - lse_tile).abs().max()),
                    "tiled_top8_max_diff": float((r_one - r_tile).abs().max()),
                    "lse_vs_numpy_float64_max_diff": float(np.max(np.abs(lse_one.cpu().numpy() - np.logaddexp.reduce(lj_gemm, axis=1)))),
                }
            except ImportError:
                torch_bench = {"status": "skipped: torch is not installed (torch path untested on this machine)"}

        payload = {
            "experiment_id": "exp_04_kernel_and_memory_bench",
            "mode": args.mode,
            "platform": "CPU / NumPy",
            "shapes": {"B": B, "d": d, "k": k, "r": r, "k_lr": k_lr},
            "numpy_gemm_expansion_vs_naive": {
                "gemm_expansion_ms_median": round(t_gemm, 3),
                "naive_broadcast_ms_median": round(t_naive, 3),
                "speedup_vs_naive": round(t_naive / max(t_gemm, 1e-9), 2),
                "max_abs_log_diff": float(np.max(np.abs(lj_gemm - lj_naive))),
            },
            "float32_precision_vs_mean_offset": precision,
            "numpy_woodbury_vs_cholesky": {
                "woodbury_ms_median": round(t_wb, 3),
                "cholesky_ms_median": round(t_ch, 3),
                "speedup_vs_cholesky": round(t_ch / max(t_wb, 1e-9), 2),
                "max_abs_log_diff": float(np.max(np.abs(lj_wb - lj_ch))),
            },
            "torch_bench": torch_bench,
            "not_measured": ["GPU throughput", "GPU memory", "fused/Triton kernel", "bf16 E-step on GPU"],
        }
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        g, w = payload["numpy_gemm_expansion_vs_naive"], payload["numpy_woodbury_vs_cholesky"]
        print(f"  [x^2,x] GEMM vs naive : {g['gemm_expansion_ms_median']} ms vs {g['naive_broadcast_ms_median']} ms "
              f"({g['speedup_vs_naive']}x), max diff {g['max_abs_log_diff']:.2e}")
        print(f"  Woodbury vs Cholesky  : {w['woodbury_ms_median']} ms vs {w['cholesky_ms_median']} ms "
              f"({w['speedup_vs_cholesky']}x), max diff {w['max_abs_log_diff']:.2e}")
        for row in precision:
            print(f"  float32 at mean offset {row['mean_offset']:>6g}: centered err {row['float32_centered_max_abs_err']:.2e}, "
                  f"uncentered err {row['float32_uncentered_max_abs_err']:.2e}")
        print(f"  torch: {torch_bench}")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
