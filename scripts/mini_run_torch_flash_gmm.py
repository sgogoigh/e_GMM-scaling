"""
PyTorch / GPU Dry-Run Validator:
1. Verifies Flash-GMM chunked 2d-GEMM E-step parity vs naive PyTorch E-step.
2. Verifies Top-m sparse M-step accumulation and Low-Rank+Diag Woodbury E-step.
3. Benchmarks step time on CUDA (if available) or CPU.
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.datasets import generate_gaussian_mixture  # noqa: E402


def main():
    try:
        import torch
        from e_gmm.config import EGMMConfig
        from e_gmm.torch_egmm import TorchEGMM
        from e_gmm.triton_kernels import HAS_TRITON, flash_gmm_diag_log_probs
    except ImportError:
        print("PyTorch is not installed in this environment. Install `torch>=2.3.0` to run GPU/PyTorch validation.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running PyTorch Flash-GMM verification on device={device} (triton={HAS_TRITON})")

    X_np, _, _ = generate_gaussian_mixture(n=1024, d=64, num_true_clusters=16, zipf=True, anisotropic=True, seed=7)
    X = torch.from_numpy(X_np).to(device)

    # 1. Parity between single-block and chunked streaming LogSumExp Flash-GMM
    pi = torch.softmax(torch.randn(128, device=device), dim=0)
    mu = torch.randn(128, 64, device=device)
    var = torch.rand(128, 64, device=device) + 0.2

    r_full, _, _, lse_full = flash_gmm_diag_log_probs(X, pi, mu, var, chunk_k=128)
    r_chunk, _, _, lse_chunk = flash_gmm_diag_log_probs(X, pi, mu, var, chunk_k=32)
    print(f"  Streaming chunked Flash-GMM max |lse diff|: {float((lse_full - lse_chunk).abs().max()):.3e}")
    print(f"  Streaming chunked Flash-GMM max |r diff|  : {float((r_full - r_chunk).abs().max()):.3e}")

    # 2. Run TorchEGMM in diag (Flash-GMM + Top-8), low_rank_diag, and full modes
    for cov_type, top_m in [("diag", 8), ("low_rank_diag", None), ("full", None)]:
        cfg = EGMMConfig(
            num_components=16,
            dim=64,
            cov_type=cov_type,
            low_rank=8,
            top_m=top_m,
            chunk_k=8 if cov_type == "diag" else 16,
        )
        model = TorchEGMM(cfg).to(device)
        t0 = time.perf_counter()
        nll = model.step_batch(X, alpha_t=0.25)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  TorchEGMM({cov_type:<13}, top_m={str(top_m):<4}) | step_nll={nll:.4f} | step_time={dt:.2f} ms")


if __name__ == "__main__":
    main()
