"""
Zero-Download Memory, HBM & FLOP Explorer for e-GMM Parametrizations.
Profiles Diagonal (Flash-GMM), Low-Rank+Diag (Woodbury), and Full Covariance
footprints across SIFT1M, DEEP1B, DINOv2-100M, ImageNet-1k, and the 275 GB wall.
"""

import pathlib
import sys
from typing import Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import DATASET_PRESETS, DatasetScaleSpec  # noqa: E402


def profile_egmm_spec(spec: DatasetScaleSpec, cov_type: str, rank: int = 16, top_m: int = 8) -> Dict[str, float]:
    k, d, B = spec.num_components, spec.dim, spec.default_batch_size

    # 3-tuple FP32 state bytes: N (k) + S (k*d) + Q
    if cov_type == "diag":
        q_elems = k * d
        u_elems = 0
        e_flops = 2 * B * k * (2 * d)  # single 2d-augmented GEMM
        m_flops_dense = 4 * B * k * d
        m_flops_topm = 4 * B * top_m * d
    elif cov_type == "low_rank_diag":
        q_elems = k * d
        u_elems = k * d * rank
        e_flops = B * k * (2 * d + 2 * d * rank) + k * (d * rank * rank)
        m_flops_dense = 4 * B * k * d
        m_flops_topm = 4 * B * top_m * d
    else:  # full
        q_elems = k * d * d
        u_elems = 0
        e_flops = B * k * (d * d) + (k * (d ** 3)) // 3
        m_flops_dense = B * k * (d * d)
        m_flops_topm = B * top_m * (d * d)

    state_bytes_fp32 = 4 * (k + k * d + q_elems + u_elems)
    # With passive Slow-EMA shadow buffers
    state_bytes_with_slow = 2 * state_bytes_fp32

    naive_bk_hbm_gb = (4.0 * B * k) / (1024 ** 3)
    tiled_hbm_gb = (4.0 * B * min(2048, k)) / (1024 ** 3)
    if top_m is not None:
        topm_hbm_gb = (8.0 * B * top_m) / (1024 ** 3)
    else:
        topm_hbm_gb = naive_bk_hbm_gb

    return {
        "state_gb": state_bytes_fp32 / (1024 ** 3),
        "state_with_slow_gb": state_bytes_with_slow / (1024 ** 3),
        "naive_bk_gb": naive_bk_hbm_gb,
        "tiled_hbm_gb": tiled_hbm_gb,
        "topm_hbm_gb": topm_hbm_gb,
        "e_gflops": e_flops / 1e9,
        "m_speedup_topm": m_flops_dense / max(m_flops_topm, 1),
    }


def main():
    print("=" * 106)
    print("OFFLINE e-GMM MEMORY, HBM & FLOP PROFILER (ZERO DOWNLOADS)")
    print("=" * 106)

    rows = [
        ("sift-1m", "diag", 16),
        ("deep-1b", "diag", 16),
        ("spacev-100m", "diag", 16),
        ("dinov2-100m", "diag", 16),
        ("dinov2-100m", "low_rank_diag", 16),
        ("dinov2-100m", "full", 16),
        ("imagenet-1k", "full", 16),
    ]
    header = (
        f"{'Preset':<14} {'CovMode':<14} {'k':>7} {'d':>5} {'B':>8} "
        f"{'StateGB':>8} {'+SlowGB':>8} {'NaiveHBM':>9} {'TiledHBM':>9} {'E_GFLOP':>9} {'TopM_x':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, ctype, rank in rows:
        spec = DATASET_PRESETS[name]
        p = profile_egmm_spec(spec, ctype, rank=rank, top_m=8)
        print(
            f"{name:<14} {ctype:<14} {spec.num_components:>7d} {spec.dim:>5d} {spec.default_batch_size:>8d} "
            f"{p['state_gb']:>8.3f} {p['state_with_slow_gb']:>8.3f} {p['naive_bk_gb']:>8.2f}G "
            f"{p['tiled_hbm_gb']:>8.3f}G {p['e_gflops']:>9.1f} {p['m_speedup_topm']:>6.0f}x"
        )

    # Highlight the 275 GB wall explicitly
    wall_gib = (4.0 * 65536 * (1024 ** 2)) / (1024 ** 3)
    wall_gb_dec = (4.0 * 65536 * (1024 ** 2)) / 1e9
    print("\nKey Takeaways:")
    print(f"  1. Full Covariance Memory Wall (k=65,536, d=1024, FP32 Q_j): {wall_gib:.2f} GiB ({wall_gb_dec:.1f} GB) -> >3.2x an 80GB H100!")
    print("  2. Diagonal Flash-GMM reduces state to 0.54 GB (512x smaller) and runs as ONE 2d-augmented GEMM.")
    print("  3. Low-Rank+Diag (r=16) fits correlated DINOv2-100M (d=1024) into ~1.2-4.8 GB via Woodbury + Det Lemma.")
    print("  4. Streaming LogSumExp + Top-8 M-step eliminates multi-GB (B x k) HBM buffers and cuts M-step cost by k/8.")


if __name__ == "__main__":
    main()
