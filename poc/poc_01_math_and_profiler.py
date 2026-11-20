"""
PoC 1 (Offline CPU, $0 Cost): Mathematical Identities, PSD Proof & Scale Profiler
================================================================================

STANDALONE RUN INSTRUCTIONS:
  - Hardware Needed : Any CPU (NumPy only, zero downloads).
  - Run             : python poc/poc_01_math_and_profiler.py
  - Saved Artifacts :
      * Telemetry JSON : `poc/results/poc_01_results.json`
      * Run Log File   : `poc/logs/poc_01/run_<YYYYMMDD_HHMMSS>.log` (+ `latest.log`)
"""

import json
import math
import pathlib
import sys
import numpy as np

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import DATASET_PRESETS, EGMMConfig  # noqa: E402
from e_gmm.metrics import (  # noqa: E402
    gaussian_log_prob_diag,
    gaussian_log_prob_full,
    gaussian_log_prob_low_rank_diag,
)
from e_gmm.numpy_egmm import NumPyEGMM  # noqa: E402
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402


def verify_identities() -> dict:
    rng = np.random.default_rng(7)
    B, k, d, r = 64, 12, 24, 6
    X = rng.normal(size=(B, d))
    pi = rng.dirichlet(np.ones(k))
    mu = rng.normal(size=(k, d))
    var = rng.uniform(0.2, 2.5, size=(k, d))
    U = rng.normal(scale=0.4, size=(k, d, r))

    diff = X[:, None, :] - mu[None, :, :]
    lj_naive = np.log(pi)[None, :] - 0.5 * (
        np.sum((diff ** 2) / var[None, :, :], axis=2) + np.sum(np.log(2.0 * math.pi * var), axis=1)[None, :]
    )
    lj_flash, _ = gaussian_log_prob_diag(X, pi, mu, var)
    flash_max_diff = float(np.max(np.abs(lj_flash - lj_naive)))

    cov_full = np.array([np.diag(var[j]) + U[j] @ U[j].T for j in range(k)])
    lj_wb, _ = gaussian_log_prob_low_rank_diag(X, pi, mu, var, U)
    lj_dense, _, fb = gaussian_log_prob_full(X, pi, mu, cov_full)
    woodbury_max_diff = float(np.max(np.abs(lj_wb - lj_dense)))

    # PSD check on rank-1 collapsed batch
    floor = 1e-3
    m = NumPyEGMM(EGMMConfig(num_components=8, dim=d, cov_type="full", cov_floor=floor, split_strategy="none"))
    x_rep = np.repeat(rng.normal(size=(1, d)), repeats=16, axis=0)
    m.init_from_data(x_rep + rng.normal(scale=1e-6, size=(16, d)))
    m.step_batch(x_rep, alpha_t=0.95, allow_split=False)
    _, _, cov_out, _ = m.readout("main_ema")
    min_eig_rank1 = min(float(np.min(np.linalg.eigvalsh(cov_out[j]))) for j in range(8))

    # Spectral split conservation check
    ms = NumPyEGMM(EGMMConfig(num_components=4, dim=16, cov_type="full", cov_floor=1e-4, split_delta=0.5))
    Xs = rng.normal(size=(64, 16))
    ms.init_from_data(Xs)
    ms.N[:] = np.array([100.0, 40.0, 40.0, 0.01])
    mu0 = rng.normal(size=(16,))
    A = rng.normal(size=(16, 16))
    cov0 = (A @ A.T) / 16.0 + 0.2 * np.eye(16)
    ms.S[0] = 100.0 * mu0
    ms.Q[0] = 100.0 * (cov0 + np.outer(mu0, mu0))
    ms._revive_collapsed_components(Xs)
    _, mu_sp, cov_sp, _ = ms.readout("main_ema")
    split_mean_diff = float(np.max(np.abs(0.5 * (mu_sp[0] + mu_sp[3]) - mu0)))
    split_min_eig = min(float(np.min(np.linalg.eigvalsh(cov_sp[0]))), float(np.min(np.linalg.eigvalsh(cov_sp[3]))))

    return {
        "flash_gemm_max_log_diff": flash_max_diff,
        "woodbury_max_log_diff": woodbury_max_diff,
        "woodbury_chol_fallbacks": int(fb),
        "rank1_batch_min_eig": min_eig_rank1,
        "rank1_batch_floor": floor,
        "spectral_split_mean_diff": split_mean_diff,
        "spectral_split_min_eig": split_min_eig,
    }


def profile_scale_presets() -> list:
    targets = [
        ("sift-1m", "diag", 16),
        ("deep-1b", "diag", 16),
        ("spacev-100m", "diag", 16),
        ("dinov2-100m", "diag", 16),
        ("dinov2-100m", "low_rank_diag", 16),
        ("dinov2-100m", "full", 16),
        ("imagenet-1k", "full", 16),
    ]
    profiles = []
    for pname, cov_type, rank in targets:
        s = DATASET_PRESETS[pname]
        k, d, B = s.num_components, s.dim, s.default_batch_size
        if cov_type == "diag":
            q_el, u_el = k * d, 0
        elif cov_type == "low_rank_diag":
            q_el, u_el = k * d, k * d * rank
        else:
            q_el, u_el = k * d * d, 0
        state_gib = 4.0 * (k + k * d + q_el + u_el) / (1024 ** 3)
        naive_bk_gib = (4.0 * B * k) / (1024 ** 3)
        tiled_hbm_gib = (4.0 * B * min(2048, k)) / (1024 ** 3)
        profiles.append(
            {
                "preset": pname,
                "cov_type": cov_type,
                "k": k,
                "d": d,
                "B": B,
                "state_fp32_GiB": round(state_gib, 4),
                "naive_bk_HBM_GiB": round(naive_bk_gib, 3),
                "tiled_HBM_GiB": round(tiled_hbm_gib, 4),
            }
        )
    return profiles


def main():
    with tee_poc_logs("poc_01"):
        out_dir = POC_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        identities = verify_identities()
        profiles = profile_scale_presets()
        full_wall_gib = round((4.0 * 65536 * (1024 ** 2)) / (1024 ** 3), 2)
        full_wall_gb = round((4.0 * 65536 * (1024 ** 2)) / 1e9, 2)

        payload = {
            "poc_id": "poc_01",
            "title": "Offline Mathematical Identities, PSD Floor Proof & Memory/HBM Scale Audit",
            "identities": identities,
            "scale_profiles": profiles,
            "full_cov_wall_k65k_d1024": {"GiB": full_wall_gib, "GB_decimal": full_wall_gb},
        }
        out_path = out_dir / "poc_01_results.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== PoC 1 COMPLETE ===")
        print(f"  Flash-GMM 2d-GEMM max |delta log p| : {identities['flash_gemm_max_log_diff']:.3e}")
        print(f"  Woodbury Det Lemma max |delta log p|: {identities['woodbury_max_log_diff']:.3e}")
        print(f"  Rank-1 collapsed batch min_eig      : {identities['rank1_batch_min_eig']:.6e} (floor={identities['rank1_batch_floor']:.1e})")
        print(f"  Spectral split mean symmetry diff   : {identities['spectral_split_mean_diff']:.3e}")
        print(f"  Full-Cov Memory Wall (k=65k,d=1024) : {full_wall_gib} GiB ({full_wall_gb} GB)")
        print(f"Saved JSON telemetry -> {out_path}")


if __name__ == "__main__":
    main()
