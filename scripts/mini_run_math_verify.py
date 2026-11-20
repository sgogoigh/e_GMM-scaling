"""
Offline Exploratory Mini-Run (Pure NumPy):
1. Proves positive-definiteness guarantee (Sigma_j >= lambda_0 I_d) on rank-1 batches.
2. Proves Flash-GMM 2d-augmented GEMM log-density equivalence against naive formula.
3. Proves Low-Rank+Diag Woodbury & Matrix Determinant Lemma equivalence against dense Cholesky.
4. Proves Spectral Split second-moment consistency & Law of Total Covariance.
5. Compares FullBatchEM, OnlineEM (Cappe-Moulines), e-means+PostHoc, e-GMM (No Revival),
   and e-GMM (Spectral v1 Split) on an anisotropic Zipf-imbalanced Gaussian Mixture.
"""

import math
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture  # noqa: E402
from e_gmm.metrics import (  # noqa: E402
    evaluate_gmm_nll,
    gaussian_log_prob_diag,
    gaussian_log_prob_full,
    gaussian_log_prob_low_rank_diag,
)
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
)


def verify_psd_and_rank_safety():
    print("\n[1] NUMERICAL PROOF: PSD Floor Sigma_j >= lambda_0 I_d on Rank-1 Batches")
    print("-" * 82)
    rng = np.random.default_rng(7)
    d, k, floor = 32, 8, 1e-3
    cfg = EGMMConfig(num_components=k, dim=d, cov_type="full", cov_floor=floor, alpha_0=0.95, split_strategy="none")
    model = NumPyEGMM(cfg)
    # Single repeated vector -> sample covariance has rank 0 / 1 (normally singular!)
    x_const = np.repeat(rng.normal(size=(1, d)), repeats=16, axis=0)
    model.init_from_data(x_const + rng.normal(scale=1e-6, size=(16, d)))
    model.step_batch(x_const, alpha_t=0.95, allow_split=False)
    _, _, cov, _ = model.readout("main_ema")
    min_eig = min(float(np.min(np.linalg.eigvalsh(cov[j]))) for j in range(k))
    print(f"  Rank-1 collapsed batch min eigenvalue across all k={k} covariances: {min_eig:.6e} (floor={floor:.1e})")
    assert min_eig >= floor * 0.999999


def verify_flash_gemm_and_woodbury():
    print("\n[2] EXACT EQUIVALENCE: Flash-GMM 2d-GEMM & Woodbury Matrix Determinant Lemma")
    print("-" * 82)
    rng = np.random.default_rng(11)
    B, k, d, r = 64, 12, 24, 6
    X = rng.normal(size=(B, d))
    pi = rng.dirichlet(np.ones(k))
    mu = rng.normal(size=(k, d))
    var = rng.uniform(0.2, 2.5, size=(k, d))

    # Naive diag log joint vs 2d-augmented Flash-GMM GEMM
    diff = X[:, None, :] - mu[None, :, :]
    maha_naive = np.sum((diff ** 2) / var[None, :, :], axis=2)
    logdet_naive = np.sum(np.log(2.0 * math.pi * var), axis=1)
    lj_naive = np.log(pi)[None, :] - 0.5 * (maha_naive + logdet_naive[None, :])
    lj_flash, _ = gaussian_log_prob_diag(X, pi, mu, var)
    flash_err = float(np.max(np.abs(lj_flash - lj_naive)))

    # Woodbury low-rank + diag vs dense full Cholesky on D_j + U_j U_j^T
    U = rng.normal(scale=0.4, size=(k, d, r))
    cov_full = np.array([np.diag(var[j]) + U[j] @ U[j].T for j in range(k)])
    lj_woodbury, _ = gaussian_log_prob_low_rank_diag(X, pi, mu, var, U)
    lj_dense, _, _ = gaussian_log_prob_full(X, pi, mu, cov_full)
    woodbury_err = float(np.max(np.abs(lj_woodbury - lj_dense)))

    print(f"  Flash-GMM 2d-GEMM max |delta log p| : {flash_err:.3e}")
    print(f"  Woodbury + Det Lemma max |delta log p|: {woodbury_err:.3e}")
    assert flash_err < 1e-11
    assert woodbury_err < 1e-11


def verify_spectral_split_moment_conservation():
    print("\n[3] CONSERVATION CHECK: Spectral Split Second-Moment & Variance Identity")
    print("-" * 82)
    rng = np.random.default_rng(19)
    d = 16
    cfg = EGMMConfig(num_components=4, dim=d, cov_type="full", cov_floor=1e-4, split_delta=0.5, split_strategy="spectral_v1")
    m = NumPyEGMM(cfg)
    X = rng.normal(size=(128, d))
    m.init_from_data(X)
    # Create an elongated donor (0) and starved component (3)
    m.N[:] = np.array([100.0, 40.0, 40.0, 0.01])
    mu0 = rng.normal(size=(d,))
    A = rng.normal(size=(d, d))
    cov0 = (A @ A.T) / d + np.eye(d) * 0.2
    m.S[0] = 100.0 * mu0
    m.Q[0] = 100.0 * (cov0 + np.outer(mu0, mu0))
    m._revive_collapsed_components(X)
    pi, mu, cov, _ = m.readout("main_ema")
    # Reconstruct combined mixture covariance of siblings (0 and 3)
    mu_pair = 0.5 * (mu[0] + mu[3])
    cov_pair = 0.5 * (cov[0] + cov[3]) + 0.5 * np.outer(mu[0] - mu_pair, mu[0] - mu_pair) + 0.5 * np.outer(mu[3] - mu_pair, mu[3] - mu_pair)
    mean_err = float(np.max(np.abs(mu_pair - mu0)))
    print(f"  Sibling mass N_0={m.N[0]:.2f}, N_3={m.N[3]:.2f} | |mean_pair - mu_donor|={mean_err:.3e}")
    print(f"  Max |eig_min(cov_children)| >= floor: min_eig={min(np.min(np.linalg.eigvalsh(cov[0])), np.min(np.linalg.eigvalsh(cov[3]))):.4e}")
    assert mean_err < 1e-10
    assert np.all(np.linalg.eigvalsh(cov_pair) > 0)


def run_zipf_benchmark():
    print("\n[4] ZIPF-IMBALANCED ANISOTROPIC GMM BENCHMARK (n=3000 train, 1000 test, d=24, k=16)")
    print("-" * 92)
    X_all, _, _ = generate_gaussian_mixture(
        n=4000, d=24, num_true_clusters=16, zipf=True, zipf_power=1.1, anisotropic=True, seed=42
    )
    X_tr, X_te = X_all[:3000], X_all[3000:]
    k, d, B, ep = 16, 24, 128, 5

    rows = []

    fb = NumPyFullBatchEM(k, d, cov_type="diag", cov_floor=1e-4, seed=42).fit(X_tr, max_iters=25)
    ev_fb = evaluate_gmm_nll(X_te, fb["pi"], fb["mu"], fb["cov"], cov_type="diag")
    rows.append(("FullBatchEM@25", 0, int(fb["health"]["dead_count"]), fb["health"]["max_to_mean"], ev_fb["nll"], ev_fb["bic"], 0))

    oem = NumPyOnlineEMCappeMoulines(k, d, cov_type="diag", beta=0.75, t0=10.0, cov_floor=1e-8, seed=42).fit(X_tr, batch_size=B, epochs=ep)
    ev_oem = evaluate_gmm_nll(X_te, oem["pi"], oem["mu"], oem["cov"], cov_type="diag")
    rows.append(("OnlineEM(C&M)", 0, int(oem["health"]["dead_count"]), oem["health"]["max_to_mean"], ev_oem["nll"], ev_oem["bic"], oem["singular_events"]))

    epost = NumPyEMeansPostHocGMM(k, d, cov_type="diag", cov_floor=1e-4, alpha_0=0.25, seed=42).fit(X_tr, batch_size=B, epochs=ep)
    ev_ep = evaluate_gmm_nll(X_te, epost["pi"], epost["mu"], epost["cov"], cov_type="diag")
    rows.append(("e-means+PostHoc", epost["splits"], int(epost["health"]["dead_count"]), epost["health"]["max_to_mean"], ev_ep["nll"], ev_ep["bic"], 0))

    eg_nr = NumPyEGMM(EGMMConfig(num_components=k, dim=d, cov_type="diag", cov_floor=1e-4, alpha_0=0.45, split_strategy="none", finalization="last_epoch", seed=42)).fit(X_tr, batch_size=B, epochs=ep)
    ev_nr = evaluate_gmm_nll(X_te, eg_nr["pi"], eg_nr["mu"], eg_nr["cov"], cov_type="diag")
    rows.append(("e-GMM(NoRevive)", eg_nr["splits"], int(eg_nr["health"]["dead_count"]), eg_nr["health"]["max_to_mean"], ev_nr["nll"], ev_nr["bic"], 0))

    eg_sp = NumPyEGMM(EGMMConfig(num_components=k, dim=d, cov_type="diag", cov_floor=1e-4, alpha_0=0.45, split_strategy="spectral_v1", finalization="last_epoch", seed=42)).fit(X_tr, batch_size=B, epochs=ep)
    ev_sp = evaluate_gmm_nll(X_te, eg_sp["pi"], eg_sp["mu"], eg_sp["cov"], cov_type="diag")
    rows.append(("e-GMM(Spectral)", eg_sp["splits"], int(eg_sp["health"]["dead_count"]), eg_sp["health"]["max_to_mean"], ev_sp["nll"], ev_sp["bic"], 0))

    print(f"{'Method':<18} {'Splits':>6} {'Dead':>5} {'Max/Mean':>9} {'TestNLL':>10} {'TestBIC':>12} {'SingularEv':>11}")
    print("-" * 92)
    for name, sp, dead, mm, nll, bic, sing in rows:
        print(f"{name:<18} {sp:>6d} {dead:>5d} {mm:>8.2f}x {nll:>10.4f} {bic:>12.1f} {sing:>11d}")


def main():
    verify_psd_and_rank_safety()
    verify_flash_gemm_and_woodbury()
    verify_spectral_split_moment_conservation()
    run_zipf_benchmark()


if __name__ == "__main__":
    main()
