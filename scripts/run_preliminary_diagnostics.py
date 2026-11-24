"""
Preliminary Diagnostic Suite (Offline CPU, Pure NumPy):
Produces a structured diagnostic audit of:
(Matched baselines: every method uses the same covariance floor COV_FLOOR=1e-4 and init_sigma0=1.0 on
training-normalized data. The only floor override is DIAG 2, which is explicitly about the floor.
These are single-seed, untuned diagnostics, NOT method comparisons; see poc_02 for the tuned paired benchmark.)
  1. Cappé & Moulines (2009) Robbins-Monro (beta, t0) sensitivity vs. e-GMM cosine schedule
  2. Singularity / Cholesky instability when B/k < d without covariance floor
  3. Covariance floor lambda_0 sensitivity (floor_hit_rate, condition number, NLL)
  4. Dataset regime suitability matrix (Zipf-Separated vs. Overlapping-Anisotropic vs. Stretched)
     clarifying when each baseline (`FullBatchEM`, `OnlineEM`, `e-means+PostHoc`) is required.
"""

import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture, generate_stretched_gaussian, normalize_scale  # noqa: E402
from e_gmm.metrics import evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
)
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402

COV_FLOOR = 1e-4  # one floor for every method (DIAG 2 varies it on purpose)


def _split_norm(X_all, n_train):
    (X_tr, X_te), _ = normalize_scale(X_all[:n_train], X_all[n_train:])
    return X_tr, X_te


def diag_robbins_monro_sensitivity():
    print("\n[DIAG 1] Robbins-Monro (beta, t0) Sensitivity vs. e-GMM Cosine Schedule")
    print("-" * 78)
    X_all, _, _ = generate_gaussian_mixture(n=2400, d=16, num_true_clusters=12, zipf=True, anisotropic=True, seed=7)
    X_tr, X_te = _split_norm(X_all, 1800)

    print(f"  {'Method':<28} {'Dead':>5} {'SingEv':>7} {'TestNLL':>10}")
    for beta, t0 in ((0.55, 1.0), (0.75, 10.0), (0.95, 1.0), (0.95, 100.0)):
        res = NumPyOnlineEMCappeMoulines(12, 16, cov_type="diag", beta=beta, t0=t0, cov_floor=COV_FLOOR, init_sigma0=1.0, seed=7).fit(
            X_tr, batch_size=64, epochs=4
        )
        nll = evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"])["nll"]
        print(f"  OnlineEM(beta={beta:.2f}, t0={t0:<5.1f}) {int(res['health']['dead_count']):>5d} {res['singular_events']:>7d} {nll:>10.4f}")

    res_eg = NumPyEGMM(EGMMConfig(num_components=12, dim=16, cov_type="diag", cov_floor=COV_FLOOR, init_sigma0=1.0, alpha_0=0.35, split_strategy="spectral_v1", seed=7)).fit(
        X_tr, batch_size=64, epochs=4
    )
    nll_eg = evaluate_gmm_nll(X_te, res_eg["pi"], res_eg["mu"], res_eg["cov"])["nll"]
    print(f"  {'e-GMM(cosine + spectral_v1)':<28} {int(res_eg['health']['dead_count']):>5d} {res_eg['chol_fallbacks']:>7d} {nll_eg:>10.4f}")


def diag_floor_sensitivity():
    print("\n[DIAG 2] (floor override is the point of this diagnostic) Covariance Floor lambda_0 Sensitivity on StretchedGaussian (d=24, exp=2.0)")
    print("-" * 78)
    X_tr, _ = generate_stretched_gaussian(n=1200, d=24, exponent=2.0, seed=11)
    X_te, _ = generate_stretched_gaussian(n=600, d=24, exponent=2.0, seed=12)

    print(f"  {'cov_floor (lambda_0)':<20} {'FloorHit%':>10} {'MaxCond':>12} {'EffRank':>9} {'TestNLL':>10}")
    for floor in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        res = NumPyEGMM(EGMMConfig(num_components=8, dim=24, cov_type="diag", cov_floor=floor, alpha_0=0.3, seed=11)).fit(
            X_tr, batch_size=64, epochs=4
        )
        nll = evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"])["nll"]
        h = res["health"]
        print(
            f"  {floor:<20.1e} {h['floor_hit_rate']*100:>9.2f}% {h['max_condition_number']:>12.1f} "
            f"{h['mean_effective_rank']:>9.2f} {nll:>10.4f}"
        )


def diag_baseline_regime_matrix():
    print("\n[DIAG 3] Baseline Comparison Matrix Across Synthetic Regimes (d=16, k=12)")
    print("-" * 78)
    regimes = [
        ("Zipf-Separated", dict(zipf=True, anisotropic=True, rotated=True, sigma_mu_sq=2.0 / 3.0, sigma_w_sq=1.0 / 3.0)),
        ("Overlapping-Rot", dict(zipf=False, anisotropic=True, rotated=True, sigma_mu_sq=0.20, sigma_w_sq=0.80)),
    ]
    for rname, kw in regimes:
        X_all, _, _ = generate_gaussian_mixture(n=2400, d=16, num_true_clusters=12, seed=19, **kw)
        X_all = X_all[np.random.default_rng(19).permutation(len(X_all))]
        X_tr, X_te = _split_norm(X_all, 1800)
        fb = NumPyFullBatchEM(12, 16, cov_type="full", cov_floor=COV_FLOOR, seed=19).fit(X_tr, max_iters=20)
        oem = NumPyOnlineEMCappeMoulines(12, 16, cov_type="full", beta=0.75, cov_floor=COV_FLOOR, init_sigma0=1.0, seed=19).fit(X_tr, batch_size=64, epochs=5)
        post = NumPyEMeansPostHocGMM(12, 16, cov_type="full", cov_floor=COV_FLOOR, alpha_0=0.35, seed=19).fit(X_tr, batch_size=64, epochs=5)
        eg = NumPyEGMM(EGMMConfig(num_components=12, dim=16, cov_type="full", cov_floor=COV_FLOOR, init_sigma0=1.0, alpha_0=0.35, split_strategy="spectral_v1", seed=19)).fit(
            X_tr, batch_size=64, epochs=5
        )
        n_fb = evaluate_gmm_nll(X_te, fb["pi"], fb["mu"], fb["cov"], cov_type="full")["nll"]
        n_oem = evaluate_gmm_nll(X_te, oem["pi"], oem["mu"], oem["cov"], cov_type="full")["nll"]
        n_post = evaluate_gmm_nll(X_te, post["pi"], post["mu"], post["cov"], cov_type="full")["nll"]
        n_eg = evaluate_gmm_nll(X_te, eg["pi"], eg["mu"], eg["cov"], cov_type="full")["nll"]
        print(f"  {rname:<16} | FullBatch={n_fb:.4f} | OnlineEM={n_oem:.4f} | e-means+PH={n_post:.4f} | e-GMM={n_eg:.4f}")


def main():
    with tee_poc_logs("preliminary_diagnostics"):
        print("=" * 84)
        print("PRELIMINARY DIAGNOSTIC & BASELINE COMPARISON READINESS SUITE")
        print("=" * 84)
        diag_robbins_monro_sensitivity()
        diag_floor_sensitivity()
        diag_baseline_regime_matrix()
        print("\n[Baseline Requirements Checklist]")
        print("  1. Always share exact held-out NLL/BIC evaluator (`evaluate_gmm_nll`) across comparators.")
        print("  2. Match epoch budget (T passes over n) and batch size B across mini-batch methods.")
        print("  3. Include BOTH separated-cluster and overlapping-anisotropic regimes when benchmarking against e-means+PostHoc.")
        print("  4. Log singular/floor-hit events alongside NLL so OnlineEM instability is visible, not hidden by silent jitter.")


if __name__ == "__main__":
    main()
