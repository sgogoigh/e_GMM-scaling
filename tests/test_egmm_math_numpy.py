"""Offline mathematical & algorithmic unit tests (runnable with Python stdlib + NumPy)."""

import math
import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import DATASET_PRESETS, EGMMConfig  # noqa: E402
from e_gmm.datasets import (  # noqa: E402
    generate_gaussian_mixture,
    generate_stretched_gaussian,
    generate_uniform_cube,
)
from e_gmm.metrics import (  # noqa: E402
    compute_gmm_health,
    evaluate_gmm_nll,
    gaussian_log_prob_diag,
    gaussian_log_prob_full,
    gaussian_log_prob_low_rank_diag,
)
from e_gmm.numpy_egmm import NumPyEGMM, NumPyEMeansPostHocGMM, NumPyOnlineEMCappeMoulines  # noqa: E402
from e_gmm.scaling_law import (  # noqa: E402
    DimensionScalingObservation,
    compute_nll_regret,
    evaluate_lodo_d,
    evaluate_loso_n,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
)
from e_gmm.schedule import cosine_alpha, robbins_monro_gamma  # noqa: E402


class TestEGMMMathAndPrimitives(unittest.TestCase):
    def test_presets_present(self):
        for key in ("tiny-debug", "sift-1m", "deep-1m", "deep-1b", "dinov2-100m", "imagenet-1k"):
            self.assertIn(key, DATASET_PRESETS)

    def test_cosine_and_robbins_monro_schedules(self):
        a_start = cosine_alpha(0, 101, alpha_0=0.4, eps_floor=0.1)
        a_mid = cosine_alpha(50, 101, alpha_0=0.4, eps_floor=0.1)
        a_end = cosine_alpha(100, 101, alpha_0=0.4, eps_floor=0.1)
        self.assertAlmostEqual(a_start, 0.4, places=6)
        self.assertAlmostEqual(a_mid, 0.4 * 0.55, places=6)
        self.assertAlmostEqual(a_end, 0.04, places=6)
        self.assertGreater(robbins_monro_gamma(0, beta=0.75, t0=10.0), robbins_monro_gamma(50, beta=0.75, t0=10.0))

    def test_flash_gemm_diag_equivalence(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(32, 16))
        pi = rng.dirichlet(np.ones(6))
        mu = rng.normal(size=(6, 16))
        var = rng.uniform(0.3, 2.0, size=(6, 16))

        diff = X[:, None, :] - mu[None, :, :]
        lj_naive = np.log(pi)[None, :] - 0.5 * (
            np.sum((diff ** 2) / var[None, :, :], axis=2)
            + np.sum(np.log(2.0 * math.pi * var), axis=1)[None, :]
        )
        lj_flash, lse = gaussian_log_prob_diag(X, pi, mu, var)
        self.assertLess(float(np.max(np.abs(lj_flash - lj_naive))), 1e-11)
        self.assertEqual(lse.shape, (32,))

    def test_woodbury_low_rank_diag_equivalence(self):
        rng = np.random.default_rng(2)
        B, k, d, r = 24, 5, 18, 4
        X = rng.normal(size=(B, d))
        pi = rng.dirichlet(np.ones(k))
        mu = rng.normal(size=(k, d))
        d_var = rng.uniform(0.2, 1.5, size=(k, d))
        U = rng.normal(scale=0.5, size=(k, d, r))
        cov_full = np.array([np.diag(d_var[j]) + U[j] @ U[j].T for j in range(k)])

        lj_wb, _ = gaussian_log_prob_low_rank_diag(X, pi, mu, d_var, U)
        lj_full, _, fb = gaussian_log_prob_full(X, pi, mu, cov_full)
        self.assertEqual(fb, 0)
        self.assertLess(float(np.max(np.abs(lj_wb - lj_full))), 1e-11)

    def test_psd_floor_on_rank_deficient_batch(self):
        rng = np.random.default_rng(3)
        d, k, floor = 16, 4, 1e-3
        cfg = EGMMConfig(num_components=k, dim=d, cov_type="full", cov_floor=floor, split_strategy="none")
        m = NumPyEGMM(cfg)
        x_rep = np.repeat(rng.normal(size=(1, d)), repeats=10, axis=0)
        m.init_from_data(x_rep + rng.normal(scale=1e-6, size=(10, d)))
        m.step_batch(x_rep, alpha_t=0.95, allow_split=False)
        _, _, cov, _ = m.readout("main_ema")
        for j in range(k):
            min_eig = float(np.min(np.linalg.eigvalsh(cov[j])))
            self.assertGreaterEqual(min_eig, floor * 0.9999)

    def test_spectral_split_mean_and_covariance_conservation(self):
        rng = np.random.default_rng(4)
        d = 12
        cfg = EGMMConfig(num_components=4, dim=d, cov_type="full", cov_floor=1e-4, split_delta=0.5, split_strategy="spectral_v1")
        m = NumPyEGMM(cfg)
        X = rng.normal(size=(64, d))
        m.init_from_data(X)
        m.N[:] = np.array([80.0, 20.0, 20.0, 0.01])
        mu0 = rng.normal(size=(d,))
        A = rng.normal(size=(d, d))
        cov0 = (A @ A.T) / d + 0.15 * np.eye(d)
        m.S[0] = 80.0 * mu0
        m.Q[0] = 80.0 * (cov0 + np.outer(mu0, mu0))
        splits = m._revive_collapsed_components(X)
        self.assertEqual(splits, 1)
        self.assertAlmostEqual(float(m.N[0]), 40.0, places=6)
        self.assertAlmostEqual(float(m.N[3]), 40.0, places=6)
        _, mu, cov, _ = m.readout("main_ema")
        self.assertLess(float(np.max(np.abs(0.5 * (mu[0] + mu[3]) - mu0))), 1e-10)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(cov[3]))), 1e-4 * 0.999)

    def test_top_m_and_dual_finalization(self):
        X, _, _ = generate_gaussian_mixture(n=600, d=12, num_true_clusters=8, zipf=True, anisotropic=True, seed=9)
        cfg = EGMMConfig(
            num_components=8,
            dim=12,
            cov_type="diag",
            top_m=4,
            finalization="slow_ema",
            slow_ema_alpha=0.05,
            seed=9,
        )
        m = NumPyEGMM(cfg)
        res = m.fit(X, batch_size=64, epochs=3)
        ev_slow = evaluate_gmm_nll(X, *m.readout("slow_ema")[:3], cov_type="diag")
        ev_last = evaluate_gmm_nll(X, *m.readout("last_epoch")[:3], cov_type="diag")
        self.assertTrue(np.isfinite(ev_slow["nll"]))
        self.assertTrue(np.isfinite(ev_last["nll"]))
        self.assertEqual(res["health"]["dead_count"], 0.0)

    def test_dimension_aware_scaling_law_loso_lodo(self):
        obs = []
        for n in (1e4, 5e4, 1e5):
            for d in (16, 32, 64):
                for k in (16, 64):
                    for B in (128, 512):
                        a0 = 1.2 * (k ** -0.45) * (B ** 0.85) * (n ** -0.42) * (3.0 ** -0.1) * (d ** -0.25)
                        bk = 0.04 * (k ** 0.13) * (n ** 0.88) * (3.0 ** 0.81) * (d ** 0.05)
                        obs.append(DimensionScalingObservation(n, k, d, B, 3.0, a0, b_knee_opt=bk))
        fit = fit_dimension_aware_alpha0(obs, include_dim=True)
        self.assertGreater(fit.r2_log, 0.999999)
        self.assertAlmostEqual(fit.a_d, -0.25, places=5)
        loso = evaluate_loso_n(obs, include_dim=True)
        lodo = evaluate_lodo_d(obs, include_dim=True)
        self.assertLess(loso["loso_log_rmse"], 1e-6)
        self.assertLess(lodo["lodo_log_rmse"], 1e-6)

        fit_b = fit_dimension_aware_batch_knee(obs, include_dim=True)
        self.assertGreater(fit_b.r2_log, 0.999999)

        reg = compute_nll_regret(-5.0, -5.2, nll_reference=2.0)
        self.assertAlmostEqual(reg["nll_gap_nats"], 0.2, places=6)

    def test_datasets_and_baselines_run(self):
        Xs, lam = generate_stretched_gaussian(200, 10, exponent=1.0, seed=5)
        self.assertEqual(Xs.shape, (200, 10))
        self.assertAlmostEqual(float(np.mean(lam)), 1.0, places=5)
        Xu = generate_uniform_cube(200, 10, seed=5)
        self.assertEqual(Xu.shape, (200, 10))

        oem = NumPyOnlineEMCappeMoulines(4, 10, cov_type="diag", seed=5).fit(Xs, batch_size=50, epochs=2)
        epost = NumPyEMeansPostHocGMM(4, 10, cov_type="diag", seed=5).fit(Xs, batch_size=50, epochs=2)
        self.assertEqual(oem["mu"].shape, (4, 10))
        self.assertEqual(epost["mu"].shape, (4, 10))
        h = compute_gmm_health(np.array([25.0, 25.0, 25.0, 25.0]))
        self.assertAlmostEqual(h["norm_entropy"], 1.0, places=6)

    def test_low_rank_slow_ema_and_variance_mass_donor(self):
        X, _, _ = generate_gaussian_mixture(
            n=400, d=12, num_true_clusters=6, zipf=True, anisotropic=True, rotated=True, seed=17
        )
        cfg = EGMMConfig(
            num_components=6,
            dim=12,
            cov_type="low_rank_diag",
            low_rank=4,
            donor_weight_by="variance_mass",
            donor_min_mult=1.25,
            finalization="slow_ema",
            slow_ema_alpha=0.1,
            seed=17,
        )
        m = NumPyEGMM(cfg)
        res = m.fit(X, batch_size=64, epochs=3)
        self.assertIsNotNone(m.U_slow)
        assert m.U_slow is not None
        self.assertGreater(float(np.linalg.norm(m.U_slow)), 0.0)
        ev = evaluate_gmm_nll(X, res["pi"], res["mu"], res["cov"], cov_type="low_rank_diag", U=m.U_slow)
        self.assertTrue(np.isfinite(ev["nll"]))


if __name__ == "__main__":
    unittest.main()
