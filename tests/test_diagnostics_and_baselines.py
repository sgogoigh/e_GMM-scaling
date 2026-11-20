"""
Intensive diagnostic tests for preliminary understanding of failure modes,
numerical edge cases, and strict requirements for fair baseline comparisons.
"""

import pathlib
import sys
import tempfile
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.config import EGMMConfig  # noqa: E402
from e_gmm.datasets import (  # noqa: E402
    generate_gaussian_mixture,
    generate_stretched_gaussian,
    load_vector_memmap,
)
from e_gmm.metrics import compute_gmm_health, evaluate_gmm_nll  # noqa: E402
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEGMM,
    NumPyEMeansPostHocGMM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
    _softmax_from_log_joint,
)


class TestPreliminarySituationsAndFailureModes(unittest.TestCase):
    """Tests diagnosing why unregularized Online EM and naive setups fail."""

    def test_online_em_rank_deficiency_and_egmm_psd_guarantee(self):
        """When B/k < d and prior/floor are tiny, OnlineEM hits singular events; e-GMM stays strictly PSD."""
        rng = np.random.default_rng(7)
        # Repeat a tiny cluster of 2 distinct points across a high-d batch (rank <= 2 < d=16)
        pts = rng.normal(size=(2, 16))
        X_rank2 = np.vstack([pts[0]] * 16 + [pts[1]] * 16).astype(np.float64)

        oem = NumPyOnlineEMCappeMoulines(
            num_components=8, dim=16, cov_type="full", beta=0.55, t0=1.0, cov_floor=1e-12, init_sigma0=1e-4, seed=7
        )
        res_oem = oem.fit(X_rank2, batch_size=16, epochs=2)
        self.assertGreater(res_oem["singular_events"], 0)

        egmm = NumPyEGMM(
            EGMMConfig(
                num_components=8,
                dim=16,
                cov_type="full",
                cov_floor=1e-3,
                init_prior_mass=1e-10,
                alpha_0=0.95,
                split_strategy="none",
                seed=7,
            )
        )
        egmm.fit(X_rank2, batch_size=16, epochs=2)
        _, _, cov, _ = egmm.readout("main_ema")
        min_eig = min(float(np.min(np.linalg.eigvalsh(cov[j]))) for j in range(8))
        self.assertGreaterEqual(min_eig, 1e-3 * 0.9999)

    def test_robbins_monro_beta_and_t0_sensitivity_diagnostic(self):
        """Verify Robbins-Monro (beta, t0) extremes cause large NLL variance across schedules."""
        X_all, _, _ = generate_gaussian_mixture(
            n=1600, d=12, num_true_clusters=8, zipf=True, anisotropic=True, seed=11
        )
        X_tr, X_te = X_all[:1200], X_all[1200:]
        nlls = []
        for beta, t0 in ((0.55, 1.0), (0.75, 10.0), (0.99, 1.0), (0.99, 200.0)):
            res = NumPyOnlineEMCappeMoulines(8, 12, cov_type="diag", beta=beta, t0=t0, cov_floor=1e-4, seed=11).fit(
                X_tr, batch_size=64, epochs=3
            )
            nll = evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"], cov_type="diag")["nll"]
            self.assertTrue(np.isfinite(nll))
            nlls.append(nll)
        # Fast-decay small-t0 (beta=0.99, t0=1) vs tuned schedules exhibits non-trivial spread
        self.assertGreater(max(nlls) - min(nlls), 0.1)

    def test_batch1_prior_mass_collapse_vs_spectral_recovery(self):
        """Without revival on B/k < d rotated full-cov data, e-GMM collapses; spectral_v1 recovers."""
        X_all, _, _ = generate_gaussian_mixture(
            n=1600, d=16, num_true_clusters=16, zipf=True, anisotropic=True, rotated=True, seed=3
        )
        X_tr, X_te = X_all[:1200], X_all[1200:]

        no_rev = NumPyEGMM(
            EGMMConfig(
                num_components=16, dim=16, cov_type="full", init_prior_mass=1e-10, alpha_0=0.35, split_strategy="none", seed=3
            )
        ).fit(X_tr, batch_size=64, epochs=4)
        sp_rev = NumPyEGMM(
            EGMMConfig(
                num_components=16, dim=16, cov_type="full", init_prior_mass=1.0, alpha_0=0.35, split_strategy="spectral_v1", seed=3
            )
        ).fit(X_tr, batch_size=64, epochs=4)

        self.assertGreater(no_rev["health"]["dead_count"], sp_rev["health"]["dead_count"])
        nll_no = evaluate_gmm_nll(X_te, no_rev["pi"], no_rev["mu"], no_rev["cov"], cov_type="full")["nll"]
        nll_sp = evaluate_gmm_nll(X_te, sp_rev["pi"], sp_rev["mu"], sp_rev["cov"], cov_type="full")["nll"]
        self.assertLess(nll_sp, nll_no)

    def test_catastrophic_cancellation_large_mean_offset(self):
        """Large mean offset ||mu|| = 1e4 with small variance must remain finite and strictly PSD."""
        rng = np.random.default_rng(21)
        X_offset = (10_000.0 + rng.normal(scale=0.1, size=(300, 12))).astype(np.float64)
        for ctype in ("diag", "full"):
            cfg = EGMMConfig(num_components=4, dim=12, cov_type=ctype, cov_floor=1e-4, alpha_0=0.3, seed=21)
            res = NumPyEGMM(cfg).fit(X_offset, batch_size=50, epochs=3)
            ev = evaluate_gmm_nll(X_offset, res["pi"], res["mu"], res["cov"], cov_type=ctype)
            self.assertTrue(np.isfinite(ev["nll"]))
            self.assertGreater(res["health"]["mean_trace"], 0.0)

    def test_cov_floor_sensitivity_and_condition_diagnostics(self):
        """Verify compute_gmm_health accurately tracks floor_hit_rate and conditioning across lambda_0."""
        X_s, _ = generate_stretched_gaussian(n=600, d=16, exponent=2.0, seed=13)
        res_small = NumPyEGMM(
            EGMMConfig(num_components=4, dim=16, cov_type="diag", cov_floor=1e-6, seed=13)
        ).fit(X_s, batch_size=100, epochs=3)
        res_large = NumPyEGMM(
            EGMMConfig(num_components=4, dim=16, cov_type="diag", cov_floor=2.0, seed=13)
        ).fit(X_s, batch_size=100, epochs=3)

        self.assertGreater(res_large["health"]["floor_hit_rate"], res_small["health"]["floor_hit_rate"])
        self.assertLess(res_large["health"]["max_condition_number"], res_small["health"]["max_condition_number"])


class TestBaselineComparisonRequirements(unittest.TestCase):
    """Contracts ensuring fair, apples-to-apples baseline comparisons."""

    def test_shared_evaluator_and_parameter_accounting_contract(self):
        """All comparators must output valid normalized distributions and consistent BIC/AIC param counts."""
        X_all, _, _ = generate_gaussian_mixture(n=800, d=10, num_true_clusters=6, anisotropic=True, seed=5)
        X_tr, X_te = X_all[:600], X_all[600:]
        k, d = 6, 10

        models = [
            ("full_batch", NumPyFullBatchEM(k, d, cov_type="diag", cov_floor=1e-4, seed=5).fit(X_tr, max_iters=10)),
            ("online_em", NumPyOnlineEMCappeMoulines(k, d, cov_type="diag", cov_floor=1e-4, seed=5).fit(X_tr, batch_size=60, epochs=3)),
            ("emeans_posthoc", NumPyEMeansPostHocGMM(k, d, cov_type="diag", cov_floor=1e-4, seed=5).fit(X_tr, batch_size=60, epochs=3)),
            ("egmm", NumPyEGMM(EGMMConfig(num_components=k, dim=d, cov_type="diag", cov_floor=1e-4, seed=5)).fit(X_tr, batch_size=60, epochs=3)),
        ]
        param_counts = set()
        for name, res in models:
            self.assertAlmostEqual(float(np.sum(res["pi"])), 1.0, places=6, msg=name)
            self.assertTrue(np.all(res["cov"] >= 1e-4 * 0.999), msg=name)
            ev = evaluate_gmm_nll(X_te, res["pi"], res["mu"], res["cov"], cov_type="diag")
            self.assertTrue(np.isfinite(ev["nll"]), msg=name)
            param_counts.add(int(ev["n_effective_params"]))
        # Every diag comparator at fixed (k, d) must share the exact same effective parameter count in BIC/AIC
        self.assertEqual(len(param_counts), 1)

    def test_overlapping_anisotropic_regime_favors_soft_egmm_over_posthoc(self):
        """On overlapping rotated anisotropic Gaussians, soft e-GMM must strictly beat hard e-means+PostHoc."""
        X_all, _, _ = generate_gaussian_mixture(
            n=2400,
            d=12,
            num_true_clusters=8,
            zipf=False,
            anisotropic=True,
            rotated=True,
            sigma_mu_sq=0.20,
            sigma_w_sq=0.80,
            seed=31,
        )
        X_tr, X_te = X_all[:1800], X_all[1800:]

        res_post = NumPyEMeansPostHocGMM(8, 12, cov_type="full", cov_floor=1e-4, alpha_0=0.35, seed=31).fit(
            X_tr, batch_size=64, epochs=5
        )
        res_egmm = NumPyEGMM(
            EGMMConfig(num_components=8, dim=12, cov_type="full", cov_floor=1e-4, alpha_0=0.35, finalization="last_epoch", seed=31)
        ).fit(X_tr, batch_size=64, epochs=5)

        nll_post = evaluate_gmm_nll(X_te, res_post["pi"], res_post["mu"], res_post["cov"], cov_type="full")["nll"]
        nll_egmm = evaluate_gmm_nll(X_te, res_egmm["pi"], res_egmm["mu"], res_egmm["cov"], cov_type="full")["nll"]
        self.assertLess(nll_egmm, nll_post - 0.3)

    def test_low_rank_diag_captures_rotated_correlations_below_diag_nll(self):
        """Low-Rank+Diag (Woodbury) must beat Diagonal on rotated correlated embeddings with fewer params than Full."""
        X_all, _, _ = generate_gaussian_mixture(
            n=2200, d=16, num_true_clusters=6, anisotropic=True, rotated=True, seed=41
        )
        X_tr, X_te = X_all[:1600], X_all[1600:]

        res_d = NumPyEGMM(EGMMConfig(num_components=6, dim=16, cov_type="diag", alpha_0=0.35, seed=41)).fit(
            X_tr, batch_size=128, epochs=4
        )
        res_lr = NumPyEGMM(
            EGMMConfig(num_components=6, dim=16, cov_type="low_rank_diag", low_rank=4, alpha_0=0.35, seed=41)
        ).fit(X_tr, batch_size=128, epochs=4)
        res_f = NumPyEGMM(EGMMConfig(num_components=6, dim=16, cov_type="full", alpha_0=0.35, seed=41)).fit(
            X_tr, batch_size=128, epochs=4
        )

        ev_d = evaluate_gmm_nll(X_te, res_d["pi"], res_d["mu"], res_d["cov"], cov_type="diag")
        ev_lr = evaluate_gmm_nll(X_te, res_lr["pi"], res_lr["mu"], res_lr["cov"], cov_type="low_rank_diag", U=res_lr["U"])
        ev_f = evaluate_gmm_nll(X_te, res_f["pi"], res_f["mu"], res_f["cov"], cov_type="full")

        self.assertLess(ev_lr["nll"], ev_d["nll"] - 0.5)
        self.assertLess(ev_d["n_effective_params"], ev_lr["n_effective_params"])
        self.assertLess(ev_lr["n_effective_params"], ev_f["n_effective_params"])

    def test_dual_finalization_train_vs_test_contract(self):
        """last_epoch optimizes Train NLL; slow_ema reduces small-n/k Test NLL overfitting."""
        X_tr, _ = generate_stretched_gaussian(n=800, d=16, exponent=1.0, seed=51)
        X_te, _ = generate_stretched_gaussian(n=800, d=16, exponent=1.0, seed=52)
        m = NumPyEGMM(
            EGMMConfig(num_components=16, dim=16, cov_type="diag", alpha_0=0.45, slow_ema_alpha=0.05, seed=51)
        )
        m.fit(X_tr, batch_size=64, epochs=4)

        pi_l, mu_l, cov_l, _ = m.readout("last_epoch")
        pi_s, mu_s, cov_s, _ = m.readout("slow_ema")
        tr_last = evaluate_gmm_nll(X_tr, pi_l, mu_l, cov_l)["nll"]
        tr_slow = evaluate_gmm_nll(X_tr, pi_s, mu_s, cov_s)["nll"]
        te_last = evaluate_gmm_nll(X_te, pi_l, mu_l, cov_l)["nll"]
        te_slow = evaluate_gmm_nll(X_te, pi_s, mu_s, cov_s)["nll"]

        self.assertLessEqual(tr_last, tr_slow)
        self.assertLessEqual(te_slow, te_last + 0.05)

    def test_top_m_responsibility_sparsity_and_normalization(self):
        """Top-m softmax must sum to 1.0 per row and keep at most m non-zeros."""
        rng = np.random.default_rng(61)
        logits = rng.normal(scale=2.0, size=(32, 20))
        r, lse = _softmax_from_log_joint(logits, top_m=5)
        self.assertEqual(lse.shape, (32,))
        np.testing.assert_allclose(np.sum(r, axis=1), np.ones(32), atol=1e-12)
        self.assertTrue(np.all(np.sum(r > 0.0, axis=1) <= 5))

    def test_offline_memmap_readers_npy_fvecs_bvecs(self):
        """Verify local .npy, .fvecs, and .bvecs readers without any network access."""
        with tempfile.TemporaryDirectory() as tmp:
            p_dir = pathlib.Path(tmp)
            arr = np.arange(24, dtype=np.float32).reshape(6, 4)

            npy_p = p_dir / "vec.npy"
            np.save(npy_p, arr)
            np.testing.assert_allclose(load_vector_memmap(str(npy_p), max_rows=4), arr[:4])

            fvecs_p = p_dir / "vec.fvecs"
            with fvecs_p.open("wb") as fh:
                for row in arr:
                    fh.write(np.int32(4).tobytes())
                    fh.write(row.astype(np.float32).tobytes())
            np.testing.assert_allclose(load_vector_memmap(str(fvecs_p)), arr)

            bvecs_p = p_dir / "vec.bvecs"
            u8 = (arr * 3).astype(np.uint8)
            with bvecs_p.open("wb") as fh:
                for row in u8:
                    fh.write(np.int32(4).tobytes())
                    fh.write(row.tobytes())
            np.testing.assert_array_equal(load_vector_memmap(str(bvecs_p)), u8)


if __name__ == "__main__":
    unittest.main()
