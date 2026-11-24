"""
Checks for the 2026-11-25 review fixes (docs/review/2026-11-25-FIX-PLAN.md, units U1-U8).
One class per unit; each class is the "Verified by" check named in the plan.
"""

import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.benchmark import (  # noqa: E402
    ALPHA_EXTEND,
    MethodSpec,
    RunContext,
    default_methods,
    run_benchmark,
    tune_method,
)
from e_gmm.datasets import (  # noqa: E402
    estimate_intrinsic_dim_twonn,
    find_vector_file,
    generate_gaussian_mixture,
    load_vector_memmap,
    load_vectors,
    normalize_scale,
    pca_project,
    random_splits,
)
from e_gmm.metrics import compute_gmm_health, gaussian_log_prob_diag  # noqa: E402
from e_gmm.numpy_egmm import (  # noqa: E402
    NumPyEMeansThenEM,
    NumPyFullBatchEM,
    NumPyOnlineEMCappeMoulines,
)
from e_gmm.scaling_law import (  # noqa: E402
    DimensionScalingObservation,
    evaluate_leave_one_out,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
    fit_power_law,
    nll_at_alpha,
)
from e_gmm.schedule import predict_d_aware_alpha0, predict_d_aware_batch_knee  # noqa: E402
from e_gmm.stats import equivalence_verdict, paired_comparison, sign_flip_p_value, verdict  # noqa: E402
from e_gmm.sweep import (  # noqa: E402
    SweepGrid,
    alpha_ladder_search,
    batch_knee_search,
    power_of_two_batches,
    run_sweep,
    synthetic_data_fn,
)


def _mixture_split(seed, n_tr=300, n_val=150, n_te=150, d=4, k=3):
    X, _, _ = generate_gaussian_mixture(n=n_tr + n_val + n_te, d=d, num_true_clusters=k, anisotropic=True, seed=seed)
    (a, b, c), _ = normalize_scale(X[:n_tr], X[n_tr : n_tr + n_val], X[n_tr + n_val :])
    return a, b, c


class TestStats(unittest.TestCase):
    def test_exact_sign_flip(self):
        # All 10 diffs positive: only the all-+ and all-- sign patterns are as extreme.
        self.assertAlmostEqual(sign_flip_p_value(np.ones(10)), 2.0 / 1024.0)
        self.assertEqual(sign_flip_p_value(np.zeros(5)), 1.0)

    def test_paired_comparison_and_verdicts(self):
        ref = np.linspace(1.0, 2.0, 10)
        res = paired_comparison(ref - 0.05, ref)
        self.assertAlmostEqual(res["mean_diff"], -0.05)
        self.assertEqual(res["wins"], 10.0)
        self.assertLess(res["p_value"], 0.01)
        self.assertEqual(verdict(res["mean_diff"], res["p_value"], 0.01), "Supported")
        self.assertEqual(verdict(+0.05, 0.002, 0.01), "Contradicted")
        self.assertTrue(verdict(-0.001, 0.002, 0.01).startswith("Inconclusive (significant"))
        self.assertEqual(verdict(-0.5, 0.3, 0.01), "Inconclusive")
        self.assertEqual(verdict(-0.5, 0.001, 0.01, min_n=10, n=3), "Inconclusive (too few seeds)")
        with self.assertRaises(ValueError):
            paired_comparison([1.0, 2.0], [1.0])

    def test_monte_carlo_p_is_never_zero_and_equivalence(self):
        p = sign_flip_p_value(np.ones(40), n_mc=999)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0 / 1000 + 1e-12)
        tight = equivalence_verdict(np.full(10, 0.001) + np.linspace(-1e-4, 1e-4, 10), margin=0.01)
        self.assertEqual(tight["verdict"], "Supported")
        far = equivalence_verdict(np.full(10, 0.05) + np.linspace(-1e-3, 1e-3, 10), margin=0.01)
        self.assertEqual(far["verdict"], "Contradicted")
        self.assertEqual(equivalence_verdict([0.0] * 3, 0.01, min_n=10)["verdict"], "Inconclusive (too few seeds)")


def _obs(k, B, n, T, d, alpha, censored=False, curve=None, knee=None, knee_censored=False):
    return DimensionScalingObservation(
        num_train=n, num_components=k, dim=d, batch_size=B, epochs=T, alpha_0_opt=alpha,
        alpha_censored=censored, alpha_curve=curve, b_knee_opt=knee, knee_censored=knee_censored,
    )


class TestScalingLaw(unittest.TestCase):
    def test_constant_feature_is_unidentified_not_fitted(self):
        obs = [_obs(k, B, n, 3, 32, 0.5 * k ** -0.4 * B ** 0.8 * n ** -0.4)
               for k in (4, 16) for B in (32, 128) for n in (1000, 4000)]
        fit = fit_dimension_aware_alpha0(obs, include_dim=True)
        self.assertIsNone(fit.a_t)
        self.assertIsNone(fit.a_d)
        self.assertIn("T", fit.unidentified)
        self.assertIn("d", fit.unidentified)
        self.assertAlmostEqual(fit.a_k, -0.4, places=6)

    def test_r2_is_none_without_variance_and_rank_deficiency_raises(self):
        self.assertIsNone(fit_power_law({"x": [1.0, 2.0, 4.0]}, [3.0, 3.0, 3.0]).r2_log)
        with self.assertRaises(ValueError):
            fit_power_law({"a": [1.0, 2.0, 4.0, 8.0], "b": [2.0, 4.0, 8.0, 16.0]}, [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            fit_power_law({"a": [1.0]}, [1.0])

    def test_censored_observations_are_excluded(self):
        obs = [_obs(k, B, n, T, 16, 0.3 * k ** -0.5 * B ** 0.9 * n ** -0.5 * T ** -0.1)
               for k in (4, 16) for B in (32, 128) for n in (1000, 4000) for T in (2, 4)]
        obs.append(_obs(8, 64, 2000, 3, 16, 1.0, censored=True))
        fit = fit_dimension_aware_alpha0(obs, include_dim=False)
        self.assertEqual(fit.n_censored_excluded, 1)
        self.assertAlmostEqual(fit.a_t, -0.1, places=6)

        knees = [_obs(k, 64, n, T, 16, 0.1, knee=0.01 * k ** 0.2 * n ** 0.9 * T ** 0.5)
                 for k in (4, 16) for n in (1000, 4000) for T in (2, 4)]
        knees.append(_obs(8, 64, 2000, 3, 16, 0.1, knee=512.0, knee_censored=True))
        kfit = fit_dimension_aware_batch_knee(knees, include_dim=False)
        self.assertEqual(kfit.n_censored_excluded, 1)
        self.assertAlmostEqual(kfit.b_n, 0.9, places=6)

    def test_nll_at_alpha_and_regret_points(self):
        val, ex = nll_at_alpha(([0.1, 0.2, 0.4], [3.0, 2.0, 3.0]), 0.2)
        self.assertAlmostEqual(val, 2.0)
        self.assertFalse(ex)
        self.assertTrue(nll_at_alpha(([0.1, 0.2], [1.0, 2.0]), 0.8)[1])

        obs = []
        for k in (4, 16):
            for B in (32, 128):
                for n in (1000, 4000):
                    for T in (2, 4):
                        for d in (8, 32):
                            a = 0.3 * k ** -0.5 * B ** 0.9 * n ** -0.5 * T ** -0.1
                            alphas = [a / 2, a, a * 2]
                            obs.append(_obs(k, B, n, T, d, a, curve=(alphas, [d * 1.1, d * 1.0, d * 1.1])))
        res = evaluate_leave_one_out(obs, "dim", include_dim=False, return_points=True)
        self.assertLess(res["log_rmse"], 1e-6)
        self.assertLess(abs(res["regret_nats_per_dim_mean"]), 1e-9)
        self.assertEqual(len(res["points"]), len(obs))


class TestSchedule(unittest.TestCase):
    def test_d_exponents_default_to_zero(self):
        self.assertEqual(predict_d_aware_alpha0(64, 256, 1e6, 3, 16), predict_d_aware_alpha0(64, 256, 1e6, 3, 768))
        self.assertEqual(predict_d_aware_batch_knee(64, 1e6, 3, 16), predict_d_aware_batch_knee(64, 1e6, 3, 768))

    def test_paper_batch_rule(self):
        raw = 0.035 * 64 ** 0.13 * 4e6 ** 0.88 * 3 ** 0.81
        b = predict_d_aware_batch_knee(64, 4e6, 3, 96, round_pow2=False)
        self.assertEqual(b, int(round(raw / 8.0)))
        self.assertLessEqual(predict_d_aware_batch_knee(16, 10, 50, 8), 10)
        # No hidden B >= k clamp.
        self.assertLess(predict_d_aware_batch_knee(4096, 200, 1, 8), 4096)


class TestMetrics(unittest.TestCase):
    def test_diag_log_prob_translation_invariant_and_float32_stable(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 8))
        mu = rng.normal(size=(3, 8))
        var = rng.uniform(0.5, 2.0, size=(3, 8))
        pi = np.array([0.2, 0.3, 0.5])
        lj, lse = gaussian_log_prob_diag(X, pi, mu, var)
        shift = 1e4
        lj2, lse2 = gaussian_log_prob_diag(X + shift, pi, mu + shift, var, compute_dtype=np.float32)
        np.testing.assert_allclose(lse2, lse, atol=1e-2)
        np.testing.assert_allclose(lj2, lj, atol=1e-2)

    def test_floor_hit_rate_counts_floor_dominated_components(self):
        floor = 1e-4
        cov = np.full((4, 3), 1.0)
        cov[0] = floor * 1.01  # raw variance ~0 -> floor-dominated
        cov[1] = floor * 1.5
        h = compute_gmm_health(np.full(4, 0.25), cov, cov_type="diag", cov_floor=floor)
        self.assertAlmostEqual(h["floor_hit_rate"], 0.5)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.X = _mixture_split(3, n_tr=400)[0]

    def test_fullbatch_chunking_is_exact(self):
        a = NumPyFullBatchEM(3, 4, cov_type="full", seed=1, chunk_size=10_000).fit(self.X, max_iters=5, tol=0.0)
        b = NumPyFullBatchEM(3, 4, cov_type="full", seed=1, chunk_size=37).fit(self.X, max_iters=5, tol=0.0)
        np.testing.assert_allclose(a["mu"], b["mu"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a["nll_history"], b["nll_history"], rtol=1e-12)

    def test_fullbatch_inits(self):
        res = NumPyFullBatchEM(3, 4, cov_type="diag", seed=1, init="emeans").fit(self.X, max_iters=5)
        self.assertTrue(np.all(np.isfinite(res["mu"])))
        pi0, mu0, cov0 = np.full(3, 1 / 3), self.X[:3].astype(np.float64), np.ones((3, 4))
        res0 = NumPyFullBatchEM(3, 4, cov_type="diag", seed=1).fit(self.X, max_iters=0, init_params=(pi0, mu0, cov0))
        np.testing.assert_allclose(res0["mu"], mu0)
        with self.assertRaises(ValueError):
            NumPyFullBatchEM(3, 4, init="kmeans++")
        with self.assertRaises(ValueError):
            NumPyFullBatchEM(3, 4, cov_type="low_rank_diag")

    def test_emeans_then_em_and_matched_online_floor(self):
        res = NumPyEMeansThenEM(3, 4, cov_type="full", em_iters=3, seed=2).fit(self.X, batch_size=32, epochs=2)
        self.assertEqual(len(res["nll_history"]), 3)
        on = NumPyOnlineEMCappeMoulines(3, 4, cov_type="diag", seed=2)
        self.assertEqual(on.cov_floor, 1e-4)
        out = on.fit(self.X, batch_size=32, epochs=2)
        self.assertGreaterEqual(float(out["cov"].min()), 1e-4)


class TestDatasets(unittest.TestCase):
    def test_bigann_and_texmex_readers(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(20, 6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "x.fbin"
            with open(p, "wb") as fh:
                np.array([20, 6], dtype=np.uint32).tofile(fh)
                A.tofile(fh)
            np.testing.assert_array_equal(np.asarray(load_vectors(str(p))), A)
            self.assertEqual(load_vectors(str(p), max_rows=5).shape, (5, 6))
            self.assertEqual(find_vector_file(tmp, "x"), p)
            with self.assertRaises(FileNotFoundError):
                find_vector_file(tmp, "missing")

            short = pathlib.Path(tmp) / "short.fbin"
            with open(short, "wb") as fh:
                np.array([20, 6], dtype=np.uint32).tofile(fh)
                A[:10].tofile(fh)
            with self.assertRaises(ValueError):
                load_vectors(str(short))

            U = rng.integers(0, 255, size=(7, 5), dtype=np.uint8)
            pu = pathlib.Path(tmp) / "x.u8bin"
            with open(pu, "wb") as fh:
                np.array([7, 5], dtype=np.uint32).tofile(fh)
                U.tofile(fh)
            np.testing.assert_array_equal(np.asarray(load_vector_memmap(str(pu))), U)

            bad = pathlib.Path(tmp) / "bad.fvecs"
            recs = np.zeros((3, 5), dtype=np.float32)
            heads = np.array([4, 4, 3], dtype=np.int32)
            recs[:, 0] = heads.view(np.float32)
            recs.tofile(bad)
            with self.assertRaises(ValueError):
                load_vectors(str(bad))
            csv = pathlib.Path(tmp) / "x.csv"
            csv.write_text("1,2\n")
            with self.assertRaises(ValueError):
                load_vectors(str(csv))
        with self.assertRaises(FileNotFoundError):
            load_vectors("definitely/not/here.npy")

    def test_random_splits_are_disjoint_and_deduplicated(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(100, 3)).astype(np.float32)
        arr = np.concatenate([base, base])  # every row appears twice
        (tr, te), info = random_splits(arr, [60, 30], seed=4)
        train_rows = {r.tobytes() for r in tr}
        self.assertFalse(any(r.tobytes() in train_rows for r in te))
        self.assertGreater(info["dups_replaced"], 0)
        with self.assertRaises(ValueError):
            random_splits(arr, [150, 100])

    def test_normalize_pca_and_twonn(self):
        rng = np.random.default_rng(2)
        X = (rng.normal(size=(1500, 3)) @ rng.normal(size=(3, 20)) * 50.0 + 7.0).astype(np.float32)
        (Xn,), info = normalize_scale(X)
        self.assertAlmostEqual(float(np.mean(Xn.astype(np.float64).var(axis=0))), 1.0, places=4)
        self.assertGreater(info["scale"], 1.0)
        (P,), pinfo = pca_project(Xn, 3)
        self.assertEqual(P.shape, (1500, 3))
        self.assertGreater(pinfo["explained_variance_ratio"], 0.999)
        with self.assertRaises(ValueError):
            pca_project(Xn, 21)
        est = estimate_intrinsic_dim_twonn(Xn, max_points=1500)
        self.assertTrue(2.4 < est < 3.6, est)


class TestBenchmark(unittest.TestCase):
    def test_tuned_paired_benchmark(self):
        methods, skipped = default_methods(
            egmm_alphas=(0.1, 0.35), online_betas=(0.6, 0.9), online_t0s=(10.0,), emeans_alphas=(0.2,),
            include_sklearn=False,
        )
        res = run_benchmark(_mixture_split, k=3, cov_type="full", batch_size=32, epochs=2, seeds=[1, 2, 3],
                            methods=methods, skipped=skipped)
        self.assertEqual(set(res["summary"]), {m.name for m in methods})
        self.assertIn("online_em", res["paired_vs_reference"])
        self.assertNotIn("egmm_spectral", res["paired_vs_reference"])
        self.assertEqual(res["paired_vs_reference"]["online_em"]["n"], 3.0)
        self.assertIn(res["tuning"]["online_em"]["best_params"]["beta"], (0.6, 0.9))
        self.assertEqual(res["setup"]["unit"], "nats per dimension (test NLL / d)")
        with self.assertRaises(ValueError):
            run_benchmark(_mixture_split, k=3, cov_type="full", batch_size=32, epochs=2, seeds=[1], tuning_seed=1,
                          methods=methods)
        with self.assertRaises(ValueError):
            run_benchmark(_mixture_split, k=3, cov_type="low_rank_diag", batch_size=32, epochs=2, seeds=[1],
                          methods=methods)

    def test_tuning_extends_grid_past_an_edge(self):
        # Validation NLL of a single N(0, var) with var = alpha / 0.8 on unit-variance data: optimum at alpha = 0.8.
        rng = np.random.default_rng(0)
        X_val = rng.standard_normal((4000, 2))
        X_val = (X_val - X_val.mean(0)) / X_val.std(0)
        ctx = RunContext(k=1, cov_type="diag", cov_floor=1e-4, batch_size=8, epochs=1, seed=0)

        def fit_for(scale):
            return lambda X, c, p: {"pi": np.ones(1), "mu": np.zeros((1, 2)), "cov": np.full((1, 2), p["alpha_0"] / scale)}

        spec = MethodSpec("toy", [{"alpha_0": 0.1}, {"alpha_0": 0.2}], fit_for(0.8), extend=ALPHA_EXTEND)
        res = tune_method(spec, X_val, X_val, ctx)
        self.assertAlmostEqual(res["best_params"]["alpha_0"], 0.8)
        self.assertEqual(res["at_grid_edge"], [])
        self.assertEqual(res["extensions"], 3)  # 0.4, 0.8, then 1.0 (clipped) confirms 0.8 is interior

        # Optimum far below the grid: extension stops after MAX_GRID_EXTENSIONS and the edge is still reported.
        spec_low = MethodSpec("toy_low", [{"alpha_0": 0.1}, {"alpha_0": 0.2}], fit_for(1e-4), extend=ALPHA_EXTEND)
        res_low = tune_method(spec_low, X_val, X_val, ctx)
        self.assertEqual(res_low["at_grid_edge"], ["alpha_0"])
        self.assertAlmostEqual(res_low["best_params"]["alpha_0"], 0.1 / 2 ** 4)

        # Without an extension rule the grid is left alone.
        spec_fixed = MethodSpec("toy_fixed", [{"alpha_0": 0.1}, {"alpha_0": 0.2}], fit_for(0.8))
        res_fixed = tune_method(spec_fixed, X_val, X_val, ctx)
        self.assertEqual((res_fixed["extensions"], res_fixed["at_grid_edge"]), (0, ["alpha_0"]))


class TestSweep(unittest.TestCase):
    def test_alpha_ladder_interior_and_censored(self):
        res = alpha_ladder_search(lambda a: (math.log(a) - math.log(0.1)) ** 2)
        self.assertFalse(res["censored"])
        self.assertAlmostEqual(res["alpha_star"], 0.1, places=6)

        low = alpha_ladder_search(lambda a: a)  # monotone increasing: wants alpha -> 0
        self.assertTrue(low["censored"])
        self.assertEqual(low["censored_side"], "low")
        self.assertLess(min(low["alphas"]), 0.02)

        high = alpha_ladder_search(lambda a: -a)
        self.assertTrue(high["censored"])
        self.assertEqual(max(high["alphas"]), 1.0)

    def test_knee_search_and_batch_range(self):
        curve = {16: 1.0, 32: 1.0, 64: 1.001, 128: 1.002, 256: 1.5}
        res = batch_knee_search(lambda b: curve[b], list(curve), dim=1, tol_nats_per_dim=0.002)
        self.assertEqual(res["knee"], 128)
        self.assertFalse(res["censored"])
        flat = batch_knee_search(lambda b: 1.0, list(curve), dim=1)
        self.assertTrue(flat["censored"])
        self.assertEqual(power_of_two_batches(1200), [16, 32, 64, 128, 256])
        with self.assertRaises(ValueError):
            power_of_two_batches(40)

    def test_tiny_sweep_varies_t_and_reports_censoring(self):
        grid = SweepGrid(dims=[4, 8], ks=[2], ns=[400], batch_sizes=[16, 32], epochs=[1, 2], seeds=[0], n_val=200,
                         ood=[{"d": 6, "k": 2, "n": 400, "B": 32, "T": 2}])
        res = run_sweep(synthetic_data_fn(), grid)
        self.assertEqual(res["censoring"]["alpha_total"], 8)
        for name in ("no_d", "ambient_d", "intrinsic_d"):
            self.assertIn(name, res["alpha_laws"])
        fit = res["alpha_laws"]["no_d"].get("fit")
        if fit is not None:  # T varies in this grid, so it must not be reported as unidentified
            self.assertNotIn("T", fit["unidentified"])
        self.assertEqual(len(res["ood"]["configs"]), 1)
        self.assertEqual(set(res["intrinsic_dims"]), {"d=4,n=400", "d=8,n=400", "d=6,n=400"})  # grid + OOD


if __name__ == "__main__":
    unittest.main()
