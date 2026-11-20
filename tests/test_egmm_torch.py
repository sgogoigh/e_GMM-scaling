"""PyTorch unit tests for Flash-GMM, Woodbury low-rank, Top-m sparsity, and Spectral Splitting."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment.")
class TestEGMMTorch(unittest.TestCase):
    def test_flash_gmm_streaming_parity(self):
        from e_gmm.triton_kernels import flash_gmm_diag_log_probs

        torch.manual_seed(0)
        X = torch.randn(64, 16)
        pi = torch.softmax(torch.randn(36), dim=0)
        mu = torch.randn(36, 16)
        var = torch.rand(36, 16) + 0.2

        r_full, _, _, lse_full = flash_gmm_diag_log_probs(X, pi, mu, var, chunk_k=36)
        r_chunk, _, _, lse_chunk = flash_gmm_diag_log_probs(X, pi, mu, var, chunk_k=10)
        self.assertLess(float((lse_full - lse_chunk).abs().max()), 1e-5)
        self.assertLess(float((r_full - r_chunk).abs().max()), 1e-5)

    def test_torch_egmm_all_cov_modes_and_top_m(self):
        from e_gmm.config import EGMMConfig
        from e_gmm.torch_egmm import TorchEGMM

        torch.manual_seed(1)
        X = torch.randn(128, 12)
        for cov_type, top_m in [("diag", 4), ("diag", None), ("low_rank_diag", None), ("full", None)]:
            cfg = EGMMConfig(num_components=8, dim=12, cov_type=cov_type, low_rank=4, top_m=top_m, chunk_k=4)
            m = TorchEGMM(cfg)
            nll = m.step_batch(X, alpha_t=0.25, record_last_epoch=True)
            self.assertTrue(torch.isfinite(torch.tensor(nll)))
            pi, mu, cov = m.readout("main_ema")
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
            self.assertEqual(mu.shape, (8, 12))
            self.assertTrue(torch.isfinite(cov).all())

    def test_torch_spectral_split_revives_dead(self):
        from e_gmm.config import EGMMConfig
        from e_gmm.torch_egmm import TorchEGMM

        torch.manual_seed(2)
        X = torch.randn(64, 8)
        cfg = EGMMConfig(num_components=4, dim=8, cov_type="full", split_interval=1, split_tau=0.2)
        m = TorchEGMM(cfg)
        m.init_from_data(X)
        m.N.copy_(torch.tensor([90.0, 30.0, 30.0, 0.01]))
        nll = m.step_batch(X, alpha_t=0.1, allow_split=True)
        self.assertTrue(torch.isfinite(torch.tensor(nll)))
        self.assertGreaterEqual(m.splits_done, 1)

    def test_torch_egmm_fit_parity(self):
        from e_gmm.config import EGMMConfig
        from e_gmm.torch_egmm import TorchEGMM

        torch.manual_seed(3)
        X = torch.randn(256, 8)
        cfg = EGMMConfig(num_components=4, dim=8, cov_type="low_rank_diag", low_rank=3, donor_weight_by="variance_mass")
        m = TorchEGMM(cfg)
        res = m.fit(X, batch_size=64, epochs=2)
        self.assertEqual(res["mu"].shape, (4, 8))
        self.assertIsNotNone(m.U_slow)


if __name__ == "__main__":
    unittest.main()
