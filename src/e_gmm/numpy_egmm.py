"""Pure-NumPy e-GMM engine and baseline comparators (Online EM, Full-Batch EM, e-means+PostHoc)."""

import math
from typing import Dict, List, Optional, Tuple, TypedDict
import numpy as np


class EGMMFitResult(TypedDict):
    pi: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    U: Optional[np.ndarray]
    splits: int
    chol_fallbacks: int
    batch_nlls: List[float]
    health: Dict[str, float]


class OnlineEMFitResult(TypedDict):
    pi: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    singular_events: int
    health: Dict[str, float]


class FullBatchEMFitResult(TypedDict):
    pi: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    nll_history: List[float]
    health: Dict[str, float]


class EMeansPostHocFitResult(TypedDict):
    pi: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    splits: int
    health: Dict[str, float]

from .config import EGMMConfig
from .datasets import iter_minibatches
from .metrics import (
    compute_gmm_health,
    evaluate_gmm_nll,
    gaussian_log_prob_diag,
    gaussian_log_prob_full,
    gaussian_log_prob_low_rank_diag,
    logsumexp_1d,
)
from .schedule import cosine_alpha, robbins_monro_gamma


def _power_iteration_top_eig(mat: np.ndarray, steps: int = 5, rng: Optional[np.random.Generator] = None) -> Tuple[float, np.ndarray]:
    """Compute principal eigenvalue and unit eigenvector via fast power iteration."""
    d = mat.shape[0]
    if rng is None:
        v = np.ones(d, dtype=np.float64) / math.sqrt(d)
    else:
        v = rng.normal(size=(d,))
        nrm = np.linalg.norm(v)
        v = v / max(nrm, 1e-12)
    for _ in range(steps):
        mv = mat @ v
        nrm = float(np.linalg.norm(mv))
        if nrm < 1e-15:
            break
        v = mv / nrm
    lam = float(v @ (mat @ v))
    return max(lam, 0.0), v


def _softmax_from_log_joint(log_joint: np.ndarray, top_m: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (B, k) log joint probabilities into responsibilities (B, k) and per-point log-likelihoods.
    If top_m is set (< k), zero out non-top-m entries and renormalize over top-m.
    """
    B, k = log_joint.shape
    lse_full = logsumexp_1d(log_joint, axis=1, keepdims=False)
    if top_m is not None and 0 < top_m < k:
        idx = np.argpartition(log_joint, -top_m, axis=1)[:, -top_m:]
        top_vals = np.take_along_axis(log_joint, idx, axis=1)
        m_max = np.max(top_vals, axis=1, keepdims=True)
        w = np.exp(top_vals - m_max)
        w /= np.sum(w, axis=1, keepdims=True)
        r = np.zeros_like(log_joint)
        np.put_along_axis(r, idx, w, axis=1)
        return r, lse_full

    m_max = np.max(log_joint, axis=1, keepdims=True)
    w = np.exp(log_joint - m_max)
    r = w / np.sum(w, axis=1, keepdims=True)
    return r, lse_full


class NumPyEGMM:
    """
    Pure-NumPy 3-tuple EMA Gaussian Mixture Model (N_j, S_j, Q_j) with:
      - Covariance floor lambda_0 I_d
      - Cosine-annealed step size alpha(t)
      - Diagonal (Flash-GMM 2d-GEMM), Low-Rank+Diag (Woodbury), and Full (Cholesky) modes
      - Spectral component splitting along principal eigenvector v_1
      - Dual finalization: Last-Epoch Exact EM and passive Slow-EMA shadow
    """

    def __init__(self, cfg: EGMMConfig):
        self.cfg = cfg
        self.k = cfg.num_components
        self.d = cfg.dim
        self.cov_type = cfg.cov_type
        self.rank = min(cfg.low_rank, cfg.dim)
        self.rng = np.random.default_rng(cfg.seed)

        self.N: np.ndarray = np.zeros(self.k, dtype=np.float64)
        self.S: np.ndarray = np.zeros((self.k, self.d), dtype=np.float64)
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q: np.ndarray = np.zeros((self.k, self.d), dtype=np.float64)
        elif self.cov_type == "full":
            self.Q = np.zeros((self.k, self.d, self.d), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported cov_type: {self.cov_type}")

        # For low_rank_diag: Q stores diagonal 2nd moments (k, d), plus factor U (k, d, r)
        self.U: Optional[np.ndarray] = (
            np.zeros((self.k, self.d, self.rank), dtype=np.float64)
            if self.cov_type == "low_rank_diag"
            else None
        )

        # Passive Slow-EMA shadow accumulators
        self.N_slow: np.ndarray = np.zeros_like(self.N)
        self.S_slow: np.ndarray = np.zeros_like(self.S)
        self.Q_slow: np.ndarray = np.zeros_like(self.Q)
        self.U_slow: Optional[np.ndarray] = (
            np.zeros_like(self.U) if self.U is not None else None
        )

        # Last-epoch exact EM accumulators
        self.N_last: np.ndarray = np.zeros_like(self.N)
        self.S_last: np.ndarray = np.zeros_like(self.S)
        self.Q_last: np.ndarray = np.zeros_like(self.Q)

        self.initialized: bool = False
        self.cooldown = np.zeros(self.k, dtype=np.int64)
        self.step_count = 0
        self.splits_done = 0
        self.chol_fallbacks = 0

    def init_from_data(self, X: np.ndarray, mu0: Optional[np.ndarray] = None) -> None:
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        if d != self.d:
            raise ValueError(f"Expected dim={self.d}, got {d}.")

        if mu0 is None:
            replace = n < self.k
            idx = self.rng.choice(n, size=self.k, replace=replace)
            mu_init = X64[idx].copy()
        else:
            mu_init = np.asarray(mu0, dtype=np.float64).copy()

        n0 = float(self.cfg.init_prior_mass)
        s0_sq = float(self.cfg.init_sigma0) ** 2
        self.N = np.full(self.k, n0, dtype=np.float64)
        self.S = self.N[:, None] * mu_init

        if self.cov_type == "diag":
            self.Q = self.N[:, None] * (mu_init ** 2 + s0_sq)
            self.U = None
        elif self.cov_type == "low_rank_diag":
            self.Q = self.N[:, None] * (mu_init ** 2 + s0_sq)
            self.U = self.rng.normal(scale=1e-3, size=(self.k, d, self.rank))
        elif self.cov_type == "full":
            eye = np.eye(d, dtype=np.float64)[None, :, :]
            self.Q = self.N[:, None, None] * (
                mu_init[:, :, None] * mu_init[:, None, :] + s0_sq * eye
            )
            self.U = None
        else:
            raise ValueError(f"Unsupported cov_type: {self.cov_type}")

        # Passive Slow-EMA mirrors
        self.N_slow = self.N.copy()
        self.S_slow = self.S.copy()
        self.Q_slow = self.Q.copy()
        self.U_slow = self.U.copy() if self.U is not None else None
        self.reset_last_epoch_accumulators()
        self.initialized = True
        self.cooldown.fill(0)
        self.step_count = 0
        self.splits_done = 0
        self.chol_fallbacks = 0

    def reset_last_epoch_accumulators(self) -> None:
        self.N_last = np.zeros(self.k, dtype=np.float64)
        self.S_last = np.zeros((self.k, self.d), dtype=np.float64)
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q_last = np.zeros((self.k, self.d), dtype=np.float64)
        else:
            self.Q_last = np.zeros((self.k, self.d, self.d), dtype=np.float64)

    def _readout_from_tuples(
        self,
        N: np.ndarray,
        S: np.ndarray,
        Q: np.ndarray,
        U: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        N_safe = np.maximum(N, 1e-30)
        pi = N_safe / np.sum(N_safe)
        mu = S / N_safe[:, None]
        floor = float(self.cfg.cov_floor)

        if self.cov_type == "diag":
            var = np.maximum(Q / N_safe[:, None] - mu ** 2, 0.0) + floor
            return pi, mu, var, None

        if self.cov_type == "low_rank_diag":
            total_diag = np.maximum(Q / N_safe[:, None] - mu ** 2, 0.0) + floor
            if U is not None:
                u_diag = np.sum(U ** 2, axis=2)
                d_var = np.maximum(total_diag - u_diag, floor)
                return pi, mu, d_var, U
            return pi, mu, total_diag, None

        eye = np.eye(self.d, dtype=np.float64)[None, :, :]
        cov = Q / N_safe[:, None, None] - mu[:, :, None] * mu[:, None, :]
        cov = 0.5 * (cov + np.swapaxes(cov, -1, -2)) + floor * eye
        return pi, mu, cov, None

    def readout(self, mode: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read out (pi, mu, cov_or_var, U) from active, slow_ema, or last_epoch tuples."""
        use_mode = mode or self.cfg.finalization
        if use_mode == "slow_ema":
            return self._readout_from_tuples(self.N_slow, self.S_slow, self.Q_slow, self.U_slow)
        if use_mode == "last_epoch" and np.sum(self.N_last) > 1e-12:
            N_eff = np.maximum(self.N_last, 1e-12)
            # Fall back for completely empty last-epoch components to active tuples
            empty = self.N_last <= 1e-12
            S_eff = np.where(empty[:, None], self.S / np.maximum(self.N[:, None], 1e-30) * N_eff[:, None], self.S_last)
            if self.cov_type in ("diag", "low_rank_diag"):
                Q_fallback = (self.Q / np.maximum(self.N[:, None], 1e-30)) * N_eff[:, None]
                Q_eff = np.where(empty[:, None], Q_fallback, self.Q_last)
            else:
                Q_fallback = (self.Q / np.maximum(self.N[:, None, None], 1e-30)) * N_eff[:, None, None]
                Q_eff = np.where(empty[:, None, None], Q_fallback, self.Q_last)
            return self._readout_from_tuples(N_eff, S_eff, Q_eff, self.U)
        return self._readout_from_tuples(self.N, self.S, self.Q, self.U)

    def _compute_log_joint(self, X_batch: np.ndarray, pi: np.ndarray, mu: np.ndarray, cov_or_var: np.ndarray, U: Optional[np.ndarray]) -> np.ndarray:
        if self.cov_type == "diag":
            if self.cfg.e_step_mode == "flash_gemm":
                log_joint, _ = gaussian_log_prob_diag(X_batch, pi, mu, cov_or_var)
                return log_joint
            # Naive reference path
            X64 = np.asarray(X_batch, dtype=np.float64)
            var64 = np.maximum(cov_or_var, 1e-12)
            diff = X64[:, None, :] - mu[None, :, :]
            maha = np.sum((diff ** 2) / var64[None, :, :], axis=2)
            logdet = np.sum(np.log(2.0 * math.pi * var64), axis=1)
            return np.log(np.maximum(pi, 1e-30))[None, :] - 0.5 * (maha + logdet[None, :])

        if self.cov_type == "low_rank_diag":
            if U is None:
                raise ValueError("U must be initialized for cov_type='low_rank_diag'.")
            log_joint, _ = gaussian_log_prob_low_rank_diag(X_batch, pi, mu, cov_or_var, U)
            return log_joint

        log_joint, _, fb = gaussian_log_prob_full(X_batch, pi, mu, cov_or_var)
        self.chol_fallbacks += fb
        return log_joint

    def _update_low_rank_factors(
        self,
        X64: np.ndarray,
        r: np.ndarray,
        mu: np.ndarray,
        n_b: np.ndarray,
        alpha_t: float,
    ) -> None:
        """Online Oja subspace tracking for low_rank_diag U_j in O(k d r) memory."""
        if self.U is None:
            return
        floor = float(self.cfg.cov_floor)
        for j in range(self.k):
            if n_b[j] <= 1e-6:
                continue
            weights = r[:, j]
            res = X64 - mu[j][None, :]  # (B, d)
            # Extract current orthonormal basis Q_j from U_j
            u_norm = np.linalg.norm(self.U[j], axis=0)
            if np.all(u_norm > 1e-10):
                V_cur, _ = np.linalg.qr(self.U[j])
            else:
                V_cur, _ = np.linalg.qr(self.rng.normal(size=(self.d, self.rank)))

            # Batch covariance action on subspace: C_b @ V_cur = (res^T (w * (res @ V_cur))) / n_b
            proj = res.T @ (weights[:, None] * (res @ V_cur)) / max(n_b[j], 1e-8)
            # Smooth subspace update with EMA step alpha_t and sign-consistent QR
            proj_norm = max(float(np.linalg.norm(proj)), 1e-8)
            cand = (1.0 - alpha_t) * V_cur + alpha_t * (proj / proj_norm)
            q_mat, _ = np.linalg.qr(cand)
            # Align column signs to V_cur so basis never flips
            signs = np.where(np.sum(q_mat * V_cur, axis=0) >= 0.0, 1.0, -1.0)
            q_mat = q_mat * signs[None, :]

            # Directional variances along q_mat columns
            dir_vars = np.sum(weights[:, None] * ((res @ q_mat) ** 2), axis=0) / max(n_b[j], 1e-8)
            # Residual average per-coordinate variance outside top-r subspace
            tot_var_est = float(np.sum(weights[:, None] * (res ** 2)) / max(n_b[j], 1e-8))
            rem_dim = max(self.d - self.rank, 1)
            bg_var = max((tot_var_est - float(np.sum(dir_vars))) / rem_dim, floor)
            excess_vars = np.maximum(dir_vars - bg_var, 0.0)
            # Smooth directional variance rather than raw vector to prevent cancellation
            old_vars = np.sum(self.U[j] ** 2, axis=0)
            smoothed_vars = (1.0 - alpha_t) * old_vars + alpha_t * excess_vars
            self.U[j] = q_mat * np.sqrt(np.maximum(smoothed_vars, 0.0))[None, :]

    def _revive_collapsed_components(self, X64: np.ndarray) -> int:
        strategy = self.cfg.split_strategy
        if strategy == "none":
            return 0

        mean_n = float(np.mean(self.N))
        dead_ids = np.where((self.N < self.cfg.split_tau * mean_n) & (self.cooldown == 0))[0]
        if dead_ids.size == 0:
            return 0

        if strategy == "batch_resample":
            n_pick = min(int(dead_ids.size), self.cfg.max_splits_per_step, X64.shape[0])
            picks = self.rng.choice(X64.shape[0], size=n_pick, replace=False)
            for idx_d, x_idx in zip(dead_ids[:n_pick], picks):
                x_new = X64[x_idx]
                n_new = max(mean_n * 0.25, 1e-4)
                self.N[idx_d] = n_new
                self.S[idx_d] = n_new * x_new
                s0 = float(self.cfg.init_sigma0) ** 2
                if self.cov_type in ("diag", "low_rank_diag"):
                    self.Q[idx_d] = n_new * (x_new ** 2 + s0)
                else:
                    self.Q[idx_d] = n_new * (np.outer(x_new, x_new) + s0 * np.eye(self.d))
                self.cooldown[idx_d] = self.cfg.cooldown_steps
                self.splits_done += 1
            return n_pick

        donor_pool = np.where((self.N > self.cfg.donor_min_mult * mean_n) & (self.cooldown == 0))[0]
        if donor_pool.size == 0:
            donor_pool = np.where((self.N > mean_n) & (self.cooldown == 0))[0]
        if donor_pool.size == 0:
            donor_pool = np.where(self.N > mean_n)[0]
        if donor_pool.size == 0:
            return 0

        n_pairs = min(int(dead_ids.size), int(donor_pool.size), self.cfg.max_splits_per_step)
        _, mu_cur, cov_cur, U_cur = self._readout_from_tuples(self.N, self.S, self.Q, self.U)
        if self.cfg.donor_weight_by == "variance_mass":
            if self.cov_type == "diag":
                traces = np.sum(cov_cur[donor_pool], axis=1)
            elif self.cov_type == "low_rank_diag":
                u_tr = np.sum(U_cur[donor_pool] ** 2, axis=(1, 2)) if U_cur is not None else 0.0
                traces = np.sum(cov_cur[donor_pool], axis=1) + u_tr
            else:
                traces = np.trace(cov_cur[donor_pool], axis1=1, axis2=2)
            probs = np.maximum(self.N[donor_pool] * np.maximum(traces, 1e-8), 1e-12)
        else:
            probs = np.maximum(self.N[donor_pool], 1e-12)
        probs /= probs.sum()
        donors = self.rng.choice(donor_pool, size=n_pairs, replace=False, p=probs)
        delta = float(self.cfg.split_delta)
        floor = float(self.cfg.cov_floor)

        for dead_i, don_i in zip(dead_ids[:n_pairs], donors):
            mu_d = mu_cur[don_i]
            n_half = 0.5 * self.N[don_i]

            if strategy == "faiss_alt":
                # Alternating hyperplane perturbation (Faiss / e-means Eq. 8 style)
                alt = np.where(np.arange(self.d) % 2 == 0, 1.0, -1.0)
                eps_alt = 1e-2
                mu_plus = mu_d * (1.0 + eps_alt * alt)
                mu_minus = mu_d * (1.0 - eps_alt * alt)
                self._assign_split_tuples(don_i, dead_i, n_half, mu_plus, mu_minus, cov_cur[don_i], cov_cur[don_i])
            else:
                if self.cov_type == "diag":
                    var_d = cov_cur[don_i]
                    if strategy == "spectral_v1":
                        m_star = int(np.argmax(var_d))
                        lam_max = float(var_d[m_star])
                        v1 = np.zeros(self.d, dtype=np.float64)
                        v1[m_star] = 1.0
                    else:  # random_orth
                        v1 = self.rng.normal(size=(self.d,))
                        v1 /= max(float(np.linalg.norm(v1)), 1e-12)
                        lam_max = float(np.sum((v1 ** 2) * var_d))

                    shift = delta * math.sqrt(max(lam_max, floor)) * v1
                    mu_plus = mu_d + shift
                    mu_minus = mu_d - shift
                    var_child = np.maximum(var_d - (delta ** 2) * lam_max * (v1 ** 2), floor)
                    self._assign_split_tuples(don_i, dead_i, n_half, mu_plus, mu_minus, var_child, var_child)

                elif self.cov_type == "low_rank_diag":
                    if U_cur is None:
                        raise ValueError("U_cur must be present for low_rank_diag splitting.")
                    d_var = cov_cur[don_i]
                    U_d = U_cur[don_i]
                    if strategy == "spectral_v1":
                        v = self.rng.normal(size=(self.d,))
                        v /= max(float(np.linalg.norm(v)), 1e-12)
                        for _ in range(self.cfg.power_iter_steps):
                            mv = d_var * v + U_d @ (U_d.T @ v)
                            nrm = float(np.linalg.norm(mv))
                            if nrm > 1e-15:
                                v = mv / nrm
                        v1 = v
                        lam_max = float(np.sum(d_var * (v1 ** 2)) + np.sum((U_d.T @ v1) ** 2))
                    else:
                        v1 = self.rng.normal(size=(self.d,))
                        v1 /= max(float(np.linalg.norm(v1)), 1e-12)
                        lam_max = float(np.sum(d_var * (v1 ** 2)) + np.sum((U_d.T @ v1) ** 2))

                    shift = delta * math.sqrt(max(lam_max, floor)) * v1
                    mu_plus = mu_d + shift
                    mu_minus = mu_d - shift
                    total_diag = np.maximum(d_var + np.sum(U_d ** 2, axis=1) - (delta ** 2) * lam_max * (v1 ** 2), floor)
                    self._assign_split_tuples(don_i, dead_i, n_half, mu_plus, mu_minus, total_diag, total_diag)
                    if self.U is not None:
                        self.U[dead_i] = U_d.copy()
                        if self.U_slow is not None:
                            self.U_slow[dead_i] = self.U_slow[don_i].copy()
                else:
                    cov_d = cov_cur[don_i]
                    if strategy == "spectral_v1":
                        lam_max, v1 = _power_iteration_top_eig(cov_d, steps=self.cfg.power_iter_steps, rng=self.rng)
                    else:
                        v1 = self.rng.normal(size=(self.d,))
                        v1 /= max(float(np.linalg.norm(v1)), 1e-12)
                        lam_max = float(v1 @ (cov_d @ v1))

                    shift = delta * math.sqrt(max(lam_max, floor)) * v1
                    mu_plus = mu_d + shift
                    mu_minus = mu_d - shift
                    cov_child = cov_d - (delta ** 2) * lam_max * np.outer(v1, v1)
                    # Guarantee strictly PSD child covariance
                    cov_child = 0.5 * (cov_child + cov_child.T)
                    evals, evecs = np.linalg.eigh(cov_child)
                    evals = np.maximum(evals, floor)
                    cov_child = (evecs * evals[None, :]) @ evecs.T
                    self._assign_split_tuples(don_i, dead_i, n_half, mu_plus, mu_minus, cov_child, cov_child)

            self.cooldown[don_i] = self.cfg.cooldown_steps
            self.cooldown[dead_i] = self.cfg.cooldown_steps
            self.splits_done += 1

        return n_pairs

    def _assign_split_tuples(
        self,
        don_i: int,
        dead_i: int,
        n_half: float,
        mu_plus: np.ndarray,
        mu_minus: np.ndarray,
        cov_plus: np.ndarray,
        cov_minus: np.ndarray,
    ) -> None:
        self.N[don_i] = n_half
        self.N[dead_i] = n_half
        self.S[don_i] = n_half * mu_plus
        self.S[dead_i] = n_half * mu_minus
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q[don_i] = n_half * (cov_plus + mu_plus ** 2)
            self.Q[dead_i] = n_half * (cov_minus + mu_minus ** 2)
        else:
            self.Q[don_i] = n_half * (cov_plus + np.outer(mu_plus, mu_plus))
            self.Q[dead_i] = n_half * (cov_minus + np.outer(mu_minus, mu_minus))

        # Mirror split onto passive Slow-EMA shadow buffers
        ns_half = 0.5 * self.N_slow[don_i]
        self.N_slow[don_i] = ns_half
        self.N_slow[dead_i] = ns_half
        self.S_slow[don_i] = ns_half * mu_plus
        self.S_slow[dead_i] = ns_half * mu_minus
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q_slow[don_i] = ns_half * (cov_plus + mu_plus ** 2)
            self.Q_slow[dead_i] = ns_half * (cov_minus + mu_minus ** 2)
        else:
            self.Q_slow[don_i] = ns_half * (cov_plus + np.outer(mu_plus, mu_plus))
            self.Q_slow[dead_i] = ns_half * (cov_minus + np.outer(mu_minus, mu_minus))

    def step_batch(
        self,
        X_batch: np.ndarray,
        alpha_t: float,
        record_last_epoch: bool = False,
        allow_split: bool = True,
    ) -> float:
        """Perform one E-step + 3-tuple EMA M-step on X_batch. Returns batch mean NLL."""
        X64 = np.asarray(X_batch, dtype=np.float64)
        if not self.initialized:
            self.init_from_data(X64)
        pi, mu, cov_or_var, U = self._readout_from_tuples(self.N, self.S, self.Q, self.U)
        log_joint = self._compute_log_joint(X64, pi, mu, cov_or_var, U)
        r, lse = _softmax_from_log_joint(log_joint, top_m=self.cfg.top_m)

        n_b = np.sum(r, axis=0)
        s_b = r.T @ X64
        if self.cov_type in ("diag", "low_rank_diag"):
            q_b = r.T @ (X64 ** 2)
        else:
            # r.T @ outer(x_i, x_i) -> einsum('bi,bd,be->ide', r, X, X)
            q_b = np.einsum("bi,bd,be->ide", r, X64, X64, optimize=True)

        gamma = 1.0 - alpha_t
        self.N = gamma * self.N + alpha_t * n_b
        self.S = gamma * self.S + alpha_t * s_b
        self.Q = gamma * self.Q + alpha_t * q_b
        if self.cov_type == "low_rank_diag":
            self._update_low_rank_factors(X64, r, mu, n_b, alpha_t)

        # Passive Slow-EMA update
        a_s = float(self.cfg.slow_ema_alpha)
        g_s = 1.0 - a_s
        self.N_slow = g_s * self.N_slow + a_s * n_b
        self.S_slow = g_s * self.S_slow + a_s * s_b
        self.Q_slow = g_s * self.Q_slow + a_s * q_b
        if self.cov_type == "low_rank_diag" and self.U is not None and self.U_slow is not None:
            self.U_slow = g_s * self.U_slow + a_s * self.U

        if record_last_epoch:
            self.N_last += n_b
            self.S_last += s_b
            self.Q_last += q_b

        self.step_count += 1
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        if allow_split and self.cfg.split_interval > 0 and (self.step_count % self.cfg.split_interval == 0):
            self._revive_collapsed_components(X64)

        return float(-np.mean(lse))

    def fit(
        self,
        X: np.ndarray,
        batch_size: int = 512,
        epochs: int = 3,
        alpha_0: Optional[float] = None,
    ) -> EGMMFitResult:
        """Train e-GMM over X for `epochs` passes."""
        X64 = np.asarray(X, dtype=np.float64)
        if not self.initialized:
            self.init_from_data(X64)

        a0 = float(alpha_0 if alpha_0 is not None else self.cfg.alpha_0)
        batches_per_epoch = max(1, math.ceil(X64.shape[0] / batch_size))
        total_steps = epochs * batches_per_epoch
        batch_nlls: List[float] = []

        t = 0
        for ep in range(epochs):
            is_last = ep == (epochs - 1)
            if is_last:
                self.reset_last_epoch_accumulators()
            for xb in iter_minibatches(X64, batch_size, rng=self.rng, shuffle=True):
                alpha_t = cosine_alpha(t, total_steps, a0, self.cfg.eps_floor, self.cfg.use_cosine_schedule)
                if epochs > 1:
                    allow_split = not is_last
                else:
                    allow_split = t < (total_steps - max(2, batches_per_epoch // 4))
                nll_b = self.step_batch(xb, alpha_t, record_last_epoch=is_last, allow_split=allow_split)
                batch_nlls.append(nll_b)
                t += 1

        pi, mu, cov_or_var, U = self.readout()
        health = compute_gmm_health(
            self.N,
            cov_or_var,
            cov_type=self.cov_type,
            cov_floor=self.cfg.cov_floor,
            dead_tau=self.cfg.split_tau,
            U=U,
        )
        return {
            "pi": pi,
            "mu": mu,
            "cov": cov_or_var,
            "U": U,
            "splits": self.splits_done,
            "chol_fallbacks": self.chol_fallbacks,
            "batch_nlls": batch_nlls,
            "health": health,
        }


class NumPyOnlineEMCappeMoulines:
    """
    Cappé & Moulines (2009) Online / Stochastic EM baseline using Robbins-Monro
    step sizes gamma_t = (t + t0)^(-beta), beta in (0.5, 1], with no revival.
    """

    def __init__(
        self,
        num_components: int,
        dim: int,
        cov_type: str = "diag",
        beta: float = 0.75,
        t0: float = 10.0,
        cov_floor: float = 1e-8,
        init_sigma0: float = 1.0,
        seed: int = 42,
    ):
        self.k = num_components
        self.d = dim
        self.cov_type = cov_type
        self.beta = beta
        self.t0 = t0
        self.cov_floor = cov_floor
        self.init_sigma0 = init_sigma0
        self.rng = np.random.default_rng(seed)
        self.singular_events = 0

    def fit(self, X: np.ndarray, batch_size: int = 512, epochs: int = 3) -> OnlineEMFitResult:
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        idx = self.rng.choice(n, size=self.k, replace=n < self.k)
        mu0 = X64[idx].copy()

        # Normalized sufficient statistics (expectation form: sum(N_j) ~= 1)
        N = np.full(self.k, 1.0 / self.k, dtype=np.float64)
        S = N[:, None] * mu0
        s0_sq = self.init_sigma0 ** 2
        if self.cov_type == "diag":
            Q = N[:, None] * (mu0 ** 2 + s0_sq)
        else:
            Q = N[:, None, None] * (mu0[:, :, None] * mu0[:, None, :] + s0_sq * np.eye(d))

        self.singular_events = 0
        t = 0
        for _ in range(epochs):
            for xb in iter_minibatches(X64, batch_size, rng=self.rng, shuffle=True):
                B = xb.shape[0]
                gamma_t = robbins_monro_gamma(t, beta=self.beta, t0=self.t0)
                N_safe = np.maximum(N, 1e-30)
                pi = N_safe / N_safe.sum()
                mu = S / N_safe[:, None]
                if self.cov_type == "diag":
                    raw_var = Q / N_safe[:, None] - mu ** 2
                    if np.any(raw_var <= 1e-10):
                        self.singular_events += int(np.sum(np.min(raw_var, axis=1) <= 1e-10))
                    cov = np.maximum(raw_var, self.cov_floor)
                    log_joint, _ = gaussian_log_prob_diag(xb, pi, mu, cov)
                else:
                    raw_cov = Q / N_safe[:, None, None] - mu[:, :, None] * mu[:, None, :]
                    raw_cov = 0.5 * (raw_cov + np.swapaxes(raw_cov, -1, -2))
                    evals = np.linalg.eigvalsh(raw_cov)
                    if np.any(evals <= 1e-10):
                        self.singular_events += int(np.sum(np.min(evals, axis=1) <= 1e-10))
                    cov = raw_cov + self.cov_floor * np.eye(d)[None, :, :]
                    log_joint, _, fb = gaussian_log_prob_full(xb, pi, mu, cov)
                    self.singular_events += fb

                r, _ = _softmax_from_log_joint(log_joint)
                n_b = np.mean(r, axis=0)
                s_b = (r.T @ xb) / B
                N = (1.0 - gamma_t) * N + gamma_t * n_b
                S = (1.0 - gamma_t) * S + gamma_t * s_b
                if self.cov_type == "diag":
                    q_b = (r.T @ (xb ** 2)) / B
                else:
                    q_b = np.einsum("bi,bd,be->ide", r, xb, xb, optimize=True) / B
                Q = (1.0 - gamma_t) * Q + gamma_t * q_b
                t += 1

        N_safe = np.maximum(N, 1e-30)
        pi = N_safe / N_safe.sum()
        mu = S / N_safe[:, None]
        if self.cov_type == "diag":
            cov = np.maximum(Q / N_safe[:, None] - mu ** 2, self.cov_floor)
        else:
            cov = 0.5 * (Q / N_safe[:, None, None] - mu[:, :, None] * mu[:, None, :] + np.swapaxes(Q / N_safe[:, None, None] - mu[:, :, None] * mu[:, None, :], -1, -2)) + self.cov_floor * np.eye(d)[None, :, :]

        health = compute_gmm_health(N, cov, cov_type=self.cov_type, cov_floor=self.cov_floor)
        return {"pi": pi, "mu": mu, "cov": cov, "singular_events": self.singular_events, "health": health}


class NumPyFullBatchEM:
    """Classical Full-Batch EM baseline (used as oracle reference on tractable sizes)."""

    def __init__(
        self,
        num_components: int,
        dim: int,
        cov_type: str = "diag",
        cov_floor: float = 1e-4,
        init_sigma0: float = 1.0,
        seed: int = 42,
    ):
        self.k = num_components
        self.d = dim
        self.cov_type = cov_type
        self.cov_floor = cov_floor
        self.init_sigma0 = init_sigma0
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, max_iters: int = 25, tol: float = 1e-4) -> FullBatchEMFitResult:
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        idx = self.rng.choice(n, size=self.k, replace=n < self.k)
        mu = X64[idx].copy()
        pi = np.full(self.k, 1.0 / self.k, dtype=np.float64)
        s0_sq = self.init_sigma0 ** 2 + self.cov_floor
        if self.cov_type == "diag":
            cov = np.full((self.k, d), s0_sq, dtype=np.float64)
        else:
            cov = np.broadcast_to(s0_sq * np.eye(d), (self.k, d, d)).copy()

        prev_nll = float("inf")
        nll_history: List[float] = []
        N_eff = np.full(self.k, n / self.k, dtype=np.float64)

        for _ in range(max_iters):
            if self.cov_type == "diag":
                log_joint, lse = gaussian_log_prob_diag(X64, pi, mu, cov)
            else:
                log_joint, lse, _ = gaussian_log_prob_full(X64, pi, mu, cov)
            r, _ = _softmax_from_log_joint(log_joint)
            nll = float(-np.mean(lse))
            nll_history.append(nll)

            N_eff = np.maximum(np.sum(r, axis=0), 1e-12)
            pi = N_eff / np.sum(N_eff)
            mu = (r.T @ X64) / N_eff[:, None]
            if self.cov_type == "diag":
                var = (r.T @ (X64 ** 2)) / N_eff[:, None] - mu ** 2
                cov = np.maximum(var, 0.0) + self.cov_floor
            else:
                q_sum = np.einsum("bi,bd,be->ide", r, X64, X64, optimize=True)
                raw = q_sum / N_eff[:, None, None] - mu[:, :, None] * mu[:, None, :]
                cov = 0.5 * (raw + np.swapaxes(raw, -1, -2)) + self.cov_floor * np.eye(d)[None, :, :]

            if abs(prev_nll - nll) < tol:
                break
            prev_nll = nll

        health = compute_gmm_health(N_eff, cov, cov_type=self.cov_type, cov_floor=self.cov_floor)
        return {"pi": pi, "mu": mu, "cov": cov, "nll_history": nll_history, "health": health}


class NumPyEMeansPostHocGMM:
    """
    Strong baseline: runs hard e-means (2-tuple EMA + cosine schedule + donor splitting)
    followed by 1 exact hard-assignment M-step covariance readout.
    Proves that joint soft EM in e-GMM outperforms hard clustering + post-hoc covariance!
    """

    def __init__(
        self,
        num_components: int,
        dim: int,
        cov_type: str = "diag",
        cov_floor: float = 1e-4,
        alpha_0: float = 0.25,
        seed: int = 42,
    ):
        self.k = num_components
        self.d = dim
        self.cov_type = cov_type
        self.cov_floor = cov_floor
        self.alpha_0 = alpha_0
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, batch_size: int = 512, epochs: int = 3) -> EMeansPostHocFitResult:
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        idx = self.rng.choice(n, size=self.k, replace=n < self.k)
        c = X64[idx].copy()
        N = np.full(self.k, 1e-10, dtype=np.float64)
        S = N[:, None] * c

        batches_per_ep = max(1, math.ceil(n / batch_size))
        total_steps = epochs * batches_per_ep
        t = 0
        splits = 0

        for _ in range(epochs):
            for xb in iter_minibatches(X64, batch_size, rng=self.rng, shuffle=True):
                alpha_t = cosine_alpha(t, total_steps, self.alpha_0, 0.1, True)
                gamma = 1.0 - alpha_t
                c = S / N[:, None]
                c_norm = np.sum(c ** 2, axis=1)[None, :]
                assign = np.argmin(c_norm - 2.0 * (xb @ c.T), axis=1)

                n_b = np.bincount(assign, minlength=self.k).astype(np.float64)
                s_b = np.zeros((self.k, d), dtype=np.float64)
                np.add.at(s_b, assign, xb)

                N = gamma * N + alpha_t * n_b
                S = gamma * S + alpha_t * s_b
                t += 1

                if t % 5 == 0 and t < total_steps - 2:
                    mean_n = float(np.mean(N))
                    dead = np.where(N < 0.1 * mean_n)[0]
                    alive = np.where(N > mean_n)[0]
                    if dead.size > 0 and alive.size > 0:
                        npick = min(dead.size, alive.size, 4)
                        p = N[alive] / N[alive].sum()
                        donors = self.rng.choice(alive, size=npick, replace=False, p=p)
                        alt = np.where(np.arange(d) % 2 == 0, 1.0, -1.0) * 7.8125e-3
                        for di, ai in zip(dead[:npick], donors):
                            c_don = S[ai] / N[ai]
                            nh = 0.5 * N[ai]
                            N[ai] = nh
                            N[di] = nh
                            S[ai] = nh * c_don * (1.0 + alt)
                            S[di] = nh * c_don * (1.0 - alt)
                            splits += 1

        # Post-hoc 1-pass exact covariance M-step on hard Voronoi cells
        c = S / N[:, None]
        c_norm = np.sum(c ** 2, axis=1)[None, :]
        assign = np.argmin(c_norm - 2.0 * (X64 @ c.T), axis=1)
        counts = np.bincount(assign, minlength=self.k).astype(np.float64)
        pi = np.maximum(counts, 1e-6)
        pi /= pi.sum()
        mu = np.zeros((self.k, d), dtype=np.float64)
        if self.cov_type == "diag":
            cov = np.full((self.k, d), self.cov_floor, dtype=np.float64)
        else:
            cov = np.broadcast_to(self.cov_floor * np.eye(d), (self.k, d, d)).copy()

        for j in range(self.k):
            pts = X64[assign == j]
            if pts.shape[0] > 0:
                mu[j] = pts.mean(axis=0)
                centered = pts - mu[j][None, :]
                if self.cov_type == "diag":
                    cov[j] = np.mean(centered ** 2, axis=0) + self.cov_floor
                else:
                    cov[j] = (centered.T @ centered) / pts.shape[0] + self.cov_floor * np.eye(d)
            else:
                mu[j] = c[j]

        health = compute_gmm_health(counts, cov, cov_type=self.cov_type, cov_floor=self.cov_floor)
        return {"pi": pi, "mu": mu, "cov": cov, "splits": splits, "health": health}
