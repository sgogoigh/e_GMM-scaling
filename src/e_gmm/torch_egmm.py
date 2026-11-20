"""PyTorch GPU/CPU e-GMM implementation with Flash-GMM 2d-GEMM, Woodbury low-rank, and Spectral Splitting."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .config import EGMMConfig
from .triton_kernels import flash_gmm_diag_log_probs


class TorchEGMM(nn.Module):
    """
    PyTorch 3-tuple EMA Gaussian Mixture Model supporting:
      - `diag`: Flash-GMM 2d-augmented GEMM + tiled streaming LogSumExp + Top-m M-step
      - `low_rank_diag`: Woodbury + Matrix Determinant Lemma in O(k d r) memory
      - `full`: Cholesky solve with adaptive jitter + 5-step power iteration spectral split
    """

    def __init__(self, cfg: EGMMConfig):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.num_components
        self.d = cfg.dim
        self.cov_type = cfg.cov_type
        self.rank = min(cfg.low_rank, cfg.dim)

        self.register_buffer("N", torch.zeros(self.k, dtype=torch.float32))
        self.register_buffer("S", torch.zeros(self.k, self.d, dtype=torch.float32))
        if self.cov_type in ("diag", "low_rank_diag"):
            self.register_buffer("Q", torch.zeros(self.k, self.d, dtype=torch.float32))
        else:
            self.register_buffer("Q", torch.zeros(self.k, self.d, self.d, dtype=torch.float32))

        if self.cov_type == "low_rank_diag":
            self.register_buffer("U", torch.zeros(self.k, self.d, self.rank, dtype=torch.float32))
        else:
            self.U = None

        # Passive Slow-EMA shadow buffers
        self.register_buffer("N_slow", torch.zeros_like(self.N))
        self.register_buffer("S_slow", torch.zeros_like(self.S))
        self.register_buffer("Q_slow", torch.zeros_like(self.Q))

        # Last-epoch exact accumulators
        self.register_buffer("N_last", torch.zeros_like(self.N))
        self.register_buffer("S_last", torch.zeros_like(self.S))
        self.register_buffer("Q_last", torch.zeros_like(self.Q))

        self.register_buffer("cooldown", torch.zeros(self.k, dtype=torch.long))
        self.initialized = False
        self.step_count = 0
        self.splits_done = 0
        self.chol_fallbacks = 0

    @torch.no_grad()
    def init_from_data(self, X: torch.Tensor, sigma0: Optional[float] = None) -> None:
        device = X.device
        n = X.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        if n >= self.k:
            idx = torch.randperm(n, generator=g)[: self.k].to(device)
        else:
            idx = torch.randint(0, n, (self.k,), generator=g).to(device)
        mu0 = X[idx].float()
        s0_sq = float(sigma0 if sigma0 is not None else self.cfg.init_sigma0) ** 2

        self.N.fill_(float(self.cfg.init_prior_mass))
        self.S.copy_(self.N.unsqueeze(1) * mu0)
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q.copy_(self.N.unsqueeze(1) * (mu0.pow(2) + s0_sq))
        else:
            eye = torch.eye(self.d, device=device, dtype=torch.float32).unsqueeze(0)
            self.Q.copy_(self.N.view(-1, 1, 1) * (mu0.unsqueeze(2) * mu0.unsqueeze(1) + s0_sq * eye))

        if self.U is not None:
            u_init = torch.randn(self.k, self.d, self.rank, generator=g, dtype=torch.float32).to(device) * 1e-3
            self.U.copy_(u_init)

        self.N_slow.copy_(self.N)
        self.S_slow.copy_(self.S)
        self.Q_slow.copy_(self.Q)
        self.reset_last_epoch()
        self.cooldown.zero_()
        self.initialized = True
        self.step_count = 0
        self.splits_done = 0
        self.chol_fallbacks = 0

    @torch.no_grad()
    def reset_last_epoch(self) -> None:
        self.N_last.zero_()
        self.S_last.zero_()
        self.Q_last.zero_()

    def _readout_tuples(self, N: torch.Tensor, S: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_safe = N.clamp_min(1e-30)
        pi = n_safe / n_safe.sum()
        mu = S / n_safe.unsqueeze(1)
        floor = float(self.cfg.cov_floor)

        if self.cov_type == "diag":
            var = (Q / n_safe.unsqueeze(1) - mu.pow(2)).clamp_min(0.0) + floor
            return pi, mu, var

        if self.cov_type == "low_rank_diag":
            total_diag = (Q / n_safe.unsqueeze(1) - mu.pow(2)).clamp_min(0.0) + floor
            u_diag = torch.sum(self.U.pow(2), dim=2)
            d_var = (total_diag - u_diag).clamp_min(floor)
            return pi, mu, d_var

        eye = torch.eye(self.d, device=mu.device, dtype=torch.float32).unsqueeze(0)
        cov = Q / n_safe.view(-1, 1, 1) - mu.unsqueeze(2) * mu.unsqueeze(1)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + floor * eye
        return pi, mu, cov

    def readout(self, mode: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_mode = mode or self.cfg.finalization
        if use_mode == "slow_ema":
            return self._readout_tuples(self.N_slow, self.S_slow, self.Q_slow)
        if use_mode == "last_epoch" and float(self.N_last.sum()) > 1e-12:
            empty = self.N_last <= 1e-12
            n_eff = self.N_last.clamp_min(1e-12)
            s_eff = torch.where(empty.unsqueeze(1), (self.S / self.N.clamp_min(1e-30).unsqueeze(1)) * n_eff.unsqueeze(1), self.S_last)
            if self.cov_type in ("diag", "low_rank_diag"):
                q_fb = (self.Q / self.N.clamp_min(1e-30).unsqueeze(1)) * n_eff.unsqueeze(1)
                q_eff = torch.where(empty.unsqueeze(1), q_fb, self.Q_last)
            else:
                q_fb = (self.Q / self.N.clamp_min(1e-30).view(-1, 1, 1)) * n_eff.view(-1, 1, 1)
                q_eff = torch.where(empty.view(-1, 1, 1), q_fb, self.Q_last)
            return self._readout_tuples(n_eff, s_eff, q_eff)
        return self._readout_tuples(self.N, self.S, self.Q)

    def _woodbury_log_joint(self, x_f32: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, d_var: torch.Tensor) -> torch.Tensor:
        B, d = x_f32.shape
        const_term = d * math.log(2.0 * math.pi)
        eye_r = torch.eye(self.rank, device=x_f32.device, dtype=torch.float32).unsqueeze(0)
        inv_d = 1.0 / d_var.clamp_min(1e-12)                                 # (k, d)
        dinv_u = inv_d.unsqueeze(-1) * self.U                                 # (k, d, r)
        cap = eye_r + torch.bmm(self.U.transpose(1, 2), dinv_u)               # (k, r, r)
        cap = 0.5 * (cap + cap.transpose(-1, -2))
        L_c = torch.linalg.cholesky(cap)                                      # (k, r, r)
        logdet = torch.sum(torch.log(d_var.clamp_min(1e-12)), dim=1) + 2.0 * torch.sum(
            torch.log(torch.diagonal(L_c, dim1=-2, dim2=-1)), dim=1
        )                                                                     # (k,)

        log_joint = torch.empty((B, self.k), device=x_f32.device, dtype=torch.float32)
        for j in range(self.k):
            diff = x_f32 - mu[j].unsqueeze(0)                                 # (B, d)
            diag_q = torch.sum(diff.pow(2) * inv_d[j].unsqueeze(0), dim=1)
            proj = dinv_u[j].t() @ diff.t()                                   # (r, B)
            solved = torch.linalg.solve_triangular(L_c[j], proj, upper=False)
            maha = diag_q - torch.sum(solved.pow(2), dim=0)
            log_joint[:, j] = torch.log(pi[j].clamp_min(1e-30)) - 0.5 * (const_term + logdet[j] + maha)
        return log_joint

    def _full_cov_log_joint(self, x_f32: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        B, d = x_f32.shape
        const_term = d * math.log(2.0 * math.pi)
        eye_d = torch.eye(d, device=x_f32.device, dtype=torch.float32).unsqueeze(0)
        L, info = torch.linalg.cholesky_ex(cov)
        if torch.any(info != 0):
            self.chol_fallbacks += int((info != 0).sum().item())
            traces = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / d
            cov_reg = cov + (1e-4 * traces.clamp_min(1.0)) * eye_d
            L, info2 = torch.linalg.cholesky_ex(cov_reg)
            if torch.any(info2 != 0):
                evals, evecs = torch.linalg.eigh(cov_reg)
                evals = evals.clamp_min(1e-5)
                cov_psd = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
                L = torch.linalg.cholesky(cov_psd)

        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)), dim=1)
        log_joint = torch.empty((B, self.k), device=x_f32.device, dtype=torch.float32)
        for j in range(self.k):
            diff = x_f32 - mu[j].unsqueeze(0)
            solved = torch.linalg.solve_triangular(L[j], diff.t(), upper=False)
            maha = torch.sum(solved.pow(2), dim=0)
            log_joint[:, j] = torch.log(pi[j].clamp_min(1e-30)) - 0.5 * (const_term + logdet[j] + maha)
        return log_joint

    @torch.no_grad()
    def _revive_collapsed(self, x_f32: torch.Tensor) -> int:
        if self.cfg.split_strategy == "none":
            return 0
        mean_n = float(self.N.mean().item())
        dead_ids = torch.where((self.N < self.cfg.split_tau * mean_n) & (self.cooldown == 0))[0]
        if dead_ids.numel() == 0:
            return 0

        donor_pool = torch.where((self.N > mean_n) & (self.cooldown == 0))[0]
        if donor_pool.numel() == 0:
            donor_pool = torch.where(self.N > mean_n)[0]
        if donor_pool.numel() == 0:
            return 0

        n_pairs = min(int(dead_ids.numel()), int(donor_pool.numel()), self.cfg.max_splits_per_step)
        probs = self.N[donor_pool].clamp_min(1e-12)
        probs = probs / probs.sum()
        donors = donor_pool[torch.multinomial(probs, num_samples=n_pairs, replacement=False)]

        _, mu_cur, cov_cur = self._readout_tuples(self.N, self.S, self.Q)
        delta = float(self.cfg.split_delta)
        floor = float(self.cfg.cov_floor)

        for idx in range(n_pairs):
            dead_i = int(dead_ids[idx].item())
            don_i = int(donors[idx].item())
            mu_d = mu_cur[don_i]
            n_half = 0.5 * self.N[don_i]

            if self.cov_type == "diag":
                var_d = cov_cur[don_i]
                m_star = int(torch.argmax(var_d).item())
                lam_max = float(var_d[m_star].item())
                v1 = torch.zeros_like(mu_d)
                v1[m_star] = 1.0
                shift = delta * math.sqrt(max(lam_max, floor)) * v1
                mu_plus, mu_minus = mu_d + shift, mu_d - shift
                var_c = (var_d - (delta ** 2) * lam_max * v1.pow(2)).clamp_min(floor)
                self._write_split_buffers(don_i, dead_i, n_half, mu_plus, mu_minus, var_c)
            elif self.cov_type == "low_rank_diag":
                d_var = cov_cur[don_i]
                U_d = self.U[don_i]
                v = torch.randn_like(mu_d)
                v = v / v.norm().clamp_min(1e-12)
                for _ in range(self.cfg.power_iter_steps):
                    mv = d_var * v + U_d @ (U_d.t() @ v)
                    nrm = mv.norm()
                    if float(nrm.item()) > 1e-15:
                        v = mv / nrm
                lam_max = float((torch.sum(d_var * v.pow(2)) + torch.sum((U_d.t() @ v).pow(2))).item())
                shift = delta * math.sqrt(max(lam_max, floor)) * v
                mu_plus, mu_minus = mu_d + shift, mu_d - shift
                total_diag = (d_var + torch.sum(U_d.pow(2), dim=1) - (delta ** 2) * lam_max * v.pow(2)).clamp_min(floor)
                self._write_split_buffers(don_i, dead_i, n_half, mu_plus, mu_minus, total_diag)
                self.U[dead_i].copy_(U_d)
            else:
                cov_d = cov_cur[don_i]
                v = torch.randn_like(mu_d)
                v = v / v.norm().clamp_min(1e-12)
                for _ in range(self.cfg.power_iter_steps):
                    mv = cov_d @ v
                    nrm = mv.norm()
                    if float(nrm.item()) > 1e-15:
                        v = mv / nrm
                lam_max = max(float((v @ (cov_d @ v)).item()), 0.0)
                shift = delta * math.sqrt(max(lam_max, floor)) * v
                mu_plus, mu_minus = mu_d + shift, mu_d - shift
                cov_c = cov_d - (delta ** 2) * lam_max * torch.outer(v, v)
                cov_c = 0.5 * (cov_c + cov_c.t())
                evals, evecs = torch.linalg.eigh(cov_c)
                cov_c = (evecs * evals.clamp_min(floor).unsqueeze(0)) @ evecs.t()
                self._write_split_buffers(don_i, dead_i, n_half, mu_plus, mu_minus, cov_c)

            self.cooldown[don_i] = self.cfg.cooldown_steps
            self.cooldown[dead_i] = self.cfg.cooldown_steps
            self.splits_done += 1

        return n_pairs

    def _write_split_buffers(
        self,
        don_i: int,
        dead_i: int,
        n_half: torch.Tensor,
        mu_plus: torch.Tensor,
        mu_minus: torch.Tensor,
        cov_child: torch.Tensor,
    ) -> None:
        self.N[don_i] = n_half
        self.N[dead_i] = n_half
        self.S[don_i] = n_half * mu_plus
        self.S[dead_i] = n_half * mu_minus
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q[don_i] = n_half * (cov_child + mu_plus.pow(2))
            self.Q[dead_i] = n_half * (cov_child + mu_minus.pow(2))
        else:
            self.Q[don_i] = n_half * (cov_child + torch.outer(mu_plus, mu_plus))
            self.Q[dead_i] = n_half * (cov_child + torch.outer(mu_minus, mu_minus))

        ns_half = 0.5 * self.N_slow[don_i]
        self.N_slow[don_i] = ns_half
        self.N_slow[dead_i] = ns_half
        self.S_slow[don_i] = ns_half * mu_plus
        self.S_slow[dead_i] = ns_half * mu_minus
        if self.cov_type in ("diag", "low_rank_diag"):
            self.Q_slow[don_i] = ns_half * (cov_child + mu_plus.pow(2))
            self.Q_slow[dead_i] = ns_half * (cov_child + mu_minus.pow(2))
        else:
            self.Q_slow[don_i] = ns_half * (cov_child + torch.outer(mu_plus, mu_plus))
            self.Q_slow[dead_i] = ns_half * (cov_child + torch.outer(mu_minus, mu_minus))

    @torch.no_grad()
    def step_batch(
        self,
        x_batch: torch.Tensor,
        alpha_t: float,
        record_last_epoch: bool = False,
        allow_split: bool = True,
        use_bf16_gemm: bool = False,
    ) -> float:
        """Execute one E-step + FP32 3-tuple EMA M-step on x_batch. Returns batch NLL."""
        if not self.initialized:
            self.init_from_data(x_batch)

        x_f32 = x_batch.float()
        x_sq = x_f32.pow(2)
        pi, mu, cov_or_var = self._readout_tuples(self.N, self.S, self.Q)
        gamma = 1.0 - float(alpha_t)

        if self.cov_type == "diag" and self.cfg.e_step_mode == "flash_gemm":
            r_or_sparse, top_idx, _, lse = flash_gmm_diag_log_probs(
                x_f32,
                pi,
                mu,
                cov_or_var,
                chunk_k=self.cfg.chunk_k,
                top_m=self.cfg.top_m,
                use_bf16_gemm=use_bf16_gemm,
            )
            if top_idx is not None:
                # Sparse O(B * m * d) M-step scatter accumulation
                n_b = torch.zeros_like(self.N)
                s_b = torch.zeros_like(self.S)
                q_b = torch.zeros_like(self.Q)
                flat_idx = top_idx.reshape(-1)
                flat_w = r_or_sparse.reshape(-1)
                m_top = top_idx.shape[1]
                x_rep = x_f32.unsqueeze(1).expand(-1, m_top, -1).reshape(-1, self.d)
                xsq_rep = x_sq.unsqueeze(1).expand(-1, m_top, -1).reshape(-1, self.d)
                n_b.index_add_(0, flat_idx, flat_w)
                s_b.index_add_(0, flat_idx, flat_w.unsqueeze(1) * x_rep)
                q_b.index_add_(0, flat_idx, flat_w.unsqueeze(1) * xsq_rep)
            else:
                n_b = r_or_sparse.sum(dim=0)
                s_b = r_or_sparse.t() @ x_f32
                q_b = r_or_sparse.t() @ x_sq
        else:
            if self.cov_type == "diag":
                diff = x_f32.unsqueeze(1) - mu.unsqueeze(0)
                maha = torch.sum(diff.pow(2) / cov_or_var.unsqueeze(0), dim=-1)
                logdet = torch.sum(torch.log(2.0 * math.pi * cov_or_var), dim=-1)
                log_joint = torch.log(pi.clamp_min(1e-30)).unsqueeze(0) - 0.5 * (maha + logdet.unsqueeze(0))
            elif self.cov_type == "low_rank_diag":
                log_joint = self._woodbury_log_joint(x_f32, pi, mu, cov_or_var)
            else:
                log_joint = self._full_cov_log_joint(x_f32, pi, mu, cov_or_var)

            lse = torch.logsumexp(log_joint, dim=1)
            r = torch.softmax(log_joint, dim=1)
            n_b = r.sum(dim=0)
            s_b = r.t() @ x_f32
            if self.cov_type in ("diag", "low_rank_diag"):
                q_b = r.t() @ x_sq
            else:
                q_b = torch.einsum("bk,bd,be->kde", r, x_f32, x_f32)

            if self.cov_type == "low_rank_diag":
                floor = float(self.cfg.cov_floor)
                rem_dim = max(self.d - self.rank, 1)
                for j in range(self.k):
                    nb_j = float(n_b[j].item())
                    if nb_j <= 1e-6:
                        continue
                    w = r[:, j]
                    res = x_f32 - mu[j].unsqueeze(0)
                    v_cur, _ = torch.linalg.qr(self.U[j])
                    proj = res.t() @ (w.unsqueeze(1) * (res @ v_cur)) / nb_j
                    cand = (1.0 - alpha_t) * v_cur + alpha_t * (proj / proj.norm().clamp_min(1e-8))
                    q_mat, _ = torch.linalg.qr(cand)
                    signs = torch.where((q_mat * v_cur).sum(dim=0) >= 0.0, 1.0, -1.0)
                    q_mat = q_mat * signs.unsqueeze(0)
                    dir_vars = torch.sum(w.unsqueeze(1) * (res @ q_mat).pow(2), dim=0) / nb_j
                    tot_var = float(torch.sum(w.unsqueeze(1) * res.pow(2)).item()) / nb_j
                    bg_var = max((tot_var - float(dir_vars.sum().item())) / rem_dim, floor)
                    excess = (dir_vars - bg_var).clamp_min(0.0)
                    old_vars = self.U[j].pow(2).sum(dim=0)
                    smoothed_vars = (1.0 - alpha_t) * old_vars + alpha_t * excess
                    self.U[j].copy_(q_mat * torch.sqrt(smoothed_vars.clamp_min(0.0)).unsqueeze(0))

        self.N.mul_(gamma).add_(n_b, alpha=alpha_t)
        self.S.mul_(gamma).add_(s_b, alpha=alpha_t)
        self.Q.mul_(gamma).add_(q_b, alpha=alpha_t)

        a_s = float(self.cfg.slow_ema_alpha)
        g_s = 1.0 - a_s
        self.N_slow.mul_(g_s).add_(n_b, alpha=a_s)
        self.S_slow.mul_(g_s).add_(s_b, alpha=a_s)
        self.Q_slow.mul_(g_s).add_(q_b, alpha=a_s)

        if record_last_epoch:
            self.N_last.add_(n_b)
            self.S_last.add_(s_b)
            self.Q_last.add_(q_b)

        self.step_count += 1
        self.cooldown.sub_(1).clamp_min_(0)
        if allow_split and self.cfg.split_interval > 0 and (self.step_count % self.cfg.split_interval == 0):
            self._revive_collapsed(x_f32)

        return float(-lse.mean().item())
