"""Triton Flash-GMM diagonal log-density kernel with automatic PyTorch tiled fallback."""

import math
from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_gmm_diag_lse_kernel(
        X_aug_ptr,
        W_aug_ptr,
        B_ptr,
        LSE_ptr,
        N_rows,
        K_cols,
        D2_dim,
        stride_xm,
        stride_xd,
        stride_wk,
        stride_wd,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused streaming LogSumExp over k for diagonal GMMs using 2d-augmented dot products:
            logits(i, j) = sum_d X_aug[i, d] * W_aug[j, d] + b[j]
        Never materializes the (B, k) logit matrix in HBM.
        """
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_rows

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, K_cols, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < K_cols
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for start_d in range(0, D2_dim, BLOCK_K):
                offs_d = start_d + tl.arange(0, BLOCK_K)
                mask_d = offs_d < D2_dim
                x_ptrs = X_aug_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd
                w_ptrs = W_aug_ptr + offs_n[None, :] * stride_wk + offs_d[:, None] * stride_wd
                x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
                w_tile = tl.load(w_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(x_tile, w_tile)

            b_tile = tl.load(B_ptr + offs_n, mask=mask_n, other=float("-inf"))
            logits = acc + b_tile[None, :]
            logits = tl.where(mask_n[None, :], logits, float("-inf"))

            block_max = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, block_max)
            exp_logits = tl.exp(logits - new_m[:, None])
            l_i = l_i * tl.exp(m_i - new_m) + tl.sum(exp_logits, axis=1)
            m_i = new_m

        lse = m_i + tl.log(l_i)
        tl.store(LSE_ptr + offs_m, lse, mask=mask_m)


def flash_gmm_diag_log_probs(
    x_batch: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    chunk_k: int = 2048,
    top_m: Optional[int] = None,
    use_bf16_gemm: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Compute diagonal Gaussian log-joint probabilities / Top-m sparse responsibilities
    using the 2d-augmented Flash-GMM GEMM reformulation + tiled streaming LogSumExp.

    Returns:
        - If top_m is None: (dense_responsibilities (B, k), None, None, lse (B,))
        - If top_m is set: (sparse_weights (B, m), top_indices (B, m), None, lse (B,))
    """
    x_f32 = x_batch.float()
    var_safe = var.float().clamp_min(1e-12)
    inv_var = 1.0 / var_safe
    mu_f32 = mu.float()
    pi_safe = pi.float().clamp_min(1e-30)

    c_j = torch.sum((mu_f32 ** 2) * inv_var + torch.log(2.0 * math.pi * var_safe), dim=1)
    b_j = torch.log(pi_safe) - 0.5 * c_j  # (k,)

    x_aug = torch.cat([x_f32 * x_f32, x_f32], dim=1)          # (B, 2d)
    w_aug = torch.cat([-0.5 * inv_var, mu_f32 * inv_var], dim=1)  # (k, 2d)
    if use_bf16_gemm and x_aug.is_cuda:
        x_aug_mm = x_aug.bfloat16()
        w_aug_mm = w_aug.bfloat16()
    else:
        x_aug_mm = x_aug
        w_aug_mm = w_aug

    B, k = x_f32.shape[0], mu_f32.shape[0]
    device = x_f32.device

    # Streaming tiled LSE + optional top-m selection (never materializes full B x k in HBM when chunked)
    if chunk_k < k:
        m_run = torch.full((B,), float("-inf"), device=device, dtype=torch.float32)
        l_run = torch.zeros((B,), device=device, dtype=torch.float32)

        top_vals_acc = None
        top_idx_acc = None

        for start in range(0, k, chunk_k):
            end = min(start + chunk_k, k)
            logits_c = (x_aug_mm @ w_aug_mm[start:end].t()).float() + b_j[start:end].unsqueeze(0)
            b_max = torch.max(logits_c, dim=1).values
            new_m = torch.maximum(m_run, b_max)
            l_run = l_run * torch.exp(m_run - new_m) + torch.sum(torch.exp(logits_c - new_m.unsqueeze(1)), dim=1)
            m_run = new_m

            if top_m is not None and 0 < top_m < k:
                m_cur = min(top_m, end - start)
                tv, ti = torch.topk(logits_c, k=m_cur, dim=1)
                ti = ti + start
                if top_vals_acc is None:
                    top_vals_acc, top_idx_acc = tv, ti
                else:
                    cat_v = torch.cat([top_vals_acc, tv], dim=1)
                    cat_i = torch.cat([top_idx_acc, ti], dim=1)
                    keep = min(top_m, cat_v.shape[1])
                    top_vals_acc, sel = torch.topk(cat_v, k=keep, dim=1)
                    top_idx_acc = torch.gather(cat_i, 1, sel)

        lse = m_run + torch.log(l_run)
        if top_m is not None and 0 < top_m < k:
            r_sparse = torch.softmax(top_vals_acc, dim=1)
            return r_sparse, top_idx_acc, None, lse

        # Pass 2 for dense responsibilities when chunked without top_m
        r_dense = torch.empty((B, k), device=device, dtype=torch.float32)
        for start in range(0, k, chunk_k):
            end = min(start + chunk_k, k)
            logits_c = (x_aug_mm @ w_aug_mm[start:end].t()).float() + b_j[start:end].unsqueeze(0)
            r_dense[:, start:end] = torch.exp(logits_c - lse.unsqueeze(1))
        return r_dense, None, None, lse

    # Single-block fast path
    logits = (x_aug_mm @ w_aug_mm.t()).float() + b_j.unsqueeze(0)
    lse = torch.logsumexp(logits, dim=1)
    if top_m is not None and 0 < top_m < k:
        tv, ti = torch.topk(logits, k=top_m, dim=1)
        r_sparse = torch.softmax(tv, dim=1)
        return r_sparse, ti, None, lse
    r_dense = torch.exp(logits - lse.unsqueeze(1))
    return r_dense, None, None, lse
