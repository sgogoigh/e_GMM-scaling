# Compute Allocation Proposal Template (`1–2× A100 / H100` Request Backed by PoC 1–4)

Use the empirical telemetry automatically compiled into `poc/results/COMPILED-POC-REPORT.md` to fill in the bracketed fields below when requesting university HPC hours, lab GPUs, or cloud research credits.

---

## 1. Project Summary
- **Title**: *e-GMM: Scaling Expectation-Maximization to Billion-Scale Datasets via Exponential Moving Averages*
- **Target Benchmarks**: `SIFT1M` ($d=128$), `DEEP1B` ($d=96, k=65{,}536$), `DINOv2-100M` ($d=1024$), and `ImageNet-1k` embeddings ($d=768$).
- **Core Bottleneck Addressed**: Classical full-batch EM is $O(nkd^2)$ per pass; Cappé & Moulines (2009) Online EM suffers from Robbins-Monro $(\beta, t_0)$ tuning fragility, rank-deficient covariance singularities ($\det \Sigma_j \to 0$), unrevived dead components, and a `256 GiB (274.9 GB)` $O(kd^2)$ full-covariance memory wall at $k=65{,}536, d=1024$.

## 2. Empirical Proof-of-Concept Evidence Completed
1. **Exact Numerical & Algebraic Identities (PoC 1)**:
   - Proved strict positive-definiteness ($\lambda_{\min}(\Sigma_j) \ge \lambda_0$) even on rank-1 collapsed mini-batches.
   - Verified `2d`-augmented **Flash-GMM GEMM** equivalence ($\max |\Delta \ln p| \le$ `[PoC1 Flash diff]`) and **Woodbury + Matrix Determinant Lemma** equivalence ($\max |\Delta \ln p| \le$ `[PoC1 Woodbury diff]`), cutting `DINOv2-100M` state memory from `64.06 GB` (`full`) to `1.13 GB` (`low_rank_diag`, $r=16$) and `DEEP1B` `B x k` HBM from `64.0 GB` to `< 2.0 GB`.
2. **Collapse Prevention & Spectral Revival Advantage (PoC 2)**:
   - On Zipf-imbalanced anisotropic mixtures, `e-GMM (Spectral v1)` achieved `0` dead components and Test NLL `[PoC2 spectral NLL]`, outperforming `e-means + PostHoc` (`[PoC2 posthoc NLL]`), `e-GMM (No Revival)` (`[PoC2 norevive NLL]`, `[dead]` dead), and Cappé & Moulines Online EM (`[PoC2 oem NLL]`).
3. **Dimension-Aware Scaling Law & OOD Transfer (PoC 3)**:
   - Across an empirical $(n, d, k, B)$ sweep on `StretchedGaussian`, adding $d^{a_d}$ improved LODO-$d$ extrapolation error from `[PoC3 no-d LODO %]` to `[PoC3 with-d LODO %]` ($R^2 = \text{`[PoC3 R2]`}$) and transferred zero-shot to OOD `Zipf-GMM` with NLL gap `[PoC3 OOD gap]`.
4. **Systems Optimizations & Generalization Finalizer (PoC 4)**:
   - Verified Top-$m$ sparse responsibility truncation (`k/m` M-step speedup), `low_rank_diag` closing the rotated-correlation NLL gap vs `diag`, and `slow_ema` reducing Test NLL overfitting relative to `main_ema` / `last_epoch` at low $n/k$.

## 3. Requested Compute Breakdown (`200–300 A100/H100 GPU-Hours`)
- **Phase A (Dimension-Aware Scaling Sweeps)**: `~80 GPU-hrs` — Fit $\alpha_0(k, B, n, T, d)$ & $B_{\text{knee}}$ across $d \in \{32, 96, 256, 768\}$, $k \in [64, 16384]$, $n \in [10^5, 10^8]$ on `DEEP` & `StretchedGaussian`; verify LOSO, LODO, and OOD NLL regret.
- **Phase B (Billion-Scale Vector & Vision Benchmarks)**: `~120 GPU-hrs` — `SIFT1M`, `DEEP1B`, `DINOv2-100M`, `ImageNet-1k` against Full-Batch EM, Cappé & Moulines Online EM, and `e-means + PostHoc`.
- **Phase C (Ablations & Kernel Profiling)**: `~50 GPU-hrs` — Spectral $\mathbf{v}_1$ vs. random/Faiss splits, $\lambda_0$ & covariance mode trade-offs, Top-$m$ truncation, and Last-Epoch vs. Slow-EMA.
- **Expected Risk**: **Low** — every mathematical invariant, Woodbury identity, Flash-GMM kernel parity, and scaling-law pipeline has been verified across PoCs 1–4.
