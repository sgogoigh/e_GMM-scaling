# Research Design & Formal Hypotheses (`e-GMM`)

**Working Paper Title**: *"e-GMM: Scaling Expectation-Maximization to Billion-Scale Datasets via Exponential Moving Averages"*  
**Research Track**: Tier 1 Extension of `e-means` (`arXiv:2611.09501`)  
**Total Cluster Allocation**: `200–300 GPU-hours` on `1× to 2× NVIDIA A100 / H100 (80 GB)`

---

## 1. The Research Subject & Core Crux

### Why Expectation-Maximization Still Fails at Billion Scale
Gaussian Mixture Models (GMMs) provide calibrated density estimation, soft posterior responsibilities, and anisotropic uncertainty modeling that hard Euclidean `k-means` lacks. Yet two decades of scaling attempts hit four intertwined barriers:
1. **Full-Batch EM Wall ($O(nkd^2)$ / $O(nkd)$)**: Requires 25–100 full dataset passes; intractable when $n \ge 10^8$ or $k \ge 10^4$.
2. **Robbins-Monro Fragility in Online EM (Cappé & Moulines, 2009)**: Relies on polynomial step decay $\gamma_t = (t + t_0)^{-\beta}$, $\beta \in (0.5, 1]$, with no scaling formula for $(n, k, d, B, T)$. Aggressive decay stalls initialization; slow decay prevents convergence.
3. **Covariance Singularity & Dead Components**: Whenever a mini-batch routes fewer than $d$ effective points to component $j$, sample covariance becomes rank-deficient ($\det \Sigma_j \to 0$), causing Cholesky `LinAlgError` or $\ln p \to +\infty$, while low-weight components permanently starve ($r_{ij} \to 0$).
4. **HBM & Parameter Memory Walls**: Storing full covariances at $k=65{,}536, d=1024$ requires **`256.0 GiB` (`274.88 GB`)** for $Q_j$ alone, while materializing $(B \times k)$ E-step matrices at $B=262{,}144, k=65{,}536$ consumes **`64.0 GiB`** of HBM.

### The `e-GMM` Solution (Three Primitives + Two Systems Enablers)
1. **3-Tuple FP32 EMA Sufficient Statistics + Regularized Readout**: Track $(N_j, S_j, Q_j)$ in FP32 with cosine schedule $\alpha(t)$ down to $0.1\alpha_0$, and read out $\Sigma_j = Q_j/N_j - \mu_j\mu_j^\top + \lambda_0 I_d \succeq \lambda_0 I_d \succ 0$ (with Batch-1 prior mass $\ge 1.0$ to prevent early rank collapse).
2. **Dimension-Aware Power-Law Tuning**: Fit $\alpha_0(k, B, n, T, d)$ and $B_{\text{knee}}(k, n, T, d)$ once on a proxy domain to eliminate manual hyperparameter sweeps on new datasets.
3. **Spectral Component Splitting Along $\mathbf{v}_1$**: Every 5 batches, detect starved slots ($N_j < \tau \bar{N}$) and split overloaded donors (weighted by variance mass $N_{j^*} \operatorname{tr}(\Sigma_{j^*})$) along their principal covariance eigenvector $\mathbf{v}_1$ with exact second-moment conservation.
4. **Flash-GMM `2d`-GEMM + Top-$m$ + Low-Rank Woodbury**: Turn diagonal E-steps into a single $(B \times 2d) \times (2d \times k)$ Tensor-Core GEMM with streaming `logsumexp` + Top-$m$ sparse M-step accumulation, and handle correlated high-$d$ embeddings via $\Sigma_j = D_j + U_j U_j^\top$ ($O(kdr)$).

---

## 2. Formal Research Questions & Testable Hypotheses (`H1`–`H7`)

| ID | Research Question | Formal Hypothesis & Falsification Criterion | Primary Experiment |
| :---: | :--- | :--- | :--- |
| **H1** | Does ambient dimension $d$ stabilize `e-GMM` scaling laws? | **H1**: Including $d^{a_d}$ in $\ln \alpha_0 = \ln C_\alpha + a_k \ln k + a_B \ln B + a_n \ln n + a_T \ln T + a_d \ln d$ achieves $a_d < 0$, raises OLS $R^2$ over the $d$-agnostic law, and cuts LODO-$d$ extrapolation error by $\ge 1.5\times$. *Falsified if $R^2_{\text{with } d} \le R^2_{\text{no } d}$ or LODO error worsens.* | `exp_01` (Phase A) |
| **H2** | Do fitted $(n, k, d, B, T)$ laws generalize zero-shot OOD? | **H2**: Laws fit on `DEEP` / `StretchedGaussian` transfer zero-shot to unseen $(d, n)$ and OOD datasets (`SIFT`, `SPACEV`, `ImageNet-1k`) with low NLL regret vs. exhaustive oracle grid search. | `exp_01` (Phase A) |
| **H3** | Does `e-GMM` eliminate singularities and beat Online EM & Full-Batch EM at matched epoch budgets? | **H3**: Across 1–3 epochs on `SIFT1M`, `DEEP1B`, `DINOv2-100M`, and `ImageNet-1k`, `e-GMM` produces **0** Cholesky/singularity failures, lower collapsed-component rate, and strictly faster wall-clock time to target Test NLL than Full-Batch EM and swept Cappé & Moulines Online EM. | `exp_02` (Phase B) |
| **H4** | When does joint soft `e-GMM` beat `e-means + PostHoc`? | **H4**: On cleanly separated clusters (`SIFT`/`Zipf-Separated`), `e-means + PostHoc` is a near-tie baseline; on **overlapping, correlated, or anisotropic mixtures** (`DINOv2`, `ImageNet-1k`, `Overlapping-Aniso`), joint soft `e-GMM` achieves statistically significantly lower Test NLL ($\Delta \text{NLL} < 0$, paired $t$-test $p < 0.05$ across $\ge 5$ seeds). | `exp_02` (Phase B) |
| **H5** | Does spectral $\mathbf{v}_1$ splitting outperform alternative revival policies? | **H5**: Spectral splitting along $\mathbf{v}_1$ with variance-mass donor selection achieves lower Test NLL and fewer dead components than no-revival, batch resampling, Faiss alternating $\pm \delta$, and random orthogonal splits. | `exp_03` (Phase C) |
| **H6** | Do `low_rank_diag` (Woodbury) and Top-$m$ sparsity preserve density quality at huge memory/FLOP savings? | **H6**: `low_rank_diag` ($r \in \{8, 16\}$) strictly beats `diag` on correlated embeddings and achieves superior BIC vs. `full` while saving $>20\times$ VRAM; Top-$m$ ($m \in \{8, 16\}$) stays within $< 0.01$ nats/dim of dense soft EM while cutting M-step work by $k/m$. | `exp_03` & `exp_04` |
| **H7** | Does dual finalization separate Train NLL optimality from Test NLL generalization? | **H7**: `last_epoch` exact EM minimizes Train NLL, whereas passive `slow_ema` shadow statistics reduce finite-sample variance and win held-out Test NLL in low-$n/k$ regimes. | `exp_03` (Phase C) |

---

## 3. Experimental Controls & Baseline Fairness Requirements

Enforced automatically across `exp_01`–`exp_04` and verified by `tests/test_diagnostics_and_baselines.py`:
1. **Shared Holdout & Exact Evaluator**: Every comparator outputs $(\pi, \mu, \Sigma)$ evaluated through the identical `evaluate_gmm_nll()` function over identically deduplicated 10,000-vector validation / 10,000-vector test splits (`e-means` Appendix A protocol).
2. **Matched Data & Epoch Budgets**: All mini-batch comparators consume identical total passes $T \cdot n$ and identical batch size $B$ (except when $B_{\text{knee}}$ itself is under study in Phase A).
3. **Transparent Stability Accounting**: Online EM baselines must log `singular_events` and `dead_components` rather than hiding rank collapse behind silent jitter.
4. **Both Separated & Overlapping Regimes**: Every benchmark suite against `e-means + PostHoc` must include overlapping/correlated mixtures where hard Voronoi partitioning fails.
5. **Multi-Seed Replication**: Smoke mode uses `2–3` seeds; full GPU runs use **`5–10` seeds** and report `mean ± std` + paired effect size.
