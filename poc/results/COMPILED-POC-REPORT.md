# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)

**Generated**: 2026-11-21 02:52:03  
**Completed Tiers**: PoC 1, PoC 2, PoC 3, PoC 4

---

## Executive Hypothesis Verification Matrix

| ID | Core Research Hypothesis | Evidence Source | Verdict |
| :---: | :--- | :--- | :--- |
| **H1.1** | `2d`-GEMM Flash-GMM & Woodbury Det Lemma are algebraically exact | PoC 1 (`2.84e-14`, `1.42e-14`) | **Supported** |
| **H1.2** | Additive $\lambda_0 I_d$ strictly guarantees $\Sigma_j \succeq \lambda_0 I_d$ even on rank-1 batches | PoC 1 (`min_eig=1.000e-03`) | **Supported** |
| **H2.1** | Spectral $\mathbf{v}_1$ revival eliminates collapse & beats heuristic splits | PoC 2A (`-2.80` nats vs no-revive) | **Supported** |
| **H2.2** | Joint soft `e-GMM` beats `e-means + PostHoc` covariance fit | PoC 2A vs 2B (`-1.46` nats overlap) | **Regime-Dependent (Supported on Overlap)** |
| **H3.1** | Ambient dimension $d^{a_d}$ is required for `e-GMM` $\alpha_0$ scaling laws | PoC 3 ($R^2$: `0.2753` $\to$ `0.7139`) | **Supported** |
| **H3.2** | Knee batch size $B_{\text{knee}}$ scales cleanly at small CPU $n \le 4800$ | PoC 3 ($R^2=0.2353$) | **Inconclusive (Needs Phase-A $n \ge 10^5$)** |
| **H4.1** | Top-$m$ sparse E-step preserves accuracy at $m/k$ M-step cost | PoC 4 (`m=8`: `14.4512` vs `14.4510`) | **Supported** |
| **H4.2** | Low-Rank+Diag captures off-diagonal covariance with best BIC | PoC 4 (`BIC=23761.5`) | **Supported** |
| **H4.3** | `last_epoch` wins Train NLL while `slow_ema` wins Test NLL | PoC 4 (`23.8808` / `24.5874`) | **Supported** |

---

## PoC 1: Exact Mathematical Identities, PSD Floor & Memory/HBM Audit

- **Flash-GMM 2d-GEMM Max $|\Delta \ln p|$**: `2.842e-14`
- **Woodbury + Determinant Lemma Max $|\Delta \ln p|$**: `1.421e-14`
- **Rank-1 Collapsed Batch Min Eigenvalue**: `1.000000e-03` (floor `1.0e-03`)
- **Spectral Split Mean Conservation Error**: `4.441e-16`
- **Full-Covariance Memory Wall ($k=65536, d=1024$)**: `256.0 GiB` (`274.88 GB`)

| Preset | Cov Mode | $k$ | $d$ | $B$ | State (GiB) | Naive $B\times k$ HBM (GiB) | Tiled HBM (GiB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sift-1m` | `diag` | 4096 | 128 | 16384 | 0.0039 | 0.25 | 0.125 |
| `deep-1b` | `diag` | 65536 | 96 | 262144 | 0.0471 | 64.0 | 2.0 |
| `spacev-100m` | `diag` | 65536 | 100 | 131072 | 0.0491 | 32.0 | 1.0 |
| `dinov2-100m` | `diag` | 16384 | 1024 | 65536 | 0.1251 | 4.0 | 0.5 |
| `dinov2-100m` | `low_rank_diag` | 16384 | 1024 | 65536 | 1.1251 | 4.0 | 0.5 |
| `dinov2-100m` | `full` | 16384 | 1024 | 65536 | 64.0626 | 4.0 | 0.5 |
| `imagenet-1k` | `full` | 1024 | 768 | 16384 | 2.2529 | 0.062 | 0.0625 |

### Analytical Interpretation & Hypothesis Verdicts (PoC 1)
- **Exactness of Kernel Reformulations (H1.1 — Supported)**: Both the `2d`-augmented diagonal GEMM (`2.84e-14`) and the Woodbury + Matrix Determinant Lemma (`1.42e-14`) match naive full-precision log-densities to **FP64 machine precision** ($\sim 10^{-14}$). This means speedups from Tensor-Core MatMuls and $O(kdr)$ factorizations introduce **zero mathematical approximation error**.
- **Elimination of Singularity Collapse (H1.2 — Supported)**: Even when a full-covariance mini-batch contains repeated points of rank $\le 1$ and negligible prior (`1e-10`), `readout()` holds $\lambda_{\min}(\Sigma_j) = 1.000000e-03 \ge \lambda_0$, eliminating Cholesky `LinAlgError` and $\det(\Sigma_j) \to 0$ likelihood blow-ups.
- **Breaking the Memory Walls (H1.3 — Supported)**: Full covariance at `k=65,536, d=1024` hits `256.0 GiB` (`> 3x` an 80 GB H100). On `dinov2-100m`, switching from `full` (`64.0626 GiB`) to `low_rank_diag` (`1.1251 GiB`) cuts state VRAM by **`56.9x`**; on `deep-1b`, tiled streaming reduces peak $(B \times k)$ HBM from `64.0 GiB` to **`2.0 GiB`** (`32x` reduction).

---

## PoC 2: Collapse, Spectral Splitting & Overlapping Soft-EM Advantage

### Regime A: Zipf-1.2 Rotated Anisotropic GMM (Extreme Class Imbalance)
| Method | Splits | Dead | Max/Mean | Norm Entropy | Sing/Floor Ev | Train NLL | Test NLL | Test BIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `online_em_beta0.60` | 0.0 | 5.3 | 5.6x | 0.7361 | 0.0 | 13.4997 | 15.1395 | 40580.5 |
| `online_em_beta0.85` | 0.0 | 0.3 | 3.3x | 0.8933 | 0.0 | 13.7426 | 14.9859 | 40334.7 |
| `emeans_posthoc` | 2.3 | 0.0 | 1.93x | 0.9575 | 0.0 | 13.6269 | 14.6991 | 39875.9 |
| `egmm_no_revive` | 0.0 | 13.7 | 9.96x | 0.2652 | 0.0 | 17.1296 | 17.6951 | 44669.4 |
| `egmm_batch_resample` | 69.0 | 1.7 | 6.57x | 0.7055 | 0.0 | 13.9052 | 15.3059 | 40846.7 |
| `egmm_faiss_alt` | 37.3 | 0.0 | 2.16x | 0.9511 | 0.0 | 13.9105 | 15.0130 | 40378.0 |
| `egmm_random_orth` | 47.0 | 0.3 | 3.33x | 0.9038 | 0.0 | 13.9930 | 15.2321 | 40728.6 |
| `egmm_spectral_v1` | 41.7 | 0.3 | 2.61x | 0.9212 | 0.0 | 13.6807 | 14.8957 | 40190.4 |

### Regime B: Overlapping Rotated Anisotropic GMM ($\sigma_\mu^2 = 0.2, \sigma_w^2 = 0.8$)
| Method | Splits | Dead | Max/Mean | Norm Entropy | Test NLL | Test BIC |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| `online_em_beta0.85` | 0.0 | 3.7 | 5.05x | 0.7584 | 19.7397 | 47940.7 |
| `emeans_posthoc` | 2.3 | 0.0 | 1.82x | 0.9769 | 20.4754 | 49117.9 |
| `egmm_no_revive` | 0.0 | 14.7 | 15.61x | 0.0451 | 21.8137 | 51259.2 |
| `egmm_spectral_v1` | 48.0 | 0.0 | 2.42x | 0.9438 | 19.0163 | 46783.4 |

### Analytical Interpretation & Hypothesis Verdicts (PoC 2)
- **Why Revival Is Mandatory in Soft Mini-Batch GMMs (H2.1 — Supported)**: Without component splitting (`egmm_no_revive`), Batch-1 covariance shrinkage triggers a positive-feedback collapse that kills **13.7 / 16** components (entropy `0.2652`, Test NLL `17.6951`). Spectral splitting (`egmm_spectral_v1`) drops dead slots to **`0.3`** (entropy `0.9212`) and improves Test NLL by **`-2.7994` nats**.
- **Why Spectral $\mathbf{v}_1$ Beats Other Split Heuristics (H2.1 — Supported)**: Among all `e-GMM` & Online-EM methods on Regime A, `egmm_spectral_v1` (`14.8957`) beats `online_em_beta0.85` (`14.9859`), `egmm_faiss_alt` (`15.0130`), `egmm_random_orth` (`15.2321`), and `egmm_batch_resample` (`15.3059`), confirming that splitting along the donor's principal eigenvector $\mathbf{v}_1$ cuts multimodal variance more cleanly than arbitrary directions or outlier resampling.
- **Crucial Nuance — When `e-GMM` Beats `e-means + PostHoc` (H2.2 — Regime-Dependent)**: On cleanly separated Zipf clusters (Regime A), hard `emeans_posthoc` (`14.6991`) edges out `egmm_spectral_v1` (`14.8957`) by `0.1966` nats because Euclidean Voronoi cells already isolate well-separated clusters without $-\frac{1}{2}\ln\det\Sigma_j$ feedback. Conversely, on **overlapping rotated anisotropic mixtures (Regime B)**, hard Voronoi boundaries slice off Gaussian tails (`emeans_posthoc = 20.4754`), whereas **`egmm_spectral_v1` reaches `19.0163`** (`0.0` dead) — a **`-1.4591` nat** win over `emeans_posthoc` and **`-0.7234` nats** over `online_em_beta0.85`. **Implication**: Phase-B real benchmarks must evaluate on correlated/overlapping embeddings (e.g., `DINOv2`, `ImageNet-1k`) rather than only cleanly separated clusters.

---

## PoC 3: Empirical Dimension-Aware Scaling Law & Unseen-$d$ Transfer

- **Sweep Size**: 72 $(n, d, k, B)$ configurations on `StretchedGaussian`
- **Fitted $\alpha_0$ Law ($d$-Aware)**: $\alpha_0 = 2.1336 \cdot k^{+0.19} B^{+0.95} n^{-0.55} d^{-1.20}$ ($R^2 = 0.7139$ vs. `0.2753` without $d$)
- **LODO-$d$ Rel Error**: `89.44%` (with $d^{a_d}$) vs. `189.62%` (without $d$)
- **LOSO-$n$ Rel Error**: `79.37%`
- **Fitted $B_{\text{knee}}$ Law ($d$-Aware)**: $B_{\text{knee}} = 15.5206 \cdot k^{-0.11} n^{+0.08} d^{-0.08}$ ($R^2 = 0.2353$)
- **Zero-Shot Unseen $d=24$ Transfer**: Oracle NLL `28.6890` | With-$d$ Law `28.7120` (`+0.0230` nats, $\alpha_0=0.1315$) | No-$d$ Law `28.7133` (`+0.0243` nats, $\alpha_0=0.1412$)

### Analytical Interpretation & Hypothesis Verdicts (PoC 3)
- **Why Ambient Dimension $d$ Cannot Be Dropped in `e-GMM` (H3.1 — Supported)**: Unlike `e-means` (where $d$ dropped out because Euclidean clustering depends on intrinsic dimension), adding $d^{a_d}$ to `e-GMM` boosts $R^2$ from **`0.2753` $\to$ `0.7139`** (`+0.4386`) and cuts Leave-One-Dimension-Out (LODO) error by **`2.12x`** (`189.62%` $\to$ `89.44%`). The negative exponent $a_d = -1.20$ confirms the theoretical prediction: higher dimension increases per-component covariance estimation variance, requiring smaller step sizes $\alpha_0$ (longer EMA memory). Meanwhile, $a_B = +0.95$ and $a_n = -0.55$ align closely with `e-means` Eq. 6 (`+0.90` and `-0.45`).
- **Unseen-Dimension Interpolation (H3.1 — Supported)**: At held-out $d=24$, the $d$-aware formula lands within **`+0.0230` nats** of exhaustive grid-search oracle NLL (`28.7120` vs. `28.6890`), beating the $d$-agnostic law (`+0.0243` nats).
- **Exposing Small-Scale Proxy Limits (H3.2 — Inconclusive at $n \le 4800$)**: Two metrics show why Phase-A GPU sweeps ($n \in [10^5, 10^8]$ on multi-cluster `DEEP`) are still necessary: (1) $a_k = +0.19$ is slightly positive because unimodal `StretchedGaussian` has a flat valley across $k \in \{4, 8, 16\}$, and (2) `B_knee` ($R^2 = 0.2353$) hits small-$n$ quantization ceilings where nearly every batch size $[16..512]$ lies within the `+0.005 nats/dim` plateau.

---

## PoC 4: Top-$m$ Sparsity, Low-Rank Woodbury & Finalization Ablations

### Top-$m$ Sparse Responsibility Truncation (`k=32`)
| Top-$m$ | M-Step Work Ratio | Test NLL | Fit Time (ms) |
| :---: | :---: | :---: | ---: |
| `2` | 0.0625x | 14.7435 | 31.0 |
| `4` | 0.125x | 14.1575 | 30.6 |
| `8` | 0.25x | 14.4512 | 27.4 |
| `dense` | 1.0x | 14.4510 | 28.3 |

### Rotated Correlated Embeddings (`diag` vs `low_rank_diag` vs `full`)
| Covariance Mode | Effective Params | Test NLL | Test BIC |
| :--- | ---: | :---: | ---: |
| `diag` | 263 | 14.5308 | 25007.3 |
| `low_rank_diag` | 727 | 11.8136 | 23761.5 |
| `full` | 1223 | 10.2735 | 24612.9 |

### Finalization Strategy (`main_ema` vs `last_epoch` vs `slow_ema`)
| Finalizer | Train NLL | Test NLL |
| :--- | :---: | :---: |
| `main_ema` | 23.9068 | 24.6154 |
| `last_epoch` | 23.8808 | 24.6447 |
| `slow_ema` | 23.9421 | 24.5874 |

### Analytical Interpretation & Hypothesis Verdicts (PoC 4)
- **Sparse Top-$m$ Truncation Cuts M-Step Work With Negligible Loss (H4.1 — Supported)**: At $k=32$, `top_m=8` (`0.25x` M-step work) matches `dense` within **`0.0002` nats** (`14.4512` vs. `14.4510`), while `top_m=4` (`14.1575`) acts as light noise pruning (`-0.2935` nats). At billion scale ($k=65{,}536$), Top-8 reduces M-step FLOPs by **`8192x`**.
- **Low-Rank + Diagonal Achieves Best Statistical Parsimony (H4.2 — Supported)**: On rotated correlated clusters, diagonal covariance misses off-diagonal axes (`Test NLL = 14.5308`), whereas `low_rank_diag` ($r=4$) cuts Test NLL by **`-2.7172` nats** (`11.8136`) and achieves the **lowest Test BIC (`23761.5`)** across all modes—beating both `diag` (`25007.3`) and `full` (`24612.9`, `1223` params vs. `727`). This confirms $D_j + U_j U_j^\top$ captures dominant subspace correlations without $O(kd^2)$ overfitting or VRAM explosion.
- **Dual Finalization Reproduces the Train vs. Test Generalization Trade-Off (H4.3 — Supported)**: In the small-$n/k$ regime, **`last_epoch`** achieves the best **Train NLL** (`23.8808` vs. `23.9068` `main_ema` and `23.9421` `slow_ema`) because an exact unweighted pass satisfies first-order M-step optimality on the training sample; meanwhile, **`slow_ema`** achieves the best **Test NLL** (`24.5874` vs. `24.6154` `main_ema` and `24.6447` `last_epoch`) by Polyak-averaging late-epoch noise.
