# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)

**Generated**: 2026-11-21 02:38:34  
**Completed Tiers**: PoC 1, PoC 2, PoC 3, PoC 4

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

---

## PoC 3: Empirical Dimension-Aware Scaling Law & Unseen-$d$ Transfer

- **Sweep Size**: 72 $(n, d, k, B)$ configurations on `StretchedGaussian`
- **Fitted $\alpha_0$ Law ($d$-Aware)**: $\alpha_0 = 2.1336 \cdot k^{+0.19} B^{+0.95} n^{-0.55} d^{-1.20}$ ($R^2 = 0.7139$ vs. `0.2753` without $d$)
- **LODO-$d$ Rel Error**: `89.44%` (with $d^{a_d}$) vs. `189.62%` (without $d$)
- **LOSO-$n$ Rel Error**: `79.37%`
- **Fitted $B_{\text{knee}}$ Law ($d$-Aware)**: $B_{\text{knee}} = 15.5206 \cdot k^{-0.11} n^{+0.08} d^{-0.08}$ ($R^2 = 0.2353$)
- **Zero-Shot Unseen $d=24$ Transfer**: Oracle NLL `28.6890` | With-$d$ Law `28.7120` (`+0.0230` nats, $\alpha_0=0.1315$) | No-$d$ Law `28.7133` (`+0.0243` nats, $\alpha_0=0.1412$)

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
