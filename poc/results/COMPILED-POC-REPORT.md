# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)

**Generated**: 2026-11-21 02:32:08  
**Completed Tiers**: PoC 1, PoC 2, PoC 3, PoC 4

---

## PoC 1: Exact Mathematical Identities, PSD Floor & Memory/HBM Audit

- **Flash-GMM 2d-GEMM Max $|\Delta \ln p|$**: `2.842e-14`
- **Woodbury + Determinant Lemma Max $|\Delta \ln p|$**: `1.421e-14`
- **Rank-1 Collapsed Batch Min Eigenvalue**: `2.664103e-02` (floor `1.0e-03`)
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

## PoC 2: Anisotropic Zipf-GMM Collapse & Spectral Splitting Ablation

| Method | Splits | Dead | Max/Mean | Norm Entropy | Sing/Floor Ev | Train NLL | Test NLL | Test BIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `online_em_beta0.60` | 0.0 | 5.3 | 5.6x | 0.7361 | 0.0 | 13.4997 | 15.1395 | 40580.5 |
| `online_em_beta0.85` | 0.0 | 0.3 | 3.3x | 0.8933 | 0.0 | 13.7426 | 14.9859 | 40334.7 |
| `emeans_posthoc` | 2.3 | 0.0 | 1.93x | 0.9575 | 0.0 | 13.6269 | 14.6991 | 39875.9 |
| `egmm_no_revive` | 0.0 | 13.7 | 9.96x | 0.2652 | 0.0 | 17.1296 | 17.6951 | 44669.4 |
| `egmm_batch_resample` | 72.3 | 1.7 | 6.43x | 0.7119 | 0.0 | 13.7360 | 15.0559 | 40446.7 |
| `egmm_faiss_alt` | 39.7 | 0.0 | 2.33x | 0.9416 | 0.0 | 13.8253 | 14.9383 | 40258.6 |
| `egmm_random_orth` | 54.3 | 0.3 | 2.18x | 0.9279 | 0.0 | 14.4389 | 15.5511 | 41239.0 |
| `egmm_spectral_v1` | 44.7 | 0.3 | 2.44x | 0.9369 | 0.0 | 13.7731 | 14.9151 | 40221.5 |

---

## PoC 3: Empirical Dimension-Aware Scaling Law & Unseen-$d$ Transfer

- **Sweep Size**: 48 $(n, d, k, B)$ configurations on `StretchedGaussian`
- **Fitted $d$-Aware Law**: $\alpha_0 = 2.6728 \cdot k^{+0.16} B^{+0.85} n^{-0.57} d^{-1.18}$ ($R^2 = 0.7218$ vs. `0.2524` without $d$)
- **LODO-$d$ Rel Error**: `82.09%` (with $d^{a_d}$) vs. `161.79%` (without $d$)
- **LOSO-$n$ Rel Error**: `68.71%`
- **Zero-Shot Unseen $d=24$ Transfer**: Oracle NLL `28.6920` | With-$d$ Law `28.7086` (`+0.0166` nats, $\alpha_0=0.1216$) | No-$d$ Law `28.7102` (`+0.0182` nats, $\alpha_0=0.1304$)

---

## PoC 4: Top-$m$ Sparsity, Low-Rank Woodbury & Finalization Ablations

### Top-$m$ Sparse Responsibility Truncation (`k=32`)
| Top-$m$ | M-Step Work Ratio | Test NLL | Fit Time (ms) |
| :---: | :---: | :---: | ---: |
| `2` | 0.0625x | 15.3042 | 38.3 |
| `4` | 0.125x | 15.0457 | 88.6 |
| `8` | 0.25x | 14.8106 | 39.3 |
| `dense` | 1.0x | 14.8101 | 39.3 |

### Rotated Correlated Embeddings (`diag` vs `low_rank_diag` vs `full`)
| Covariance Mode | Effective Params | Test NLL | Test BIC |
| :--- | ---: | :---: | ---: |
| `diag` | 263 | 14.5308 | 25007.3 |
| `low_rank_diag` | 727 | 11.8136 | 23761.5 |
| `full` | 1223 | 10.2735 | 24612.9 |

### Finalization Strategy (`main_ema` vs `last_epoch` vs `slow_ema`)
| Finalizer | Train NLL | Test NLL |
| :--- | :---: | :---: |
| `main_ema` | 23.8844 | 24.6012 |
| `last_epoch` | 23.8566 | 24.6297 |
| `slow_ema` | 23.9431 | 24.5706 |
