# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)

**Generated**: 2026-11-21 02:08:28  
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

## PoC 2: Anisotropic Zipf-GMM Collapse & Spectral Splitting Ablation

| Method | Splits | Dead | Max/Mean | Norm Entropy | Sing/Floor Ev | Train NLL | Test NLL | Test BIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `online_em_beta0.60` | 0 | 6 | 8.191x | 0.7225 | 0 | 21.0275 | 21.1067 | 58986.9 |
| `online_em_beta0.85` | 0 | 7 | 8.58x | 0.7034 | 0 | 21.0650 | 21.0571 | 58868.0 |
| `emeans_posthoc` | 4 | 0 | 2.394x | 0.9496 | 0 | 21.1020 | 20.9692 | 58657.0 |
| `egmm_no_revive` | 0 | 10 | 5.856x | 0.7188 | 0 | 21.7207 | 21.7269 | 60475.4 |
| `egmm_batch_resample` | 14 | 0 | 3.749x | 0.8517 | 0 | 19.3578 | 19.4987 | 55127.7 |
| `egmm_faiss_alt` | 15 | 0 | 2.682x | 0.936 | 0 | 21.4842 | 21.3730 | 59626.0 |
| `egmm_random_orth` | 16 | 0 | 2.769x | 0.9445 | 0 | 20.7027 | 20.7642 | 58165.0 |
| `egmm_spectral_v1` | 15 | 0 | 3.508x | 0.9175 | 0 | 20.9349 | 20.9188 | 58535.9 |

---

## PoC 3: Empirical Dimension-Aware Scaling Law & OOD Transfer

- **Sweep Size**: 24 $(n, d, k, B)$ configurations on `AnisotropicGMM`
- **Fitted $d$-Aware Law**: $\alpha_0 = 0.1168 \cdot k^{+1.92} B^{-0.63} n^{+0.01} d^{+0.04}$ ($R^2 = 0.456$)
- **LODO-$d$ Rel Error**: `111.98%` (with $d^{a_d}$) vs. `128.6%` (without $d$)
- **LOSO-$n$ Rel Error**: `100.53%`
- **Zero-Shot OOD Zipf-GMM**: Oracle NLL `20.7874` | With-$d$ Law `21.3761` (`+0.5888` nats) | No-$d$ Law `21.3761` (`+0.5888` nats)

---

## PoC 4: Top-$m$ Sparsity, Low-Rank Woodbury & Finalization Ablations

### Top-$m$ Sparse Responsibility Truncation (`k=32`)
| Top-$m$ | M-Step Work Ratio | Test NLL | Fit Time (ms) |
| :---: | :---: | :---: | ---: |
| `2` | 0.0625x | 14.9609 | 34.1 |
| `4` | 0.125x | 15.3510 | 32.5 |
| `8` | 0.25x | 15.5054 | 38.5 |
| `dense` | 1.0x | 15.5458 | 32.2 |

### Rotated Correlated Embeddings (`diag` vs `low_rank_diag` vs `full`)
| Covariance Mode | Effective Params | Test NLL | Test BIC |
| :--- | ---: | :---: | ---: |
| `diag` | 263 | 14.5308 | 25007.3 |
| `low_rank_diag` | 727 | 13.9630 | 27200.4 |
| `full` | 1223 | 10.7477 | 25371.6 |

### Finalization Strategy (`main_ema` vs `last_epoch` vs `slow_ema`)
| Finalizer | Train NLL | Test NLL |
| :--- | :---: | :---: |
| `main_ema` | 23.9652 | 24.5946 |
| `last_epoch` | 23.9404 | 24.6211 |
| `slow_ema` | 23.9935 | 24.5688 |
