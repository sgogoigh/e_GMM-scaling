# Base Terms & Core Concepts Index (`docs/basic-terms/`)

This directory breaks down every foundational equation, numerical stability trap, kernel optimization, and statistical concept required before reading or running the **`e_GMM-scaling`** codebase.

---

## Recommended Learning Sequence

| Order | Document | Core Topics Covered |
| :---: | :--- | :--- |
| **01** | [**`e-means` Primitives & 3-Tuple EMA**](./01-e-means-Primitives-and-3Tuple-EMA.md) | Extending $(N_j, S_j)$ hard clustering to 3-tuple Gaussian sufficient statistics $(N_j, S_j, Q_j)$, convex hull & PSD guarantees, FP32/FP64 variance readout, and catastrophic cancellation prevention |
| **02** | [**Online EM (Cappé-Moulines) & Collapse**](./02-Online-EM-Cappe-Moulines-and-Collapse.md) | Full-batch EM vs. Cappé & Moulines (2009) curved exponential family stochastic EM, Robbins-Monro $\gamma_t = (t+t_0)^{-\beta}$ sensitivity, covariance singularity ($\det\Sigma_j \to 0$), starved components, and the 275 GB $O(kd^2)$ memory wall |
| **03** | [**Covariance Parametrizations & Woodbury**](./03-Covariance-Parametrizations-and-Woodbury.md) | Diagonal $O(kd)$, Low-Rank + Diagonal $O(kdr)$ ($\Sigma_j = D_j + U_j U_j^\top$), and Full $O(kd^2)$ covariances; Woodbury matrix identity, matrix determinant lemma, and Cholesky jitter fallback |
| **04** | [**Flash-GMM Kernels & Top-$m$ Sparsity**](./04-Flash-GMM-Kernels-and-TopM-Sparsity.md) | Exact $2d$-augmented single-GEMM diagonal E-step reformulation, streaming SRAM `logsumexp`, precision-split rule (BF16 E-step + FP32 M-step), and Top-$m$ sparse responsibility truncation |
| **05** | [**Spectral Component Splitting & Finalization**](./05-Spectral-Component-Splitting-and-Finalization.md) | Collapsed component detection ($N_j < \tau \bar{N}$), donor sampling $\propto N_{j^*}$, principal eigenvector $\mathbf{v}_1$ (argmax vs. 5-step power iteration), exact $Q_{\text{child}\pm}$ second-moment consistency, and Last-Epoch vs. Slow-EMA finalization |
| **06** | [**Dimension-Aware Scaling Laws & Master Glossary**](./06-Dimension-Aware-Scaling-Laws-and-Glossary.md) | Why ambient $d$ dropped out in `e-means` vs. why Mahalanobis quadratic forms & $\log\det\Sigma_j$ require $d^{a_d}$ in `e-GMM`, NLL regret, LOSO/LODO/OOD checks, and complete A–Z glossary |

---

## Visual Architecture Guides (`../assets/images/` & `../architecture/`)

- [**3-Tuple EMA & Spectral Splitting Pipeline**](../assets/images/egmm-3tuple-pipeline.svg)
- [**Flash-GMM 2d-Augmented GEMM Reformulation**](../assets/images/flash-gmm-gemm-reformulation.svg)
- [**Spectral Splitting Geometry Along $\mathbf{v}_1$**](../assets/images/spectral-splitting-geometry.svg)
- [**Covariance Parametrizations & The 275 GB Memory Wall**](../assets/images/covariance-parametrizations-memory.svg)
- [**Full Triton/PyTorch Implementation & 200–300 GPU-Hour Plan**](../architecture/IMPLEMENTATION-PLAN-EGMM.md)
- [**Repository README**](../../README.md)
