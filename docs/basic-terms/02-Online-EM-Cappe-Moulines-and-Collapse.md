# 02. Online EM (Cappé & Moulines, 2009) & The Four Bottlenecks

Before understanding why `e-GMM` introduces cosine schedules, covariance floors, diagonal/low-rank factorizations, and spectral splitting, it helps to examine why **classical Full-Batch EM** and **Cappé & Moulines (2009) Online EM** break down on large-scale, high-dimensional datasets.

---

## 1. Full-Batch EM (`scikit-learn`, `Faiss`)

Given dataset $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$, full-batch EM alternates over **every single point** on every iteration:
- **E-step** ($O(n k d^2)$ full / $O(n k d)$ diagonal):
  $$
  r_{ij} = \frac{\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{\ell=1}^{k} \pi_\ell \, \mathcal{N}(x_i \mid \mu_\ell, \Sigma_\ell)}
  $$
- **M-step**:
  $$
  N_j = \sum_{i=1}^{n} r_{ij}, \quad \pi_j = \frac{N_j}{n}, \quad \mu_j = \frac{1}{N_j}\sum_{i=1}^{n} r_{ij} x_i, \quad \Sigma_j = \frac{1}{N_j}\sum_{i=1}^{n} r_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top
  $$

**Why it fails at scale**:
- When $n = 10^8$ or $10^9$ and $k = 65{,}536$, a single E-step touches $n \cdot k = 6.55 \times 10^{12}$ pairs per iteration—just like Lloyd's algorithm requires 25–100 full passes!
- In `scikit-learn` (`GaussianMixture`), execution is single-node CPU, capped well below $n = 10^6, k = 10^3$.

---

## 2. Cappé & Moulines (2009) Online / Stochastic EM

Cappé & Moulines (2009) formulated Online EM as a **Robbins-Monro stochastic approximation** in the space of curved exponential family sufficient statistics $\hat{S}$:

$$
\hat{S}_t = (1 - \gamma_t) \hat{S}_{t-1} + \gamma_t \, \mathbb{E}_{\hat{\theta}_{t-1}}\big[ s(x_t, z_t) \mid x_t \big], \qquad \hat{\theta}_t = \bar{\theta}(\hat{S}_t)
$$

using polynomial Robbins-Monro step sizes:
$$
\gamma_t = (t + t_0)^{-\beta}, \qquad \beta \in (0.5, 1]
$$
satisfying the classical asymptotic conditions $\sum_t \gamma_t = \infty$ and $\sum_t \gamma_t^2 < \infty$.

---

## 3. Four Critical Bottlenecks of Classical Online EM

### Bottleneck 1: Extreme Sensitivity to $(\beta, t_0)$ and Batch Size $B$
Robbins-Monro theory only guarantees **asymptotic** convergence as $t \to \infty$; it provides **no closed-form formula** to choose $(\beta, t_0, B)$ for a fixed finite epoch budget $T \in [1.5, 5]$ across arbitrary $(n, k, d)$:
- If $\beta$ is too large (e.g., $\beta = 0.95$) or $t_0$ is too small, $\gamma_t$ decays too fast in Epoch 1, **freezing components near their bad random initialization**.
- If $\beta$ is too small (e.g., $\beta = 0.55$), late-epoch noise never settles, degrading held-out NLL.

### Bottleneck 2: Covariance Singularity & Likelihood Blow-Up
In maximum-likelihood GMMs, if a component $j$ has fewer than $d$ effective points ($N_j < d$)—which happens constantly in early mini-batches when $k$ is large and $d \ge 100$—its empirical covariance $\Sigma_j = \frac{1}{N_j}\sum r_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top$ has rank $< d$:
$$
\det(\Sigma_j) \to 0 \implies \ln \mathcal{N}(x_i \mid \mu_j, \Sigma_j) \to +\infty \quad \text{or} \quad \texttt{LinAlgError / NaN in Cholesky!}
$$
In ML literature (Bishop, 2006), this is the notorious **pathological singularity of MLE for Gaussian mixtures**: shrinking one Gaussian onto a single data point drives the log-likelihood to $+\infty$.

### Bottleneck 3: Dead / Starved Components (Zero Revival Mechanism)
Once a component's weight $\pi_j \to 0$ or its variance shrinks tightly onto a tiny outlier region, its log-density $\ln \pi_j + \ln \mathcal{N}(x_i \mid \mu_j, \Sigma_j)$ drops far below neighboring components for all $x_i \in \mathcal{B}$:
$$
r_{ij} \to 0 \implies \text{component } j \text{ receives zero future gradients or stats!}
$$
Classical Online EM has **no mechanism to detect and revive dead Gaussians**, leaving up to 15–40% of capacity permanently wasted on skewed / Zipf mixtures.

### Bottleneck 4: The $O(k d^2)$ Memory Wall
Storing full $d \times d$ covariance matrices for modern embeddings hits a hard GPU VRAM wall:
- At $k = 65{,}536, d = 1024$ (e.g., `DINO10B` / `DINOv3`):
  $$
  \text{Memory}(Q) = 65{,}536 \times 1024^2 \times 4\text{ bytes (FP32)} = \mathbf{274.88\text{ GB}}
  $$
  Even before storing $\mu_j$, Cholesky factors $L_j$, or Slow-EMA shadow buffers, $Q_j$ alone is **$3.4\times$ larger than an entire 80 GB H100 GPU**!

---

## 4. How `e-GMM` Solves All Four Bottlenecks Simultaneously

| Bottleneck in Online EM | `e-GMM` Solution | Module in `src/e_gmm/` |
| :--- | :--- | :--- |
| 1. Robbins-Monro $(\beta, t_0)$ fragility | Cosine schedule $\alpha(t)$ down to $0.1\alpha_0$ + **dimension-aware scaling law** $\alpha_0(k, B, n, T, d)$ | `schedule.py`, `scaling_law.py` |
| 2. Singular covariance ($\det\Sigma_j \to 0$) | Unconditional **covariance floor** $\Sigma_j = \frac{Q_j}{N_j} - \mu_j\mu_j^\top + \lambda_0 I_d \succeq \lambda_0 I_d$ | `numpy_egmm.py`, `torch_egmm.py` |
| 3. Dead / starved components | Periodic **spectral donor splitting** ($N_j < \tau \bar{N}$) along principal eigenvector $\mathbf{v}_1$ | `numpy_egmm.py`, `torch_egmm.py` |
| 4. $O(kd^2)$ VRAM & $O(Bk)$ HBM walls | **Diagonal Flash-GMM** ($O(kd)$, 2d-GEMM) & **Low-Rank+Diag Woodbury** ($O(kdr)$) | `torch_egmm.py`, `triton_kernels.py` |

---

## 5. Sources & Videos
- **Cappé & Moulines (2009)**: *On-line expectation–maximization algorithm for latent data models*, JRSS-B 71(3):593–613 ([arXiv:0712.4273](https://arxiv.org/abs/0712.4273)).
- **Bishop (2006)**: *Pattern Recognition and Machine Learning*, Chapter 9 (Mixture Models & EM singularities).
- **Video**: *Datamlistic — Gaussian Mixture Models & EM Explained* ([YouTube Search](https://www.youtube.com/results?search_query=Datamlistic+Gaussian+Mixture+Models+Expectation+Maximization)).
