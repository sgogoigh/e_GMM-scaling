# 06. Dimension-Aware Scaling Laws & Master A–Z Glossary

> **Goal**: Explain why `e-GMM` adds $d^{a_d}$ back into the `e-means` power law, define NLL regret and LOSO/LODO/OOD validation, and provide an A–Z reference for every symbol in `e_GMM-scaling`.

---

## 1. Why Dimension $d$ Drops Out in `e-means`—But Matters in `e-GMM`

In **`e-means`** (Paper Section 3.3 & Appendix B.1):
- The assignment objective is isotropic squared Euclidean distance $\|x_i - c_j\|_2^2$.
- When $x$ is normalized per coordinate, the concentration of pairwise distances is governed by **Local Intrinsic Dimensionality (LID)** (Amsaleg et al., 2015), not raw ambient $d$, so including $d$ hurt OOD generalization across datasets (`DEEP` $d=96$ vs. `SimpleWiki` $d=3072$).

In **`e-GMM`**, two mathematical terms introduce **explicit $O(d)$ degrees of freedom and variance scaling**:
1. **Covariance Estimation Degrees of Freedom**: Estimating $\Sigma_j$ (even diagonal, $d$ parameters per component; full, $d(d+1)/2$ parameters) requires effective batch sample support $N_j \propto d$ before ratio readout $\frac{Q_j}{N_j} - \mu_j\mu_j^\top$ stabilizes!
2. **Mahalanobis & Log-Determinant Terms in E-Step Logits**:
   $$
   \ln \mathcal{N}(x_i \mid \mu_j, \Sigma_j) = -\frac{1}{2}(x_i - \mu_j)^\top \Sigma_j^{-1}(x_i - \mu_j) - \frac{1}{2}\ln\det \Sigma_j - \frac{d}{2}\ln(2\pi)
   $$
   In higher $d$, small covariance estimation noise across $d$ coordinates compounds inside the softmax logits $r_{ij} = \text{softmax}_j(\ln \pi_j + \ln \mathcal{N})$, sharpening or destabilizing responsibilities unless $\alpha_0$ adapts to $d$!

Therefore, `e-GMM` fits **dimension-aware log-linear power laws**:

$$
\alpha_0 = \min\Big( C_\alpha \cdot k^{a_k} \cdot B^{a_B} \cdot n^{a_n} \cdot T^{a_T} \cdot d^{a_d},\; 1 \Big)
$$
$$
B_{\text{knee}} = \min\Big( C_B \cdot k^{b_k} \cdot n^{b_n} \cdot T^{b_T} \cdot d^{b_d},\; n \Big)
$$

Taking natural logarithms converts both into Ordinary Least Squares (OLS) multiple linear regression (`src/e_gmm/scaling_law.py`):
$$
\ln \alpha_0 = \ln C_\alpha + a_k \ln k + a_B \ln B + a_n \ln n + a_T \ln T + a_d \ln d
$$

### Expected Exponent Signs
- **$a_B > 0$**: Larger batches carry lower E-step variance and visit more components per step $\implies$ larger step $\alpha_0$.
- **$a_k < 0$, $a_n < 0$**: More clusters or larger datasets reduce per-component batch hit fraction / increase total updates $\implies$ gentler $\alpha_0$.
- **$a_d < 0$**: Higher dimension $d$ increases covariance estimation variance per component ($d$ or $d^2$ parameters) $\implies$ requires longer EMA smoothing window (smaller $\alpha_0$) to avoid variance collapse!

---

## 2. Regret Metric & Validation Protocols (LOSO, LODO, OOD)

### Held-Out NLL Regret
Unlike MSE (always $> 0$), Negative Log-Likelihood ($\text{NLL} = -\frac{1}{n_{\text{val}}}\sum_i \ln p(x_i)$) can be **negative** for concentrated densities! To measure scale-invariant relative sub-optimality cleanly, `e_GMM-scaling` tracks both:
1. **Bits-per-dimension / Nats gap**: $\Delta \text{NLL} = \text{NLL}_{\text{law}} - \text{NLL}_*$ (or normalized per dimension $\Delta \text{NLL} / d$).
2. **Relative Shift-Invariant NLL Regret** (or normalized against initial/single-Gaussian NLL $\text{NLL}_{\text{ref}}$):
   $$
   \text{Regret}_{\text{NLL}} = \frac{\text{NLL}_{\text{law}} - \text{NLL}_*}{\max(|\text{NLL}_{\text{ref}} - \text{NLL}_*|, 10^{-8})}
   $$
   where $\text{NLL}_*$ is the oracle grid-search minimum NLL for $(n, k, d, B, T)$.

### Three Extrapolation Tests
1. **LOSO ($n$-extrapolation)**: Leave-One-Size-Out across $n$ (train on smaller $n$, predict largest $n$).
2. **LODO ($d$-extrapolation)**: Leave-One-Dimension-Out across $d$ (train on e.g. $d \in \{16, 32, 64\}$ or $\{32, 96, 256\}$, predict $d=128$ or $768$).
3. **OOD (Dataset transfer)**: Fit on `DEEP` / `StretchedGaussian`, predict zero-shot on `SIFT`, `SPACEV`, `ImageNet-1k`, `Zipf-GMM`.

---

## 3. Complete A–Z Glossary & Symbol Table

| Symbol / Term | Meaning & Role in `e_GMM-scaling` |
| :--- | :--- |
| **$\alpha_0, \alpha(t), \varepsilon$** | Peak step size, step-$t$ cosine-annealed step size, and cosine floor ratio ($\varepsilon = 0.1$). |
| **$\gamma_t$** | EMA decay factor ($\gamma_t = 1 - \alpha(t)$); Robbins-Monro $(t+t_0)^{-\beta}$ in Cappé & Moulines baseline. |
| **$(N_j, S_j, Q_j)$** | 3-tuple EMA sufficient statistics: effective count, weighted vector sum, weighted 2nd moment. |
| **$\pi_j, \mu_j, \Sigma_j$** | Mixture weight ($N_j/\sum N_\ell$), mean ($S_j/N_j$), and regularized covariance ($Q_j/N_j - \mu_j\mu_j^\top + \lambda_0 I_d$). |
| **$\lambda_0$ (`cov_floor`)** | Positive-definite covariance floor preventing $\det(\Sigma_j) \to 0$ singularities. |
| **$r_{ij}$** | Soft E-step posterior responsibility $P(z_i = j \mid x_i; \theta)$. |
| **$\delta$ (`split_delta`)** | Spectral displacement scale in $\mu_{\text{child}\pm} = \mu_{j^*} \pm \delta \sqrt{\lambda_{\max}}\, \mathbf{v}_1$. |
| **$\mathbf{v}_1, \lambda_{\max}$** | Principal eigenvector and largest eigenvalue of donor covariance $\Sigma_{j^*}$. |
| **$\tau$ (`split_tau`)** | Dead-component threshold fraction (default `0.1`): triggers split when $N_j < \tau \bar{N}$. |
| **BIC / AIC** | Bayesian / Akaike Information Criterion penalizing log-likelihood by parameter count $p_{\text{eff}}$. |
| **Cholesky ($L_j L_j^\top$)** | Lower-triangular factorization of $\Sigma_j \succ 0$ for stable inverse quadratic forms & $\ln\det\Sigma_j$. |
| **Flash-GMM** | Tiled $2d$-augmented GEMM reformulation of diagonal GMM E-step + streaming `logsumexp`. |
| **LODO / LOSO / OOD** | Leave-One-Dimension-Out / Leave-One-Size-Out / Out-of-Distribution scaling-law validations. |
| **Low-Rank + Diag** | $\Sigma_j = D_j + U_j U_j^\top$ ($U_j \in \mathbb{R}^{d \times r}$) breaking the $O(kd^2)$ memory wall via Woodbury. |
| **NLL** | Negative Log-Likelihood $-\frac{1}{n}\sum_i \ln \sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)$. |
| **Robbins-Monro** | Classical stochastic EM step schedule $\gamma_t = (t + t_0)^{-\beta}$, $\beta \in (0.5, 1]$. |
| **Slow-EMA** | Passive shadow 3-tuple $(N^{\text{slow}}, S^{\text{slow}}, Q^{\text{slow}})$ optimizing held-out Test NLL. |
| **Top-$m$ Truncation** | Keeping only the top-$m$ largest responsibilities per point ($m \in \{4, 8, 16\}$) for $O(Bmd)$ M-step. |
| **Woodbury Identity** | Expressing $(D_j + U_j U_j^\top)^{-1}$ using an $r \times r$ capacitance solve instead of $d \times d$. |
| **Zipf Mixture** | Heavily power-law imbalanced GMM ($\pi_g \propto 1/g$) stress-testing dead-component revival. |
