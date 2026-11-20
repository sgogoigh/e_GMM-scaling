# 01. `e-means` Primitives & 3-Tuple Sufficient Statistics $(N_j, S_j, Q_j)$

> **Origin**: Extending Bessa et al. (Meta FAIR, Nov 2026), *"e-means: outperforming k-means (in one-tenth the epochs)"* ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)) from hard isotropic clustering to soft **Gaussian Mixture Models (`e-GMM`)**.

---

## 1. From 2-Tuple Hard `e-means` to 3-Tuple Soft `e-GMM`

In `k-means` / `e-means`, every cluster is modeled as an isotropic ball with fixed infinitesimal variance $\sigma^2 \to 0$. That requires tracking only **two** exponentially decayed sufficient statistics per cluster $j \in \{1, \dots, k\}$:
1. **0th-order moment (effective count)**: $N_j \in \mathbb{R}_+$
2. **1st-order moment (weighted vector sum)**: $S_j \in \mathbb{R}^d$

In a **Gaussian Mixture Model (GMM)**, $p(x) = \sum_{j=1}^{k} \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j)$, each component carries a mixture weight $\pi_j$, a mean $\mu_j \in \mathbb{R}^d$, and a **covariance** $\Sigma_j \in \mathbb{S}_{++}^d$. Under soft E-step responsibilities $r_{ij} = P(z_i = j \mid x_i; \theta)$, the complete-data exponential family sufficient statistics expand from a **2-tuple** to a **3-tuple $(N_j, S_j, Q_j)$**:

| Concept | In Hard `e-means` (`k-means`) | In Soft `e-GMM` (Gaussian Mixtures) |
| :--- | :--- | :--- |
| **Assignment** | Hard Voronoi $a(x_i) = \arg\min_j \|x_i - c_j\|_2^2$ | Soft log-space posterior $r_{ij} = \text{softmax}_j(\ln \pi_j + \ln \mathcal{N}(x_i \mid \mu_j, \Sigma_j))$ |
| **Tracking Tuple** | $(N_j, S_j)$ (count & vector sum) | **$(N_j, S_j, Q_j)$** (count, sum, uncentered 2nd moment) |
| **Ratio Readout** | $c_j = S_j / N_j$ | $\pi_j = \frac{N_j}{\sum_\ell N_\ell}$, $\mu_j = \frac{S_j}{N_j}$, $\Sigma_j = \frac{Q_j}{N_j} - \mu_j \mu_j^\top + \lambda_0 I_d$ |
| **Step Law** | $\alpha_0 = \min(0.824 k^{-0.46} B^{0.90} n^{-0.45} T^{-0.11}, 1)$ | **Dimension-aware**: $\alpha_0 = \min(C_\alpha k^{a_k} B^{a_B} n^{a_n} T^{a_T} d^{a_d}, 1)$ |
| **Dead Revival** | Hyperplane sign flip $\odot [1 \pm \delta, \dots]$ | **Spectral split** along top eigenvector $\mathbf{v}_1$ of $\Sigma_{j^*}$ |

---

## 2. Exact 3-Tuple EMA Equations & Initialization

### Initialization (Zero-Inertia Exact Prior)
Choose $k$ initial means $\mu_j^{(0)} \sim \text{RandomChoice}(X, k)$ (or `k-means++` / `e-means` warm start), initial isotropic variance $\sigma_0^2 > 0$, and tiny prior mass $N_j^{(0)} = 10^{-10}$:
$$
N_j^{(0)} = 10^{-10}, \qquad S_j^{(0)} = N_j^{(0)} \mu_j^{(0)}, \qquad Q_j^{(0)} = \begin{cases} N_j^{(0)} \big( (\mu_j^{(0)})^{\odot 2} + \sigma_0^2 \mathbf{1}_d \big) & \text{(diagonal)} \\ N_j^{(0)} \big( \mu_j^{(0)} (\mu_j^{(0)})^\top + \sigma_0^2 I_d \big) & \text{(full)} \end{cases}
$$
**Why $10^{-10}$?** Evaluating the ratio readout at step $0$ yields $\mu_j = \mu_j^{(0)}$ and $\Sigma_j = (\sigma_0^2 + \lambda_0) I_d$ **exactly**, while carrying essentially zero inertia once real mini-batches arrive!

### Mini-Batch EMA Update (With Cosine Decay $\gamma_t = 1 - \alpha(t)$)
On mini-batch $\mathcal{B}$ of size $B$, compute batch sufficient statistics in **FP32** (or FP64 for verification):
$$
n_{b,j} = \sum_{x_i \in \mathcal{B}} r_{ij}, \qquad s_{b,j} = \sum_{x_i \in \mathcal{B}} r_{ij} x_i, \qquad q_{b,j} = \begin{cases} \sum_{x_i \in \mathcal{B}} r_{ij} (x_i^{\odot 2}) & \text{(diagonal)} \\ \sum_{x_i \in \mathcal{B}} r_{ij} (x_i x_i^\top) & \text{(full)} \end{cases}
$$
Then apply the EMA recurrence with $\alpha_t = \alpha(t) = 1 - \gamma_t$:
$$
N_j \leftarrow \gamma_t N_j + (1 - \gamma_t) n_{b,j}
$$
$$
S_j \leftarrow \gamma_t S_j + (1 - \gamma_t) s_{b,j}
$$
$$
Q_j \leftarrow \gamma_t Q_j + (1 - \gamma_t) q_{b,j}
$$

---

## 3. Why $(N_j, S_j, Q_j)$ Guarantees Positive-Definiteness & Convex Hull Stability

At any step $t$, unrolling the EMA recurrence shows that $(N_j, S_j, Q_j)$ is a **strictly non-negative weighted sum** of historical data moments plus the positive-definite prior $Q_j^{(0)}$:
$$
N_j^{(t)} = w_0 + \sum_{u=1}^{t} \sum_{x_i \in \mathcal{B}_u} w_{u,i,j}, \qquad w_{u,i,j} = \alpha_u \left(\prod_{v=u+1}^{t} \gamma_v\right) r_{ij}^{(u)} \ge 0
$$
Dividing by $N_j^{(t)}$ defines a valid **convex probability measure** over all seen vectors. Therefore:
1. **Mean Convex Hull Property**: $\mu_j = S_j / N_j$ is strictly a convex combination of $\mu_j^{(0)}$ and seen vectors—it **cannot overshoot** the data convex hull (even at $\alpha_t = 1$).
2. **Covariance PSD Property**: By Jensen's inequality / variance identity:
   $$
   \frac{Q_j}{N_j} - \mu_j \mu_j^\top = \sum_{\text{history}} \tilde{w}_i (x_i - \mu_j)(x_i - \mu_j)^\top + \frac{w_0}{N_j}\big(\sigma_0^2 I_d + (\mu_j^{(0)} - \mu_j)(\mu_j^{(0)} - \mu_j)^\top\big) \succeq 0
   $$
3. **Strict Positive-Definiteness via Floor $\lambda_0 I_d$**:
   $$
   \Sigma_j = \frac{Q_j}{N_j} - \mu_j \mu_j^\top + \lambda_0 I_d \succeq \lambda_0 I_d \succ 0
   $$
   Every eigenvalue of $\Sigma_j$ is strictly bounded below by $\lambda_0 > 0$, making $\Sigma_j$ **unconditionally invertible** with bounded log-determinant $\ln\det\Sigma_j \ge d \ln \lambda_0$!

---

## 4. Numerical Precision Trap: Catastrophic Cancellation in $Q_j/N_j - \mu_j\mu_j^\top$

In exact real arithmetic, $\mathbb{E}[x x^\top] - \mathbb{E}[x]\mathbb{E}[x]^\top$ is always PSD. In **BF16 or FP16**, however, if vectors have large mean offsets (e.g., $\|\mu_j\| \gg \sigma_j$), subtracting two nearly equal large numbers $Q_j/N_j$ and $\mu_j \mu_j^\top$ suffers **catastrophic cancellation**, yielding tiny negative variances or non-PSD matrices!

**Mandatory Precision Split Rule** (implemented throughout `src/e_gmm/`):
1. Store and update $(N_j, S_j, Q_j)$ **strictly in FP32** (or FP64 for small-$k$ verification).
2. Enforce `.clamp_min(lambda_0)` (diagonal) or `+ lambda_0 * I_d` (full covariance) **after** subtracting $\mu_j \mu_j^\top$ and symmetrizing $\frac{1}{2}(\Sigma_j + \Sigma_j^\top)$.
3. Optionally center data once by global batch mean $\bar{x}_0$ (which shifts $\mu_j$ without changing covariances or Gaussian log-likelihoods!).

---

## 5. Sources & References
- **Bessa et al. (2026)**: *e-means: outperforming k-means (in one-tenth the epochs)*, `arXiv:2611.09501v1` (Sections 2–3, Appendix E).
- **Neal & Hinton (1998)**: *A view of the EM algorithm that justifies incremental, sparse, and other variants*.
- **Cappé & Moulines (2009)**: *On-line expectation–maximization algorithm for latent data models*, JRSS-B 71(3):593–613.
- **Nowlan (1991)**: *Soft Competitive Adaptation*, PhD thesis, CMU.
