# 02. Online EM (Cappé & Moulines, 2009) and where it struggles

This primer covers classical full-batch EM and the Cappé & Moulines online EM. **e-GMM uses the same update rule as Cappé–Moulines**; what it changes is the step schedule and a few add-ons. This primer explains which problems those changes target, and how strong the evidence is for each problem.

> Revised 2026-11-25 after the [review](../review/2026-11-25-REVIEW.md). The earlier version overstated several of these problems as established.

---

## 1. Full-batch EM

Given data X = {x₁, …, x_n} ⊂ ℝ^d, full-batch EM processes **every point** in every iteration.
- **E-step.** Cost is O(nkd) for diagonal covariances and O(nkd²) for full ones:
  $$
  r_{ij} = \frac{\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{\ell=1}^{k} \pi_\ell \, \mathcal{N}(x_i \mid \mu_\ell, \Sigma_\ell)}
  $$
- **M-step.**
  $$
  N_j = \sum_{i} r_{ij}, \quad \pi_j = \frac{N_j}{n}, \quad \mu_j = \frac{1}{N_j}\sum_{i} r_{ij} x_i, \quad \Sigma_j = \frac{1}{N_j}\sum_{i} r_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top
  $$

**Cost at scale.** At n = 10⁸ and k = 65 536, one E-step touches 6.6 × 10¹² point–component pairs, and EM usually needs tens of iterations. scikit-learn's `GaussianMixture` runs on a single CPU node and keeps an n × k responsibility matrix in memory. As far as we know, **FAISS has no GMM trainer**; it is a k-means and nearest-neighbour library.

---

## 2. Online EM (Cappé & Moulines, 2009)

Online EM is a stochastic approximation on the expected sufficient statistics Ŝ of an exponential-family model:

$$
\hat{S}_t = (1 - \gamma_t)\, \hat{S}_{t-1} + \gamma_t \, \mathbb{E}_{\hat{\theta}_{t-1}}\big[ s(x_t, z_t) \mid x_t \big], \qquad \hat{\theta}_t = \bar{\theta}(\hat{S}_t)
$$

It typically uses the step size γ_t = (t + t₀)^−β with β ∈ (0.5, 1]. This satisfies Σγ_t = ∞ and Σγ_t² < ∞.

**e-GMM is this update** with Ŝ = (N, S, Q), mini-batch expectations, and γ_t replaced by e-means' cosine schedule α(t), which runs from α₀ down to 0.1·α₀. Related earlier work:
- Nowlan (1991): decayed statistics.
- Neal & Hinton (1998): incremental and sparse EM.
- Sato & Ishii (2000): discounted online EM.
- Liang & Klein (2009): stepwise EM in practice.

A full list is in [the review, Section 7](../review/2026-11-25-REVIEW.md#7-prior-art-to-position-against-from-memory--verify-before-citing).

---

## 3. Practical problems, and the evidence for each

### 3.1 Step-size choice at a fixed epoch budget (well supported)
Robbins–Monro conditions guarantee convergence only as t → ∞. They do not say how to choose (β, t₀) or B for a budget of T ≈ 1–5 epochs.
- If the step decays too fast, components stay near their initialization.
- If it decays too slowly, late-epoch noise never settles.

The same problem applies to e-GMM's α₀. In the review's 10-seed re-check (d = 16, k = 16, 6 epochs):
- at α₀ = 0.35, e-GMM without revival starved 13–15 of 16 components and lost 2.2–3.0 nats;
- at α₀ = 0.10 it did not.

This is the motivation for a fitted α₀ law. Whether the law needs a d term is open: the e-means authors tried one for k-means and dropped it as unstable (their Appendix B.1).

### 3.2 Singular covariances (standard; prevented by the floor)
The maximum-likelihood GMM objective is unbounded: a component that collapses onto a single point drives the likelihood to +∞ (Bishop, 2006, Chapter 9). With fewer than d effective points, a component's sample covariance is rank-deficient.
- **The usual fix is the one e-GMM uses:** add λ₀·I to every covariance. This is a MAP-like regularizer and is the same as scikit-learn's `reg_covar`.
- With the floor added, no covariance can become singular, so **"zero singularity failures" is not evidence** that e-GMM is better.
- No method in the repository's runs recorded a singular event, including online EM with a floor of 10⁻⁸.
- The floor is absolute, so it only means something on normalized data.

### 3.3 Starved components (depends on the setting)
A component whose log-joint stays far below its neighbours' receives almost no responsibility, and so its statistics stop updating. Classical online EM has no revival step.
- **In the repository's runs, starvation depended on the step size** (Section 3.1), not on soft EM as such. Plain online EM did not collapse in the same setting.
- **"Dead" means mixture weight below 0.1 × the mean.** That is a k-means notion; a GMM with a rare mode *should* have a small weight there. In one smoke benchmark, online EM had 10 "dead" components and still a better test NLL than e-GMM. Revival can remove legitimate rare modes.
- The earlier claim that "15–40 % of capacity is wasted on Zipf mixtures" had no source and has been removed.
- Whether revival (spectral or otherwise) helps against tuned baselines is hypothesis H5. It is not established.

### 3.4 Memory for full covariances (correct arithmetic)
At k = 65 536 and d = 1024, Q alone is 65 536 × 1024² × 4 bytes = **256 GiB (274.9 GB)** in FP32, more than an 80 GB GPU.
- Diagonal covariances cost O(kd); low-rank + diagonal costs O(kdr).
- The E-step's B × k responsibility matrix is a second memory cost. The repository's dense path materializes it, and the fused Triton kernel that would avoid this is not wired in.

---

## 4. What e-GMM changes, and the status of each change

| Problem | e-GMM's change | Status (2026-11-25) |
| :--- | :--- | :--- |
| Step-size choice | Cosine α(t), and a fitted law for α₀(k, B, n, T[, d]) | Law not yet fitted on identifiable data. The defaults use the e-means k-means law with d exponent 0. |
| Singular covariances | Additive floor λ₀·I | Works by construction. This is standard practice, not a contribution. |
| Starved components | Revival every 5 batches; moment-preserving split along v₁ | The split preserves moments exactly. Its benefit at a tuned α₀ is unproven (H5). |
| Memory | Diagonal ([x², x] GEMM); low-rank + diagonal (Woodbury) | The float64 identities are verified. The GPU kernel is not wired in, the torch low-rank and full paths loop over k in Python, and the torch path has not run. |

---

## 5. Sources
- **Cappé & Moulines (2009).** *On-line expectation–maximization algorithm for latent data models.* JRSS-B 71(3):593–613. [arXiv:0712.4273](https://arxiv.org/abs/0712.4273).
- **Neal & Hinton (1998).** *A view of the EM algorithm that justifies incremental, sparse, and other variants.*
- **Bishop (2006).** *Pattern Recognition and Machine Learning*, Chapter 9.
- **Liang & Klein (2009).** *Online EM for unsupervised models.* NAACL.
- For the rest, see [the review's prior-art list](../review/2026-11-25-REVIEW.md#7-prior-art-to-position-against-from-memory--verify-before-citing) (compiled from memory; verify before citing).
