# Research Design and Hypotheses (e-GMM)

**Working title:** *e-GMM: fitted step and batch laws, and component revival, for soft mini-batch EM*
**Track:** Tier 1, candidate 2 of the e-means novelty report (extending arXiv 2611.09501)
**Budget asked:** 200–300 GPU-hours on 1–2 × A100/H100 80 GB. This is an estimate: no GPU job has run yet.

> Revised 2026-11-25 after the [review](../docs/review/2026-11-25-REVIEW.md).
> - The earlier version stated barriers that the repository's own runs did not show.
> - It also had no numeric pass/fail criteria.
> - Its verdicts were hard-coded in the report compilers.
>
> This version fixes all three. Verdicts are now computed by the compilers using the rule in Section 3.

---

## 1. Subject

### What e-GMM is
- **Update rule.** A stochastic-approximation online EM on the sufficient statistics (N_j, S_j, Q_j), as in Cappé & Moulines (2009): Ŝ_t = (1 − α_t)·Ŝ_{t−1} + α_t·E[s(x) | x, θ_{t−1}].
- **Step schedule.** α_t is a cosine schedule from α₀ down to 0.1·α₀, as in e-means, instead of (t + t₀)^−β.
- **Covariance readout.** Σ_j = Q_j/N_j − μ_jμ_jᵀ + λ₀·I.
- **Revival.** Every 5 batches, components with N_j < τ·mean(N) are rebuilt by splitting a donor. The default split is moment-preserving, along the donor's principal eigenvector v₁.
- **Finalization.** Either the average of the last epoch's statistics, or a slow EMA of the statistics.

The update rule is not new. The e-means paper itself credits EMA-statistics EM to Nowlan (1991) and Neal & Hinton (1998). The contribution, if any, is empirical:
- **Step and batch laws.** Can α₀ and the batch-size knee be predicted from (k, B, n, T) for soft EM? Does adding d (ambient or intrinsic) help?
- **Revival.** Does revival improve held-out NLL against *tuned* baselines?

### Barriers, as far as the evidence supports them
1. **Full-batch EM costs O(nkd) per iteration** (O(nkd²) for full covariance), and needs many passes. This is standard, and it is the reason to use mini-batches.
2. **Online EM has no rule for choosing its step size at a fixed epoch budget.** Robbins–Monro conditions are asymptotic.
   - The re-check in the review shows the cost of a bad choice: at α₀ = 0.35, e-GMM without revival starves 13–15 of 16 components, but at α₀ = 0.10 it does not.
   - A fitted law for α₀ is therefore useful. Whether the law needs d is open: the e-means authors dropped a d exponent as unstable (their Appendix B.1).
3. **Starved components.** Mini-batch soft EM can leave components with negligible responsibility. However:
   - In the repository's runs, this happened at a large hand-set step size, not intrinsically.
   - Plain online EM did not collapse in the same setting.
   - For a GMM, a small mixture weight can be correct (a rare mode), so "dead" (weight below τ × mean) is a diagnostic, not an error.
4. **Singular covariances.** The maximum-likelihood GMM objective is unbounded when a component collapses onto a point. An additive floor λ₀·I prevents this by construction.
   - "Zero singularity failures" is therefore a property of the floor, not a finding.
   - No method in the PoCs recorded a singular event.
5. **Memory.** Full covariance at k = 65 536 and d = 1024 needs 256 GiB in FP32 for Q alone.
   - Diagonal costs O(kd) and low-rank + diagonal costs O(kdr). This arithmetic is correct.
   - The claimed GPU savings (a fused Triton E-step, no B × k materialization) are **not implemented**. The kernel is not wired in, and the dense path materializes B × k.

---

## 2. Hypotheses

Every hypothesis is judged by the rule in Section 3.
Every comparison is paired by seed: each method sees the same splits and the same budget.
"Tuned" means the step hyperparameters were chosen on a validation split, with the same grid size for every method (Section 4, control 2).

| ID | Question | Hypothesis, and how it is judged | Experiment |
| :---: | :--- | :--- | :--- |
| **H0** | Is e-GMM better than the standard alternatives once they are all tuned? | **H0 (the null to beat):** at matched epochs and batch size, tuned e-GMM's test NLL is *no better* than the best of tuned online EM (Cappé–Moulines), e-means followed by a few full-batch EM iterations, and full-batch EM from a k-means/e-means initialization. **e-GMM has a contribution only where H0 is rejected** (verdict *Supported* for "e-GMM < baseline"). | `exp_02` |
| **H1** | Does a d term improve the α₀ law? | **H1:** adding a_d·ln d (ambient d, and separately intrinsic d from TwoNN) lowers the **leave-one-d-out (LODO) test-NLL regret**: the NLL at the law-predicted α₀, minus the NLL at the best grid α₀, in nats/dim. Supported if the regret drops by at least 0.01 nats/dim with p < 0.05 across held-out configurations. **In-sample R² is reported but does not decide H1.** An exponent is reported only if its regressor varies in the sweep (T must take at least 2 values). | `exp_01` |
| **H2** | Does the fitted law transfer zero-shot? | **H2:** fitted on one data source, the law's predicted α₀ gives a held-out NLL regret of at most 0.01 nats/dim (median over configurations) on unseen (n, k, d) and on an unseen dataset. Batch-knee fits use only uncensored knees: a knee at the largest batch tried is excluded and counted. | `exp_01` |
| **H3** | Does e-GMM beat online EM and full-batch EM at a matched epoch budget? | **H3:** at T ∈ {1, 2, 3} epochs, tuned e-GMM has lower test NLL than tuned online EM and than full-batch EM given the same number of data passes. Judged per dataset. Wall-clock is compared **only on the same backend and device**. | `exp_02` |
| **H4** | When does joint soft e-GMM beat a hard e-means pipeline? | **H4:** on overlapping or anisotropic mixtures, e-GMM beats both e-means + post-hoc covariance and e-means → EM. On well-separated mixtures the difference is within the threshold. Both regimes must be reported. | `exp_02` |
| **H5** | Does spectral v₁ splitting beat other revival policies? | **H5:** at each method's *tuned* α₀, spectral splitting has lower test NLL than no revival, batch resampling, the Faiss ±δ split and a random orthogonal split. No revival at tuned α₀ is the key comparison: at α₀ = 0.10 the re-check found no significant difference. | `exp_03` |
| **H6** | Do low-rank + diagonal covariance and top-m responsibilities preserve quality? | **H6a:** `low_rank_diag` (r ∈ {8, 16}) beats `diag` on test NLL for correlated data, and beats `full` on BIC **computed on the training set**. **H6b:** top-m (m ∈ {8, 16}) stays within 0.01 nats/dim of dense responsibilities. | `exp_03`, `exp_04` |
| **H7** | Does the finalization method matter? | **H7:** the slow-EMA readout beats the last-epoch average on test NLL when n/k is small. The last-epoch average is **not** exact EM: its responsibilities come from parameters that move during the epoch. | `exp_03` |

---

## 3. Verdict rule

The rule is implemented in [`src/e_gmm/stats.py`](../src/e_gmm/stats.py) and used by both report compilers.

1. **Seeds:** at least 10. Smoke runs use fewer, and the compiler reports them as *Inconclusive (too few seeds)*.
2. **Test:** for the claim "A beats B" (lower NLL), compute diff = NLL_A − NLL_B on the test split for each seed, and run a two-sided **exact sign-flip permutation test** on the diffs.
3. **Practical threshold:** 0.01 nats per dimension. The per-point threshold is 0.01·d nats.
4. **Verdict:**
   - **Supported:** p < 0.05 and mean(diff) ≤ −threshold.
   - **Contradicted:** p < 0.05 and mean(diff) > 0.
   - **Inconclusive:** everything else, including significant differences smaller than the threshold.
5. **Multiple claims:** each report lists every claim it tested, not only the ones that passed.

---

## 4. Controls

The compilers check these where they can. A run that violates one is flagged in the report.

1. **One evaluator.** Every method outputs (π, μ, Σ), which are scored by the same `evaluate_gmm_nll` on the same deduplicated validation and test splits.
2. **Validation tuning for every method.** Each method's step hyperparameters are chosen on the validation split, with a grid of the same size per method:
   - α₀ for e-GMM and e-means;
   - (β, t₀) for online EM;
   - the iteration count for full-batch EM.
   No method uses a hand-set value that another method's grid would have beaten.
3. **Matched regularization and initialization.** Every method uses the same covariance floor λ₀ (additive) and the same initialization scale. Full-batch EM is also run from a k-means/e-means initialization, which is scikit-learn's default.
4. **Matched budget.** Every mini-batch method uses the same number of passes T·n and the same batch size B. The only exception is the batch-size sweep in Phase A.
5. **Strict data handling.**
   - A missing file is an error; there is no synthetic substitution.
   - There is no silent dimension truncation; PCA projection is explicit and recorded.
   - Rows are sampled at random, not taken from the start of the file.
   - Data is scale-normalized before fitting, so λ₀ means the same thing on every dataset.
6. **Diagnostics are not results.** Dead-component counts (weight below τ × mean) and the floor-dominated rate (raw variance ≤ λ₀) are reported to explain results. They are never used as quality metrics on their own.
7. **Information criteria.** BIC and AIC are computed on the training set.
8. **Sweep identifiability.**
   - Every regressor in a law fit must vary.
   - Censored knees are excluded.
   - The sweep data is a mixture, not a single Gaussian.
   - Initialization does not depend on B or k.
9. **Scope of claims.** A report states the largest n, d and k actually run, the backend and device, and whether the data was real or synthetic.
