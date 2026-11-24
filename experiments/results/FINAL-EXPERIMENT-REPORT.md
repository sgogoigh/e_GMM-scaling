# Experiment report (e_GMM-scaling)

Generated 2026-11-25 01:10:57 by experiments/compile_experiment_report.py from experiments/results/*.json.

Verdict rule (experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md §3): paired by seed, exact sign-flip test, p < 0.05, practical threshold 0.01 nats/dim, at least 10 seeds. Smoke runs are small CPU runs on synthetic data unless stated otherwise; they are evidence about the method at that scale only.

## Verdicts

Totals: {'Contradicted': 6, 'Inconclusive': 25, 'H0 not rejected': 4, 'Supported': 13}

| ID | Claim | Verdict |
|---|---|---|
| H1 | ambient_d lowers LODO regret by >= 0.01 nats/dim | Contradicted (preliminary: 2 seeds per curve, design asks >= 10) |
| H2 | law transfers with median regret <= 0.01 nats/dim | Inconclusive (only 2 OOD configurations; median regret 0.01087) |
| H0 | [Zipf-Separated-Diag] e-GMM beats the best tuned standard baseline (emeans_then_em3) | H0 not rejected: Inconclusive |
| H3 | [Zipf-Separated-Diag] e-GMM beats online_em at epochs=5 | Inconclusive |
| H3 | [Zipf-Separated-Diag] e-GMM beats fullbatch_em_emeans at epochs=5 | Inconclusive |
| H3 | [Zipf-Separated-Diag] e-GMM beats fullbatch_em_random at epochs=5 | Supported |
| H4 | [Zipf-Separated-Diag] e-GMM beats emeans_posthoc | Supported |
| H4 | [Zipf-Separated-Diag] e-GMM beats emeans_then_em3 | Inconclusive |
| H0/H5 | [Zipf-Separated-Diag] spectral revival beats no revival (both alpha_0 tuned) | Inconclusive |
| H0 | [Zipf-Rotated-Full] e-GMM beats the best tuned standard baseline (emeans_then_em3) | H0 not rejected: Inconclusive |
| H3 | [Zipf-Rotated-Full] e-GMM beats online_em at epochs=6 | Inconclusive |
| H3 | [Zipf-Rotated-Full] e-GMM beats fullbatch_em_emeans at epochs=6 | Inconclusive |
| H3 | [Zipf-Rotated-Full] e-GMM beats fullbatch_em_random at epochs=6 | Supported |
| H4 | [Zipf-Rotated-Full] e-GMM beats emeans_posthoc | Inconclusive |
| H4 | [Zipf-Rotated-Full] e-GMM beats emeans_then_em3 | Inconclusive |
| H0/H5 | [Zipf-Rotated-Full] spectral revival beats no revival (both alpha_0 tuned) | Inconclusive |
| H0 | [Overlapping-Rotated-Full] e-GMM beats the best tuned standard baseline (fullbatch_em_emeans) | H0 not rejected: Inconclusive |
| H3 | [Overlapping-Rotated-Full] e-GMM beats online_em at epochs=6 | Supported |
| H3 | [Overlapping-Rotated-Full] e-GMM beats fullbatch_em_emeans at epochs=6 | Inconclusive |
| H3 | [Overlapping-Rotated-Full] e-GMM beats fullbatch_em_random at epochs=6 | Supported |
| H4 | [Overlapping-Rotated-Full] e-GMM beats emeans_posthoc | Supported |
| H4 | [Overlapping-Rotated-Full] e-GMM beats emeans_then_em3 | Supported |
| H0/H5 | [Overlapping-Rotated-Full] spectral revival beats no revival (both alpha_0 tuned) | Supported |
| H0 | [Stretched-PowerLaw-Diag] e-GMM beats the best tuned standard baseline (fullbatch_em_emeans) | H0 not rejected: Contradicted |
| H3 | [Stretched-PowerLaw-Diag] e-GMM beats online_em at epochs=5 | Inconclusive (significant but below practical threshold) |
| H3 | [Stretched-PowerLaw-Diag] e-GMM beats fullbatch_em_emeans at epochs=5 | Contradicted |
| H3 | [Stretched-PowerLaw-Diag] e-GMM beats fullbatch_em_random at epochs=5 | Inconclusive |
| H4 | [Stretched-PowerLaw-Diag] e-GMM beats emeans_posthoc | Inconclusive |
| H4 | [Stretched-PowerLaw-Diag] e-GMM beats emeans_then_em3 | Contradicted |
| H0/H5 | [Stretched-PowerLaw-Diag] spectral revival beats no revival (both alpha_0 tuned) | Inconclusive |
| H0/H5 | spectral_v1/variance_mass beats none/count (tuned alpha_0) | Inconclusive |
| H5 | spectral_v1/variance_mass beats batch_resample/count (tuned alpha_0) | Inconclusive |
| H5 | spectral_v1/variance_mass beats faiss_alt/count (tuned alpha_0) | Inconclusive |
| H5 | spectral_v1/variance_mass beats random_orth/count (tuned alpha_0) | Inconclusive (significant but below practical threshold) |
| H5 | spectral_v1/variance_mass beats spectral_v1/count (tuned alpha_0) | Inconclusive |
| H5 (context) | spectral_v1/variance_mass beats no revival at the old hand-set alpha_0 = 0.35 | Supported |
| H6a | low_rank_r2 beats diag on test NLL | Supported |
| H6a | low_rank_r2 has lower training-set BIC than full (diff +8778.2) | Contradicted |
| H6a | low_rank_r4 beats diag on test NLL | Inconclusive |
| H6a | low_rank_r4 has lower training-set BIC than full (diff +14093.2) | Contradicted |
| H6a | low_rank_r8 beats diag on test NLL | Supported |
| H6a | low_rank_r8 has lower training-set BIC than full (diff +15546.5) | Contradicted |
| H6b | m2 within ±0.01 nats/dim of dense (95% CI -0.0174..+0.0238) | Inconclusive |
| H6b | m4 within ±0.01 nats/dim of dense (95% CI -0.0074..+0.0319) | Inconclusive |
| H6b | m8 within ±0.01 nats/dim of dense (95% CI -0.0148..+0.0113) | Inconclusive |
| H6b | m16 within ±0.01 nats/dim of dense (95% CI -0.0099..+0.0039) | Supported |
| H7 | [small_n_over_k] slow-EMA beats last-epoch on test NLL | Supported |
| H7 | [medium_n_over_k] slow-EMA beats last-epoch on test NLL | Inconclusive |

## Exp 00: environment

- CPU cores: 16; torch installed: False; CUDA: False; Triton: False; scikit-learn: False
- Data directory: None; datasets audited: 0

Every torch/Triton/GPU claim in this repository is untested on this machine.

## Exp 01: scaling-law sweep (H1, H2)

- Data: synthetic: 32-cluster anisotropic rotated Zipf mixture, scale-normalized; backend numpy; mode smoke; 2 seeds averaged per curve point.
- Axes: d=[8, 16, 32, 64], k=[4, 8, 16], n=[1200, 4800], B=[32, 128], T=[2, 4].
- Censored alpha_0 optima (excluded from fits): 17/96; censored knees: 0/48.

| Law | c | a_k | a_B | a_n | a_T | a_d | unidentified | R² (in-sample) | LODO regret (nats/dim) | LOSO regret (nats/dim) |
|---|---|---|---|---|---|---|---|---|---|---|
| no_d | 1.4767 | -0.694 | 0.381 | -0.198 | -0.111 | - | - | 0.427 | 0.00343 | 0.00399 |
| ambient_d | 1.6499 | -0.697 | 0.386 | -0.198 | -0.109 | -0.038 | - | 0.429 | 0.00377 | 0.00417 |
| intrinsic_d | 1.7570 | -0.698 | 0.387 | -0.194 | -0.109 | -0.089 | - | 0.430 | 0.00382 | 0.00417 |

R² is shown for reference only; H1 and H2 are decided by held-out NLL regret.

**H1 (LODO, ambient_d vs no d):** mean regret change +0.00034 nats/dim over 79 held-out configurations, p = 5.0e-05, d-law better on 25. Verdict: **Contradicted** (preliminary: 2 seeds per curve, design asks >= 10).

_Caveat: the p-value treats held-out configurations as independent. They share fitted laws, folds and data draws, so it is optimistic; read the sign and size of the mean change, not the p-value alone._

| OOD config | alpha* | censored | regret paper_prior_no_d | regret no_d | regret ambient_d | regret intrinsic_d |
|---|---|---|---|---|---|---|
| {'d': 128, 'k': 8, 'n': 4800, 'B': 64, 'T': 3} | 0.5446 | False | 0.01152 | 0.01015 | 0.01129 | 0.01121 |
| {'d': 24, 'k': 12, 'n': 2400, 'B': 64, 'T': 3} | 0.1443 | False | 0.01522 | 0.01044 | 0.01044 | 0.01026 |

**H2 (OOD transfer, ambient_d law):** median regret 0.01087 nats/dim over 2 configurations. Verdict: **Inconclusive (only 2 OOD configurations; median regret 0.01087)**. No unseen *dataset* was tested in this run.

Batch-knee laws (uncensored knees only, alpha from the ambient_d law):

- no_d: c_B=5.8141, b_k=0.219, b_n=0.146, b_T=0.375, b_d=-, unidentified=-, n_used=48
- ambient_d: c_B=6.9743, b_k=0.219, b_n=0.146, b_T=0.375, b_d=-0.058, unidentified=-, n_used=48
- intrinsic_d: c_B=6.7788, b_k=0.219, b_n=0.150, b_T=0.375, b_d=-0.074, unidentified=-, n_used=48

## Exp 02: tuned, matched, paired benchmark (H0, H3, H4)

Backend numpy (tested locally: True); data synthetic; seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. diff = e-GMM − baseline in nats/dim; negative means e-GMM is better.

### Zipf-Separated-Diag

cov=diag, n_train data from synthetic, d=16, k=16, B=64, epochs=5. Skipped: {'sklearn_gmm': 'scikit-learn is not installed'}.

| Method | test NLL/d | dead | tuned setting | e-GMM − method | p | e-GMM wins | verdict: e-GMM better |
|---|---|---|---|---|---|---|---|
| **egmm_spectral** (reference) | 0.9875 | 0.0 | {'alpha_0': 0.1} | | | | |
| egmm_no_revive | 0.9952 | 2.0 | {'alpha_0': 0.1} | -0.00778 | 0.1797 | 5/10 | Inconclusive |
| egmm_faiss_alt | 0.9857 | 0.0 | {'alpha_0': 0.1} | +0.00174 | 0.2051 | 1/10 | Inconclusive |
| online_em | 0.9938 | 2.5 | {'beta': 0.65, 't0': 10.0} | -0.00634 | 0.2148 | 5/10 | Inconclusive |
| emeans_posthoc | 1.0020 | 0.0 | {'alpha_0': 0.2} | -0.01458 | 0.0195 | 9/10 | Supported |
| emeans_then_em3 | 0.9935 | 0.0 | {'alpha_0': 0.1, 'em_iters': 3} | -0.00602 | 0.2539 | 6/10 | Inconclusive |
| fullbatch_em_random | 1.0132 | 0.3 | {'max_iters': 25} | -0.02570 | 0.0020 | 10/10 | Supported |
| fullbatch_em_emeans | 0.9959 | 0.0 | {'max_iters': 25} | -0.00848 | 0.1113 | 6/10 | Inconclusive |

### Zipf-Rotated-Full

cov=full, n_train data from synthetic, d=16, k=16, B=64, epochs=6. Skipped: {'sklearn_gmm': 'scikit-learn is not installed'}.

| Method | test NLL/d | dead | tuned setting | e-GMM − method | p | e-GMM wins | verdict: e-GMM better |
|---|---|---|---|---|---|---|---|
| **egmm_spectral** (reference) | 1.0145 | 0.0 | {'alpha_0': 0.1} | | | | |
| egmm_no_revive | 1.0163 | 3.1 | {'alpha_0': 0.1} | -0.00184 | 0.6621 | 4/10 | Inconclusive |
| egmm_faiss_alt | 1.0132 | 0.0 | {'alpha_0': 0.05} | +0.00126 | 0.8262 | 5/10 | Inconclusive |
| online_em | 1.0107 | 2.4 | {'beta': 0.85, 't0': 10.0} | +0.00379 | 0.5723 | 5/10 | Inconclusive |
| emeans_posthoc | 1.0145 | 0.0 | {'alpha_0': 0.1} | -0.00008 | 0.9922 | 6/10 | Inconclusive |
| emeans_then_em3 | 1.0096 | 0.0 | {'alpha_0': 0.1, 'em_iters': 3} | +0.00491 | 0.1211 | 4/10 | Inconclusive |
| fullbatch_em_random | 1.0259 | 0.1 | {'max_iters': 25} | -0.01145 | 0.0293 | 8/10 | Supported |
| fullbatch_em_emeans | 1.0108 | 0.0 | {'max_iters': 25} | +0.00363 | 0.4746 | 5/10 | Inconclusive |

### Overlapping-Rotated-Full

cov=full, n_train data from synthetic, d=16, k=16, B=64, epochs=6. Skipped: {'sklearn_gmm': 'scikit-learn is not installed'}.

| Method | test NLL/d | dead | tuned setting | e-GMM − method | p | e-GMM wins | verdict: e-GMM better |
|---|---|---|---|---|---|---|---|
| **egmm_spectral** (reference) | 1.1945 | 0.0 | {'alpha_0': 0.1} | | | | |
| egmm_no_revive | 1.2071 | 2.9 | {'alpha_0': 0.1} | -0.01256 | 0.0020 | 10/10 | Supported |
| egmm_faiss_alt | 1.1956 | 0.0 | {'alpha_0': 0.2} | -0.00102 | 0.7949 | 5/10 | Inconclusive |
| online_em | 1.2111 | 2.0 | {'beta': 0.65, 't0': 10.0} | -0.01652 | 0.0137 | 8/10 | Supported |
| emeans_posthoc | 1.2907 | 0.0 | {'alpha_0': 0.2} | -0.09611 | 0.0020 | 10/10 | Supported |
| emeans_then_em3 | 1.2472 | 0.0 | {'alpha_0': 0.35, 'em_iters': 3} | -0.05264 | 0.0020 | 10/10 | Supported |
| fullbatch_em_random | 1.2116 | 0.9 | {'max_iters': 25} | -0.01706 | 0.0039 | 9/10 | Supported |
| fullbatch_em_emeans | 1.1903 | 0.0 | {'max_iters': 25} | +0.00421 | 0.3711 | 4/10 | Inconclusive |

### Stretched-PowerLaw-Diag

cov=diag, n_train data from synthetic, d=24, k=16, B=64, epochs=5. Skipped: {'sklearn_gmm': 'scikit-learn is not installed'}.

| Method | test NLL/d | dead | tuned setting | e-GMM − method | p | e-GMM wins | verdict: e-GMM better |
|---|---|---|---|---|---|---|---|
| **egmm_spectral** (reference) | 1.0961 | 0.0 | {'alpha_0': 0.1} | | | | |
| egmm_no_revive | 1.0983 | 0.3 | {'alpha_0': 0.1} | -0.00220 | 0.0703 | 7/10 | Inconclusive |
| egmm_faiss_alt | 1.0965 | 0.0 | {'alpha_0': 0.1} | -0.00042 | 0.3867 | 5/10 | Inconclusive |
| online_em | 1.0990 | 0.2 | {'beta': 0.65, 't0': 10.0} | -0.00287 | 0.0332 | 8/10 | Inconclusive (significant but below practical threshold) |
| emeans_posthoc | 1.0942 | 0.0 | {'alpha_0': 0.35} | +0.00193 | 0.1582 | 3/10 | Inconclusive |
| emeans_then_em3 | 1.0933 | 0.0 | {'alpha_0': 0.35, 'em_iters': 3} | +0.00285 | 0.0312 | 2/10 | Contradicted |
| fullbatch_em_random | 1.0942 | 0.0 | {'max_iters': 25} | +0.00188 | 0.0840 | 3/10 | Inconclusive |
| fullbatch_em_emeans | 1.0927 | 0.0 | {'max_iters': 25} | +0.00345 | 0.0059 | 1/10 | Contradicted |

H3 in the design doc is stated for T ∈ {1, 2, 3}; the epochs actually used are listed per suite.

## Exp 03: paired ablations (H0, H5, H6, H7)

Seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; alpha_0 tuning seeds [1001, 1002, 1003]. diff columns are variant − reference (nats/dim).

**Revival at fixed alpha_0 = 0.05** (reference: spectral_v1/variance_mass)

| Variant | test_nll_per_dim | dead_count | floor_hit_rate | diff vs ref | p |
|---|---|---|---|---|---|
| none/count | 1.0188 | 2.1000 | 0.0688 | +0.00628 | 0.1855 |
| batch_resample/count | 1.0164 | 0.6000 | 0.0000 | +0.00392 | 0.4141 |
| faiss_alt/count | 1.0136 | 0.0000 | 0.0000 | +0.00110 | 0.7305 |
| random_orth/count | 1.0193 | 0.0000 | 0.0000 | +0.00677 | 0.0254 |
| spectral_v1/count | 1.0140 | 0.0000 | 0.0000 | +0.00151 | 0.5000 |
| spectral_v1/variance_mass | 1.0125 | 0.0000 | 0.0000 | ref |  |

**Revival at fixed alpha_0 = 0.1** (reference: spectral_v1/variance_mass)

| Variant | test_nll_per_dim | dead_count | floor_hit_rate | diff vs ref | p |
|---|---|---|---|---|---|
| none/count | 1.0170 | 2.7000 | 0.1625 | +0.00346 | 0.3613 |
| batch_resample/count | 1.0173 | 0.6000 | 0.0000 | +0.00372 | 0.4785 |
| faiss_alt/count | 1.0156 | 0.0000 | 0.0000 | +0.00205 | 0.4551 |
| random_orth/count | 1.0172 | 0.0000 | 0.0000 | +0.00363 | 0.0098 |
| spectral_v1/count | 1.0141 | 0.0000 | 0.0000 | +0.00057 | 0.5000 |
| spectral_v1/variance_mass | 1.0135 | 0.0000 | 0.0000 | ref |  |

**Revival at fixed alpha_0 = 0.2** (reference: spectral_v1/variance_mass)

| Variant | test_nll_per_dim | dead_count | floor_hit_rate | diff vs ref | p |
|---|---|---|---|---|---|
| none/count | 1.0546 | 8.1000 | 0.5062 | +0.03481 | 0.0020 |
| batch_resample/count | 1.0179 | 2.1000 | 0.0125 | -0.00189 | 0.7070 |
| faiss_alt/count | 1.0237 | 0.0000 | 0.0000 | +0.00391 | 0.4648 |
| random_orth/count | 1.0247 | 0.0000 | 0.0000 | +0.00497 | 0.2441 |
| spectral_v1/count | 1.0204 | 0.0000 | 0.0000 | +0.00066 | 0.8125 |
| spectral_v1/variance_mass | 1.0197 | 0.0000 | 0.0000 | ref |  |

**Revival at fixed alpha_0 = 0.35** (reference: spectral_v1/variance_mass)

| Variant | test_nll_per_dim | dead_count | floor_hit_rate | diff vs ref | p |
|---|---|---|---|---|---|
| none/count | 1.1622 | 13.3000 | 0.8313 | +0.14303 | 0.0020 |
| batch_resample/count | 1.0410 | 1.5000 | 0.0063 | +0.02193 | 0.0547 |
| faiss_alt/count | 1.0302 | 0.0000 | 0.0000 | +0.01104 | 0.0137 |
| random_orth/count | 1.0447 | 0.0000 | 0.0000 | +0.02557 | 0.0273 |
| spectral_v1/count | 1.0257 | 0.1000 | 0.0000 | +0.00661 | 0.2402 |
| spectral_v1/variance_mass | 1.0191 | 0.1000 | 0.0063 | ref |  |

**Revival at tuned alpha_0 {'none/count': 0.1, 'batch_resample/count': 0.1, 'faiss_alt/count': 0.1, 'random_orth/count': 0.025, 'spectral_v1/count': 0.025, 'spectral_v1/variance_mass': 0.05}** (reference: spectral_v1/variance_mass)

| Variant | test_nll_per_dim | dead_count | floor_hit_rate | diff vs ref | p |
|---|---|---|---|---|---|
| none/count | 1.0170 | 2.7000 | 0.1625 | +0.00452 | 0.4688 |
| batch_resample/count | 1.0173 | 0.6000 | 0.0000 | +0.00479 | 0.5488 |
| faiss_alt/count | 1.0156 | 0.0000 | 0.0000 | +0.00311 | 0.6289 |
| random_orth/count | 1.0198 | 0.0000 | 0.0000 | +0.00733 | 0.0449 |
| spectral_v1/count | 1.0073 | 0.0000 | 0.0000 | -0.00517 | 0.0508 |
| spectral_v1/variance_mass | 1.0125 | 0.0000 | 0.0000 | ref |  |

**Covariance floor lambda_0** (reference: 0.0001)

| Variant | test_nll_per_dim | floor_hit_rate | max_condition_number | diff vs ref | p |
|---|---|---|---|---|---|
| 1e-06 | 0.9351 | 0.0000 | 11.0317 | +0.00000 | 0.4141 |
| 0.0001 | 0.9351 | 0.0000 | 11.0355 | ref |  |
| 0.01 | 0.9345 | 0.0000 | 10.4949 | -0.00060 | 0.0117 |
| 0.1 | 0.9524 | 0.0052 | 6.1349 | +0.01725 | 0.0020 |

**Covariance structure** (reference: diag)

| Variant | test_nll_per_dim | train_bic | test_bic | n_params | diff vs ref | p |
|---|---|---|---|---|---|---|
| diag | 1.0457 | 61615.3307 | 21759.0043 | 263.0000 | ref |  |
| full | 0.7549 | 49958.3904 | 22316.8216 | 1223.0000 | -0.29079 | 0.0020 |
| low_rank_r2 | 0.9662 | 58736.5987 | 21819.5300 | 511.0000 | -0.07947 | 0.0059 |
| low_rank_r4 | 1.0341 | 64051.5839 | 24504.7072 | 727.0000 | -0.01159 | 0.7305 |
| low_rank_r8 | 1.0170 | 65504.8542 | 26325.7949 | 1063.0000 | -0.02869 | 0.0020 |

_low_rank_diag uses an Oja-style subspace heuristic, not a principled MFA M-step; train_bic (used for H6a) is computed on the training split, test_bic on the test split_

**Top-m responsibilities** (reference: dense)

| Variant | test_nll_per_dim | diff vs ref | p |
|---|---|---|---|
| dense | 0.7413 | ref |  |
| m2 | 0.7445 | +0.00322 | 0.7227 |
| m4 | 0.7535 | +0.01223 | 0.1953 |
| m8 | 0.7395 | -0.00180 | 0.7852 |
| m16 | 0.7382 | -0.00302 | 0.7246 |

**Finalization, small_n_over_k** (reference: last_epoch)

| Variant | train_nll_per_dim | test_nll_per_dim | diff vs ref | p |
|---|---|---|---|---|
| last_epoch | 0.8264 | 1.1718 | ref |  |
| slow_ema | 0.9952 | 1.0593 | -0.11256 | 0.0020 |
| main_ema | 0.8633 | 1.0698 | -0.10203 | 0.0020 |

**Finalization, medium_n_over_k** (reference: last_epoch)

| Variant | train_nll_per_dim | test_nll_per_dim | diff vs ref | p |
|---|---|---|---|---|
| last_epoch | 0.8900 | 0.9512 | ref |  |
| slow_ema | 0.9116 | 0.9530 | +0.00187 | 0.1973 |
| main_ema | 0.8914 | 0.9513 | +0.00009 | 0.9961 |

## Exp 04: CPU E-step timing and float32 precision (no GPU claims)

- [x², x] GEMM expansion vs naive broadcast (NumPy, CPU, median): 106.066 ms vs 4397.757 ms (41.46×), max |Δ log p| 2.0e-13.
- Woodbury vs Cholesky (NumPy, CPU, median): 124.599 ms vs 272.28 ms (2.19×), max |Δ log p| 1.1e-13.
- torch: {'status': 'skipped: torch is not installed (torch path untested on this machine)'}. Not measured: GPU throughput, GPU memory, fused/Triton kernel, bf16 E-step on GPU.

| data mean offset | float32 error, centered | float32 error, uncentered |
|---|---|---|
| 0 | 1.6e-04 | 1.4e-04 |
| 10 | 1.6e-04 | 3.6e-03 |
| 100 | 1.6e-04 | 3.3e-01 |
| 1000 | 1.6e-04 | 3.6e+01 |

