# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)

**Generated**: 2026-11-25 01:10:59  
**Completed Tiers**: PoC 1, PoC 2, PoC 3, PoC 4  
**Verdict rule**: `e_gmm.stats.verdict`, threshold 0.01 nats/dim, min_n 10, alpha 0.05, exact paired sign-flip p. Synthetic data only; nothing here supports a claim about real embeddings.

---

## Hypothesis verdict matrix

| ID | Claim | Evidence | Verdict |
| :--- | :--- | :--- | :--- |
| H1.1 | 2d-GEMM Flash-GMM and Woodbury determinant lemma match naive log-densities (tol 1e-10) | PoC 1 (`4.26e-14`, `1.42e-14`) | **Supported (deterministic check)** |
| H1.2 | Additive floor keeps min eigenvalue >= lambda_0 on a rank-1 batch | PoC 1 (`1.000e-03` vs `1.0e-03`) | **Supported (deterministic check)** |
| H2.zipf.egmm_no_revive | e-GMM spectral better than tuned `egmm_no_revive` on zipf | PoC 2 (diff -0.0079, p 0.0352, n 10) | **Inconclusive (significant but below practical threshold)** |
| H2.zipf.egmm_faiss_alt | e-GMM spectral better than tuned `egmm_faiss_alt` on zipf | PoC 2 (diff -0.0008, p 0.8379, n 10) | **Inconclusive** |
| H2.zipf.online_em | e-GMM spectral better than tuned `online_em` on zipf | PoC 2 (diff -0.0072, p 0.1758, n 10) | **Inconclusive** |
| H2.zipf.emeans_posthoc | e-GMM spectral better than tuned `emeans_posthoc` on zipf | PoC 2 (diff -0.0014, p 0.8477, n 10) | **Inconclusive** |
| H2.zipf.emeans_then_em3 | e-GMM spectral better than tuned `emeans_then_em3` on zipf | PoC 2 (diff -0.0030, p 0.5625, n 10) | **Inconclusive** |
| H2.zipf.fullbatch_em_random | e-GMM spectral better than tuned `fullbatch_em_random` on zipf | PoC 2 (diff -0.0093, p 0.0430, n 10) | **Inconclusive (significant but below practical threshold)** |
| H2.zipf.fullbatch_em_emeans | e-GMM spectral better than tuned `fullbatch_em_emeans` on zipf | PoC 2 (diff -0.0066, p 0.1562, n 10) | **Inconclusive** |
| H2.zipf.alpha0.05 | spectral revival beats no revival at alpha_0=0.05 on zipf | PoC 2 (diff -0.0070, p 0.0820) | **Inconclusive** |
| H2.zipf.alpha0.1 | spectral revival beats no revival at alpha_0=0.1 on zipf | PoC 2 (diff -0.0068, p 0.0879) | **Inconclusive** |
| H2.zipf.alpha0.2 | spectral revival beats no revival at alpha_0=0.2 on zipf | PoC 2 (diff -0.0430, p 0.0020) | **Supported** |
| H2.zipf.alpha0.35 | spectral revival beats no revival at alpha_0=0.35 on zipf | PoC 2 (diff -0.1691, p 0.0020) | **Supported** |
| H2.overlap.egmm_no_revive | e-GMM spectral better than tuned `egmm_no_revive` on overlap | PoC 2 (diff -0.0029, p 0.2109, n 10) | **Inconclusive** |
| H2.overlap.egmm_faiss_alt | e-GMM spectral better than tuned `egmm_faiss_alt` on overlap | PoC 2 (diff -0.0079, p 0.1738, n 10) | **Inconclusive** |
| H2.overlap.online_em | e-GMM spectral better than tuned `online_em` on overlap | PoC 2 (diff -0.0116, p 0.0820, n 10) | **Inconclusive** |
| H2.overlap.emeans_posthoc | e-GMM spectral better than tuned `emeans_posthoc` on overlap | PoC 2 (diff -0.0900, p 0.0020, n 10) | **Supported** |
| H2.overlap.emeans_then_em3 | e-GMM spectral better than tuned `emeans_then_em3` on overlap | PoC 2 (diff -0.0467, p 0.0020, n 10) | **Supported** |
| H2.overlap.fullbatch_em_random | e-GMM spectral better than tuned `fullbatch_em_random` on overlap | PoC 2 (diff -0.0040, p 0.3594, n 10) | **Inconclusive** |
| H2.overlap.fullbatch_em_emeans | e-GMM spectral better than tuned `fullbatch_em_emeans` on overlap | PoC 2 (diff +0.0095, p 0.0059, n 10) | **Contradicted** |
| H2.overlap.alpha0.05 | spectral revival beats no revival at alpha_0=0.05 on overlap | PoC 2 (diff -0.0008, p 0.5000) | **Inconclusive** |
| H2.overlap.alpha0.1 | spectral revival beats no revival at alpha_0=0.1 on overlap | PoC 2 (diff -0.0029, p 0.2109) | **Inconclusive** |
| H2.overlap.alpha0.2 | spectral revival beats no revival at alpha_0=0.2 on overlap | PoC 2 (diff -0.0735, p 0.0020) | **Supported** |
| H2.overlap.alpha0.35 | spectral revival beats no revival at alpha_0=0.35 on overlap | PoC 2 (diff -0.1742, p 0.0020) | **Supported** |
| H3.LODO.ambient_d_vs_no_d | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff +0.0007, p 0.0031, n 80) | **Contradicted** |
| H3.LODO.intrinsic_d_vs_no_d | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff +0.0006, p 0.0023, n 80) | **Contradicted** |
| H3.OOD.ambient_d_vs_no_d | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff +0.0000, p 0.7500, n 3) | **Inconclusive (too few seeds)** |
| H3.OOD.intrinsic_d_vs_no_d | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff +0.0000, p 0.7500, n 3) | **Inconclusive (too few seeds)** |
| H3.OOD.paper_prior_no_d_vs_no_d | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff +0.0009, p 0.7500, n 3) | **Inconclusive (too few seeds)** |
| H3.knee | B_knee depends on d | PoC 3 (no paired regret stored) | **Untested** |
| H4.1 | Top-m sparse E-step preserves accuracy at m/k M-step cost | PoC 4 (single seed) | **Untested** |
| H4.2 | Low-rank+diag has the best BIC on rotated embeddings | PoC 4 (single seed) | **Untested** |
| H4.3 | last_epoch wins train NLL while slow_ema wins test NLL | PoC 4 (single seed) | **Untested** |

---

## PoC 1: Mathematical identities, PSD floor and memory arithmetic

- Flash-GMM 2d-GEMM max |delta ln p|: `4.263e-14`; Woodbury max |delta ln p|: `1.421e-14`; tolerance `1e-10`
- Rank-1 batch min eigenvalue `1.000000e-03` (floor `1.0e-03`)
- Spectral split mean conservation error `4.441e-16`
- Full-covariance state at k=65536, d=1024: `256.0 GiB` (arithmetic, not a measurement)

| Preset | Cov | k | d | B | State (GiB) | Naive BxK HBM (GiB) | Tiled HBM (GiB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sift-1m` | `diag` | 4096 | 128 | 16384 | 0.0039 | 0.25 | 0.125 |
| `deep-1b` | `diag` | 65536 | 96 | 262144 | 0.0471 | 64.0 | 2.0 |
| `spacev-100m` | `diag` | 65536 | 100 | 131072 | 0.0491 | 32.0 | 1.0 |
| `dinov2-100m` | `diag` | 16384 | 1024 | 65536 | 0.1251 | 4.0 | 0.5 |
| `dinov2-100m` | `low_rank_diag` | 16384 | 1024 | 65536 | 1.1251 | 4.0 | 0.5 |
| `dinov2-100m` | `full` | 16384 | 1024 | 65536 | 64.0626 | 4.0 | 0.5 |
| `imagenet-1k` | `full` | 1024 | 768 | 16384 | 2.2529 | 0.062 | 0.0625 |

The memory table is closed-form arithmetic on the stated presets. Nothing here was run on a GPU, and no claim about speed or about real-data accuracy follows from it (untested).

---

## PoC 2: Matched tuned baselines and alpha_0 sensitivity

Setup: d=16, k=16, B=64, epochs=6, train/val/test = 2400/800/800, seeds = 10, full covariance, floor 1e-4 for every method, each method tuned on validation with a separate tuning seed. Unit: test NLL in nats/dim. Paired difference column is (egmm_spectral - baseline): negative means e-GMM is better. Verdict uses threshold 0.01 nats/dim, min_n=10, alpha 0.05, no multiple-comparison correction (14 benchmark comparisons plus the sensitivity block).

### Regime `zipf` ({"sigma_mu_sq": 0.6666666666666666, "sigma_w_sq": 0.3333333333333333, "zipf": true, "zipf_power": 1.2})

| Baseline | n | egmm_spectral - baseline (nats/dim) | p (sign-flip) | baseline mean dead | Verdict: e-GMM better |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `egmm_spectral` (reference) | 10 | - | - | 0.0 | - |
| `egmm_no_revive` | 10 | -0.0079 | 0.0352 | 2.4 | Inconclusive (significant but below practical threshold) |
| `egmm_faiss_alt` | 10 | -0.0008 | 0.8379 | 0.0 | Inconclusive |
| `online_em` | 10 | -0.0072 | 0.1758 | 3.4 | Inconclusive |
| `emeans_posthoc` | 10 | -0.0014 | 0.8477 | 0.0 | Inconclusive |
| `emeans_then_em3` | 10 | -0.0030 | 0.5625 | 0.0 | Inconclusive |
| `fullbatch_em_random` | 10 | -0.0093 | 0.0430 | 0.0 | Inconclusive (significant but below practical threshold) |
| `fullbatch_em_emeans` | 10 | -0.0066 | 0.1562 | 0.0 | Inconclusive |

Methods not run: `{"sklearn_gmm": "scikit-learn is not installed"}`

Tuned settings: `egmm_spectral`={'alpha_0': 0.1}; `egmm_no_revive`={'alpha_0': 0.025}; `egmm_faiss_alt`={'alpha_0': 0.1}; `online_em`={'beta': 0.55, 't0': 100.0}; `emeans_posthoc`={'alpha_0': 0.7}; `emeans_then_em3`={'alpha_0': 0.2, 'em_iters': 3}; `fullbatch_em_random`={'max_iters': 25}; `fullbatch_em_emeans`={'max_iters': 25}

#### alpha_0 sensitivity on `zipf`: is revival only needed because alpha_0 is too large?

| alpha_0 | spectral NLL | no-revive NLL | spectral - no-revive | p | dead (spectral / none) | Verdict: revival helps |
| ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| 0.05 | 1.0075 | 1.0145 | -0.0070 | 0.0820 | 0.0 / 1.5 | Inconclusive |
| 0.1 | 1.0069 | 1.0137 | -0.0068 | 0.0879 | 0.0 / 2.7 | Inconclusive |
| 0.2 | 1.0178 | 1.0607 | -0.0430 | 0.0020 | 0.0 / 9.5 | Supported |
| 0.35 | 1.0081 | 1.1772 | -0.1691 | 0.0020 | 0.0 / 13.8 | Supported |

At the smallest alpha_0 the revival verdict is **Inconclusive**. If revival only helps at large alpha_0 the H0 (revival compensates for a too-large step) is not rejected for that regime.

### Regime `overlap` ({"sigma_mu_sq": 0.2, "sigma_w_sq": 0.8, "zipf": false, "zipf_power": 1.0})

| Baseline | n | egmm_spectral - baseline (nats/dim) | p (sign-flip) | baseline mean dead | Verdict: e-GMM better |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `egmm_spectral` (reference) | 10 | - | - | 0.0 | - |
| `egmm_no_revive` | 10 | -0.0029 | 0.2109 | 2.1 | Inconclusive |
| `egmm_faiss_alt` | 10 | -0.0079 | 0.1738 | 0.0 | Inconclusive |
| `online_em` | 10 | -0.0116 | 0.0820 | 1.4 | Inconclusive |
| `emeans_posthoc` | 10 | -0.0900 | 0.0020 | 0.0 | Supported |
| `emeans_then_em3` | 10 | -0.0467 | 0.0020 | 0.0 | Supported |
| `fullbatch_em_random` | 10 | -0.0040 | 0.3594 | 0.2 | Inconclusive |
| `fullbatch_em_emeans` | 10 | +0.0095 | 0.0059 | 0.0 | Contradicted |

Methods not run: `{"sklearn_gmm": "scikit-learn is not installed"}`

Tuned settings: `egmm_spectral`={'alpha_0': 0.1}; `egmm_no_revive`={'alpha_0': 0.1}; `egmm_faiss_alt`={'alpha_0': 0.35}; `online_em`={'beta': 0.55, 't0': 100.0}; `emeans_posthoc`={'alpha_0': 0.1}; `emeans_then_em3`={'alpha_0': 0.2, 'em_iters': 3}; `fullbatch_em_random`={'max_iters': 25}; `fullbatch_em_emeans`={'max_iters': 25}

#### alpha_0 sensitivity on `overlap`: is revival only needed because alpha_0 is too large?

| alpha_0 | spectral NLL | no-revive NLL | spectral - no-revive | p | dead (spectral / none) | Verdict: revival helps |
| ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| 0.05 | 1.2213 | 1.2221 | -0.0008 | 0.5000 | 0.0 / 0.2 | Inconclusive |
| 0.1 | 1.1959 | 1.1989 | -0.0029 | 0.2109 | 0.0 / 2.1 | Inconclusive |
| 0.2 | 1.1949 | 1.2684 | -0.0735 | 0.0020 | 0.0 / 10.1 | Supported |
| 0.35 | 1.2021 | 1.3763 | -0.1742 | 0.0020 | 0.1 / 14.8 | Supported |

At the smallest alpha_0 the revival verdict is **Inconclusive**. If revival only helps at large alpha_0 the H0 (revival compensates for a too-large step) is not rejected for that regime.


---

## PoC 3: Does a dimension term improve alpha_0 / B_knee prediction? (synthetic, small grid)

- Grid: dims=[8, 16, 32, 64], ks=[4, 8, 16], ns=[1200, 4800], B=[32, 128], T=[2, 4], seeds=[1, 2]. Small synthetic grid (see 'grid'), 2 seeds; chosen to keep the run under ~10 minutes.
- Alpha-ladder optima censored (on the ladder edge): **16 / 96**; knee optima censored: **2 / 48**. Skipped configs: 0.
- Intrinsic dims (TwoNN): `{"d=8,n=1200": 7.06997410147607, "d=8,n=4800": 7.314328701862799, "d=16,n=1200": 10.419704020833143, "d=16,n=4800": 10.865454260106452, "d=32,n=1200": 14.875072395612635, "d=32,n=4800": 15.939535111725736, "d=64,n=1200": 18.50109664643137, "d=64,n=4800": 20.72821001896936, "d=24,n=2400": 13.694327782196991, "d=96,n=2400": 24.556550681054727, "d=16,n=2400": 10.898461954843098}`

| Law | n used | censored excluded | exponents (k, B, n, T, d) | unidentified | LOSO-n | LODO-d |
| :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| `no_d` | 80 | 16 | -0.90, 0.52, -0.13, 0.05, n/a | none | `{"loso_log_rmse": 0.48945678068423637, "loso_mean_rel_err": 0.45218935588862513, "loso_regret_nats_per_dim_mean": 0.0038954881318692567, "loso_regret_nats_per_dim_median": 0.0029489633078154665, "loso_n_regret_extrapolated": 1.0, "loso_n_heldout_points": 80.0, "n_folds": 2.0}` | `{"lodo_log_rmse": 0.4688532067728101, "lodo_mean_rel_err": 0.43102054210443363, "lodo_regret_nats_per_dim_mean": 0.003498475591841607, "lodo_regret_nats_per_dim_median": 0.00259442803733112, "lodo_n_regret_extrapolated": 1.0, "lodo_n_heldout_points": 80.0, "n_folds": 4.0}` |
| `ambient_d` | 80 | 16 | -0.90, 0.52, -0.13, 0.05, -0.01 | none | `{"loso_log_rmse": 0.4903995549586724, "loso_mean_rel_err": 0.4535801383313548, "loso_regret_nats_per_dim_mean": 0.003944126030547992, "loso_regret_nats_per_dim_median": 0.0029522158567244494, "loso_n_regret_extrapolated": 1.0, "loso_n_heldout_points": 80.0, "n_folds": 2.0}` | `{"lodo_log_rmse": 0.529995489577082, "lodo_mean_rel_err": 0.4648009208601156, "lodo_regret_nats_per_dim_mean": 0.0042388626385074235, "lodo_regret_nats_per_dim_median": 0.002601591467357167, "lodo_n_regret_extrapolated": 1.0, "lodo_n_heldout_points": 80.0, "n_folds": 4.0}` |
| `intrinsic_d` | 80 | 16 | -0.90, 0.52, -0.13, 0.05, -0.05 | none | `{"loso_log_rmse": 0.4891039337602895, "loso_mean_rel_err": 0.45245019828236116, "loso_regret_nats_per_dim_mean": 0.003943695354417814, "loso_regret_nats_per_dim_median": 0.0029536086437751408, "loso_n_regret_extrapolated": 1.0, "loso_n_heldout_points": 80.0, "n_folds": 2.0}` | `{"lodo_log_rmse": 0.5222674865185677, "lodo_mean_rel_err": 0.45958415841281264, "lodo_regret_nats_per_dim_mean": 0.004130616328895118, "lodo_regret_nats_per_dim_median": 0.0027277548052543343, "lodo_n_regret_extrapolated": 1.0, "lodo_n_heldout_points": 80.0, "n_folds": 4.0}` |

### Paired regret of d-aware law vs no-d law (nats/dim; negative = d term helps)

| Test | Comparison | n | mean regret diff | p | Verdict: d term beats no-d |
| :--- | :--- | ---: | ---: | ---: | :--- |
| LODO (held-out d) | `ambient_d_vs_no_d` | 80 | +0.0007 | 0.0031 | Contradicted |
| LODO (held-out d) | `intrinsic_d_vs_no_d` | 80 | +0.0006 | 0.0023 | Contradicted |
| OOD configs | `ambient_d_vs_no_d` | 3 | +0.0000 | 0.7500 | Inconclusive (too few seeds) |
| OOD configs | `intrinsic_d_vs_no_d` | 3 | +0.0000 | 0.7500 | Inconclusive (too few seeds) |
| OOD configs | `paper_prior_no_d_vs_no_d` | 3 | +0.0009 | 0.7500 | Inconclusive (too few seeds) |

Caveats: LODO points are held-out (config, d) cells pooled over the grid, so they are not independent draws and the p-value is optimistic. `verdict` labels a significant wrong-direction difference Contradicted even when it is far below the 0.01 nats/dim practical threshold, so read the size column. For OOD, n counts configs (not seeds), hence the min_n rule makes it Inconclusive.

OOD configs (regret in nats/dim at each law's predicted alpha_0; `extrapolated` means the prediction fell outside the measured ladder):

| config | alpha* | censored | ambient_d | intrinsic_d | no_d | paper_prior_no_d |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: |
| `{"d": 24, "k": 12, "n": 2400, "B": 64, "T": 3}` | 0.2985 | False | 0.0034 | 0.0034 | 0.0034 | 0.0009 |
| `{"d": 96, "k": 8, "n": 2400, "B": 64, "T": 3}` | 0.4228 | False | 0.0002 | 0.0001 | 0.0003 | 0.0003 |
| `{"d": 16, "k": 12, "n": 2400, "B": 64, "T": 3}` | 0.1688 | False | 0.0097 | 0.0098 | 0.0097 | 0.0146 |

`*` = extrapolated beyond the measured alpha ladder.

### B_knee law

`no_d`: n_used=46, censored excluded=2, unidentified=none; `ambient_d`: n_used=46, censored excluded=2, unidentified=none; `intrinsic_d`: n_used=46, censored excluded=2, unidentified=none

Knee skip reason: `None`.
Knee optima that are censored or unidentified carry no claim about B_knee scaling. No paired regret comparison is stored for the knee law, so it is **Untested**.

---

## PoC 4: Top-m sparsity, low-rank Woodbury, finalization (single seed, descriptive)

This JSON holds one run per setting, with no seeds and no paired statistics, so none of its claims can receive a verdict. The numbers are shown for inspection; every claim below is **Untested**.

| Top-m | M-step work | Test NLL | Fit ms |
| :---: | :---: | :---: | ---: |
| `2` | 0.0625x | 14.7435 | 256.3 |
| `4` | 0.125x | 14.1575 | 321.1 |
| `8` | 0.25x | 14.4512 | 305.0 |
| `dense` | 1.0x | 14.4510 | 225.4 |

| Cov mode | params | Test NLL | Test BIC |
| :--- | ---: | ---: | ---: |
| `diag` | 263 | 14.5308 | 25007.3 |
| `low_rank_diag` | 727 | 11.8136 | 23761.5 |
| `full` | 1223 | 10.2735 | 24612.9 |

| Finalizer | Train NLL | Test NLL |
| :--- | :---: | :---: |
| `main_ema` | 23.9068 | 24.6154 |
| `last_epoch` | 23.8808 | 24.6447 |
| `slow_ema` | 23.9421 | 24.5874 |

Torch parity: **Not run** (no torch result in the JSON).
