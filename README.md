# e_GMM-scaling

**e-GMM** applies the [e-means](https://arxiv.org/abs/2611.09501) training recipe to Gaussian mixture models. The recipe is:
- exponentially decayed sufficient statistics,
- a cosine step schedule,
- dead-component revival,
- a finalization step,
- fitted power laws for the initial step size α₀ and the batch size.

Concretely, e-GMM:
- tracks (N_j, S_j, Q_j) per component,
- reads out covariances with an additive floor λ₀·I,
- revives starved components by splitting a donor along its principal eigenvector v₁,
- offers diagonal, low-rank + diagonal (Woodbury) and full covariances.

> **What this is, and is not.** The update rule is the stochastic-approximation online EM of Cappé & Moulines (2009) with a cosine schedule instead of (t + t₀)^−β. The e-means paper itself credits EMA-statistics EM to Nowlan (1991) and Neal & Hinton (1998).
> The open research questions are empirical:
> 1. Do fitted step/batch laws transfer for *soft* mini-batch EM?
> 2. Does component revival help at scale against tuned baselines?
>
> See the [prior-art list](./docs/review/2026-11-25-REVIEW.md#7-prior-art-to-position-against-from-memory--verify-before-citing) before writing any novelty claim.

> **Status (2026-11-25)**
> - **Has run:** the NumPy CPU path only, on synthetic data. The PoCs and smoke experiments use n ≤ 4 800, d ≤ 64 and k ≤ 32. The NumPy unit tests pass.
> - **Has not run:**
>   - The PyTorch path: `torch` is not installed on the development machine, so the torch tests skip.
>   - The Triton kernel, which is defined but not called anywhere.
>   - Any real dataset.
>   - Any GPU job. Every GPU-hour figure in this repo is an estimate.
> - **Reviewed:** an external review on 2026-11-25 found that several earlier "Supported" verdicts were hard-coded and contradicted their own numbers. See [docs/review/2026-11-25-REVIEW.md](./docs/review/2026-11-25-REVIEW.md). Report verdicts are now computed from paired multi-seed statistics ([experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md](./experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md)). Results produced before that review should not be cited.
> - **Download policy:** nothing in the repository downloads datasets or launches GPU jobs.

---

## Repository map

| Folder / file | Purpose |
| :--- | :--- |
| [`docs/review/`](./docs/review/2026-11-25-REVIEW.md) | The 2026-11-25 review (findings, 10-seed re-check, prior art) and the [fix plan](./docs/review/2026-11-25-FIX-PLAN.md), with the fix log |
| [`docs/basic-terms/`](./docs/basic-terms/00-INDEX.md) | Primers: EMA statistics (N, S, Q), Cappé–Moulines online EM, covariance parametrizations and Woodbury, the [x², x] GEMM form of the diagonal E-step, spectral splitting, scaling laws, glossary |
| [`docs/architecture/`](./docs/architecture/IMPLEMENTATION-PLAN-EGMM.md) | Implementation plan, memory/FLOP tables and the phased experiment plan, with a status-and-caveats section |
| [`docs/assets/images/`](./docs/assets/images/) | SVG diagrams |
| `src/e_gmm/` | `config`, `schedule`, `numpy_egmm` (e-GMM and baselines), `torch_egmm`, `triton_kernels`, `metrics`, `datasets`, `scaling_law`, `stats` (paired tests), `benchmark` (method registry and validation tuning), `sweep` (scaling-law sweeps) |
| `scripts/` | Memory/FLOP profiler, NumPy identity checks, sweep planner, preliminary diagnostics, torch validator |
| `configs/` | JSON presets (`tiny_egmm_debug`, `sift1m_diag_k4096`, `deep1b_diag_k65536`, `dinov2_100m_lowrank_k16384`, `imagenet1k_full_k1024`) and the Phase-A sweep manifest |
| [`poc/`](./poc/README.md) | PoC 1–4 and the PoC report compiler |
| [`experiments/`](./experiments/README.md) | Hypotheses H0–H7, Exp 00–04, logs, JSON results and the experiment report compiler |
| `tests/` | NumPy unit tests, review-fix tests (`test_review_fixes.py`) and torch tests (skipped without torch) |

---

## Architecture

![e-GMM pipeline](./docs/assets/images/egmm-3tuple-pipeline.svg)

---

## Quick start (CPU, NumPy, no downloads)

```bash
# Unit tests (NumPy; the torch tests skip when torch is absent)
python -m unittest discover -s tests -v

# Memory/FLOP tables and float64 identity checks
python scripts/explore_egmm_memory_and_flops.py
python scripts/mini_run_math_verify.py

# Proofs of concept (small synthetic data), then compile poc/results/COMPILED-POC-REPORT.md
python poc/poc_01_math_and_profiler.py
python poc/poc_02_collapse_and_spectral_split.py
python poc/poc_03_dimension_scaling_law.py
python poc/compile_poc_report.py

# Experiment smoke runs, then compile experiments/results/FINAL-EXPERIMENT-REPORT.md
python experiments/exp_00_preflight_and_resource_audit.py
python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --seeds 2      # ~5 min
python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --seeds 10  # ~10 min
python experiments/exp_03_phase_c_ablations.py --mode smoke --seeds 10
python experiments/exp_04_kernel_and_memory_bench.py --mode smoke
python experiments/compile_experiment_report.py
```

Smoke runs are small CPU runs on synthetic data. Their verdicts are evidence about the method at that scale only, and
many come out *Inconclusive*; that is an outcome, not a failure. Run each script with `--help` to see its flags.
[experiments/README.md](./experiments/README.md) has the real-data commands.

## Running on real data

Put the files in one directory and pass it as `--data-dir` (exp_02) or pass one file as `--data-file` (exp_01).
`exp_00` lists which expected files it finds. Only the `*_train` file is read: train, validation and test rows are drawn
from it at random, disjoint, with held-out duplicates of training rows replaced.

| File | Dataset | d |
| :--- | :--- | :---: |
| `deep_train.fvecs`, `deep_test.fvecs` | DEEP | 96 |
| `sift_train.fvecs`, `sift_test.fvecs` | SIFT | 128 |
| `spacev_train.bvecs`, `spacev_test.bvecs` | SPACEV (uint8) | 100 |
| `dinov2_100m_train.npy`, `dinov2_100m_test.npy` | DINOv2 embeddings | 1024 |
| `imagenet1k_train.npy`, `imagenet1k_test.npy` | ImageNet-1k embeddings | 768 |

Supported formats are `.npy`, `.fvecs`, `.bvecs`, `.fbin`, `.u8bin` and `.i8bin`. The rules below come from the review fixes; the [fix log](./docs/review/2026-11-25-REVIEW.md#fix-log) records what has been verified.
- **A missing or malformed file is an error.** The loader no longer substitutes synthetic data under a real dataset's name.
- **Dimensions are never silently truncated.** A lower dimension must be requested explicitly as a PCA projection, and the report records it.
- **Rows are sampled at random**, not taken from the start of the file. Held-out points that duplicate training points are removed.
- **Data is scale-normalized before fitting.** The covariance floor λ₀ is absolute, so it only means something on a known scale.

On a machine with `torch` and a GPU:

```bash
python scripts/mini_run_torch_flash_gmm.py
python poc/poc_04_flash_gmm_and_lowrank.py --backend torch
python -m unittest tests.test_egmm_torch -v
python experiments/exp_00_preflight_and_resource_audit.py --data-dir /data/egmm --strict
python experiments/exp_01_phase_a_scaling_laws.py --mode full --backend torch --data real --data-file /data/egmm/dinov2_100m_train.npy --seeds 3
python experiments/exp_02_phase_b_billion_benchmarks.py --mode full --backend torch --data real --data-dir /data/egmm --seeds 10
python experiments/exp_03_phase_c_ablations.py --mode full --seeds 10   # NumPy only
python experiments/exp_04_kernel_and_memory_bench.py --mode full --backend torch
python experiments/compile_experiment_report.py
```

Before spending GPU-hours, note three limitations:
- The torch low-rank and full-covariance paths loop over components in Python.
- The Triton kernel is not wired in.
- The torch path has never been executed.

Run the torch tests and a timing check at the target k first.
