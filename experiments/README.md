# Experiments (`experiments/`)

Scripts, logs (`experiments/logs/`) and results (`experiments/results/`) for the e-GMM study: an EMA-of-sufficient-statistics
(Cappé–Moulines-style online EM) Gaussian mixture trainer with a cosine step schedule, component revival and a proposed
d-aware step/batch law. The design was rewritten on 2026-11-25 after a review; see
[docs/review/2026-11-25-REVIEW.md](../docs/review/2026-11-25-REVIEW.md) for the findings and
[docs/review/2026-11-25-FIX-PLAN.md](../docs/review/2026-11-25-FIX-PLAN.md) for what changed.

---

## 1. Hypotheses and verdict rule

[HYPOTHESES-AND-RESEARCH-DESIGN.md](./HYPOTHESES-AND-RESEARCH-DESIGN.md) defines H0–H7. In short:

- **H0** (exp_02, exp_03): with every method tuned on validation data, is e-GMM better than online EM, e-means → EM and
  full-batch EM? Is revival still needed once α₀ is tuned?
- **H1–H2** (exp_01): does a d term lower held-out α₀ regret, and does the law transfer?
- **H3–H4** (exp_02): matched-budget comparisons against online / full-batch EM and against hard e-means pipelines.
- **H5–H7** (exp_03): revival strategies at tuned α₀, low-rank and top-m, finalization.

Verdicts are computed by [compile_experiment_report.py](./compile_experiment_report.py) with
[src/e_gmm/stats.py](../src/e_gmm/stats.py): paired by seed, exact sign-flip test, p < 0.05, practical threshold
0.01 nats per dimension, at least 10 seeds. Nothing in the scripts states a conclusion.

---

## 2. What each script does

| Script | What it measures | Smoke runtime (16-core CPU) |
| :--- | :--- | :--- |
| [exp_00_preflight_and_resource_audit.py](./exp_00_preflight_and_resource_audit.py) | Environment (torch/CUDA/Triton/sklearn) and local dataset files | seconds |
| [exp_01_phase_a_scaling_laws.py](./exp_01_phase_a_scaling_laws.py) | Adaptive α₀ ladders with censoring, knee search, laws with no d / ambient d / intrinsic d, LODO/LOSO/OOD regret | ~5 min |
| [exp_02_phase_b_billion_benchmarks.py](./exp_02_phase_b_billion_benchmarks.py) | Tuned, matched, paired benchmark of e-GMM against every baseline in `e_gmm.benchmark` | see log |
| [exp_03_phase_c_ablations.py](./exp_03_phase_c_ablations.py) | Revival × α₀, floor, covariance structure, top-m, finalization, all paired | see log |
| [exp_04_kernel_and_memory_bench.py](./exp_04_kernel_and_memory_bench.py) | CPU timing and parity of E-step formulations; float32 precision with/without centering | < 1 min |
| [compile_experiment_report.py](./compile_experiment_report.py) | Writes `results/FINAL-EXPERIMENT-REPORT.md` with data-driven verdicts | < 1 s |

GPU budgets in the original proposal (200–300 GPU-hours) are estimates for a torch backend that has **not been executed**
on the review machine: torch is not installed there, every torch test skips, and the Triton kernel is not wired in.

### Real data (no downloads, no fallbacks)

Put pre-downloaded files in one directory. Any of `.npy`, `.fvecs`, `.bvecs`, `.fbin`, `.u8bin`, `.i8bin` is read:

- `sift_train.*` (d=128), `deep_train.*` (d=96), `spacev_train.*` (d=100, uint8),
  `dinov2_100m_train.*` (d=1024), `imagenet1k_train.*` (d=768).

Only the `*_train` file is read. Train, validation and test rows are sampled at random from it, are disjoint, and held-out
rows that duplicate a training row are replaced. A missing file raises; there is no synthetic fallback and no dimension
truncation. A lower dimension is used only when asked for, by PCA fitted on the training split. All splits are
scale-normalized on the training split so the covariance floor λ₀ = 1e-4 means the same thing on every dataset.

---

## 3. Commands

### CPU smoke runs (synthetic data)

```bash
python experiments/exp_00_preflight_and_resource_audit.py
python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --seeds 2
python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --seeds 10
python experiments/exp_03_phase_c_ablations.py --mode smoke --seeds 10
python experiments/exp_04_kernel_and_memory_bench.py --mode smoke
python experiments/compile_experiment_report.py
```

`--dry-run` on exp_01–exp_04 prints the grid and exits without fitting.

### Real data

```bash
# Scaling laws on one source: d is varied by PCA of that source, so every d must be <= its dimension
python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --data real --data-file /data/egmm/dinov2_100m_train.npy --seeds 2
python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --data real --data-file /data/egmm/deep_train.fvecs --dims 8,16,32,64

# Benchmark on real embeddings (native d, or --project-dim N), 10 paired seeds; --epochs 2 for H3's low-epoch regime
python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --data real --data-dir /data/egmm --seeds 10
python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --data real --data-dir /data/egmm --project-dim 64 --epochs 2
```

### Full mode (untested torch backend)

```bash
python experiments/exp_00_preflight_and_resource_audit.py --data-dir /data/egmm --strict
python experiments/exp_01_phase_a_scaling_laws.py --mode full --backend torch --data real \
    --data-file /data/egmm/dinov2_100m_train.npy --manifest configs/phase_a_scaling_manifest.json --seeds 3
python experiments/exp_02_phase_b_billion_benchmarks.py --mode full --data real --data-dir /data/egmm --backend torch --seeds 10
python experiments/exp_03_phase_c_ablations.py --mode full --seeds 10
python experiments/compile_experiment_report.py
```

Full-mode exp_01 takes its axes (n, d, k, B, T) from the manifest, which varies T ∈ {2, 3, 5}. exp_03 has only the
NumPy backend. Run the torch paths on a small case and compare them with NumPy before trusting a full run.
