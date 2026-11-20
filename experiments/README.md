# Full Experimental Suite (`experiments/`)

This folder contains the complete, self-contained experimental harness for **`e-GMM: Scaling Expectation-Maximization to Billion-Scale Datasets via Exponential Moving Averages`**. All scripts, runbooks, logs (`experiments/logs/`), and compiled JSON/Markdown results (`experiments/results/`) live strictly inside this directory.

---

## 1. Research Subject & Hypotheses

See [**`HYPOTHESES-AND-RESEARCH-DESIGN.md`**](./HYPOTHESES-AND-RESEARCH-DESIGN.md) for the complete theoretical framing, null/alternative criteria, and baseline fairness contracts covering **H1–H7**:
- **H1–H2 (Phase A)**: Dimension-aware $\alpha_0(k, B, n, T, d)$ & $B_{\text{knee}}(k, n, T, d)$ laws, LODO-$d$, LOSO-$n$, and OOD transfer.
- **H3–H4 (Phase B)**: Controlled benchmarks vs. Full-Batch EM, Cappé & Moulines (2009) Online EM, and `e-means + PostHoc` across separated & overlapping/correlated regimes.
- **H5–H7 (Phase C & Systems)**: Spectral $\mathbf{v}_1$ splitting vs. heuristics, $\lambda_0$ & `low_rank_diag` ($r \in \{4, 8, 16, 32\}$) vs. `diag`/`full`, Top-$m$ sparse truncation, `last_epoch` vs. `slow_ema` finalization, and Flash-GMM `2d`-GEMM / Woodbury throughput.

---

## 2. Extensive Resource, Hardware & Dataset Requirements

### Hardware & Compute Budget (`200–300 GPU-Hours`)
| Phase | Script | GPU Target | Peak VRAM | Host RAM | SSD Storage | Est. GPU-Hours |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Preflight** | [`exp_00_preflight_and_resource_audit.py`](./exp_00_preflight_and_resource_audit.py) | Local CPU / Any Node | `0 GB` | `>= 4 GB` | `< 10 MB` | `< 1 min` (`$0`) |
| **Phase A** | [`exp_01_phase_a_scaling_laws.py`](./exp_01_phase_a_scaling_laws.py) | `1× A100/H100 (80GB)` | `16–48 GB` | `128 GB` | `150 GB` | `~80 hrs` |
| **Phase B** | [`exp_02_phase_b_billion_benchmarks.py`](./exp_02_phase_b_billion_benchmarks.py) | `1–2× A100/H100 (80GB)` | `24–72 GB` | `256–512 GB` | `1.2 TB` NVMe | `~120 hrs` |
| **Phase C** | [`exp_03_phase_c_ablations.py`](./exp_03_phase_c_ablations.py) | `1× A100/H100 (80GB)` | `24–64 GB` | `128 GB` | `200 GB` | `~50 hrs` |
| **Systems** | [`exp_04_kernel_and_memory_bench.py`](./exp_04_kernel_and_memory_bench.py) | `1× A100/H100 (80GB)` | `16–48 GB` | `64 GB` | `< 1 GB` | `~5 hrs` |
| **Report** | [`compile_experiment_report.py`](./compile_experiment_report.py) | Any CPU (Stdlib) | `0 GB` | `< 1 GB` | `< 10 MB` | `< 1 sec` |

### Software Dependencies
- **Offline Smoke / CPU Mode**: Python `>= 3.10`, `numpy>=1.26.0`
- **Full GPU Cluster Mode**: `torch>=2.3.0` (CUDA), `triton>=3.0.0` (Linux CUDA), `scikit-learn>=1.4.0`, `scipy>=1.12.0` (for full-batch CPU oracle checks up to 1M).

### Real Dataset Preparation (`--mode full` — Zero Auto-Downloads)
Place pre-downloaded memmap files (`.npy`, `.fvecs`, or `.bvecs`) under a local directory (e.g., `/data/egmm/`) and pass `--data-dir /data/egmm`:
- `deep_train.fvecs` / `deep_test.fvecs` (`d=96`, BigANN DEEP1B; 20k deduplicated holdout split 10k val / 10k test)
- `sift_train.fvecs` / `sift_test.fvecs` (`d=128`, SIFT1M / SIFT1B)
- `spacev_train.bvecs` / `spacev_test.bvecs` (`d=100`, Microsoft SPACEV)
- `dinov2_100m_train.npy` / `dinov2_100m_test.npy` (`d=1024`, BF16/FP32)
- `imagenet1k_train.npy` / `imagenet1k_test.npy` (`d=768`, frozen ViT/DINO embeddings)

---

## 3. How to Conduct the Full Run (Step-by-Step)

### Option A: Local / Smoke End-to-End Execution (Runs Immediately on Any CPU/GPU)
```bash
# 0. Inspect system resources, dataset presets & VRAM/HBM budgets
python experiments/exp_00_preflight_and_resource_audit.py

# 1. Phase A Smoke: Fit d-aware alpha_0 & B_knee + LODO/LOSO/OOD validation
python experiments/exp_01_phase_a_scaling_laws.py --mode smoke

# 2. Phase B Smoke: Multi-seed baselines on Zipf-Separated & Overlapping-Anisotropic regimes
python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke

# 3. Phase C Smoke: Revival, lambda_0, covariance mode, Top-m, and finalization ablations
python experiments/exp_03_phase_c_ablations.py --mode smoke

# 4. Systems Smoke: Flash-GMM 2d-GEMM vs naive, Woodbury vs Cholesky & memory benchmark
python experiments/exp_04_kernel_and_memory_bench.py --mode smoke

# 5. Compile all JSON telemetry & hypothesis verdicts into FINAL-EXPERIMENT-REPORT.md
python experiments/compile_experiment_report.py
```

### Option B: Full GPU Cluster Execution (`1–2× A100/H100 80GB`, `200–300 GPU-Hours`)
```bash
# 0. Verify cluster GPUs, CUDA, Triton, RAM, and local dataset memmaps before spending hours
python experiments/exp_00_preflight_and_resource_audit.py --data-dir /data/egmm --strict

# 1. Phase A (~80 GPU-hrs): Full scaling-law sweep on DEEP + StretchedGaussian + OOD transfer
python experiments/exp_01_phase_a_scaling_laws.py --mode full --backend torch --data-dir /data/egmm --seeds 5

# 2. Phase B (~120 GPU-hrs): SIFT1M, DEEP1B, DINOv2-100M, ImageNet-1k vs FullBatch/OnlineEM/e-means+PH
python experiments/exp_02_phase_b_billion_benchmarks.py --mode full --backend torch --data-dir /data/egmm --seeds 5

# 3. Phase C (~50 GPU-hrs): Full multi-seed ablations (spectral splits, lambda_0, low-rank r, top-m, slow-ema)
python experiments/exp_03_phase_c_ablations.py --mode full --backend torch --data-dir /data/egmm --seeds 5

# 4. Systems Profiling (~5 GPU-hrs): H100/A100 Flash-GMM & Triton kernel throughput vs naive & Cholesky
python experiments/exp_04_kernel_and_memory_bench.py --mode full --backend torch

# 5. Compile final manuscript tables & hypothesis verdicts
python experiments/compile_experiment_report.py
```
*(Pass `--dry-run` to any `exp_01`–`exp_04` script to print exact sweep counts, VRAM checks, and estimated GPU-hours without running training.)*
