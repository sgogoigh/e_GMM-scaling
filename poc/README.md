# Proof-of-Concept (PoC) Suite — De-Risking Compute Requests

This folder holds the **4-Tier Proof-of-Concept (PoC) suite** designed to verify every theoretical, numerical, and systems claim of **`e-GMM`** using **minimal local resources** before launching the full **200–300 GPU-hour (`1–2× A100/H100 80GB`)** benchmark suite on `SIFT1M`, `DEEP1B`, `DINOv2-100M`, and `ImageNet-1k`.

> **Safe Defaults**: Scripts never download external datasets behind your back. All offline CPU/NumPy PoCs (`poc_01`, `poc_02`, `poc_03`, and `poc_04` NumPy/dry-run mode) run cleanly out of the box.

---

## PoC Tiers at a Glance

| Tier | Script | Target Hardware | Est. Runtime / Cost | Core Evidence Produced |
| :---: | :--- | :--- | :--- | :--- |
| **PoC 1** | [`poc_01_math_and_profiler.py`](./poc_01_math_and_profiler.py) | Local CPU (NumPy only) | `< 3s` • `\$0` | `< 1e-13` Flash-GMM & Woodbury equivalence + PSD floor proof + Spectral conservation + `256 GiB` full-cov memory wall audit |
| **PoC 2** | [`poc_02_collapse_and_spectral_split.py`](./poc_02_collapse_and_spectral_split.py) | Local CPU | `~8-15 min` (10 seeds) | Matched, validation-tuned baselines (e-GMM alpha_0 grid, online EM beta x t0, e-means+PostHoc, e-means then 3 EM iters, full-batch EM random / e-means init) vs tuned e-GMM spectral revival on a Zipf and an Overlap full-cov mixture; paired test NLL in nats/dim; dead counts; alpha_0 sensitivity of revival vs no revival (alpha_0 in 0.05, 0.1, 0.2, 0.35) |
| **PoC 3** | [`poc_03_dimension_scaling_law.py`](./poc_03_dimension_scaling_law.py) | Local CPU | `~7 min` | Small synthetic sweep that varies d, k, n, B and T (`e_gmm.sweep.run_sweep`): alpha ladders with censoring counts, laws with no d / ambient d / intrinsic d (TwoNN), paired LODO and OOD regret (incl. d outside the grid and the k-means paper prior). Judged by paired regret, not R^2 |
| **PoC 4** | [`poc_04_flash_gmm_and_lowrank.py`](./poc_04_flash_gmm_and_lowrank.py) | CPU (NumPy) or 1× GPU (PyTorch) | `~5–20s` • `\$0` | Top-$m$ sparsity ($m \in \{2,4,8,\text{full}\}$), Rotated Correlated GMM (`diag` vs `low_rank_diag` vs `full`), and `last_epoch` vs `slow_ema` Test NLL check |
| **Report** | [`compile_poc_report.py`](./compile_poc_report.py) | Local CPU (stdlib + `e_gmm.stats`) | `< 1s` | Computes every verdict from the JSON (`stats.verdict`, 0.01 nats/dim, min_n 10) and aggregates `poc/results/poc_0*.json` into `poc/results/COMPILED-POC-REPORT.md` |

---

## Automatic Run Logging (`poc/logs/`)

Every PoC script automatically tees console output + system/compute headers into structured subfolders via [`poc_logging.py`](../src/e_gmm/poc_logging.py):
- `poc/logs/poc_01/run_<YYYYMMDD_HHMMSS>.log` & `latest.log`
- `poc/logs/poc_02/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/poc_03/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/poc_04/run_<YYYYMMDD_HHMMSS>[_dry_run].log` & `latest.log`
- `poc/logs/compile_report/run_<YYYYMMDD_HHMMSS>.log` & `latest.log`

---

## Step-by-Step Execution Workflow

```bash
# Run PoCs 1-4 and compile the evidence report
python poc/poc_01_math_and_profiler.py
python poc/poc_02_collapse_and_spectral_split.py
python poc/poc_03_dimension_scaling_law.py
python poc/poc_04_flash_gmm_and_lowrank.py
python poc/compile_poc_report.py
```

On a GPU machine with `torch` installed, run PoC 4 with PyTorch Flash-GMM benchmarking:
```bash
python poc/poc_04_flash_gmm_and_lowrank.py --backend torch
python poc/compile_poc_report.py
```

Then pair `poc/results/COMPILED-POC-REPORT.md` with [`PROPOSAL-TEMPLATE.md`](./PROPOSAL-TEMPLATE.md).

## How to read the claims

- Verdicts are computed in `compile_poc_report.py` from the stored paired results via `e_gmm.stats.verdict`
  (threshold 0.01 nats/dim, at least 10 paired observations, exact sign-flip p < 0.05). A missing result reads `Not run`.
  No multiple-comparison correction is applied.
- PoC 2 compares every method after tuning on validation data with the same covariance floor (1e-4), batch size and epochs.
  The alpha_0 sensitivity block tests whether revival still helps when alpha_0 is small.
- PoC 3 uses a small synthetic grid (see `shrunk_note` in its JSON), so censored ladder optima and unidentified exponents are
  reported rather than hidden. A scaling law is judged by whether the d-aware law has lower paired LODO/OOD regret than the
  no-d law. OOD has only 3 configs, so it cannot be Supported on its own.
- PoC 1 is deterministic identity checks and arithmetic. PoC 4 is single-seed: its claims are `Untested`, shown descriptively.
- `scripts/run_preliminary_diagnostics.py` is a single-seed, untuned diagnostic with matched floors (1e-4); only its floor
  sweep changes the floor on purpose. It is not a method comparison.
- Everything is synthetic data on CPU. No claim about real embeddings or GPU speed follows.

Run `python poc/poc_02_collapse_and_spectral_split.py --seeds 10 --out <path>` to change the seed count or output file.
