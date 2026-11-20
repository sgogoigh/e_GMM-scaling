# Proof-of-Concept (PoC) Suite — De-Risking Compute Requests

This folder holds the **4-Tier Proof-of-Concept (PoC) suite** designed to verify every theoretical, numerical, and systems claim of **`e-GMM`** using **minimal local resources** before launching the full **200–300 GPU-hour (`1–2× A100/H100 80GB`)** benchmark suite on `SIFT1M`, `DEEP1B`, `DINOv2-100M`, and `ImageNet-1k`.

> **Safe Defaults**: Scripts never download external datasets behind your back. All offline CPU/NumPy PoCs (`poc_01`, `poc_02`, `poc_03`, and `poc_04` NumPy/dry-run mode) run cleanly out of the box.

---

## PoC Tiers at a Glance

| Tier | Script | Target Hardware | Est. Runtime / Cost | Core Evidence Produced |
| :---: | :--- | :--- | :--- | :--- |
| **PoC 1** | [`poc_01_math_and_profiler.py`](./poc_01_math_and_profiler.py) | Local CPU (NumPy only) | `< 3s` • `\$0` | `< 1e-13` Flash-GMM & Woodbury equivalence + PSD floor proof + Spectral conservation + `256 GiB` full-cov memory wall audit |
| **PoC 2** | [`poc_02_collapse_and_spectral_split.py`](./poc_02_collapse_and_spectral_split.py) | Local CPU (or 1× GPU) | `~5–20s` • `\$0` | 8-way Anisotropic Zipf-GMM ablation (`OnlineEM beta=0.6/0.85` vs `e-means+PostHoc` vs `NoRevive` vs `BatchResample` vs `FaissAlt` vs `RandomOrth` vs `Spectral v1`) |
| **PoC 3** | [`poc_03_dimension_scaling_law.py`](./poc_03_dimension_scaling_law.py) | Local CPU (or 1× GPU) | `~10–30s` • `\$0` | Real empirical $(n, d, k, B)$ sweep fitting $\alpha_0(k, B, n, T, d)$ on `StretchedGaussian` + LODO-$d$ / LOSO-$n$ & OOD transfer to `Zipf-GMM` |
| **PoC 4** | [`poc_04_flash_gmm_and_lowrank.py`](./poc_04_flash_gmm_and_lowrank.py) | CPU (NumPy) or 1× GPU (PyTorch) | `~5–20s` • `\$0` | Top-$m$ sparsity ($m \in \{2,4,8,\text{full}\}$), Rotated Correlated GMM (`diag` vs `low_rank_diag` vs `full`), and `last_epoch` vs `slow_ema` Test NLL check |
| **Report** | [`compile_poc_report.py`](./compile_poc_report.py) | Local CPU (Stdlib only) | `< 1s` | Aggregates `poc/results/poc_0*.json` into `poc/results/COMPILED-POC-REPORT.md` |

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
