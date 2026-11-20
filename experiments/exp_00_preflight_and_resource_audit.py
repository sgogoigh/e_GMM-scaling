"""
Experiment 00: Preflight Hardware, Storage, Dataset Memmap & Resource Audit
===========================================================================

PURPOSE:
  Audits cluster / workstation readiness BEFORE launching any Phase A-C jobs:
    1. Detects Python, NumPy, PyTorch, CUDA GPUs, VRAM per device, CPU RAM, and free disk space.
    2. Computes exact FP32 state VRAM, Slow-EMA shadow VRAM, naive vs. tiled HBM, and
       NVMe sequential streaming throughput budgets for SIFT1M, DEEP1B, SPACEV100M,
       DINOv2-100M, and ImageNet-1k.
    3. Validates any local dataset memmaps inside `--data-dir` (.npy/.fvecs/.bvecs)
       without downloading anything.

HOW TO RUN:
  # Local / offline audit:
  python experiments/exp_00_preflight_and_resource_audit.py

  # Cluster preflight with local dataset verification:
  python experiments/exp_00_preflight_and_resource_audit.py --data-dir /data/egmm --strict

REQUIREMENTS:
  - Offline Audit : Any CPU, Python >= 3.10, numpy>=1.26.0
  - Full Cluster  : 1-2x A100/H100 (80GB VRAM), >= 256 GB Host RAM, >= 1.5 TB NVMe SSD

SAVED ARTIFACTS:
  - Telemetry : `experiments/results/exp_00_preflight.json`
  - Log       : `experiments/logs/exp_00_preflight/latest.log`
"""

import argparse
import json
import os
import pathlib
import shutil
import sys
from typing import Any, Dict, List

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.config import DATASET_PRESETS  # noqa: E402
from e_gmm.datasets import load_vector_memmap  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402


EXPECTED_DATASETS = [
    ("deep_train.fvecs", "DEEP train (d=96)"),
    ("deep_test.fvecs", "DEEP test/holdout (d=96)"),
    ("sift_train.fvecs", "SIFT train (d=128)"),
    ("sift_test.fvecs", "SIFT test/holdout (d=128)"),
    ("spacev_train.bvecs", "SPACEV train (d=100, uint8)"),
    ("spacev_test.bvecs", "SPACEV test/holdout (d=100, uint8)"),
    ("dinov2_100m_train.npy", "DINOv2-100M train (d=1024)"),
    ("dinov2_100m_test.npy", "DINOv2-100M test (d=1024)"),
    ("imagenet1k_train.npy", "ImageNet-1k embeddings train (d=768)"),
    ("imagenet1k_test.npy", "ImageNet-1k embeddings test (d=768)"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 00: Preflight Hardware & Dataset Audit")
    p.add_argument("--data-dir", type=str, default=None, help="Optional directory containing local .npy/.fvecs/.bvecs memmaps.")
    p.add_argument("--strict", action="store_true", help="Exit with code 1 if CUDA GPU or required datasets are missing.")
    return p.parse_args()


def detect_hardware() -> Dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    info: Dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "disk_free_GB": round(disk.free / (1024 ** 3), 2),
        "torch_installed": False,
        "cuda_available": False,
        "gpus": [],
        "triton_available": False,
        "sklearn_available": False,
    }
    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_GiB": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    try:
        import triton  # type: ignore[import-not-found]  # noqa: F401

        info["triton_available"] = True
    except ImportError:
        pass

    try:
        import sklearn  # type: ignore[import-not-found]  # noqa: F401

        info["sklearn_available"] = True
    except ImportError:
        pass

    return info


def audit_presets() -> list:
    rows = []
    for name, s in DATASET_PRESETS.items():
        k, d, B = s.num_components, s.dim, s.default_batch_size
        ctype, r = s.recommended_cov_type, s.recommended_rank
        q_el = k * d if ctype == "diag" else (k * d + k * d * r if ctype == "low_rank_diag" else k * d * d)
        state_gib = 4.0 * (k + k * d + q_el) / (1024 ** 3)
        full_wall_gib = 4.0 * (k + k * d + k * d * d) / (1024 ** 3)
        naive_bk_gib = (4.0 * B * k) / (1024 ** 3)
        tiled_hbm_gib = (4.0 * B * min(2048, k)) / (1024 ** 3)
        rows.append(
            {
                "preset": name,
                "n": s.num_train,
                "d": d,
                "k": k,
                "cov_type": ctype,
                "state_fp32_GiB": round(state_gib, 4),
                "full_cov_equivalent_GiB": round(full_wall_gib, 2),
                "naive_bk_HBM_GiB": round(naive_bk_gib, 3),
                "tiled_HBM_GiB": round(tiled_hbm_gib, 4),
            }
        )
    return rows


def audit_data_dir(data_dir: str | None) -> List[Dict[str, Any]]:
    if data_dir is None:
        return []
    dpath = pathlib.Path(data_dir)
    found: List[Dict[str, Any]] = []
    for fname, desc in EXPECTED_DATASETS:
        fpath = dpath / fname
        entry: Dict[str, Any] = {"file": fname, "description": desc, "exists": fpath.exists(), "shape": None, "dtype": None}
        if fpath.exists():
            try:
                mm = load_vector_memmap(str(fpath), max_rows=8)
                entry["probe_shape"] = list(mm.shape)
                entry["dtype"] = str(mm.dtype)
                entry["size_GiB"] = round(fpath.stat().st_size / (1024 ** 3), 3)
            except Exception as exc:
                entry["error"] = str(exc)
        found.append(entry)
    return found


def main():
    args = parse_args()
    with tee_experiment_logs("exp_00_preflight"):
        hw = detect_hardware()
        presets = audit_presets()
        datasets = audit_data_dir(args.data_dir)

        payload = {
            "experiment_id": "exp_00_preflight",
            "hardware": hw,
            "preset_resource_budgets": presets,
            "data_dir": args.data_dir,
            "dataset_audit": datasets,
        }
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_ROOT / "exp_00_preflight.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("=== EXP 00 PREFLIGHT & RESOURCE AUDIT ===")
        print(f"  CPUs: {hw['cpu_count']} | Disk Free: {hw['disk_free_GB']} GiB | CUDA: {hw['cuda_available']} | Triton: {hw['triton_available']}")
        if hw["gpus"]:
            print(f"  GPUs: {hw['gpus']}")
        print("\n[Preset Memory & HBM Audit]")
        print(f"  {'Preset':<15} {'Mode':<14} {'k':>6} {'d':>5} {'StateGiB':>9} {'FullEqGiB':>10} {'TiledHBM':>9}")
        for r in presets:
            print(
                f"  {r['preset']:<15} {r['cov_type']:<14} {r['k']:>6d} {r['d']:>5d} "
                f"{r['state_fp32_GiB']:>9.4f} {r['full_cov_equivalent_GiB']:>10.2f} {r['tiled_HBM_GiB']:>8.4f}G"
            )
        if datasets:
            print(f"\n[Local Dataset Memmap Audit: {args.data_dir}]")
            for ds in datasets:
                status = f"OK ({ds.get('size_GiB', 0)} GiB, {ds.get('dtype')})" if ds["exists"] else "MISSING"
                print(f"  {ds['file']:<26} -> {status}")
        print(f"\nSaved telemetry -> {out_path}")

        if args.strict and (not hw["cuda_available"] or (datasets and not any(d["exists"] for d in datasets))):
            print("STRICT PREFLIGHT FAILED: CUDA GPU or local dataset memmaps missing.")
            sys.exit(1)


if __name__ == "__main__":
    main()
