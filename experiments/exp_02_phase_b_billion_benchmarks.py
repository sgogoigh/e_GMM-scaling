"""
Experiment 02 (Phase B): tuned, matched, multi-seed benchmark of e-GMM against its baselines
=============================================================================================

Rewritten 2026-11-25 (docs/review/2026-11-25-REVIEW.md). Every method gets the same split, covariance
type, additive covariance floor (1e-4 after scale normalization), batch size and epoch budget, and
its own step/decay hyper-parameters are tuned on a validation split (e_gmm.benchmark). Results are
paired by seed against tuned e-GMM with spectral revival and tested with an exact sign-flip test.

Real data never falls back to synthetic data and is never truncated: a missing file raises, rows are
sampled at random (held-out rows that duplicate a training row are replaced), and a lower dimension
is only used when --project-dim asks for a PCA projection fitted on the training split.

HYPOTHESES (experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md):
  H0: tuned e-GMM without revival is as good as tuned e-GMM with revival.
  H3: tuned e-GMM beats tuned online EM, e-means->EM and full-batch EM at a matched budget.
  H4: e-GMM's advantage over e-means+post-hoc covariance is larger on overlapping/rotated regimes.

HOW TO RUN:
  python experiments/exp_02_phase_b_billion_benchmarks.py --dry-run
  python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --seeds 10
  python experiments/exp_02_phase_b_billion_benchmarks.py --mode smoke --data real --data-dir /data/egmm --project-dim 64
  python experiments/exp_02_phase_b_billion_benchmarks.py --mode full --data real --data-dir /data/egmm --backend torch

  Real files are looked up by stem in --data-dir with any of .npy/.fvecs/.bvecs/.fbin/.u8bin/.i8bin:
  sift_train, deep_train, dinov2_100m_train, imagenet1k_train (the names exp_00 audits). Only the
  *_train file is read; validation and test rows are drawn from it, disjoint and deduplicated.
  --backend torch is NOT tested locally.

SAVED ARTIFACTS:
  - Results : experiments/results/exp_02_phase_b_benchmarks.json
  - Log     : experiments/logs/exp_02_phase_b/latest.log
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.benchmark import default_methods, run_benchmark  # noqa: E402
from e_gmm.datasets import (  # noqa: E402
    find_vector_file,
    generate_gaussian_mixture,
    generate_stretched_gaussian,
    load_vectors,
    normalize_scale,
    pca_project,
    random_splits,
)
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402

# name, cov_type, n_train, n_val, n_test, d, k, B, epochs, generator kwargs
SYNTHETIC_SUITES = {
    "smoke": [
        ("Zipf-Separated-Diag", "diag", 2400, 800, 800, 16, 16, 64, 5,
         dict(kind="gmm", zipf=True, zipf_power=1.2, anisotropic=True, rotated=False, sigma_mu_sq=2 / 3, sigma_w_sq=1 / 3)),
        ("Zipf-Rotated-Full", "full", 2400, 800, 800, 16, 16, 64, 6,
         dict(kind="gmm", zipf=True, zipf_power=1.2, anisotropic=True, rotated=True, sigma_mu_sq=2 / 3, sigma_w_sq=1 / 3)),
        ("Overlapping-Rotated-Full", "full", 2400, 800, 800, 16, 16, 64, 6,
         dict(kind="gmm", zipf=False, anisotropic=True, rotated=True, sigma_mu_sq=0.20, sigma_w_sq=0.80)),
        ("Stretched-PowerLaw-Diag", "diag", 2400, 800, 800, 24, 16, 64, 5, dict(kind="stretched", exponent=1.5)),
    ],
    "full": [
        ("Zipf-Rotated-Full-L", "full", 200_000, 20_000, 20_000, 64, 256, 2048, 6,
         dict(kind="gmm", zipf=True, zipf_power=1.2, anisotropic=True, rotated=True, sigma_mu_sq=2 / 3, sigma_w_sq=1 / 3)),
        ("Overlapping-Rotated-Full-L", "full", 200_000, 20_000, 20_000, 64, 256, 2048, 6,
         dict(kind="gmm", zipf=False, anisotropic=True, rotated=True, sigma_mu_sq=0.20, sigma_w_sq=0.80)),
    ],
}
# name, file stem, cov_type, n_train, n_val, n_test, k, B, epochs
REAL_SUITES = {
    "smoke": [
        ("SIFT-Diag", "sift_train", "diag", 5000, 1000, 1000, 32, 256, 4),
        ("DEEP-Diag", "deep_train", "diag", 5000, 1000, 1000, 32, 256, 4),
        ("DINOv2-Full", "dinov2_100m_train", "full", 5000, 1000, 1000, 16, 256, 4),
        ("ImageNet1k-Full", "imagenet1k_train", "full", 5000, 1000, 1000, 16, 256, 4),
    ],
    "full": [
        ("SIFT-1M-Diag", "sift_train", "diag", 900_000, 50_000, 50_000, 1024, 4096, 8),
        ("DEEP-1M-Diag", "deep_train", "diag", 900_000, 50_000, 50_000, 1024, 4096, 5),
        ("DINOv2-Full", "dinov2_100m_train", "full", 250_000, 20_000, 20_000, 512, 2048, 8),
        ("ImageNet1k-Full", "imagenet1k_train", "full", 250_000, 20_000, 20_000, 1000, 2048, 8),
    ],
}
EGMM_STRATEGIES = {"egmm_spectral": "spectral_v1", "egmm_no_revive": "none", "egmm_faiss_alt": "faiss_alt"}


def parse_args():
    p = argparse.ArgumentParser(description="Exp 02 (Phase B): tuned, matched, paired benchmark")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None, help="Directory with the real embedding files (required with --data real).")
    p.add_argument("--project-dim", type=int, default=None, help="Real data: PCA-project to this d (fitted on train). Default: native d.")
    p.add_argument("--seeds", type=int, default=10, help="Evaluation seeds (paired); the verdict rule needs >= 10.")
    p.add_argument("--epochs", type=int, default=None, help="Override every suite's epoch budget (H3 is stated for T in {1,2,3}).")
    p.add_argument("--suites", type=str, default=None, help="Comma-separated subset of suite names.")
    p.add_argument("--skip-methods", type=str, default="", help="Comma-separated method names to leave out.")
    p.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--out", type=str, default=str(RESULTS_ROOT / "exp_02_phase_b_benchmarks.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def synthetic_split_fn(n_tr, n_val, n_te, d, k, spec):
    def make(seed):
        n = n_tr + n_val + n_te
        if spec["kind"] == "gmm":
            kw = {key: v for key, v in spec.items() if key != "kind"}
            X, _, _ = generate_gaussian_mixture(n=n, d=d, num_true_clusters=k, seed=seed, **kw)
        else:
            X, _ = generate_stretched_gaussian(n=n, d=d, exponent=spec["exponent"], rotate=True, seed=seed)
        (tr, va, te), _ = normalize_scale(X[:n_tr], X[n_tr : n_tr + n_val], X[n_tr + n_val :])
        return tr, va, te

    return make


def real_split_fn(arr, n_tr, n_val, n_te, project_dim, info):
    def make(seed):
        (tr, va, te), split_info = random_splits(arr, [n_tr, n_val, n_te], seed=seed)
        info.setdefault("dups_replaced", {})[str(seed)] = split_info["dups_replaced"]
        if project_dim is not None:
            (tr, va, te), pinfo = pca_project(tr, project_dim, va, te)
            info.setdefault("pca_explained_variance", {})[str(seed)] = pinfo["explained_variance_ratio"]
        (tr, va, te), _ = normalize_scale(tr, va, te)
        return tr, va, te

    return make


def main():
    args = parse_args()
    if args.data == "real":
        if not args.data_dir:
            raise SystemExit("--data real needs --data-dir (no synthetic fallback).")
        suites = [("real",) + s for s in REAL_SUITES[args.mode]]
    else:
        suites = [("synthetic",) + s for s in SYNTHETIC_SUITES[args.mode]]
    if args.suites:
        wanted = set(args.suites.split(","))
        suites = [s for s in suites if s[1] in wanted]
        if not suites:
            raise SystemExit(f"No suite matches {sorted(wanted)}.")
    seeds = list(range(1, args.seeds + 1))
    skip = {m for m in args.skip_methods.split(",") if m}

    print(f"Exp 02 | mode={args.mode} data={args.data} backend={args.backend} seeds={seeds}")
    print(f"  suites: {[s[1] for s in suites]}; skipped methods: {sorted(skip)}")
    if args.dry_run:
        if args.data == "real":
            for s in suites:
                print(f"  {s[1]}: {find_vector_file(args.data_dir, s[2])}")
        print("Dry run: nothing fitted.")
        return

    with tee_experiment_logs("exp_02_phase_b", run_tag=f"{args.mode}_{args.data}_{args.backend}"):
        t_start = time.perf_counter()
        results = {}
        for suite in suites:
            if suite[0] == "real":
                _, name, stem, cov_type, n_tr, n_val, n_te, k, B, epochs = suite
                path = find_vector_file(args.data_dir, stem)
                arr = load_vectors(str(path))
                info = {"source": str(path), "native_dim": int(arr.shape[1]), "rows_in_file": int(arr.shape[0]),
                        "project_dim": args.project_dim}
                if args.project_dim is not None and args.project_dim > arr.shape[1]:
                    raise SystemExit(f"{path} has d={arr.shape[1]} < --project-dim {args.project_dim}.")
                make_split = real_split_fn(arr, n_tr, n_val, n_te, args.project_dim, info)
                d = args.project_dim or int(arr.shape[1])
            else:
                _, name, cov_type, n_tr, n_val, n_te, d, k, B, epochs, spec = suite
                info = {"source": "synthetic", "generator": spec}
                make_split = synthetic_split_fn(n_tr, n_val, n_te, d, k, spec)

            methods, skipped = default_methods()
            methods = [m for m in methods if m.name not in skip]
            skipped.update({m: "skipped by --skip-methods" for m in skip})
            if args.backend == "torch":
                from e_gmm.benchmark import torch_egmm_runner

                for m in methods:
                    if m.name in EGMM_STRATEGIES:
                        m.run = torch_egmm_runner(EGMM_STRATEGIES[m.name])

            if args.epochs is not None:
                epochs = args.epochs
            print(f"\n[{name}] cov={cov_type} n_train={n_tr} d={d} k={k} B={B} epochs={epochs}")
            t0 = time.perf_counter()
            res = run_benchmark(make_split, k=k, cov_type=cov_type, batch_size=B, epochs=epochs, seeds=seeds,
                                methods=methods, skipped=skipped, verbose=False)
            res["data"] = info
            res["dim"] = d
            res["seconds"] = time.perf_counter() - t0
            results[name] = res

            print(f"  {'method':22s} {'NLL/d mean':>11s} {'diff vs ref':>12s} {'p':>7s} {'wins':>5s} {'dead':>5s}  tuned")
            for mname, summ in res["summary"].items():
                cmp = res["paired_vs_reference"].get(mname, {})
                diff = f"{cmp['mean_diff']:+.5f}" if "mean_diff" in cmp else "ref"
                pval = f"{cmp['p_value']:.4f}" if "p_value" in cmp else ""
                wins = f"{int(cmp['wins'])}/{int(cmp['n'])}" if "wins" in cmp else ""
                dead = f"{summ['dead_count_mean']:.1f}" if summ["dead_count_mean"] is not None else "-"
                mean = f"{summ['test_nll_per_dim_mean']:.5f}" if summ["test_nll_per_dim_mean"] is not None else "nan"
                edge = res["tuning"][mname]["at_grid_edge"]
                print(f"  {mname:22s} {mean:>11s} {diff:>12s} {pval:>7s} {wins:>5s} {dead:>5s}  "
                      f"{res['tuning'][mname]['best_params']}{' (grid edge: ' + ','.join(edge) + ')' if edge else ''}")

        payload = {
            "experiment_id": "exp_02_phase_b_billion_benchmarks",
            "mode": args.mode,
            "data": args.data,
            "backend": args.backend,
            "backend_tested_locally": args.backend == "numpy",
            "seeds": seeds,
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
            "suites": results,
        }
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)),
                       encoding="utf-8")
        print(f"\nSaved -> {out} ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
