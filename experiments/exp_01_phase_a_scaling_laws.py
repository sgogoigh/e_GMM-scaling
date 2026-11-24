"""
Experiment 01 (Phase A): can a d-aware alpha_0 / B_knee law be identified, and does the d term help?
====================================================================================================

Rewritten 2026-11-25 (docs/review/2026-11-25-REVIEW.md, findings on the scaling-law evidence). The
sweep is e_gmm.sweep.run_sweep:
  - T is varied, so a_T is identifiable (it was collinear with the intercept when T was fixed).
  - alpha_0 is searched on an adaptive ladder; optima on the ladder edge are flagged as censored and
    excluded from the fit. Knees are searched up to n/4; a knee at the top is flagged as censored.
  - d varies within ONE data source: a 32-cluster anisotropic synthetic mixture, or PCA projections
    of one real embedding file. Nothing falls back to synthetic data and nothing is truncated.
  - The law is fitted with no d term, with ambient d and with intrinsic (TwoNN) d. H1 is judged by
    held-out NLL regret (paired leave-one-d-out and out-of-distribution configs), not by R^2.

HYPOTHESES (experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md):
  H1: adding a d term lowers held-out alpha_0 regret versus the same law without d.
  H2: a law fitted on the grid transfers to unseen (d, n) configurations with small regret.

HOW TO RUN:
  python experiments/exp_01_phase_a_scaling_laws.py --dry-run
  python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --seeds 2
  python experiments/exp_01_phase_a_scaling_laws.py --mode smoke --data real --data-file /data/dinov2_768.npy
  python experiments/exp_01_phase_a_scaling_laws.py --mode full --backend torch --data real \
      --data-file /data/dinov2_768.fbin --manifest configs/phase_a_scaling_manifest.json --seeds 3

  --backend torch has NOT been executed on the review machine (torch not installed).
  Full mode takes its axes (n, d, k, B, T) from the manifest; every d must be <= the source dimension.

SAVED ARTIFACTS:
  - Results : experiments/results/exp_01_phase_a_scaling_laws.json
  - Log     : experiments/logs/exp_01_phase_a/latest.log
"""

import argparse
import json
import pathlib
import sys
import time

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.datasets import load_vectors  # noqa: E402
from e_gmm.sweep import SweepGrid, real_data_fn, run_sweep, synthetic_data_fn  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402

SMOKE_GRID = dict(
    dims=[8, 16, 32, 64], ks=[4, 8, 16], ns=[1200, 4800], batch_sizes=[32, 128], epochs=[2, 4],
    n_val=1000, knee_b_min=16,
    ood=[{"d": 128, "k": 8, "n": 4800, "B": 64, "T": 3}, {"d": 24, "k": 12, "n": 2400, "B": 64, "T": 3}],
)


def parse_args():
    p = argparse.ArgumentParser(description="Exp 01 (Phase A): identifiable d-aware scaling-law sweep")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-file", type=str, default=None, help="Embedding file (.npy/.fvecs/.bvecs/.fbin/.u8bin/.i8bin); required with --data real.")
    p.add_argument("--manifest", type=str, default=str(ROOT / "configs" / "phase_a_scaling_manifest.json"),
                   help="Full mode: JSON list of {num_train, dim, num_components, batch_size, epochs}; the axes are its unique values.")
    p.add_argument("--dims", type=str, default=None, help="Comma-separated override of the d axis.")
    p.add_argument("--dim-feature", choices=["ambient", "intrinsic"], default="ambient",
                   help="Which d-law is the primary H1 candidate in the summary (both are always fitted).")
    p.add_argument("--seeds", type=int, default=2, help="Seeds averaged per point of every curve.")
    p.add_argument("--cov-type", choices=["diag", "full"], default="diag")
    p.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--out", type=str, default=str(RESULTS_ROOT / "exp_01_phase_a_scaling_laws.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_grid(args) -> SweepGrid:
    if args.mode == "smoke":
        g = dict(SMOKE_GRID)
    else:
        rows = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        axis = lambda key: sorted({int(r[key]) for r in rows})  # noqa: E731
        ns = axis("num_train")
        g = dict(dims=axis("dim"), ks=axis("num_components"), ns=ns, batch_sizes=axis("batch_size"),
                 epochs=axis("epochs"), n_val=20_000, knee_b_min=256,
                 ood=[{"d": axis("dim")[-1], "k": axis("num_components")[1], "n": ns[-1] * 2,
                       "B": axis("batch_size")[1], "T": 3}])
    if args.dims:
        g["dims"] = [int(x) for x in args.dims.split(",")]
    return SweepGrid(seeds=list(range(args.seeds)), cov_type=args.cov_type, **g)


def main():
    args = parse_args()
    grid = build_grid(args)
    all_dims = sorted(set(grid.dims) | {int(o["d"]) for o in grid.ood})

    if args.data == "real":
        if not args.data_file:
            raise SystemExit("--data real needs --data-file (no synthetic fallback).")
        arr = load_vectors(args.data_file)
        if max(all_dims) > arr.shape[1]:
            raise SystemExit(f"{args.data_file} has d={arr.shape[1]} but the grid/OOD needs d up to {max(all_dims)}; "
                             "pass --dims or a higher-dimensional source (no truncation, no fallback).")
        need = max(grid.ns + [int(o["n"]) for o in grid.ood]) + grid.n_val
        if arr.shape[0] < need:
            raise SystemExit(f"{args.data_file} has {arr.shape[0]} rows but the grid needs {need}.")
        data_fn, source = real_data_fn(arr), f"real:{pathlib.Path(args.data_file).name} (PCA to d, scale-normalized)"
    else:
        data_fn, source = synthetic_data_fn(), "synthetic: 32-cluster anisotropic rotated Zipf mixture, scale-normalized"

    runner = None
    if args.backend == "torch":
        from e_gmm.benchmark import torch_egmm_runner

        runner = torch_egmm_runner()

    n_ladders = sum(1 for d in grid.dims for n in grid.ns for k in grid.ks for _ in grid.epochs for B in grid.batch_sizes
                    if B <= n // 2 and n >= 10 * k)
    print(f"Exp 01 | mode={args.mode} data={source} backend={args.backend} seeds={len(grid.seeds)}")
    print(f"  axes: d={grid.dims} k={grid.ks} n={grid.ns} B={grid.batch_sizes} T={grid.epochs}; {n_ladders} alpha ladders; OOD={grid.ood}")
    if args.dry_run:
        print("Dry run: nothing fitted.")
        return

    with tee_experiment_logs("exp_01_phase_a"):
        t0 = time.perf_counter()
        res = run_sweep(data_fn, grid, source=source, runner=runner, verbose=True)
        primary = "ambient_d" if args.dim_feature == "ambient" else "intrinsic_d"
        payload = {
            "experiment_id": "exp_01_phase_a_scaling_laws",
            "mode": args.mode,
            "data": args.data,
            "data_file": args.data_file,
            "backend": args.backend,
            "backend_tested_locally": args.backend == "numpy",
            "primary_d_law": primary,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            **res,
        }
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

        print("=== EXP 01 COMPLETE ===")
        c = res["censoring"]
        print(f"  censored alpha optima: {c['alpha_censored']}/{c['alpha_total']}; censored knees: {c['knee_censored']}/{c['knee_total']}")
        for name, law in res["alpha_laws"].items():
            if "error" in law:
                print(f"  alpha law [{name}]: not fitted ({law['error']})")
                continue
            f = law["fit"]
            ex = {k: f[k] for k in ("a_k", "a_b", "a_n", "a_t", "a_d")}
            print(f"  alpha law [{name}]: {ex} unidentified={f['unidentified']} R2={f['r2_log']} "
                  f"LODO regret/dim={law['lodo_d'].get('lodo_regret_nats_per_dim_mean')}")
        for key, cmp in res["lodo_paired_regret"].items():
            print(f"  LODO paired {key}: {cmp}")
        for key, cmp in res["ood"].get("paired_vs_no_d", {}).items():
            print(f"  OOD paired {key}: mean_diff={cmp['mean_diff']:+.5f} p={cmp['p_value']:.3f}")
        print(f"Saved -> {out} ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
