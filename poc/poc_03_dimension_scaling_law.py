"""
PoC 3: does a dimension term improve alpha_0 / B_knee prediction? (synthetic sweep, CPU)
=======================================================================================

Uses `e_gmm.sweep.run_sweep` on a small synthetic grid that varies d, k, n, B AND T (epochs), so every exponent of
alpha_0(k, B, n, T, d) is identifiable in principle. Alpha ladders are censored when the optimum sits on the
ladder boundary; censored points are counted and kept out of the "identified" claims. Laws are fitted with no d,
ambient d and intrinsic d (TwoNN). The judge is paired leave-one-d-out (LODO) regret and zero-shot out-of-distribution
(OOD) regret, d-aware vs no-d, never R^2. OOD configs include a d outside the training grid.

The grid is deliberately small so the script runs in minutes; it is recorded in the JSON ("grid", "shrunk_note").
Absence of evidence on a small grid is Inconclusive, not support.
"""

import argparse
import json
import pathlib
import sys
import time

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.poc_logging import tee_poc_logs  # noqa: E402
from e_gmm.sweep import SweepGrid, run_sweep, synthetic_data_fn  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PoC 3: dimension-aware scaling-law sweep with paired LODO/OOD regret")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--ns", type=int, nargs="+", default=[1200, 4800])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--epochs", type=int, nargs="+", default=[2, 4])
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--out", type=str, default=str(POC_DIR / "results" / "poc_03_results.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else "sweep"
    with tee_poc_logs("poc_03", run_tag=tag):
        # OOD: d=24 and d=96 lie outside every trained d value; (d=16, k=12, n=2400, B=64, T=3) is an unseen combination.
        ood = [
            {"d": 24, "k": 12, "n": 2400, "B": 64, "T": 3},
            {"d": 96, "k": 8, "n": 2400, "B": 64, "T": 3},
            {"d": 16, "k": 12, "n": 2400, "B": 64, "T": 3},
        ]
        grid = SweepGrid(
            dims=args.dims, ks=args.ks, ns=args.ns, batch_sizes=args.batch_sizes, epochs=args.epochs,
            seeds=list(range(1, args.seeds + 1)), n_val=args.n_val, cov_type="diag", ood=ood,
        )
        n_cfg = len(grid.dims) * len(grid.ks) * len(grid.ns) * len(grid.batch_sizes) * len(grid.epochs)
        if args.dry_run:
            print(f"=== PoC 3 DRY RUN === {n_cfg} configs x ladder x {args.seeds} seeds; OOD={ood}")
            return
        t0 = time.perf_counter()
        res = run_sweep(synthetic_data_fn(), grid, source="synthetic", verbose=True)
        res.update({
            "poc_id": "poc_03",
            "title": "Dimension-aware alpha_0 / B_knee laws: censoring, paired LODO and OOD regret (synthetic)",
            "shrunk_note": "Small synthetic grid (see 'grid'), 2 seeds; chosen to keep the run under ~10 minutes.",
            "wall_seconds": time.perf_counter() - t0,
        })
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

        print("=== PoC 3 COMPLETE ===")
        print(f"  censoring: {res['censoring']}  ({res['wall_seconds']:.0f}s)")
        for name, law in res["alpha_laws"].items():
            f = law.get("fit")
            print(f"  alpha law {name}: " + (str({k: f.get(k) for k in ('a_k', 'a_b', 'a_n', 'a_t', 'a_d')}) if f else str(law)))
        print(f"  LODO paired regret: {res['lodo_paired_regret']}")
        print(f"  OOD paired: {res['ood'].get('paired_vs_no_d')}")
        print(f"Saved JSON -> {out}")


if __name__ == "__main__":
    main()
