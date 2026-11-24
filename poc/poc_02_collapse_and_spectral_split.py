"""
PoC 2: matched, tuned benchmark of e-GMM (spectral revival) against baselines, plus an alpha_0 sensitivity block
================================================================================================================

Two regimes (anisotropic, rotated, full-covariance mixtures from `generate_gaussian_mixture`; d=16, k=16, B=64,
6 epochs, train 2400 / val 800 / test 800, `normalize_scale` fitted on the training split):
  Zipf    sigma_mu^2=2/3, sigma_w^2=1/3, Zipf power 1.2 (cluster imbalance / starvation)
  Overlap sigma_mu^2=0.2, sigma_w^2=0.8, equal weights (heavily overlapping tails)

Every method is tuned on a validation split drawn with a separate tuning seed (`e_gmm.benchmark.run_benchmark`),
uses the same covariance floor (1e-4) and the same batch/epoch budget, and is scored on test NLL in nats per
dimension, paired by seed against tuned e-GMM with spectral revival (`egmm_spectral`).

alpha_0 sensitivity block (H0: "revival is only needed because alpha_0 is too large"): for each alpha_0 in
{0.05, 0.1, 0.2, 0.35}, e-GMM with spectral revival vs the same e-GMM with revival off, paired by seed.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.benchmark import RunContext, _nll_per_dim, egmm_runner, run_benchmark  # noqa: E402
from e_gmm.datasets import generate_gaussian_mixture, normalize_scale  # noqa: E402
from e_gmm.poc_logging import tee_poc_logs  # noqa: E402
from e_gmm.stats import paired_comparison  # noqa: E402

REGIMES = {
    "zipf": dict(sigma_mu_sq=2.0 / 3.0, sigma_w_sq=1.0 / 3.0, zipf=True, zipf_power=1.2),
    "overlap": dict(sigma_mu_sq=0.2, sigma_w_sq=0.8, zipf=False, zipf_power=1.0),
}
SENS_ALPHAS = (0.05, 0.1, 0.2, 0.35)
COV_FLOOR = 1e-4


def parse_args():
    p = argparse.ArgumentParser(description="PoC 2: matched tuned benchmark + alpha_0 sensitivity")
    p.add_argument("--n-train", type=int, default=2400)
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--num-components", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", type=str, default=str(POC_DIR / "results" / "poc_02_results.json"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_split_fn(args, regime: dict):
    n_tot = args.n_train + args.n_val + args.n_test

    def make_split(seed: int):
        X, _, _ = generate_gaussian_mixture(
            n=n_tot, d=args.dim, num_true_clusters=args.num_components, anisotropic=True, rotated=True,
            seed=int(seed), **regime,
        )
        X = X[np.random.default_rng(int(seed) + 7919).permutation(n_tot)]  # no cluster-ordered slices
        tr, va, te = X[: args.n_train], X[args.n_train : args.n_train + args.n_val], X[args.n_train + args.n_val :]
        (tr, va, te), _ = normalize_scale(tr, va, te)
        return tr, va, te

    return make_split


def sensitivity(args, make_split, seeds) -> dict:
    """Spectral revival vs no revival at each fixed alpha_0, paired by seed."""
    out = {}
    for a in SENS_ALPHAS:
        rec = {"spectral": [], "no_revive": [], "dead_spectral": [], "dead_no_revive": []}
        for s in seeds:
            tr, _, te = make_split(s)
            ctx = RunContext(args.num_components, "full", COV_FLOOR, args.batch_size, args.epochs, s)
            for key, strat in (("spectral", "spectral_v1"), ("no_revive", "none")):
                fit = egmm_runner(strat)(tr, ctx, {"alpha_0": a})
                rec[key].append(_nll_per_dim(te, fit, "full"))
                rec["dead_" + key].append(float(fit["health"]["dead_count"]))
        out[f"{a:g}"] = {
            "alpha_0": a,
            "no_revive_vs_spectral": paired_comparison(rec["no_revive"], rec["spectral"]),
            "dead_spectral_mean": float(np.mean(rec["dead_spectral"])),
            "dead_no_revive_mean": float(np.mean(rec["dead_no_revive"])),
            "per_seed": rec,
        }
    return out


def main():
    args = parse_args()
    tag = "dry_run" if args.dry_run else "full"
    with tee_poc_logs("poc_02", run_tag=tag):
        seeds = list(range(1, args.seeds + 1))
        if args.dry_run:
            print(f"=== PoC 2 DRY RUN === regimes={list(REGIMES)} d={args.dim} k={args.num_components} seeds={seeds}")
            return
        t_all = time.perf_counter()
        payload = {
            "poc_id": "poc_02",
            "title": "Matched tuned baselines + alpha_0 sensitivity, Zipf and Overlap full-cov mixtures",
            "config": {k: v for k, v in vars(args).items()}, "seeds": seeds, "regimes": {},
        }
        for name, regime in REGIMES.items():
            print(f"[{name}] benchmark ...", flush=True)
            t0 = time.perf_counter()
            make_split = make_split_fn(args, regime)
            bench = run_benchmark(make_split, args.num_components, "full", args.batch_size, args.epochs, seeds,
                                  cov_floor=COV_FLOOR, verbose=True)
            print(f"[{name}] sensitivity ...", flush=True)
            sens = sensitivity(args, make_split, seeds)
            payload["regimes"][name] = {
                "generator": regime, "benchmark": bench, "alpha_sensitivity": sens,
                "seconds": time.perf_counter() - t0,
            }
            print(f"\n[{name}] test NLL nats/dim, paired vs egmm_spectral (mean_diff>0 means egmm_spectral better)")
            for m, c in bench["paired_vs_reference"].items():
                print(f"  {m:22s} n={c.get('n')} mean_diff={c.get('mean_diff', float('nan')):+.4f} "
                      f"p={c.get('p_value', float('nan')):.4f} dead={bench['summary'][m]['dead_count_mean']}")
            for a, r in sens.items():
                c = r["no_revive_vs_spectral"]
                print(f"  alpha_0={a:>5}: no_revive-spectral mean_diff={c['mean_diff']:+.4f} p={c['p_value']:.4f} "
                      f"dead(spectral/none)={r['dead_spectral_mean']:.1f}/{r['dead_no_revive_mean']:.1f}")
        payload["seconds"] = time.perf_counter() - t_all
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {out} ({payload['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
