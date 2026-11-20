"""
Phase-A Dimension-Aware Scaling-Law Sweep Planner & OLS / LOSO / LODO Verifier.
Generates `configs/phase_a_scaling_manifest.json` and demonstrates why including
d^{a_d} stabilizes NLL scaling-law prediction across dimensions.
"""

import itertools
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.scaling_law import (  # noqa: E402
    DimensionScalingObservation,
    evaluate_lodo_d,
    evaluate_loso_n,
    fit_dimension_aware_alpha0,
    fit_dimension_aware_batch_knee,
)


def main():
    grid_n = [100_000, 1_000_000, 10_000_000]
    grid_d = [32, 96, 256, 768]
    grid_k = [64, 512, 4096, 16384]
    grid_b = [4096, 16384, 65536]
    grid_t = [2.0, 3.0, 5.0]

    combos = [
        (n, d, k, b, t)
        for (n, d, k, b, t) in itertools.product(grid_n, grid_d, grid_k, grid_b, grid_t)
        if b <= n and (n // k) >= 8 and b >= k // 4
    ]

    manifest_path = ROOT / "configs" / "phase_a_scaling_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"num_train": n, "dim": d, "num_components": k, "batch_size": b, "epochs": t}
                for (n, d, k, b, t) in combos
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Planned Phase-A Dimension-Aware Scaling Matrix: {len(combos)} configurations -> {manifest_path}")

    # Verify OLS power-law recovery + LOSO/LODO ablation (with vs without d^{a_d})
    rng = np.random.default_rng(42)
    true_c, ak, ab, an, at, ad = 0.25, -0.44, 0.85, -0.42, -0.12, -0.28
    cb, bk, bn, bt, bd = 0.04, 0.14, 0.86, 0.78, 0.08
    obs = []
    for n, d, k, b, t in combos:
        a_true = min(true_c * (k ** ak) * (b ** ab) * (n ** an) * (t ** at) * (d ** ad), 0.99)
        b_true = cb * (k ** bk) * (n ** bn) * (t ** bt) * (d ** bd)
        noisy_a = float(np.clip(a_true * np.exp(rng.normal(scale=0.04)), 1e-5, 0.99))
        noisy_b = float(b_true * np.exp(rng.normal(scale=0.04)))
        obs.append(DimensionScalingObservation(n, k, d, b, t, noisy_a, b_knee_opt=noisy_b))

    fit_with_d = fit_dimension_aware_alpha0(obs, include_dim=True)
    fit_no_d = fit_dimension_aware_alpha0(obs, include_dim=False)
    lodo_with_d = evaluate_lodo_d(obs, include_dim=True)
    lodo_no_d = evaluate_lodo_d(obs, include_dim=False)
    loso_with_d = evaluate_loso_n(obs, include_dim=True)
    fit_b = fit_dimension_aware_batch_knee(obs, include_dim=True)

    print("\n[Dimension-Aware Scaling Law Recovery Verification]")
    print(
        f"  With d^{{a_d}} : C={fit_with_d.c_alpha:.3f}, k={fit_with_d.a_k:+.2f}, B={fit_with_d.a_b:+.2f}, "
        f"n={fit_with_d.a_n:+.2f}, T={fit_with_d.a_t:+.2f}, d={fit_with_d.a_d:+.2f} | "
        f"R^2={fit_with_d.r2_log:.4f} | LODO rel err={lodo_with_d['lodo_mean_rel_err']*100:.2f}% | "
        f"LOSO rel err={loso_with_d['loso_mean_rel_err']*100:.2f}%"
    )
    print(
        f"  No d (ablat): R^2={fit_no_d.r2_log:.4f} | LODO rel err={lodo_no_d['lodo_mean_rel_err']*100:.2f}% "
        "(demonstrates dimension extrapolation failure without d^{a_d})"
    )
    print(
        f"  Batch Knee B: C_B={fit_b.c_b:.4f}, k={fit_b.b_k:+.2f}, n={fit_b.b_n:+.2f}, "
        f"T={fit_b.b_t:+.2f}, d={fit_b.b_d:+.2f} | R^2={fit_b.r2_log:.4f}"
    )


if __name__ == "__main__":
    main()
