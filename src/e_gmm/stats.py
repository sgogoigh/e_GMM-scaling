"""Paired statistics for multi-seed method comparisons (no SciPy dependency).

Every comparison in the experiment reports is paired by seed: method A and method B
see the same data split for a given seed. The test used is the two-sided sign-flip
permutation test on the per-seed differences, which is exact for up to 16 seeds and
makes no normality assumption.

Verdict rule used by the report compilers (see experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md):
    claim "A is better than B" (lower NLL), diff = A - B per seed
    - Supported:     p < alpha and mean(diff) <= -threshold
    - Contradicted:  p < alpha and mean(diff) > 0
    - Inconclusive:  everything else (including significant but below the practical threshold)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

EXACT_MAX_N = 16


def sign_flip_p_value(diffs: Sequence[float], n_mc: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for H0: mean(diff) = 0."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n == 0:
        return float("nan")
    observed = abs(float(d.mean()))
    if observed == 0.0:
        return 1.0
    tol = 1e-12 * max(1.0, observed)
    if n <= EXACT_MAX_N:
        codes = np.arange(2 ** n, dtype=np.int64)[:, None]
        signs = 1.0 - 2.0 * ((codes >> np.arange(n, dtype=np.int64)[None, :]) & 1)
    else:
        # Monte Carlo: (hits + 1) / (n_mc + 1) so the estimate is never exactly 0.
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, n))
        null = np.abs(signs @ d) / n
        return float((np.sum(null >= observed - tol) + 1) / (n_mc + 1))
    null = np.abs(signs @ d) / n
    return float(np.mean(null >= observed - tol))


def paired_comparison(
    method: Sequence[float],
    reference: Sequence[float],
    lower_is_better: bool = True,
) -> Dict[str, float]:
    """Summarize per-seed paired differences `method - reference`."""
    a = np.asarray(method, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"Paired comparison needs equal-length 1-D arrays, got {a.shape} vs {b.shape}.")
    diff = a - b
    n = int(diff.size)
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    t_stat = float(diff.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    better = diff < 0 if lower_is_better else diff > 0
    return {
        "n": float(n),
        "method_mean": float(a.mean()) if n else float("nan"),
        "reference_mean": float(b.mean()) if n else float("nan"),
        "mean_diff": float(diff.mean()) if n else float("nan"),
        "std_diff": sd,
        "t_stat": t_stat,
        "p_value": sign_flip_p_value(diff),
        "wins": float(np.sum(better)),
        "ties": float(np.sum(diff == 0)),
    }


def verdict(
    mean_diff: float,
    p_value: float,
    threshold: float,
    alpha: float = 0.05,
    lower_is_better: bool = True,
    min_n: Optional[int] = None,
    n: Optional[float] = None,
) -> str:
    """Map a paired comparison onto Supported / Contradicted / Inconclusive (see module docstring)."""
    if min_n is not None and n is not None and n < min_n:
        return "Inconclusive (too few seeds)"
    if not (np.isfinite(mean_diff) and np.isfinite(p_value)):
        return "Inconclusive"
    signed = mean_diff if lower_is_better else -mean_diff
    if p_value < alpha and signed <= -abs(threshold):
        return "Supported"
    if p_value < alpha and signed > 0:
        return "Contradicted"
    if p_value < alpha:
        return "Inconclusive (significant but below practical threshold)"
    return "Inconclusive"


# Two-sided 97.5% Student-t quantiles for df = 1..30 (above 30 the normal 1.96 is used).
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042)


def equivalence_verdict(diffs: Sequence[float], margin: float, min_n: Optional[int] = None) -> Dict[str, object]:
    """
    Verdict for an equivalence claim ("A is within +/- margin of B") from per-seed paired diffs.

    Supported when the 95% t confidence interval of mean(diff) lies inside [-margin, +margin];
    Contradicted when it lies entirely outside; Inconclusive otherwise.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    out: Dict[str, object] = {"n": float(n), "margin": float(margin)}
    if min_n is not None and n < min_n:
        out["verdict"] = "Inconclusive (too few seeds)"
        return out
    if n < 2:
        out["verdict"] = "Inconclusive"
        return out
    mean = float(d.mean())
    half = (_T975[n - 2] if n - 1 <= len(_T975) else 1.96) * float(d.std(ddof=1)) / math.sqrt(n)
    lo, hi = mean - half, mean + half
    out.update({"mean_diff": mean, "ci95_low": lo, "ci95_high": hi})
    if -margin <= lo and hi <= margin:
        out["verdict"] = "Supported"
    elif lo > margin or hi < -margin:
        out["verdict"] = "Contradicted"
    else:
        out["verdict"] = "Inconclusive"
    return out
