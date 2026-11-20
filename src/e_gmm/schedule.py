"""Step-size schedules for e-GMM and Cappé & Moulines (2009) Online EM baselines."""

import math


def cosine_alpha(
    step: int,
    total_steps: int,
    alpha_0: float,
    eps_floor: float = 0.1,
    use_cosine: bool = True,
) -> float:
    """
    Paper Eq. 5 cosine annealing schedule for EMA step size alpha(t) = 1 - gamma(t).

    Args:
        step: Zero-based or one-based batch index t in [0, total_steps].
        total_steps: Total batch updates M = T * floor(n / B).
        alpha_0: Peak initial step size at t = 0.
        eps_floor: Minimum floor ratio at t = M (default 0.1 -> ends at 0.1 * alpha_0).
        use_cosine: If False, returns constant alpha_0.
    """
    if not use_cosine or total_steps <= 1:
        return float(min(max(alpha_0, 1e-8), 1.0))
    progress = min(max(step / float(max(total_steps - 1, 1)), 0.0), 1.0)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    alpha_t = alpha_0 * (eps_floor + (1.0 - eps_floor) * cos_term)
    return float(min(max(alpha_t, 1e-8), 1.0))


def robbins_monro_gamma(step: int, beta: float = 0.75, t0: float = 10.0) -> float:
    """
    Classical Cappé & Moulines (2009) Robbins-Monro step size: gamma_t = (t + t0)^(-beta).

    Note: In Cappé & Moulines notation, gamma_t is the weight on the NEW batch
    (corresponding to alpha_t in e-means / e-GMM). beta must lie in (0.5, 1.0].
    """
    if not (0.5 < beta <= 1.0):
        raise ValueError(f"Robbins-Monro exponent beta must lie in (0.5, 1.0], got {beta}.")
    return float(min(max((max(step, 0) + t0) ** (-beta), 1e-8), 1.0))


def predict_d_aware_alpha0(
    num_components: float,
    batch_size: float,
    num_train: float,
    epochs: float,
    dim: float,
    c_alpha: float = 0.824,
    a_k: float = -0.46,
    a_b: float = 0.90,
    a_n: float = -0.45,
    a_t: float = -0.11,
    a_d: float = -0.25,
) -> float:
    """
    Dimension-aware power law for peak step size alpha_0:
        alpha_0 = min(C_alpha * k^a_k * B^a_B * n^a_n * T^a_T * d^a_d, 1.0)
    Defaults extrapolate from e-means Eq. 6 with negative dimension exponent a_d < 0
    prior to dataset-specific Phase-A OLS fitting.
    """
    val = (
        c_alpha
        * (num_components ** a_k)
        * (batch_size ** a_b)
        * (num_train ** a_n)
        * (epochs ** a_t)
        * (dim ** a_d)
    )
    return float(min(max(val, 1e-6), 1.0))


def predict_d_aware_batch_knee(
    num_components: float,
    num_train: float,
    epochs: float,
    dim: float,
    c_b: float = 0.035,
    b_k: float = 0.13,
    b_n: float = 0.88,
    b_t: float = 0.81,
    b_d: float = 0.05,
    round_pow2: bool = True,
) -> int:
    """Dimension-aware power law for knee batch size B."""
    val = (
        c_b
        * (num_components ** b_k)
        * (num_train ** b_n)
        * (epochs ** b_t)
        * (dim ** b_d)
    )
    val = min(max(val, float(num_components)), float(num_train))
    if round_pow2:
        pow2 = 2 ** round(math.log2(max(val, 1.0)))
        return int(min(max(pow2, 1), int(num_train)))
    return int(round(val))
