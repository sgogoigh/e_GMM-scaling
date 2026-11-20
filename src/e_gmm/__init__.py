"""e_GMM-scaling: Scaling Gaussian Mixture Models via Exponential Moving Averages."""

from .config import DATASET_PRESETS, DatasetScaleSpec, EGMMConfig
from .poc_logging import tee_poc_logs

__all__ = [
    "EGMMConfig",
    "DatasetScaleSpec",
    "DATASET_PRESETS",
    "tee_poc_logs",
]

try:
    import numpy  # noqa: F401
    from .datasets import (
        generate_gaussian_mixture,
        generate_isotropic_gaussian,
        generate_stretched_gaussian,
        generate_uniform_cube,
        iter_minibatches,
        load_vector_memmap,
    )
    from .metrics import compute_gmm_health, evaluate_gmm_nll
    from .scaling_law import (
        DimensionAlphaLawFit,
        DimensionBatchLawFit,
        DimensionScalingObservation,
        compute_nll_regret,
        evaluate_lodo_d,
        evaluate_loso_n,
        fit_dimension_aware_alpha0,
        fit_dimension_aware_batch_knee,
    )
    from .schedule import (
        cosine_alpha,
        predict_d_aware_alpha0,
        predict_d_aware_batch_knee,
        robbins_monro_gamma,
    )

    __all__ += [
        "cosine_alpha",
        "robbins_monro_gamma",
        "predict_d_aware_alpha0",
        "predict_d_aware_batch_knee",
        "compute_gmm_health",
        "evaluate_gmm_nll",
        "DimensionScalingObservation",
        "DimensionAlphaLawFit",
        "DimensionBatchLawFit",
        "fit_dimension_aware_alpha0",
        "fit_dimension_aware_batch_knee",
        "evaluate_loso_n",
        "evaluate_lodo_d",
        "compute_nll_regret",
        "generate_stretched_gaussian",
        "generate_gaussian_mixture",
        "generate_isotropic_gaussian",
        "generate_uniform_cube",
        "iter_minibatches",
        "load_vector_memmap",
    ]
except ImportError:
    pass

try:
    import numpy  # noqa: F401
    from .numpy_egmm import (
        NumPyEGMM,
        NumPyEMeansPostHocGMM,
        NumPyFullBatchEM,
        NumPyOnlineEMCappeMoulines,
    )

    __all__ += [
        "NumPyEGMM",
        "NumPyOnlineEMCappeMoulines",
        "NumPyFullBatchEM",
        "NumPyEMeansPostHocGMM",
    ]
except ImportError:
    pass

try:
    import torch  # noqa: F401
    from .torch_egmm import TorchEGMM
    from .triton_kernels import HAS_TRITON, flash_gmm_diag_log_probs

    __all__ += [
        "TorchEGMM",
        "HAS_TRITON",
        "flash_gmm_diag_log_probs",
    ]
except ImportError:
    pass
