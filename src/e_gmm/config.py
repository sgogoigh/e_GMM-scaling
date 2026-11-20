"""Configuration dataclasses and benchmark dataset presets for e-GMM."""

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class DatasetScaleSpec:
    """Architectural and scale specification for benchmark datasets."""

    name: str
    num_train: int
    dim: int
    num_components: int
    recommended_cov_type: str
    recommended_rank: int
    default_batch_size: int
    default_epochs: float
    dtype_input: str = "float32"
    notes: str = ""


DATASET_PRESETS: Dict[str, DatasetScaleSpec] = {
    "tiny-debug": DatasetScaleSpec(
        name="tiny-debug",
        num_train=10_000,
        dim=32,
        num_components=16,
        recommended_cov_type="full",
        recommended_rank=8,
        default_batch_size=512,
        default_epochs=5.0,
        dtype_input="float32",
        notes="Instant CPU/GPU verification preset for math and unit tests.",
    ),
    "sift-1m": DatasetScaleSpec(
        name="sift-1m",
        num_train=1_000_000,
        dim=128,
        num_components=4_096,
        recommended_cov_type="diag",
        recommended_rank=16,
        default_batch_size=16_384,
        default_epochs=3.0,
        dtype_input="float32",
        notes="BigANN SIFT1M image descriptors (128-d).",
    ),
    "deep-1m": DatasetScaleSpec(
        name="deep-1m",
        num_train=1_000_000,
        dim=96,
        num_components=16_384,
        recommended_cov_type="diag",
        recommended_rank=16,
        default_batch_size=32_768,
        default_epochs=3.0,
        dtype_input="float32",
        notes="DEEP1M CNN embeddings (96-d); primary scaling-law fit domain.",
    ),
    "deep-1b": DatasetScaleSpec(
        name="deep-1b",
        num_train=1_000_000_000,
        dim=96,
        num_components=65_536,
        recommended_cov_type="diag",
        recommended_rank=16,
        default_batch_size=262_144,
        default_epochs=2.5,
        dtype_input="float32",
        notes="DEEP1B billion-scale benchmark with Flash-GMM 2d-GEMM + Top-m M-step.",
    ),
    "spacev-100m": DatasetScaleSpec(
        name="spacev-100m",
        num_train=100_000_000,
        dim=100,
        num_components=65_536,
        recommended_cov_type="diag",
        recommended_rank=16,
        default_batch_size=131_072,
        default_epochs=3.0,
        dtype_input="int8",
        notes="Microsoft SPACEV100M web-search text embeddings (100-d).",
    ),
    "dinov2-100m": DatasetScaleSpec(
        name="dinov2-100m",
        num_train=100_000_000,
        dim=1024,
        num_components=16_384,
        recommended_cov_type="low_rank_diag",
        recommended_rank=16,
        default_batch_size=65_536,
        default_epochs=2.5,
        dtype_input="bfloat16",
        notes="High-d correlated ViT embeddings; Low-Rank+Diag avoids 275 GB full-cov wall.",
    ),
    "imagenet-1k": DatasetScaleSpec(
        name="imagenet-1k",
        num_train=1_281_167,
        dim=768,
        num_components=1_024,
        recommended_cov_type="full",
        recommended_rank=32,
        default_batch_size=16_384,
        default_epochs=4.0,
        dtype_input="float32",
        notes="ImageNet-1k frozen vision backbone embeddings (768-d, k=1024).",
    ),
    "stretched-d768": DatasetScaleSpec(
        name="stretched-d768",
        num_train=500_000,
        dim=768,
        num_components=2_048,
        recommended_cov_type="diag",
        recommended_rank=16,
        default_batch_size=8_192,
        default_epochs=3.0,
        dtype_input="float32",
        notes="Anisotropic power-law stretched Gaussian for Phase-A d-aware sweep.",
    ),
}


@dataclass
class EGMMConfig:
    """Hyperparameters for 3-tuple EMA Gaussian Mixture Models (e-GMM)."""

    num_components: int = 16
    dim: int = 32
    cov_type: str = "diag"  # "diag", "low_rank_diag", "full"
    low_rank: int = 8

    # Covariance positive-definiteness & initialization
    cov_floor: float = 1e-4       # lambda_0 I_d
    init_sigma0: float = 1.0      # initial isotropic std
    init_prior_mass: float = 1e-10
    center_data: bool = False

    # Cosine EMA step schedule
    alpha_0: float = 0.25
    eps_floor: float = 0.1
    use_cosine_schedule: bool = True

    # E-step / M-step optimizations
    e_step_mode: str = "flash_gemm"  # "flash_gemm" or "naive" (diag mode)
    top_m: Optional[int] = None      # Sparse top-m responsibilities (e.g., 4, 8, 16)
    chunk_k: int = 2048              # Tiling size over k for streaming LSE

    # Spectral component splitting (reviving collapsed Gaussians)
    split_interval: int = 5
    split_tau: float = 0.10
    split_delta: float = 0.50
    split_strategy: str = "spectral_v1"  # "spectral_v1", "random_orth", "faiss_alt", "batch_resample", "none"
    max_splits_per_step: int = 4
    cooldown_steps: int = 10
    power_iter_steps: int = 5

    # Finalization
    finalization: str = "last_epoch"  # "last_epoch", "slow_ema", "main_ema"
    slow_ema_alpha: float = 0.01

    seed: int = 42

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
