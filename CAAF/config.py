"""Public configuration dictionaries; callers supply only overrides."""

from typing import Any, Callable, Literal, TypedDict

from numpy.typing import ArrayLike
from torch import nn
from torch.optim import Optimizer

NormalizationMethod = Literal["centered_max", "standard", "minmax", "centered_range", "none"]


# Normalization can be selected independently for inputs and targets.
class NormalizationConfig(TypedDict, total=False):
    x: NormalizationMethod
    y: NormalizationMethod


# Each clustering method accepts only its applicable settings.
class ClusteringConfig(TypedDict, total=False):
    name: Literal["ap", "kmeans", "spectral", "agglomerative"]
    preference: float | None
    damping: float
    max_iter: int
    convergence_iter: int
    n_clusters: int
    n_init: int
    tol: float
    assign_labels: str
    distance_threshold: float
    linkage: Literal["average", "complete", "single"]


# The functional form permits the reserved keyword "class" as a dictionary key.
ModelConfig = TypedDict("ModelConfig", {
    "name": Literal["mlp", "lstm", "tcn"], "class": type[nn.Module],
    "factory": Callable[..., nn.Module], "kwargs": dict[str, Any],
    "hidden_sizes": tuple[int, ...], "activation": str, "negative_slope": float,
    "dropout": float, "batch_norm": bool, "bias": bool, "hidden_size": int,
    "num_layers": int, "bidirectional": bool, "channels": tuple[int, ...],
    "kernel_size": int,
}, total=False)


# A window ending at t predicts the target at t + horizon.
class SequenceConfig(TypedDict, total=False):
    length: int
    horizon: int
    stride: int
    groups: ArrayLike | None


# Scheduler and stopping settings are merged with their defaults.
class SchedulerConfig(TypedDict, total=False):
    factor: float
    patience: int
    min_lr: float
    eps: float


class EarlyStoppingConfig(TypedDict, total=False):
    patience: int
    min_delta: float
    min_epochs: int


# Custom optimizers receive only optimizer_kwargs as constructor options.
class TrainingConfig(TypedDict, total=False):
    optimizer: Literal["adam", "adamw"] | type[Optimizer]
    optimizer_kwargs: dict[str, Any]
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    validation_fraction: float
    n_runs: int
    dtype: Literal["float64", "float32"]
    verbose: bool
    scheduler: SchedulerConfig | None
    early_stopping: EarlyStoppingConfig | None


# Vector baselines are specified in the original input coordinates.
class IGConfig(TypedDict, total=False):
    baseline: Any
    n_steps: int
    method: str
    batch_size: int
    internal_batch_size: int | None
    max_samples: int | None
    targets: list[int] | None
    target_weights: list[float] | None
    aggregation: Literal["mean_abs", "mean_signed"]
    convergence_delta: bool
