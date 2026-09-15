# CAAF API reference

[Back to the quick start](README.md)

## Public interface

Supply finite numeric `X` with shape `(samples, sensors)` and `y` with shape `(samples,)` or `(samples, outputs)`.

```python
import CAAF

indices = CAAF.rank_sensors(X, y)
indices, percentages = CAAF.rank_sensors(X, y, return_percentages=True)
details = CAAF.rank_sensors(X, y, n_sensors=10, return_details=True)
```

Indices are zero-based original input columns. `n_sensors=None` returns every cluster representative; a positive integer selects the top k. Constant input columns are excluded while preserving original indices. Constant output columns are supported, but entirely constant targets are rejected.

The complete signature exposes overrides as keyword arguments.

```python
rank_sensors(
    X, y, *, n_sensors=None, normalization=None, clustering=None, model=None,
    sequence=None, training=None, ig=None, random_state=42, device="auto", verbose=True,
    return_percentages=False, return_details=False,
)
```

Configuration dictionaries contain overrides only. Unknown or inapplicable options raise errors. Normalization, clustering, and built-in model names also accept strings. `NormalizationConfig`, `ClusteringConfig`, `ModelConfig`, `SequenceConfig`, `TrainingConfig`, and `IGConfig` are exported `TypedDict` annotations; return forms have overload annotations.

`verbose=True` reports input dimensions, normalization and excluded columns, clustering and original cluster center indices, device and dtype, split sizes, model/run seeds, epoch training and validation losses, learning rates, restored checkpoints, attribution progress for each output, and the final ranking. Attribution updates include completed samples and percentage progress. Set `verbose=False` to silence CAAF progress output across all stages. The existing `training={"verbose": False}` option suppresses epoch details only; it cannot override a top-level `verbose=False`.

## Normalization and clustering

Normalization uses all supplied samples, independently per column. `centered_max` (default) subtracts the mean and divides by the maximum absolute centered value. `standard` divides centered values by population standard deviation, `minmax` maps to [0, 1], `centered_range` divides centered values by max minus min, and `none` preserves values. Constant-column scales are replaced by one.

Select input and output normalization separately when needed.

```python
indices = CAAF.rank_sensors(X, y, normalization={"x": "standard", "y": "centered_range"})
```

Clustering operates on complete sensor histories before temporal windows are constructed.

| `name` | Default options | Representative |
|---|---|---|
| `ap` | `preference=None`, `damping=0.5`, `max_iter=10000`, `convergence_iter=20` | Affinity-propagation exemplar using absolute Pearson correlation |
| `kmeans` | Required `n_clusters`; `n_init=10`, `max_iter=300`, `tol=1e-4` | Actual member nearest the centroid in normalized history space |
| `spectral` | Required `n_clusters`; `assign_labels="kmeans"`, `n_init=10` | Member with greatest mean within-cluster absolute correlation |
| `agglomerative` | Required `n_clusters` or `distance_threshold`; `linkage="average"` | Member with greatest mean within-cluster absolute correlation |

Spectral clustering uses absolute correlation affinity. Agglomerative clustering uses `1 - abs(correlation)` with `average`, `complete`, or `single` linkage. Nonconvergence, missing clusters, and requesting more sensors than representatives raise errors. A K-means run reaching `max_iter` is conservatively rejected; increase the limit.

## Models and temporal inputs

| `name` | Defaults |
|---|---|
| `mlp` | `hidden_sizes=(64,64,64)`, `activation="leaky_relu"`, `negative_slope=0.01`, `dropout=0`, `batch_norm=False`, `bias=True` |
| `lstm` | `hidden_size=64`, `num_layers=2`, `dropout=0`, `bidirectional=False` |
| `tcn` | `channels=(64,64)`, `kernel_size=3`, `activation="relu"`, `dropout=0` |

MLP flattens its input history, LSTM uses the last layer's final hidden states (both directions when bidirectional), and TCN applies causal temporal convolutions followed by a flattened linear output. Configurable activations are `relu`, `leaky_relu`, `gelu`, and `tanh`. Built-in options may appear directly in the model dictionary or in `kwargs`, with no duplicate definitions.

`sequence={"length": 1, "horizon": 0, "stride": 1}` is the default. A window ending at t predicts `y[t + horizon]`. Any nondefault temporal configuration splits the raw samples chronologically before window construction, so training and validation windows cannot overlap. Instantaneous inputs use a seeded random split. Default IG evaluates every valid window across the complete supplied input, including validation samples.

For concatenated independent trajectories, set `sequence.groups` to a one-dimensional integer array with one identifier per input row. Each identifier must occupy one contiguous block. Every trajectory is split chronologically using `validation_fraction`, including when `length=1`; training and validation windows are constructed independently within those partitions. IG windows stay inside each complete trajectory but can span its internal train/validation boundary. Each partition must have at least `length + horizon` rows, otherwise the call raises an error identifying the short row range. Omitting groups or passing `None` preserves the ungrouped behavior.

Supply trajectory identifiers together with the history length.

```python
indices = CAAF.rank_sensors(
    X, y, model="tcn",
    sequence={"length": 10, "horizon": 0, "groups": probe_ids},
)
```

Diagnostics retain original input/target row indices and include `group_ids` aligned with each window's target when groups are supplied. Normalization and clustering still use all supplied samples; grouped splits prevent temporal-window overlap but do not make the validation loss an independent predictive benchmark.

Custom models receive the inferred input shape and output dimension after clustering.

```python
import math
import torch
from torch import nn

# Flatten the inferred history shape and predict all target quantities.
class CustomRegressor(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_size=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(math.prod(input_shape), hidden_size),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# The pipeline constructs a fresh instance for each run.
indices, percentages = CAAF.rank_sensors(
    X, y,
    model={"class": CustomRegressor, "kwargs": {"hidden_size": 128, "dropout": 0.1}},
    return_percentages=True,
)
```

Construction uses `CustomRegressor(input_shape=..., output_dim=..., **kwargs)`. The shape excludes the batch dimension: `(n_representatives,)` for length 1, otherwise `(length, n_representatives)`. Forward must return a differentiable tensor shaped `(batch_size, output_dim)`, including `(batch_size, 1)` for scalar targets. The pipeline moves the model to the selected device and floating-point dtype and validates output shape and input gradients before training.

A class may also be passed directly as `model=CustomRegressor`. Use `model={"factory": callable, "kwargs": {...}}` to adapt existing constructors: the callable receives the same inferred arguments and must return a fresh `nn.Module` each time. Exactly one selector (`name`, `class`, or `factory`) is allowed; omitting all selects MLP. User kwargs cannot override inferred dimensions. Initialized model instances are unsupported.

Detached predictions, nondifferentiable forward operations, and models requiring additional forward arguments do not satisfy this interface. Custom models must have trainable parameters and support evaluation-mode input gradients. IG runs in evaluation mode; cuDNN is disabled locally for recurrent-model attribution on CUDA.

## Training and optimizers

Training minimizes mean squared error. Defaults are Adam, `lr=1e-4`, `weight_decay=1e-5`, `epochs=150`, `batch_size=64`, `validation_fraction=0.1`, `n_runs=1`, `dtype="float64"`, and `verbose=True`. `dtype="float32"` is supported. `device="auto"` selects CUDA when available, otherwise CPU; an explicit PyTorch device string is also accepted.

Each run uses seed `random_state + run_index`, a fresh model and optimizer, and the same train/validation split. Losses are weighted by sample count, and each run restores its lowest-validation-loss checkpoint before IG. Batch normalization merges a final singleton training batch into the preceding batch. Its configured batch size must be at least two.

The default ReduceLROnPlateau scheduler has `factor=0.5`, `patience=3`, `min_lr=1e-7`, and `eps=1e-8`. Override individual scheduler options or set `training={"scheduler": None}`. Early stopping is disabled by default; enable it with `early_stopping={"patience": 10, "min_delta": 0, "min_epochs": 0}`. The stopping counter tracks improvements exceeding `min_delta`; checkpoint selection always uses the actual minimum validation loss.

Choose `"adam"`, `"adamw"`, or a `torch.optim.Optimizer` subclass. Custom classes receive exactly `OptimizerClass(model.parameters(), **optimizer_kwargs)` after device placement.

```python
# Custom optimizer constructor options belong exclusively in optimizer_kwargs.
indices = CAAF.rank_sensors(
    X, y,
    model={"class": CustomRegressor, "kwargs": {"hidden_size": 128}},
    training={"optimizer": torch.optim.SGD,
              "optimizer_kwargs": {"lr": 1e-3, "momentum": 0.9},
              "epochs": 150, "n_runs": 3},
)
```

Custom optimizer classes reject top-level `lr` and `weight_decay` overrides and receive no Adam defaults. Built-in names accept those shortcuts and extra `optimizer_kwargs` such as `betas` and `eps`; duplicate definitions of `lr` or `weight_decay` are rejected. Initialized optimizer instances are rejected because they already reference another model's parameters.

Optimization calls `optimizer.step(closure)`, allowing repeated loss evaluation (for example, LBFGS). A custom optimizer must invoke the closure; metrics count its first evaluation once per training batch. Scheduling requires ordinary parameter groups containing `lr`; otherwise set `scheduler=None`. Custom loss functions and training callbacks are outside this interface.

## Integrated gradients and results

| Option | Default |
|---|---|
| `baseline` | `"mean"` |
| `n_steps`, `method` | `50`, `"gausslegendre"` |
| `batch_size`, `internal_batch_size` | `128`, `1024` |
| `max_samples` | `None` (all valid input windows) |
| `targets`, `target_weights` | `None` (all outputs, equal weights) |
| `aggregation`, `convergence_delta` | `"mean_abs"`, `False` |

Baselines are defined in original physical input coordinates: `"mean"` uses the mean of all supplied samples, `"zero"` uses physical zero, and a finite vector supplies one value per **original** sensor column, including excluded columns. The normalization transform and representative selection are then applied. A temporal baseline repeats the vector across history steps.

IG is computed separately for each selected output. Scores average absolute attribution over samples, history steps, outputs, and independent runs. `mean_signed` preserves attribution signs before those averages. `targets` is a list of unique zero-based output indices; nonnegative `target_weights` align with that list and are normalized to sum to one. Scores refer to normalized model outputs; cross-output physical-unit scaling must be supplied through those weights when needed. A positive `max_samples` selects a seeded random subset shared across runs. Internal batch size must cover at least one complete evaluation batch; use `None` to remove the internal limit.

Ranking uses decreasing aggregated score, breaking ties by original sensor index. For `mean_signed`, ranking is by signed score, while percentages still use absolute magnitude. Each percentage is `100 * abs(score) / sum(abs(all_representative_scores))`. Top-k values retain their share of the full ranking. All-zero scores produce zero percentages.

`return_details=True` takes precedence over `return_percentages` and returns these diagnostics.

| Keys | Alignment or meaning |
|---|---|
| `indices`, `percentages` | Selected ranking and aligned percentages |
| `ranking`, `ranking_percentages` | Full ranking before top-k selection |
| `representatives`, `scores` | Original representative indices and scores in cluster order |
| `per_output_scores` | `(selected_outputs, representatives)` in target-list order |
| `run_scores` | `(runs, selected_outputs, representatives)` |
| `labels`, `excluded_columns` | One label per original sensor; excluded constants have label -1 |
| `normalization`, `baseline` | Input/output offsets, scales, means, constant masks; transformed IG baseline |
| `settings` | Resolved configurations, device, and random seed |
| `split`, `evaluation` | Original row indices for histories and targets |
| `run_histories` | Sample-weighted losses, learning rates, seed, best epoch and validation loss |
| `convergence` | Optional per-run, per-output mean/max absolute IG completeness residuals |
