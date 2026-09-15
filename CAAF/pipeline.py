"""Normalize, cluster, fit regression models, and rank sensors using IG."""

from contextlib import nullcontext
import copy
import random
from typing import Any, Literal, overload
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn
from captum.attr import IntegratedGradients
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             KMeans, SpectralClustering)
from sklearn.exceptions import ConvergenceWarning

from .config import (ClusteringConfig, IGConfig, ModelConfig, NormalizationConfig,
                     SequenceConfig, TrainingConfig)
from .models import MLP, LSTM, TCN


# Shared defaults are copied before applying overrides.
_CLUSTER = {
    "ap": dict(preference=None, damping=0.5, max_iter=10000, convergence_iter=20),
    "kmeans": dict(n_clusters=None, n_init=10, max_iter=300, tol=1e-4),
    "spectral": dict(n_clusters=None, assign_labels="kmeans", n_init=10),
    "agglomerative": dict(n_clusters=None, distance_threshold=None, linkage="average"),
}
_MODEL = {
    "mlp": dict(hidden_sizes=(64, 64, 64), activation="leaky_relu", negative_slope=0.01,
                dropout=0, batch_norm=False, bias=True),
    "lstm": dict(hidden_size=64, num_layers=2, dropout=0, bidirectional=False),
    "tcn": dict(channels=(64, 64), kernel_size=3, activation="relu", dropout=0),
}
_TRAIN = dict(optimizer="adam", optimizer_kwargs={}, lr=1e-4, weight_decay=1e-5,
              epochs=150, batch_size=64, validation_fraction=0.1, n_runs=1,
              dtype="float64", verbose=True,
              scheduler=dict(factor=0.5, patience=3, min_lr=1e-7, eps=1e-8),
              early_stopping=None)
_IG = dict(baseline="mean", n_steps=50, method="gausslegendre", batch_size=128,
           internal_batch_size=1024, max_samples=None, targets=None,
           target_weights=None, aggregation="mean_abs", convergence_delta=False)


# Reject misspelled or inapplicable options instead of silently ignoring them.
def _merge(defaults, overrides, group):
    if overrides is None:
        return copy.deepcopy(defaults)
    if not isinstance(overrides, dict):
        raise TypeError(f"{group} must be a configuration dictionary.")
    unknown = overrides.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {group} settings: {sorted(unknown)}.")
    return {**copy.deepcopy(defaults), **overrides}


# Integer parameters reject booleans and fractional values explicitly.
def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


# Store the affine transform so physical baselines use the same model coordinates.
def _normalize(values, method):
    mean = values.mean(axis=0)
    span = np.ptp(values, axis=0)
    if method == "centered_max":
        offset, scale = mean, np.max(np.abs(values - mean), axis=0)
    elif method == "standard":
        offset, scale = mean, values.std(axis=0)
    elif method == "minmax":
        offset, scale = values.min(axis=0), span
    elif method == "centered_range":
        offset, scale = mean, span
    elif method == "none":
        offset, scale = np.zeros_like(mean), np.ones_like(mean)
    else:
        raise ValueError(f"Unknown normalization method {method!r}.")
    scale = np.where(scale == 0, 1, scale)
    return (values - offset) / scale, dict(method=method, offset=offset, scale=scale,
                                         mean=mean, constant=span == 0)


# Cluster sensor histories and retain actual member indices as representatives.
def _cluster(X, config, seed, verbose=False):
    config = {"name": config} if isinstance(config, str) else dict(config or {})
    name = config.pop("name", "ap").lower()
    if name not in _CLUSTER:
        raise ValueError(f"Unknown clustering method {name!r}.")
    settings = _merge(_CLUSTER[name], config, "clustering")
    if verbose:
        print(f"Clustering: {name}; {X.shape[1]} sensor histories with {len(X)} samples each.", flush=True)
    count = settings.get("n_clusters")
    if name in ("kmeans", "spectral") and count is None:
        raise ValueError(f"{name} requires n_clusters.")
    if count is not None:
        _integer(count, "n_clusters")
        if count > X.shape[1]:
            raise ValueError("n_clusters exceeds the number of nonconstant sensors.")
    if name == "agglomerative":
        if (count is None) == (settings["distance_threshold"] is None):
            raise ValueError("Agglomerative clustering requires exactly one of n_clusters or distance_threshold.")
        if settings["linkage"] not in ("average", "complete", "single"):
            raise ValueError("Agglomerative linkage must be average, complete, or single.")

    # A single nonconstant sensor is already its own valid cluster.
    if X.shape[1] == 1:
        return np.array([0]), np.array([0]), {"name": name, **settings}
    affinity = np.clip(np.abs(np.corrcoef(X.T)), 0, 1)
    np.fill_diagonal(affinity, 1)
    if not np.isfinite(affinity).all():
        raise ValueError("Sensor correlations are not finite; check input magnitudes.")
    if name == "ap":
        estimator = AffinityPropagation(affinity="precomputed", random_state=seed, **settings)
        data = affinity
    elif name == "kmeans":
        estimator = KMeans(random_state=seed, **settings)
        data = X.T
    elif name == "spectral":
        estimator = SpectralClustering(affinity="precomputed", random_state=seed, **settings)
        data = affinity
    else:
        estimator = AgglomerativeClustering(metric="precomputed", **settings)
        data = 1 - affinity
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        labels = estimator.fit_predict(data)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError(f"{name} clustering did not converge to valid clusters; adjust clustering settings.")
    for item in caught:
        warnings.warn(str(item.message), item.category, stacklevel=2)
    if np.any(labels < 0) or (count is not None and len(np.unique(labels)) != count):
        raise ValueError("Clustering returned invalid or empty clusters.")
    if name == "kmeans" and estimator.n_iter_ >= settings["max_iter"]:
        raise ValueError("K-means reached max_iter; increase it to ensure convergence.")
    if name == "ap":
        representatives = np.asarray(estimator.cluster_centers_indices_, dtype=int)
    else:
        representatives = []
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            if name == "kmeans":
                distances = np.sum((X[:, members].T - estimator.cluster_centers_[label]) ** 2, axis=1)
                representative = members[np.argmin(distances)]
            else:
                representative = members[np.argmax(affinity[np.ix_(members, members)].mean(axis=1))]
            representatives.append(representative)
        representatives = np.asarray(representatives)
    if not len(representatives) or len(representatives) != len(np.unique(labels)):
        raise ValueError("Clustering produced no valid representatives.")
    return representatives, labels, {"name": name, **settings}


# Window targets follow the last input timestep plus the requested horizon.
def _windows(X, y, settings, offset=0, segments=None):
    length, horizon, stride = (settings[k] for k in ("length", "horizon", "stride"))
    segments = [(0, len(X))] if segments is None else segments
    ends = []
    for start, stop in segments:
        if stop - start < length + horizon:
            raise ValueError(f"Each data split must contain enough samples for sequence length and horizon; "
                             f"rows [{start + offset}, {stop + offset}) are too short.")
        ends.append(np.arange(start + length - 1, stop - horizon, stride))
    ends = np.concatenate(ends)
    indices = ends[:, None] - np.arange(length - 1, -1, -1)
    inputs = X[indices] if length > 1 else X[ends]
    return inputs, y[ends + horizon], dict(input_indices=indices + offset,
                                          target_indices=ends + horizon + offset)


# Resolve model construction before allocating training state.
def _model_settings(model):
    if isinstance(model, type) and issubclass(model, nn.Module):
        model = {"class": model}
    config = {"name": model} if isinstance(model, str) else dict(model or {})
    selectors = config.keys() & {"name", "class", "factory"}
    if len(selectors) > 1:
        raise ValueError("Exactly one of model name, class, or factory is permitted.")
    if selectors & {"class", "factory"}:
        unknown = config.keys() - {"class", "factory", "kwargs"}
        if unknown:
            raise ValueError(f"Custom model settings belong in kwargs: {sorted(unknown)}.")
        kwargs = dict(config.get("kwargs", {}))
        if kwargs.keys() & {"input_shape", "output_dim"}:
            raise ValueError("Model kwargs cannot override inferred input_shape or output_dim.")
        constructor = config.get("class", config.get("factory"))
        if "class" in config and not (isinstance(constructor, type) and issubclass(constructor, nn.Module)):
            raise TypeError("model['class'] must be an nn.Module subclass.")
        if not callable(constructor):
            raise TypeError("model['factory'] must be callable.")
        return constructor, kwargs, {**config, "kwargs": kwargs}
    name = config.pop("name", "mlp").lower()
    if name not in _MODEL:
        raise ValueError(f"Unknown built-in model {name!r}.")
    kwargs = config.pop("kwargs", {})
    if config.keys() & kwargs.keys():
        raise ValueError("Duplicate built-in model options in kwargs.")
    settings = _merge(_MODEL[name], {**config, **kwargs}, "model")
    return {"mlp": MLP, "lstm": LSTM, "tcn": TCN}[name], settings, {"name": name, **settings}


# Preserve optimizer ownership by constructing it from each new model's parameters.
def _training_settings(overrides):
    overrides = overrides or {}
    settings = _merge(_TRAIN, overrides, "training")
    for key in ("epochs", "batch_size", "n_runs"):
        _integer(settings[key], f"training.{key}")
    if not 0 < settings["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if settings["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64.")
    optimizer = settings["optimizer"]
    kwargs = dict(settings["optimizer_kwargs"])
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer not in ("adam", "adamw"):
            raise ValueError("Built-in optimizer must be adam or adamw.")
        for key in ("lr", "weight_decay"):
            if key in overrides and key in kwargs:
                raise ValueError(f"Duplicate optimizer option {key}.")
            kwargs.setdefault(key, settings[key])
            settings[key] = kwargs[key]
        constructor = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer]
        settings["optimizer"] = optimizer
    else:
        if not isinstance(optimizer, type) or not issubclass(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a name or Optimizer subclass, not an initialized instance.")
        if overrides.keys() & {"lr", "weight_decay"}:
            raise ValueError("Custom optimizer constructor options must be supplied exclusively in optimizer_kwargs.")
        constructor = optimizer
        settings.pop("lr")
        settings.pop("weight_decay")
    settings["optimizer_kwargs"] = kwargs
    if settings["scheduler"] is not None:
        settings["scheduler"] = _merge(_TRAIN["scheduler"], settings["scheduler"], "scheduler")
    if settings["early_stopping"] is not None:
        stop = _merge(dict(patience=10, min_delta=0, min_epochs=0), settings["early_stopping"], "early_stopping")
        _integer(stop["patience"], "early_stopping.patience")
        _integer(stop["min_epochs"], "early_stopping.min_epochs", 0)
        if not np.isfinite(stop["min_delta"]) or stop["min_delta"] < 0:
            raise ValueError("early_stopping.min_delta must be finite and nonnegative.")
        settings["early_stopping"] = stop
    return constructor, settings


# Check both the output contract and the gradient path before training starts.
def _validate_model(model, batch, output_dim):
    if not isinstance(model, nn.Module):
        raise TypeError("The model factory must return an nn.Module.")
    model.eval()
    probe = batch[:2].detach().requires_grad_(True)
    output = model(probe)
    if not isinstance(output, torch.Tensor) or output.shape != (len(probe), output_dim):
        raise ValueError(f"Model forward must return a tensor shaped (batch_size, {output_dim}).")
    if not torch.isfinite(output).all():
        raise ValueError("Model output must be finite.")
    if not output.requires_grad:
        raise ValueError("Model outputs must remain differentiable with respect to inputs for IG.")
    gradient = torch.autograd.grad(output.sum(), probe, allow_unused=True)[0]
    if gradient is None:
        raise ValueError("Model forward must support input gradients for IG.")


# Training counts each batch once even if an optimizer reevaluates its closure.
def _train(model, optimizer_class, train_X, train_y, val_X, val_y, settings, seed):
    optimizer = optimizer_class(model.parameters(), **settings["optimizer_kwargs"])
    if settings["scheduler"] is not None and any("lr" not in group for group in optimizer.param_groups):
        raise ValueError("Scheduling requires optimizer parameter groups with lr; set scheduler=None.")
    scheduler = None if settings["scheduler"] is None else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **settings["scheduler"])
    best_loss, stop_best, stale = float("inf"), float("inf"), 0
    history = dict(seed=seed, train_loss=[], validation_loss=[], learning_rates=[], best_epoch=None)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    uses_batch_norm = any(isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in model.modules())
    if uses_batch_norm and (settings["batch_size"] < 2 or len(train_X) < 2):
        raise ValueError("Batch normalization requires batch_size and training sample count >= 2.")
    for epoch in range(settings["epochs"]):
        model.train()
        order = torch.randperm(len(train_X), generator=generator)
        batches = list(order.split(settings["batch_size"]))
        if uses_batch_norm and len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        total = 0.0
        for indices in batches:
            xb, yb = train_X[indices].to(device), train_y[indices].to(device)
            batch_loss = None

            # The optimizer may call this closure repeatedly; log its first evaluation only.
            def closure():
                nonlocal batch_loss
                with torch.enable_grad():
                    optimizer.zero_grad()
                    prediction = model(xb)
                    if prediction.shape != yb.shape:
                        raise ValueError("Model output shape changed during training.")
                    loss = nn.functional.mse_loss(prediction, yb)
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is nonfinite; check data, model, and optimizer settings.")
                    loss.backward()
                    if batch_loss is None:
                        batch_loss = loss.detach().item()
                    return loss

            optimizer.step(closure)
            if batch_loss is None:
                raise ValueError("Custom optimizer.step must evaluate the supplied closure.")
            total += batch_loss * len(indices)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for start in range(0, len(val_X), settings["batch_size"]):
                xb = val_X[start:start + settings["batch_size"]].to(device)
                yb = val_y[start:start + settings["batch_size"]].to(xb.device)
                prediction = model(xb)
                if prediction.shape != yb.shape:
                    raise ValueError("Model output shape changed during validation.")
                validation_total += nn.functional.mse_loss(prediction, yb).item() * len(xb)
        validation_loss = validation_total / len(val_X)
        if not np.isfinite(validation_loss):
            raise ValueError("Validation loss is nonfinite.")
        history["train_loss"].append(total / len(train_X))
        history["validation_loss"].append(validation_loss)
        history["learning_rates"].append([group.get("lr") for group in optimizer.param_groups])
        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
        if scheduler is not None:
            scheduler.step(validation_loss)
        if settings["verbose"]:
            print(f"  Epoch {epoch + 1}/{settings['epochs']}: train_loss={total / len(train_X):.6g}, "
                  f"validation_loss={validation_loss:.6g}, lr={history['learning_rates'][-1]}", flush=True)
        stop = settings["early_stopping"]
        if stop is not None:
            if validation_loss < stop_best - stop["min_delta"]:
                stop_best, stale = validation_loss, 0
            else:
                stale += 1
            if epoch + 1 >= stop["min_epochs"] and stale >= stop["patience"]:
                if settings["verbose"]:
                    print(f"  Early stopping after {epoch + 1} epochs.", flush=True)
                break
    model.load_state_dict(checkpoint)
    model.eval()
    history["best_validation_loss"] = best_loss
    if settings["verbose"]:
        print(f"  Restored epoch {history['best_epoch']}: best validation_loss={best_loss:.6g}.", flush=True)
    return history


# IG accumulates sample-weighted output scores without retaining attribution tensors.
def _attribute(model, X, baseline, targets, config, verbose=False):
    model.eval()
    parameter = next(model.parameters())
    baseline = torch.as_tensor(baseline, dtype=parameter.dtype, device=parameter.device).unsqueeze(0)
    scores = np.zeros((len(targets), X.shape[-1]))
    delta_summaries = []
    explainer = IntegratedGradients(model)
    # Report approximately ten updates per output, plus the first and final batches.
    n_batches = (len(X) + config["batch_size"] - 1) // config["batch_size"]
    report_every = max(1, (n_batches + 9) // 10)
    recurrent = any(isinstance(layer, nn.RNNBase) for layer in model.modules())
    context = torch.backends.cudnn.flags(enabled=False) if recurrent and parameter.is_cuda else nullcontext()
    with context:
        for target_i, target in enumerate(targets):
            if verbose:
                print(f"Attribution: output {target} ({target_i + 1}/{len(targets)}); "
                      f"{len(X)} samples, {config['n_steps']} IG steps, method={config['method']}.", flush=True)
            delta_sum, delta_max, delta_count = 0.0, 0.0, 0
            for start in range(0, len(X), config["batch_size"]):
                batch = X[start:start + config["batch_size"]].to(parameter.device)
                result = explainer.attribute(batch, baselines=baseline, target=int(target),
                    n_steps=config["n_steps"], method=config["method"],
                    internal_batch_size=config["internal_batch_size"],
                    return_convergence_delta=config["convergence_delta"])
                if config["convergence_delta"]:
                    attribution, delta = result
                    delta_sum += delta.abs().sum().item()
                    delta_max = max(delta_max, delta.abs().max().item())
                    delta_count += len(delta)
                else:
                    attribution = result
                if not torch.isfinite(attribution).all():
                    raise ValueError("IG produced nonfinite attributions; check the model's input gradients.")
                if config["aggregation"] == "mean_abs":
                    attribution = attribution.abs()
                if attribution.ndim == 3:
                    attribution = attribution.mean(dim=1)
                scores[target_i] += attribution.sum(dim=0).detach().cpu().numpy()
                batch_number = start // config["batch_size"] + 1
                if verbose and (batch_number == 1 or batch_number % report_every == 0 or batch_number == n_batches):
                    completed = start + len(batch)
                    print(f"  Attribution output {target}: {completed}/{len(X)} samples "
                          f"({100 * completed / len(X):.1f}%), batch {batch_number}/{n_batches}.", flush=True)
            if delta_count:
                delta_summaries.append(dict(target=int(target), mean_abs=delta_sum / delta_count,
                                            max_abs=delta_max, n_samples=delta_count))
                if verbose:
                    print(f"  IG convergence residual: mean_abs={delta_sum / delta_count:.6g}, "
                          f"max_abs={delta_max:.6g}.", flush=True)
    return scores / len(X), delta_summaries


# Overloads expose the three public return forms to type checkers.
@overload
def rank_sensors(X: ArrayLike, y: ArrayLike, *, n_sensors: int | None = None,
    normalization: str | NormalizationConfig | None = None, clustering: str | ClusteringConfig | None = None,
    model: str | ModelConfig | type[nn.Module] | None = None, sequence: SequenceConfig | None = None,
    training: TrainingConfig | None = None, ig: IGConfig | None = None, random_state: int = 42,
    device: str = "auto", verbose: bool = True, return_percentages: bool = False, return_details: Literal[True]) -> dict[str, Any]: ...


@overload
def rank_sensors(X: ArrayLike, y: ArrayLike, *, n_sensors: int | None = None,
    normalization: str | NormalizationConfig | None = None, clustering: str | ClusteringConfig | None = None,
    model: str | ModelConfig | type[nn.Module] | None = None, sequence: SequenceConfig | None = None,
    training: TrainingConfig | None = None, ig: IGConfig | None = None, random_state: int = 42,
    device: str = "auto", verbose: bool = True, return_percentages: Literal[True], return_details: Literal[False] = False
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...


@overload
def rank_sensors(X: ArrayLike, y: ArrayLike, *, n_sensors: int | None = None,
    normalization: str | NormalizationConfig | None = None, clustering: str | ClusteringConfig | None = None,
    model: str | ModelConfig | type[nn.Module] | None = None, sequence: SequenceConfig | None = None,
    training: TrainingConfig | None = None, ig: IGConfig | None = None, random_state: int = 42,
    device: str = "auto", verbose: bool = True, return_percentages: Literal[False] = False, return_details: Literal[False] = False
) -> NDArray[np.int64]: ...


def rank_sensors(X: ArrayLike, y: ArrayLike, *, n_sensors: int | None = None,
    normalization: str | NormalizationConfig | None = None, clustering: str | ClusteringConfig | None = None,
    model: str | ModelConfig | type[nn.Module] | None = None, sequence: SequenceConfig | None = None,
    training: TrainingConfig | None = None, ig: IGConfig | None = None, random_state: int = 42,
    device: str = "auto", verbose: bool = True, return_percentages: bool = False, return_details: bool = False
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]] | dict[str, Any]:
    """Rank original sensor indices by integrated gradients on cluster representatives.

    X has shape (samples, sensors); y has shape (samples,) or (samples, outputs).
    Dictionary arguments contain overrides only. Percentages use all representatives
    as their denominator, including when n_sensors selects only the top k. Details
    take precedence over percentages. verbose=True reports each processing stage;
    verbose=False disables CAAF progress output. See README.md for configuration contracts.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim not in (1, 2):
        raise ValueError("X must be (samples, sensors); y must be (samples,) or (samples, outputs).")
    if y.ndim == 1:
        y = y[:, None]
    if len(X) != len(y) or len(X) < 3 or not X.shape[1] or not y.shape[1]:
        raise ValueError("Provide matching X/y with at least three samples and nonempty columns.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a boolean.")
    if verbose:
        print(f"CAAF input: X={X.shape}, y={y.shape}; {len(X)} samples, "
              f"{X.shape[1]} sensors, {y.shape[1]} outputs.", flush=True)
    _integer(random_state, "random_state", 0)
    if random_state >= 2 ** 32:
        raise ValueError("random_state must be less than 2**32 for clustering.")
    random_state = int(random_state)
    if n_sensors is not None:
        _integer(n_sensors, "n_sensors")
    active = np.flatnonzero(np.ptp(X, axis=0) != 0)
    if not len(active):
        raise ValueError("All input sensor columns are constant.")
    if not np.any(np.ptp(y, axis=0)):
        raise ValueError("All target columns are constant.")

    # Normalize all supplied samples and map cluster members back to original indices.
    norm = {"x": normalization, "y": normalization} if isinstance(normalization, str) else normalization
    norm = _merge(dict(x="centered_max", y="centered_max"), norm, "normalization")
    Xn, x_stats = _normalize(X, norm["x"])
    yn, y_stats = _normalize(y, norm["y"])
    if verbose:
        print(f"Normalization: X={norm['x']}, y={norm['y']}; "
              f"excluded constant sensor indices={np.flatnonzero(x_stats['constant']).tolist()}.", flush=True)
    representatives, labels, cluster_settings = _cluster(Xn[:, active], clustering, random_state, verbose=verbose)
    representatives = active[representatives]
    if verbose:
        print(f"Clusters: {len(representatives)}; cluster center indices "
              f"(original sensor representatives)={representatives.tolist()}.", flush=True)
    if n_sensors is not None and n_sensors > len(representatives):
        raise ValueError(f"Requested {n_sensors} sensors, but clustering produced only {len(representatives)} representatives.")
    all_labels = np.full(X.shape[1], -1, dtype=int)
    all_labels[active] = labels
    Xr = Xn[:, representatives]
    seq = _merge(dict(length=1, horizon=0, stride=1, groups=None), sequence, "sequence")
    for key in ("length", "horizon", "stride"):
        _integer(seq[key], f"sequence.{key}", 0 if key == "horizon" else 1)
    groups = seq["groups"]
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(X),) or groups.dtype.kind not in "iu":
            raise ValueError("sequence.groups must be a one-dimensional integer array with one entry per sample.")
        starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
        if len(np.unique(groups[starts])) != len(starts):
            raise ValueError("Each sequence.groups identifier must occupy one contiguous block.")
        stops = np.r_[starts[1:], len(X)]
        seq["groups"] = groups.copy()
    constructor, model_kwargs, model_settings = _model_settings(model)
    optimizer_class, train_settings = _training_settings(training)
    train_settings["verbose"] = bool(verbose and train_settings["verbose"])

    # Temporal splits precede window creation so no history crosses the split boundary.
    n_val = max(1, int(np.ceil(len(X) * train_settings["validation_fraction"])))
    if groups is not None:
        # Split each trajectory before constructing any histories or forecast targets.
        counts = np.maximum(1, np.ceil((stops - starts) * train_settings["validation_fraction"]).astype(int))
        boundaries = stops - counts
        train_X, train_y, train_indices = _windows(Xr, yn, seq, segments=zip(starts, boundaries))
        val_X, val_y, val_indices = _windows(Xr, yn, seq, segments=zip(boundaries, stops))
        evaluation_X, _, evaluation_indices = _windows(Xr, yn, seq, segments=zip(starts, stops))
        for rows in (train_indices, val_indices, evaluation_indices):
            rows["group_ids"] = groups[rows["target_indices"]]
    elif any(seq[key] != value for key, value in dict(length=1, horizon=0, stride=1).items()):
        boundary = len(X) - n_val
        train_X, train_y, train_indices = _windows(Xr[:boundary], yn[:boundary], seq)
        val_X, val_y, val_indices = _windows(Xr[boundary:], yn[boundary:], seq, boundary)
    else:
        order = np.random.default_rng(random_state).permutation(len(X))
        val_rows, train_rows = order[:n_val], order[n_val:]
        if not len(train_rows):
            raise ValueError("The validation split leaves no training samples.")
        train_X, train_y = Xr[train_rows], yn[train_rows]
        val_X, val_y = Xr[val_rows], yn[val_rows]
        train_indices = dict(input_indices=train_rows[:, None], target_indices=train_rows)
        val_indices = dict(input_indices=val_rows[:, None], target_indices=val_rows)
    if groups is None:
        evaluation_X, _, evaluation_indices = _windows(Xr, yn, seq)

    # Validate IG settings and transform physical baselines before training any runs.
    ig_settings = _merge(_IG, ig, "ig")
    for key in ("n_steps", "batch_size"):
        _integer(ig_settings[key], f"ig.{key}")
    if ig_settings["method"] not in ("gausslegendre", "riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid"):
        raise ValueError("Unknown IG integration method.")
    if ig_settings["method"].startswith("riemann") and ig_settings["n_steps"] < 2:
        raise ValueError("Riemann IG methods require at least two steps.")
    if ig_settings["aggregation"] not in ("mean_abs", "mean_signed"):
        raise ValueError("IG aggregation must be mean_abs or mean_signed.")
    if ig_settings["max_samples"] is not None:
        _integer(ig_settings["max_samples"], "ig.max_samples")
        selected = np.sort(np.random.default_rng(random_state).choice(len(evaluation_X),
                          min(len(evaluation_X), ig_settings["max_samples"]), replace=False))
        evaluation_X = evaluation_X[selected]
        evaluation_indices = {key: value[selected] for key, value in evaluation_indices.items()}
    if ig_settings["internal_batch_size"] is not None:
        _integer(ig_settings["internal_batch_size"], "ig.internal_batch_size")
        if ig_settings["internal_batch_size"] < min(ig_settings["batch_size"], len(evaluation_X)):
            raise ValueError("IG internal_batch_size must be at least the effective IG batch_size.")
    targets = list(range(y.shape[1])) if ig_settings["targets"] is None else list(ig_settings["targets"])
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("IG targets must be nonempty and unique.")
    for target in targets:
        _integer(target, "IG target", 0)
        if target >= y.shape[1]:
            raise ValueError("IG target index exceeds the number of outputs.")
    weights = np.ones(len(targets)) if ig_settings["target_weights"] is None else np.asarray(ig_settings["target_weights"], dtype=float)
    if weights.shape != (len(targets),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("target_weights must match targets and be finite, nonnegative, with positive sum.")
    # Rescale before summation so large finite weights retain their relative shares.
    weights = weights / weights.max()
    weights = weights / weights.sum()
    baseline = ig_settings["baseline"]
    if isinstance(baseline, str):
        if baseline == "mean":
            baseline = X.mean(axis=0)
        elif baseline == "zero":
            baseline = np.zeros(X.shape[1])
        else:
            raise ValueError("IG baseline must be mean, zero, or a vector with one entry per original sensor.")
    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != (X.shape[1],) or not np.isfinite(baseline).all():
        raise ValueError("IG baseline vector must be finite and match the original sensor count.")
    baseline = ((baseline - x_stats["offset"]) / x_stats["scale"])[representatives]
    if seq["length"] > 1:
        baseline = np.broadcast_to(baseline, (seq["length"], len(representatives))).copy()

    # Reuse one split and evaluation set while initializing independent runs.
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    dtype = getattr(torch, train_settings["dtype"])
    if verbose:
        print(f"Device: {resolved_device}; dtype={train_settings['dtype']}.", flush=True)
        print(f"Data split: {len(train_X)} training, {len(val_X)} validation, "
              f"{len(evaluation_X)} attribution samples; input shape={train_X.shape[1:]}; "
              f"sequence={ {key: seq[key] for key in ('length', 'horizon', 'stride')} }.", flush=True)
        if groups is not None:
            print(f"Sequence groups: {len(starts)} separate trajectories, each split chronologically.", flush=True)
        print(f"Training: {train_settings['n_runs']} runs, up to {train_settings['epochs']} epochs/run, "
              f"batch_size={train_settings['batch_size']}, optimizer={optimizer_class.__name__}.", flush=True)
    train_X, train_y, val_X, val_y, evaluation_X = [torch.as_tensor(value, dtype=dtype)
        for value in (train_X, train_y, val_X, val_y, evaluation_X)]
    run_histories, run_scores, convergence = [], [], []
    for run in range(train_settings["n_runs"]):
        seed = random_state + run
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        network = constructor(input_shape=tuple(train_X.shape[1:]), output_dim=y.shape[1], **model_kwargs)
        if not isinstance(network, nn.Module):
            raise TypeError("The model factory must return an nn.Module.")
        network.to(device=resolved_device, dtype=dtype)
        if not any(parameter.requires_grad for parameter in network.parameters()):
            raise ValueError("The model must expose trainable parameters.")
        if verbose:
            print(f"Run {run + 1}/{train_settings['n_runs']}: seed={seed}, model={type(network).__name__}; "
                  "starting training.", flush=True)
        recurrent = any(isinstance(layer, nn.RNNBase) for layer in network.modules())
        context = torch.backends.cudnn.flags(enabled=False) if recurrent and resolved_device.type == "cuda" else nullcontext()
        with context:
            _validate_model(network, train_X[:2].to(resolved_device), y.shape[1])
        history = _train(network, optimizer_class, train_X, train_y, val_X, val_y, train_settings, seed)
        output_scores, deltas = _attribute(network, evaluation_X, baseline, targets, ig_settings, verbose=verbose)
        if verbose:
            print(f"Run {run + 1}/{train_settings['n_runs']} complete.", flush=True)
        run_histories.append(history)
        run_scores.append(output_scores)
        convergence.append(deltas)

    # Percentages use all representatives before any requested top-k selection.
    per_output_scores = np.mean(run_scores, axis=0)
    scores = weights @ per_output_scores
    order = np.lexsort((representatives, -scores))
    magnitude = np.abs(scores)
    relative = magnitude / magnitude.max() if magnitude.max() else np.zeros_like(scores)
    percentages = 100 * (relative / relative.sum()) if relative.sum() else relative
    ranking, ranked_percentages = representatives[order], percentages[order]
    indices, selected_percentages = ranking[:n_sensors], ranked_percentages[:n_sensors]
    if verbose:
        print(f"Ranking complete: selected {len(indices)}/{len(representatives)} representatives.", flush=True)
        print(f"Sensor indices (ranked): {indices.tolist()}", flush=True)
        print(f"Attribution percentages: {selected_percentages.round(4).tolist()}", flush=True)
    if return_details:
        return dict(indices=indices, percentages=selected_percentages, ranking=ranking,
            ranking_percentages=ranked_percentages, scores=scores, per_output_scores=per_output_scores,
            representatives=representatives, labels=all_labels, excluded_columns=np.flatnonzero(all_labels < 0),
            normalization=dict(x=x_stats, y=y_stats), baseline=baseline,
            settings=dict(normalization=norm, clustering=cluster_settings, model=model_settings,
                          sequence=seq, training=train_settings, ig={**ig_settings, "targets": targets,
                          "target_weights": weights}, random_state=random_state, device=str(resolved_device), verbose=bool(verbose)),
            split=dict(train=train_indices, validation=val_indices), evaluation=evaluation_indices,
            run_histories=run_histories, run_scores=np.asarray(run_scores),
            convergence=convergence if ig_settings["convergence_delta"] else None)
    if return_percentages:
        return indices, selected_percentages
    return indices
