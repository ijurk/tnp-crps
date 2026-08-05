from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.base import Batch
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.experiment_utils import deep_convert_dict, extract_config
from tnp_crps.data.tabular import TabularRegressionGenerator
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn

from evaluation.predictive_sampling import sample_model_chunked


BASELINE_KINDS = {
    "context_mean",
    "context_gaussian",
    "context_resample",
}


@dataclass(frozen=True)
class LoadedSource:
    name: str
    display_name: str
    kind: str
    checkpoint_path: str
    sampling_seed_offset: int
    model: Optional[torch.nn.Module]
    training_alpha: Optional[float]
    eval_context_sizes: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class RawTaskRecord:
    accepted_index: int
    scanned_index: int
    x_raw: torch.Tensor
    y_raw: torch.Tensor
    row_permutation: torch.Tensor
    feature_permutation: torch.Tensor
    metadata: Dict[str, Any]
    task_fingerprint: str
    target_fingerprint: str


class NestedTaskRejection(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


def load_merged_config(
    *,
    config_paths: Sequence[str],
    overrides: Optional[Sequence[str]] = None,
):
    raw_config = deep_convert_dict(
        hiyapyco.load(
            list(config_paths),
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    config, _ = extract_config(
        raw_config,
        config_changes=list(overrides or []),
        combine_default=True,
    )
    OmegaConf.resolve(config)
    return config


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    batch_kwargs: Dict[str, Any] = {}

    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
        batch_kwargs[field.name] = value

    return type(batch)(**batch_kwargs)


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    attempts = [("direct", state_dict)]

    if any(key.startswith("model.") for key in state_dict):
        attempts.append(
            (
                "strip_model_prefix",
                {
                    key[len("model.") :]: value
                    for key, value in state_dict.items()
                    if key.startswith("model.")
                },
            )
        )

    if any(key.startswith("lit_model.model.") for key in state_dict):
        attempts.append(
            (
                "strip_lit_model_model_prefix",
                {
                    key[len("lit_model.model.") :]: value
                    for key, value in state_dict.items()
                    if key.startswith("lit_model.model.")
                },
            )
        )

    last_error: Optional[RuntimeError] = None

    for mode, candidate in attempts:
        try:
            model.load_state_dict(candidate, strict=True)
            print(f"Loaded checkpoint using state_dict mode: {mode}")
            return
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to load checkpoint into model. "
        f"checkpoint_path={checkpoint_path}\nLast error:\n{last_error}"
    )


def _training_alpha(entry: Mapping[str, Any], config: Any) -> Optional[float]:
    explicit = entry.get("training_alpha", None)
    config_value = None

    if hasattr(config, "params"):
        config_value = getattr(config.params, "crps_alpha", None)

    if explicit is None:
        resolved = config_value
    else:
        resolved = explicit

    if resolved is None:
        return None

    value = float(resolved)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"training_alpha must be in [0,1] or null, got {value}."
        )

    if explicit is not None and config_value is not None:
        if not math.isclose(
            value,
            float(config_value),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Explicit training_alpha does not match the resolved "
                f"model configuration for {entry.get('name')!r}."
            )

    return value


def load_sources(
    *,
    entries: Sequence[Mapping[str, Any]],
    base_generator_config: str,
    device: torch.device,
) -> List[LoadedSource]:
    offsets: List[int] = []
    sources: List[LoadedSource] = []

    for entry in entries:
        name = str(entry["name"])
        display_name = str(entry.get("display_name", name))
        kind = str(entry.get("kind", "model"))
        offset = int(entry["sampling_seed_offset"])
        offsets.append(offset)

        eval_context_sizes_raw = entry.get("eval_context_sizes", None)
        eval_context_sizes = (
            tuple(int(value) for value in eval_context_sizes_raw)
            if eval_context_sizes_raw is not None
            else None
        )

        if kind in BASELINE_KINDS:
            if entry.get("training_alpha", None) is not None:
                raise ValueError(
                    f"Baseline {name!r} must use training_alpha: null."
                )

            sources.append(
                LoadedSource(
                    name=name,
                    display_name=display_name,
                    kind=kind,
                    checkpoint_path=f"<baseline:{kind}>",
                    sampling_seed_offset=offset,
                    model=None,
                    training_alpha=None,
                    eval_context_sizes=eval_context_sizes,
                )
            )
            continue

        if kind != "model":
            raise ValueError(
                f"Unknown source kind {kind!r} for source {name!r}."
            )

        model_config = str(entry["model_config"])
        checkpoint_path = str(entry["checkpoint_path"])

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        config = load_merged_config(
            config_paths=[base_generator_config, model_config],
            overrides=list(entry.get("overrides", []) or []),
        )

        pl.seed_everything(int(config.misc.seed))
        model = instantiate(config.model)
        load_model_state(model, checkpoint_path)
        model.to(device)
        model.eval()

        sources.append(
            LoadedSource(
                name=name,
                display_name=display_name,
                kind=kind,
                checkpoint_path=checkpoint_path,
                sampling_seed_offset=offset,
                model=model,
                training_alpha=_training_alpha(entry, config),
                eval_context_sizes=eval_context_sizes,
            )
        )

    if len(offsets) != len(set(offsets)):
        raise ValueError(
            f"sampling_seed_offset values must be unique, got {offsets}."
        )
    if any(value < 1 for value in offsets):
        raise ValueError(
            f"sampling_seed_offset values must be positive, got {offsets}."
        )

    return sources


@torch.no_grad()
def sample_tabular_baseline(
    *,
    baseline_kind: str,
    batch: Batch,
    num_samples: int,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    num_samples = int(num_samples)
    if num_samples < 2:
        raise ValueError("At least two baseline samples are required.")

    yc = batch.yc
    batch_size = int(yc.shape[0])
    num_context = int(yc.shape[1])
    num_targets = int(batch.yt.shape[1])
    dim_y = int(yc.shape[-1])

    context_mean = yc.mean(dim=1, keepdim=True)

    if baseline_kind == "context_mean":
        return (
            context_mean.unsqueeze(0)
            .expand(num_samples, batch_size, num_targets, dim_y)
            .clone()
        )

    if baseline_kind == "context_gaussian":
        context_std = yc.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(float(epsilon))
        noise = torch.randn(
            num_samples,
            batch_size,
            num_targets,
            dim_y,
            device=yc.device,
            dtype=yc.dtype,
        )
        return context_mean.unsqueeze(0) + context_std.unsqueeze(0) * noise

    if baseline_kind == "context_resample":
        indices = torch.randint(
            low=0,
            high=num_context,
            size=(num_samples, batch_size, num_targets),
            device=yc.device,
        )
        source = yc.unsqueeze(0).expand(
            num_samples,
            batch_size,
            num_context,
            dim_y,
        )
        gather_indices = indices.unsqueeze(-1).expand(
            num_samples,
            batch_size,
            num_targets,
            dim_y,
        )
        return torch.gather(source, dim=2, index=gather_indices)

    raise ValueError(
        f"Unknown baseline kind {baseline_kind!r}; "
        f"expected one of {sorted(BASELINE_KINDS)}."
    )


@torch.no_grad()
def sample_loaded_source(
    *,
    source: LoadedSource,
    batch: Batch,
    num_samples: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if source.model is None:
        samples = sample_tabular_baseline(
            baseline_kind=source.kind,
            batch=batch,
            num_samples=num_samples,
        )
        return samples, None, None

    if isinstance(source.model, DirectTNP):
        samples = sample_model_chunked(
            model=source.model,
            batch=batch,
            num_samples=num_samples,
            chunk_size=chunk_size,
        )
        return samples, None, None

    pred_dist = np_pred_fn(
        model=source.model,
        batch=batch,
        num_samples=num_samples,
    )
    if not isinstance(pred_dist, torch.distributions.Normal):
        raise TypeError(
            "Gaussian tabular source must return torch.distributions.Normal; "
            f"got {type(pred_dist)}."
        )
    samples = pred_dist.sample((int(num_samples),))
    if not torch.isfinite(samples).all():
        raise FloatingPointError(
            f"Non-finite predictive samples for source {source.name!r}."
        )
    return samples, pred_dist.loc, pred_dist.scale


def tensor_fingerprint(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()

    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


def batch_task_fingerprints(batch: Batch) -> List[str]:
    return [
        tensor_fingerprint(
            batch.xc[index],
            batch.yc[index],
            batch.xt[index],
            batch.yt[index],
        )
        for index in range(int(batch.yt.shape[0]))
    ]


def stable_sampling_seed(
    *,
    base_seed: int,
    source_offset: int,
    batch_index: int,
    condition_index: int = 0,
) -> int:
    if source_offset < 1:
        raise ValueError("source_offset must be positive.")
    if batch_index < 0 or condition_index < 0:
        raise ValueError("batch_index and condition_index must be non-negative.")
    return (
        int(base_seed)
        + 10_000_000 * int(source_offset)
        + 1_000_000 * int(condition_index)
        + int(batch_index)
    )


def build_generator(
    *,
    base_generator_config: str,
    overrides: Sequence[str],
    samples_per_epoch: int,
    batch_size: int,
) -> TabularRegressionGenerator:
    config = load_merged_config(
        config_paths=[base_generator_config],
        overrides=list(overrides),
    )
    config.generators.test.samples_per_epoch = int(samples_per_epoch)
    config.generators.test.batch_size = int(batch_size)

    pl.seed_everything(int(config.misc.seed))
    generator = instantiate(config.generators.test)

    if not isinstance(generator, TabularRegressionGenerator):
        raise TypeError(
            "Expected TabularRegressionGenerator, got "
            f"{type(generator)}."
        )
    return generator


def _validate_raw_task(
    *,
    generator: TabularRegressionGenerator,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    x_raw = x_raw.detach().to(device="cpu", dtype=torch.float32)
    y_raw = y_raw.detach().to(device="cpu", dtype=torch.float32)

    if y_raw.ndim == 1:
        y_raw = y_raw.unsqueeze(-1)

    if x_raw.ndim != 2:
        raise NestedTaskRejection(
            f"raw_x_rank:{tuple(x_raw.shape)}"
        )
    if y_raw.ndim != 2 or int(y_raw.shape[-1]) != 1:
        raise NestedTaskRejection(
            f"raw_y_shape:{tuple(y_raw.shape)}"
        )
    if int(x_raw.shape[0]) != int(y_raw.shape[0]):
        raise NestedTaskRejection("raw_row_count_mismatch")
    if not torch.isfinite(x_raw).all():
        raise NestedTaskRejection("nonfinite_raw_x")
    if not torch.isfinite(y_raw).all():
        raise NestedTaskRejection("nonfinite_raw_y")

    num_features = int(x_raw.shape[-1])
    if num_features < 1:
        raise NestedTaskRejection("zero_active_features")
    if num_features > int(generator.max_input_features):
        raise NestedTaskRejection(
            f"too_many_features:{num_features}"
        )

    return x_raw, y_raw, num_features


def prepare_nested_rung(
    *,
    generator: TabularRegressionGenerator,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    context_pool_indices: torch.Tensor,
    target_indices: torch.Tensor,
    feature_permutation: torch.Tensor,
    num_context: int,
) -> Tuple[SyntheticBatch, Dict[str, float]]:
    x_raw, y_raw, num_features = _validate_raw_task(
        generator=generator,
        x_raw=x_raw,
        y_raw=y_raw,
    )

    num_context = int(num_context)
    if num_context < 1 or num_context > int(context_pool_indices.numel()):
        raise ValueError(
            f"Invalid num_context={num_context} for context pool of size "
            f"{context_pool_indices.numel()}."
        )

    context_indices = context_pool_indices[:num_context]
    xc_raw = x_raw[context_indices]
    yc_raw = y_raw[context_indices]
    xt_raw = x_raw[target_indices]
    yt_raw = y_raw[target_indices]

    context_target_std = yc_raw.std(
        dim=0,
        unbiased=False,
    )
    if (
        not torch.isfinite(context_target_std).all()
        or float(context_target_std.min().item())
        < float(generator.min_context_target_std)
    ):
        raise NestedTaskRejection("context_y_std")

    xc, xt, _, _ = generator._preprocess_pair(
        xc_raw,
        xt_raw,
        mode=generator.x_preprocessing_mode,
        zero_constant_dimensions=True,
    )
    max_abs_x = float(torch.maximum(xc.abs().max(), xt.abs().max()).item())
    if (
        not math.isfinite(max_abs_x)
        or max_abs_x > float(generator.max_abs_standardized_input)
    ):
        raise NestedTaskRejection("standardized_x_bound")

    yc, yt, y_mean, y_std = generator._preprocess_pair(
        yc_raw,
        yt_raw,
        mode=generator.y_preprocessing_mode,
        zero_constant_dimensions=False,
    )
    max_abs_y = float(torch.maximum(yc.abs().max(), yt.abs().max()).item())
    if (
        not math.isfinite(max_abs_y)
        or max_abs_y > float(generator.max_abs_standardized_target)
    ):
        raise NestedTaskRejection("standardized_y_bound")

    padding = int(generator.max_input_features) - num_features
    if padding > 0:
        zero_context = torch.zeros(
            int(xc.shape[0]),
            padding,
            dtype=xc.dtype,
        )
        zero_target = torch.zeros(
            int(xt.shape[0]),
            padding,
            dtype=xt.dtype,
        )
        xc = torch.cat([xc, zero_context], dim=-1)
        xt = torch.cat([xt, zero_target], dim=-1)

    if tuple(feature_permutation.shape) != (
        int(generator.max_input_features),
    ):
        raise ValueError(
            "feature_permutation has the wrong shape: "
            f"{tuple(feature_permutation.shape)}."
        )

    xc = xc[:, feature_permutation]
    xt = xt[:, feature_permutation]

    if not (
        torch.isfinite(xc).all()
        and torch.isfinite(xt).all()
        and torch.isfinite(yc).all()
        and torch.isfinite(yt).all()
    ):
        raise NestedTaskRejection("nonfinite_processed_task")

    x = torch.cat([xc, xt], dim=0).unsqueeze(0).contiguous()
    y = torch.cat([yc, yt], dim=0).unsqueeze(0).contiguous()

    batch = SyntheticBatch(
        x=x,
        y=y,
        xc=xc.unsqueeze(0).contiguous(),
        yc=yc.unsqueeze(0).contiguous(),
        xt=xt.unsqueeze(0).contiguous(),
        yt=yt.unsqueeze(0).contiguous(),
        gt_pred=None,
    )

    diagnostics = {
        "active_num_features": float(num_features),
        "raw_context_y_std": float(context_target_std.item()),
        "context_y_mean": float(y_mean.item()),
        "context_y_std": float(y_std.item()),
        "max_abs_standardized_x": max_abs_x,
        "max_abs_standardized_y": max_abs_y,
    }
    return batch, diagnostics


def stack_single_task_batches(batches: Sequence[SyntheticBatch]) -> SyntheticBatch:
    if not batches:
        raise ValueError("Cannot stack an empty batch sequence.")

    return SyntheticBatch(
        x=torch.cat([batch.x for batch in batches], dim=0),
        y=torch.cat([batch.y for batch in batches], dim=0),
        xc=torch.cat([batch.xc for batch in batches], dim=0),
        yc=torch.cat([batch.yc for batch in batches], dim=0),
        xt=torch.cat([batch.xt for batch in batches], dim=0),
        yt=torch.cat([batch.yt for batch in batches], dim=0),
        gt_pred=None,
    )


def raw_task_fingerprints(
    *,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    context_pool_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> Tuple[str, str]:
    task_fingerprint = tensor_fingerprint(
        x_raw[context_pool_indices],
        y_raw[context_pool_indices],
        x_raw[target_indices],
        y_raw[target_indices],
    )
    target_fingerprint = tensor_fingerprint(
        x_raw[target_indices],
        y_raw[target_indices],
    )
    return task_fingerprint, target_fingerprint


def checkpoint_training_state(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }


def per_task_metric_rows_efficient(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    num_context: int,
    model_name: str,
    display_name: str,
    checkpoint_path: str,
    eval_set: str,
    task_index_start: int,
    alpha: float = 1.0,
    interval_levels: Iterable[float] = (0.9,),
    compute_energy_score: bool = True,
) -> List[Dict[str, Any]]:
    """Memory-bounded task metrics for large finite ensembles."""
    from evaluation.metrics import (
        energy_score_per_task,
        level_suffix,
    )

    if samples.shape[1:] != target.shape:
        raise ValueError(
            "Expected samples.shape[1:] == target.shape; "
            f"got {samples.shape} and {target.shape}."
        )

    num_samples = int(samples.shape[0])
    batch_size = int(target.shape[0])
    num_targets = int(target.shape[1])
    output_dim = int(target.shape[2])
    num_elements = num_targets * output_dim

    if num_samples < 2:
        raise ValueError("At least two predictive samples are required.")

    pred_mean = samples.mean(dim=0)
    squared_error = (pred_mean - target).pow(2).reshape(batch_size, -1)
    task_rmse = squared_error.mean(dim=1).sqrt()

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}.")

    # One sort supplies both the almost-fair CRPS and the exact
    # off-diagonal ensemble diversity, avoiding a second O(M log M) pass.
    sorted_samples = samples.sort(dim=0).values
    coefficients = torch.arange(
        1,
        num_samples + 1,
        device=samples.device,
        dtype=samples.dtype,
    )
    coefficients = (
        2.0 * coefficients - float(num_samples) - 1.0
    ).reshape(num_samples, *([1] * (samples.ndim - 1)))
    weighted_order_sum = (coefficients * sorted_samples).sum(dim=0)

    target_term = torch.abs(
        samples - target.unsqueeze(0)
    ).mean(dim=0)
    pairwise_weight = (
        float(alpha) / (num_samples * (num_samples - 1))
        + (1.0 - float(alpha)) / (num_samples * num_samples)
    )
    crps = (
        target_term - pairwise_weight * weighted_order_sum
    ).reshape(batch_size, -1)
    task_crps = crps.mean(dim=1)

    sample_var = samples.var(
        dim=0,
        unbiased=True,
    ).reshape(batch_size, -1)
    task_spread = sample_var.mean(dim=1).sqrt()

    offdiag_diversity = (
        2.0 * weighted_order_sum
        / (num_samples * (num_samples - 1))
    ).reshape(batch_size, -1)
    task_diversity = offdiag_diversity.mean(dim=1)

    if compute_energy_score:
        task_energy = energy_score_per_task(
            samples=samples,
            target=target,
            mask=None,
        )
    else:
        task_energy = torch.full(
            (batch_size,),
            float("nan"),
            device=target.device,
            dtype=target.dtype,
        )

    finite_m_correction = math.sqrt((num_samples + 1.0) / num_samples)
    task_ssr = finite_m_correction * task_spread / (task_rmse + 1.0e-12)

    interval_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level in interval_levels:
        suffix = level_suffix(float(level))
        lower_q = (1.0 - float(level)) / 2.0
        upper_q = 1.0 - lower_q
        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        coverage = (
            ((target >= lower) & (target <= upper))
            .to(target.dtype)
            .reshape(batch_size, -1)
            .mean(dim=1)
        )
        width = (upper - lower).reshape(batch_size, -1).mean(dim=1)
        interval_metrics[suffix] = (coverage, width)

    rows: List[Dict[str, Any]] = []
    for local_index in range(batch_size):
        row: Dict[str, Any] = {
            "model_name": model_name,
            "display_name": display_name,
            "checkpoint_path": checkpoint_path,
            "eval_set": eval_set,
            "region": "all",
            "context_bucket": f"nc_{int(num_context):03d}",
            "task_index": int(task_index_start) + local_index,
            "num_context": int(num_context),
            "num_targets": num_targets,
            "output_dim": output_dim,
            "num_target_elements": num_elements,
            "num_eval_samples": num_samples,
            "rmse": float(task_rmse[local_index].item()),
            "crps": float(task_crps[local_index].item()),
            "energy_score": float(task_energy[local_index].item()),
            "ensemble_spread": float(task_spread[local_index].item()),
            "spread_skill_ratio": float(task_ssr[local_index].item()),
            "sample_diversity_offdiag": float(
                task_diversity[local_index].item()
            ),
        }
        for suffix, (coverage, width) in interval_metrics.items():
            row[f"coverage_{suffix}"] = float(coverage[local_index].item())
            row[f"width_{suffix}"] = float(width[local_index].item())
        rows.append(row)

    return rows
