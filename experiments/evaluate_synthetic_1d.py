from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import hiyapyco
import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.base import Batch
from tnp.utils.experiment_utils import deep_convert_dict, extract_config
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn

from evaluation.metrics import (
    batch_metric_rows,
    batch_metric_rows_tabular,
    finalise_metric_rows,
    per_task_metric_rows_tabular,
    per_task_shape_rows_tabular,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--num_eval_samples", default=None, type=int)
    parser.add_argument("--samples_per_eval_set", default=None, type=int)
    parser.add_argument("--eval_batch_size", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def load_merged_config(
    *,
    config_paths: List[str],
    overrides: Optional[List[str]] = None,
):
    raw_config = deep_convert_dict(
        hiyapyco.load(
            config_paths,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    config, _ = extract_config(
        raw_config,
        config_changes=overrides or [],
        combine_default=True,
    )
    OmegaConf.resolve(config)
    return config


def apply_eval_kernel(config: Any, kernel_name: str) -> None:
    """Restrict generator to one kernel unless kernel_name == mixture."""
    if kernel_name in {"mixture", "mixed", "all"}:
        return

    if not hasattr(config, kernel_name):
        raise KeyError(
            f"Unknown kernel_name={kernel_name}. "
            f"Expected one of rbf_kernel, matern12_kernel, matern32_kernel, "
            f"matern52_kernel, periodic_kernel, or mixture."
        )

    kernel_cfg = OmegaConf.to_container(getattr(config, kernel_name), resolve=True)

    for split in ("train", "val", "test"):
        if hasattr(config.generators, split):
            config.generators[split].kernel = [kernel_cfg]


def apply_eval_dataset_overrides(
    config: Any,
    *,
    samples_per_eval_set: Optional[int],
    eval_batch_size: Optional[int],
) -> None:
    """Set evaluation sample count and batch size before generator instantiation."""
    if samples_per_eval_set is not None:
        config.generators.test.samples_per_epoch = int(samples_per_eval_set)

    if eval_batch_size is not None:
        config.generators.test.batch_size = int(eval_batch_size)


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    batch_kwargs = {}

    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
        batch_kwargs[field.name] = value

    return type(batch)(**batch_kwargs)


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    attempts = []

    attempts.append(("direct", state_dict))

    if any(key.startswith("model.") for key in state_dict.keys()):
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

    if any(key.startswith("lit_model.model.") for key in state_dict.keys()):
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

    last_error = None

    for attempt_name, candidate_state in attempts:
        try:
            model.load_state_dict(candidate_state, strict=True)
            print(f"Loaded checkpoint using state_dict mode: {attempt_name}")
            return
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to load checkpoint into model. checkpoint_path={checkpoint_path}\n"
        f"Last error:\n{last_error}"
    )


@torch.no_grad()
def sample_model(
    *,
    model: torch.nn.Module,
    batch: Batch,
    num_eval_samples: int,
) -> torch.Tensor:
    """Return samples with shape [M, B, Nt, Dy]."""
    if isinstance(model, DirectTNP):
        return model.sample(
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            num_samples=num_eval_samples,
        )

    pred_dist = np_pred_fn(
        model=model,
        batch=batch,
        num_samples=num_eval_samples,
    )
    return pred_dist.sample((num_eval_samples,))


@torch.no_grad()
def sample_model_for_shape_analysis(
    *,
    model: torch.nn.Module,
    batch: Batch,
    num_eval_samples: int,
    sample_chunk_size: int,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Generate samples and retain exact Gaussian parameters when available.

    Returns:
        samples:
            [M, B, Nt, Dy].
        gaussian_loc:
            [B, Nt, Dy] for the Gaussian TNP, otherwise None.
        gaussian_scale:
            [B, Nt, Dy] for the Gaussian TNP, otherwise None.
    """
    num_eval_samples = int(
        num_eval_samples
    )
    sample_chunk_size = int(
        sample_chunk_size
    )

    if num_eval_samples < 2:
        raise ValueError(
            "Shape analysis requires at least two samples."
        )

    if sample_chunk_size < 1:
        raise ValueError(
            "sample_chunk_size must be positive."
        )

    if isinstance(
        model,
        DirectTNP,
    ):
        chunks = []
        remaining = (
            num_eval_samples
        )

        while remaining > 0:
            chunk_size = min(
                sample_chunk_size,
                remaining,
            )

            chunk = model.sample(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                num_samples=chunk_size,
            )

            chunks.append(
                chunk
            )

            remaining -= (
                chunk_size
            )

        return (
            torch.cat(
                chunks,
                dim=0,
            ),
            None,
            None,
        )

    pred_dist = np_pred_fn(
        model=model,
        batch=batch,
        num_samples=(
            num_eval_samples
        ),
    )

    if not isinstance(
        pred_dist,
        torch.distributions.Normal,
    ):
        raise TypeError(
            "Analytic Gaussian shape analysis requires a "
            "torch.distributions.Normal prediction. "
            f"Got {type(pred_dist)}."
        )

    samples = pred_dist.sample(
        (
            num_eval_samples,
        )
    )

    return (
        samples,
        pred_dist.loc,
        pred_dist.scale,
    )


@torch.no_grad()
def sample_tabular_baseline(
    *,
    baseline_kind: str,
    batch: Batch,
    num_eval_samples: int,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return baseline samples with shape [M, B, Nt, Dy].

    Supported baselines:

    ``context_mean``
        Repeats the context target mean as a deterministic prediction.

    ``context_gaussian``
        Samples independently from a Gaussian fitted to the context targets.

    ``context_resample``
        Resamples observed context targets with replacement independently for
        every target position and ensemble member.
    """
    num_eval_samples = int(num_eval_samples)

    if num_eval_samples < 2:
        raise ValueError(
            "Tabular baseline evaluation requires at least two samples, "
            f"got {num_eval_samples}."
        )

    yc = batch.yc
    batch_size = yc.shape[0]
    num_context = yc.shape[1]
    num_targets = batch.yt.shape[1]
    dim_y = yc.shape[-1]

    context_mean = yc.mean(dim=1, keepdim=True)

    if baseline_kind == "context_mean":
        return (
            context_mean.unsqueeze(0)
            .expand(
                num_eval_samples,
                batch_size,
                num_targets,
                dim_y,
            )
            .clone()
        )

    if baseline_kind == "context_gaussian":
        context_std = yc.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(float(epsilon))

        noise = torch.randn(
            num_eval_samples,
            batch_size,
            num_targets,
            dim_y,
            device=yc.device,
            dtype=yc.dtype,
        )

        return (
            context_mean.unsqueeze(0)
            + context_std.unsqueeze(0) * noise
        )

    if baseline_kind == "context_resample":
        indices = torch.randint(
            low=0,
            high=num_context,
            size=(
                num_eval_samples,
                batch_size,
                num_targets,
            ),
            device=yc.device,
        )

        source = yc.unsqueeze(0).expand(
            num_eval_samples,
            batch_size,
            num_context,
            dim_y,
        )

        gather_indices = indices.unsqueeze(-1).expand(
            num_eval_samples,
            batch_size,
            num_targets,
            dim_y,
        )

        return torch.gather(
            source,
            dim=2,
            index=gather_indices,
        )

    raise ValueError(
        "Unknown tabular baseline kind "
        f"{baseline_kind!r}. Expected one of: "
        "'context_mean', 'context_gaussian', 'context_resample'."
    )


def resolve_training_alpha(
    *,
    model_entry: Dict[str, Any],
    config: Any,
    is_learned_model: bool,
) -> Optional[float]:
    """Resolve and validate the CRPS alpha used during training.

    Resolution order:

    1. Explicit ``training_alpha`` in the evaluation model entry.
    2. ``config.params.crps_alpha`` from the resolved model configuration.
    3. ``None`` for models or baselines not trained with CRPS.

    When both an explicit value and a configuration value are available,
    they must agree. This catches stale evaluation metadata or a missing
    training override.
    """
    config_alpha_raw = None

    if is_learned_model and hasattr(config, "params"):
        config_alpha_raw = getattr(
            config.params,
            "crps_alpha",
            None,
        )

    has_explicit_alpha = "training_alpha" in model_entry

    if has_explicit_alpha:
        resolved_raw = model_entry["training_alpha"]
    else:
        resolved_raw = config_alpha_raw

    if resolved_raw is None:
        training_alpha = None
    else:
        training_alpha = float(resolved_raw)

        if not 0.0 <= training_alpha <= 1.0:
            raise ValueError(
                "training_alpha must be in [0, 1] or null. "
                f"Got {training_alpha} for "
                f"model={model_entry.get('name', '<unnamed>')}."
            )

    if not is_learned_model and training_alpha is not None:
        raise ValueError(
            "Context baselines must use training_alpha: null. "
            f"Got {training_alpha} for "
            f"source={model_entry.get('name', '<unnamed>')}."
        )

    if has_explicit_alpha and config_alpha_raw is not None:
        config_alpha = float(config_alpha_raw)

        if (
            training_alpha is None
            or not math.isclose(
                training_alpha,
                config_alpha,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "Explicit training_alpha does not match the resolved "
                "model configuration. "
                f"model={model_entry.get('name', '<unnamed>')}, "
                f"training_alpha={training_alpha}, "
                f"config.params.crps_alpha={config_alpha}. "
                "Add the training-time alpha override to the model entry's "
                "'overrides' list or correct the metadata."
            )

    return training_alpha


def evaluate_one_model_on_one_set(
    *,
    model_entry: Dict[str, Any],
    eval_set: Dict[str, Any],
    base_generator_config: str,
    num_eval_samples: int,
    samples_per_eval_set: Optional[int],
    eval_batch_size: Optional[int],
    max_batches: Optional[int],
    device: torch.device,
    evaluation_kind: str,
    metric_alpha: float,
    sampling_seed: int,
    shape_analysis: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model_name = model_entry["name"]
    entry_kind = str(model_entry.get("kind", "model"))
    model_overrides = list(model_entry.get("overrides", []) or [])

    supported_baselines = {
        "context_mean",
        "context_gaussian",
        "context_resample",
    }

    is_learned_model = entry_kind == "model"

    if is_learned_model:
        model_config = model_entry["model_config"]
        checkpoint_path = model_entry["checkpoint_path"]
        config_paths = [
            base_generator_config,
            model_config,
        ]
    elif entry_kind in supported_baselines:
        model_config = None
        checkpoint_path = f"<baseline:{entry_kind}>"
        config_paths = [base_generator_config]
    else:
        raise ValueError(
            f"Unknown model-entry kind {entry_kind!r}. "
            "Expected 'model' or one of "
            f"{sorted(supported_baselines)}."
        )

    eval_name = eval_set["name"]
    kernel_name = eval_set.get("kernel", None)
    eval_overrides = list(eval_set.get("overrides", []) or [])

    print("=" * 80)
    print(f"Evaluating source={model_name}, kind={entry_kind}")

    if kernel_name is None:
        print(f"Eval set={eval_name}")
    else:
        print(f"Eval set={eval_name}, kernel={kernel_name}")

    if is_learned_model:
        print(f"Checkpoint={checkpoint_path}")

    print("=" * 80)

    if is_learned_model and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    config = load_merged_config(
        config_paths=config_paths,
        overrides=model_overrides + eval_overrides,
    )

    training_alpha = resolve_training_alpha(
        model_entry=model_entry,
        config=config,
        is_learned_model=is_learned_model,
    )

    if kernel_name is not None:
        apply_eval_kernel(config, kernel_name)

    apply_eval_dataset_overrides(
        config,
        samples_per_eval_set=samples_per_eval_set,
        eval_batch_size=eval_batch_size,
    )

    # Seed model construction and generator construction.
    pl.seed_everything(int(config.misc.seed))

    model = None

    if is_learned_model:
        model = instantiate(config.model)
        load_model_state(model, checkpoint_path)
        model.to(device)
        model.eval()

    generator = instantiate(config.generators.test)

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    if evaluation_kind == "synthetic_1d":
        if not hasattr(config.params, "context_range"):
            raise ValueError(
                "synthetic_1d evaluation requires params.context_range."
            )

        context_range = OmegaConf.to_container(
            config.params.context_range,
            resolve=True,
        )

    elif evaluation_kind == "tabular":
        context_range = None

    else:
        raise ValueError(
            f"Unknown evaluation_kind={evaluation_kind!r}. "
            "Expected 'synthetic_1d' or 'tabular'."
        )

    resolved_sampling_seed = int(eval_set.get("sampling_seed", sampling_seed))
    resolved_metric_alpha = float(eval_set.get("metric_alpha", metric_alpha))

    if not 0.0 <= resolved_metric_alpha <= 1.0:
        raise ValueError(
            "Resolved metric_alpha must be in [0, 1]. "
            f"Got {resolved_metric_alpha} for eval_set={eval_name}."
        )

    shape_cfg = (
        shape_analysis
        if shape_analysis is not None
        else {}
    )

    shape_enabled = bool(shape_cfg.get("enabled",False))

    if (shape_enabled and evaluation_kind != "tabular"):
        raise ValueError(
            "Shape analysis is currently supported only "
            "for tabular evaluation."
        )

    headline_num_samples = int(shape_cfg.get("headline_num_samples", num_eval_samples))

    shape_sample_counts = [
        int(value)
        for value in shape_cfg.get(
            "sample_counts",
            [
                headline_num_samples
            ],
        )
    ]

    rank_sample_count = int(shape_cfg.get("rank_sample_count", headline_num_samples))

    sample_chunk_size = int(shape_cfg.get("sample_chunk_size", headline_num_samples))

    generated_num_samples = (
        max(
            [
                headline_num_samples,
                rank_sample_count,
                *shape_sample_counts,
            ]
        )
        if shape_enabled
        else num_eval_samples
    )

    # Reset after model construction so predictive Monte Carlo randomness is
    # not affected by architecture-specific parameter initialization.
    pl.seed_everything(resolved_sampling_seed)

    rows: List[Dict[str, Any]] = []
    per_task_rows: List[Dict[str, Any]] = []
    task_index_start = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        gaussian_loc = None
        gaussian_scale = None

        if is_learned_model:
            assert model is not None

            if shape_enabled:
                (
                    samples,
                    gaussian_loc,
                    gaussian_scale,
                ) = (
                    sample_model_for_shape_analysis(
                        model=model,
                        batch=batch,
                        num_eval_samples=(
                            generated_num_samples
                        ),
                        sample_chunk_size=(
                            sample_chunk_size
                        ),
                    )
                )

                metric_samples = samples[:headline_num_samples]
            else:
                samples = sample_model(model=model, batch=batch, num_eval_samples=(num_eval_samples))
                metric_samples = (samples)

        else:
            if evaluation_kind != "tabular":
                raise ValueError(
                    "The context-based baselines are currently defined only "
                    "for tabular evaluation."
                )

            samples = sample_tabular_baseline(
                baseline_kind=entry_kind,
                batch=batch,
                num_eval_samples=num_eval_samples,
            )
            metric_samples = samples

        if evaluation_kind == "tabular":
            batch_rows = batch_metric_rows_tabular(
                samples=metric_samples,
                target=batch.yt,
                num_context=batch.xc.shape[1],
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                eval_set=eval_name,
                alpha=resolved_metric_alpha,
            )

            batch_per_task_rows = (
                per_task_metric_rows_tabular(
                    samples=metric_samples,
                    target=batch.yt,
                    num_context=batch.xc.shape[1],
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    eval_set=eval_name,
                    task_index_start=task_index_start,
                    alpha=resolved_metric_alpha,
                )
            )

            if (shape_enabled and is_learned_model):
                shape_rows = (
                    per_task_shape_rows_tabular(
                        samples=samples,
                        target=batch.yt,
                        num_context=(
                            batch.xc.shape[1]
                        ),
                        model_name=(
                            model_name
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                        eval_set=(
                            eval_name
                        ),
                        task_index_start=(
                            task_index_start
                        ),
                        sample_counts=(
                            shape_sample_counts
                        ),
                        rank_sample_count=(
                            rank_sample_count
                        ),
                        gaussian_loc=(
                            gaussian_loc
                        ),
                        gaussian_scale=(
                            gaussian_scale
                        ),
                    )
                )

                if len(shape_rows) != len(batch_per_task_rows):
                    raise RuntimeError(
                        "Shape-analysis and standard per-task row "
                        "counts do not match."
                    )

                for (standard_row,shape_row) in zip(batch_per_task_rows, shape_rows):
                    if (standard_row["task_index"] != shape_row["task_index"]):
                        raise RuntimeError(
                            "Shape-analysis task indices do not "
                            "match standard metric task indices."
                        )

                    for (key,value) in shape_row.items():
                        if key not in standard_row:
                            standard_row[key] = value

            task_index_start += int(batch.yt.shape[0])

        else:
            assert context_range is not None

            batch_rows = batch_metric_rows(
                samples=metric_samples,
                target=batch.yt,
                xt=batch.xt,
                num_context=batch.xc.shape[1],
                context_range=context_range,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                eval_set=eval_name,
                alpha=resolved_metric_alpha,
            )

            batch_per_task_rows = []

        for metric_row in batch_rows:
            metric_row["training_alpha"] = training_alpha
            metric_row["metric_alpha"] = resolved_metric_alpha

        for task_row in batch_per_task_rows:
            task_row["training_alpha"] = training_alpha
            task_row["metric_alpha"] = resolved_metric_alpha

        rows.extend(batch_rows)
        per_task_rows.extend(batch_per_task_rows)

        if batch_idx % 25 == 0:
            print(
                f"  processed batch "
                f"{batch_idx + 1}/{generator.num_batches}"
            )

    return rows, per_task_rows


def main() -> None:
    args = parse_args()

    eval_config = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = args.output_dir or eval_config["output_dir"]
    num_eval_samples = (
        args.num_eval_samples
        if args.num_eval_samples is not None
        else int(
            eval_config["num_eval_samples"]
        )
    )

    shape_analysis_cfg = (eval_config.get("shape_analysis",{}) or {})

    samples_per_eval_set = (
        args.samples_per_eval_set
        if args.samples_per_eval_set is not None
        else eval_config.get("samples_per_eval_set", None)
    )

    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else eval_config.get("eval_batch_size", None)
    )

    max_batches = args.max_batches
    if max_batches is None:
        max_batches = eval_config.get("max_batches", None)

    device_name = args.device or eval_config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but CUDA is not available.")

    device = torch.device(device_name)

    evaluation_kind = str(
        eval_config.get("evaluation_kind", "synthetic_1d")
    )

    if evaluation_kind not in {"synthetic_1d", "tabular"}:
        raise ValueError(
            f"Unknown evaluation_kind={evaluation_kind!r}."
        )

    # Every source is evaluated with the same scoring rule, independently of
    # the objective or alpha value used during training.
    metric_alpha = float(eval_config.get("metric_alpha", 1.0))

    if not 0.0 <= metric_alpha <= 1.0:
        raise ValueError(
            f"metric_alpha must be in [0, 1], got {metric_alpha}."
        )

    sampling_seed = int(eval_config.get("sampling_seed", 20260724))

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "eval_config_resolved.json"), "w") as f:
        json.dump(eval_config, f, indent=2)

    all_rows: List[Dict[str, Any]] = []
    all_per_task_rows: List[Dict[str, Any]] = []

    for model_entry in eval_config["models"]:
        for eval_set in eval_config["eval_sets"]:
            rows, per_task_rows = (
                evaluate_one_model_on_one_set(
                    model_entry=model_entry,
                    eval_set=eval_set,
                    base_generator_config=eval_config["base_generator_config"],
                    num_eval_samples=num_eval_samples,
                    samples_per_eval_set=samples_per_eval_set,
                    eval_batch_size=eval_batch_size,
                    max_batches=max_batches,
                    device=device,
                    evaluation_kind=evaluation_kind,
                    metric_alpha=metric_alpha,
                    sampling_seed=sampling_seed,
                    shape_analysis=shape_analysis_cfg,
                )
            )
            all_rows.extend(rows)
            all_per_task_rows.extend(per_task_rows)

            raw_so_far = pd.DataFrame(all_rows)
            raw_so_far.to_csv(
                os.path.join(output_dir, "raw_metric_sums_partial.csv"),
                index=False,
            )

            final_so_far = finalise_metric_rows(all_rows)
            final_so_far.to_csv(
                os.path.join(output_dir, "metrics_partial.csv"),
                index=False,
            )

            if all_per_task_rows:
                pd.DataFrame(all_per_task_rows).to_csv(
                    os.path.join(output_dir, "per_task_metrics_partial.csv"),
                    index=False,
                )

    raw = pd.DataFrame(all_rows)
    raw_path = os.path.join(output_dir, "raw_metric_sums.csv")
    raw.to_csv(raw_path, index=False)

    final = finalise_metric_rows(all_rows)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    final.to_csv(metrics_path, index=False)

    per_task_path = None

    if all_per_task_rows:
        per_task_path = os.path.join(output_dir,"per_task_metrics.csv")
        pd.DataFrame(all_per_task_rows).to_csv(per_task_path,index=False)

    print(f"Wrote raw metric sums to: {raw_path}")
    print(f"Wrote final metrics to:   {metrics_path}")
    if per_task_path is not None:
        print(f"Wrote per-task metrics to: {per_task_path}")

    display_cols = [
        "model_name",
        "training_alpha",
        "metric_alpha",
        "eval_set",
        "region",
        "context_bucket",
        "rmse_pooled",
        "crps",
        "energy_score",
        "ensemble_spread",
        "spread_skill_ratio",
        "coverage_90",
        "width_90",
    ]

    print(final[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()