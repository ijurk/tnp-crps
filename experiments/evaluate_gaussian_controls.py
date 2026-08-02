from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.synthetic import SyntheticBatch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn

from evaluate_synthetic_1d import (
    apply_eval_dataset_overrides,
    apply_eval_kernel,
    load_merged_config,
    load_model_state,
    move_batch_to_device,
)
from evaluation.gaussian_controls_metrics import (
    per_task_rows_gaussian,
    per_task_rows_sampled,
    task_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired fixed-hyperparameter Gaussian-control evaluation."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--samples_per_eval_set", default=None, type=int)
    parser.add_argument("--eval_batch_size", default=None, type=int)
    parser.add_argument("--num_eval_samples", default=None, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def _git_value(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _runtime_metadata(device: torch.device) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "created_unix_time": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "repo_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "tnp_submodule_commit": _git_value(
            ["git", "-C", "external/tnp", "rev-parse", "HEAD"]
        ),
        "git_status_short": _git_value(["git", "status", "--short"]),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        metadata.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
            }
        )

    return metadata


def _normalise_metadata(entry: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(entry.get("metadata", {}) or {})
    metadata.setdefault("training_alpha", None)
    metadata.setdefault("training_num_samples", None)
    metadata.setdefault("training_p_dropout", None)
    metadata.setdefault("training_layernorm_noise_dim", None)
    return metadata


def _sampling_seed(
    *,
    base_sampling_seed: int,
    entry: Mapping[str, Any],
    batch_index: int,
) -> int:
    """Return a source-order-independent predictive-sampling seed."""
    if "sampling_seed_offset" not in entry:
        raise KeyError(
            "Every metric_mode='sampled' source must define "
            f"sampling_seed_offset; missing for {entry.get('name')!r}."
        )

    offset = int(entry["sampling_seed_offset"])
    if offset < 1:
        raise ValueError(
            "sampling_seed_offset must be a positive integer; "
            f"got {offset} for {entry.get('name')!r}."
        )

    return int(base_sampling_seed) + 1_000_000 * offset + int(batch_index)


def _validate_sampling_seed_offsets(source_entries: List[Dict[str, Any]]) -> None:
    sampled = [
        entry
        for entry in source_entries
        if str(entry.get("metric_mode")) == "sampled"
    ]
    offsets = [int(entry.get("sampling_seed_offset", -1)) for entry in sampled]

    if any(offset < 1 for offset in offsets):
        missing = [
            str(entry.get("name"))
            for entry in sampled
            if int(entry.get("sampling_seed_offset", -1)) < 1
        ]
        raise ValueError(
            "Sampled sources require positive sampling_seed_offset values: "
            f"{missing}."
        )

    if len(offsets) != len(set(offsets)):
        raise ValueError(
            "sampling_seed_offset values must be unique across sampled sources."
        )


def _load_learned_sources(
    *,
    source_entries: List[Dict[str, Any]],
    base_generator_config: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []

    for source_index, entry in enumerate(source_entries):
        kind = str(entry.get("kind", "model"))
        if kind == "exact_gp":
            loaded.append(
                {
                    "entry": entry,
                    "model": None,
                    "source_index": source_index,
                }
            )
            continue

        if kind != "model":
            raise ValueError(
                f"Unsupported source kind {kind!r} for {entry.get('name')!r}."
            )

        checkpoint_path = str(entry["checkpoint_path"])
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        overrides = list(entry.get("overrides", []) or [])
        model_config = str(entry["model_config"])
        config = load_merged_config(
            config_paths=[base_generator_config, model_config],
            overrides=overrides,
        )

        pl.seed_everything(int(config.misc.seed), workers=False)
        model = instantiate(config.model)
        load_model_state(model, checkpoint_path)
        model.to(device)
        model.eval()

        loaded.append(
            {
                "entry": entry,
                "model": model,
                "source_index": source_index,
                "resolved_model_config": OmegaConf.to_container(
                    config,
                    resolve=True,
                ),
            }
        )

        print(
            f"Loaded source={entry['name']} metric_mode={entry['metric_mode']} "
            f"checkpoint={checkpoint_path}"
        )

    return loaded


def _ensure_last_dim(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape == target.shape:
        return x
    if x.shape == target.shape[:-1]:
        return x.unsqueeze(-1)
    raise ValueError(
        f"Cannot align tensor shape {tuple(x.shape)} with target {tuple(target.shape)}."
    )


def _validate_fixed_gp_batch(
    *,
    batch: SyntheticBatch,
    eval_set: Mapping[str, Any],
) -> None:
    predictor = batch.gt_pred
    if predictor is None or not hasattr(predictor, "kernel"):
        raise TypeError("Gaussian controls require a GP ground-truth predictor.")

    expected_lengthscale = eval_set.get("expected_lengthscale", None)
    if expected_lengthscale is not None:
        actual_lengthscale = float(
            predictor.kernel.lengthscale.detach().cpu().reshape(-1)[0].item()
        )
        if abs(actual_lengthscale - float(expected_lengthscale)) > 1.0e-6:
            raise RuntimeError(
                "Fixed lengthscale assertion failed: "
                f"expected={expected_lengthscale}, actual={actual_lengthscale}."
            )

    expected_period = eval_set.get("expected_period", None)
    if expected_period is not None:
        if not hasattr(predictor.kernel, "period_length"):
            raise RuntimeError("Expected a periodic kernel but period_length is absent.")
        actual_period = float(
            predictor.kernel.period_length.detach().cpu().reshape(-1)[0].item()
        )
        if abs(actual_period - float(expected_period)) > 1.0e-6:
            raise RuntimeError(
                "Fixed period assertion failed: "
                f"expected={expected_period}, actual={actual_period}."
            )



@torch.no_grad()
def _exact_gp_posterior_from_reference(
    *,
    batch: SyntheticBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the canonical noisy-observation GP posterior from batch.gt_pred.

    The generator attaches a GPGroundTruthPredictor containing the realised
    kernel hyperparameters and Gaussian likelihood. Reusing that predictor
    avoids maintaining a second numerically distinct exact-GP implementation.
    """
    if batch.gt_pred is None:
        raise RuntimeError("Synthetic GP batch has no gt_pred oracle.")

    mean, scale, _ = batch.gt_pred(
        xc=batch.xc,
        yc=batch.yc,
        xt=batch.xt,
        yt=batch.yt,
    )

    mean = _ensure_last_dim(mean, batch.yt)
    scale = _ensure_last_dim(scale, batch.yt).clamp_min(1.0e-12)

    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise FloatingPointError(
            "Exact-GP oracle returned non-finite values."
        )

    return (
        mean.to(device=device, non_blocking=True),
        scale.to(device=device, non_blocking=True),
    )


def _assert_exact_gp_oracle_valid(
    *,
    batch: SyntheticBatch,
    mean: torch.Tensor,
    scale: torch.Tensor,
    noise_std: float,
    verbose: bool,
) -> None:
    expected_shape = tuple(batch.yt.shape)

    if tuple(mean.shape) != expected_shape:
        raise RuntimeError(
            "Exact-GP mean has wrong shape: "
            f"expected={expected_shape}, got={tuple(mean.shape)}."
        )

    if tuple(scale.shape) != expected_shape:
        raise RuntimeError(
            "Exact-GP scale has wrong shape: "
            f"expected={expected_shape}, got={tuple(scale.shape)}."
        )

    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise FloatingPointError(
            "Exact-GP oracle returned non-finite values."
        )

    if not (scale > 0).all():
        raise FloatingPointError(
            "Exact-GP oracle returned a non-positive scale."
        )

    # The noisy predictive posterior satisfies scale >= noise_std everywhere.
    # A violation means the oracle returned the noiseless function posterior,
    # which would silently overstate every model's calibration deficit.
    min_scale = float(scale.min().item())
    if min_scale < float(noise_std) * (1.0 - 1.0e-3):
        raise FloatingPointError(
            "Exact-GP oracle scale fell below the observation-noise floor: "
            f"min_scale={min_scale:.6e} < noise_std={float(noise_std):.6e}. "
            "The oracle must include the Gaussian likelihood noise."
        )

    if verbose:
        print(
            "Exact-GP oracle check PASS: "
            "source=batch.gt_pred, "
            f"min_scale={min_scale:.3e}, "
            f"max_scale={float(scale.max().item()):.3e}, "
            f"noise_floor={float(noise_std):.3e}."
        )


def _sample_model(
    *,
    model: torch.nn.Module,
    batch: SyntheticBatch,
    num_eval_samples: int,
) -> torch.Tensor:
    if isinstance(model, DirectTNP):
        samples = model.sample(
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            num_samples=num_eval_samples,
        )
    else:
        pred_dist = np_pred_fn(
            model=model,
            batch=batch,
            num_samples=num_eval_samples,
        )
        samples = pred_dist.sample((num_eval_samples,))

    expected = (
        num_eval_samples,
        batch.yt.shape[0],
        batch.yt.shape[1],
        batch.yt.shape[2],
    )
    if tuple(samples.shape) != expected:
        raise ValueError(
            f"Predictive samples have wrong shape. Expected {expected}, "
            f"got {tuple(samples.shape)}."
        )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Predictive samples contain non-finite values.")

    return samples


def _analytic_model_distribution(
    *,
    model: torch.nn.Module,
    batch: SyntheticBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_dist = np_pred_fn(model=model, batch=batch, num_samples=None)
    if not isinstance(pred_dist, torch.distributions.Normal):
        raise TypeError(
            "metric_mode='analytic_gaussian' requires torch.distributions.Normal; "
            f"got {type(pred_dist)}."
        )

    loc = _ensure_last_dim(pred_dist.loc, batch.yt)
    scale = _ensure_last_dim(pred_dist.scale, batch.yt).clamp_min(1.0e-8)

    if not torch.isfinite(loc).all() or not torch.isfinite(scale).all():
        raise FloatingPointError("Analytic Gaussian parameters contain non-finite values.")

    return loc, scale


def _check_kernel_pairing(
    *,
    rows: pd.DataFrame,
    source_names: List[str],
    expected_tasks: int,
    eval_set_name: str,
) -> None:
    all_region = rows.loc[rows["region"] == "all"].copy()

    counts = all_region.groupby(["task_index", "task_fingerprint"])[
        "model_name"
    ].nunique()
    if not (counts == len(source_names)).all():
        bad = counts[counts != len(source_names)].head()
        raise RuntimeError(
            "Task pairing failed: not every fingerprint appears for every source.\n"
            f"{bad}"
        )

    fingerprint_counts = all_region.groupby("task_index")[
        "task_fingerprint"
    ].nunique()
    if not (fingerprint_counts == 1).all():
        bad = fingerprint_counts[fingerprint_counts != 1].head()
        raise RuntimeError(
            "Task pairing failed: a task_index maps to multiple fingerprints.\n"
            f"{bad}"
        )

    source_task_counts = all_region.groupby("model_name")["task_index"].nunique()
    if not (source_task_counts == expected_tasks).all():
        raise RuntimeError(
            f"Unexpected task count for eval_set={eval_set_name}. "
            f"Expected {expected_tasks}; got\n{source_task_counts}"
        )

    print(
        f"Pairing PASS for eval_set={eval_set_name}: "
        f"{expected_tasks} tasks x {len(source_names)} sources."
    )


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = Path(args.output_dir or cfg["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}. "
            "Use a new versioned path or pass --overwrite deliberately."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or str(cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)

    samples_per_eval_set = int(
        args.samples_per_eval_set
        if args.samples_per_eval_set is not None
        else cfg["samples_per_eval_set"]
    )
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else cfg["eval_batch_size"]
    )
    num_eval_samples = int(
        args.num_eval_samples
        if args.num_eval_samples is not None
        else cfg["num_eval_samples"]
    )
    max_batches = (
        args.max_batches
        if args.max_batches is not None
        else cfg.get("max_batches", None)
    )

    if samples_per_eval_set % eval_batch_size != 0:
        raise ValueError(
            "samples_per_eval_set must be divisible by eval_batch_size. "
            f"Got {samples_per_eval_set} and {eval_batch_size}."
        )
    if num_eval_samples < 2:
        raise ValueError("num_eval_samples must be at least two.")

    base_generator_config = str(cfg["base_generator_config"])
    source_entries = list(cfg["sources"])
    _validate_sampling_seed_offsets(source_entries)
    interval_levels = tuple(float(x) for x in cfg.get("interval_levels", [0.90]))

    loaded_sources = _load_learned_sources(
        source_entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    runtime_metadata = _runtime_metadata(device)
    resolved = dict(cfg)
    resolved.update(
        {
            "output_dir": str(output_dir),
            "samples_per_eval_set": samples_per_eval_set,
            "eval_batch_size": eval_batch_size,
            "num_eval_samples": num_eval_samples,
            "max_batches": max_batches,
            "runtime_metadata": runtime_metadata,
        }
    )

    with open(output_dir / "eval_config_resolved.json", "w") as handle:
        json.dump(resolved, handle, indent=2)

    source_names = [str(entry["name"]) for entry in source_entries]
    combined_path = output_dir / "per_task_metrics.csv"
    fingerprint_path = output_dir / "task_fingerprints.csv"
    wrote_combined_header = False
    wrote_fingerprint_header = False

    for eval_set in cfg["eval_sets"]:
        eval_set = dict(eval_set)
        eval_name = str(eval_set["name"])
        kernel_name = str(eval_set["kernel"])
        deterministic_seed = int(eval_set["deterministic_seed"])
        base_sampling_seed = int(eval_set["sampling_seed"])
        eval_overrides = list(eval_set.get("overrides", []) or [])

        generator_config = load_merged_config(
            config_paths=[base_generator_config],
            overrides=eval_overrides,
        )
        apply_eval_kernel(generator_config, kernel_name)
        apply_eval_dataset_overrides(
            generator_config,
            samples_per_eval_set=samples_per_eval_set,
            eval_batch_size=eval_batch_size,
        )
        generator_config.generators.test.deterministic = True
        generator_config.generators.test.deterministic_seed = deterministic_seed

        generator_noise_std = float(generator_config.generators.test.noise_std)

        generator = instantiate(generator_config.generators.test)
        loader = torch.utils.data.DataLoader(
            generator,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
        )

        expected_batches = int(generator.num_batches)
        if max_batches is not None:
            expected_batches = min(expected_batches, int(max_batches))
        expected_tasks = expected_batches * eval_batch_size

        print("=" * 88)
        print(
            f"EVAL SET {eval_name}: kernel={kernel_name}, tasks={expected_tasks}, "
            f"batch_size={eval_batch_size}, deterministic_seed={deterministic_seed}"
        )
        print("=" * 88)

        kernel_rows: List[Dict[str, Any]] = []
        task_index_start = 0

        for batch_index, batch_cpu in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            if not isinstance(batch_cpu, SyntheticBatch):
                raise TypeError(
                    "Gaussian controls require SyntheticBatch; "
                    f"got {type(batch_cpu)}."
                )

            _validate_fixed_gp_batch(batch=batch_cpu, eval_set=eval_set)

            fingerprints = task_fingerprints(batch_cpu)
            num_context = int(batch_cpu.xc.shape[1])
            batch = move_batch_to_device(batch_cpu, device)

            # The repository's canonical GP oracle and all learned models
            # are evaluated on the same paired batch. Exact-GP sources may be
            # analytic (closed-form marginals) or sampled (an M-member draw
            # from those marginals, scored with the same finite-ensemble
            # estimators as the CRPS models) so that headline comparisons can
            # be made under identical estimators.
            exact_sources = [
                item
                for item in loaded_sources
                if str(item["entry"].get("kind", "model")) == "exact_gp"
            ]
            if not exact_sources:
                raise RuntimeError(
                    "Gaussian controls require at least one kind='exact_gp' source."
                )
            if not any(
                str(item["entry"]["metric_mode"]) == "analytic_gaussian"
                for item in exact_sources
            ):
                raise RuntimeError(
                    "Gaussian controls require an analytic exact-GP source."
                )

            gp_mean, gp_std = _exact_gp_posterior_from_reference(
                batch=batch_cpu,
                device=device,
            )

            _assert_exact_gp_oracle_valid(
                batch=batch_cpu,
                mean=gp_mean,
                scale=gp_std,
                noise_std=generator_noise_std,
                verbose=batch_index == 0,
            )

            for exact_item in exact_sources:
                exact_entry = exact_item["entry"]
                exact_mode = str(exact_entry["metric_mode"])

                if exact_mode == "analytic_gaussian":
                    kernel_rows.extend(
                        per_task_rows_gaussian(
                            loc=gp_mean,
                            scale=gp_std,
                            target=batch.yt,
                            xt=batch.xt,
                            xc=batch.xc,
                            task_index_start=task_index_start,
                            generator_batch_index=batch_index,
                            fingerprints=fingerprints,
                            model_name=str(exact_entry["name"]),
                            source_kind="exact_gp",
                            checkpoint_path="<exact_gp_oracle>",
                            eval_set=eval_name,
                            kernel_name=kernel_name,
                            num_context=num_context,
                            metadata=_normalise_metadata(exact_entry),
                            interval_levels=interval_levels,
                        )
                    )
                elif exact_mode == "sampled":
                    source_seed = _sampling_seed(
                        base_sampling_seed=base_sampling_seed,
                        entry=exact_entry,
                        batch_index=batch_index,
                    )
                    pl.seed_everything(source_seed, workers=False)
                    marginal_noise = torch.randn(
                        (num_eval_samples,) + tuple(gp_mean.shape),
                        device=gp_mean.device,
                        dtype=gp_mean.dtype,
                    )
                    gp_samples = (
                        gp_mean.unsqueeze(0)
                        + gp_std.unsqueeze(0) * marginal_noise
                    )
                    kernel_rows.extend(
                        per_task_rows_sampled(
                            samples=gp_samples,
                            target=batch.yt,
                            xt=batch.xt,
                            xc=batch.xc,
                            task_index_start=task_index_start,
                            generator_batch_index=batch_index,
                            fingerprints=fingerprints,
                            model_name=str(exact_entry["name"]),
                            source_kind="exact_gp",
                            checkpoint_path="<exact_gp_oracle_sampled>",
                            eval_set=eval_name,
                            kernel_name=kernel_name,
                            num_context=num_context,
                            metadata=_normalise_metadata(exact_entry),
                            interval_levels=interval_levels,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown metric_mode={exact_mode!r} for "
                        f"{exact_entry['name']!r}."
                    )

            for loaded in loaded_sources:
                entry = loaded["entry"]
                if str(entry.get("kind", "model")) == "exact_gp":
                    continue

                model = loaded["model"]
                assert model is not None

                # Explicit per-source offsets keep predictive Monte Carlo
                # reproducible even if sources are reordered or diagnostics are added.
                source_seed = _sampling_seed(
                    base_sampling_seed=base_sampling_seed,
                    entry=entry,
                    batch_index=batch_index,
                ) if str(entry["metric_mode"]) == "sampled" else base_sampling_seed
                pl.seed_everything(source_seed, workers=False)

                metric_mode = str(entry["metric_mode"])
                metadata = _normalise_metadata(entry)

                if metric_mode == "analytic_gaussian":
                    loc, scale = _analytic_model_distribution(
                        model=model,
                        batch=batch,
                    )
                    source_rows = per_task_rows_gaussian(
                        loc=loc,
                        scale=scale,
                        target=batch.yt,
                        xt=batch.xt,
                        xc=batch.xc,
                        task_index_start=task_index_start,
                        generator_batch_index=batch_index,
                        fingerprints=fingerprints,
                        model_name=str(entry["name"]),
                        source_kind="model",
                        checkpoint_path=str(entry["checkpoint_path"]),
                        eval_set=eval_name,
                        kernel_name=kernel_name,
                        num_context=num_context,
                        metadata=metadata,
                        interval_levels=interval_levels,
                    )

                elif metric_mode == "sampled":
                    samples = _sample_model(
                        model=model,
                        batch=batch,
                        num_eval_samples=num_eval_samples,
                    )
                    source_rows = per_task_rows_sampled(
                        samples=samples,
                        target=batch.yt,
                        xt=batch.xt,
                        xc=batch.xc,
                        task_index_start=task_index_start,
                        generator_batch_index=batch_index,
                        fingerprints=fingerprints,
                        model_name=str(entry["name"]),
                        source_kind="model",
                        checkpoint_path=str(entry["checkpoint_path"]),
                        eval_set=eval_name,
                        kernel_name=kernel_name,
                        num_context=num_context,
                        metadata=metadata,
                        interval_levels=interval_levels,
                    )
                else:
                    raise ValueError(
                        f"Unknown metric_mode={metric_mode!r} for {entry['name']}."
                    )

                kernel_rows.extend(source_rows)

            task_index_start += int(batch_cpu.yt.shape[0])

            if batch_index % 25 == 0 or batch_index + 1 == expected_batches:
                print(
                    f"  {eval_name}: processed batch {batch_index + 1}/"
                    f"{expected_batches}; tasks={task_index_start}"
                )

        kernel_df = pd.DataFrame(kernel_rows)
        _check_kernel_pairing(
            rows=kernel_df,
            source_names=source_names,
            expected_tasks=expected_tasks,
            eval_set_name=eval_name,
        )

        kernel_path = output_dir / f"per_task_metrics_{eval_name}.csv"
        kernel_df.to_csv(kernel_path, index=False)

        kernel_df.to_csv(
            combined_path,
            mode="a",
            header=not wrote_combined_header,
            index=False,
        )
        wrote_combined_header = True

        fingerprint_df = (
            kernel_df.loc[
                (kernel_df["region"] == "all")
                & (kernel_df["model_name"] == source_names[0]),
                [
                    "eval_set",
                    "kernel_name",
                    "task_index",
                    "generator_batch_index",
                    "within_batch_index",
                    "num_context",
                    "num_targets",
                    "task_fingerprint",
                ],
            ]
            .drop_duplicates()
            .sort_values("task_index")
        )
        fingerprint_df.to_csv(
            fingerprint_path,
            mode="a",
            header=not wrote_fingerprint_header,
            index=False,
        )
        wrote_fingerprint_header = True

        print(f"Wrote {kernel_path}")

        del kernel_rows, kernel_df, loader, generator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Wrote combined per-task metrics: {combined_path}")
    print(f"Wrote task fingerprints:        {fingerprint_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
