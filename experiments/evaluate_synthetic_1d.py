from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

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

from evaluation.metrics import batch_metric_rows, finalise_metric_rows


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
) -> List[Dict[str, Any]]:
    model_name = model_entry["name"]
    model_config = model_entry["model_config"]
    checkpoint_path = model_entry["checkpoint_path"]
    model_overrides = list(model_entry.get("overrides", []) or [])

    eval_name = eval_set["name"]
    kernel_name = eval_set["kernel"]

    print("=" * 80)
    print(f"Evaluating model={model_name}")
    print(f"Eval set={eval_name}, kernel={kernel_name}")
    print(f"Checkpoint={checkpoint_path}")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = load_merged_config(
        config_paths=[base_generator_config, model_config],
        overrides=model_overrides,
    )

    apply_eval_kernel(config, kernel_name)
    apply_eval_dataset_overrides(
        config,
        samples_per_eval_set=samples_per_eval_set,
        eval_batch_size=eval_batch_size,
    )

    pl.seed_everything(int(config.misc.seed))

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

    context_range = OmegaConf.to_container(config.params.context_range, resolve=True)
    alpha = float(getattr(config.params, "crps_alpha", 1.0))

    rows: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        samples = sample_model(
            model=model,
            batch=batch,
            num_eval_samples=num_eval_samples,
        )

        batch_rows = batch_metric_rows(
            samples=samples,
            target=batch.yt,
            xt=batch.xt,
            num_context=batch.xc.shape[1],
            context_range=context_range,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            eval_set=eval_name,
            alpha=alpha,
        )
        rows.extend(batch_rows)

        if batch_idx % 25 == 0:
            print(f"  processed batch {batch_idx + 1}/{generator.num_batches}")

    return rows


def main() -> None:
    args = parse_args()

    eval_config = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = args.output_dir or eval_config["output_dir"]
    num_eval_samples = args.num_eval_samples or int(eval_config["num_eval_samples"])

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

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "eval_config_resolved.json"), "w") as f:
        json.dump(eval_config, f, indent=2)

    all_rows: List[Dict[str, Any]] = []

    for model_entry in eval_config["models"]:
        for eval_set in eval_config["eval_sets"]:
            rows = evaluate_one_model_on_one_set(
                model_entry=model_entry,
                eval_set=eval_set,
                base_generator_config=eval_config["base_generator_config"],
                num_eval_samples=num_eval_samples,
                samples_per_eval_set=samples_per_eval_set,
                eval_batch_size=eval_batch_size,
                max_batches=max_batches,
                device=device,
            )
            all_rows.extend(rows)

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

    raw = pd.DataFrame(all_rows)
    raw_path = os.path.join(output_dir, "raw_metric_sums.csv")
    raw.to_csv(raw_path, index=False)

    final = finalise_metric_rows(all_rows)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    final.to_csv(metrics_path, index=False)

    print(f"Wrote raw metric sums to: {raw_path}")
    print(f"Wrote final metrics to:   {metrics_path}")

    display_cols = [
        "model_name",
        "eval_set",
        "region",
        "context_bucket",
        "rmse_pooled",
        "crps",
        "ensemble_spread",
        "spread_skill_ratio",
        "coverage_90",
        "width_90",
    ]

    print(final[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()