from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.base import Batch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn

from evaluate_synthetic_1d import (
    apply_eval_dataset_overrides,
    apply_eval_kernel,
    load_merged_config,
    load_model_state,
    move_batch_to_device,
    sample_model,
)
from evaluation.plotting import plot_function_comparison

NORMAL_Z_50 = 0.6744897501960817
NORMAL_Z_80 = 1.2815515655446004
NORMAL_Z_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def dataclass_replace_batch(batch: Batch, **kwargs) -> Batch:
    return dataclasses.replace(batch, **kwargs)


def load_models(
    *,
    model_entries: List[Dict[str, Any]],
    base_generator_config: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    loaded = []

    for model_entry in model_entries:
        model_name = model_entry["name"]
        model_config = model_entry["model_config"]
        checkpoint_path = model_entry["checkpoint_path"]
        overrides = list(model_entry.get("overrides", []) or [])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        config = load_merged_config(
            config_paths=[base_generator_config, model_config],
            overrides=overrides,
        )

        model = instantiate(config.model)
        load_model_state(model, checkpoint_path)
        model.to(device)
        model.eval()

        loaded.append(
            {
                "name": model_name,
                "model": model,
                "checkpoint_path": checkpoint_path,
                "show_sample_paths": bool(model_entry.get("show_sample_paths", True)),
            }
        )

        print(f"Loaded plotting model: {model_name}")

    return loaded


def get_plot_batch(
    *,
    base_generator_config: str,
    plot_spec: Dict[str, Any],
    search_batches: int,
) -> Batch:
    """Get deterministic batch size 1 task matching optional Nc constraints."""
    config = load_merged_config(
        config_paths=[base_generator_config],
        overrides=[],
    )

    apply_eval_kernel(config, plot_spec["kernel"])

    apply_eval_dataset_overrides(
        config,
        samples_per_eval_set=search_batches,
        eval_batch_size=1,
    )

    config.generators.test.deterministic = True
    config.generators.test.deterministic_seed = int(plot_spec.get("seed", config.misc.seed))

    generator = instantiate(config.generators.test)
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    min_nc = plot_spec.get("min_nc", None)
    max_nc = plot_spec.get("max_nc", None)
    task_index = int(plot_spec.get("task_index", 0))

    qualifying_seen = 0

    for batch_idx, batch in enumerate(loader):
        nc = int(batch.xc.shape[1])

        if min_nc is not None and nc < int(min_nc):
            continue
        if max_nc is not None and nc > int(max_nc):
            continue

        if qualifying_seen == task_index:
            print(
                f"Selected plot batch for {plot_spec['name']}: "
                f"batch_idx={batch_idx}, nc={nc}, kernel={plot_spec['kernel']}"
            )
            return batch

        qualifying_seen += 1

    raise RuntimeError(
        f"Could not find plot batch for spec={plot_spec} "
        f"within search_batches={search_batches}."
    )


@torch.no_grad()
def make_prediction_row(
    *,
    item: Dict[str, Any],
    plot_batch: Batch,
    num_plot_samples: int,
) -> Dict[str, Any]:
    """Create one prediction row for plotting.

    DirectTNP models are represented by empirical samples.
    Gaussian/non-Direct models are represented analytically by mean/std bands.
    """
    model = item["model"]

    if isinstance(model, DirectTNP):
        samples = sample_model(
            model=model,
            batch=plot_batch,
            num_eval_samples=num_plot_samples,
        )

        return {
            "name": item["name"],
            "samples": samples,
            "show_sample_paths": item.get("show_sample_paths", True),
        }

    pred_dist = np_pred_fn(
        model=model,
        batch=plot_batch,
        num_samples=num_plot_samples,
    )

    mean = pred_dist.mean
    std = pred_dist.stddev.clamp_min(1e-8)

    return {
        "name": item["name"],
        "samples": None,
        "mean": mean,
        "q025": mean - NORMAL_Z_95 * std,
        "q10": mean - NORMAL_Z_80 * std,
        "q25": mean - NORMAL_Z_50 * std,
        "q75": mean + NORMAL_Z_50 * std,
        "q90": mean + NORMAL_Z_80 * std,
        "q975": mean + NORMAL_Z_95 * std,
        "show_sample_paths": False,
    }
    

@torch.no_grad()
def make_one_plot(
    *,
    batch: Batch,
    models: List[Dict[str, Any]],
    plot_spec: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_plot_samples: int,
    num_sample_paths: int,
    x_range: List[float],
    points_per_unit: int,
) -> None:
    batch = move_batch_to_device(batch, device)

    x_min = float(x_range[0])
    x_max = float(x_range[1])
    num_points = int(points_per_unit * (x_max - x_min))

    x_plot = torch.linspace(
        x_min,
        x_max,
        num_points,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]

    y_placeholder = torch.zeros_like(x_plot)

    plot_batch = dataclass_replace_batch(
        batch,
        xt=x_plot,
        yt=y_placeholder,
    )

    prediction_rows = []

    for item in models:
        prediction_rows.append(
            make_prediction_row(
                item=item,
                plot_batch=plot_batch,
                num_plot_samples=num_plot_samples,
            )
        )

    title = (
        f"{plot_spec['name']} | kernel={plot_spec['kernel']} | "
        f"Nc={batch.xc.shape[1]}"
    )

    y_lim = plot_spec.get("y_lim", None)
    if y_lim is not None:
        y_lim = (float(y_lim[0]), float(y_lim[1]))

    output_path_base = output_dir / plot_spec["name"]

    plot_function_comparison(
        batch=batch,
        x_plot=x_plot,
        prediction_rows=prediction_rows,
        output_path_base=output_path_base,
        title=title,
        num_sample_paths=num_sample_paths,
        y_lim=y_lim,
        show_targets=bool(plot_spec.get("show_targets", True)),
        show_ground_truth=bool(plot_spec.get("show_ground_truth", True)),
    )


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "plot_config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device_name = args.device or cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but CUDA is not available.")
    device = torch.device(device_name)

    base_generator_config = cfg["base_generator_config"]

    models = load_models(
        model_entries=cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    for plot_spec in cfg["plot_specs"]:
        batch = get_plot_batch(
            base_generator_config=base_generator_config,
            plot_spec=plot_spec,
            search_batches=int(cfg.get("search_batches", 4096)),
        )

        make_one_plot(
            batch=batch,
            models=models,
            plot_spec=plot_spec,
            output_dir=output_dir,
            device=device,
            num_plot_samples=int(cfg["num_plot_samples"]),
            num_sample_paths=int(cfg["num_sample_paths"]),
            x_range=cfg["x_range"],
            points_per_unit=int(cfg["points_per_unit"]),
        )


if __name__ == "__main__":
    main()