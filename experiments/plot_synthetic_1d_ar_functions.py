from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from tnp.data.base import Batch

from evaluate_synthetic_1d import move_batch_to_device
from evaluation.autoregressive import autoregressive_sample_model
from evaluation.plotting import plot_function_comparison
from plot_synthetic_1d_functions import (
    dataclass_replace_batch,
    get_plot_batch,
    load_models,
    make_prediction_row,
    maybe_resample_batch_for_exact_dense_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)

    # AR plotting can be expensive because it requires one model call per
    # plotted target location. These CLI options make smoke tests easy.
    parser.add_argument("--num_ar_samples", default=None, type=int)
    parser.add_argument("--points_per_unit", default=None, type=int)
    parser.add_argument("--num_sample_paths", default=None, type=int)

    parser.add_argument(
        "--target_order",
        default="ascending",
        choices=["ascending", "descending", "given"],
        type=str,
        help="Order in which target locations are autoregressively sampled.",
    )

    parser.add_argument("--max_plots", default=None, type=int)
    parser.add_argument("--only_plot", default=None, type=str)

    # Default is AR-only. This optional flag gives the old debugging layout:
    # one-shot row followed by AR row for each model.
    parser.add_argument(
        "--include_one_shot",
        action="store_true",
        help="Also include one-shot prediction rows for comparison/debugging.",
    )

    return parser.parse_args()


@torch.no_grad()
def make_ar_prediction_row(
    *,
    item: Dict[str, Any],
    plot_batch: Batch,
    num_ar_samples: int,
    target_order: str,
) -> Dict[str, Any]:
    """Create one AR prediction row for plotting."""
    samples = autoregressive_sample_model(
        model=item["model"],
        batch=plot_batch,
        num_samples=num_ar_samples,
        target_order=target_order,  # type: ignore[arg-type]
    )

    return {
        "name": f"{item['name']} AR",
        "samples": samples,
        # For AR plots, show sample paths for all models, including Gaussian.
        # AR samples are function-level rollouts, so these paths are informative.
        "show_sample_paths": bool(item.get("show_ar_sample_paths", True)),
    }


@torch.no_grad()
def make_one_ar_plot(
    *,
    batch: Batch,
    models: List[Dict[str, Any]],
    plot_spec: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_plot_samples: int,
    num_ar_samples: int,
    num_sample_paths: int,
    x_range: List[float],
    points_per_unit: int,
    target_order: str,
    include_one_shot: bool,
) -> None:
    batch = move_batch_to_device(batch, device)

    x_min = float(x_range[0])
    x_max = float(x_range[1])
    num_points = max(2, int(points_per_unit * (x_max - x_min)))

    x_plot = torch.linspace(
        x_min,
        x_max,
        num_points,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]

    # For latent fork plots, this creates/stores exact dense realised truth
    # via joint finite-dimensional GP sampling.
    #
    # For sawtooth plots, dense truth is handled by gt_pred.latent_function.
    batch = maybe_resample_batch_for_exact_dense_truth(
        batch=batch,
        x_plot=x_plot,
        enabled=bool(plot_spec.get("resample_dense_ground_truth", True)),
    )

    y_placeholder = torch.zeros(
        x_plot.shape[0],
        x_plot.shape[1],
        batch.yc.shape[-1],
        device=device,
        dtype=batch.yc.dtype,
    )

    plot_batch = dataclass_replace_batch(
        batch,
        xt=x_plot,
        yt=y_placeholder,
    )

    prediction_rows: List[Dict[str, Any]] = []

    for item in models:
        if include_one_shot:
            one_shot_row = make_prediction_row(
                item=item,
                plot_batch=plot_batch,
                num_plot_samples=num_plot_samples,
            )
            one_shot_row["name"] = f"{one_shot_row['name']} one-shot"
            prediction_rows.append(one_shot_row)

        prediction_rows.append(
            make_ar_prediction_row(
                item=item,
                plot_batch=plot_batch,
                num_ar_samples=num_ar_samples,
                target_order=target_order,
            )
        )

    dataset_label = plot_spec.get(
        "kernel",
        plot_spec.get("dataset", "native_generator"),
    )

    sampling_label = (
        f"one-shot vs AR({target_order})"
        if include_one_shot
        else f"AR({target_order})"
    )

    title = (
        f"{plot_spec['name']} | dataset={dataset_label} | "
        f"sampling={sampling_label} | "
        f"Nc={batch.xc.shape[1]} | M_AR={num_ar_samples}"
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
        show_realised_task=bool(plot_spec.get("show_realised_task", True)),
        show_oracle_posterior=bool(plot_spec.get("show_oracle_posterior", True)),
    )


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # AR plotting defaults are deliberately conservative. If the normal function
    # plot config uses points_per_unit=100 over [-4,4], AR would require 800
    # sequential model calls per model per plot. Use ar_points_per_unit in YAML
    # or --points_per_unit on the CLI to override.
    default_points_per_unit = int(
        cfg.get(
            "ar_points_per_unit",
            min(int(cfg["points_per_unit"]), 35),
        )
    )

    default_num_ar_samples = int(
        cfg.get(
            "num_ar_samples",
            min(int(cfg["num_plot_samples"]), 32),
        )
    )

    num_ar_samples = int(
        args.num_ar_samples
        if args.num_ar_samples is not None
        else default_num_ar_samples
    )

    points_per_unit = int(
        args.points_per_unit
        if args.points_per_unit is not None
        else default_points_per_unit
    )

    num_sample_paths = int(
        args.num_sample_paths
        if args.num_sample_paths is not None
        else cfg["num_sample_paths"]
    )

    with open(output_dir / "ar_plot_config_resolved.json", "w") as f:
        json.dump(
            {
                "config": cfg,
                "cli": {
                    "num_ar_samples": args.num_ar_samples,
                    "points_per_unit": args.points_per_unit,
                    "num_sample_paths": args.num_sample_paths,
                    "target_order": args.target_order,
                    "max_plots": args.max_plots,
                    "only_plot": args.only_plot,
                    "include_one_shot": args.include_one_shot,
                },
                "resolved_ar_defaults": {
                    "num_ar_samples": num_ar_samples,
                    "points_per_unit": points_per_unit,
                    "num_sample_paths": num_sample_paths,
                },
            },
            f,
            indent=2,
        )

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

    plot_specs = list(cfg["plot_specs"])

    if args.only_plot is not None:
        plot_specs = [p for p in plot_specs if p["name"] == args.only_plot]
        if not plot_specs:
            raise RuntimeError(f"No plot spec named {args.only_plot!r} found.")

    if args.max_plots is not None:
        plot_specs = plot_specs[: int(args.max_plots)]

    for plot_spec in plot_specs:
        batch = get_plot_batch(
            base_generator_config=base_generator_config,
            plot_spec=plot_spec,
            search_batches=int(cfg.get("search_batches", 4096)),
        )

        make_one_ar_plot(
            batch=batch,
            models=models,
            plot_spec=plot_spec,
            output_dir=output_dir,
            device=device,
            num_plot_samples=int(cfg["num_plot_samples"]),
            num_ar_samples=num_ar_samples,
            num_sample_paths=num_sample_paths,
            x_range=cfg["x_range"],
            points_per_unit=points_per_unit,
            target_order=args.target_order,
            include_one_shot=bool(args.include_one_shot),
        )


if __name__ == "__main__":
    main()