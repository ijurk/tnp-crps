from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

from tnp.data.base import Batch

from evaluate_synthetic_1d import move_batch_to_device
from evaluation.autoregressive import (
    autoregressive_sample_model,
    denoise_autoregressive_samples,
)
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
        choices=[
            "ascending",
            "descending",
            "given",
            "nearest_context",
            "random",
        ],
        type=str,
        help="Order in which target locations are autoregressively sampled.",
    )

    parser.add_argument(
        "--stochln_noise_mode",
        default="refresh",
        choices=["refresh", "fixed"],
        type=str,
        help=(
            "How to handle StochLN noise during AR. "
            "'refresh' resamples at every AR step and is the main/default mode. "
            "'fixed' reuses one StochLN noise vector per rollout path and should "
            "be treated as an ablation."
        ),
    )

    parser.add_argument(
        "--denoise_ar_samples",
        action="store_true",
        help=(
            "After raw AR sampling, condition on each sampled rollout and plot "
            "the predictive mean as the smoothed/denoised sample path."
        ),
    )

    parser.add_argument(
        "--num_denoise_samples",
        default=32,
        type=int,
        help=(
            "Number of stochastic samples used to approximate the denoising "
            "predictive mean for DirectTNP/CRPS models."
        ),
    )

    parser.add_argument(
        "--num_ar_anchors",
        default=None,
        type=int,
        help=(
            "Number of sparse inputs used for the sequential AR rollout. "
            "If omitted, uses num_ar_anchors from the YAML. If neither is "
            "specified, AR uses the dense plotting grid as before."
        ),
    )

    parser.add_argument(
        "--training_max_nc",
        default=None,
        type=int,
        help=(
            "Maximum context size seen during training. Sparse-anchor AR "
            "enforces Nc + K <= training_max_nc."
        ),
    )

    parser.add_argument(
        "--ar_anchor_range",
        nargs=2,
        default=None,
        type=float,
        metavar=("X_MIN", "X_MAX"),
        help=(
            "Input range for sparse AR anchors. For binary fork use "
            "'--ar_anchor_range 0.1 4.0'. If omitted, uses the YAML value."
        ),
    )

    parser.add_argument(
        "--anchor_placement",
        default=None,
        choices=["linspace", "uniform_random"],
        type=str,
        help=(
            "How to place sparse AR support inputs. "
            "Use 'linspace' for the binary fork and 'uniform_random' "
            "for periodic sawtooth tasks."
        ),
    )

    parser.add_argument(
        "--anchor_seed",
        default=None,
        type=int,
        help=(
            "Random seed for uniform_random anchor placement. "
            "If omitted, uses anchor_seed from the YAML."
        ),
    )

    parser.add_argument(
        "--denoise_chunk_size",
        default=None,
        type=int,
        help=(
            "Number of AR rollout paths denoised together. Smaller values "
            "reduce GPU memory use."
        ),
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
def denoise_ar_samples_in_chunks(
    *,
    model,
    ar_batch: Batch,
    raw_samples: torch.Tensor,
    query_xt: torch.Tensor,
    num_denoise_samples: int,
    chunk_size: int,
) -> torch.Tensor:
    """Denoise AR rollout paths in memory-safe sample chunks."""
    if raw_samples.ndim != 4:
        raise ValueError(
            "Expected raw_samples [M, B, K, Dy], "
            f"got {tuple(raw_samples.shape)}."
        )

    chunk_size = int(chunk_size)

    if chunk_size < 1:
        raise ValueError(
            f"chunk_size must be at least 1, got {chunk_size}."
        )

    chunks = []

    for start in range(
        0,
        raw_samples.shape[0],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            raw_samples.shape[0],
        )

        chunk = denoise_autoregressive_samples(
            model=model,
            batch=ar_batch,
            ar_samples=raw_samples[start:end],
            query_xt=query_xt,
            num_denoise_samples=num_denoise_samples,
        )

        chunks.append(chunk)

    return torch.cat(
        chunks,
        dim=0,
    ).contiguous()
    

@torch.no_grad()
def make_ar_prediction_row(
    *,
    item: Dict[str, Any],
    ar_batch: Batch,
    query_xt: torch.Tensor,
    num_ar_samples: int,
    target_order: str,
    stochln_noise_mode: str,
    denoise_ar_samples: bool,
    num_denoise_samples: int,
    denoise_chunk_size: int,
) -> Dict[str, Any]:
    """Create one raw or denoised AR prediction row."""
    raw_samples = autoregressive_sample_model(
        model=item["model"],
        batch=ar_batch,
        num_samples=num_ar_samples,
        target_order=target_order,  # type: ignore[arg-type]
        stochln_noise_mode=stochln_noise_mode,  # type: ignore[arg-type]
    )

    if denoise_ar_samples:
        samples = denoise_ar_samples_in_chunks(
            model=item["model"],
            ar_batch=ar_batch,
            raw_samples=raw_samples,
            query_xt=query_xt,
            num_denoise_samples=num_denoise_samples,
            chunk_size=denoise_chunk_size,
        )

        row_suffix = "AR denoised"
        sample_path_label = "Denoised sample paths"
        interval_label = "95% denoised path envelope"

    else:
        # Raw samples can only be plotted directly when their support inputs
        # are the same inputs as the plotting/query grid.
        if (
            ar_batch.xt.shape != query_xt.shape
            or not torch.allclose(ar_batch.xt, query_xt)
        ):
            raise RuntimeError(
                "Raw sparse-anchor AR samples are defined only at the anchor "
                "locations and therefore cannot be plotted on the dense query "
                "grid. Use --denoise_ar_samples for sparse-anchor plotting."
            )

        samples = raw_samples
        row_suffix = "AR raw"
        sample_path_label = "Predictive sample paths"
        interval_label = "95% interval"

    return {
        "name": f"{item['name']} {row_suffix}",
        "samples": samples,
        "show_sample_paths": bool(
            item.get("show_ar_sample_paths", True)
        ),
        "sample_path_label": sample_path_label,
        "interval_label": interval_label,
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
    stochln_noise_mode: str,
    denoise_ar_samples: bool,
    num_denoise_samples: int,
    denoise_chunk_size: int,
    include_one_shot: bool,
    num_ar_anchors: Optional[int],
    training_max_nc: Optional[int],
    ar_anchor_range: Optional[List[float]],
    anchor_placement: str,
    anchor_seed: Optional[int],
) -> None:
    batch = move_batch_to_device(
        batch,
        device,
    )

    x_min = float(x_range[0])
    x_max = float(x_range[1])

    num_dense_points = max(
        2,
        int(points_per_unit * (x_max - x_min)),
    )

    x_plot = torch.linspace(
        x_min,
        x_max,
        num_dense_points,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]

    # Create an exact realised dense truth for generators that support it.
    batch = maybe_resample_batch_for_exact_dense_truth(
        batch=batch,
        x_plot=x_plot,
        enabled=bool(
            plot_spec.get(
                "resample_dense_ground_truth",
                True,
            )
        ),
    )

    dense_y_placeholder = torch.zeros(
        x_plot.shape[0],
        x_plot.shape[1],
        batch.yc.shape[-1],
        device=device,
        dtype=batch.yc.dtype,
    )

    dense_plot_batch = dataclass_replace_batch(
        batch,
        xt=x_plot,
        yt=dense_y_placeholder,
    )

    use_sparse_anchors = num_ar_anchors is not None

    if use_sparse_anchors:
        if not denoise_ar_samples:
            raise ValueError(
                "Sparse-anchor plotting requires --denoise_ar_samples because "
                "raw AR samples exist only at the sparse anchor locations."
            )

        if training_max_nc is None:
            raise ValueError(
                "Sparse-anchor AR requires training_max_nc so that "
                "Nc + K remains within the training context-size range."
            )

        requested_anchors = int(num_ar_anchors)

        if requested_anchors < 1:
            raise ValueError(
                "num_ar_anchors must be at least 1, "
                f"got {requested_anchors}."
            )

        nc = int(batch.xc.shape[1])
        max_allowed_anchors = int(training_max_nc) - nc

        if max_allowed_anchors < 1:
            raise RuntimeError(
                "No in-distribution AR anchors are available: "
                f"Nc={nc}, training_max_nc={training_max_nc}."
            )

        actual_num_anchors = min(
            requested_anchors,
            max_allowed_anchors,
        )

        if actual_num_anchors < requested_anchors:
            print(
                "Capping sparse AR anchors to remain within the training "
                "context-size range: "
                f"requested={requested_anchors}, "
                f"allowed={actual_num_anchors}, "
                f"Nc={nc}, "
                f"training_max_nc={training_max_nc}."
            )

        anchor_bounds = plot_spec.get(
            "ar_anchor_range",
            ar_anchor_range,
        )

        if anchor_bounds is None:
            anchor_bounds = x_range

        if len(anchor_bounds) != 2:
            raise ValueError(
                "ar_anchor_range must contain [x_min, x_max], "
                f"got {anchor_bounds}."
            )

        anchor_min = float(anchor_bounds[0])
        anchor_max = float(anchor_bounds[1])

        if not anchor_min < anchor_max:
            raise ValueError(
                "ar_anchor_range must satisfy x_min < x_max, "
                f"got [{anchor_min}, {anchor_max}]."
            )

        resolved_anchor_placement = str(
            plot_spec.get(
                "anchor_placement",
                anchor_placement,
            )
        )

        if resolved_anchor_placement not in (
            "linspace",
            "uniform_random",
        ):
            raise ValueError(
                "anchor_placement must be 'linspace' or "
                f"'uniform_random', got {resolved_anchor_placement!r}."
            )

        resolved_anchor_seed_value = plot_spec.get(
            "anchor_seed",
            anchor_seed,
        )

        resolved_anchor_seed = (
            None
            if resolved_anchor_seed_value is None
            else int(resolved_anchor_seed_value)
        )

        batch_size = int(batch.xc.shape[0])

        if resolved_anchor_placement == "linspace":
            x_anchor = (
                torch.linspace(
                    anchor_min,
                    anchor_max,
                    actual_num_anchors,
                    device=device,
                    dtype=batch.xc.dtype,
                )
                .view(1, actual_num_anchors, 1)
                .expand(batch_size, -1, -1)
                .contiguous()
            )

        else:
            if resolved_anchor_seed is None:
                raise ValueError(
                    "uniform_random anchor placement requires "
                    "anchor_seed in the YAML or --anchor_seed."
                )

            anchor_generator = torch.Generator(
                device="cpu",
            )

            anchor_generator.manual_seed(
                resolved_anchor_seed,
            )

            unit_anchor = torch.rand(
                batch_size,
                actual_num_anchors,
                1,
                generator=anchor_generator,
                dtype=torch.float32,
            )

            x_anchor = (
                anchor_min
                + (anchor_max - anchor_min) * unit_anchor
            ).to(
                device=device,
                dtype=batch.xc.dtype,
            )

            # Store anchors in increasing x order. When target_order="random",
            # each rollout still receives its own random AR permutation.
            x_anchor = torch.sort(
                x_anchor,
                dim=1,
            ).values.contiguous()

        anchor_y_placeholder = torch.zeros(
            x_anchor.shape[0],
            x_anchor.shape[1],
            batch.yc.shape[-1],
            device=device,
            dtype=batch.yc.dtype,
        )

        ar_batch = dataclass_replace_batch(
            batch,
            xt=x_anchor,
            yt=anchor_y_placeholder,
        )

        support_label = (
            f"K_AR={actual_num_anchors}, "
            f"anchors={resolved_anchor_placement}"
        )

    else:
        ar_batch = dense_plot_batch
        actual_num_anchors = x_plot.shape[1]
        support_label = (
            f"N_AR={actual_num_anchors}"
        )

    prediction_rows: List[Dict[str, Any]] = []

    for item in models:
        if include_one_shot:
            one_shot_row = make_prediction_row(
                item=item,
                plot_batch=dense_plot_batch,
                num_plot_samples=num_plot_samples,
            )

            one_shot_row["name"] = (
                f"{one_shot_row['name']} one-shot"
            )

            prediction_rows.append(
                one_shot_row
            )

        prediction_rows.append(
            make_ar_prediction_row(
                item=item,
                ar_batch=ar_batch,
                query_xt=x_plot,
                num_ar_samples=num_ar_samples,
                target_order=target_order,
                stochln_noise_mode=stochln_noise_mode,
                denoise_ar_samples=denoise_ar_samples,
                num_denoise_samples=num_denoise_samples,
                denoise_chunk_size=denoise_chunk_size,
            )
        )

    dataset_label = plot_spec.get(
        "kernel",
        plot_spec.get(
            "dataset",
            "native_generator",
        ),
    )

    ar_label = (
        f"AR-denoised({target_order}, noise={stochln_noise_mode})"
        if denoise_ar_samples
        else f"AR-raw({target_order}, noise={stochln_noise_mode})"
    )

    sampling_label = (
        f"one-shot vs {ar_label}"
        if include_one_shot
        else ar_label
    )

    title = (
        f"{plot_spec['name']} | "
        f"dataset={dataset_label} | "
        f"sampling={sampling_label} | "
        f"Nc={batch.xc.shape[1]} | "
        f"{support_label} | "
        f"M_AR={num_ar_samples}"
    )

    y_lim = plot_spec.get(
        "y_lim",
        None,
    )

    if y_lim is not None:
        y_lim = (
            float(y_lim[0]),
            float(y_lim[1]),
        )

    output_path_base = (
        output_dir
        / plot_spec["name"]
    )

    plot_function_comparison(
        batch=batch,
        x_plot=x_plot,
        prediction_rows=prediction_rows,
        output_path_base=output_path_base,
        title=title,
        num_sample_paths=num_sample_paths,
        y_lim=y_lim,
        show_targets=bool(
            plot_spec.get(
                "show_targets",
                True,
            )
        ),
        show_ground_truth=bool(
            plot_spec.get(
                "show_ground_truth",
                True,
            )
        ),
        show_realised_task=bool(
            plot_spec.get(
                "show_realised_task",
                True,
            )
        ),
        show_oracle_posterior=bool(
            plot_spec.get(
                "show_oracle_posterior",
                True,
            )
        ),
        training_ranges=plot_spec.get(
            "training_ranges",
            None,
        ),
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


    num_ar_anchors = None

    if args.num_ar_anchors is not None:
        num_ar_anchors = int(args.num_ar_anchors)
    elif cfg.get("num_ar_anchors", None) is not None:
        num_ar_anchors = int(cfg["num_ar_anchors"])
    
    
    training_max_nc = None
    
    if args.training_max_nc is not None:
        training_max_nc = int(args.training_max_nc)
    elif cfg.get("training_max_nc", None) is not None:
        training_max_nc = int(cfg["training_max_nc"])
    
    
    ar_anchor_range = (
        list(args.ar_anchor_range)
        if args.ar_anchor_range is not None
        else cfg.get("ar_anchor_range", None)
    )
    
    if ar_anchor_range is not None:
        ar_anchor_range = [
            float(value)
            for value in ar_anchor_range
        ]
    
    
    anchor_placement = str(
        args.anchor_placement
        if args.anchor_placement is not None
        else cfg.get("anchor_placement", "linspace")
    )

    if anchor_placement not in (
        "linspace",
        "uniform_random",
    ):
        raise ValueError(
            "anchor_placement must be 'linspace' or "
            f"'uniform_random', got {anchor_placement!r}."
        )

    anchor_seed = (
        int(args.anchor_seed)
        if args.anchor_seed is not None
        else (
            int(cfg["anchor_seed"])
            if cfg.get("anchor_seed", None) is not None
            else None
        )
    )


    denoise_chunk_size = int(
        args.denoise_chunk_size
        if args.denoise_chunk_size is not None
        else cfg.get("denoise_chunk_size", 16)
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
                    "stochln_noise_mode": args.stochln_noise_mode,
                    "denoise_ar_samples": args.denoise_ar_samples,
                    "num_denoise_samples": args.num_denoise_samples,
                    "max_plots": args.max_plots,
                    "only_plot": args.only_plot,
                    "include_one_shot": args.include_one_shot,
                    "num_ar_anchors": args.num_ar_anchors,
                    "training_max_nc": args.training_max_nc,
                    "ar_anchor_range": args.ar_anchor_range,
                    "anchor_placement": args.anchor_placement,
                    "anchor_seed": args.anchor_seed,
                    "denoise_chunk_size": args.denoise_chunk_size,
                },
                "resolved_ar_defaults": {
                    "num_ar_samples": num_ar_samples,
                    "points_per_unit": points_per_unit,
                    "num_sample_paths": num_sample_paths,
                    "num_ar_anchors": num_ar_anchors,
                    "training_max_nc": training_max_nc,
                    "ar_anchor_range": ar_anchor_range,
                    "anchor_placement": anchor_placement,
                    "anchor_seed": anchor_seed,
                    "denoise_chunk_size": denoise_chunk_size,
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
            stochln_noise_mode=args.stochln_noise_mode,
            denoise_ar_samples=bool(args.denoise_ar_samples),
            num_denoise_samples=int(args.num_denoise_samples),
            include_one_shot=bool(args.include_one_shot),
            denoise_chunk_size=denoise_chunk_size,
            num_ar_anchors=num_ar_anchors,
            training_max_nc=training_max_nc,
            ar_anchor_range=ar_anchor_range,
            anchor_placement=anchor_placement,
            anchor_seed=anchor_seed,
        )


if __name__ == "__main__":
    main()