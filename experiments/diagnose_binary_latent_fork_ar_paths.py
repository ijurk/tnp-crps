from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from evaluate_synthetic_1d import move_batch_to_device
from evaluation.autoregressive import autoregressive_sample_model
from plot_synthetic_1d_ar_functions import denoise_ar_samples_in_chunks
from plot_synthetic_1d_functions import (
    dataclass_replace_batch,
    get_plot_batch,
    load_models,
    maybe_resample_batch_for_exact_dense_truth,
)


MODEL_GROUPS = ("old", "mixctx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--old_config", required=True, type=str)
    parser.add_argument("--new_config", required=True, type=str)

    parser.add_argument(
        "--output_dir",
        default=(
            "results/synthetic_1d/"
            "binary_latent_fork_ar_path_diagnostics"
        ),
        type=str,
    )

    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--only_plot", default=None, type=str)
    parser.add_argument("--max_plots", default=None, type=int)

    parser.add_argument("--seed", default=20260714, type=int)

    # Force low-context tasks so that Nc + K <= 32 for both old and new models.
    parser.add_argument("--min_nc", default=8, type=int)
    parser.add_argument("--max_nc", default=16, type=int)

    parser.add_argument("--num_ar_samples", default=None, type=int)
    parser.add_argument("--num_ar_anchors", default=None, type=int)
    parser.add_argument("--points_per_unit", default=None, type=int)
    parser.add_argument("--num_denoise_samples", default=32, type=int)
    parser.add_argument("--denoise_chunk_size", default=None, type=int)

    parser.add_argument(
        "--target_order",
        default="random",
        choices=[
            "ascending",
            "descending",
            "given",
            "nearest_context",
            "random",
        ],
    )

    parser.add_argument(
        "--stochln_noise_mode",
        default="refresh",
        choices=["refresh", "fixed"],
    )

    parser.add_argument(
        "--ar_anchor_range",
        nargs=2,
        default=None,
        type=float,
        metavar=("X_MIN", "X_MAX"),
    )

    # Ignore the transition when analysing branch identity.
    parser.add_argument("--branch_start", default=0.5, type=float)

    # A point is treated as clearly upper/lower only if it lies at least this
    # fraction of delta away from the oracle mixture centre.
    parser.add_argument("--deadband_fraction", default=0.25, type=float)

    # A whole path is assigned upper/lower according to its average normalised
    # branch score. Paths inside this threshold are classified as middle.
    parser.add_argument(
        "--assignment_threshold_fraction",
        default=0.50,
        type=float,
    )

    parser.add_argument("--bins", default=51, type=int)

    return parser.parse_args()


def _load_cfg(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
    )

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Expected config {path!r} to resolve to a dictionary, "
            f"got {type(cfg)}."
        )

    return cfg


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _first_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None

        return float(
            value.detach().reshape(-1)[0].cpu()
        )

    if isinstance(value, (list, tuple)):
        if not value:
            return None

        return _first_scalar(value[0])

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gt_scalar(
    gt: Any,
    names: List[str],
    fallback: float,
) -> float:
    for name in names:
        if hasattr(gt, name):
            value = _first_scalar(getattr(gt, name))

            if value is not None:
                return float(value)

    return float(fallback)


def _regime_label(batch: Any) -> str:
    gt = getattr(batch, "gt_pred", None)

    if gt is None:
        return "unknown"

    sampled_regimes = getattr(
        gt,
        "sampled_regimes",
        None,
    )

    if sampled_regimes is None:
        regime_z = _first_scalar(
            getattr(gt, "regime_z", None)
        )

        if regime_z is None:
            return "unknown"

        return "upper" if regime_z > 0 else "lower"

    regime_id = int(
        sampled_regimes.reshape(-1)[0].item()
    )

    if hasattr(gt, "regime_name"):
        return str(gt.regime_name(regime_id))

    return str(regime_id)


def _validate_configs(
    old_cfg: Dict[str, Any],
    new_cfg: Dict[str, Any],
) -> None:
    if (
        old_cfg["base_generator_config"]
        != new_cfg["base_generator_config"]
    ):
        raise ValueError(
            "Old and new configs must use the same evaluation generator. "
            f"Got {old_cfg['base_generator_config']!r} and "
            f"{new_cfg['base_generator_config']!r}."
        )

    if list(old_cfg["x_range"]) != list(new_cfg["x_range"]):
        raise ValueError(
            "Old and new configs must use the same x_range. "
            f"Got {old_cfg['x_range']} and {new_cfg['x_range']}."
        )

    old_names = [
        entry["name"]
        for entry in old_cfg["models"]
    ]

    new_names = [
        entry["name"]
        for entry in new_cfg["models"]
    ]

    if old_names != new_names:
        raise ValueError(
            "Old and new configs must list the same model names in the "
            f"same order. Got old={old_names}, new={new_names}."
        )

    old_plots = [
        entry["name"]
        for entry in old_cfg["plot_specs"]
    ]

    new_plots = [
        entry["name"]
        for entry in new_cfg["plot_specs"]
    ]

    if old_plots != new_plots:
        raise ValueError(
            "Old and new configs must list the same plot specs in the "
            f"same order. Got old={old_plots}, new={new_plots}."
        )


def _compute_path_metrics(
    *,
    samples: torch.Tensor,
    x_dense: torch.Tensor,
    oracle_mean: torch.Tensor,
    delta: float,
    branch_start: float,
    deadband_fraction: float,
    assignment_threshold_fraction: float,
    plot_name: str,
    task_index: int,
    regime: str,
    checkpoint_group: str,
    model_name: str,
) -> pd.DataFrame:
    """Return one metric row per denoised dense AR path."""
    if samples.ndim != 4:
        raise ValueError(
            f"Expected samples [M, B, N, Dy], got {tuple(samples.shape)}."
        )

    if samples.shape[1] != 1 or samples.shape[-1] != 1:
        raise ValueError(
            "This diagnostic expects plotting batch size 1 and Dy=1, "
            f"got {tuple(samples.shape)}."
        )

    x = (
        x_dense[0, :, 0]
        .detach()
        .float()
        .cpu()
    )

    centre = (
        oracle_mean[0]
        .reshape(-1)
        .detach()
        .float()
        .cpu()
    )

    y = (
        samples[:, 0, :, 0]
        .detach()
        .float()
        .cpu()
    )

    post_mask = x >= float(branch_start)

    if int(post_mask.sum()) < 3:
        raise ValueError(
            "Need at least three dense post-fork points. "
            f"branch_start={branch_start}, "
            f"selected={int(post_mask.sum())}."
        )

    x_post = x[post_mask]
    y_post = y[:, post_mask]

    diff_post = (
        y_post
        - centre[post_mask].unsqueeze(0)
    )

    delta_abs = max(
        abs(float(delta)),
        1.0e-8,
    )

    deadband = (
        float(deadband_fraction)
        * delta_abs
    )

    assignment_threshold = (
        float(assignment_threshold_fraction)
        * delta_abs
    )

    branch_score = diff_post.mean(dim=1)
    branch_score_norm = branch_score / delta_abs

    # -1 = clearly below centre, +1 = clearly above centre,
    #  0 = inside the middle deadband.
    side = torch.zeros_like(diff_post)
    side[diff_post > deadband] = 1.0
    side[diff_post < -deadband] = -1.0

    dx = float(
        (x_post[1] - x_post[0]).abs()
    )

    rows: List[Dict[str, Any]] = []

    for sample_idx in range(samples.shape[0]):
        score = float(branch_score[sample_idx])
        score_norm = float(
            branch_score_norm[sample_idx]
        )

        if score > assignment_threshold:
            assignment = "upper"
        elif score < -assignment_threshold:
            assignment = "lower"
        else:
            assignment = "middle"

        side_i = side[sample_idx]
        active = side_i[side_i != 0]
        active_count = int(active.numel())

        if active_count >= 2:
            switch_count = int(
                (
                    active[1:]
                    != active[:-1]
                ).sum().item()
            )

            zero_switch = int(switch_count == 0)

            active_coherence = float(
                active.sum().abs()
                / active_count
            )

        elif active_count == 1:
            switch_count = 0
            zero_switch = 0
            active_coherence = 1.0

        else:
            switch_count = 0
            zero_switch = 0
            active_coherence = 0.0

        active_fraction = (
            active_count
            / int(side_i.numel())
        )

        middle_fraction = (
            1.0 - active_fraction
        )

        # Includes the middle points as zeros, so this is conservative.
        side_consistency = float(
            side_i.sum().abs()
            / side_i.numel()
        )

        y_i = y_post[sample_idx]

        first_difference_mae = float(
            torch.diff(y_i).abs().mean()
        )

        second_difference_mae = float(
            torch.diff(y_i, n=2).abs().mean()
        )

        rows.append(
            {
                "plot_name": plot_name,
                "task_index": int(task_index),
                "regime": regime,
                "checkpoint_group": checkpoint_group,
                "model_name": model_name,
                "sample_idx": int(sample_idx),
                "num_postfork_points": int(post_mask.sum()),
                "branch_start": float(branch_start),
                "delta": float(delta_abs),
                "branch_score": score,
                "branch_score_norm": score_norm,
                "abs_branch_score_norm": abs(score_norm),
                "assignment": assignment,
                "active_fraction": float(active_fraction),
                "middle_fraction": float(middle_fraction),
                "active_coherence": float(active_coherence),
                "side_consistency": float(side_consistency),
                "switch_count": int(switch_count),
                "zero_switch": int(zero_switch),
                "first_difference_mae": first_difference_mae,
                "second_difference_mae": second_difference_mae,
                "dense_dx": dx,
            }
        )

    return pd.DataFrame(rows)


def _summarise_group(
    group: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "num_paths": int(len(group)),
        "branch_score_norm_mean": (
            group["branch_score_norm"].mean()
        ),
        "branch_score_norm_std": (
            group["branch_score_norm"].std(ddof=1)
        ),
        "branch_score_norm_q10": (
            group["branch_score_norm"].quantile(0.10)
        ),
        "branch_score_norm_q50": (
            group["branch_score_norm"].quantile(0.50)
        ),
        "branch_score_norm_q90": (
            group["branch_score_norm"].quantile(0.90)
        ),
        "mean_abs_branch_score_norm": (
            group["abs_branch_score_norm"].mean()
        ),
        "frac_upper": (
            group["assignment"] == "upper"
        ).mean(),
        "frac_middle": (
            group["assignment"] == "middle"
        ).mean(),
        "frac_lower": (
            group["assignment"] == "lower"
        ).mean(),
        "zero_switch_fraction": (
            group["zero_switch"].mean()
        ),
        "mean_switch_count": (
            group["switch_count"].mean()
        ),
        "mean_active_fraction": (
            group["active_fraction"].mean()
        ),
        "mean_middle_fraction": (
            group["middle_fraction"].mean()
        ),
        "mean_active_coherence": (
            group["active_coherence"].mean()
        ),
        "mean_side_consistency": (
            group["side_consistency"].mean()
        ),
        "mean_first_difference_mae": (
            group["first_difference_mae"].mean()
        ),
        "mean_second_difference_mae": (
            group["second_difference_mae"].mean()
        ),
    }


def _summarise(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    detailed_keys = [
        "plot_name",
        "task_index",
        "regime",
        "checkpoint_group",
        "model_name",
    ]

    for keys, group in df.groupby(
        detailed_keys,
        sort=False,
    ):
        row = dict(
            zip(detailed_keys, keys)
        )

        row.update(
            _summarise_group(group)
        )

        rows.append(row)

    # Aggregate across all evaluated tasks.
    for keys, group in df.groupby(
        ["checkpoint_group", "model_name"],
        sort=False,
    ):
        checkpoint_group, model_name = keys

        row = {
            "plot_name": "ALL_TASKS",
            "task_index": -1,
            "regime": "mixed",
            "checkpoint_group": checkpoint_group,
            "model_name": model_name,
        }

        row.update(
            _summarise_group(group)
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _plot_branch_histograms(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    plot_name: str,
    model_names: List[str],
    bins: int,
    assignment_threshold_fraction: float,
) -> None:
    plot_df = df[
        df["plot_name"] == plot_name
    ]

    fig, axes = plt.subplots(
        len(model_names),
        len(MODEL_GROUPS),
        figsize=(
            4.0 * len(MODEL_GROUPS),
            2.6 * len(model_names),
        ),
        squeeze=False,
        sharex=True,
    )

    all_scores = (
        plot_df["branch_score_norm"]
        .to_numpy()
    )

    if all_scores.size == 0:
        plt.close(fig)
        return

    q_low, q_high = np.quantile(
        all_scores,
        [0.01, 0.99],
    )

    limit = max(
        1.5,
        abs(float(q_low)),
        abs(float(q_high)),
    )

    edges = np.linspace(
        -limit,
        limit,
        int(bins) + 1,
    )

    for row_idx, model_name in enumerate(model_names):
        for col_idx, checkpoint_group in enumerate(MODEL_GROUPS):
            ax = axes[row_idx, col_idx]

            values = plot_df[
                (plot_df["model_name"] == model_name)
                & (
                    plot_df["checkpoint_group"]
                    == checkpoint_group
                )
            ]["branch_score_norm"].to_numpy()

            ax.hist(
                values,
                bins=edges,
                density=True,
                alpha=0.75,
            )

            ax.axvline(
                0.0,
                linewidth=1.0,
            )

            # Ideal branch scores.
            ax.axvline(
                -1.0,
                linestyle=":",
                linewidth=1.0,
            )

            ax.axvline(
                1.0,
                linestyle=":",
                linewidth=1.0,
            )

            # Upper/lower assignment thresholds.
            ax.axvline(
                -float(assignment_threshold_fraction),
                linestyle="--",
                linewidth=0.9,
            )

            ax.axvline(
                float(assignment_threshold_fraction),
                linestyle="--",
                linewidth=0.9,
            )

            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                ax.set_title(checkpoint_group)

            if col_idx == 0:
                ax.set_ylabel(model_name)

    fig.suptitle(
        f"{plot_name}: denoised sparse-anchor AR branch scores"
    )

    fig.supxlabel(
        "normalised branch score"
    )

    fig.supylabel("density")

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    png_path = (
        output_dir
        / f"{plot_name}_branch_score_histograms.png"
    )

    pdf_path = (
        output_dir
        / f"{plot_name}_branch_score_histograms.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def _plot_score_vs_switches(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    plot_name: str,
    model_names: List[str],
) -> None:
    plot_df = df[
        df["plot_name"] == plot_name
    ]

    fig, axes = plt.subplots(
        len(model_names),
        len(MODEL_GROUPS),
        figsize=(
            4.0 * len(MODEL_GROUPS),
            2.6 * len(model_names),
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_idx, model_name in enumerate(model_names):
        for col_idx, checkpoint_group in enumerate(MODEL_GROUPS):
            ax = axes[row_idx, col_idx]

            sub = plot_df[
                (plot_df["model_name"] == model_name)
                & (
                    plot_df["checkpoint_group"]
                    == checkpoint_group
                )
            ]

            ax.scatter(
                sub["branch_score_norm"].to_numpy(),
                sub["switch_count"].to_numpy(),
                s=12,
                alpha=0.45,
            )

            ax.axvline(
                0.0,
                linewidth=1.0,
            )

            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                ax.set_title(checkpoint_group)

            if col_idx == 0:
                ax.set_ylabel(model_name)

    fig.suptitle(
        f"{plot_name}: branch score vs dense-path switches"
    )

    fig.supxlabel(
        "normalised branch score"
    )

    fig.supylabel(
        "switch count"
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    png_path = (
        output_dir
        / f"{plot_name}_branch_score_vs_switches.png"
    )

    pdf_path = (
        output_dir
        / f"{plot_name}_branch_score_vs_switches.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()

    old_cfg = _load_cfg(args.old_config)
    new_cfg = _load_cfg(args.new_config)

    _validate_configs(
        old_cfg,
        new_cfg,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        args.device
        or new_cfg.get("device", "cuda")
    )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Requested cuda but CUDA is not available."
        )

    device = torch.device(device_name)

    num_ar_samples = int(
        args.num_ar_samples
        if args.num_ar_samples is not None
        else new_cfg.get("num_ar_samples", 128)
    )

    num_ar_anchors = int(
        args.num_ar_anchors
        if args.num_ar_anchors is not None
        else new_cfg.get("num_ar_anchors", 16)
    )

    points_per_unit = int(
        args.points_per_unit
        if args.points_per_unit is not None
        else new_cfg.get(
            "ar_points_per_unit",
            min(
                int(new_cfg["points_per_unit"]),
                32,
            ),
        )
    )

    denoise_chunk_size = int(
        args.denoise_chunk_size
        if args.denoise_chunk_size is not None
        else new_cfg.get(
            "denoise_chunk_size",
            16,
        )
    )

    anchor_range = (
        list(args.ar_anchor_range)
        if args.ar_anchor_range is not None
        else list(
            new_cfg.get(
                "ar_anchor_range",
                [0.1, 4.0],
            )
        )
    )

    anchor_min, anchor_max = map(
        float,
        anchor_range,
    )

    if not anchor_min < anchor_max:
        raise ValueError(
            f"Invalid ar_anchor_range={anchor_range}; "
            "expected min < max."
        )

    old_training_max_nc = int(
        old_cfg.get("training_max_nc", 32)
    )

    new_training_max_nc = int(
        new_cfg.get("training_max_nc", 64)
    )

    # The matched comparison must stay in distribution for both groups.
    comparison_training_max_nc = min(
        old_training_max_nc,
        new_training_max_nc,
    )

    if (
        args.min_nc < 1
        or args.max_nc < args.min_nc
    ):
        raise ValueError(
            f"Invalid context bounds: min_nc={args.min_nc}, "
            f"max_nc={args.max_nc}."
        )

    if (
        int(args.max_nc)
        + num_ar_anchors
        > comparison_training_max_nc
    ):
        raise ValueError(
            "The requested comparison is context-size OOD for the old "
            f"models: max_nc={args.max_nc}, K={num_ar_anchors}, "
            f"common maximum={comparison_training_max_nc}. "
            "Reduce max_nc or num_ar_anchors."
        )

    base_generator_config = (
        old_cfg["base_generator_config"]
    )

    print("Loading old checkpoints...")

    old_models = load_models(
        model_entries=old_cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    print("Loading mixed-context checkpoints...")

    new_models = load_models(
        model_entries=new_cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    old_model_names = [
        item["name"]
        for item in old_models
    ]

    new_model_names = [
        item["name"]
        for item in new_models
    ]

    if old_model_names != new_model_names:
        raise RuntimeError(
            f"Loaded model names differ: old={old_model_names}, "
            f"new={new_model_names}."
        )

    plot_specs = [
        dict(spec)
        for spec in old_cfg["plot_specs"]
    ]

    if args.only_plot is not None:
        plot_specs = [
            spec
            for spec in plot_specs
            if spec["name"] == args.only_plot
        ]

        if not plot_specs:
            raise RuntimeError(
                f"No plot spec named {args.only_plot!r} found."
            )

    if args.max_plots is not None:
        plot_specs = plot_specs[
            : int(args.max_plots)
        ]

    resolved = {
        "old_config": args.old_config,
        "new_config": args.new_config,
        "device": str(device),
        "seed": int(args.seed),
        "min_nc": int(args.min_nc),
        "max_nc": int(args.max_nc),
        "num_ar_samples": num_ar_samples,
        "num_ar_anchors": num_ar_anchors,
        "comparison_training_max_nc": (
            comparison_training_max_nc
        ),
        "points_per_unit": points_per_unit,
        "num_denoise_samples": int(
            args.num_denoise_samples
        ),
        "denoise_chunk_size": denoise_chunk_size,
        "target_order": args.target_order,
        "stochln_noise_mode": args.stochln_noise_mode,
        "ar_anchor_range": [
            anchor_min,
            anchor_max,
        ],
        "branch_start": float(args.branch_start),
        "deadband_fraction": float(
            args.deadband_fraction
        ),
        "assignment_threshold_fraction": float(
            args.assignment_threshold_fraction
        ),
        "plot_specs": [
            spec["name"]
            for spec in plot_specs
        ],
    }

    with open(
        output_dir
        / "ar_path_diagnostics_resolved.json",
        "w",
    ) as f:
        json.dump(
            resolved,
            f,
            indent=2,
        )

    all_frames: List[pd.DataFrame] = []

    x_min, x_max = map(
        float,
        old_cfg["x_range"],
    )

    num_dense_points = max(
        2,
        int(
            points_per_unit
            * (x_max - x_min)
        ),
    )

    for plot_idx, plot_spec in enumerate(plot_specs):
        # Force the same low-context regime for old and new checkpoints.
        plot_spec["min_nc"] = int(args.min_nc)
        plot_spec["max_nc"] = int(args.max_nc)

        task_seed = (
            int(args.seed)
            + 10000 * plot_idx
        )

        _set_seed(task_seed)

        # The task is generated only once and is shared by every model.
        batch = get_plot_batch(
            base_generator_config=base_generator_config,
            plot_spec=plot_spec,
            search_batches=int(
                old_cfg.get(
                    "search_batches",
                    4096,
                )
            ),
        )

        batch = move_batch_to_device(
            batch,
            device,
        )

        nc = int(batch.xc.shape[1])

        if (
            nc + num_ar_anchors
            > comparison_training_max_nc
        ):
            raise RuntimeError(
                f"Selected task {plot_spec['name']} has Nc={nc}; "
                f"Nc+K={nc + num_ar_anchors} exceeds the common "
                f"training maximum {comparison_training_max_nc}."
            )

        x_dense = torch.linspace(
            x_min,
            x_max,
            num_dense_points,
            device=device,
            dtype=batch.xc.dtype,
        )[None, :, None]

        batch = maybe_resample_batch_for_exact_dense_truth(
            batch=batch,
            x_plot=x_dense,
            enabled=bool(
                plot_spec.get(
                    "resample_dense_ground_truth",
                    True,
                )
            ),
        )

        x_anchor = torch.linspace(
            anchor_min,
            anchor_max,
            num_ar_anchors,
            device=device,
            dtype=batch.xc.dtype,
        )[None, :, None]

        anchor_y_placeholder = torch.zeros(
            1,
            num_ar_anchors,
            batch.yc.shape[-1],
            device=device,
            dtype=batch.yc.dtype,
        )

        ar_batch = dataclass_replace_batch(
            batch,
            xt=x_anchor,
            yt=anchor_y_placeholder,
        )

        gt = getattr(
            batch,
            "gt_pred",
            None,
        )

        if gt is None:
            raise RuntimeError(
                "Binary-fork path diagnostics require batch.gt_pred."
            )

        oracle_mean, _, _ = gt(
            xc=batch.xc,
            yc=batch.yc,
            xt=x_dense,
        )

        if oracle_mean.ndim == 2:
            oracle_mean = (
                oracle_mean.unsqueeze(-1)
            )

        delta = _gt_scalar(
            gt,
            [
                "delta",
                "delta_value",
                "branch_delta",
            ],
            fallback=2.0,
        )

        regime = _regime_label(batch)

        task_index = int(
            plot_spec.get(
                "task_index",
                -1,
            )
        )

        grouped_models: List[
            Tuple[str, List[Dict[str, Any]]]
        ] = [
            ("old", old_models),
            ("mixctx", new_models),
        ]

        for checkpoint_group, models in grouped_models:
            for model_idx, item in enumerate(models):
                # Paired old/new versions of the same architecture receive
                # the same target-order and random-number seed.
                rollout_seed = (
                    task_seed
                    + 100 * (model_idx + 1)
                )

                _set_seed(rollout_seed)

                print(
                    f"[{plot_spec['name']}] "
                    f"{checkpoint_group} | "
                    f"{item['name']} | "
                    f"Nc={nc} | "
                    f"K={num_ar_anchors} | "
                    f"M={num_ar_samples}"
                )

                raw_samples = autoregressive_sample_model(
                    model=item["model"],
                    batch=ar_batch,
                    num_samples=num_ar_samples,
                    target_order=args.target_order,  # type: ignore[arg-type]
                    stochln_noise_mode=args.stochln_noise_mode,  # type: ignore[arg-type]
                )

                dense_samples = denoise_ar_samples_in_chunks(
                    model=item["model"],
                    ar_batch=ar_batch,
                    raw_samples=raw_samples,
                    query_xt=x_dense,
                    num_denoise_samples=int(
                        args.num_denoise_samples
                    ),
                    chunk_size=denoise_chunk_size,
                )

                frame = _compute_path_metrics(
                    samples=dense_samples,
                    x_dense=x_dense,
                    oracle_mean=oracle_mean,
                    delta=delta,
                    branch_start=float(
                        args.branch_start
                    ),
                    deadband_fraction=float(
                        args.deadband_fraction
                    ),
                    assignment_threshold_fraction=float(
                        args.assignment_threshold_fraction
                    ),
                    plot_name=plot_spec["name"],
                    task_index=task_index,
                    regime=regime,
                    checkpoint_group=checkpoint_group,
                    model_name=item["name"],
                )

                all_frames.append(frame)

                del raw_samples
                del dense_samples

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    all_metrics = pd.concat(
        all_frames,
        ignore_index=True,
    )

    all_summary = _summarise(
        all_metrics
    )

    metrics_path = (
        output_dir
        / "dense_denoised_ar_path_metrics.csv"
    )

    summary_path = (
        output_dir
        / "dense_denoised_ar_path_summary.csv"
    )

    all_metrics.to_csv(
        metrics_path,
        index=False,
    )

    all_summary.to_csv(
        summary_path,
        index=False,
    )

    for plot_name in (
        all_metrics["plot_name"]
        .drop_duplicates()
        .tolist()
    ):
        _plot_branch_histograms(
            df=all_metrics,
            output_dir=output_dir,
            plot_name=plot_name,
            model_names=old_model_names,
            bins=int(args.bins),
            assignment_threshold_fraction=float(
                args.assignment_threshold_fraction
            ),
        )

        _plot_score_vs_switches(
            df=all_metrics,
            output_dir=output_dir,
            plot_name=plot_name,
            model_names=old_model_names,
        )

    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")

    overall = all_summary[
        all_summary["plot_name"]
        == "ALL_TASKS"
    ]

    columns = [
        "checkpoint_group",
        "model_name",
        "frac_lower",
        "frac_middle",
        "frac_upper",
        "zero_switch_fraction",
        "mean_switch_count",
        "mean_side_consistency",
        "mean_abs_branch_score_norm",
    ]

    print(
        "\n=== Overall dense denoised AR path summary ==="
    )

    print(
        overall[columns].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
