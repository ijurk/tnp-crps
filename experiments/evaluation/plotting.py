from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch

from tnp.data.synthetic import SyntheticBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def _to_cpu_1d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().reshape(-1)

def _first_batch_y_to_cpu_1d(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 3:
        return _to_cpu_1d(y[0, :, 0])
    if y.ndim == 2:
        return _to_cpu_1d(y[0])
    return _to_cpu_1d(y)


def _first_batch_x_to_cpu_1d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return _to_cpu_1d(x[0, :, 0])
    if x.ndim == 2:
        return _to_cpu_1d(x[0])
    return _to_cpu_1d(x)

def _realised_task_line(batch) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return sorted realised finite task values, if available.

    This is useful as a fallback for processes where no exact dense realised
    function has been stored.
    """
    if not hasattr(batch, "x") or not hasattr(batch, "y"):
        return None

    x = getattr(batch, "x")
    y = getattr(batch, "y")

    if x is None or y is None:
        return None

    x_1d = _to_cpu_1d(x[0, :, 0])
    y_1d = _to_cpu_1d(y[0, :, 0])

    order = torch.argsort(x_1d)
    return x_1d[order], y_1d[order]

def _dense_ground_truth_line(
    batch,
    x_plot: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, str]]:
    """Return dense realised ground truth line, if available.

    Priority:
      1. stored dense_ground_truth_x/y from a jointly sampled latent-fork plot;
      2. gt_pred.latent_function(x_plot), e.g. sawtooth.
    """
    gt_pred = getattr(batch, "gt_pred", None)
    if gt_pred is None:
        return None

    label = str(getattr(gt_pred, "dense_ground_truth_label", "GT realised function"))

    stored_y = getattr(gt_pred, "dense_ground_truth_y", None)
    if stored_y is not None:
        stored_x = getattr(gt_pred, "dense_ground_truth_x", x_plot)
        return (
            _first_batch_x_to_cpu_1d(stored_x),
            _first_batch_y_to_cpu_1d(stored_y),
            label,
        )

    if hasattr(gt_pred, "latent_function"):
        with torch.no_grad():
            y_plot = gt_pred.latent_function(x_plot)
        return (
            _to_cpu_1d(x_plot[0, :, 0]),
            _first_batch_y_to_cpu_1d(y_plot),
            label,
        )

    return None

def _latent_fork_plot_metadata(batch) -> Dict[str, object]:
    """Extract latent-fork metadata if present."""
    gt_pred = getattr(batch, "gt_pred", None)
    if gt_pred is None:
        return {}

    out: Dict[str, object] = {}

    if hasattr(gt_pred, "fork_locations"):
        fork_x = gt_pred.fork_locations.detach().cpu().reshape(-1)
        if fork_x.numel() > 0:
            out["fork_x"] = float(fork_x[0])

    sampled_regimes = getattr(gt_pred, "sampled_regimes", None)
    if sampled_regimes is not None:
        regime = sampled_regimes.detach().cpu().reshape(-1)
        if regime.numel() > 0:
            regime_id = int(regime[0])
            out["regime_id"] = regime_id
            if hasattr(gt_pred, "regime_name"):
                out["regime_name"] = gt_pred.regime_name(regime_id)
            else:
                out["regime_name"] = str(regime_id)

    regime_z = getattr(gt_pred, "regime_z", None)
    if regime_z is not None:
        z = regime_z.detach().cpu().reshape(-1)
        if z.numel() > 0:
            out["regime_z"] = int(z[0])

    for attr, key in [
        ("delta", "delta"),
        ("transition_width", "transition_width"),
        ("base_scale", "base_scale"),
    ]:
        if hasattr(gt_pred, attr):
            try:
                out[key] = float(getattr(gt_pred, attr))
            except TypeError:
                pass

    return out

def _prediction_bundle(row: Dict) -> Dict[str, Optional[torch.Tensor]]:
    """Return mean, quantiles and optional sample paths for plotting.

    For CRPS models:
        row["samples"] is expected with shape [M, 1, Nplot, 1].

    For Gaussian baseline:
        row should contain analytic mean/q025/q10/q25/q75/q90/q975.
    """
    samples = row.get("samples", None)

    if samples is not None:
        samples_2d = samples[:, 0, :, 0].detach().cpu()

        return {
            "sample_paths": samples_2d,
            "mean": samples_2d.mean(dim=0),
            "q025": torch.quantile(samples_2d, 0.025, dim=0),
            "q10": torch.quantile(samples_2d, 0.10, dim=0),
            "q25": torch.quantile(samples_2d, 0.25, dim=0),
            "q75": torch.quantile(samples_2d, 0.75, dim=0),
            "q90": torch.quantile(samples_2d, 0.90, dim=0),
            "q975": torch.quantile(samples_2d, 0.975, dim=0),
        }

    return {
        "sample_paths": None,
        "mean": _to_cpu_1d(row["mean"]),
        "q025": _to_cpu_1d(row["q025"]),
        "q10": _to_cpu_1d(row["q10"]),
        "q25": _to_cpu_1d(row["q25"]),
        "q75": _to_cpu_1d(row["q75"]),
        "q90": _to_cpu_1d(row["q90"]),
        "q975": _to_cpu_1d(row["q975"]),
    }

def _compute_y_limits(
    *,
    batch,
    prediction_rows: List[Dict],
    padding_fraction: float = 0.08,
    extra_values: Optional[List[torch.Tensor]] = None,
) -> Tuple[float, float]:
    values = [
        _to_cpu_1d(batch.yc[..., 0]),
        _to_cpu_1d(batch.yt[..., 0]),
    ]

    for row in prediction_rows:
        bundle = _prediction_bundle(row)
        values.extend([bundle["q025"], bundle["q975"]])

    if extra_values is not None:
        for value in extra_values:
            if value is not None:
                values.append(_to_cpu_1d(value))

    all_values = torch.cat(values)
    y_min = float(all_values.min())
    y_max = float(all_values.max())

    if y_min == y_max:
        return y_min - 1.0, y_max + 1.0

    pad = padding_fraction * (y_max - y_min)
    return y_min - pad, y_max + pad

def _normalise_training_ranges(
    training_ranges: Optional[List[List[float]]],
) -> List[Tuple[float, float]]:
    if training_ranges is None:
        return []

    out = []

    for item in training_ranges:
        if len(item) != 2:
            raise ValueError(
                f"Each training range must have length 2, got {item}."
            )

        lo = float(item[0])
        hi = float(item[1])

        if hi <= lo:
            raise ValueError(
                f"Invalid training range [{lo}, {hi}]."
            )

        out.append((lo, hi))

    return out


def _shade_training_ranges(
    *,
    ax,
    training_ranges: Optional[List[List[float]]],
    x_min: float,
    x_max: float,
) -> None:
    for lo, hi in _normalise_training_ranges(training_ranges):
        lo = max(lo, x_min)
        hi = min(hi, x_max)

        if hi > lo:
            ax.axvspan(
                lo,
                hi,
                facecolor="0.93",
                edgecolor="none",
                alpha=1.0,
                zorder=0,
            )

def plot_function_comparison(
    *,
    batch,
    x_plot: torch.Tensor,
    prediction_rows: List[Dict],
    output_path_base: Path,
    title: str,
    num_sample_paths: int = 6,
    y_lim: Optional[Tuple[float, float]] = None,
    figsize_per_row: Tuple[float, float] = (8.0, 3.0),
    show_targets: bool = True,
    show_ground_truth: bool = True,
    show_realised_task: bool = True,
    show_oracle_posterior: bool = True,
    training_ranges: Optional[List[List[float]]] = None,
) -> None:
    """Plot model sample paths and empirical/analytic quantile bands for one 1D task.

    Args:
        batch: Batch with batch size 1.
        x_plot: [1, Nplot, 1] dense plotting inputs.
        prediction_rows: list of model prediction dictionaries.
        output_path_base: path without extension. Saves .png and .pdf.
    """
    if batch.xc.shape[0] != 1:
        raise ValueError("plot_function_comparison expects batch size 1.")

    output_path_base.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(prediction_rows)
    figsize = (figsize_per_row[0], figsize_per_row[1] * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    x_dense = _to_cpu_1d(x_plot[0, :, 0])
    xc = _to_cpu_1d(batch.xc[0, :, 0])
    yc = _to_cpu_1d(batch.yc[0, :, 0])
    xt = _to_cpu_1d(batch.xt[0, :, 0])
    yt = _to_cpu_1d(batch.yt[0, :, 0])

    dense_ground_truth = (
        _dense_ground_truth_line(batch, x_plot)
        if show_ground_truth
        else None
    )
    realised_task = (
        _realised_task_line(batch)
        if show_realised_task
        else None
    )
    latent_meta = _latent_fork_plot_metadata(batch)

    extra_y_values: List[torch.Tensor] = []
    if dense_ground_truth is not None:
        extra_y_values.append(dense_ground_truth[1])
    elif realised_task is not None:
        extra_y_values.append(realised_task[1])

    if y_lim is None:
        y_lim = _compute_y_limits(
            batch=batch,
            prediction_rows=prediction_rows,
            extra_values=extra_y_values,
        )

    gt_mean = None
    gt_std = None

    gt_pred = getattr(batch, "gt_pred", None)
    plot_posterior_summary = bool(
        getattr(gt_pred, "plot_posterior_summary", True)
    )

    if (
        show_ground_truth
        and show_oracle_posterior
        and plot_posterior_summary
        and isinstance(batch, SyntheticBatch)
        and batch.gt_pred is not None
    ):
        with torch.no_grad():
            gt_mean, gt_std, _ = batch.gt_pred(
                xc=batch.xc,
                yc=batch.yc,
                xt=x_plot,
            )
        gt_mean = _to_cpu_1d(gt_mean[0])
        gt_std = _to_cpu_1d(gt_std[0])

    for ax, row in zip(axes, prediction_rows):
        model_name = row["name"]
        bundle = _prediction_bundle(row)

        sample_paths = bundle["sample_paths"]
        mean = bundle["mean"]
        q025 = bundle["q025"]
        q975 = bundle["q975"]

        show_sample_paths = bool(row.get("show_sample_paths", True))

        sample_path_label = str(
            row.get(
                "sample_path_label",
                "Predictive sample paths",
            )
        )
        
        interval_label = str(
            row.get(
                "interval_label",
                "95% interval",
            )
        )

        # Shade the training / interpolation window(s).
        # White regions are extrapolation.
        if training_ranges is not None:
            x_min = float(x_dense.min())
            x_max = float(x_dense.max())

            for range_item in training_ranges:
                if len(range_item) != 2:
                    raise ValueError(
                        f"Each training range must have length 2, got {range_item}."
                    )

                lo = float(range_item[0])
                hi = float(range_item[1])

                if hi <= lo:
                    raise ValueError(
                        f"Invalid training range [{lo}, {hi}]."
                    )

                lo = max(lo, x_min)
                hi = min(hi, x_max)

                if hi > lo:
                    ax.axvspan(
                        lo,
                        hi,
                        color="0.93",
                        alpha=1.0,
                        zorder=-10,
                    )

        if show_sample_paths and sample_paths is not None:
            max_paths = min(num_sample_paths, sample_paths.shape[0])
            for sample_idx in range(max_paths):
                ax.plot(
                    x_dense,
                    sample_paths[sample_idx],
                    color="0.45",
                    alpha=0.30,
                    linewidth=0.80,
                    label=sample_path_label if sample_idx == 0 else None,
                    zorder=1,
                )

        ax.fill_between(
            x_dense,
            q025,
            q975,
            alpha=0.20,
            label=interval_label,
            zorder=0,
        )

        # Predictive mean, matching RMSE evaluation.
        ax.plot(
            x_dense,
            mean,
            linewidth=2.2,
            label="Predictive mean",
            zorder=3,
        )

        if dense_ground_truth is not None:
            gt_x, gt_y, gt_label = dense_ground_truth
            ax.plot(
                gt_x,
                gt_y,
                color="tab:orange",
                linewidth=2.0,
                alpha=0.95,
                label=gt_label,
                zorder=4,
            )
        elif realised_task is not None:
            realised_x, realised_y = realised_task
            ax.plot(
                realised_x,
                realised_y,
                color="tab:orange",
                linewidth=1.4,
                alpha=0.85,
                label="Realised task values",
                zorder=2,
            )

        if gt_mean is not None and gt_std is not None:
            ax.plot(
                x_dense,
                gt_mean,
                linestyle="--",
                linewidth=1.8,
                color="tab:purple",
                label="GT posterior mean",
                zorder=3,
            )
            ax.plot(
                x_dense,
                gt_mean + 2.0 * gt_std,
                linestyle="--",
                linewidth=1.2,
                color="tab:purple",
                alpha=0.8,
                zorder=3,
            )
            ax.plot(
                x_dense,
                gt_mean - 2.0 * gt_std,
                linestyle="--",
                linewidth=1.2,
                color="tab:purple",
                alpha=0.8,
                label="GT ±2 std",
                zorder=3,
            )

        if "fork_x" in latent_meta:
            ax.axvline(
                float(latent_meta["fork_x"]),
                color="tab:green",
                linestyle=":",
                linewidth=1.8,
                alpha=0.9,
                label="Fork",
                zorder=4,
            )

        ax.scatter(
            xc,
            yc,
            color="black",
            s=24,
            zorder=6,
            label="Context",
        )

        if show_targets:
            ax.scatter(
                xt,
                yt,
                color="tab:red",
                s=12,
                alpha=0.35,
                zorder=5,
                label="Targets",
            )

        ax.set_ylim(y_lim)
        ax.set_xlim(float(x_dense.min()), float(x_dense.max()))
        ax.set_ylabel(model_name)
        ax.grid(True, alpha=0.35)

    title_bits = [title]

    if "regime_id" in latent_meta:
        if "regime_z" in latent_meta:
            title_bits.append(
                f"regime={latent_meta['regime_name']} "
                f"(z={int(latent_meta['regime_z']):+d})"
            )
        else:
            title_bits.append(
                f"regime={latent_meta['regime_name']} "
                f"(id={latent_meta['regime_id']})"
            )
            
    if "fork_x" in latent_meta:
        title_bits.append(f"fork x0={latent_meta['fork_x']:.3f}")

    if "delta" in latent_meta:
        title_bits.append(f"delta={latent_meta['delta']:.2f}")

    if "transition_width" in latent_meta:
        title_bits.append(f"tw={latent_meta['transition_width']:.2f}")

    axes[0].set_title(" | ".join(title_bits))
    axes[-1].set_xlabel("x")

    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label and label is not None:
            by_label[label] = handle

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(5, max(1, len(by_label))),
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    png_path = output_path_base.with_suffix(".png")
    pdf_path = output_path_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

def plot_predictive_histogram_comparison(
    *,
    batch,
    x_hist: torch.Tensor,
    prediction_rows: List[Dict],
    output_path_base: Path,
    title: str,
    num_oracle_samples: int = 2048,
    bins: int = 50,
) -> None:
    """Plot predictive histograms as a model-by-location grid.

    Layout:
        rows    = models
        columns = selected x locations

    Each panel overlays only:
        - oracle posterior samples, if available
        - the corresponding model samples

    This is much cleaner than overlaying all models in the same panel.
    """
    import numpy as np

    if batch.xc.shape[0] != 1:
        raise ValueError("plot_predictive_histogram_comparison expects batch size 1.")

    output_path_base.parent.mkdir(parents=True, exist_ok=True)

    x_values = _to_cpu_1d(x_hist[0, :, 0])
    num_locations = int(x_values.numel())
    num_models = len(prediction_rows)

    fig, axes = plt.subplots(
        num_models,
        num_locations,
        figsize=(4.2 * num_locations, 2.6 * num_models),
        squeeze=False,
        sharex="col",
        sharey="col",
    )

    oracle_samples = None
    if (
        isinstance(batch, SyntheticBatch)
        and batch.gt_pred is not None
        and hasattr(batch.gt_pred, "predictive_samples")
    ):
        with torch.no_grad():
            oracle_samples = batch.gt_pred.predictive_samples(
                batch.xc,
                batch.yc,
                x_hist,
                num_samples=num_oracle_samples,
            )
        oracle_samples = oracle_samples.detach().cpu()  # [M, 1, K, 1]

    # Compute common bin edges per column/location.
    # This makes row-by-row comparison visually fair.
    column_bin_edges = []

    for loc_idx in range(num_locations):
        vals_for_column = []

        if oracle_samples is not None:
            vals_for_column.append(
                oracle_samples[:, 0, loc_idx, 0].reshape(-1).numpy()
            )

        for row in prediction_rows:
            samples = row.get("samples", None)
            if samples is not None:
                vals_for_column.append(
                    samples[:, 0, loc_idx, 0]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .numpy()
                )

        all_vals = np.concatenate(vals_for_column)
        column_bin_edges.append(np.histogram_bin_edges(all_vals, bins=bins))

    for model_idx, row in enumerate(prediction_rows):
        model_name = row["name"]
        samples = row.get("samples", None)

        if samples is None:
            continue

        for loc_idx, ax in enumerate(axes[model_idx]):
            bin_edges = column_bin_edges[loc_idx]

            if oracle_samples is not None:
                oracle_vals = (
                    oracle_samples[:, 0, loc_idx, 0]
                    .reshape(-1)
                    .numpy()
                )

                ax.hist(
                    oracle_vals,
                    bins=bin_edges,
                    density=True,
                    alpha=0.45,
                    label="Oracle",
                )

            model_vals = (
                samples[:, 0, loc_idx, 0]
                .detach()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            ax.hist(
                model_vals,
                bins=bin_edges,
                density=True,
                alpha=0.45,
                label=model_name,
            )

            if model_idx == 0:
                ax.set_title(f"x = {float(x_values[loc_idx]):.2f}")

            if loc_idx == 0:
                ax.set_ylabel(model_name)

            if model_idx == num_models - 1:
                ax.set_xlabel("y")

            ax.grid(True, alpha=0.25)

    # Collect legend entries from all panels.
    handles = []
    labels = []
    for ax in axes.reshape(-1):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(4, len(by_label)),
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(title, y=1.06, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    png_path = output_path_base.with_suffix(".png")
    pdf_path = output_path_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")