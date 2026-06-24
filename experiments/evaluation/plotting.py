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
) -> Tuple[float, float]:
    values = [
        _to_cpu_1d(batch.yc[..., 0]),
        _to_cpu_1d(batch.yt[..., 0]),
    ]

    for row in prediction_rows:
        bundle = _prediction_bundle(row)
        values.extend([bundle["q025"], bundle["q975"]])

    all_values = torch.cat(values)
    y_min = float(all_values.min())
    y_max = float(all_values.max())

    if y_min == y_max:
        return y_min - 1.0, y_max + 1.0

    pad = padding_fraction * (y_max - y_min)
    return y_min - pad, y_max + pad


def plot_function_comparison(
    *,
    batch,
    x_plot: torch.Tensor,
    prediction_rows: List[Dict],
    output_path_base: Path,
    title: str,
    num_sample_paths: int = 5,
    y_lim: Optional[Tuple[float, float]] = None,
    figsize_per_row: Tuple[float, float] = (11.0, 3.6),
    legend_fontsize: float = 17.0,
    title_fontsize: float = 17.0,
    panel_label_fontsize: float = 15.0,
    axis_label_fontsize: float = 15.0,
    tick_fontsize: float = 13.0,
    show_targets: bool = True,
    show_ground_truth: bool = True,
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

    if y_lim is None:
        y_lim = _compute_y_limits(batch=batch, prediction_rows=prediction_rows)

    gt_mean = None
    gt_std = None

    if (
        show_ground_truth
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
        q10 = bundle["q10"]
        q25 = bundle["q25"]
        q75 = bundle["q75"]
        q90 = bundle["q90"]
        q975 = bundle["q975"]

        show_sample_paths = bool(row.get("show_sample_paths", True))

        if show_sample_paths and sample_paths is not None:
            max_paths = min(num_sample_paths, sample_paths.shape[0])
            for sample_idx in range(max_paths):
                ax.plot(
                    x_dense,
                    sample_paths[sample_idx],
                    color="0.65",
                    alpha=0.20,
                    linewidth=0.6,
                    label="Predictive sample paths" if sample_idx == 0 else None,
                )

        ax.fill_between(
            x_dense,
            q025,
            q975,
            color="tab:blue",
            alpha=0.20,
            label="95% interval",
        )

        # Predictive mean, matching RMSE evaluation.
        ax.plot(
            x_dense,
            mean,
            color="tab:blue",
            linewidth=2.2,
            label="Predictive mean",
        )

        if gt_mean is not None and gt_std is not None:
            ax.plot(
                x_dense,
                gt_mean,
                linestyle="--",
                linewidth=1.8,
                color="tab:purple",
                label="GT posterior mean",
            )
            ax.plot(
                x_dense,
                gt_mean + 2.0 * gt_std,
                linestyle="--",
                linewidth=1.2,
                color="tab:purple",
                alpha=0.8,
            )
            ax.plot(
                x_dense,
                gt_mean - 2.0 * gt_std,
                linestyle="--",
                linewidth=1.2,
                color="tab:purple",
                alpha=0.8,
                label="GT ±2 std",
            )

        ax.scatter(
            xc,
            yc,
            color="black",
            s=24,
            zorder=5,
            label="Context",
        )

        if show_targets:
            ax.scatter(
                xt,
                yt,
                color="tab:red",
                s=12,
                alpha=0.35,
                zorder=4,
                label="Targets",
            )

        # Lightly shade extrapolation regions.
        ax.axvspan(-4.0, -2.0, color="0.95", zorder=-10)
        ax.axvspan(2.0, 4.0, color="0.95", zorder=-10)

        ax.set_ylim(y_lim)
        ax.set_xlim(float(x_dense.min()), float(x_dense.max()))
        ax.set_ylabel(model_name,fontsize=panel_label_fontsize,labelpad=10)
        ax.tick_params(axis="both",labelsize=tick_fontsize)
        ax.grid(True, alpha=0.35)

    axes[0].set_title(title, fontsize=title_fontsize, pad=10)
    axes[-1].set_xlabel("x", fontsize=axis_label_fontsize, labelpad=8)

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
        ncol=4,
        fontsize=legend_fontsize,
        markerscale=1.25,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.2,
        labelspacing=0.6,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )
    
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.87), h_pad=1.1)

    png_path = output_path_base.with_suffix(".png")
    pdf_path = output_path_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")