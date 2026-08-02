from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch


matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLUMN_ORDER = [
    "Exact GP oracle",
    "Gaussian TNP",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]

COLUMN_TITLES = {
    "Exact GP oracle": "Exact GP",
    "Gaussian TNP": "Gaussian TNP",
    "Dropout CRPS-TNP": "Dropout CRPS-TNP",
    "StochLN CRPS-TNP": "StochLN CRPS-TNP",
}


PREDICTION_COLOUR = "#5B4B8A"
TRUTH_COLOUR = "#333333"
INTERVAL_COLOUR = "#9B8CC2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--visible_paths", default=None, type=int)
    parser.add_argument("--dpi", default=350, type=int)
    return parser.parse_args()


def _samples_for_panel(cache: Dict, model_name: str, row_index: int) -> torch.Tensor:
    if model_name == "Exact GP oracle":
        return cache["exact_gp"]["direct_samples"]
    if row_index == 0:
        return cache["models"][model_name]["direct_samples"]
    return cache["models"][model_name]["ar_denoised_paths"]


def _panel_summary(
    cache: Dict,
    model_name: str,
    samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_name == "Exact GP oracle":
        mean = cache["exact_gp"]["mean"][0, :, 0]
        scale = cache["exact_gp"]["scale"][0, :, 0]
        z90 = 1.6448536269514722
        return mean, mean - z90 * scale, mean + z90 * scale

    paths = samples[:, 0, :, 0]
    mean = paths.mean(dim=0)
    lower = torch.quantile(paths, 0.05, dim=0)
    upper = torch.quantile(paths, 0.95, dim=0)
    return mean, lower, upper


def _global_y_limits(cache: Dict, visible_paths: int) -> Tuple[float, float]:
    values: List[torch.Tensor] = [cache["task"]["latent_truth"].reshape(-1)]

    for row_index in range(2):
        for model_name in COLUMN_ORDER:
            samples = _samples_for_panel(cache, model_name, row_index)
            _, lower, upper = _panel_summary(cache, model_name, samples)
            values.extend([lower, upper])
            values.append(samples[:visible_paths].reshape(-1))

    combined = torch.cat([value.detach().cpu().reshape(-1) for value in values])
    y_min = float(combined.min())
    y_max = float(combined.max())
    padding = 0.07 * max(y_max - y_min, 1.0)
    return y_min - padding, y_max + padding


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    visible_paths = int(
        args.visible_paths
        if args.visible_paths is not None
        else min(8, int(cache["exact_gp"]["direct_samples"].shape[0]))
    )

    x_plot = cache["task"]["x_plot"][0, :, 0]
    latent_truth = cache["task"]["latent_truth"][0, :, 0]
    xc = cache["task"]["xc"][0, :, 0]
    yc = cache["task"]["yc"][0, :, 0]

    y_limits = _global_y_limits(cache, visible_paths)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.25, 3.65),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for row_index in range(2):
        for column_index, model_name in enumerate(COLUMN_ORDER):
            ax = axes[row_index, column_index]
            samples = _samples_for_panel(cache, model_name, row_index)
            paths = samples[:, 0, :, 0]
            mean, lower, upper = _panel_summary(cache, model_name, samples)

            for path_index in range(min(visible_paths, paths.shape[0])):
                ax.plot(
                    x_plot,
                    paths[path_index],
                    color=PREDICTION_COLOUR,
                    alpha=0.20,
                    linewidth=0.70,
                    zorder=1,
                )

            ax.fill_between(
                x_plot,
                lower,
                upper,
                color=INTERVAL_COLOUR,
                alpha=0.16,
                linewidth=0.0,
                zorder=0,
            )
            ax.plot(
                x_plot,
                mean,
                color=PREDICTION_COLOUR,
                linewidth=1.25,
                zorder=3,
            )
            ax.plot(
                x_plot,
                latent_truth,
                color=TRUTH_COLOUR,
                linewidth=1.05,
                alpha=0.95,
                zorder=4,
            )
            ax.scatter(
                xc,
                yc,
                s=10,
                facecolor="black",
                edgecolor="white",
                linewidth=0.25,
                zorder=6,
            )

            ax.set_xlim(float(x_plot.min()), float(x_plot.max()))
            ax.set_ylim(*y_limits)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7, pad=1.5)

            if row_index == 0:
                ax.set_title(COLUMN_TITLES[model_name], fontsize=8.5, pad=5)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$x$", fontsize=8)

            if column_index > 0:
                ax.tick_params(labelleft=False)

    axes[0, 0].set_ylabel(r"$y$", fontsize=8)
    axes[1, 0].set_ylabel(r"$y$", fontsize=8)

    fig.text(
        0.018,
        0.735,
        "Direct",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.5,
    )
    fig.text(
        0.018,
        0.285,
        "AR-conditioned\nmean paths",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.5,
        linespacing=0.95,
    )

    legend_handles = [
        Line2D([0], [0], color=TRUTH_COLOUR, linewidth=1.1, label="Latent truth"),
        Line2D(
            [0],
            [0],
            color=PREDICTION_COLOUR,
            linewidth=0.8,
            alpha=0.35,
            label="Predictive paths",
        ),
        Line2D([0], [0], color=PREDICTION_COLOUR, linewidth=1.3, label="Predictive mean"),
        Patch(
            facecolor=INTERVAL_COLOUR,
            alpha=0.16,
            edgecolor="none",
            label="Central 90% interval",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.25,
            markersize=4.5,
            label="Context observations",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.01),
        ncol=5,
        frameon=False,
        fontsize=7.2,
        handlelength=1.8,
        columnspacing=1.1,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.91,
        bottom=0.16,
        wspace=0.10,
        hspace=0.13,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output.with_suffix(".pdf")
    png_path = output.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
