from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260923)
    parser.add_argument(
        "--bootstrap_chunk_size",
        type=int,
        default=128,
        help="Number of bootstrap replicates processed per host-memory chunk.",
    )
    return parser.parse_args()


def _paired_arrays(frame: pd.DataFrame, models: List[str], column: str) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    tasks = sorted(frame["task_index"].unique())
    for model in models:
        subset = (
            frame.loc[frame["model_name"] == model, ["task_index", column]]
            .set_index("task_index")
            .reindex(tasks)
        )
        if subset[column].isna().any():
            raise ValueError(f"Missing {column!r} values for model {model!r}.")
        arrays[model] = subset[column].to_numpy(dtype=float)
    return arrays


def _bootstrap_components(
    frame: pd.DataFrame,
    *,
    models: List[str],
    gaussian_model: str,
    sample_counts: List[int],
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = sorted(frame["task_index"].unique())
    if len(tasks) < 2:
        raise ValueError("At least two paired tasks are required.")
    if replicates < 1 or bootstrap_chunk_size < 1:
        raise ValueError("Bootstrap replicate and chunk counts must be positive.")

    gaussian_row = (
        frame.loc[
            frame["model_name"] == gaussian_model,
            ["task_index", "crps_gaussian_analytic"],
        ]
        .set_index("task_index")
        .reindex(tasks)
    )
    if gaussian_row["crps_gaussian_analytic"].isna().any():
        raise ValueError("Analytic Gaussian baseline CRPS is incomplete.")
    s_gaussian = gaussian_row["crps_gaussian_analytic"].to_numpy(dtype=float)

    components_by_key: Dict[tuple[str, int, str], np.ndarray] = {}
    shape_by_key: Dict[tuple[str, int], np.ndarray] = {}

    for sample_count in sample_counts:
        empirical = _paired_arrays(
            frame,
            models,
            f"crps_empirical_m{sample_count}",
        )
        moment_matched = _paired_arrays(
            frame,
            models,
            f"crps_mm_gaussian_m{sample_count}",
        )
        for model in models:
            shape = moment_matched[model] - empirical[model]
            total = s_gaussian - empirical[model]
            moment = s_gaussian - moment_matched[model]
            if not np.allclose(total, moment + shape, rtol=0.0, atol=1.0e-10):
                raise RuntimeError("Score decomposition identity failed.")
            shape_by_key[(model, sample_count)] = shape
            components_by_key[(model, sample_count, "total")] = total
            components_by_key[(model, sample_count, "moment")] = moment
            components_by_key[(model, sample_count, "shape")] = shape

    adjusted_by_key = {
        (model, sample_count): (
            shape_by_key[(model, sample_count)]
            - shape_by_key[(gaussian_model, sample_count)]
        )
        for sample_count in sample_counts
        for model in models
    }

    component_bootstrap = {
        key: np.empty(replicates, dtype=float) for key in components_by_key
    }
    adjusted_bootstrap = {
        key: np.empty(replicates, dtype=float) for key in adjusted_by_key
    }

    rng = np.random.default_rng(seed)
    for chunk_start in range(0, replicates, bootstrap_chunk_size):
        chunk_stop = min(chunk_start + bootstrap_chunk_size, replicates)
        draw = rng.integers(
            0,
            len(tasks),
            size=(chunk_stop - chunk_start, len(tasks)),
            endpoint=False,
        )
        for key, values in components_by_key.items():
            component_bootstrap[key][chunk_start:chunk_stop] = values[draw].mean(
                axis=1
            )
        for key, values in adjusted_by_key.items():
            adjusted_bootstrap[key][chunk_start:chunk_stop] = values[draw].mean(
                axis=1
            )

    contribution_rows = []
    for (model, sample_count, component_name), values in components_by_key.items():
        bootstrap = component_bootstrap[(model, sample_count, component_name)]
        low = float(np.quantile(bootstrap, 0.025))
        high = float(np.quantile(bootstrap, 0.975))
        contribution_rows.append(
            {
                "model_name": model,
                "sample_count": sample_count,
                "component": component_name,
                "estimate": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
            }
        )

    null_adjusted_rows = []
    for (model, sample_count), values in adjusted_by_key.items():
        bootstrap = adjusted_bootstrap[(model, sample_count)]
        low = float(np.quantile(bootstrap, 0.025))
        high = float(np.quantile(bootstrap, 0.975))
        null_adjusted_rows.append(
            {
                "model_name": model,
                "reference_model": gaussian_model,
                "sample_count": sample_count,
                "metric": "shape_contribution_minus_gaussian_null",
                "estimate_delta": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
            }
        )

    return pd.DataFrame(contribution_rows), pd.DataFrame(null_adjusted_rows)


def _rank_histograms(
    frame: pd.DataFrame,
    *,
    models: List[str],
    rank_sample_count: int,
    num_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_columns = [f"rank_{index:03d}" for index in range(rank_sample_count + 1)]
    missing = [column for column in rank_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Rank-count columns are incomplete; first missing columns={missing[:5]}."
        )

    histogram_rows = []
    calibration_rows = []

    rank_indices = np.arange(rank_sample_count + 1)
    bin_indices = np.minimum(
        (rank_indices * num_bins) // (rank_sample_count + 1),
        num_bins - 1,
    )

    for model in models:
        counts = (
            frame.loc[frame["model_name"] == model, rank_columns]
            .sum(axis=0)
            .to_numpy(dtype=float)
        )
        total = float(counts.sum())
        if total <= 0.0:
            raise ValueError(f"No rank counts for model {model!r}.")

        binned = np.zeros(num_bins, dtype=float)
        for rank_index, count in enumerate(counts):
            binned[bin_indices[rank_index]] += count
        probabilities = binned / binned.sum()
        expected = 1.0 / num_bins

        for bin_index, probability in enumerate(probabilities):
            histogram_rows.append(
                {
                    "model_name": model,
                    "bin_index": bin_index,
                    "bin_left": bin_index / num_bins,
                    "bin_right": (bin_index + 1) / num_bins,
                    "probability": float(probability),
                    "uniform_probability": expected,
                    "rank_sample_count": rank_sample_count,
                    "num_plot_bins": num_bins,
                }
            )

        l1_distance = 0.5 * float(np.abs(probabilities - expected).sum())
        chi_square = float(
            np.sum((binned - total / num_bins) ** 2 / (total / num_bins))
        )
        calibration_rows.append(
            {
                "model_name": model,
                "rank_sample_count": rank_sample_count,
                "num_plot_bins": num_bins,
                "num_rank_observations": int(total),
                "rank_histogram_l1_from_uniform": l1_distance,
                "rank_histogram_chi_square": chi_square,
            }
        )

    return pd.DataFrame(histogram_rows), pd.DataFrame(calibration_rows)


def _latex_table(
    contributions: pd.DataFrame,
    adjusted: pd.DataFrame,
    *,
    order: List[str],
    display_map: Dict[str, str],
    primary_count: int,
) -> str:
    subset = contributions.loc[
        contributions["sample_count"] == primary_count
    ].pivot(index="model_name", columns="component", values="estimate")
    adjusted_values = adjusted.loc[
        adjusted["sample_count"] == primary_count
    ].set_index("model_name")["estimate_delta"]

    lines = [
        r"\begin{tabular}{@{}l c c c c@{}}",
        r"\toprule",
        (
            r"Model & $\Delta_{\mathrm{total}}$ & "
            r"$\Delta_{\mathrm{moment}}$ & $\Delta_{\mathrm{shape}}$ & "
            r"Shape above Gaussian null \\"
        ),
        r"\midrule",
    ]
    for model in order:
        lines.append(
            f"{display_map[model]} & "
            f"{subset.loc[model, 'total']:.3f} & "
            f"{subset.loc[model, 'moment']:.3f} & "
            f"{subset.loc[model, 'shape']:.3f} & "
            f"{adjusted_values.loc[model]:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    models = [str(value) for value in cfg["shape_order"]]
    if set(frame["model_name"]) != set(models):
        raise ValueError("Shape-analysis model set does not match the config.")
    fingerprints = frame.groupby("task_index")["task_fingerprint"].nunique()
    if not fingerprints.eq(1).all():
        raise ValueError("Shape-analysis tasks are not paired across models.")
    if frame.duplicated(["model_name", "task_index"]).any():
        raise ValueError("Duplicate shape-analysis rows.")

    shape_cfg = dict(cfg["shape_analysis"])
    sample_counts = [int(value) for value in shape_cfg["sample_counts"]]
    rank_sample_count = int(shape_cfg["rank_sample_count"])
    num_plot_bins = int(shape_cfg.get("rank_plot_bins", 21))
    gaussian_model = str(cfg["roles"]["gaussian"])

    contributions, adjusted = _bootstrap_components(
        frame,
        models=models,
        gaussian_model=gaussian_model,
        sample_counts=sample_counts,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    histograms, calibration = _rank_histograms(
        frame,
        models=models,
        rank_sample_count=rank_sample_count,
        num_bins=num_plot_bins,
    )

    display_map = {
        str(entry["name"]): str(entry.get("display_name", entry["name"]))
        for entry in cfg["sources"]
    }
    for result in (contributions, adjusted, histograms, calibration):
        result["display_name"] = result["model_name"].map(display_map)

    contributions.to_csv(
        output_dir / "tabular_shape_contribution_summary.csv", index=False
    )
    adjusted.to_csv(
        output_dir / "tabular_shape_null_adjusted.csv", index=False
    )
    histograms.to_csv(
        output_dir / "tabular_rank_histogram_binned.csv", index=False
    )
    calibration.to_csv(
        output_dir / "tabular_rank_calibration_summary.csv", index=False
    )
    (output_dir / "tabular_shape_decomposition_table.tex").write_text(
        _latex_table(
            contributions,
            adjusted,
            order=models,
            display_map=display_map,
            primary_count=max(sample_counts),
        )
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_unit": "task",
                "bootstrap_chunk_size": args.bootstrap_chunk_size,
                "primary_shape_sample_count": max(sample_counts),
                "finite_sample_check": min(sample_counts),
                "rank_sample_count": rank_sample_count,
                "rank_plot_bins": num_plot_bins,
                "interpretation_guard": (
                    "Shape contribution is measured relative to the sampled "
                    "Gaussian null; no exact tabular posterior oracle is claimed."
                ),
            },
            indent=2,
        )
    )

    print("TABULAR SHAPE DECOMPOSITION")
    print(contributions.to_string(index=False))
    print()
    print(adjusted.to_string(index=False))
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
