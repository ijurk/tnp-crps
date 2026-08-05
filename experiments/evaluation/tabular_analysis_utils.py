from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "rmse",
    "crps",
    "energy_score",
    "spread_skill_ratio",
    "coverage_90",
    "width_90",
)


def validate_paired_rows(
    frame: pd.DataFrame,
    *,
    task_columns: Sequence[str] = ("task_index",),
    model_column: str = "model_name",
    expected_models: Iterable[str] | None = None,
) -> None:
    key_columns = [*task_columns, model_column]
    if frame.duplicated(key_columns).any():
        raise ValueError(f"Duplicate rows for keys {key_columns}.")

    if expected_models is not None:
        expected = set(expected_models)
        actual = set(frame[model_column].astype(str))
        if actual != expected:
            raise ValueError(
                f"Model set mismatch. expected={sorted(expected)}, "
                f"actual={sorted(actual)}."
            )

    counts = frame.groupby(list(task_columns))[model_column].nunique()
    if counts.nunique() != 1:
        raise ValueError("Not every paired task has the same number of models.")


def aggregate_task_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        raise ValueError("Cannot aggregate an empty frame.")

    weights = frame["num_target_elements"].to_numpy(dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("Non-positive total target count.")

    rmse_values = frame["rmse"].to_numpy(dtype=float)
    spread_values = frame["ensemble_spread"].to_numpy(dtype=float)
    num_samples = frame["num_eval_samples"].to_numpy(dtype=int)
    if np.unique(num_samples).size != 1:
        raise ValueError("A summary row mixes finite ensemble sizes.")
    m = int(num_samples[0])

    pooled_rmse = math.sqrt(
        float(np.sum(weights * rmse_values**2) / total_weight)
    )
    pooled_spread = math.sqrt(
        float(np.sum(weights * spread_values**2) / total_weight)
    )
    finite_m_correction = math.sqrt((m + 1.0) / m)

    return {
        "num_tasks": int(frame["task_index"].nunique()),
        "num_eval_samples": m,
        "rmse": pooled_rmse,
        "crps": float(np.sum(weights * frame["crps"]) / total_weight),
        "energy_score": float(frame["energy_score"].mean()),
        "ensemble_spread": pooled_spread,
        "spread_skill_ratio": (
            finite_m_correction * pooled_spread / (pooled_rmse + 1.0e-12)
        ),
        "coverage_90": float(
            np.sum(weights * frame["coverage_90"]) / total_weight
        ),
        "width_90": float(np.sum(weights * frame["width_90"]) / total_weight),
    }


def summarise_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in frame.groupby("model_name", sort=False):
        summary = aggregate_task_metrics(group)
        summary["model_name"] = model_name
        if "display_name" in group:
            summary["display_name"] = str(group["display_name"].iloc[0])
        rows.append(summary)
    return pd.DataFrame(rows)


def _task_draws(
    num_tasks: int,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0,
        int(num_tasks),
        size=(int(replicates), int(num_tasks)),
        endpoint=False,
    )


def paired_bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    reference_model: str | None = None,
    metrics: Sequence[str] = METRIC_COLUMNS,
    bootstrap_chunk_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Task-level paired bootstrap without a replicates-by-tasks draw matrix.

    The same resampled task indices are reused across every model in a chunk,
    preserving paired comparisons while bounding host memory.
    """
    models = list(dict.fromkeys(frame["model_name"].astype(str).tolist()))
    tasks = sorted(frame["task_index"].unique().tolist())
    num_tasks = len(tasks)
    if num_tasks < 2:
        raise ValueError("At least two paired tasks are required.")
    if int(replicates) < 1:
        raise ValueError("replicates must be positive.")
    if int(bootstrap_chunk_size) < 1:
        raise ValueError("bootstrap_chunk_size must be positive.")

    task_to_position = {task: index for index, task in enumerate(tasks)}
    model_frames: Dict[str, pd.DataFrame] = {}
    point: Dict[str, Dict[str, float]] = {}
    arrays: Dict[str, Dict[str, np.ndarray | int]] = {}

    required_columns = {
        "num_target_elements",
        "num_eval_samples",
        "rmse",
        "crps",
        "energy_score",
        "ensemble_spread",
        "coverage_90",
        "width_90",
    }

    for model in models:
        subset = frame.loc[frame["model_name"] == model].copy()
        subset["_position"] = subset["task_index"].map(task_to_position)
        subset = subset.sort_values("_position").reset_index(drop=True)
        if len(subset) != num_tasks or subset["_position"].isna().any():
            raise ValueError(f"Model {model!r} is missing paired tasks.")
        missing = required_columns.difference(subset.columns)
        if missing:
            raise ValueError(
                f"Model {model!r} is missing bootstrap columns {sorted(missing)}."
            )
        sample_counts = subset["num_eval_samples"].to_numpy(dtype=int)
        if np.unique(sample_counts).size != 1:
            raise ValueError(f"Model {model!r} mixes finite ensemble sizes.")

        model_frames[model] = subset
        point[model] = aggregate_task_metrics(subset)
        arrays[model] = {
            "weights": subset["num_target_elements"].to_numpy(dtype=float),
            "rmse_sq": subset["rmse"].to_numpy(dtype=float) ** 2,
            "crps": subset["crps"].to_numpy(dtype=float),
            "energy_score": subset["energy_score"].to_numpy(dtype=float),
            "spread_sq": subset["ensemble_spread"].to_numpy(dtype=float) ** 2,
            "coverage_90": subset["coverage_90"].to_numpy(dtype=float),
            "width_90": subset["width_90"].to_numpy(dtype=float),
            "num_eval_samples": int(sample_counts[0]),
        }

    bootstrap: Dict[str, Dict[str, np.ndarray]] = {
        model: {metric: np.empty(int(replicates), dtype=float) for metric in metrics}
        for model in models
    }

    rng = np.random.default_rng(int(seed))
    for chunk_start in range(0, int(replicates), int(bootstrap_chunk_size)):
        chunk_stop = min(
            chunk_start + int(bootstrap_chunk_size),
            int(replicates),
        )
        chunk_replicates = chunk_stop - chunk_start
        draw = rng.integers(
            0,
            num_tasks,
            size=(chunk_replicates, num_tasks),
            endpoint=False,
        )

        for model in models:
            values = arrays[model]
            weights = values["weights"][draw]  # type: ignore[index]
            total_weight = weights.sum(axis=1)
            if np.any(total_weight <= 0.0):
                raise ValueError("Bootstrap produced a non-positive target weight.")

            rmse = np.sqrt(
                np.sum(weights * values["rmse_sq"][draw], axis=1)  # type: ignore[index]
                / total_weight
            )
            spread = np.sqrt(
                np.sum(weights * values["spread_sq"][draw], axis=1)  # type: ignore[index]
                / total_weight
            )
            m = int(values["num_eval_samples"])
            correction = math.sqrt((m + 1.0) / m)

            chunk_values = {
                "rmse": rmse,
                "crps": (
                    np.sum(weights * values["crps"][draw], axis=1)  # type: ignore[index]
                    / total_weight
                ),
                "energy_score": np.nanmean(
                    values["energy_score"][draw],  # type: ignore[index]
                    axis=1,
                ),
                "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
                "coverage_90": (
                    np.sum(weights * values["coverage_90"][draw], axis=1)  # type: ignore[index]
                    / total_weight
                ),
                "width_90": (
                    np.sum(weights * values["width_90"][draw], axis=1)  # type: ignore[index]
                    / total_weight
                ),
            }
            for metric in metrics:
                bootstrap[model][metric][chunk_start:chunk_stop] = chunk_values[
                    metric
                ]

    ci_rows: List[Dict[str, object]] = []
    for model in models:
        for metric in metrics:
            values = bootstrap[model][metric]
            ci_rows.append(
                {
                    "model_name": model,
                    "metric": metric,
                    "estimate": point[model][metric],
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_unit": "task",
                }
            )

    delta_rows: List[Dict[str, object]] = []
    if reference_model is not None:
        if reference_model not in models:
            raise KeyError(reference_model)
        for model in models:
            if model == reference_model:
                continue
            for metric in metrics:
                delta = bootstrap[model][metric] - bootstrap[reference_model][metric]
                estimate = point[model][metric] - point[reference_model][metric]
                low = float(np.nanquantile(delta, 0.025))
                high = float(np.nanquantile(delta, 0.975))
                delta_rows.append(
                    {
                        "model_name": model,
                        "reference_model": reference_model,
                        "metric": metric,
                        "estimate_delta": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_contains_zero": bool(low <= 0.0 <= high),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_unit": "task",
                    }
                )

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def bootstrap_mean_difference(
    values: pd.DataFrame,
    *,
    value_column: str,
    model_column: str,
    task_column: str,
    reference_model: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    pivot = values.pivot(
        index=task_column,
        columns=model_column,
        values=value_column,
    ).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Paired mean-difference input contains missing cells.")
    models = list(pivot.columns)
    if reference_model not in models:
        raise KeyError(reference_model)

    array = pivot.to_numpy(dtype=float)
    draws = _task_draws(len(pivot), replicates=replicates, seed=seed)
    rows = []
    reference_index = models.index(reference_model)

    for model_index, model in enumerate(models):
        if model == reference_model:
            continue
        task_difference = array[:, model_index] - array[:, reference_index]
        boot = task_difference[draws].mean(axis=1)
        estimate = float(task_difference.mean())
        low = float(np.quantile(boot, 0.025))
        high = float(np.quantile(boot, 0.975))
        rows.append(
            {
                "model_name": model,
                "reference_model": reference_model,
                "metric": value_column,
                "estimate_delta": estimate,
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": int(replicates),
                "bootstrap_unit": "task",
            }
        )

    return pd.DataFrame(rows)
