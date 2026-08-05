from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from evaluation.tabular_analysis_utils import aggregate_task_metrics


ARCHITECTURE_ORDER = [
    "Gaussian TNP",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]

TRAINING_REGIME_ORDER = [
    "fixed32",
    "fixed64",
    "fixed128",
    "variable",
]

CRPS_VARIANT_ORDER = [
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260922)
    parser.add_argument(
        "--expected_tasks",
        type=int,
        default=None,
        help=(
            "Optional smoke-test override for the paired task count. "
            "The final analysis defaults to nested_tasks.accepted_tasks from YAML."
        ),
    )
    parser.add_argument(
        "--bootstrap_chunk_size",
        type=int,
        default=128,
        help="Number of bootstrap replicates processed per host-memory chunk.",
    )
    return parser.parse_args()


def _load_groups(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*/per_task_metrics.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No group per_task_metrics.csv files found under {input_dir}."
        )
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    duplicates = result.duplicated(["model_name", "num_context", "task_index"])
    if duplicates.any():
        duplicate_rows = result.loc[
            duplicates,
            ["model_name", "num_context", "task_index", "source_group"],
        ]
        raise ValueError(
            "Duplicate ladder rows after merging groups:\n"
            + duplicate_rows.head().to_string(index=False)
        )
    return result


def _validate_full_grid(
    frame: pd.DataFrame,
    *,
    context_sizes: List[int],
    expected_models: List[str],
    expected_tasks: int,
) -> None:
    expected_model_set = set(expected_models)

    for num_context in context_sizes:
        rung = frame.loc[frame["num_context"] == num_context]
        expected_rows = expected_tasks * len(expected_models)

        if len(rung) != expected_rows:
            raise ValueError(
                f"Nc={num_context}: expected {expected_rows} rows from the "
                f"complete 15-source grid, found {len(rung)}."
            )

        if set(rung["model_name"].astype(str)) != expected_model_set:
            missing = expected_model_set.difference(
                set(rung["model_name"].astype(str))
            )
            extra = set(rung["model_name"].astype(str)).difference(
                expected_model_set
            )
            raise ValueError(
                f"Nc={num_context}: incomplete source grid; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

        counts = rung.groupby("task_index")["model_name"].nunique()
        if not counts.eq(len(expected_models)).all():
            raise ValueError(
                f"Nc={num_context}: one or more tasks lack a configured source."
            )

        targets = rung.groupby("task_index")["target_fingerprint"].nunique()
        if not targets.eq(1).all():
            raise ValueError(
                f"Nc={num_context}: target fingerprints differ across sources."
            )

    target_counts = frame.groupby("task_index")["target_fingerprint"].nunique()
    if not target_counts.eq(1).all():
        raise ValueError("Raw target fingerprints differ across context-size rungs.")


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (num_context, model_name), group in frame.groupby(
        ["num_context", "model_name"], sort=False
    ):
        row = aggregate_task_metrics(group)
        row["num_context"] = int(num_context)
        row["model_name"] = str(model_name)
        row["display_name"] = str(group["display_name"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    *,
    values_by_key: Dict[tuple, np.ndarray],
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> Dict[tuple, tuple[float, float]]:
    if replicates < 1 or bootstrap_chunk_size < 1:
        raise ValueError("Bootstrap replicate and chunk counts must be positive.")
    if not values_by_key:
        return {}

    task_counts = {len(values) for values in values_by_key.values()}
    if len(task_counts) != 1:
        raise ValueError(
            f"Paired bootstrap inputs do not share one task count: {task_counts}."
        )
    num_tasks = task_counts.pop()
    if num_tasks < 2:
        raise ValueError("At least two paired tasks are required.")

    draws_by_key = {
        key: np.empty(replicates, dtype=float)
        for key in values_by_key
    }
    rng = np.random.default_rng(seed)

    for chunk_start in range(0, replicates, bootstrap_chunk_size):
        chunk_stop = min(chunk_start + bootstrap_chunk_size, replicates)
        draw = rng.integers(
            0,
            num_tasks,
            size=(chunk_stop - chunk_start, num_tasks),
            endpoint=False,
        )
        for key, values in values_by_key.items():
            draws_by_key[key][chunk_start:chunk_stop] = values[draw].mean(axis=1)

    return {
        key: (
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        )
        for key, bootstrap in draws_by_key.items()
    }


def _bootstrap_margin_summary(
    frame: pd.DataFrame,
    *,
    context_sizes: List[int],
    baseline_name: str,
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> pd.DataFrame:
    task_ids = sorted(frame["task_index"].unique())
    task_to_position = {task: index for index, task in enumerate(task_ids)}
    paired_margins = []

    for num_context in context_sizes:
        rung = frame.loc[frame["num_context"] == num_context]
        baseline = (
            rung.loc[rung["model_name"] == baseline_name, ["task_index", "crps"]]
            .rename(columns={"crps": "baseline_crps"})
            .set_index("task_index")
        )
        if len(baseline) != len(task_ids):
            raise ValueError(
                f"Baseline {baseline_name!r} is incomplete at Nc={num_context}."
            )

        for model_name, model_group in rung.groupby("model_name", sort=False):
            if model_name == baseline_name:
                continue
            model = model_group[["task_index", "crps", "display_name"]].set_index(
                "task_index"
            )
            paired = baseline.join(model, how="inner")
            if len(paired) != len(task_ids):
                raise ValueError(
                    f"Model {model_name!r} is not paired at Nc={num_context}."
                )
            paired["_position"] = [task_to_position[index] for index in paired.index]
            paired = paired.sort_values("_position")
            paired_margins.append(
                {
                    "num_context": int(num_context),
                    "model_name": str(model_name),
                    "display_name": str(model_group["display_name"].iloc[0]),
                    "values": (
                        paired["baseline_crps"].to_numpy(dtype=float)
                        - paired["crps"].to_numpy(dtype=float)
                    ),
                }
            )

    values_by_key = {
        (item["num_context"], item["model_name"]): item["values"]
        for item in paired_margins
    }
    intervals = _bootstrap_mean_ci(
        values_by_key=values_by_key,
        replicates=replicates,
        seed=seed,
        bootstrap_chunk_size=bootstrap_chunk_size,
    )

    rows = []
    for item in paired_margins:
        key = (item["num_context"], item["model_name"])
        low, high = intervals[key]
        rows.append(
            {
                "num_context": item["num_context"],
                "model_name": item["model_name"],
                "display_name": item["display_name"],
                "crps_margin_over_context_resample": float(item["values"].mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
            }
        )

    return pd.DataFrame(rows)


def _paired_regime_deltas(
    frame: pd.DataFrame,
    *,
    config: Dict,
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> pd.DataFrame:
    """Variable-context CRPS minus fixed-128 CRPS at every evaluation rung."""
    comparisons = []
    architecture_roles = dict(config["architecture_roles"])

    for architecture, names in architecture_roles.items():
        fixed128 = str(names["fixed128"])
        variable = str(names["variable"])
        for num_context in config["nested_tasks"]["context_sizes"]:
            rung = frame.loc[frame["num_context"] == int(num_context)]
            pivot = rung.loc[
                rung["model_name"].isin([fixed128, variable]),
                ["task_index", "model_name", "crps"],
            ].pivot(index="task_index", columns="model_name", values="crps")
            if pivot.isna().any().any():
                raise ValueError(
                    f"Incomplete fixed-vs-variable pairing for {architecture} "
                    f"at Nc={num_context}."
                )
            comparisons.append(
                {
                    "architecture": str(architecture),
                    "num_context": int(num_context),
                    "model_name": variable,
                    "reference_model": fixed128,
                    "values": (
                        pivot[variable].to_numpy(dtype=float)
                        - pivot[fixed128].to_numpy(dtype=float)
                    ),
                }
            )

    values_by_key = {
        (item["architecture"], item["num_context"]): item["values"]
        for item in comparisons
    }
    intervals = _bootstrap_mean_ci(
        values_by_key=values_by_key,
        replicates=replicates,
        seed=seed,
        bootstrap_chunk_size=bootstrap_chunk_size,
    )

    rows = []
    for item in comparisons:
        key = (item["architecture"], item["num_context"])
        low, high = intervals[key]
        rows.append(
            {
                "architecture": item["architecture"],
                "num_context": item["num_context"],
                "model_name": item["model_name"],
                "reference_model": item["reference_model"],
                "metric": "crps_variable_minus_fixed128",
                "estimate_delta": float(item["values"].mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
            }
        )

    return pd.DataFrame(rows)


def _paired_matched_regime_deltas(
    frame: pd.DataFrame,
    *,
    config: Dict,
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> pd.DataFrame:
    """Fixed specialist CRPS minus variable-context CRPS at matched rungs.

    Positive values mean the variable-context model has lower CRPS.
    """
    training_roles = dict(config["training_regime_roles"])
    variable_models = dict(training_roles["variable"]["models"])
    comparisons = []

    for num_context in (32, 64, 128):
        fixed_key = f"fixed{num_context}"
        fixed_models = dict(training_roles[fixed_key]["models"])
        rung = frame.loc[frame["num_context"] == num_context]

        for architecture in ARCHITECTURE_ORDER:
            fixed_name = str(fixed_models[architecture])
            variable_name = str(variable_models[architecture])
            pivot = rung.loc[
                rung["model_name"].isin([fixed_name, variable_name]),
                ["task_index", "model_name", "crps"],
            ].pivot(index="task_index", columns="model_name", values="crps")
            if pivot.isna().any().any():
                raise ValueError(
                    f"Incomplete matched fixed-vs-variable pairing for "
                    f"{architecture} at Nc={num_context}."
                )
            comparisons.append(
                {
                    "architecture": architecture,
                    "num_context": num_context,
                    "fixed_model": fixed_name,
                    "variable_model": variable_name,
                    "values": (
                        pivot[fixed_name].to_numpy(dtype=float)
                        - pivot[variable_name].to_numpy(dtype=float)
                    ),
                }
            )

    values_by_key = {
        (item["architecture"], item["num_context"]): item["values"]
        for item in comparisons
    }
    intervals = _bootstrap_mean_ci(
        values_by_key=values_by_key,
        replicates=replicates,
        seed=seed,
        bootstrap_chunk_size=bootstrap_chunk_size,
    )

    rows = []
    for item in comparisons:
        key = (item["architecture"], item["num_context"])
        low, high = intervals[key]
        rows.append(
            {
                "architecture": item["architecture"],
                "num_context": item["num_context"],
                "variable_model": item["variable_model"],
                "reference_model": item["fixed_model"],
                "metric": "crps_fixed_minus_variable",
                "estimate_delta": float(item["values"].mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
                "interpretation": (
                    "positive means variable-context training has lower CRPS"
                ),
            }
        )

    return pd.DataFrame(rows)


def _paired_variant_vs_gaussian(
    frame: pd.DataFrame,
    *,
    config: Dict,
    replicates: int,
    seed: int,
    bootstrap_chunk_size: int,
) -> pd.DataFrame:
    """Paired CRPS gain of each CRPS model over Gaussian.

    The reported quantity is Gaussian CRPS minus variant CRPS, so positive
    values mean the CRPS-trained variant is better.
    """
    training_roles = dict(config["training_regime_roles"])
    comparisons = []

    for regime_key in TRAINING_REGIME_ORDER:
        regime = dict(training_roles[regime_key])
        models = dict(regime["models"])
        gaussian_name = str(models["Gaussian TNP"])

        for architecture in CRPS_VARIANT_ORDER:
            variant_name = str(models[architecture])
            for num_context in config["nested_tasks"]["context_sizes"]:
                rung = frame.loc[frame["num_context"] == int(num_context)]
                pivot = rung.loc[
                    rung["model_name"].isin([gaussian_name, variant_name]),
                    ["task_index", "model_name", "crps"],
                ].pivot(index="task_index", columns="model_name", values="crps")
                if pivot.isna().any().any():
                    raise ValueError(
                        f"Incomplete {architecture}-vs-Gaussian pairing for "
                        f"regime={regime_key}, Nc={num_context}."
                    )
                comparisons.append(
                    {
                        "training_regime": regime_key,
                        "training_display_name": str(regime["display_name"]),
                        "training_context_size": regime.get(
                            "training_context_size"
                        ),
                        "architecture": architecture,
                        "num_context": int(num_context),
                        "variant_model": variant_name,
                        "gaussian_model": gaussian_name,
                        "values": (
                            pivot[gaussian_name].to_numpy(dtype=float)
                            - pivot[variant_name].to_numpy(dtype=float)
                        ),
                    }
                )

    values_by_key = {
        (
            item["training_regime"],
            item["architecture"],
            item["num_context"],
        ): item["values"]
        for item in comparisons
    }
    intervals = _bootstrap_mean_ci(
        values_by_key=values_by_key,
        replicates=replicates,
        seed=seed,
        bootstrap_chunk_size=bootstrap_chunk_size,
    )

    rows = []
    for item in comparisons:
        key = (
            item["training_regime"],
            item["architecture"],
            item["num_context"],
        )
        low, high = intervals[key]
        rows.append(
            {
                "training_regime": item["training_regime"],
                "training_display_name": item["training_display_name"],
                "training_context_size": item["training_context_size"],
                "architecture": item["architecture"],
                "num_context": item["num_context"],
                "variant_model": item["variant_model"],
                "reference_model": item["gaussian_model"],
                "metric": "crps_gaussian_minus_variant",
                "estimate_delta": float(item["values"].mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_contains_zero": bool(low <= 0.0 <= high),
                "variant_better": bool(float(item["values"].mean()) > 0.0),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "task",
                "interpretation": (
                    "positive means the CRPS-trained variant has lower CRPS"
                ),
            }
        )

    return pd.DataFrame(rows)


def _figure_data(margins: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    training_roles = dict(cfg["training_regime_roles"])
    rows = []

    for regime_key in TRAINING_REGIME_ORDER:
        regime = dict(training_roles[regime_key])
        models = dict(regime["models"])
        training_context_size = regime.get("training_context_size")

        for architecture in ARCHITECTURE_ORDER:
            model_name = str(models[architecture])
            selected = margins.loc[margins["model_name"] == model_name].copy()
            selected = selected.sort_values("num_context")
            expected_contexts = selected["num_context"].astype(int).tolist()
            if expected_contexts != [16, 32, 64, 128]:
                raise ValueError(
                    f"Incomplete full-grid line for regime={regime_key}, "
                    f"architecture={architecture}: {expected_contexts}."
                )
            selected["architecture"] = architecture
            selected["training_regime"] = regime_key
            selected["training_display_name"] = str(regime["display_name"])
            selected["training_context_size"] = training_context_size
            selected["in_training_support"] = (
                True
                if training_context_size is None
                else selected["num_context"].eq(int(training_context_size))
            )
            selected["annotation_point"] = (
                selected["num_context"].eq(128)
                if training_context_size is None
                else selected["num_context"].eq(int(training_context_size))
            )
            rows.append(selected)

    result = pd.concat(rows, ignore_index=True)
    expected_rows = len(TRAINING_REGIME_ORDER) * len(ARCHITECTURE_ORDER) * 4
    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} context-figure rows, found {len(result)}."
        )
    return result


def _latex_table(summary: pd.DataFrame, order: List[str], context_sizes: List[int]) -> str:
    values = summary.set_index(["model_name", "num_context"])
    lines = [
        r"\begin{tabular}{@{}l c c c c@{}}",
        r"\toprule",
        r"Model & $N_c=16$ & $N_c=32$ & $N_c=64$ & $N_c=128$ \\",
        r"\midrule",
    ]
    for model in order:
        if model not in summary["model_name"].values:
            continue
        display = summary.loc[summary["model_name"] == model, "display_name"].iloc[0]
        cells = []
        for num_context in context_sizes:
            if (model, num_context) in values.index:
                cells.append(f"{values.loc[(model, num_context), 'crps']:.3f}")
            else:
                cells.append("--")
        lines.append(f"{display} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_groups(Path(args.input_dir))
    context_sizes = [int(value) for value in cfg["nested_tasks"]["context_sizes"]]
    expected_tasks = (
        int(args.expected_tasks)
        if args.expected_tasks is not None
        else int(cfg["nested_tasks"]["accepted_tasks"])
    )
    expected_models = [str(source["name"]) for source in cfg["sources"]]

    if frame["task_index"].nunique() != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} paired tasks, found "
            f"{frame['task_index'].nunique()}."
        )

    _validate_full_grid(
        frame,
        context_sizes=context_sizes,
        expected_models=expected_models,
        expected_tasks=expected_tasks,
    )

    summary = _summary(frame)
    margins = _bootstrap_margin_summary(
        frame,
        context_sizes=context_sizes,
        baseline_name=str(cfg["roles"]["context_resample"]),
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    regime_deltas = _paired_regime_deltas(
        frame,
        config=cfg,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 1,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    matched_regime_deltas = _paired_matched_regime_deltas(
        frame,
        config=cfg,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 2,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    variant_vs_gaussian = _paired_variant_vs_gaussian(
        frame,
        config=cfg,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 3,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    figure_data = _figure_data(margins, cfg)

    summary.to_csv(output_dir / "tabular_ladder_absolute_summary.csv", index=False)
    margins.to_csv(output_dir / "tabular_ladder_crps_margins.csv", index=False)
    regime_deltas.to_csv(
        output_dir / "tabular_ladder_variable_minus_fixed128.csv", index=False
    )
    matched_regime_deltas.to_csv(
        output_dir / "tabular_ladder_variable_minus_matched_fixed.csv", index=False
    )
    variant_vs_gaussian.to_csv(
        output_dir / "tabular_crps_variant_vs_gaussian.csv", index=False
    )
    figure_data.to_csv(
        output_dir / "tabular_context_training_regime_figure_data.csv", index=False
    )
    (output_dir / "tabular_context_ladder_table.tex").write_text(
        _latex_table(summary, [str(x) for x in cfg["table_order"]], context_sizes)
    )

    cache_path = Path(cfg["nested_tasks"]["cache_path"])
    rejection_path = cache_path.with_suffix(".json")
    if rejection_path.is_file():
        (output_dir / "nested_task_rejection_diagnostics.json").write_text(
            rejection_path.read_text()
        )

    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_unit": "task",
                "bootstrap_chunk_size": args.bootstrap_chunk_size,
                "fixed_raw_targets_across_rungs": True,
                "intersection_task_count": expected_tasks,
                "complete_training_regime_grid": True,
                "training_regimes": TRAINING_REGIME_ORDER,
                "architectures": ARCHITECTURE_ORDER,
                "context_sizes": context_sizes,
            },
            indent=2,
        )
    )

    print("TABULAR NESTED CONTEXT LADDER: FULL TRAINING-REGIME GRID")
    print(figure_data.to_string(index=False))
    print()
    print("CRPS-TRAINED VARIANTS VERSUS MATCHED GAUSSIAN")
    print(variant_vs_gaussian.to_string(index=False))
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
