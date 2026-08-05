from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from evaluation.tabular_analysis_utils import aggregate_task_metrics


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
    if len(task_ids) < 2:
        raise ValueError("At least two paired tasks are required.")
    if replicates < 1 or bootstrap_chunk_size < 1:
        raise ValueError("Bootstrap replicate and chunk counts must be positive.")

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

    bootstrap_values = {
        (item["num_context"], item["model_name"]): np.empty(
            replicates, dtype=float
        )
        for item in paired_margins
    }
    rng = np.random.default_rng(seed)

    for chunk_start in range(0, replicates, bootstrap_chunk_size):
        chunk_stop = min(chunk_start + bootstrap_chunk_size, replicates)
        draw = rng.integers(
            0,
            len(task_ids),
            size=(chunk_stop - chunk_start, len(task_ids)),
            endpoint=False,
        )
        for item in paired_margins:
            key = (item["num_context"], item["model_name"])
            bootstrap_values[key][chunk_start:chunk_stop] = item["values"][draw].mean(
                axis=1
            )

    rows = []
    for item in paired_margins:
        key = (item["num_context"], item["model_name"])
        bootstrap = bootstrap_values[key]
        low = float(np.quantile(bootstrap, 0.025))
        high = float(np.quantile(bootstrap, 0.975))
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
    if replicates < 1 or bootstrap_chunk_size < 1:
        raise ValueError("Bootstrap replicate and chunk counts must be positive.")

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

    bootstrap_values = {
        (item["architecture"], item["num_context"]): np.empty(
            replicates, dtype=float
        )
        for item in comparisons
    }
    rng = np.random.default_rng(seed)
    num_tasks = len(comparisons[0]["values"]) if comparisons else 0

    for chunk_start in range(0, replicates, bootstrap_chunk_size):
        chunk_stop = min(chunk_start + bootstrap_chunk_size, replicates)
        draw = rng.integers(
            0,
            num_tasks,
            size=(chunk_stop - chunk_start, num_tasks),
            endpoint=False,
        )
        for item in comparisons:
            if len(item["values"]) != num_tasks:
                raise ValueError("Regime comparisons do not share one task count.")
            key = (item["architecture"], item["num_context"])
            bootstrap_values[key][chunk_start:chunk_stop] = item["values"][draw].mean(
                axis=1
            )

    rows = []
    for item in comparisons:
        key = (item["architecture"], item["num_context"])
        bootstrap = bootstrap_values[key]
        low = float(np.quantile(bootstrap, 0.025))
        high = float(np.quantile(bootstrap, 0.975))
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


def _figure_data(margins: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    roles = dict(cfg["architecture_roles"])
    rows = []

    for architecture, names in roles.items():
        for regime, model_name in (
            ("fixed128_ood", str(names["fixed128"])),
            ("variable_trained", str(names["variable"])),
        ):
            selected = margins.loc[margins["model_name"] == model_name].copy()
            selected["architecture"] = architecture
            selected["panel"] = regime
            selected["marker_type"] = "line"
            rows.append(selected)

        specialisation = dict(names.get("specialisation", {}))
        for context_size, model_name in specialisation.items():
            selected = margins.loc[
                (margins["model_name"] == str(model_name))
                & (margins["num_context"] == int(context_size))
            ].copy()
            selected["architecture"] = architecture
            selected["panel"] = "variable_trained"
            selected["marker_type"] = "specialisation"
            rows.append(selected)

    return pd.concat(rows, ignore_index=True)


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

    if frame["task_index"].nunique() != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} paired tasks, found "
            f"{frame['task_index'].nunique()}."
        )
    target_counts = frame.groupby("task_index")["target_fingerprint"].nunique()
    if not target_counts.eq(1).all():
        raise ValueError("Raw target fingerprints differ across rungs.")

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
    figure_data = _figure_data(margins, cfg)

    summary.to_csv(output_dir / "tabular_ladder_absolute_summary.csv", index=False)
    margins.to_csv(output_dir / "tabular_ladder_crps_margins.csv", index=False)
    regime_deltas.to_csv(
        output_dir / "tabular_ladder_variable_minus_fixed128.csv", index=False
    )
    figure_data.to_csv(
        output_dir / "tabular_context_dependence_figure_data.csv", index=False
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
            },
            indent=2,
        )
    )

    print("TABULAR NESTED CONTEXT LADDER")
    print(figure_data.to_string(index=False))
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
