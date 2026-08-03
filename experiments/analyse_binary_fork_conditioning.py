from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SOURCE_ORDER = [
    "Exact marginal oracle",
    "Old Gaussian TNP",
    "Old Dropout CRPS-TNP",
    "Old StochLN CRPS-TNP",
    "Mixed Gaussian TNP",
    "Mixed Dropout CRPS-TNP",
    "Mixed StochLN CRPS-TNP",
]
CONDITION_ORDER = ["ambiguous", "upper_reveal", "lower_reveal"]
OLD_MIXED_PAIRS = [
    ("Old Gaussian TNP", "Mixed Gaussian TNP"),
    ("Old Dropout CRPS-TNP", "Mixed Dropout CRPS-TNP"),
    ("Old StochLN CRPS-TNP", "Mixed StochLN CRPS-TNP"),
]
COMPONENTS = [
    "numel",
    "sse",
    "crps_sum",
    "var_sum",
    "coverage_count_90",
    "shape_numel",
    "branch_mass_error_sum",
    "gap_mass_error_sum",
    "pred_upper_mass_sum",
    "oracle_upper_mass_sum",
]
METRICS = [
    "crps",
    "branch_mass_error",
    "gap_mass_error",
    "revealed_branch_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=20260902, type=int)
    parser.add_argument("--bootstrap_chunk_size", default=500, type=int)
    parser.add_argument("--region", default="postfork", type=str)
    return parser.parse_args()


def _aggregate(group: pd.DataFrame, condition: str) -> Dict[str, float]:
    numel = float(group["numel"].sum())
    shape_numel = float(group["shape_numel"].sum())
    pred_upper = float(group["pred_upper_mass_sum"].sum()) / shape_numel
    oracle_upper = float(group["oracle_upper_mass_sum"].sum()) / shape_numel
    if condition == "upper_reveal":
        revealed_probability = pred_upper
        oracle_revealed_probability = oracle_upper
    elif condition == "lower_reveal":
        revealed_probability = 1.0 - pred_upper
        oracle_revealed_probability = 1.0 - oracle_upper
    else:
        revealed_probability = float("nan")
        oracle_revealed_probability = float("nan")
    return {
        "num_tasks": int(group["task_index"].nunique()),
        "numel": numel,
        "shape_numel": shape_numel,
        "rmse": math.sqrt(float(group["sse"].sum()) / numel),
        "crps": float(group["crps_sum"].sum()) / numel,
        "coverage_90": float(group["coverage_count_90"].sum()) / numel,
        "branch_mass_error": float(group["branch_mass_error_sum"].sum()) / shape_numel,
        "gap_mass_error": float(group["gap_mass_error_sum"].sum()) / shape_numel,
        "pred_upper_mass": pred_upper,
        "oracle_upper_mass": oracle_upper,
        "revealed_branch_probability": revealed_probability,
        "oracle_revealed_branch_probability": oracle_revealed_probability,
    }


def check_pairing(df: pd.DataFrame) -> None:
    selected = df.loc[df["region"] == "postfork"]
    if set(selected["model_name"].unique()) != set(SOURCE_ORDER):
        raise RuntimeError("Unexpected source set in conditioning evaluation.")
    if set(selected["condition"].unique()) != set(CONDITION_ORDER):
        raise RuntimeError("Unexpected condition set in conditioning evaluation.")
    for condition, condition_df in selected.groupby("condition"):
        grouped = condition_df.groupby("task_index")
        if not grouped["model_name"].nunique().eq(len(SOURCE_ORDER)).all():
            raise RuntimeError(f"Source pairing failed for {condition}.")
        if not grouped["task_fingerprint"].nunique().eq(1).all():
            raise RuntimeError(f"Fingerprint pairing failed for {condition}.")
    base_group = selected.groupby("task_index")["base_task_fingerprint"].nunique()
    if not base_group.eq(1).all():
        raise RuntimeError("Base task fingerprints differ across conditions.")


def summary(df: pd.DataFrame, region: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    selected = df.loc[df["region"] == region]
    for (model_name, condition), group in selected.groupby(
        ["model_name", "condition"], sort=False
    ):
        row: Dict[str, float] = {
            "model_name": model_name,
            "condition": condition,
        }
        row.update(_aggregate(group, condition))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=SOURCE_ORDER, ordered=True
    )
    out["condition"] = pd.Categorical(
        out["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return out.sort_values(["model_name", "condition"]).reset_index(drop=True)


def _cluster_table(df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = df.loc[df["region"] == region]
    aggregation = {column: "sum" for column in COMPONENTS}
    return (
        selected.groupby(
            ["model_name", "condition", "generator_batch_index"],
            as_index=False,
        )
        .agg(aggregation)
        .sort_values(["model_name", "condition", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays(
    components: Dict[str, np.ndarray], condition: str
) -> Dict[str, np.ndarray]:
    numel = components["numel"]
    shape_numel = components["shape_numel"]
    pred_upper = components["pred_upper_mass_sum"] / shape_numel
    if condition == "upper_reveal":
        revealed = pred_upper
    elif condition == "lower_reveal":
        revealed = 1.0 - pred_upper
    else:
        revealed = np.full_like(pred_upper, np.nan)
    return {
        "crps": components["crps_sum"] / numel,
        "branch_mass_error": components["branch_mass_error_sum"] / shape_numel,
        "gap_mass_error": components["gap_mass_error_sum"] / shape_numel,
        "revealed_branch_probability": revealed,
    }


def paired_bootstrap(
    *,
    df: pd.DataFrame,
    point: pd.DataFrame,
    region: str,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df, region)
    first = clusters.loc[
        (clusters["model_name"] == SOURCE_ORDER[0])
        & (clusters["condition"] == CONDITION_ORDER[0])
    ].sort_values("generator_batch_index")
    cluster_ids = first["generator_batch_index"].to_numpy()
    num_clusters = int(cluster_ids.size)
    rng = np.random.default_rng(seed)

    distributions: Dict[Tuple[str, str, str], np.ndarray] = {}
    for source in SOURCE_ORDER:
        for condition in CONDITION_ORDER:
            for metric in METRICS:
                distributions[(source, condition, metric)] = np.empty(
                    replicates, dtype=np.float64
                )

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        draws = rng.integers(
            0, num_clusters, size=(stop - start, num_clusters), endpoint=False
        )
        for source in SOURCE_ORDER:
            for condition in CONDITION_ORDER:
                group = clusters.loc[
                    (clusters["model_name"] == source)
                    & (clusters["condition"] == condition)
                ].sort_values("generator_batch_index")
                if not np.array_equal(
                    group["generator_batch_index"].to_numpy(), cluster_ids
                ):
                    raise RuntimeError(
                        f"Cluster pairing failed for {source}, {condition}."
                    )
                components = {
                    column: group[column].to_numpy(dtype=np.float64)[draws].sum(axis=1)
                    for column in COMPONENTS
                }
                arrays = _metric_arrays(components, condition)
                for metric in METRICS:
                    distributions[(source, condition, metric)][start:stop] = arrays[metric]

    point_index = point.set_index(["model_name", "condition"])
    ci_rows = []
    comparison_rows = []
    for source in SOURCE_ORDER:
        for condition in CONDITION_ORDER:
            for metric in METRICS:
                if condition == "ambiguous" and metric == "revealed_branch_probability":
                    continue
                values = distributions[(source, condition, metric)]
                ci_rows.append(
                    {
                        "model_name": source,
                        "condition": condition,
                        "metric": metric,
                        "estimate": float(point_index.loc[(source, condition), metric]),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "generator_batch",
                    }
                )

    for old_source, mixed_source in OLD_MIXED_PAIRS:
        for condition in CONDITION_ORDER:
            for metric in METRICS:
                if condition == "ambiguous" and metric == "revealed_branch_probability":
                    continue
                delta = (
                    distributions[(mixed_source, condition, metric)]
                    - distributions[(old_source, condition, metric)]
                )
                comparison_rows.append(
                    {
                        "mixed_model": mixed_source,
                        "old_model": old_source,
                        "condition": condition,
                        "metric": metric,
                        "estimate_delta": float(
                            point_index.loc[(mixed_source, condition), metric]
                            - point_index.loc[(old_source, condition), metric]
                        ),
                        "ci_low": float(np.quantile(delta, 0.025)),
                        "ci_high": float(np.quantile(delta, 0.975)),
                        "ci_contains_zero": bool(
                            np.quantile(delta, 0.025) <= 0.0 <= np.quantile(delta, 0.975)
                        ),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "generator_batch",
                    }
                )
    return pd.DataFrame(ci_rows), pd.DataFrame(comparison_rows)


def conditioning_table(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCE_ORDER:
        source_rows = result.loc[result["model_name"] == source].set_index(
            "condition"
        )
        if set(source_rows.index.astype(str)) != set(CONDITION_ORDER):
            raise RuntimeError(f"Missing conditioning rows for {source}.")
        upper = source_rows.loc["upper_reveal"]
        lower = source_rows.loc["lower_reveal"]
        ambiguous = source_rows.loc["ambiguous"]
        rows.append(
            {
                "model_name": source,
                "ambiguous_crps": float(ambiguous["crps"]),
                "ambiguous_branch_mass_error": float(
                    ambiguous["branch_mass_error"]
                ),
                "upper_revealed_branch_probability": float(
                    upper["revealed_branch_probability"]
                ),
                "lower_revealed_branch_probability": float(
                    lower["revealed_branch_probability"]
                ),
                "mean_reveal_crps": 0.5
                * (float(upper["crps"]) + float(lower["crps"])),
            }
        )
    return pd.DataFrame(rows)


def latex_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \begin{tabular}{@{}l c c c c c@{}}",
        r"        \toprule",
        r"        Model & Ambiguous CRPS $\downarrow$ & Ambiguous mass error $\downarrow$ & Upper reveal & Lower reveal & Reveal CRPS $\downarrow$ \\",
        r"        \midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"        {row['model_name']} & "
            f"{row['ambiguous_crps']:.3f} & "
            f"{row['ambiguous_branch_mass_error']:.3f} & "
            f"{row['upper_revealed_branch_probability']:.3f} & "
            f"{row['lower_revealed_branch_probability']:.3f} & "
            f"{row['mean_reveal_crps']:.3f} \\\\"
        )
        if str(row["model_name"]) == "Exact marginal oracle":
            lines.append(r"        \midrule")
        if str(row["model_name"]) == "Old StochLN CRPS-TNP":
            lines.append(r"        \addlinespace")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption[Binary-fork revealing-context acid test.]",
            r"    {Old ambiguous-only and mixed-context checkpoints evaluated on the same paired tasks. Upper and lower reveal columns report predictive probability assigned to the revealed branch; one is ideal. Ambiguous mass error is the absolute upper-branch probability error against the exact posterior. Reveal CRPS is averaged over upper- and lower-reveal conditions.}",
            r"    \label{table:binary_fork_conditioning}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    check_pairing(df)
    result = summary(df, args.region)
    ci, comparisons = paired_bootstrap(
        df=df,
        point=result,
        region=args.region,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        chunk_size=int(args.bootstrap_chunk_size),
    )
    result.to_csv(output_dir / "summary_by_source_condition.csv", index=False)
    compact_table = conditioning_table(result)
    compact_table.to_csv(output_dir / "summary_conditioning_table.csv", index=False)
    ci.to_csv(output_dir / "bootstrap_metric_ci.csv", index=False)
    comparisons.to_csv(output_dir / "mixed_minus_old_deltas.csv", index=False)
    (output_dir / "binary_fork_conditioning_table.tex").write_text(
        latex_table(compact_table)
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "region": args.region,
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "bootstrap_seed": int(args.bootstrap_seed),
                "bootstrap_unit": "generator_batch",
            },
            indent=2,
        )
    )
    print("\nCONDITIONING SUMMARY\n")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMIXED MINUS OLD\n")
    print(comparisons.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
