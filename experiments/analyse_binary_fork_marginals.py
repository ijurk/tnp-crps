from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Exact marginal oracle",
    "Gaussian TNP",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]
REFERENCE = "Gaussian TNP"
METRICS = [
    "rmse",
    "crps",
    "spread_skill_ratio",
    "coverage_90",
    "gap_mass_error",
    "branch_mass_error",
]
COMPONENTS = [
    "numel",
    "sse",
    "crps_sum",
    "var_sum",
    "coverage_count_90",
    "shape_numel",
    "gap_mass_error_sum",
    "branch_mass_error_sum",
    "pred_gap_mass_sum",
    "oracle_gap_mass_sum",
    "pred_upper_mass_sum",
    "oracle_upper_mass_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=20260901, type=int)
    parser.add_argument("--bootstrap_chunk_size", default=250, type=int)
    parser.add_argument("--region", default="postfork", type=str)
    return parser.parse_args()


def _aggregate(group: pd.DataFrame) -> Dict[str, float]:
    numel = float(group["numel"].sum())
    shape_numel = float(group["shape_numel"].sum())
    if numel <= 0 or shape_numel <= 0:
        raise ValueError("Cannot aggregate empty metric components.")
    num_samples = int(group["num_eval_samples"].iloc[0])
    if not group["num_eval_samples"].eq(num_samples).all():
        raise ValueError("num_eval_samples varies within an aggregate.")
    sse = float(group["sse"].sum())
    var_sum = float(group["var_sum"].sum())
    rmse = math.sqrt(sse / numel)
    spread = math.sqrt(var_sum / numel)
    correction = math.sqrt((num_samples + 1.0) / num_samples)
    return {
        "numel": numel,
        "shape_numel": shape_numel,
        "rmse": rmse,
        "crps": float(group["crps_sum"].sum()) / numel,
        "ensemble_spread": spread,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": float(group["coverage_count_90"].sum()) / numel,
        "gap_mass_error": float(group["gap_mass_error_sum"].sum()) / shape_numel,
        "branch_mass_error": float(group["branch_mass_error_sum"].sum()) / shape_numel,
        "pred_gap_mass": float(group["pred_gap_mass_sum"].sum()) / shape_numel,
        "oracle_gap_mass": float(group["oracle_gap_mass_sum"].sum()) / shape_numel,
        "pred_upper_mass": float(group["pred_upper_mass_sum"].sum()) / shape_numel,
        "oracle_upper_mass": float(group["oracle_upper_mass_sum"].sum()) / shape_numel,
        "num_eval_samples": num_samples,
    }


def check_pairing(df: pd.DataFrame) -> None:
    selected = df.loc[df["region"] == "postfork"]
    if set(selected["model_name"].unique()) != set(MODEL_ORDER):
        raise RuntimeError("Unexpected source names in marginal evaluation.")
    grouped = selected.groupby("task_index")
    if not grouped["task_fingerprint"].nunique().eq(1).all():
        raise RuntimeError("Task fingerprints are not paired.")
    if not grouped["model_name"].nunique().eq(len(MODEL_ORDER)).all():
        raise RuntimeError("Not every task is present for every source.")
    if selected.duplicated(["model_name", "task_index"]).any():
        raise RuntimeError("Duplicate source/task rows detected.")


def summary(df: pd.DataFrame, region: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    selected = df.loc[df["region"] == region]
    for model_name, group in selected.groupby("model_name", sort=False):
        row: Dict[str, float] = {"model_name": model_name}
        row.update(_aggregate(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values("model_name").reset_index(drop=True)


def _cluster_table(df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = df.loc[df["region"] == region]
    aggregation = {column: "sum" for column in COMPONENTS}
    aggregation["num_eval_samples"] = "first"
    return (
        selected.groupby(
            ["model_name", "generator_batch_index"], as_index=False
        )
        .agg(aggregation)
        .sort_values(["model_name", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays(components: Mapping[str, np.ndarray], num_samples: int) -> Dict[str, np.ndarray]:
    numel = components["numel"]
    shape_numel = components["shape_numel"]
    rmse = np.sqrt(components["sse"] / numel)
    spread = np.sqrt(components["var_sum"] / numel)
    correction = math.sqrt((num_samples + 1.0) / num_samples)
    return {
        "rmse": rmse,
        "crps": components["crps_sum"] / numel,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": components["coverage_count_90"] / numel,
        "gap_mass_error": components["gap_mass_error_sum"] / shape_numel,
        "branch_mass_error": components["branch_mass_error_sum"] / shape_numel,
    }


def paired_cluster_bootstrap(
    *,
    df: pd.DataFrame,
    point: pd.DataFrame,
    region: str,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df, region)
    reference = clusters.loc[
        clusters["model_name"] == MODEL_ORDER[0]
    ].sort_values("generator_batch_index")
    cluster_ids = reference["generator_batch_index"].to_numpy()
    num_clusters = int(cluster_ids.size)
    if num_clusters < 2:
        raise RuntimeError("Need at least two generator batches for bootstrap.")

    values: Dict[str, np.ndarray] = {
        metric: np.empty((replicates, len(MODEL_ORDER)), dtype=np.float64)
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        draws = rng.integers(
            0, num_clusters, size=(stop - start, num_clusters), endpoint=False
        )
        for model_index, model_name in enumerate(MODEL_ORDER):
            model_clusters = clusters.loc[
                clusters["model_name"] == model_name
            ].sort_values("generator_batch_index")
            if not np.array_equal(
                model_clusters["generator_batch_index"].to_numpy(), cluster_ids
            ):
                raise RuntimeError(f"Cluster pairing failed for {model_name}.")
            components = {
                column: model_clusters[column].to_numpy(dtype=np.float64)[draws].sum(axis=1)
                for column in COMPONENTS
            }
            sample_values = model_clusters["num_eval_samples"].unique()
            if len(sample_values) != 1:
                raise RuntimeError("num_eval_samples varies within a source.")
            arrays = _metric_arrays(components, int(sample_values[0]))
            for metric in METRICS:
                values[metric][start:stop, model_index] = arrays[metric]

    point_index = point.set_index("model_name")
    ci_rows = []
    delta_rows = []
    reference_index = MODEL_ORDER.index(REFERENCE)
    for model_index, model_name in enumerate(MODEL_ORDER):
        for metric in METRICS:
            distribution = values[metric][:, model_index]
            ci_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "estimate": float(point_index.loc[model_name, metric]),
                    "ci_low": float(np.quantile(distribution, 0.025)),
                    "ci_high": float(np.quantile(distribution, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_unit": "generator_batch",
                }
            )
            if model_name != REFERENCE:
                delta = distribution - values[metric][:, reference_index]
                delta_rows.append(
                    {
                        "model_name": model_name,
                        "reference_model": REFERENCE,
                        "metric": metric,
                        "estimate_delta": float(
                            point_index.loc[model_name, metric]
                            - point_index.loc[REFERENCE, metric]
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
    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def latex_table(table: pd.DataFrame) -> str:
    names = {
        "Exact marginal oracle": "Exact marginal oracle",
        "Gaussian TNP": "Gaussian TNP",
        "Dropout CRPS-TNP": "Dropout CRPS-TNP",
        "StochLN CRPS-TNP": "StochLN CRPS-TNP",
    }
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \begin{tabular}{@{}l c c c c c c@{}}",
        r"        \toprule",
        r"        Model & RMSE $\downarrow$ & CRPS $\downarrow$ & SSR & Coverage$_{90}$ & Gap-mass error $\downarrow$ & Branch-mass error $\downarrow$ \\",
        r"        \midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            "        "
            + names[str(row["model_name"])]
            + " & "
            + " & ".join(
                f"{float(row[column]):.3f}"
                for column in (
                    "rmse",
                    "crps",
                    "spread_skill_ratio",
                    "coverage_90",
                    "gap_mass_error",
                    "branch_mass_error",
                )
            )
            + r" \\"
        )
        if str(row["model_name"]) == "Exact marginal oracle":
            lines.append(r"        \midrule")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption[Ambiguous binary-fork marginal performance.]",
            r"    {Ambiguous-context performance on fully post-fork targets. All predictive distributions are represented by $M_{\mathrm{eval}}=256$ samples. Gap-mass and branch-mass errors are measured relative to the exact binary-mixture posterior.}",
            r"    \label{table:binary_fork_ambiguous}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = {
        "model_name",
        "region",
        "task_index",
        "generator_batch_index",
        "task_fingerprint",
        "num_eval_samples",
        *COMPONENTS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    check_pairing(df)
    main_summary = summary(df, args.region)
    ci, deltas = paired_cluster_bootstrap(
        df=df,
        point=main_summary,
        region=args.region,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        chunk_size=int(args.bootstrap_chunk_size),
    )

    main_summary.to_csv(output_dir / "summary_headline.csv", index=False)
    ci.to_csv(output_dir / "bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / "paired_cluster_bootstrap_deltas.csv", index=False)
    (output_dir / "binary_fork_ambiguous_table.tex").write_text(
        latex_table(main_summary)
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
    print("\nBINARY-FORK AMBIGUOUS MARGINALS\n")
    print(main_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPAIRED DIFFERENCES VS GAUSSIAN TNP\n")
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
