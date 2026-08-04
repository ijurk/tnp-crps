from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


ADDITIVE_COMPONENTS = [
    "numel",
    "sse",
    "crps_sum",
    "var_sum",
    "coverage_count_90",
    "width_sum_90",
]
SUPPORTED_METRICS = [
    "rmse",
    "crps",
    "ensemble_spread",
    "spread_skill_ratio",
    "coverage_90",
    "width_90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired cluster-bootstrap analysis of sawtooth marginals."
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=None, type=int)
    parser.add_argument("--bootstrap_chunk_size", default=250, type=int)
    return parser.parse_args()


def _aggregate(group: pd.DataFrame) -> Dict[str, float]:
    numel = float(group["numel"].sum())
    if numel <= 0:
        raise ValueError("Cannot aggregate zero target elements.")

    sample_values = group["num_eval_samples"].unique()
    if len(sample_values) != 1:
        raise ValueError("num_eval_samples varies within a source.")
    num_samples = int(sample_values[0])

    rmse = math.sqrt(float(group["sse"].sum()) / numel)
    spread = math.sqrt(float(group["var_sum"].sum()) / numel)
    correction = math.sqrt((num_samples + 1.0) / num_samples)

    return {
        "num_tasks": int(group["task_index"].nunique()),
        "numel": numel,
        "num_eval_samples": num_samples,
        "rmse": rmse,
        "crps": float(group["crps_sum"].sum()) / numel,
        "ensemble_spread": spread,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": float(group["coverage_count_90"].sum()) / numel,
        "width_90": float(group["width_sum_90"].sum()) / numel,
        "mean_num_context": float(group["num_context"].mean()),
        "minimum_num_context": int(group["num_context"].min()),
        "maximum_num_context": int(group["num_context"].max()),
    }


def check_pairing(df: pd.DataFrame, source_order: Sequence[str]) -> None:
    if set(df["model_name"].unique()) != set(source_order):
        raise RuntimeError(
            "Unexpected source names. "
            f"Expected {source_order}, got {sorted(df['model_name'].unique())}."
        )
    grouped = df.groupby("task_index")
    if not grouped["task_fingerprint"].nunique().eq(1).all():
        raise RuntimeError("Task fingerprints are not paired.")
    if not grouped["model_name"].nunique().eq(len(source_order)).all():
        raise RuntimeError("Not every task is present for every source.")
    if df.duplicated(["model_name", "task_index"]).any():
        raise RuntimeError("Duplicate source/task rows detected.")


def point_summary(df: pd.DataFrame, source_order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name in source_order:
        group = df.loc[df["model_name"] == model_name]
        if group.empty:
            raise RuntimeError(f"Missing source {model_name!r}.")
        row: Dict[str, Any] = {"model_name": model_name}
        row.update(_aggregate(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregation = {component: "sum" for component in ADDITIVE_COMPONENTS}
    aggregation["num_eval_samples"] = "first"
    return (
        df.groupby(["model_name", "generator_batch_index"], as_index=False)
        .agg(aggregation)
        .sort_values(["model_name", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays(
    components: Mapping[str, np.ndarray],
    num_samples: int,
) -> Dict[str, np.ndarray]:
    numel = components["numel"]
    rmse = np.sqrt(components["sse"] / numel)
    spread = np.sqrt(components["var_sum"] / numel)
    correction = math.sqrt((num_samples + 1.0) / num_samples)
    return {
        "rmse": rmse,
        "crps": components["crps_sum"] / numel,
        "ensemble_spread": spread,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": components["coverage_count_90"] / numel,
        "width_90": components["width_sum_90"] / numel,
    }


def paired_cluster_bootstrap(
    *,
    df: pd.DataFrame,
    point: pd.DataFrame,
    source_order: Sequence[str],
    references: Sequence[str],
    metrics: Sequence[str],
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df)
    first = clusters.loc[
        clusters["model_name"] == source_order[0]
    ].sort_values("generator_batch_index")
    cluster_ids = first["generator_batch_index"].to_numpy()
    num_clusters = int(cluster_ids.size)
    if num_clusters < 2:
        raise RuntimeError("Need at least two generator batches for bootstrap.")

    values: Dict[str, np.ndarray] = {
        metric: np.empty((replicates, len(source_order)), dtype=np.float64)
        for metric in metrics
    }
    rng = np.random.default_rng(seed)

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        draws = rng.integers(
            0,
            num_clusters,
            size=(stop - start, num_clusters),
            endpoint=False,
        )

        for model_index, model_name in enumerate(source_order):
            model_clusters = clusters.loc[
                clusters["model_name"] == model_name
            ].sort_values("generator_batch_index")
            if not np.array_equal(
                model_clusters["generator_batch_index"].to_numpy(),
                cluster_ids,
            ):
                raise RuntimeError(f"Cluster pairing failed for {model_name}.")

            components = {
                column: model_clusters[column]
                .to_numpy(dtype=np.float64)[draws]
                .sum(axis=1)
                for column in ADDITIVE_COMPONENTS
            }
            sample_values = model_clusters["num_eval_samples"].unique()
            if len(sample_values) != 1:
                raise RuntimeError("num_eval_samples varies within a source.")
            arrays = _metric_arrays(components, int(sample_values[0]))
            for metric in metrics:
                values[metric][start:stop, model_index] = arrays[metric]

    point_index = point.set_index("model_name")
    ci_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for model_index, model_name in enumerate(source_order):
        for metric in metrics:
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

    for reference in references:
        if reference not in source_order:
            raise ValueError(f"Unknown bootstrap reference {reference!r}.")
        reference_index = source_order.index(reference)
        for model_index, model_name in enumerate(source_order):
            if model_name == reference:
                continue
            for metric in metrics:
                delta = values[metric][:, model_index] - values[metric][
                    :, reference_index
                ]
                low = float(np.quantile(delta, 0.025))
                high = float(np.quantile(delta, 0.975))
                delta_rows.append(
                    {
                        "model_name": model_name,
                        "reference_model": reference,
                        "metric": metric,
                        "estimate_delta": float(
                            point_index.loc[model_name, metric]
                            - point_index.loc[reference, metric]
                        ),
                        "ci_low": low,
                        "ci_high": high,
                        "ci_contains_zero": bool(low <= 0.0 <= high),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "generator_batch",
                    }
                )

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def latex_table(
    *,
    table: pd.DataFrame,
    display_names: Mapping[str, str],
    columns: Sequence[str],
    column_labels: Mapping[str, str],
    filename_label: str,
) -> str:
    alignment = "l " + " ".join("c" for _ in columns)
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        rf"    \begin{{tabular}}{{@{{}}{alignment}@{{}}}}",
        r"        \toprule",
        "        Model & "
        + " & ".join(column_labels[column] for column in columns)
        + r" \\",
        r"        \midrule",
    ]

    for _, row in table.iterrows():
        model_name = str(row["model_name"])
        display = display_names.get(model_name, model_name)
        lines.append(
            "        "
            + display
            + " & "
            + " & ".join(f"{float(row[column]):.3f}" for column in columns)
            + r" \\"
        )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \label{{table:{filename_label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Analysis config must resolve to a dictionary.")
    analysis_cfg = dict(cfg["analysis"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    source_order = [str(value) for value in analysis_cfg["source_order"]]
    display_names = {
        str(key): str(value)
        for key, value in dict(analysis_cfg.get("display_names", {})).items()
    }
    references = [str(value) for value in analysis_cfg.get("references", [])]
    metrics = [str(value) for value in analysis_cfg.get("bootstrap_metrics", SUPPORTED_METRICS)]
    unsupported = set(metrics).difference(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Unsupported bootstrap metrics: {sorted(unsupported)}")

    check_pairing(df, source_order)
    point = point_summary(df, source_order)

    runtime_input = Path(args.input).with_name("runtime_by_source.csv")
    if runtime_input.is_file():
        runtime = pd.read_csv(runtime_input)
        point = point.merge(runtime, on="model_name", how="left")

    seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else analysis_cfg.get("bootstrap_seed", 20260911)
    )
    ci, deltas = paired_cluster_bootstrap(
        df=df,
        point=point,
        source_order=source_order,
        references=references,
        metrics=metrics,
        replicates=int(args.bootstrap_replicates),
        seed=seed,
        chunk_size=int(args.bootstrap_chunk_size),
    )

    prefix = str(analysis_cfg.get("output_prefix", "sawtooth"))
    point.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    ci.to_csv(output_dir / f"{prefix}_bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / f"{prefix}_paired_deltas.csv", index=False)

    table_cfg = dict(analysis_cfg["table"])
    table_columns = [str(value) for value in table_cfg["columns"]]
    column_labels = {
        str(key): str(value)
        for key, value in dict(table_cfg["column_labels"]).items()
    }
    table_text = latex_table(
        table=point,
        display_names=display_names,
        columns=table_columns,
        column_labels=column_labels,
        filename_label=str(table_cfg["label"]),
    )
    table_path = output_dir / str(table_cfg["filename"])
    table_path.write_text(table_text)

    analysis_record = {
        "input": str(args.input),
        "config": str(args.config),
        "source_order": source_order,
        "references": references,
        "bootstrap_metrics": metrics,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": seed,
        "bootstrap_unit": "generator_batch",
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(analysis_record, indent=2)
    )

    print("\nSAWTOOTH MARGINAL SUMMARY\n")
    print(point[["model_name", *table_columns]].to_string(index=False))
    print(f"\nWrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
