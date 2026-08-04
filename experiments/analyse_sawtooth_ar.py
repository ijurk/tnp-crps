from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


METRICS = [
    "rmse",
    "crps",
    "energy",
    "coverage90",
    "width90",
    "runtime_s_per_task",
]
ADDITIVE_COLUMNS = [
    "mse",
    "crps",
    "energy",
    "coverage90",
    "width90",
    "runtime_s_per_task",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap analysis of final sawtooth AR evaluation."
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=None, type=int)
    parser.add_argument("--bootstrap_chunk_size", default=250, type=int)
    return parser.parse_args()


def check_pairing(df: pd.DataFrame, source_order: Sequence[str]) -> None:
    if set(df["source"].unique()) != set(source_order):
        raise RuntimeError(
            f"Unexpected source names. Expected {source_order}, "
            f"got {sorted(df['source'].unique())}."
        )
    grouped = df.groupby("task_id")
    if not grouped["source"].nunique().eq(len(source_order)).all():
        raise RuntimeError("Not every AR task is present for every source.")
    if df.duplicated(["source", "task_id"]).any():
        raise RuntimeError("Duplicate source/task rows detected.")
    if grouped["nc"].nunique().gt(1).any():
        raise RuntimeError("Paired AR task rows disagree on initial context size.")


def point_summary(df: pd.DataFrame, source_order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source in source_order:
        group = df.loc[df["source"] == source]
        if group.empty:
            raise RuntimeError(f"Missing source {source!r}.")
        rows.append(
            {
                "source": source,
                "num_tasks": int(group["task_id"].nunique()),
                "nc": int(group["nc"].iloc[0]),
                "rmse": math.sqrt(float(group["mse"].mean())),
                "crps": float(group["crps"].mean()),
                "energy": float(group["energy"].mean()),
                "coverage90": float(group["coverage90"].mean()),
                "width90": float(group["width90"].mean()),
                "runtime_s_per_task": float(group["runtime_s_per_task"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _cluster_table(df: pd.DataFrame, eval_batch_size: int) -> pd.DataFrame:
    work = df.copy()
    work["generator_batch_index"] = work["task_id"].astype(int) // int(
        eval_batch_size
    )
    return (
        work.groupby(["source", "generator_batch_index"], as_index=False)[
            ADDITIVE_COLUMNS
        ]
        .sum()
        .sort_values(["source", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays(
    sums: Mapping[str, np.ndarray],
    tasks_per_replicate: int,
) -> Dict[str, np.ndarray]:
    divisor = float(tasks_per_replicate)
    return {
        "rmse": np.sqrt(sums["mse"] / divisor),
        "crps": sums["crps"] / divisor,
        "energy": sums["energy"] / divisor,
        "coverage90": sums["coverage90"] / divisor,
        "width90": sums["width90"] / divisor,
        "runtime_s_per_task": sums["runtime_s_per_task"] / divisor,
    }


def paired_cluster_bootstrap(
    *,
    df: pd.DataFrame,
    point: pd.DataFrame,
    source_order: Sequence[str],
    references: Sequence[str],
    metrics: Sequence[str],
    eval_batch_size: int,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df, eval_batch_size)
    first = clusters.loc[clusters["source"] == source_order[0]].sort_values(
        "generator_batch_index"
    )
    cluster_ids = first["generator_batch_index"].to_numpy()
    num_clusters = int(cluster_ids.size)
    if num_clusters < 2:
        raise RuntimeError("Need at least two AR generator batches.")
    tasks_per_replicate = num_clusters * int(eval_batch_size)

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
        for source_index, source in enumerate(source_order):
            source_clusters = clusters.loc[
                clusters["source"] == source
            ].sort_values("generator_batch_index")
            if not np.array_equal(
                source_clusters["generator_batch_index"].to_numpy(), cluster_ids
            ):
                raise RuntimeError(f"AR cluster pairing failed for {source}.")
            sums = {
                column: source_clusters[column]
                .to_numpy(dtype=np.float64)[draws]
                .sum(axis=1)
                for column in ADDITIVE_COLUMNS
            }
            arrays = _metric_arrays(sums, tasks_per_replicate)
            for metric in metrics:
                values[metric][start:stop, source_index] = arrays[metric]

    point_index = point.set_index("source")
    ci_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for source_index, source in enumerate(source_order):
        for metric in metrics:
            distribution = values[metric][:, source_index]
            ci_rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "estimate": float(point_index.loc[source, metric]),
                    "ci_low": float(np.quantile(distribution, 0.025)),
                    "ci_high": float(np.quantile(distribution, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_unit": "generator_batch",
                }
            )

    for reference in references:
        if reference not in source_order:
            raise ValueError(f"Unknown AR reference {reference!r}.")
        reference_index = source_order.index(reference)
        for source_index, source in enumerate(source_order):
            if source == reference:
                continue
            for metric in metrics:
                delta = values[metric][:, source_index] - values[metric][
                    :, reference_index
                ]
                low = float(np.quantile(delta, 0.025))
                high = float(np.quantile(delta, 0.975))
                delta_rows.append(
                    {
                        "source": source,
                        "reference_source": reference,
                        "metric": metric,
                        "estimate_delta": float(
                            point_index.loc[source, metric]
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


def select_figure_tasks(
    *,
    df: pd.DataFrame,
    learned_sources: Sequence[str],
    quantiles: Mapping[str, float],
) -> Dict[str, Any]:
    selected = df.loc[df["source"].isin(learned_sources)]
    pivot = selected.pivot(index="task_id", columns="source", values="rmse_task")
    if set(pivot.columns) != set(learned_sources):
        raise RuntimeError("Task-difficulty table is missing a learned source.")
    difficulty = pivot[list(learned_sources)].mean(axis=1)

    result: Dict[str, Any] = {
        "definition": (
            "Mean task-level AR RMSE across the learned Gaussian, Dropout and "
            "StochLN curriculum models; trivial baseline excluded."
        ),
        "learned_sources": list(learned_sources),
        "tasks": {},
    }

    used: set[int] = set()
    for label, quantile in quantiles.items():
        target = float(difficulty.quantile(float(quantile)))
        candidates = (
            difficulty.sub(target)
            .abs()
            .sort_values(kind="stable")
            .index.astype(int)
            .tolist()
        )
        task_id = next(task for task in candidates if task not in used)
        used.add(task_id)
        result["tasks"][str(label)] = {
            "task_id": int(task_id),
            "target_quantile": float(quantile),
            "target_difficulty": target,
            "selected_difficulty": float(difficulty.loc[task_id]),
            "source_rmse": {
                source: float(pivot.loc[task_id, source])
                for source in learned_sources
            },
        }

    return result


def latex_table(
    *,
    table: pd.DataFrame,
    display_names: Mapping[str, str],
    columns: Sequence[str],
    labels: Mapping[str, str],
    table_label: str,
) -> str:
    alignment = "l " + " ".join("c" for _ in columns)
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        rf"    \begin{{tabular}}{{@{{}}{alignment}@{{}}}}",
        r"        \toprule",
        "        Model & " + " & ".join(labels[column] for column in columns) + r" \\",
        r"        \midrule",
    ]
    for _, row in table.iterrows():
        source = str(row["source"])
        lines.append(
            "        "
            + display_names.get(source, source)
            + " & "
            + " & ".join(f"{float(row[column]):.3f}" for column in columns)
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \label{{table:{table_label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("AR config must resolve to a dictionary.")
    analysis_cfg = dict(cfg["analysis"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    source_order = [str(value) for value in analysis_cfg["source_order"]]
    learned_sources = [str(value) for value in analysis_cfg["learned_sources"]]
    references = [str(value) for value in analysis_cfg.get("references", [])]
    metrics = [str(value) for value in analysis_cfg.get("bootstrap_metrics", METRICS)]
    unsupported = set(metrics).difference(METRICS)
    if unsupported:
        raise ValueError(f"Unsupported AR bootstrap metrics: {sorted(unsupported)}")

    check_pairing(df, source_order)
    point = point_summary(df, source_order)
    seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else analysis_cfg.get("bootstrap_seed", 20260913)
    )
    ci, deltas = paired_cluster_bootstrap(
        df=df,
        point=point,
        source_order=source_order,
        references=references,
        metrics=metrics,
        eval_batch_size=int(cfg["eval_batch_size"]),
        replicates=int(args.bootstrap_replicates),
        seed=seed,
        chunk_size=int(args.bootstrap_chunk_size),
    )

    selected_tasks = select_figure_tasks(
        df=df,
        learned_sources=learned_sources,
        quantiles={
            str(key): float(value)
            for key, value in dict(analysis_cfg["figure_task_quantiles"]).items()
        },
    )

    point.to_csv(output_dir / "sawtooth_ar_summary_final.csv", index=False)
    ci.to_csv(output_dir / "sawtooth_ar_bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / "sawtooth_ar_paired_deltas.csv", index=False)
    (output_dir / "selected_figure_tasks.json").write_text(
        json.dumps(selected_tasks, indent=2)
    )

    table_cfg = dict(analysis_cfg["table"])
    table_text = latex_table(
        table=point,
        display_names={
            str(key): str(value)
            for key, value in dict(analysis_cfg.get("display_names", {})).items()
        },
        columns=[str(value) for value in table_cfg["columns"]],
        labels={
            str(key): str(value)
            for key, value in dict(table_cfg["column_labels"]).items()
        },
        table_label=str(table_cfg["label"]),
    )
    (output_dir / str(table_cfg["filename"])).write_text(table_text)

    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "config": str(args.config),
                "source_order": source_order,
                "learned_sources": learned_sources,
                "references": references,
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "bootstrap_seed": seed,
                "bootstrap_unit": "generator_batch",
                "figure_task_selection": selected_tasks,
            },
            indent=2,
        )
    )

    print("\nSAWTOOTH AR SUMMARY\n")
    print(point.to_string(index=False))
    print("\nSELECTED FIGURE TASKS\n")
    print(json.dumps(selected_tasks, indent=2))
    print(f"\nWrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
