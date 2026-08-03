from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Exact joint oracle",
    "Gaussian TNP",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]
DEPLOYMENT_ORDER = ["direct", "autoregressive"]
METRICS = [
    "zero_switch_rate",
    "mean_switch_count",
    "mid_gap_rate",
    "independence_zero_switch",
    "independence_mean_switches",
    "independence_mid_gap_rate",
    "excess_zero_switch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=20260903, type=int)
    parser.add_argument("--bootstrap_chunk_size", default=500, type=int)
    return parser.parse_args()


def check_pairing(df: pd.DataFrame) -> None:
    if set(df["model_name"].unique()) != set(MODEL_ORDER):
        raise RuntimeError("Unexpected coherence source set.")
    if set(df["deployment"].unique()) != set(DEPLOYMENT_ORDER):
        raise RuntimeError("Unexpected deployment set.")
    for deployment, group in df.groupby("deployment"):
        by_task = group.groupby("task_index")
        if not by_task["model_name"].nunique().eq(len(MODEL_ORDER)).all():
            raise RuntimeError(f"Source pairing failed for {deployment}.")
        if not by_task["task_fingerprint"].nunique().eq(1).all():
            raise RuntimeError(f"Fingerprint pairing failed for {deployment}.")
        if group.duplicated(["model_name", "task_index"]).any():
            raise RuntimeError(f"Duplicate source/task rows for {deployment}.")
    cross_deployment = df.groupby("task_index")["task_fingerprint"].nunique()
    if not cross_deployment.eq(1).all():
        raise RuntimeError("Task fingerprints differ between deployments.")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (model_name, deployment), group in df.groupby(
        ["model_name", "deployment"], sort=False
    ):
        num_paths = float(group["num_paths"].sum())
        num_tasks = float(len(group))
        row: Dict[str, float] = {
            "model_name": model_name,
            "deployment": deployment,
            "num_tasks": int(num_tasks),
            "num_paths": int(num_paths),
            "zero_switch_rate": float(group["zero_switch_sum"].sum()) / num_paths,
            "mean_switch_count": float(group["switch_count_sum"].sum()) / num_paths,
            "mid_gap_rate": float(group["gap_fraction_sum"].sum()) / num_paths,
            "independence_zero_switch": float(
                group["independence_zero_switch"].sum()
            ) / num_tasks,
            "independence_mean_switches": float(
                group["independence_mean_switches"].sum()
            ) / num_tasks,
            "independence_mid_gap_rate": float(
                group["independence_mid_gap_rate"].sum()
            ) / num_tasks,
        }
        row["excess_zero_switch"] = (
            row["zero_switch_rate"] - row["independence_zero_switch"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=MODEL_ORDER, ordered=True
    )
    out["deployment"] = pd.Categorical(
        out["deployment"], categories=DEPLOYMENT_ORDER, ordered=True
    )
    return out.sort_values(["model_name", "deployment"]).reset_index(drop=True)


def _cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        "num_paths": "sum",
        "zero_switch_sum": "sum",
        "switch_count_sum": "sum",
        "gap_fraction_sum": "sum",
        "independence_zero_switch": "sum",
        "independence_mean_switches": "sum",
        "independence_mid_gap_rate": "sum",
        "task_index": "count",
    }
    return (
        df.groupby(
            ["model_name", "deployment", "generator_batch_index"],
            as_index=False,
        )
        .agg(aggregation)
        .rename(columns={"task_index": "num_tasks"})
        .sort_values(["model_name", "deployment", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays(group: pd.DataFrame, draws: np.ndarray) -> Dict[str, np.ndarray]:
    paths = group["num_paths"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
    tasks = group["num_tasks"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
    zero = group["zero_switch_sum"].to_numpy(dtype=np.float64)[draws].sum(axis=1) / paths
    switches = group["switch_count_sum"].to_numpy(dtype=np.float64)[draws].sum(axis=1) / paths
    gap = group["gap_fraction_sum"].to_numpy(dtype=np.float64)[draws].sum(axis=1) / paths
    indep_zero = (
        group["independence_zero_switch"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        / tasks
    )
    indep_switch = (
        group["independence_mean_switches"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        / tasks
    )
    indep_gap = (
        group["independence_mid_gap_rate"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        / tasks
    )
    return {
        "zero_switch_rate": zero,
        "mean_switch_count": switches,
        "mid_gap_rate": gap,
        "independence_zero_switch": indep_zero,
        "independence_mean_switches": indep_switch,
        "independence_mid_gap_rate": indep_gap,
        "excess_zero_switch": zero - indep_zero,
    }


def paired_bootstrap(
    *,
    df: pd.DataFrame,
    point: pd.DataFrame,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df)
    reference = clusters.loc[
        (clusters["model_name"] == MODEL_ORDER[0])
        & (clusters["deployment"] == DEPLOYMENT_ORDER[0])
    ].sort_values("generator_batch_index")
    cluster_ids = reference["generator_batch_index"].to_numpy()
    num_clusters = int(cluster_ids.size)
    rng = np.random.default_rng(seed)

    distributions: Dict[Tuple[str, str, str], np.ndarray] = {}
    for model in MODEL_ORDER:
        for deployment in DEPLOYMENT_ORDER:
            for metric in METRICS:
                distributions[(model, deployment, metric)] = np.empty(
                    replicates, dtype=np.float64
                )

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        draws = rng.integers(
            0, num_clusters, size=(stop - start, num_clusters), endpoint=False
        )
        for model in MODEL_ORDER:
            for deployment in DEPLOYMENT_ORDER:
                group = clusters.loc[
                    (clusters["model_name"] == model)
                    & (clusters["deployment"] == deployment)
                ].sort_values("generator_batch_index")
                if not np.array_equal(
                    group["generator_batch_index"].to_numpy(), cluster_ids
                ):
                    raise RuntimeError(
                        f"Cluster pairing failed for {model}, {deployment}."
                    )
                arrays = _metric_arrays(group, draws)
                for metric in METRICS:
                    distributions[(model, deployment, metric)][start:stop] = arrays[metric]

    point_index = point.set_index(["model_name", "deployment"])
    ci_rows = []
    deployment_rows = []
    for model in MODEL_ORDER:
        for deployment in DEPLOYMENT_ORDER:
            for metric in METRICS:
                values = distributions[(model, deployment, metric)]
                ci_rows.append(
                    {
                        "model_name": model,
                        "deployment": deployment,
                        "metric": metric,
                        "estimate": float(point_index.loc[(model, deployment), metric]),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "generator_batch",
                    }
                )
        for metric in METRICS:
            delta = (
                distributions[(model, "autoregressive", metric)]
                - distributions[(model, "direct", metric)]
            )
            deployment_rows.append(
                {
                    "model_name": model,
                    "metric": metric,
                    "estimate_delta_ar_minus_direct": float(
                        point_index.loc[(model, "autoregressive"), metric]
                        - point_index.loc[(model, "direct"), metric]
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
    return pd.DataFrame(ci_rows), pd.DataFrame(deployment_rows)


def latex_table(result: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \begin{tabular}{@{}l c c c c c c@{}}",
        r"        \toprule",
        r"        & \multicolumn{3}{c}{Direct} & \multicolumn{3}{c}{Autoregressive} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"        Model & Zero-switch [ind.] & Mean switches & Mid-gap & Zero-switch [ind.] & Mean switches & Mid-gap \\",
        r"        \midrule",
    ]
    for model in MODEL_ORDER:
        direct = result.loc[
            (result["model_name"] == model) & (result["deployment"] == "direct")
        ].iloc[0]
        ar = result.loc[
            (result["model_name"] == model)
            & (result["deployment"] == "autoregressive")
        ].iloc[0]
        lines.append(
            f"        {model} & "
            f"{direct['zero_switch_rate']:.3f} [{direct['independence_zero_switch']:.3f}] & "
            f"{direct['mean_switch_count']:.3f} & {direct['mid_gap_rate']:.3f} & "
            f"{ar['zero_switch_rate']:.3f} [{ar['independence_zero_switch']:.3f}] & "
            f"{ar['mean_switch_count']:.3f} & {ar['mid_gap_rate']:.3f} \\\\"
        )
        if model == "Exact joint oracle":
            lines.append(r"        \midrule")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption[Direct and autoregressive binary-fork coherence.]",
            r"    {Function-level coherence over paired binary-fork tasks. Zero-switch is the fraction of sampled paths that retain one active branch across the post-fork anchors; bracketed values are the exact no-switch probabilities implied by independent draws from the same empirical pointwise lower, gap and upper probabilities. Mean switches counts changes between active lower and upper assignments, ignoring intervening gap points. The Exact joint oracle is repeated in the autoregressive columns as a reference.}",
            r"    \label{table:binary_fork_coherence}",
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
    result = summary(df)
    ci, deltas = paired_bootstrap(
        df=df,
        point=result,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        chunk_size=int(args.bootstrap_chunk_size),
    )
    result.to_csv(output_dir / "summary_coherence.csv", index=False)
    ci.to_csv(output_dir / "bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / "ar_minus_direct_deltas.csv", index=False)
    (output_dir / "binary_fork_coherence_table.tex").write_text(
        latex_table(result)
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "bootstrap_seed": int(args.bootstrap_seed),
                "bootstrap_unit": "generator_batch",
            },
            indent=2,
        )
    )
    print("\nCOHERENCE SUMMARY\n")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAR MINUS DIRECT\n")
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
