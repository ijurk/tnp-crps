from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Exact GP oracle",
    "Exact GP oracle (M=64)",
    "Gaussian TNP",
    "Gaussian TNP (M=64)",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
]

# Headline deltas compare finite-ensemble sources under identical estimators,
# so the sampled Gaussian TNP is the reference. The analytic rows remain as
# the estimator-bias diagnostic via FINITE_M_TWIN_PAIRS.
DELTA_REFERENCE = "Gaussian TNP (M=64)"

FINITE_M_TWIN_PAIRS = [
    ("Exact GP oracle", "Exact GP oracle (M=64)"),
    ("Gaussian TNP", "Gaussian TNP (M=64)"),
]

HEADLINE_MODEL_LABELS = {
    "Exact GP oracle (M=64)": "Exact GP oracle",
    "Gaussian TNP (M=64)": "Gaussian TNP",
    "Dropout CRPS-TNP": "Dropout CRPS-TNP",
    "StochLN CRPS-TNP": "StochLN CRPS-TNP",
}

METRICS = [
    "rmse",
    "crps",
    "spread_skill_ratio",
    "coverage_90",
    "width_90",
]

COMPONENT_COLUMNS = [
    "numel",
    "sse",
    "crps_sum",
    "var_sum",
    "coverage_count_90",
    "width_sum_90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and cluster-bootstrap Gaussian-control metrics."
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bootstrap_replicates", default=10_000, type=int)
    parser.add_argument("--bootstrap_seed", default=20260831, type=int)
    parser.add_argument("--region", default="all", type=str)
    return parser.parse_args()


def _finite_m_correction(finite_ensemble: bool, num_eval_samples: int) -> float:
    if not finite_ensemble:
        return 1.0
    if num_eval_samples < 2:
        raise ValueError(
            f"Finite ensemble requires num_eval_samples >= 2, got {num_eval_samples}."
        )
    return math.sqrt((float(num_eval_samples) + 1.0) / float(num_eval_samples))


def aggregate_components(group: pd.DataFrame) -> Dict[str, float]:
    numel = float(group["numel"].sum())
    if numel <= 0:
        raise ValueError("Cannot aggregate a group with zero selected target elements.")

    sse = float(group["sse"].sum())
    crps_sum = float(group["crps_sum"].sum())
    var_sum = float(group["var_sum"].sum())
    coverage_count = float(group["coverage_count_90"].sum())
    width_sum = float(group["width_sum_90"].sum())

    finite_values = group["finite_ensemble"].drop_duplicates().tolist()
    sample_values = group["num_eval_samples"].drop_duplicates().tolist()

    if len(finite_values) != 1 or len(sample_values) != 1:
        raise ValueError(
            "finite_ensemble and num_eval_samples must be constant within an aggregate."
        )

    finite_ensemble = bool(finite_values[0])
    num_eval_samples = int(sample_values[0])
    correction = _finite_m_correction(finite_ensemble, num_eval_samples)

    rmse = math.sqrt(sse / numel)
    spread = math.sqrt(var_sum / numel)

    return {
        "numel": numel,
        "rmse": rmse,
        "crps": crps_sum / numel,
        "ensemble_spread": spread,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": coverage_count / numel,
        "width_90": width_sum / numel,
        "num_eval_samples": num_eval_samples,
        "finite_ensemble": finite_ensemble,
    }


def check_pairing(df: pd.DataFrame, model_order: Sequence[str]) -> None:
    all_region = df.loc[df["region"] == "all"].copy()

    expected_models = set(model_order)
    found_models = set(all_region["model_name"].unique())
    if found_models != expected_models:
        raise RuntimeError(
            f"Unexpected model set. Expected {expected_models}, found {found_models}."
        )

    per_task = all_region.groupby(["eval_set", "task_index"])
    fingerprint_nunique = per_task["task_fingerprint"].nunique()
    if not (fingerprint_nunique == 1).all():
        raise RuntimeError(
            "Pairing failure: at least one eval_set/task_index has multiple fingerprints."
        )

    model_nunique = per_task["model_name"].nunique()
    if not (model_nunique == len(model_order)).all():
        bad = model_nunique[model_nunique != len(model_order)].head()
        raise RuntimeError(
            "Pairing failure: not every task is present for every model.\n"
            f"{bad}"
        )

    for column in ("num_context", "num_targets", "generator_batch_index"):
        nunique = per_task[column].nunique()
        if not (nunique == 1).all():
            raise RuntimeError(
                f"Pairing failure: {column} differs across models for a task."
            )

    duplicates = all_region.duplicated(
        ["model_name", "eval_set", "task_index"],
        keep=False,
    )
    if duplicates.any():
        raise RuntimeError("Duplicate model/eval_set/task_index rows were found.")


def summary_by_kernel(df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = df.loc[df["region"] == region]
    rows: List[Dict[str, float]] = []

    for (model_name, eval_set, kernel_name), group in selected.groupby(
        ["model_name", "eval_set", "kernel_name"],
        sort=False,
    ):
        row: Dict[str, float] = {
            "model_name": model_name,
            "eval_set": eval_set,
            "kernel_name": kernel_name,
        }
        row.update(aggregate_components(group))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values(["model_name", "eval_set"]).reset_index(drop=True)


def macro_summary(by_kernel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for model_name, group in by_kernel.groupby("model_name", observed=False):
        if pd.isna(model_name):
            continue
        row: Dict[str, float] = {
            "model_name": str(model_name),
            "num_kernels": int(group["eval_set"].nunique()),
        }
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        row["ensemble_spread"] = float(group["ensemble_spread"].mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values("model_name").reset_index(drop=True)


def pooled_summary(df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = df.loc[df["region"] == region]
    rows: List[Dict[str, float]] = []

    for model_name, group in selected.groupby("model_name", sort=False):
        row: Dict[str, float] = {"model_name": model_name}
        row.update(aggregate_components(group))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values("model_name").reset_index(drop=True)


def _cluster_table(df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = df.loc[df["region"] == region].copy()

    aggregation: Dict[str, str] = {
        "numel": "sum",
        "sse": "sum",
        "crps_sum": "sum",
        "var_sum": "sum",
        "coverage_count_90": "sum",
        "width_sum_90": "sum",
        "finite_ensemble": "first",
        "num_eval_samples": "first",
    }

    return (
        selected.groupby(
            ["model_name", "eval_set", "generator_batch_index"],
            as_index=False,
        )
        .agg(aggregation)
        .sort_values(["model_name", "eval_set", "generator_batch_index"])
        .reset_index(drop=True)
    )


def _metric_arrays_from_boot_components(
    *,
    numel: np.ndarray,
    sse: np.ndarray,
    crps_sum: np.ndarray,
    var_sum: np.ndarray,
    coverage_count: np.ndarray,
    width_sum: np.ndarray,
    finite_ensemble: bool,
    num_eval_samples: int,
) -> Mapping[str, np.ndarray]:
    rmse = np.sqrt(sse / numel)
    spread = np.sqrt(var_sum / numel)
    correction = _finite_m_correction(finite_ensemble, num_eval_samples)

    return {
        "rmse": rmse,
        "crps": crps_sum / numel,
        "spread_skill_ratio": correction * spread / (rmse + 1.0e-12),
        "coverage_90": coverage_count / numel,
        "width_90": width_sum / numel,
    }


def paired_cluster_bootstrap(
    *,
    df: pd.DataFrame,
    macro: pd.DataFrame,
    region: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = _cluster_table(df, region)
    kernels = list(dict.fromkeys(clusters["eval_set"].tolist()))
    models = MODEL_ORDER

    rng = np.random.default_rng(bootstrap_seed)
    boot: Dict[str, np.ndarray] = {
        metric: np.zeros((bootstrap_replicates, len(models)), dtype=np.float64)
        for metric in METRICS
    }

    for kernel in kernels:
        reference = clusters.loc[
            (clusters["model_name"] == models[0])
            & (clusters["eval_set"] == kernel)
        ].sort_values("generator_batch_index")

        cluster_ids = reference["generator_batch_index"].to_numpy()
        num_clusters = int(cluster_ids.size)
        if num_clusters < 2:
            raise RuntimeError(
                f"Need at least two generator batches for bootstrap, got {num_clusters}."
            )

        draw_indices = rng.integers(
            0,
            num_clusters,
            size=(bootstrap_replicates, num_clusters),
            endpoint=False,
        )

        for model_index, model_name in enumerate(models):
            model_clusters = clusters.loc[
                (clusters["model_name"] == model_name)
                & (clusters["eval_set"] == kernel)
            ].sort_values("generator_batch_index")

            model_ids = model_clusters["generator_batch_index"].to_numpy()
            if not np.array_equal(model_ids, cluster_ids):
                raise RuntimeError(
                    f"Cluster IDs do not pair for model={model_name}, kernel={kernel}."
                )

            def sampled_sum(column: str) -> np.ndarray:
                values = model_clusters[column].to_numpy(dtype=np.float64)
                return values[draw_indices].sum(axis=1)

            finite_values = model_clusters["finite_ensemble"].unique()
            sample_values = model_clusters["num_eval_samples"].unique()
            if len(finite_values) != 1 or len(sample_values) != 1:
                raise RuntimeError("Finite-ensemble metadata varies within a model/kernel.")

            metric_arrays = _metric_arrays_from_boot_components(
                numel=sampled_sum("numel"),
                sse=sampled_sum("sse"),
                crps_sum=sampled_sum("crps_sum"),
                var_sum=sampled_sum("var_sum"),
                coverage_count=sampled_sum("coverage_count_90"),
                width_sum=sampled_sum("width_sum_90"),
                finite_ensemble=bool(finite_values[0]),
                num_eval_samples=int(sample_values[0]),
            )

            for metric in METRICS:
                boot[metric][:, model_index] += metric_arrays[metric] / float(
                    len(kernels)
                )

    point = macro.set_index("model_name")
    ci_rows: List[Dict[str, float]] = []
    delta_rows: List[Dict[str, float]] = []
    reference_index = models.index(DELTA_REFERENCE)

    for model_index, model_name in enumerate(models):
        for metric in METRICS:
            values = boot[metric][:, model_index]
            ci_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "estimate": float(point.loc[model_name, metric]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "generator_batch_within_kernel",
                }
            )

            if model_name != DELTA_REFERENCE:
                delta = values - boot[metric][:, reference_index]
                estimate_delta = float(
                    point.loc[model_name, metric]
                    - point.loc[DELTA_REFERENCE, metric]
                )
                delta_rows.append(
                    {
                        "model_name": model_name,
                        "reference_model": DELTA_REFERENCE,
                        "metric": metric,
                        "estimate_delta": estimate_delta,
                        "ci_low": float(np.quantile(delta, 0.025)),
                        "ci_high": float(np.quantile(delta, 0.975)),
                        "ci_contains_zero": bool(
                            np.quantile(delta, 0.025) <= 0.0 <= np.quantile(delta, 0.975)
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_unit": "generator_batch_within_kernel",
                    }
                )

    # Finite-M estimator-bias diagnostic: the sampled twin minus its analytic
    # counterpart, using the same bootstrap draws so the CI is paired.
    for analytic_name, sampled_name in FINITE_M_TWIN_PAIRS:
        analytic_index = models.index(analytic_name)
        sampled_index = models.index(sampled_name)
        for metric in METRICS:
            delta = (
                boot[metric][:, sampled_index]
                - boot[metric][:, analytic_index]
            )
            delta_rows.append(
                {
                    "model_name": f"{sampled_name} minus {analytic_name}",
                    "reference_model": analytic_name,
                    "metric": metric,
                    "estimate_delta": float(
                        point.loc[sampled_name, metric]
                        - point.loc[analytic_name, metric]
                    ),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "ci_contains_zero": bool(
                        np.quantile(delta, 0.025) <= 0.0 <= np.quantile(delta, 0.975)
                    ),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "generator_batch_within_kernel",
                }
            )

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def headline_summary(macro: pd.DataFrame) -> pd.DataFrame:
    """Return the four M=64 rows used for the dissertation comparison."""
    selected = macro.loc[
        macro["model_name"].astype(str).isin(HEADLINE_MODEL_LABELS)
    ].copy()
    selected["model_name"] = selected["model_name"].astype(str)
    selected["display_name"] = selected["model_name"].map(HEADLINE_MODEL_LABELS)

    expected = list(HEADLINE_MODEL_LABELS)
    found = selected["model_name"].tolist()
    if set(found) != set(expected):
        raise RuntimeError(
            "Headline summary is missing sampled parity rows. "
            f"Expected {expected}, found {found}."
        )

    order = {name: index for index, name in enumerate(expected)}
    selected["_order"] = selected["model_name"].map(order)
    return selected.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def finite_m_diagnostics(deltas: pd.DataFrame) -> pd.DataFrame:
    labels = {
        f"{sampled} minus {analytic}"
        for analytic, sampled in FINITE_M_TWIN_PAIRS
    }
    return deltas.loc[deltas["model_name"].isin(labels)].reset_index(drop=True)


def _latex_table(headline: pd.DataFrame) -> str:
    table = headline[
        [
            "display_name",
            "rmse",
            "crps",
            "spread_skill_ratio",
            "coverage_90",
            "width_90",
        ]
    ].copy()

    table = table.rename(
        columns={
            "display_name": "Model",
            "rmse": r"RMSE $\downarrow$",
            "crps": r"CRPS $\downarrow$",
            "spread_skill_ratio": r"SSR $\to 1$",
            "coverage_90": r"Coverage$_{90}$ $\to 0.90$",
            "width_90": r"Width$_{90}$ $\downarrow$",
        }
    )

    return table.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.4f}",
        column_format="lccccc",
        caption=(
            "Gaussian-control performance using M=64 sampled predictive "
            "ensembles for all four sources, macro-averaged over the five "
            "fixed-hyperparameter GP kernels."
        ),
        label="tab:gaussian_controls",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "model_name",
        "eval_set",
        "kernel_name",
        "region",
        "task_index",
        "generator_batch_index",
        "task_fingerprint",
        "finite_ensemble",
        "num_eval_samples",
        *COMPONENT_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Input per-task CSV is missing columns: {missing}")

    check_pairing(df, MODEL_ORDER)

    by_kernel = summary_by_kernel(df, args.region)
    macro = macro_summary(by_kernel)
    pooled = pooled_summary(df, args.region)

    ci, deltas = paired_cluster_bootstrap(
        df=df,
        macro=macro,
        region=args.region,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    headline = headline_summary(macro)
    finite_m = finite_m_diagnostics(deltas)

    by_kernel.to_csv(output_dir / "summary_by_kernel.csv", index=False)
    macro.to_csv(output_dir / "summary_macro.csv", index=False)
    headline.to_csv(output_dir / "summary_macro_headline.csv", index=False)
    finite_m.to_csv(output_dir / "finite_m_twin_diagnostics.csv", index=False)
    pooled.to_csv(output_dir / "summary_pooled.csv", index=False)
    ci.to_csv(output_dir / "bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / "paired_cluster_bootstrap_deltas.csv", index=False)

    latex = _latex_table(headline)
    (output_dir / "gaussian_controls_table.tex").write_text(latex)

    metadata = {
        "input": str(Path(args.input).resolve()),
        "region": args.region,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_unit": "generator_batch_within_kernel",
        "main_aggregation": "arithmetic mean of five kernel-level metrics",
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(metadata, indent=2))

    print("\nHEADLINE M=64 MACRO-AVERAGED TABLE\n")
    print(
        headline[
            [
                "display_name",
                "rmse",
                "crps",
                "spread_skill_ratio",
                "coverage_90",
                "width_90",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\nPAIRED CLUSTER-BOOTSTRAP CRPS DIFFERENCES VS GAUSSIAN TNP (M=64)\n")
    print(
        deltas.loc[deltas["metric"] == "crps"].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nFINITE-M TWIN DIAGNOSTICS\n")
    print(
        finite_m.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nWrote analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
