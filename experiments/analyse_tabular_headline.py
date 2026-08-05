from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import OmegaConf

from evaluation.tabular_analysis_utils import (
    paired_bootstrap_metrics,
    summarise_by_model,
    validate_paired_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--consistency_input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260921)
    return parser.parse_args()


def _ordered(frame: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    index = {name: position for position, name in enumerate(order)}
    result = frame.copy()
    result["_order"] = result["model_name"].map(index)
    if result["_order"].isna().any():
        missing = result.loc[result["_order"].isna(), "model_name"].unique()
        raise ValueError(f"Unexpected models in summary: {missing}.")
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _load_locked_summary(cfg: Dict) -> pd.DataFrame:
    path = Path(cfg["locked_m64_metrics_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    locked = pd.read_csv(path)
    locked = locked.loc[
        (locked["region"] == "all")
        & (locked["context_bucket"] == "all")
    ].copy()
    name_map = dict(cfg["locked_m64_name_map"])
    locked["model_name"] = locked["model_name"].map(name_map)
    if locked["model_name"].isna().any():
        raise ValueError("Locked M=64 metrics contain unmapped source names.")
    locked = locked.rename(
        columns={
            "rmse_pooled": "rmse",
        }
    )
    return locked[
        [
            "model_name",
            "num_eval_samples",
            "rmse",
            "crps",
            "energy_score",
            "ensemble_spread",
            "spread_skill_ratio",
            "coverage_90",
            "width_90",
        ]
    ]


def _qualitative_checks(summary: pd.DataFrame, cfg: Dict) -> Dict[str, bool]:
    values = summary.set_index("model_name")
    dropout = str(cfg["roles"]["dropout"])
    stochln = str(cfg["roles"]["stochln"])
    gaussian = str(cfg["roles"]["gaussian"])
    resample = str(cfg["roles"]["context_resample"])
    learned = [dropout, stochln, gaussian]

    return {
        "dropout_crps_below_gaussian": bool(
            values.loc[dropout, "crps"] < values.loc[gaussian, "crps"]
        ),
        "stochln_crps_below_gaussian": bool(
            values.loc[stochln, "crps"] < values.loc[gaussian, "crps"]
        ),
        "dropout_crps_below_stochln": bool(
            values.loc[dropout, "crps"] < values.loc[stochln, "crps"]
        ),
        "dropout_rmse_below_gaussian": bool(
            values.loc[dropout, "rmse"] < values.loc[gaussian, "rmse"]
        ),
        "dropout_width_below_gaussian": bool(
            values.loc[dropout, "width_90"] < values.loc[gaussian, "width_90"]
        ),
        "all_learned_crps_below_context_resample": bool(
            all(
                values.loc[model, "crps"] < values.loc[resample, "crps"]
                for model in learned
            )
        ),
    }


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l c c c c c c@{}}",
        r"\toprule",
        (
            r"Model & RMSE $\downarrow$ & CRPS $\downarrow$ & "
            r"Energy $\downarrow$ & SSR & Coverage$_{90}$ & Width$_{90}$ \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['display_name']} & "
            f"{row['rmse']:.3f} & {row['crps']:.3f} & "
            f"{row['energy_score']:.3f} & "
            f"{row['spread_skill_ratio']:.3f} & "
            f"{row['coverage_90']:.3f} & {row['width_90']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(args.input)
    consistency = pd.read_csv(args.consistency_input)
    expected_models = [str(value) for value in cfg["headline_order"]]

    validate_paired_rows(primary, expected_models=expected_models)
    validate_paired_rows(consistency, expected_models=expected_models)

    primary_summary = _ordered(summarise_by_model(primary), expected_models)
    consistency_summary = _ordered(
        summarise_by_model(consistency), expected_models
    )
    locked_summary = _ordered(_load_locked_summary(cfg), expected_models)

    display_map = {
        str(entry["name"]): str(entry.get("display_name", entry["name"]))
        for entry in cfg["sources"]
    }
    for frame in (primary_summary, consistency_summary, locked_summary):
        frame["display_name"] = frame["model_name"].map(display_map)

    ci, deltas_gaussian = paired_bootstrap_metrics(
        primary,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        reference_model=str(cfg["roles"]["gaussian"]),
    )
    _, deltas_resample = paired_bootstrap_metrics(
        primary,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        reference_model=str(cfg["roles"]["context_resample"]),
    )
    deltas = pd.concat([deltas_gaussian, deltas_resample], ignore_index=True)

    checks = {
        "locked_m64": _qualitative_checks(locked_summary, cfg),
        "harmonised_m64_prefix": _qualitative_checks(consistency_summary, cfg),
        "harmonised_m256": _qualitative_checks(primary_summary, cfg),
    }
    all_pass = all(all(group.values()) for group in checks.values())
    invariance = {
        "all_pre_registered_qualitative_checks_pass": all_pass,
        "checks": checks,
        "interpretation": (
            "Qualitative ordering invariant across locked M=64, rerun M=64, "
            "and harmonised M=256."
            if all_pass
            else "At least one pre-registered qualitative ordering changed; "
            "report the inversion prominently before drafting."
        ),
    }

    primary_summary.to_csv(output_dir / "tabular_headline_summary_m256.csv", index=False)
    consistency_summary.to_csv(
        output_dir / "tabular_headline_summary_m64_consistency.csv", index=False
    )
    locked_summary.to_csv(
        output_dir / "tabular_headline_summary_locked_m64.csv", index=False
    )
    ci.to_csv(output_dir / "tabular_headline_bootstrap_metric_ci.csv", index=False)
    deltas.to_csv(output_dir / "tabular_headline_paired_deltas.csv", index=False)
    (output_dir / "harmonisation_invariance.json").write_text(
        json.dumps(invariance, indent=2)
    )
    (output_dir / "tabular_fixed_context_table.tex").write_text(
        _latex_table(primary_summary)
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_unit": "task",
                "primary_sample_count": int(
                    primary["num_eval_samples"].iloc[0]
                ),
                "consistency_sample_count": int(
                    consistency["num_eval_samples"].iloc[0]
                ),
            },
            indent=2,
        )
    )

    print("TABULAR FIXED-CONTEXT M=256 SUMMARY")
    print(primary_summary.to_string(index=False))
    print()
    print(json.dumps(invariance, indent=2))
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
