from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final optimisation-outcome table from exported trajectories."
    )
    parser.add_argument("--summary", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def _representation(name: str) -> str:
    return "raw $x$" if "raw x" in name.lower() else "Fourier"


def _alpha_text(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _schedule_text(value: str) -> str:
    replacements = {
        "variable_1_30": r"variable $1$--$30$",
        "variable_14_30": r"variable $14$--$30$",
        "variable_48_64": r"variable $48$--$64$",
        "fixed_64": r"fixed $64$",
        "fixed_96": r"fixed $96$",
        "fixed_128": r"fixed $128$",
        "64_to_56-64_to_48-64": r"$64\rightarrow[56,64]\rightarrow[48,64]$",
    }
    return replacements.get(str(value), str(value).replace("_", r"\_"))


def _model_text(name: str) -> str:
    replacements = {
        "Gaussian raw x": "Gaussian TNP",
        "Gaussian Fourier": "Gaussian TNP",
        "Dropout variable Nc1-30": "Dropout CRPS-TNP",
        "StochLN variable Nc1-30": "StochLN CRPS-TNP",
        "Dropout variable Nc14-30 alpha1": "Dropout CRPS-TNP",
        "StochLN variable Nc14-30 alpha1": "StochLN CRPS-TNP",
        "Dropout variable Nc14-30 alpha0.05": "Dropout CRPS-TNP",
        "Dropout variable Nc48-64": "Dropout CRPS-TNP",
        "Dropout fixed Nc64": "Dropout CRPS-TNP",
        "Dropout fixed Nc96": "Dropout CRPS-TNP",
        "Dropout fixed Nc128": "Dropout CRPS-TNP",
        "StochLN fixed Nc128": "StochLN CRPS-TNP",
        "Gaussian continuous curriculum": "Gaussian TNP",
        "Dropout continuous curriculum": "Dropout CRPS-TNP",
        "StochLN continuous curriculum": "StochLN CRPS-TNP",
        "Dropout pretrain-finetune alpha1": "Dropout CRPS-TNP",
        "Dropout pretrain-finetune alpha0.95": "Dropout CRPS-TNP",
    }
    return replacements.get(name, name)


def _training_path_text(value: str) -> str:
    replacements = {
        "from_scratch": "from scratch",
        "continuous_context_curriculum": "context curriculum",
        "deterministic_pretrain_then_crps_finetune": "pretrain--fine-tune",
    }
    return replacements.get(str(value), str(value).replace("_", r"\_"))


def _latex_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4.5pt}",
        r"    \begin{tabular}{@{}l l l l c c c@{}}",
        r"        \toprule",
        r"        Model & Input & Training path & Context schedule & $\alpha$ & Min. val. RMSE & Escape epoch \\",
        r"        \midrule",
    ]
    previous_panel = None
    for _, row in frame.iterrows():
        if previous_panel is not None and row["panel"] != previous_panel:
            lines.append(r"        \addlinespace")
        escape = "--" if pd.isna(row["rmse_escape_epoch"]) else str(int(row["rmse_escape_epoch"]))
        lines.append(
            "        "
            + " & ".join(
                [
                    str(row["model_display"]),
                    str(row["representation"]),
                    str(row["training_path_display"]),
                    str(row["context_schedule_display"]),
                    str(row["alpha_display"]),
                    f"{float(row['minimum_val_rmse']):.3f}",
                    escape,
                ]
            )
            + r" \\"
        )
        previous_panel = row["panel"]
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \label{table:sawtooth_optimisation_outcomes}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Trajectory config must resolve to a dictionary.")
    summary = pd.read_csv(args.summary)
    expected = [str(item["name"]) for item in config["runs"]]
    if set(summary["model_name"]) != set(expected):
        raise RuntimeError("Trajectory summary source set does not match the config.")
    order = {name: index for index, name in enumerate(expected)}
    summary = summary.assign(_order=summary["model_name"].map(order)).sort_values("_order")

    rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        name = str(row["model_name"])
        rows.append(
            {
                **row.to_dict(),
                "model_display": _model_text(name),
                "representation": _representation(name),
                "training_path_display": _training_path_text(str(row["training_path"])),
                "context_schedule_display": _schedule_text(str(row["context_schedule"])),
                "alpha_display": _alpha_text(row["alpha"]),
            }
        )
    output = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "sawtooth_optimisation_outcomes.csv", index=False)
    (output_dir / "sawtooth_optimisation_outcomes_table.tex").write_text(
        _latex_table(output)
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "summary": str(args.summary),
                "config": str(args.config),
                "row_order": expected,
                "escape_definition": config["escape"],
            },
            indent=2,
        )
    )
    print("\nSAWTOOTH OPTIMISATION OUTCOMES\n")
    print(
        output[
            [
                "model_name",
                "minimum_val_rmse",
                "rmse_escape_epoch",
                "final_val_rmse",
                "outcome",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote trajectory analysis to {output_dir}")


if __name__ == "__main__":
    main()
