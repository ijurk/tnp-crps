from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export auditable sawtooth validation trajectories from W&B."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output}. Use a new versioned "
            "directory or pass --overwrite deliberately."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _fetch_metric(run: Any, metric: str) -> pd.DataFrame:
    frame = run.history(
        samples=10_000,
        keys=[metric, "epoch"],
        x_axis="trainer/global_step",
        pandas=True,
    )
    required = {"trainer/global_step", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"Run {run.name!r} is missing {sorted(missing)} for metric {metric!r}. "
            f"Available columns: {list(frame.columns)}"
        )

    keep = ["trainer/global_step", metric]
    if "epoch" in frame.columns:
        keep.append("epoch")
    frame = frame[keep].copy()
    frame["trainer/global_step"] = pd.to_numeric(
        frame["trainer/global_step"], errors="coerce"
    )
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if "epoch" in frame.columns:
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")

    frame = (
        frame.dropna(subset=["trainer/global_step", metric])
        .sort_values("trainer/global_step", kind="stable")
        .drop_duplicates(subset=["trainer/global_step"], keep="last")
        .reset_index(drop=True)
    )
    frame["validation_index"] = np.arange(len(frame), dtype=int)

    if "epoch" in frame.columns and frame["epoch"].notna().any():
        frame["epoch_resolved"] = frame["epoch"].ffill().bfill()
    else:
        # All final runs validate once per epoch.
        frame["epoch_resolved"] = frame["validation_index"]

    return frame


def _merge_histories(rmse: pd.DataFrame, crps: Optional[pd.DataFrame]) -> pd.DataFrame:
    rmse = rmse.rename(columns={"val/rmse": "val_rmse"})
    columns = [
        "trainer/global_step",
        "epoch_resolved",
        "validation_index",
        "val_rmse",
    ]
    out = rmse[columns].copy()

    if crps is None:
        out["val_crps"] = np.nan
        return out

    crps = crps.rename(columns={"val/crps": "val_crps"})
    out = out.merge(
        crps[["trainer/global_step", "val_crps"]],
        on="trainer/global_step",
        how="left",
        validate="one_to_one",
    )
    return out


def _first_sustained_below(
    frame: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    window: int,
) -> Optional[int]:
    values = frame[column]
    below = values.lt(float(threshold)) & values.notna()
    sustained = below.astype(int).rolling(int(window)).sum().eq(int(window))
    hits = np.flatnonzero(sustained.to_numpy())
    if len(hits) == 0:
        return None
    start = int(hits[0] - int(window) + 1)
    return int(round(float(frame.loc[start, "epoch_resolved"])))


def _stage_for_epoch(
    epoch: float,
    stages: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    ordered = sorted(stages, key=lambda item: int(item["start_epoch"]))
    active: Mapping[str, Any] = ordered[0]
    for stage in ordered:
        if float(epoch) >= int(stage["start_epoch"]):
            active = stage
        else:
            break
    return {
        "stage_name": str(active["name"]),
        "stage_min_nc": int(active["min_nc"]),
        "stage_max_nc": int(active["max_nc"]),
    }


def _matches(run: Any, spec: Mapping[str, Any]) -> bool:
    name = str(run.name)
    exact = spec.get("run_name_exact")
    if exact is not None and name != str(exact):
        return False
    suffix = spec.get("run_name_suffix")
    if suffix is not None and not name.endswith(str(suffix)):
        return False
    for token in list(spec.get("run_name_contains", []) or []):
        if str(token) not in name:
            return False
    return True


def _resolve_run(all_runs: List[Any], spec: Mapping[str, Any]) -> Any:
    matches = [run for run in all_runs if _matches(run, spec)]
    if len(matches) != 1:
        print(f"\nRun matches for {spec['name']!r}:")
        for run in matches:
            print(f"  {run.name} ({run.id})")
        nearby = [
            run
            for run in all_runs
            if any(
                str(token) in str(run.name)
                for token in list(spec.get("run_name_contains", []) or [])
            )
        ]
        if nearby and not matches:
            print("Nearby project runs:")
            for run in nearby[:30]:
                print(f"  {run.name} ({run.id})")
        raise RuntimeError(
            f"Expected exactly one W&B run for {spec['name']!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Trajectory config must resolve to a dictionary.")

    output_dir = _prepare_output(
        args.output_dir or cfg["output_dir"], overwrite=args.overwrite
    )
    entity_env = str(cfg.get("entity_env", "WANDB_ENTITY"))
    entity = os.environ.get(entity_env)
    if not entity:
        raise RuntimeError(
            f"{entity_env} is not set. Export the W&B entity before running."
        )

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("The wandb package is required for trajectory export.") from exc

    project = str(cfg["project"])
    api = wandb.Api(timeout=int(cfg.get("api_timeout_seconds", 90)))
    all_runs = list(api.runs(f"{entity}/{project}"))
    print(f"Loaded {len(all_runs)} project runs from {entity}/{project}.")

    rmse_threshold = float(cfg["escape"]["rmse_threshold"])
    sustained_checks = int(cfg["escape"]["sustained_checks"])
    crps_threshold = cfg["escape"].get("crps_threshold")
    crps_threshold = None if crps_threshold is None else float(crps_threshold)

    history_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec_raw in cfg["runs"]:
        spec = dict(spec_raw)
        run = _resolve_run(all_runs, spec)
        print(f"Resolved {spec['name']} -> {run.name} ({run.id})")

        rmse = _fetch_metric(run, "val/rmse")
        crps: Optional[pd.DataFrame]
        try:
            crps = _fetch_metric(run, "val/crps")
        except RuntimeError:
            if bool(spec.get("require_val_crps", True)):
                raise
            crps = None

        history = _merge_histories(rmse, crps)
        expected_points = spec.get("expected_validation_points")
        if expected_points is not None and len(history) != int(expected_points):
            raise RuntimeError(
                f"Run {run.name!r} has {len(history)} validation points; "
                f"expected {expected_points}."
            )

        stages = list(spec.get("stages", []) or [])
        if not stages:
            stages = [
                {
                    "name": str(spec.get("context_schedule", "fixed")),
                    "start_epoch": 0,
                    "min_nc": int(spec.get("min_nc", -1)),
                    "max_nc": int(spec.get("max_nc", -1)),
                }
            ]

        stage_rows = [
            _stage_for_epoch(epoch, stages)
            for epoch in history["epoch_resolved"].tolist()
        ]
        stage_frame = pd.DataFrame(stage_rows)
        history = pd.concat([history.reset_index(drop=True), stage_frame], axis=1)
        history.insert(0, "model_name", str(spec["name"]))
        history.insert(1, "panel", str(spec["panel"]))
        history.insert(2, "wandb_run_name", str(run.name))
        history.insert(3, "wandb_run_id", str(run.id))
        history.insert(4, "alpha", spec.get("alpha"))
        history.insert(5, "training_path", str(spec.get("training_path", "from_scratch")))
        history.insert(6, "validation_support", str(spec.get("validation_support", "unspecified")))
        history_rows.append(history)

        rmse_best_index = history["val_rmse"].idxmin()
        valid_crps = history["val_crps"].dropna()
        crps_best_index = valid_crps.idxmin() if not valid_crps.empty else None
        rmse_escape = _first_sustained_below(
            history,
            column="val_rmse",
            threshold=rmse_threshold,
            window=sustained_checks,
        )
        crps_escape = None

        alpha_value = spec.get("alpha")

        is_fair_crps_run = (
            alpha_value is not None
            and math.isclose(
                float(alpha_value),
                1.0,
            )
        )

        if (
            crps_threshold is not None
            and is_fair_crps_run
            and history["val_crps"].notna().any()
        ):
            crps_escape = _first_sustained_below(
                history,
                column="val_crps",
                threshold=crps_threshold,
                window=sustained_checks,
            )

        summary_rows.append(
            {
                "model_name": str(spec["name"]),
                "panel": str(spec["panel"]),
                "wandb_run_name": str(run.name),
                "wandb_run_id": str(run.id),
                "checkpoint_path": spec.get("checkpoint_path"),
                "training_path": str(spec.get("training_path", "from_scratch")),
                "context_schedule": str(spec.get("context_schedule", "unspecified")),
                "validation_support": str(spec.get("validation_support", "unspecified")),
                "alpha": spec.get("alpha"),
                "num_validation_points": len(history),
                "rmse_escape_epoch": rmse_escape,
                "minimum_val_rmse": float(history.loc[rmse_best_index, "val_rmse"]),
                "epoch_of_minimum_rmse": int(
                    round(float(history.loc[rmse_best_index, "epoch_resolved"]))
                ),
                "final_val_rmse": float(history["val_rmse"].iloc[-1]),
                "crps_escape_epoch_alpha1_only": crps_escape,
                "minimum_val_crps": (
                    float(history.loc[crps_best_index, "val_crps"])
                    if crps_best_index is not None
                    else float("nan")
                ),
                "epoch_of_minimum_crps": (
                    int(round(float(history.loc[crps_best_index, "epoch_resolved"])))
                    if crps_best_index is not None
                    else None
                ),
                "final_val_crps": float(history["val_crps"].iloc[-1]),
                "outcome": "escaped" if rmse_escape is not None else "plateau",
            }
        )

    history_output = pd.concat(history_rows, ignore_index=True)
    summary_output = pd.DataFrame(summary_rows)
    history_path = output_dir / "sawtooth_trajectory_history.csv"
    summary_path = output_dir / "sawtooth_trajectory_summary.csv"
    resolved_path = output_dir / "trajectory_export_resolved.json"
    history_output.to_csv(history_path, index=False)
    summary_output.to_csv(summary_path, index=False)
    resolved_path.write_text(
        json.dumps(
            {
                "config": cfg,
                "entity": entity,
                "project": project,
                "resolved_runs": summary_rows,
            },
            indent=2,
        )
    )

    print("\nSAWTOOTH TRAJECTORY SUMMARY\n")
    print(
        summary_output[
            [
                "model_name",
                "panel",
                "num_validation_points",
                "rmse_escape_epoch",
                "minimum_val_rmse",
                "final_val_rmse",
                "outcome",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {history_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {resolved_path}")


if __name__ == "__main__":
    main()
