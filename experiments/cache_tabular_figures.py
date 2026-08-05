from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--figure",
        choices=("context", "shape", "all"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _output_path(root: Path, name: str, smoke: bool) -> Path:
    path = root / name
    if path.suffix != ".pt":
        raise ValueError("Figure cache names must end in .pt.")
    if smoke:
        return path.with_name(path.stem + "_smoke.pt")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    result = []
    for row in frame.to_dict(orient="records"):
        converted = {}
        for key, value in row.items():
            if pd.isna(value):
                converted[key] = None
            elif isinstance(value, (int, float, str, bool)):
                converted[key] = value
            else:
                converted[key] = str(value)
        result.append(converted)
    return result


def _write_cache(path: Path, payload: Dict[str, Any], source_paths: list[Path]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    sidecar = {
        "schema_version": payload["schema_version"],
        "cache_path": str(path),
        "cache_sha256": _sha256(path),
        "source_paths": [str(value) for value in source_paths],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def build_context_cache(cfg: Dict[str, Any], root: Path, smoke: bool) -> None:
    figure_cfg = dict(cfg["context_dependence"])
    analysis_path = Path(figure_cfg["analysis_csv"])
    variant_path = Path(figure_cfg["variant_delta_csv"])
    rejection_path = Path(figure_cfg["rejection_json"])

    for path in (analysis_path, variant_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(analysis_path)
    variant = pd.read_csv(variant_path)

    if smoke:
        frame = frame.head(12).copy()
        variant = variant.head(8).copy()

    rejection = (
        json.loads(rejection_path.read_text())
        if rejection_path.is_file()
        else {}
    )

    payload = {
        "schema_version": "tabular_context_full_grid_cache_v2",
        "metadata": {
            "smoke": bool(smoke),
            "margin_metric": "context_resample_crps_minus_model_crps",
            "variant_metric": "gaussian_crps_minus_crps_variant_crps",
            "higher_is_better": True,
            "fixed_raw_targets_across_rungs": True,
            "complete_training_regime_grid": True,
            "rejection_diagnostics": rejection,
        },
        "records": _records(frame),
        "variant_vs_gaussian": _records(variant),
        "context_sizes": torch.tensor(
            sorted(frame["num_context"].unique()),
            dtype=torch.long,
        ),
        "training_regimes": sorted(
            frame["training_regime"].astype(str).unique().tolist()
        ),
        "architectures": sorted(
            frame["architecture"].astype(str).unique().tolist()
        ),
    }
    output = _output_path(root, str(figure_cfg["output_name"]), smoke)
    _write_cache(
        output,
        payload,
        [analysis_path, variant_path, rejection_path],
    )
    print("CACHE PASS [CONTEXT FULL GRID]")


def build_shape_cache(cfg: Dict[str, Any], root: Path, smoke: bool) -> None:
    figure_cfg = dict(cfg["shape_calibration"])
    paths = {
        key: Path(value)
        for key, value in figure_cfg.items()
        if key.endswith("_csv")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    contributions = pd.read_csv(paths["contribution_csv"])
    adjusted = pd.read_csv(paths["null_adjusted_csv"])
    histogram = pd.read_csv(paths["rank_histogram_csv"])
    calibration = pd.read_csv(paths["rank_calibration_csv"])

    if smoke:
        contributions = contributions.head(12)
        adjusted = adjusted.head(6)
        histogram = histogram.head(21)
        calibration = calibration.head(3)

    payload = {
        "schema_version": "tabular_shape_calibration_cache_v1",
        "metadata": {
            "smoke": bool(smoke),
            "primary_sample_count": int(contributions["sample_count"].max()),
            "finite_sample_check": int(contributions["sample_count"].min()),
            "interpretation_guard": (
                "Non-Gaussian shape contribution is assessed relative to the "
                "sampled Gaussian null; no exact posterior oracle is available."
            ),
        },
        "contributions": _records(contributions),
        "null_adjusted": _records(adjusted),
        "rank_histogram": _records(histogram),
        "rank_calibration": _records(calibration),
    }
    output = _output_path(root, str(figure_cfg["output_name"]), smoke)
    _write_cache(output, payload, list(paths.values()))
    print("CACHE PASS [SHAPE]")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output_dir = Path(cfg["output_dir"])

    if args.figure in {"context", "all"}:
        build_context_cache(cfg, output_dir, args.smoke)
    if args.figure in {"shape", "all"}:
        build_shape_cache(cfg, output_dir, args.smoke)

    print("ALL SELECTED TABULAR FIGURE CACHES COMPLETED.")


if __name__ == "__main__":
    main()
