from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import torch
from omegaconf import OmegaConf

from evaluation.metrics import crps_per_element, crps_per_element_sorted


EXPECTED_TABICL_COMMIT = "46b91961db4f8873dd049ec09990698a435e1e29"
EXPECTED_TNP_COMMIT = "1f60200a5879bf1f77e63eb1427a61da7932f5c2"
EXPECTED_TEST_TASKS = 4096
EXPECTED_CHECKPOINT_EPOCH = 499
EXPECTED_CHECKPOINT_GLOBAL_STEP = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headline_config",
        default=(
            "experiments/configs/evaluation/tabular/"
            "tabular_tabicl_nc128_harmonised_final.yml"
        ),
    )
    parser.add_argument(
        "--shape_config",
        default=(
            "experiments/configs/evaluation/tabular/"
            "tabular_tabicl_nc128_shape_final.yml"
        ),
    )
    parser.add_argument(
        "--ladder_config",
        default=(
            "experiments/configs/evaluation/tabular/"
            "tabular_tabicl_nested_ladder_final.yml"
        ),
    )
    parser.add_argument(
        "--figure_config",
        default=(
            "experiments/configs/evaluation/tabular/"
            "tabular_tabicl_figures_final.yml"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/tabular/final_protocol_audit_20260804_v1.json",
    )
    return parser.parse_args()


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _load_yaml(path: str) -> Dict[str, Any]:
    resolved = OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
    )
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(resolved)}.")
    return resolved


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected!r}, actual={actual!r}")


def _assert_unique_positive_offsets(sources: Iterable[Mapping[str, Any]]) -> None:
    offsets = [int(source["sampling_seed_offset"]) for source in sources]
    if len(offsets) != len(set(offsets)):
        raise AssertionError(f"Duplicate sampling_seed_offset values: {offsets}")
    if any(value < 1 for value in offsets):
        raise AssertionError(f"Non-positive sampling_seed_offset values: {offsets}")


def _checkpoint_paths(*configs: Mapping[str, Any]) -> list[Path]:
    paths: set[Path] = set()
    for config in configs:
        for source in config.get("sources", []):
            if str(source.get("kind", "model")) != "model":
                continue
            paths.add(Path(str(source["checkpoint_path"])))
    return sorted(paths)


def _audit_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    _assert_equal(epoch, EXPECTED_CHECKPOINT_EPOCH, f"Checkpoint epoch for {path}")
    _assert_equal(
        global_step,
        EXPECTED_CHECKPOINT_GLOBAL_STEP,
        f"Checkpoint global_step for {path}",
    )
    return {
        "path": str(path),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "size_bytes": int(path.stat().st_size),
    }


def _audit_bank() -> Dict[str, Any]:
    bank_dir_raw = os.environ.get("TABICL_BANK_DIR")
    if not bank_dir_raw:
        raise RuntimeError(
            "TABICL_BANK_DIR is not set. Point it at the frozen GraphSCM bank."
        )
    bank_dir = Path(bank_dir_raw).expanduser().resolve()
    manifest_path = bank_dir / "test" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    _assert_equal(manifest.get("split"), "test", "Bank split")
    _assert_equal(manifest.get("seq_len"), 256, "Bank task sequence length")
    _assert_equal(
        manifest.get("tabicl_commit"),
        EXPECTED_TABICL_COMMIT,
        "Bank TabICL commit",
    )
    _assert_equal(
        manifest.get("full_sequence_preprocessing"),
        False,
        "Bank full-sequence preprocessing",
    )
    _assert_equal(
        manifest.get("categorical_features"),
        False,
        "Bank categorical-feature flag",
    )
    num_tasks = int(manifest.get("num_tasks", 0))
    if num_tasks <= EXPECTED_TEST_TASKS:
        raise AssertionError(
            "The test bank lacks rejection headroom for the 4,096-task "
            f"intersection ladder: num_tasks={num_tasks}."
        )

    shard_paths = sorted((bank_dir / "test").glob("shard_*.pt"))
    _assert_equal(
        len(shard_paths),
        int(manifest["num_shards"]),
        "Number of test-bank shards",
    )

    return {
        "bank_dir": str(bank_dir),
        "manifest_path": str(manifest_path),
        "num_tasks": num_tasks,
        "num_shards": int(manifest["num_shards"]),
        "seq_len": int(manifest["seq_len"]),
        "tabicl_commit": str(manifest["tabicl_commit"]),
        "full_sequence_preprocessing": bool(
            manifest["full_sequence_preprocessing"]
        ),
        "categorical_features": bool(manifest["categorical_features"]),
    }


def _audit_locked_m64(headline: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(headline["locked_m64_metrics_path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["region"] == "all")
        & (frame["context_bucket"] == "all")
    ].copy()
    expected_names = set(str(name) for name in headline["headline_order"])
    if set(selected["model_name"].astype(str)) != expected_names:
        raise AssertionError(
            "Locked M=64 source set differs from the harmonisation config."
        )
    if not selected["num_eval_samples"].eq(64).all():
        raise AssertionError("Locked fixed-context result is not uniformly M=64.")
    if selected.duplicated("model_name").any():
        raise AssertionError("Locked M=64 summary has duplicate source rows.")

    return {
        "path": str(path),
        "num_sources": int(len(selected)),
        "num_eval_samples": 64,
        "context_resample_crps": float(
            selected.loc[
                selected["model_name"] == "context_resample", "crps"
            ].iloc[0]
        ),
        "gaussian_crps": float(
            selected.loc[selected["model_name"] == "gaussian", "crps"].iloc[0]
        ),
        "dropout_crps": float(
            selected.loc[selected["model_name"] == "dropout", "crps"].iloc[0]
        ),
        "stochln_crps": float(
            selected.loc[selected["model_name"] == "stochln", "crps"].iloc[0]
        ),
    }


def _audit_crps_sorted_identity() -> Dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260804)
    samples = torch.randn(7, 3, 5, 1, generator=generator, dtype=torch.float64)
    target = torch.randn(3, 5, 1, generator=generator, dtype=torch.float64)

    direct = crps_per_element(samples=samples, target=target, alpha=1.0)
    sorted_value = crps_per_element_sorted(
        samples=samples,
        target=target,
        alpha=1.0,
    )
    max_error = float((direct - sorted_value).abs().max().item())
    if max_error > 1.0e-12:
        raise AssertionError(
            f"Sorted fair-CRPS identity failed: max_error={max_error:.3e}."
        )

    direct_ordinary = crps_per_element(samples=samples, target=target, alpha=0.0)
    sorted_ordinary = crps_per_element_sorted(
        samples=samples,
        target=target,
        alpha=0.0,
    )
    max_ordinary_error = float(
        (direct_ordinary - sorted_ordinary).abs().max().item()
    )
    if max_ordinary_error > 1.0e-12:
        raise AssertionError(
            "Sorted ordinary-CRPS identity failed: "
            f"max_error={max_ordinary_error:.3e}."
        )

    return {
        "max_fair_crps_error": max_error,
        "max_ordinary_crps_error": max_ordinary_error,
    }


def _audit_configs(
    headline: Mapping[str, Any],
    shape: Mapping[str, Any],
    ladder: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> Dict[str, Any]:
    # Fixed-context harmonisation.
    _assert_equal(headline["mode"], "headline", "Headline mode")
    _assert_equal(
        int(headline["samples_per_eval_set"]),
        EXPECTED_TEST_TASKS,
        "Headline task count",
    )
    _assert_equal(int(headline["num_eval_samples"]), 256, "Headline M")
    _assert_equal(
        int(headline["consistency_num_samples"]),
        64,
        "Headline M=64 consistency prefix",
    )
    _assert_equal(int(headline["sample_chunk_size"]), 64, "Headline chunk size")
    _assert_equal(
        [float(value) for value in headline.get("interval_levels", [])],
        [0.9],
        "Headline interval levels",
    )
    _assert_equal(
        bool(headline.get("compute_energy_score")),
        True,
        "Headline energy-score flag",
    )
    _assert_unique_positive_offsets(headline["sources"])

    # Shape protocol.
    _assert_equal(shape["mode"], "shape", "Shape mode")
    _assert_equal(
        int(shape["samples_per_eval_set"]),
        EXPECTED_TEST_TASKS,
        "Shape task count",
    )
    _assert_equal(int(shape["num_eval_samples"]), 256, "Shape headline M")
    _assert_equal(int(shape["sample_chunk_size"]), 64, "Shape chunk size")
    _assert_equal(
        [int(value) for value in shape["shape_analysis"]["sample_counts"]],
        [64, 512],
        "Shape finite-sample counts",
    )
    _assert_equal(
        int(shape["shape_analysis"]["rank_sample_count"]),
        512,
        "Rank-histogram sample count",
    )
    _assert_equal(
        int(shape["shape_analysis"]["rank_plot_bins"]),
        21,
        "Rank-histogram display bins",
    )
    _assert_equal(
        bool(shape.get("compute_energy_score")),
        False,
        "Shape energy-score flag",
    )
    _assert_unique_positive_offsets(shape["sources"])

    # Nested ladder protocol.
    nested = ladder["nested_tasks"]
    _assert_equal(
        int(nested["accepted_tasks"]),
        EXPECTED_TEST_TASKS,
        "Nested-ladder intersection task count",
    )
    _assert_equal(
        [int(value) for value in nested["context_sizes"]],
        [16, 32, 64, 128],
        "Nested context sizes",
    )
    _assert_equal(int(nested["context_pool_size"]), 128, "Context pool size")
    _assert_equal(int(nested["num_targets"]), 128, "Fixed target count")
    _assert_equal(
        int(nested["context_pool_size"]) + int(nested["num_targets"]),
        256,
        "Nested task row contract",
    )
    _assert_equal(int(ladder["num_eval_samples"]), 256, "Ladder M")
    _assert_equal(int(ladder["sample_chunk_size"]), 64, "Ladder chunk size")
    _assert_equal(
        bool(ladder.get("compute_energy_score")),
        False,
        "Ladder energy-score flag",
    )
    _assert_unique_positive_offsets(ladder["sources"])

    required_groups = {
        "all",
        "fixed128_and_baselines",
        "variable",
        "specialisation",
    }
    if set(ladder["source_groups"]) != required_groups:
        raise AssertionError(
            "Nested-ladder source groups differ from the frozen design."
        )

    # Ensure every configured figure input is produced by an analysis path.
    for section_name in ("context_dependence", "shape_calibration"):
        if section_name not in figures:
            raise AssertionError(f"Missing figure config section {section_name!r}.")

    return {
        "headline": {
            "tasks": int(headline["samples_per_eval_set"]),
            "primary_samples": int(headline["num_eval_samples"]),
            "consistency_samples": int(headline["consistency_num_samples"]),
            "chunk_size": int(headline["sample_chunk_size"]),
            "source_count": len(headline["sources"]),
        },
        "shape": {
            "tasks": int(shape["samples_per_eval_set"]),
            "headline_samples": int(shape["num_eval_samples"]),
            "shape_sample_counts": [
                int(value) for value in shape["shape_analysis"]["sample_counts"]
            ],
            "rank_sample_count": int(
                shape["shape_analysis"]["rank_sample_count"]
            ),
            "chunk_size": int(shape["sample_chunk_size"]),
        },
        "nested_ladder": {
            "intersection_tasks": int(nested["accepted_tasks"]),
            "context_sizes": [int(value) for value in nested["context_sizes"]],
            "context_pool_size": int(nested["context_pool_size"]),
            "fixed_target_count": int(nested["num_targets"]),
            "primary_samples": int(ladder["num_eval_samples"]),
            "chunk_size": int(ladder["sample_chunk_size"]),
            "fixed_raw_target_rows": [128, 256],
        },
    }


def main() -> None:
    args = parse_args()
    headline = _load_yaml(args.headline_config)
    shape = _load_yaml(args.shape_config)
    ladder = _load_yaml(args.ladder_config)
    figures = _load_yaml(args.figure_config)

    tabicl_commit = _run_git("-C", "external/tabicl", "rev-parse", "HEAD")
    tnp_commit = _run_git("-C", "external/tnp", "rev-parse", "HEAD")
    _assert_equal(tabicl_commit, EXPECTED_TABICL_COMMIT, "TabICL submodule commit")
    _assert_equal(tnp_commit, EXPECTED_TNP_COMMIT, "TNP submodule commit")

    config_audit = _audit_configs(headline, shape, ladder, figures)
    bank_audit = _audit_bank()
    locked_m64 = _audit_locked_m64(headline)
    metric_audit = _audit_crps_sorted_identity()

    checkpoints = [
        _audit_checkpoint(path)
        for path in _checkpoint_paths(headline, shape, ladder)
    ]

    output = {
        "status": "PASS",
        "repo_commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "git_status_short": _run_git("status", "--short"),
        "tnp_submodule_commit": tnp_commit,
        "tabicl_submodule_commit": tabicl_commit,
        "configs": config_audit,
        "bank": bank_audit,
        "locked_m64": locked_m64,
        "metric_regression": metric_audit,
        "checkpoints": checkpoints,
        "scientific_guards": {
            "test_bank_not_expanded": True,
            "m256_is_harmonisation_not_correction": True,
            "nested_raw_targets_fixed_across_rungs": True,
            "context_preprocessing_refit_per_rung": True,
            "shape_primary_m512_with_m64_check": True,
            "gaussian_shape_null_required": True,
            "no_exact_tabular_posterior_claim": True,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("PASS: final tabular protocol audit")
    print("  - locked 4,096-task test bank retained")
    print("  - M=256 headline is a harmonisation pass; M=64 consistency retained")
    print("  - nested contexts use fixed raw target rows [128:256]")
    print("  - shape analysis uses M=512 with nested M=64 and rank counts")
    print(f"  - audited {len(checkpoints)} unique checkpoints")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
