from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import OmegaConf

from evaluation.metrics import per_task_shape_rows_tabular
from evaluation.tabular_final_utils import (
    batch_task_fingerprints,
    build_generator,
    load_sources,
    move_batch_to_device,
    per_task_metric_rows_efficient,
    sample_loaded_source,
    stable_sampling_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--samples_per_eval_set", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--sample_chunk_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"},
        )
        return result.stdout.strip()
    except OSError:
        return ""


def _write_partial(
    *,
    output_dir: Path,
    primary_rows: List[Dict[str, Any]],
    consistency_rows: List[Dict[str, Any]],
) -> None:
    if primary_rows:
        pd.DataFrame(primary_rows).to_csv(
            output_dir / "per_task_metrics_partial.csv",
            index=False,
        )
    if consistency_rows:
        pd.DataFrame(consistency_rows).to_csv(
            output_dir / "per_task_metrics_m64_consistency_partial.csv",
            index=False,
        )


def _validate_pairing(
    *,
    rows: pd.DataFrame,
    expected_sources: List[str],
    expected_tasks: int,
) -> None:
    if len(rows) != expected_tasks * len(expected_sources):
        raise RuntimeError(
            "Unexpected row count: "
            f"{len(rows)} != {expected_tasks} x {len(expected_sources)}."
        )
    if set(rows["model_name"]) != set(expected_sources):
        raise RuntimeError("Source names do not match the configuration.")
    source_counts = rows.groupby("task_index")["model_name"].nunique()
    if not source_counts.eq(len(expected_sources)).all():
        raise RuntimeError("Not every task has every configured source.")
    fingerprints = rows.groupby("task_index")["task_fingerprint"].nunique()
    if not fingerprints.eq(1).all():
        raise RuntimeError("Task fingerprints differ across predictive sources.")
    duplicates = rows.duplicated(["model_name", "task_index"])
    if duplicates.any():
        raise RuntimeError("Duplicate source/task rows were produced.")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    mode = str(cfg.get("mode", "headline"))
    if mode not in {"headline", "shape"}:
        raise ValueError(f"Unknown mode={mode!r}.")

    output_dir = Path(args.output_dir or cfg["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    samples_per_eval_set = int(
        args.samples_per_eval_set
        if args.samples_per_eval_set is not None
        else cfg["samples_per_eval_set"]
    )
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else cfg["eval_batch_size"]
    )
    primary_samples = int(
        args.num_eval_samples
        if args.num_eval_samples is not None
        else cfg["num_eval_samples"]
    )
    sample_chunk_size = int(
        args.sample_chunk_size
        if args.sample_chunk_size is not None
        else cfg["sample_chunk_size"]
    )
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = cfg.get("max_batches", None)

    device_name = str(args.device or cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)

    metric_alpha = float(cfg.get("metric_alpha", 1.0))
    sampling_seed = int(cfg["sampling_seed"])
    consistency_samples = int(cfg.get("consistency_num_samples", 64))
    interval_levels = tuple(
        float(value) for value in cfg.get("interval_levels", [0.9])
    )
    compute_energy_score = bool(cfg.get("compute_energy_score", True))

    shape_cfg = dict(cfg.get("shape_analysis", {}) or {})
    if mode == "shape":
        shape_counts = [int(value) for value in shape_cfg["sample_counts"]]
        rank_sample_count = int(shape_cfg["rank_sample_count"])
        generated_samples = max(
            primary_samples,
            rank_sample_count,
            *shape_counts,
        )
    else:
        shape_counts = []
        rank_sample_count = 0
        generated_samples = max(primary_samples, consistency_samples)

    resolved = dict(cfg)
    resolved["samples_per_eval_set"] = samples_per_eval_set
    resolved["eval_batch_size"] = eval_batch_size
    resolved["num_eval_samples"] = primary_samples
    resolved["sample_chunk_size"] = sample_chunk_size
    resolved["runtime_metadata"] = {
        "repo_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "git_status_short": _git_value("status", "--short"),
        "tnp_submodule_commit": _git_value(
            "-C", "external/tnp", "rev-parse", "HEAD"
        ),
        "tabicl_submodule_commit": _git_value(
            "-C", "external/tabicl", "rev-parse", "HEAD"
        ),
    }
    (output_dir / "eval_config_resolved.json").write_text(
        json.dumps(resolved, indent=2)
    )

    generator = build_generator(
        base_generator_config=str(cfg["base_generator_config"]),
        overrides=list(cfg.get("generator_overrides", []) or []),
        samples_per_epoch=samples_per_eval_set,
        batch_size=eval_batch_size,
    )
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    sources = load_sources(
        entries=cfg["sources"],
        base_generator_config=str(cfg["base_generator_config"]),
        device=device,
    )
    if mode == "shape" and any(source.model is None for source in sources):
        raise ValueError("Shape analysis accepts learned sources only.")

    primary_rows: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []
    runtime: Dict[str, Dict[str, float]] = {
        source.name: {
            "elapsed_seconds": 0.0,
            "peak_memory_bytes": 0.0,
            "num_tasks": 0.0,
        }
        for source in sources
    }

    task_index_start = 0
    processed_batches = 0

    for batch_index, batch_cpu in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        fingerprints = batch_task_fingerprints(batch_cpu)
        batch = move_batch_to_device(batch_cpu, device)
        batch_size = int(batch.yt.shape[0])

        for source in sources:
            source_seed = stable_sampling_seed(
                base_seed=sampling_seed,
                source_offset=source.sampling_seed_offset,
                batch_index=batch_index,
            )
            pl.seed_everything(source_seed)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            samples, gaussian_loc, gaussian_scale = sample_loaded_source(
                source=source,
                batch=batch,
                num_samples=generated_samples,
                chunk_size=sample_chunk_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            runtime[source.name]["elapsed_seconds"] += elapsed
            runtime[source.name]["num_tasks"] += batch_size
            if device.type == "cuda":
                peak = float(torch.cuda.max_memory_allocated(device))
                runtime[source.name]["peak_memory_bytes"] = max(
                    runtime[source.name]["peak_memory_bytes"], peak
                )

            rows = per_task_metric_rows_efficient(
                samples=samples[:primary_samples],
                target=batch.yt,
                num_context=int(batch.xc.shape[1]),
                model_name=source.name,
                display_name=source.display_name,
                checkpoint_path=source.checkpoint_path,
                eval_set=str(cfg["eval_set_name"]),
                task_index_start=task_index_start,
                alpha=metric_alpha,
                interval_levels=interval_levels,
                compute_energy_score=compute_energy_score,
            )

            if mode == "shape":
                shape_rows = per_task_shape_rows_tabular(
                    samples=samples,
                    target=batch.yt,
                    num_context=int(batch.xc.shape[1]),
                    model_name=source.name,
                    checkpoint_path=source.checkpoint_path,
                    eval_set=str(cfg["eval_set_name"]),
                    task_index_start=task_index_start,
                    sample_counts=shape_counts,
                    rank_sample_count=rank_sample_count,
                    gaussian_loc=gaussian_loc,
                    gaussian_scale=gaussian_scale,
                )
                if len(shape_rows) != len(rows):
                    raise RuntimeError("Metric and shape row counts differ.")
                for row, shape_row in zip(rows, shape_rows):
                    for key, value in shape_row.items():
                        if key not in row:
                            row[key] = value

            for local_index, row in enumerate(rows):
                row["task_fingerprint"] = fingerprints[local_index]
                row["batch_index"] = batch_index
                row["training_alpha"] = source.training_alpha
                row["metric_alpha"] = metric_alpha
                row["sampling_seed"] = source_seed
            primary_rows.extend(rows)

            if mode == "headline":
                consistency = per_task_metric_rows_efficient(
                    samples=samples[:consistency_samples],
                    target=batch.yt,
                    num_context=int(batch.xc.shape[1]),
                    model_name=source.name,
                    display_name=source.display_name,
                    checkpoint_path=source.checkpoint_path,
                    eval_set=str(cfg["eval_set_name"]),
                    task_index_start=task_index_start,
                    alpha=metric_alpha,
                    interval_levels=interval_levels,
                    compute_energy_score=compute_energy_score,
                )
                for local_index, row in enumerate(consistency):
                    row["task_fingerprint"] = fingerprints[local_index]
                    row["batch_index"] = batch_index
                    row["training_alpha"] = source.training_alpha
                    row["metric_alpha"] = metric_alpha
                    row["sampling_seed"] = source_seed
                consistency_rows.extend(consistency)

            del samples

        task_index_start += batch_size
        processed_batches += 1

        if batch_index % 25 == 0:
            print(
                f"processed batch {batch_index + 1}/{generator.num_batches}; "
                f"tasks={task_index_start}",
                flush=True,
            )
            _write_partial(
                output_dir=output_dir,
                primary_rows=primary_rows,
                consistency_rows=consistency_rows,
            )

    expected_tasks = task_index_start
    if max_batches is None and expected_tasks != samples_per_eval_set:
        raise RuntimeError(
            f"Expected {samples_per_eval_set} tasks, processed {expected_tasks}."
        )

    primary_df = pd.DataFrame(primary_rows)
    _validate_pairing(
        rows=primary_df,
        expected_sources=[source.name for source in sources],
        expected_tasks=expected_tasks,
    )
    primary_df.to_csv(output_dir / "per_task_metrics.csv", index=False)

    if consistency_rows:
        consistency_df = pd.DataFrame(consistency_rows)
        _validate_pairing(
            rows=consistency_df,
            expected_sources=[source.name for source in sources],
            expected_tasks=expected_tasks,
        )
        consistency_df.to_csv(
            output_dir / "per_task_metrics_m64_consistency.csv",
            index=False,
        )

    fingerprint_df = (
        primary_df[["task_index", "batch_index", "task_fingerprint"]]
        .drop_duplicates()
        .sort_values("task_index")
    )
    fingerprint_df.to_csv(output_dir / "task_fingerprints.csv", index=False)

    runtime_rows = []
    for source in sources:
        record = runtime[source.name]
        num_tasks = int(record["num_tasks"])
        runtime_rows.append(
            {
                "model_name": source.name,
                "display_name": source.display_name,
                "num_tasks": num_tasks,
                "num_batches": processed_batches,
                "elapsed_seconds": record["elapsed_seconds"],
                "runtime_seconds_per_task": (
                    record["elapsed_seconds"] / num_tasks if num_tasks else float("nan")
                ),
                "peak_memory_bytes": int(record["peak_memory_bytes"]),
                "device": str(device),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else ""
                ),
            }
        )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "runtime_by_source.csv",
        index=False,
    )

    print(
        f"Pairing PASS: {expected_tasks} tasks x {len(sources)} sources.",
        flush=True,
    )
    print(f"Wrote {output_dir / 'per_task_metrics.csv'}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
