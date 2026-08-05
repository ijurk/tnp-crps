from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import OmegaConf

from evaluation.tabular_final_utils import (
    build_generator,
    load_sources,
    move_batch_to_device,
    per_task_metric_rows_efficient,
    prepare_nested_rung,
    sample_loaded_source,
    stable_sampling_seed,
    stack_single_task_batches,
    tensor_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--task_cache", default=None)
    parser.add_argument("--source_group", default="all")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--sample_chunk_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_group_pairing(
    *,
    rows: pd.DataFrame,
    context_sizes: List[int],
    expected_by_rung: Dict[int, List[str]],
    expected_tasks: int,
) -> None:
    for num_context in context_sizes:
        subset = rows.loc[rows["num_context"] == num_context]
        expected_sources = expected_by_rung[num_context]
        expected_rows = expected_tasks * len(expected_sources)
        if len(subset) != expected_rows:
            raise RuntimeError(
                f"Nc={num_context}: {len(subset)} rows != {expected_rows}."
            )
        if set(subset["model_name"]) != set(expected_sources):
            raise RuntimeError(
                f"Nc={num_context}: source set does not match the configuration."
            )
        counts = subset.groupby("task_index")["model_name"].nunique()
        if not counts.eq(len(expected_sources)).all():
            raise RuntimeError(
                f"Nc={num_context}: some tasks lack configured sources."
            )
        targets = subset.groupby("task_index")["target_fingerprint"].nunique()
        if not targets.eq(1).all():
            raise RuntimeError(
                f"Nc={num_context}: target fingerprints differ across sources."
            )
        duplicates = subset.duplicated(
            ["model_name", "num_context", "task_index"]
        )
        if duplicates.any():
            raise RuntimeError(f"Nc={num_context}: duplicate rows detected.")

    target_across_rungs = (
        rows[["task_index", "num_context", "target_fingerprint"]]
        .drop_duplicates()
        .groupby("task_index")["target_fingerprint"]
        .nunique()
    )
    if not target_across_rungs.eq(1).all():
        raise RuntimeError(
            "Fixed raw target fingerprints do not match across context rungs."
        )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    output_root = Path(args.output_dir or cfg["output_dir"])
    group_name = str(args.source_group)
    group_dir = output_root / group_name
    if group_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {group_dir}.")
    group_dir.mkdir(parents=True)

    task_cache_path = Path(args.task_cache or cfg["nested_tasks"]["cache_path"])
    if not task_cache_path.is_file():
        raise FileNotFoundError(task_cache_path)
    task_cache = torch.load(
        task_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    if task_cache.get("schema_version") != "tabular_nested_ladder_tasks_v1":
        raise RuntimeError("Unexpected nested-task cache schema.")

    context_sizes = [int(value) for value in task_cache["context_sizes"]]
    configured_context_sizes = [
        int(value) for value in cfg["nested_tasks"]["context_sizes"]
    ]
    if context_sizes != configured_context_sizes:
        raise RuntimeError(
            "Task-cache context sizes differ from the evaluation config."
        )

    available_tasks = int(task_cache["accepted_tasks"])
    num_tasks = min(
        available_tasks,
        int(args.max_tasks) if args.max_tasks is not None else available_tasks,
    )
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else cfg["eval_batch_size"]
    )
    num_eval_samples = int(
        args.num_eval_samples
        if args.num_eval_samples is not None
        else cfg["num_eval_samples"]
    )
    sample_chunk_size = int(
        args.sample_chunk_size
        if args.sample_chunk_size is not None
        else cfg["sample_chunk_size"]
    )
    metric_alpha = float(cfg.get("metric_alpha", 1.0))
    sampling_seed = int(cfg["sampling_seed"])
    interval_levels = tuple(
        float(value) for value in cfg.get("interval_levels", [0.9])
    )
    compute_energy_score = bool(cfg.get("compute_energy_score", False))

    device_name = str(args.device or cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)

    groups = dict(cfg["source_groups"])
    if group_name not in groups:
        raise KeyError(
            f"Unknown source_group={group_name!r}; available={sorted(groups)}."
        )
    selected_names = set(str(value) for value in groups[group_name])
    source_entries = [
        entry for entry in cfg["sources"] if str(entry["name"]) in selected_names
    ]
    if {str(entry["name"]) for entry in source_entries} != selected_names:
        raise RuntimeError("source_group references an unknown source.")

    generator = build_generator(
        base_generator_config=str(cfg["base_generator_config"]),
        overrides=list(cfg["nested_tasks"].get("generator_overrides", []) or []),
        samples_per_epoch=num_tasks,
        batch_size=eval_batch_size,
    )
    sources = load_sources(
        entries=source_entries,
        base_generator_config=str(cfg["base_generator_config"]),
        device=device,
    )

    resolved = dict(cfg)
    resolved["source_group"] = group_name
    resolved["selected_sources"] = [source.name for source in sources]
    resolved["task_cache"] = str(task_cache_path)
    resolved["task_cache_sha256"] = _sha256(task_cache_path)
    resolved["num_tasks"] = num_tasks
    resolved["eval_batch_size"] = eval_batch_size
    resolved["num_eval_samples"] = num_eval_samples
    resolved["sample_chunk_size"] = sample_chunk_size
    (group_dir / "eval_config_resolved.json").write_text(
        json.dumps(resolved, indent=2)
    )

    rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    runtime: Dict[str, Dict[str, float]] = {}

    expected_by_rung: Dict[int, List[str]] = {}

    for rung_index, num_context in enumerate(context_sizes):
        rung_sources = [
            source
            for source in sources
            if source.eval_context_sizes is None
            or num_context in source.eval_context_sizes
        ]
        expected_by_rung[num_context] = [source.name for source in rung_sources]
        if not rung_sources:
            continue

        for start in range(0, num_tasks, eval_batch_size):
            stop = min(start + eval_batch_size, num_tasks)
            single_batches = []
            batch_metadata = []

            for task_index in range(start, stop):
                active_features = int(
                    task_cache["active_num_features"][task_index].item()
                )
                x_raw = task_cache["x_raw_padded"][
                    task_index, :, :active_features
                ]
                y_raw = task_cache["y_raw"][task_index]
                row_permutation = task_cache["row_permutations"][task_index]
                feature_permutation = task_cache["feature_permutations"][task_index]
                context_pool_size = int(task_cache["context_pool_size"])
                num_targets = int(task_cache["num_targets"])
                context_pool_indices = row_permutation[:context_pool_size]
                target_indices = row_permutation[
                    context_pool_size : context_pool_size + num_targets
                ]

                batch, diagnostics = prepare_nested_rung(
                    generator=generator,
                    x_raw=x_raw,
                    y_raw=y_raw,
                    context_pool_indices=context_pool_indices,
                    target_indices=target_indices,
                    feature_permutation=feature_permutation,
                    num_context=num_context,
                )
                single_batches.append(batch)
                metadata = dict(task_cache["metadata"][task_index])
                batch_metadata.append(
                    {
                        "task_index": task_index,
                        "task_fingerprint": task_cache["task_fingerprints"][
                            task_index
                        ],
                        "target_fingerprint": task_cache["target_fingerprints"][
                            task_index
                        ],
                        "context_fingerprint": tensor_fingerprint(
                            x_raw[context_pool_indices[:num_context]],
                            y_raw[context_pool_indices[:num_context]],
                        ),
                        "bank_shard": metadata.get("shard", ""),
                        "bank_task_index": metadata.get("task_index", -1),
                        "scanned_index": metadata.get("scanned_index", -1),
                        **diagnostics,
                    }
                )

            batch_cpu = stack_single_task_batches(single_batches)
            batch = move_batch_to_device(batch_cpu, device)
            batch_index = start // eval_batch_size

            for source in rung_sources:
                runtime_key = f"{source.name}|nc{num_context}"
                runtime.setdefault(
                    runtime_key,
                    {
                        "elapsed_seconds": 0.0,
                        "peak_memory_bytes": 0.0,
                        "num_tasks": 0.0,
                    },
                )
                source_seed = stable_sampling_seed(
                    base_seed=sampling_seed,
                    source_offset=source.sampling_seed_offset,
                    batch_index=batch_index,
                    condition_index=rung_index,
                )
                pl.seed_everything(source_seed)

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                samples, _, _ = sample_loaded_source(
                    source=source,
                    batch=batch,
                    num_samples=num_eval_samples,
                    chunk_size=sample_chunk_size,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start_time

                runtime[runtime_key]["elapsed_seconds"] += elapsed
                runtime[runtime_key]["num_tasks"] += int(batch.yt.shape[0])
                if device.type == "cuda":
                    runtime[runtime_key]["peak_memory_bytes"] = max(
                        runtime[runtime_key]["peak_memory_bytes"],
                        float(torch.cuda.max_memory_allocated(device)),
                    )

                task_rows = per_task_metric_rows_efficient(
                    samples=samples,
                    target=batch.yt,
                    num_context=num_context,
                    model_name=source.name,
                    display_name=source.display_name,
                    checkpoint_path=source.checkpoint_path,
                    eval_set=str(cfg["eval_set_name"]),
                    task_index_start=start,
                    alpha=metric_alpha,
                    interval_levels=interval_levels,
                    compute_energy_score=compute_energy_score,
                )
                for local_index, row in enumerate(task_rows):
                    meta = batch_metadata[local_index]
                    row.update(meta)
                    row["batch_index"] = batch_index
                    row["training_alpha"] = source.training_alpha
                    row["metric_alpha"] = metric_alpha
                    row["sampling_seed"] = source_seed
                    row["source_group"] = group_name
                rows.extend(task_rows)
                del samples

            diagnostics_rows.extend(
                {
                    **meta,
                    "num_context": num_context,
                }
                for meta in batch_metadata
            )

            if batch_index % 25 == 0:
                print(
                    f"group={group_name} Nc={num_context}: "
                    f"tasks={stop}/{num_tasks}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(
                    group_dir / "per_task_metrics_partial.csv",
                    index=False,
                )

    result = pd.DataFrame(rows)
    _validate_group_pairing(
        rows=result,
        context_sizes=context_sizes,
        expected_by_rung=expected_by_rung,
        expected_tasks=num_tasks,
    )
    result.to_csv(group_dir / "per_task_metrics.csv", index=False)

    (
        pd.DataFrame(diagnostics_rows)
        .drop_duplicates(["task_index", "num_context"])
        .sort_values(["task_index", "num_context"])
        .to_csv(group_dir / "task_support_diagnostics.csv", index=False)
    )

    runtime_rows = []
    for key, record in runtime.items():
        source_name, context_label = key.split("|")
        task_count = int(record["num_tasks"])
        runtime_rows.append(
            {
                "model_name": source_name,
                "num_context": int(context_label[2:]),
                "num_tasks": task_count,
                "elapsed_seconds": record["elapsed_seconds"],
                "runtime_seconds_per_task": (
                    record["elapsed_seconds"] / task_count
                    if task_count
                    else float("nan")
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
        group_dir / "runtime_by_source_and_rung.csv",
        index=False,
    )

    print(
        f"Pairing PASS: group={group_name}, tasks={num_tasks}, "
        f"rungs={context_sizes}.",
        flush=True,
    )
    print(f"Wrote {group_dir / 'per_task_metrics.csv'}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
