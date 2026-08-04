from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.synthetic import SyntheticBatch

from evaluate_synthetic_1d import load_merged_config, move_batch_to_device
from evaluation.predictive_sampling import (
    sample_model_chunked,
    sampling_seed,
    validate_sampling_offsets,
)
from evaluation.sawtooth_final_metrics import (
    per_task_marginal_rows,
    task_fingerprints,
    theoretical_uniform_reference,
    trivial_uniform_samples,
)
from evaluation.sawtooth_final_utils import (
    load_sources,
    prepare_output_dir,
    runtime_metadata,
    source_metadata,
    validate_sawtooth_batch,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired finite-ensemble sawtooth marginal evaluation. "
            "The same estimator is used for learned and trivial sources."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--samples_per_eval_set", default=None, type=int)
    parser.add_argument("--eval_batch_size", default=None, type=int)
    parser.add_argument("--num_eval_samples", default=None, type=int)
    parser.add_argument("--sample_chunk_size", default=None, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pairing_check(
    *,
    frame: pd.DataFrame,
    source_names: List[str],
    expected_batch_size: int,
) -> None:
    if frame["task_index"].nunique() != int(expected_batch_size):
        raise RuntimeError("Batch pairing failed: task count mismatch.")
    grouped = frame.groupby("task_index")
    if not grouped["model_name"].nunique().eq(len(source_names)).all():
        raise RuntimeError("Batch pairing failed: source count mismatch.")
    if not grouped["task_fingerprint"].nunique().eq(1).all():
        raise RuntimeError("Batch pairing failed: task fingerprints differ.")
    if frame.duplicated(["model_name", "task_index"]).any():
        raise RuntimeError("Batch pairing failed: duplicate source/task rows.")


def _timed_samples(
    *,
    loaded: Dict[str, Any],
    batch: SyntheticBatch,
    num_eval_samples: int,
    sample_chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, int]:
    entry = loaded["entry"]
    kind = str(entry.get("kind", "model"))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    if kind == "uniform":
        samples = trivial_uniform_samples(
            target=batch.yt,
            num_samples=num_eval_samples,
        )
    elif kind == "model":
        model = loaded["model"]
        if model is None:
            raise RuntimeError(f"Source {entry['name']!r} was not loaded.")
        samples = sample_model_chunked(
            model=model,
            batch=batch,
            num_samples=num_eval_samples,
            chunk_size=sample_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported source kind={kind!r}.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    return samples, elapsed, peak_memory


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Evaluation config must resolve to a dictionary.")

    output_dir = prepare_output_dir(
        args.output_dir or cfg["output_dir"], overwrite=args.overwrite
    )

    device_name = args.device or str(cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)

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
    max_batches = (
        args.max_batches if args.max_batches is not None else cfg.get("max_batches")
    )

    if samples_per_eval_set % eval_batch_size != 0:
        raise ValueError("samples_per_eval_set must be divisible by eval_batch_size.")
    if num_eval_samples < 2 or sample_chunk_size < 2:
        raise ValueError("num_eval_samples and sample_chunk_size must be >=2.")

    generator_spec = dict(cfg["test_generator"])
    test_min_nc = int(generator_spec["min_nc"])
    test_max_nc = int(generator_spec["max_nc"])
    if test_min_nc < 1 or test_max_nc < test_min_nc:
        raise ValueError("Invalid test context range.")

    base_generator_config = str(cfg["base_generator_config"])
    source_entries = [dict(entry) for entry in cfg["sources"]]
    validate_sampling_offsets(source_entries)
    loaded_sources = load_sources(
        entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    generator_cfg = load_merged_config(config_paths=[base_generator_config])
    OmegaConf.set_struct(generator_cfg, False)
    generator_cfg.generators.test.min_nc = test_min_nc
    generator_cfg.generators.test.max_nc = test_max_nc
    generator_cfg.generators.test.min_nt = int(generator_spec.get("num_targets", 100))
    generator_cfg.generators.test.max_nt = int(generator_spec.get("num_targets", 100))
    generator_cfg.generators.test.samples_per_epoch = samples_per_eval_set
    generator_cfg.generators.test.batch_size = eval_batch_size
    generator_cfg.generators.test.deterministic = True
    generator_cfg.generators.test.deterministic_seed = int(
        cfg["deterministic_seed"]
    )

    generator = instantiate(generator_cfg.generators.test)
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    expected_batches = int(generator.num_batches)
    if max_batches is not None:
        expected_batches = min(expected_batches, int(max_batches))
    expected_tasks = expected_batches * eval_batch_size

    runtime = runtime_metadata(device)
    write_resolved_config(
        path=output_dir / "eval_config_resolved.json",
        config=cfg,
        runtime=runtime,
        overrides={
            "output_dir": str(output_dir),
            "samples_per_eval_set": samples_per_eval_set,
            "eval_batch_size": eval_batch_size,
            "num_eval_samples": num_eval_samples,
            "sample_chunk_size": sample_chunk_size,
            "max_batches": max_batches,
            "resolved_test_min_nc": test_min_nc,
            "resolved_test_max_nc": test_max_nc,
        },
    )

    source_names = [str(entry["name"]) for entry in source_entries]
    output_path = output_dir / "per_task_metrics.csv"
    fingerprint_path = output_dir / "task_fingerprints.csv"
    runtime_path = output_dir / "runtime_by_source.csv"
    wrote_header = False
    wrote_fingerprint_header = False
    task_index_start = 0

    timing: Dict[str, Dict[str, float]] = {
        name: {
            "elapsed_seconds": 0.0,
            "num_tasks": 0.0,
            "num_batches": 0.0,
            "peak_memory_bytes": 0.0,
        }
        for name in source_names
    }

    print("=" * 92)
    print(
        f"SAWTOOTH MARGINALS: tasks={expected_tasks}, batch_size={eval_batch_size}, "
        f"M={num_eval_samples}, chunk={sample_chunk_size}, "
        f"Nc~Uniform{{{test_min_nc},...,{test_max_nc}}}"
    )
    print("=" * 92)

    for batch_index, batch_cpu in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        if not isinstance(batch_cpu, SyntheticBatch):
            raise TypeError(f"Expected SyntheticBatch, got {type(batch_cpu)}.")

        validate_sawtooth_batch(
            batch=batch_cpu,
            min_freq=float(cfg["sawtooth"]["min_freq"]),
            max_freq=float(cfg["sawtooth"]["max_freq"]),
            noise_std=float(cfg["sawtooth"]["noise_std"]),
        )
        fingerprints = task_fingerprints(batch_cpu)
        batch = move_batch_to_device(batch_cpu, device)
        batch_rows: List[Dict[str, Any]] = []

        for loaded in loaded_sources:
            entry = loaded["entry"]
            source_name = str(entry["name"])
            seed = sampling_seed(
                base_seed=int(cfg["sampling_seed"]),
                source_offset=int(entry["sampling_seed_offset"]),
                batch_index=batch_index,
            )
            pl.seed_everything(seed, workers=False)

            samples, elapsed, peak_memory = _timed_samples(
                loaded=loaded,
                batch=batch,
                num_eval_samples=num_eval_samples,
                sample_chunk_size=sample_chunk_size,
                device=device,
            )
            kind = str(entry.get("kind", "model"))
            checkpoint_path = (
                "<sampled_uniform_baseline>"
                if kind == "uniform"
                else str(entry["checkpoint_path"])
            )

            batch_rows.extend(
                per_task_marginal_rows(
                    samples=samples,
                    target=batch.yt,
                    batch_cpu=batch_cpu,
                    model_name=source_name,
                    source_kind=kind,
                    checkpoint_path=checkpoint_path,
                    eval_set=str(cfg["eval_set_name"]),
                    task_index_start=task_index_start,
                    generator_batch_index=batch_index,
                    fingerprints=fingerprints,
                    metadata=source_metadata(entry),
                    compute_energy=False,
                )
            )

            timing[source_name]["elapsed_seconds"] += elapsed
            timing[source_name]["num_tasks"] += int(batch.yt.shape[0])
            timing[source_name]["num_batches"] += 1
            timing[source_name]["peak_memory_bytes"] = max(
                timing[source_name]["peak_memory_bytes"], float(peak_memory)
            )
            del samples

        frame = pd.DataFrame(batch_rows)
        _pairing_check(
            frame=frame,
            source_names=source_names,
            expected_batch_size=int(batch.yt.shape[0]),
        )
        frame.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        fingerprint_rows = pd.DataFrame(
            {
                "task_index": [
                    task_index_start + index
                    for index in range(int(batch.yt.shape[0]))
                ],
                "generator_batch_index": batch_index,
                "task_fingerprint": fingerprints,
                "num_context": int(batch.xc.shape[1]),
            }
        )
        fingerprint_rows.to_csv(
            fingerprint_path,
            mode="a",
            header=not wrote_fingerprint_header,
            index=False,
        )
        wrote_fingerprint_header = True

        task_index_start += int(batch.yt.shape[0])
        if batch_index == 0 or (batch_index + 1) % 100 == 0:
            print(
                f"  processed batch {batch_index + 1}/{expected_batches}; "
                f"tasks={task_index_start:,}"
            )

    if task_index_start != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} tasks, processed {task_index_start}."
        )

    runtime_rows = []
    for source_name in source_names:
        values = timing[source_name]
        num_tasks = int(values["num_tasks"])
        runtime_rows.append(
            {
                "model_name": source_name,
                "num_tasks": num_tasks,
                "num_batches": int(values["num_batches"]),
                "elapsed_seconds": float(values["elapsed_seconds"]),
                "runtime_seconds_per_task": (
                    float(values["elapsed_seconds"]) / max(num_tasks, 1)
                ),
                "peak_memory_bytes": int(values["peak_memory_bytes"]),
                "device": str(device),
                "cuda_device_name": runtime.get("cuda_device_name"),
            }
        )
    pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)

    # The sampled baseline is an estimator-parity row. These values are a
    # transparent sanity reference rather than hard equality assertions.
    uniform_reference = theoretical_uniform_reference(num_eval_samples)
    (output_dir / "uniform_reference.json").write_text(
        json.dumps(uniform_reference, indent=2)
    )

    # Every batch is checked before it is appended, and the final processed-task
    # count is asserted above. Avoid reloading the complete 80,000-task CSV here:
    # that would add a large, unnecessary memory spike after evaluation.

    print(
        f"Pairing PASS for eval_set={cfg['eval_set_name']}: "
        f"{expected_tasks} tasks x {len(source_names)} sources."
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {fingerprint_path}")
    print(f"Wrote {runtime_path}")
    print("Uniform theoretical reference:")
    for key, value in uniform_reference.items():
        print(f"  {key}: {value:.9f}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
