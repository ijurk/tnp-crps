from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.synthetic import SyntheticBatch

from evaluate_synthetic_1d import (
    apply_eval_dataset_overrides,
    load_merged_config,
    move_batch_to_device,
)
from evaluation.binary_fork_metrics import (
    per_task_marginal_rows,
    sample_marginal_mixture,
)
from evaluation.binary_fork_utils import (
    load_sources,
    prepare_output_dir,
    runtime_metadata,
    source_metadata,
    validate_binary_fork_batch,
    write_resolved_config,
)
from evaluation.gaussian_controls_metrics import task_fingerprints
from evaluation.predictive_sampling import (
    sample_model_chunked,
    sampling_seed,
    validate_sampling_offsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired binary-fork ambiguous-marginal evaluation."
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


def _check_batch_pairing(
    *,
    rows: pd.DataFrame,
    source_names: List[str],
    expected_batch_size: int,
) -> None:
    post = rows.loc[rows["region"] == "postfork"]
    counts = post.groupby(["task_index", "task_fingerprint"])[
        "model_name"
    ].nunique()
    if not (counts == len(source_names)).all():
        raise RuntimeError("Batch pairing failed: source count mismatch.")
    if post["task_index"].nunique() != expected_batch_size:
        raise RuntimeError("Batch pairing failed: task count mismatch.")


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

    base_generator_config = str(cfg["base_generator_config"])
    source_entries = [dict(entry) for entry in cfg["sources"]]
    validate_sampling_offsets(source_entries)
    loaded_sources = load_sources(
        entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    generator_cfg = load_merged_config(config_paths=[base_generator_config])
    apply_eval_dataset_overrides(
        generator_cfg,
        samples_per_eval_set=samples_per_eval_set,
        eval_batch_size=eval_batch_size,
    )
    generator_cfg.generators.test.deterministic = True
    generator_cfg.generators.test.deterministic_seed = int(cfg["deterministic_seed"])
    generator = instantiate(generator_cfg.generators.test)
    loader = torch.utils.data.DataLoader(
        generator, batch_size=None, num_workers=0, pin_memory=False
    )

    expected_batches = int(generator.num_batches)
    if max_batches is not None:
        expected_batches = min(expected_batches, int(max_batches))
    expected_tasks = expected_batches * eval_batch_size

    write_resolved_config(
        path=output_dir / "eval_config_resolved.json",
        config=cfg,
        runtime=runtime_metadata(device),
        overrides={
            "output_dir": str(output_dir),
            "samples_per_eval_set": samples_per_eval_set,
            "eval_batch_size": eval_batch_size,
            "num_eval_samples": num_eval_samples,
            "sample_chunk_size": sample_chunk_size,
            "max_batches": max_batches,
        },
    )

    source_names = [str(entry["name"]) for entry in source_entries]
    output_path = output_dir / "per_task_metrics.csv"
    fingerprint_path = output_dir / "task_fingerprints.csv"
    wrote_header = False
    wrote_fingerprint_header = False
    task_index_start = 0

    print("=" * 88)
    print(
        f"BINARY FORK MARGINALS: tasks={expected_tasks}, "
        f"batch_size={eval_batch_size}, M={num_eval_samples}"
    )
    print("=" * 88)

    for batch_index, batch_cpu in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        if not isinstance(batch_cpu, SyntheticBatch):
            raise TypeError(f"Expected SyntheticBatch, got {type(batch_cpu)}.")

        validate_binary_fork_batch(
            batch=batch_cpu,
            expected_fork_x0=float(cfg["fork_x0"]),
            expected_delta=float(cfg["delta"]),
            expected_noise_std=float(cfg["noise_std"]),
            require_ambiguous_context=True,
        )
        fingerprints = task_fingerprints(batch_cpu)
        batch = move_batch_to_device(batch_cpu, device)
        gt = batch_cpu.gt_pred
        assert gt is not None

        component_means, component_scales, regime_weights = (
            gt.posterior_marginal_components(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                include_target_noise=True,
            )
        )

        batch_rows: List[Dict[str, Any]] = []
        for loaded in loaded_sources:
            entry = loaded["entry"]
            seed = sampling_seed(
                base_seed=int(cfg["sampling_seed"]),
                source_offset=int(entry["sampling_seed_offset"]),
                batch_index=batch_index,
            )
            pl.seed_everything(seed, workers=False)
            kind = str(entry.get("kind", "model"))
            if kind == "oracle":
                samples = sample_marginal_mixture(
                    component_means=component_means,
                    component_scales=component_scales,
                    regime_weights=regime_weights,
                    num_samples=num_eval_samples,
                )
                checkpoint = "<exact_binary_marginal_oracle>"
            else:
                model = loaded["model"]
                assert model is not None
                samples = sample_model_chunked(
                    model=model,
                    batch=batch,
                    num_samples=num_eval_samples,
                    chunk_size=sample_chunk_size,
                )
                checkpoint = str(entry["checkpoint_path"])

            batch_rows.extend(
                per_task_marginal_rows(
                    samples=samples,
                    target=batch.yt,
                    xt=batch.xt,
                    component_means=component_means,
                    component_scales=component_scales,
                    regime_weights=regime_weights,
                    branch_start=float(cfg["branch_start"]),
                    gap_half_width_fraction=float(
                        cfg["gap_half_width_fraction"]
                    ),
                    task_index_start=task_index_start,
                    generator_batch_index=batch_index,
                    fingerprints=fingerprints,
                    model_name=str(entry["name"]),
                    source_kind=kind,
                    checkpoint_path=checkpoint,
                    eval_set=str(cfg["eval_set_name"]),
                    num_context=int(batch.xc.shape[1]),
                    metadata=source_metadata(entry),
                    condition="ambiguous",
                    interval_levels=tuple(
                        float(value) for value in cfg.get("interval_levels", [0.90])
                    ),
                )
            )
            del samples

        frame = pd.DataFrame(batch_rows)
        _check_batch_pairing(
            rows=frame,
            source_names=source_names,
            expected_batch_size=int(batch.yt.shape[0]),
        )
        frame.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        fingerprint_frame = (
            frame.loc[
                (frame["region"] == "postfork")
                & (frame["model_name"] == source_names[0]),
                [
                    "task_index",
                    "generator_batch_index",
                    "within_batch_index",
                    "num_context",
                    "num_targets",
                    "task_fingerprint",
                ],
            ]
            .drop_duplicates()
            .sort_values("task_index")
        )
        fingerprint_frame.to_csv(
            fingerprint_path,
            mode="a",
            header=not wrote_fingerprint_header,
            index=False,
        )
        wrote_fingerprint_header = True

        task_index_start += int(batch.yt.shape[0])
        if batch_index % 25 == 0 or batch_index + 1 == expected_batches:
            print(
                f"  processed batch {batch_index + 1}/{expected_batches}; "
                f"tasks={task_index_start}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if task_index_start != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} tasks, evaluated {task_index_start}."
        )
    print(
        f"Pairing PASS for eval_set={cfg['eval_set_name']}: "
        f"{expected_tasks} tasks x {len(source_names)} sources."
    )
    print(f"Wrote {output_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
