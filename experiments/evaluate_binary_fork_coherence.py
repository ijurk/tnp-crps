from __future__ import annotations

import argparse
import dataclasses
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
from evaluation.autoregressive import autoregressive_sample_model
from evaluation.binary_fork_metrics import per_task_path_rows
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


DEPLOYMENTS = ("direct", "autoregressive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched direct-versus-AR binary-fork coherence evaluation."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--samples_per_eval_set", default=None, type=int)
    parser.add_argument("--eval_batch_size", default=None, type=int)
    parser.add_argument("--num_paths", default=None, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _check_batch_pairing(
    *,
    frame: pd.DataFrame,
    source_names: List[str],
    expected_batch_size: int,
) -> None:
    for deployment in DEPLOYMENTS:
        selected = frame.loc[frame["deployment"] == deployment]
        grouped = selected.groupby(["task_index", "task_fingerprint"])
        if not grouped["model_name"].nunique().eq(len(source_names)).all():
            raise RuntimeError(f"Source pairing failed for {deployment}.")
        if selected["task_index"].nunique() != expected_batch_size:
            raise RuntimeError(f"Task count failed for {deployment}.")


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
    num_paths = int(args.num_paths if args.num_paths is not None else cfg["num_paths"])
    max_batches = (
        args.max_batches if args.max_batches is not None else cfg.get("max_batches")
    )
    if samples_per_eval_set % eval_batch_size != 0:
        raise ValueError("samples_per_eval_set must be divisible by eval_batch_size.")
    if num_paths < 2:
        raise ValueError("num_paths must be at least two.")

    num_anchors = int(cfg["num_ar_anchors"])
    training_max_nc = int(cfg["training_max_nc"])
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
            "num_paths": num_paths,
            "max_batches": max_batches,
        },
    )

    source_names = [str(entry["name"]) for entry in source_entries]
    output_path = output_dir / "per_task_coherence_metrics.csv"
    wrote_header = False
    task_index_start = 0

    print("=" * 88)
    print(
        f"BINARY FORK COHERENCE: tasks={expected_tasks}, paths={num_paths}, "
        f"anchors={num_anchors}"
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
        if int(batch_cpu.xc.shape[1]) + num_anchors > training_max_nc:
            raise RuntimeError(
                f"Nc+K={int(batch_cpu.xc.shape[1]) + num_anchors} exceeds "
                f"training_max_nc={training_max_nc}."
            )

        fingerprints = task_fingerprints(batch_cpu)
        batch = move_batch_to_device(batch_cpu, device)
        batch_size = int(batch.yt.shape[0])
        x_anchor = torch.linspace(
            float(cfg["ar_anchor_range"][0]),
            float(cfg["ar_anchor_range"][1]),
            num_anchors,
            device=device,
            dtype=batch.xc.dtype,
        )[None, :, None].expand(batch_size, -1, -1).contiguous()
        y_placeholder = torch.zeros(
            batch_size,
            num_anchors,
            batch.yc.shape[-1],
            device=device,
            dtype=batch.yc.dtype,
        )
        path_batch = dataclasses.replace(batch, xt=x_anchor, yt=y_placeholder)
        gt = batch_cpu.gt_pred
        assert gt is not None
        component_means, component_scales, regime_weights = (
            gt.posterior_marginal_components(
                xc=path_batch.xc,
                yc=path_batch.yc,
                xt=path_batch.xt,
                include_target_noise=True,
            )
        )

        batch_rows: List[Dict[str, Any]] = []
        for loaded in loaded_sources:
            entry = loaded["entry"]
            kind = str(entry.get("kind", "model"))
            direct_seed = sampling_seed(
                base_seed=int(cfg["sampling_seed"]),
                source_offset=int(entry["sampling_seed_offset"]),
                batch_index=batch_index,
                condition_index=0,
            )
            pl.seed_everything(direct_seed, workers=False)
            if kind == "oracle":
                direct_samples = gt.predictive_samples(
                    xc=path_batch.xc,
                    yc=path_batch.yc,
                    xt=path_batch.xt,
                    num_samples=num_paths,
                )
                checkpoint = "<exact_binary_joint_oracle>"
            else:
                model = loaded["model"]
                assert model is not None
                direct_samples = sample_model_chunked(
                    model=model,
                    batch=path_batch,
                    num_samples=num_paths,
                    chunk_size=min(int(cfg["direct_chunk_size"]), num_paths),
                )
                checkpoint = str(entry["checkpoint_path"])

            batch_rows.extend(
                per_task_path_rows(
                    samples=direct_samples,
                    x_path=x_anchor,
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
                    deployment="direct",
                    metadata=source_metadata(entry),
                    oracle_repeated=False,
                )
            )

            if kind == "oracle":
                ar_samples = direct_samples
                oracle_repeated = True
            else:
                ar_seed = sampling_seed(
                    base_seed=int(cfg["sampling_seed"]),
                    source_offset=int(entry["sampling_seed_offset"]),
                    batch_index=batch_index,
                    condition_index=1,
                )
                pl.seed_everything(ar_seed, workers=False)
                model = loaded["model"]
                assert model is not None
                ar_samples = autoregressive_sample_model(
                    model=model,
                    batch=path_batch,
                    num_samples=num_paths,
                    target_order=str(cfg["target_order"]),
                    stochln_noise_mode=str(cfg["stochln_noise_mode"]),
                )
                oracle_repeated = False

            batch_rows.extend(
                per_task_path_rows(
                    samples=ar_samples,
                    x_path=x_anchor,
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
                    deployment="autoregressive",
                    metadata=source_metadata(entry),
                    oracle_repeated=oracle_repeated,
                )
            )
            del direct_samples
            if kind != "oracle":
                del ar_samples

        frame = pd.DataFrame(batch_rows)
        _check_batch_pairing(
            frame=frame,
            source_names=source_names,
            expected_batch_size=batch_size,
        )
        frame.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        task_index_start += batch_size
        if batch_index % 5 == 0 or batch_index + 1 == expected_batches:
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
        f"Pairing PASS: {expected_tasks} tasks x {len(source_names)} sources "
        f"x {len(DEPLOYMENTS)} deployments."
    )
    print(f"Wrote {output_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
