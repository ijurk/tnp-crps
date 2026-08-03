from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


CONDITIONS = ("ambiguous", "upper_reveal", "lower_reveal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched old-versus-mixed binary-fork conditioning acid test."
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


def _paired_variants(
    *,
    batch: SyntheticBatch,
    batch_index: int,
    counterfactual_seed: int,
    inject_x: float,
) -> Dict[str, SyntheticBatch]:
    gt = batch.gt_pred
    if gt is None:
        raise RuntimeError("Binary-fork batch has no exact oracle.")
    batch_size = int(batch.xc.shape[0])
    num_context = int(batch.xc.shape[1])

    pl.seed_everything(int(counterfactual_seed) + int(batch_index), workers=False)
    x_reveal = torch.full(
        (batch_size, 1, 1),
        float(inject_x),
        device=batch.x.device,
        dtype=batch.x.dtype,
    )
    x_joint = torch.cat([batch.x, x_reveal], dim=1)
    paired = gt.sample_paired_regime_observations(x=x_joint)
    lower = paired[:, 0, :-1]
    upper = paired[:, 1, :-1]
    lower_reveal_y = paired[:, 0, -1:]
    upper_reveal_y = paired[:, 1, -1:]

    yc_lower = lower[:, :num_context]
    yc_upper = upper[:, :num_context]
    if not torch.allclose(yc_lower, yc_upper, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            "Pre-fork paired counterfactual contexts are not identical."
        )
    shared_yc = yc_lower
    lower_yt = lower[:, num_context:]
    upper_yt = upper[:, num_context:]

    regimes = gt.sampled_regimes
    if regimes is None:
        raise RuntimeError("Binary-fork batch is missing realised regimes.")
    regimes = regimes.to(device=batch.x.device).reshape(batch_size, 1, 1)
    ambiguous_y = torch.where(regimes == 1, upper, lower)
    ambiguous_yt = ambiguous_y[:, num_context:]

    ambiguous = dataclasses.replace(
        batch,
        y=ambiguous_y,
        yc=shared_yc,
        yt=ambiguous_yt,
    )
    upper_x = torch.cat([batch.xc, x_reveal, batch.xt], dim=1)
    upper_y = torch.cat([shared_yc, upper_reveal_y, upper_yt], dim=1)
    lower_x = torch.cat([batch.xc, x_reveal, batch.xt], dim=1)
    lower_y = torch.cat([shared_yc, lower_reveal_y, lower_yt], dim=1)

    upper_variant = dataclasses.replace(
        batch,
        x=upper_x,
        y=upper_y,
        xc=torch.cat([batch.xc, x_reveal], dim=1),
        yc=torch.cat([shared_yc, upper_reveal_y], dim=1),
        yt=upper_yt,
    )
    lower_variant = dataclasses.replace(
        batch,
        x=lower_x,
        y=lower_y,
        xc=torch.cat([batch.xc, x_reveal], dim=1),
        yc=torch.cat([shared_yc, lower_reveal_y], dim=1),
        yt=lower_yt,
    )
    return {
        "ambiguous": ambiguous,
        "upper_reveal": upper_variant,
        "lower_reveal": lower_variant,
    }


def _check_condition_pairing(
    *,
    frame: pd.DataFrame,
    source_names: List[str],
    condition: str,
    expected_batch_size: int,
) -> None:
    selected = frame.loc[
        (frame["condition"] == condition) & (frame["region"] == "postfork")
    ]
    grouped = selected.groupby(["task_index", "task_fingerprint"])
    if not grouped["model_name"].nunique().eq(len(source_names)).all():
        raise RuntimeError(f"Source pairing failed for condition={condition}.")
    if selected["task_index"].nunique() != expected_batch_size:
        raise RuntimeError(f"Task count failed for condition={condition}.")


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
    generator_cfg.generators.test.min_nc = int(cfg["min_nc"])
    generator_cfg.generators.test.max_nc = int(cfg["max_nc"])
    if int(cfg["max_nc"]) + 1 > int(cfg["old_training_max_nc"]):
        raise ValueError(
            "max_nc + one reveal exceeds the original checkpoints' training support."
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
    output_path = output_dir / "per_task_conditioning_metrics.csv"
    wrote_header = False
    task_index_start = 0

    print("=" * 88)
    print(
        f"BINARY FORK CONDITIONING: base tasks={expected_tasks}, "
        f"conditions={len(CONDITIONS)}, sources={len(source_names)}"
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
        variants_cpu = _paired_variants(
            batch=batch_cpu,
            batch_index=batch_index,
            counterfactual_seed=int(cfg["counterfactual_seed"]),
            inject_x=float(cfg["inject_x"]),
        )
        base_fingerprints = task_fingerprints(variants_cpu["ambiguous"])

        all_batch_rows: List[Dict[str, Any]] = []
        for condition_index, condition in enumerate(CONDITIONS):
            variant_cpu = variants_cpu[condition]
            condition_fingerprints = task_fingerprints(variant_cpu)
            batch = move_batch_to_device(variant_cpu, device)
            gt = variant_cpu.gt_pred
            assert gt is not None
            component_means, component_scales, regime_weights = (
                gt.posterior_marginal_components(
                    xc=batch.xc,
                    yc=batch.yc,
                    xt=batch.xt,
                    include_target_noise=True,
                )
            )

            for loaded in loaded_sources:
                entry = loaded["entry"]
                seed = sampling_seed(
                    base_seed=int(cfg["sampling_seed"]),
                    source_offset=int(entry["sampling_seed_offset"]),
                    batch_index=batch_index,
                    condition_index=condition_index,
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

                rows = per_task_marginal_rows(
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
                    fingerprints=condition_fingerprints,
                    model_name=str(entry["name"]),
                    source_kind=kind,
                    checkpoint_path=checkpoint,
                    eval_set=str(cfg["eval_set_name"]),
                    num_context=int(batch.xc.shape[1]),
                    metadata=source_metadata(entry),
                    condition=condition,
                    interval_levels=tuple(
                        float(value) for value in cfg.get("interval_levels", [0.90])
                    ),
                )
                for row in rows:
                    row["base_task_fingerprint"] = base_fingerprints[
                        int(row["within_batch_index"])
                    ]
                    row["revealed_regime"] = (
                        "upper"
                        if condition == "upper_reveal"
                        else "lower"
                        if condition == "lower_reveal"
                        else "none"
                    )
                all_batch_rows.extend(rows)
                del samples

        frame = pd.DataFrame(all_batch_rows)
        for condition in CONDITIONS:
            _check_condition_pairing(
                frame=frame,
                source_names=source_names,
                condition=condition,
                expected_batch_size=int(batch_cpu.yt.shape[0]),
            )
        frame.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        task_index_start += int(batch_cpu.yt.shape[0])
        if batch_index % 10 == 0 or batch_index + 1 == expected_batches:
            print(
                f"  processed batch {batch_index + 1}/{expected_batches}; "
                f"base_tasks={task_index_start}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if task_index_start != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} base tasks, evaluated {task_index_start}."
        )
    print(
        f"Pairing PASS: {expected_tasks} base tasks x {len(CONDITIONS)} "
        f"conditions x {len(source_names)} sources."
    )
    print(f"Wrote {output_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
