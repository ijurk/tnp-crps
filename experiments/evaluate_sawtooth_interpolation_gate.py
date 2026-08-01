from __future__ import annotations

import argparse
import json
import dataclasses
import time
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluate_synthetic_1d import move_batch_to_device, sample_model
from plot_synthetic_1d_functions import load_models
from evaluation.autoregressive import autoregressive_sample_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--num_tasks_per_nc", default=None, type=int)
    parser.add_argument("--num_eval_samples", default=None, type=int)

    parser.add_argument(
        "--fixed_nc",
        nargs="+",
        default=None,
        type=int,
        help="Optional override for fixed context counts from the YAML.",
    )

    parser.add_argument(
        "--evaluation_mode",
        default=None,
        choices=[
            "one_shot",
            "ar_anchors",
        ],
        type=str,
        help=(
            "Evaluation procedure. The dissertation AR configs "
            "use 'ar_anchors'."
        ),
    )

    parser.add_argument(
        "--num_tasks",
        default=None,
        type=int,
        help=(
            "Number of independent test tasks for AR evaluation."
        ),
    )

    parser.add_argument(
        "--num_ar_samples",
        default=None,
        type=int,
        help=(
            "Number of independent AR rollout paths per task."
        ),
    )

    parser.add_argument(
        "--eval_batch_size",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--eval_nc",
        default=None,
        type=int,
        help=(
            "Initial context size before adding AR anchors."
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fair_crps_per_task(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Fair empirical CRPS, averaged over target/output coordinates.

    Args:
        samples: [M, B, Nt, Dy]
        target: [B, Nt, Dy]

    Returns:
        CRPS per task: [B]
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError(
            "Expected samples [M,B,Nt,Dy] and target [B,Nt,Dy]. "
            f"Got {tuple(samples.shape)} and {tuple(target.shape)}."
        )

    num_samples = int(samples.shape[0])

    if num_samples < 2:
        raise ValueError("Fair CRPS requires at least two samples.")

    sorted_samples = samples.sort(dim=0).values

    first_term = (
        sorted_samples - target.unsqueeze(0)
    ).abs().mean(dim=0)

    ranks = torch.arange(
        1,
        num_samples + 1,
        device=samples.device,
        dtype=samples.dtype,
    ).view(num_samples, 1, 1, 1)

    # For sorted samples, this equals sum_{i<j} |x_j - x_i|.
    unordered_pair_sum = (
        (
            2.0 * ranks
            - float(num_samples)
            - 1.0
        )
        * sorted_samples
    ).sum(dim=0)

    second_term = unordered_pair_sum / (
        float(num_samples)
        * float(num_samples - 1)
    )

    pointwise_crps = first_term - second_term

    return pointwise_crps.mean(dim=(1, 2))


def fair_energy_per_task(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Fair Energy score over the complete finite target vector.

    Args:
        samples: [M, B, Nt, Dy]
        target: [B, Nt, Dy]

    Returns:
        Energy score per task: [B]
    """
    num_samples = int(samples.shape[0])
    batch_size = int(samples.shape[1])

    if num_samples < 2:
        raise ValueError(
            "Fair Energy score requires at least two samples."
        )

    sample_vectors = (
        samples.permute(1, 0, 2, 3)
        .reshape(batch_size, num_samples, -1)
    )

    target_vectors = target.reshape(batch_size, -1)

    first_term = torch.linalg.vector_norm(
        sample_vectors - target_vectors[:, None, :],
        dim=-1,
    ).mean(dim=1)

    pairwise_distances = torch.cdist(
        sample_vectors,
        sample_vectors,
        p=2.0,
    )

    second_term = pairwise_distances.sum(
        dim=(1, 2)
    ) / (
        2.0
        * float(num_samples)
        * float(num_samples - 1)
    )

    return first_term - second_term


def interval_metrics_per_task(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    levels = torch.tensor(
        [0.025, 0.05, 0.95, 0.975],
        device=samples.device,
        dtype=samples.dtype,
    )

    q025, q05, q95, q975 = torch.quantile(
        samples,
        levels,
        dim=0,
    )

    return {
        "coverage90": (
            (target >= q05)
            & (target <= q95)
        ).float().mean(dim=(1, 2)),
        "coverage95": (
            (target >= q025)
            & (target <= q975)
        ).float().mean(dim=(1, 2)),
        "width90": (
            q95 - q05
        ).mean(dim=(1, 2)),
        "width95": (
            q975 - q025
        ).mean(dim=(1, 2)),
    }


def metric_rows_for_batch(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    nc: int,
    source: str,
    task_offset: int,
) -> List[Dict[str, Any]]:
    if samples.shape[1:] != target.shape:
        raise ValueError(
            "Predictive samples do not match target shape. "
            f"samples={tuple(samples.shape)}, "
            f"target={tuple(target.shape)}."
        )

    predictive_mean = samples.mean(dim=0)

    mse = (
        predictive_mean - target
    ).square().mean(dim=(1, 2))

    crps = fair_crps_per_task(
        samples,
        target,
    )

    energy = fair_energy_per_task(
        samples,
        target,
    )

    intervals = interval_metrics_per_task(
        samples,
        target,
    )

    rows: List[Dict[str, Any]] = []

    for batch_index in range(target.shape[0]):
        rows.append(
            {
                "nc": int(nc),
                "source": source,
                "task_id": int(task_offset + batch_index),
                "mse": float(mse[batch_index].cpu()),
                "rmse_task": float(
                    mse[batch_index].sqrt().cpu()
                ),
                "crps": float(crps[batch_index].cpu()),
                "energy": float(energy[batch_index].cpu()),
                "coverage90": float(
                    intervals["coverage90"][batch_index].cpu()
                ),
                "coverage95": float(
                    intervals["coverage95"][batch_index].cpu()
                ),
                "width90": float(
                    intervals["width90"][batch_index].cpu()
                ),
                "width95": float(
                    intervals["width95"][batch_index].cpu()
                ),
            }
        )

    return rows


def summarise(
    per_task: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (nc, source), group in per_task.groupby(
        ["nc", "source"],
        sort=False,
    ):
        num_tasks = int(len(group))

        def standard_error(column: str) -> float:
            if num_tasks < 2:
                return float("nan")

            return float(
                group[column].std(ddof=1)
                / math.sqrt(num_tasks)
            )

        row: Dict[str, Any] = {
            "nc": int(nc),
            "source": source,
            "num_tasks": num_tasks,
            "rmse": math.sqrt(
                float(group["mse"].mean())
            ),
            "rmse_task_se": standard_error(
                "rmse_task"
            ),
            "crps": float(group["crps"].mean()),
            "crps_se": standard_error("crps"),
            "energy": float(group["energy"].mean()),
            "energy_se": standard_error("energy"),
            "coverage90": float(
                group["coverage90"].mean()
            ),
            "coverage90_se": standard_error(
                "coverage90"
            ),
            "coverage95": float(
                group["coverage95"].mean()
            ),
            "coverage95_se": standard_error(
                "coverage95"
            ),
            "width90": float(
                group["width90"].mean()
            ),
            "width90_se": standard_error("width90"),
            "width95": float(
                group["width95"].mean()
            ),
            "width95_se": standard_error("width95"),
        }

        if "runtime_s_per_task" in group.columns:
            row["runtime_s_per_task"] = float(
                group["runtime_s_per_task"].mean()
            )
            row["runtime_s_per_task_se"] = (
                standard_error(
                    "runtime_s_per_task"
                )
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    trivial = (
        summary[
            summary["source"] == "Trivial U(0,1)"
        ][
            [
                "nc",
                "rmse",
                "crps",
                "energy",
            ]
        ]
        .rename(
            columns={
                "rmse": "trivial_rmse",
                "crps": "trivial_crps",
                "energy": "trivial_energy",
            }
        )
    )

    summary = summary.merge(
        trivial,
        on="nc",
        how="left",
    )

    summary["delta_rmse_vs_trivial"] = (
        summary["rmse"]
        - summary["trivial_rmse"]
    )

    summary["delta_crps_vs_trivial"] = (
        summary["crps"]
        - summary["trivial_crps"]
    )

    summary["delta_energy_vs_trivial"] = (
        summary["energy"]
        - summary["trivial_energy"]
    )

    return summary


@torch.inference_mode()
def run_ar_anchor_evaluation(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> None:
    output_dir = Path(
        args.output_dir
        or cfg["output_dir"]
    )

    refuse_overwrite = bool(
        cfg.get("refuse_overwrite", False)
    )

    if (
        refuse_overwrite
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            "Output directory already contains files: "
            f"{output_dir}."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        args.device
        or cfg.get("device", "cuda")
    )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    device = torch.device(device_name)

    num_tasks = int(
        args.num_tasks
        if args.num_tasks is not None
        else cfg["num_tasks"]
    )

    num_ar_samples = int(
        args.num_ar_samples
        if args.num_ar_samples is not None
        else cfg["num_ar_samples"]
    )

    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else cfg["eval_batch_size"]
    )

    eval_nc = int(
        args.eval_nc
        if args.eval_nc is not None
        else cfg["eval_nc"]
    )

    num_anchors = int(
        cfg["num_ar_anchors"]
    )

    training_max_nc = int(
        cfg["training_max_nc"]
    )

    if eval_nc + num_anchors > training_max_nc:
        raise ValueError(
            "Completed AR context exceeds the training maximum: "
            f"{eval_nc} + {num_anchors} > "
            f"{training_max_nc}."
        )

    if num_tasks % eval_batch_size != 0:
        raise ValueError(
            "num_tasks must be divisible by eval_batch_size. "
            f"Got {num_tasks} and {eval_batch_size}."
        )

    if num_ar_samples < 2:
        raise ValueError(
            "num_ar_samples must be at least two."
        )

    seed = int(
        cfg.get("seed", 20260720)
    )

    anchor_seed = int(
        cfg.get(
            "anchor_seed",
            seed + 1,
        )
    )

    anchor_min = float(
        cfg["ar_anchor_range"][0]
    )
    anchor_max = float(
        cfg["ar_anchor_range"][1]
    )

    if not anchor_min < anchor_max:
        raise ValueError(
            "ar_anchor_range must satisfy min < max."
        )

    target_order = str(
        cfg.get(
            "target_order",
            "random",
        )
    )

    stochln_noise_mode = str(
        cfg.get(
            "stochln_noise_mode",
            "refresh",
        )
    )

    base_generator_config = str(
        cfg["base_generator_config"]
    )

    models = load_models(
        model_entries=cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    for item in models:
        item["model"].eval()

    generator_cfg = OmegaConf.load(
        base_generator_config
    )

    # Preserve the complete config tree so ${params.*}
    # interpolations remain valid.
    fixed_generator_cfg = OmegaConf.create(
        OmegaConf.to_container(
            generator_cfg,
            resolve=False,
        )
    )

    fixed_generator_cfg.generators.test.min_nc = (
        eval_nc
    )
    fixed_generator_cfg.generators.test.max_nc = (
        eval_nc
    )

    # The generator target set itself is not used. The evaluator
    # constructs its own uniformly random AR anchor set.
    fixed_generator_cfg.generators.test.min_nt = 1
    fixed_generator_cfg.generators.test.max_nt = 1

    fixed_generator_cfg.generators.test.samples_per_epoch = (
        num_tasks
    )
    fixed_generator_cfg.generators.test.batch_size = (
        eval_batch_size
    )
    fixed_generator_cfg.generators.test.deterministic = (
        True
    )

    resolved_test_cfg = OmegaConf.to_container(
        fixed_generator_cfg.generators.test,
        resolve=True,
    )

    set_seed(seed)

    generator = instantiate(
        resolved_test_cfg
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
    )

    all_rows: List[Dict[str, Any]] = []
    task_offset = 0

    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(
            batch,
            device,
        )

        batch_size = int(
            batch.xc.shape[0]
        )

        # One independently random support set per task.
        anchor_generator = torch.Generator(
            device="cpu"
        )

        anchor_generator.manual_seed(
            anchor_seed + batch_index
        )

        unit_anchor = torch.rand(
            batch_size,
            num_anchors,
            1,
            generator=anchor_generator,
            dtype=torch.float32,
        )

        x_anchor = (
            anchor_min
            + (
                anchor_max
                - anchor_min
            ) * unit_anchor
        ).to(
            device=device,
            dtype=batch.xc.dtype,
        )

        x_anchor = torch.sort(
            x_anchor,
            dim=1,
        ).values.contiguous()

        gt = getattr(
            batch,
            "gt_pred",
            None,
        )

        if (
            gt is None
            or not hasattr(
                gt,
                "latent_function",
            )
        ):
            raise RuntimeError(
                "Sawtooth batch must expose "
                "gt_pred.latent_function(...)."
            )

        y_anchor = gt.latent_function(
            x_anchor
        ).to(
            device=device,
            dtype=batch.yc.dtype,
        )

        ar_batch = dataclasses.replace(
            batch,
            xt=x_anchor,
            yt=y_anchor,
        )

        for item in models:
            model_seed = (
                seed
                + 1_000_000
                + batch_index
            )

            # Warm up the first batch before recording runtime.
            if batch_index == 0:
                set_seed(model_seed)

                _ = autoregressive_sample_model(
                    model=item["model"],
                    batch=ar_batch,
                    num_samples=min(
                        2,
                        num_ar_samples,
                    ),
                    target_order=target_order,  # type: ignore[arg-type]
                    stochln_noise_mode=stochln_noise_mode,  # type: ignore[arg-type]
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Reset before each model so the target permutations are
            # shared across models for the same task batch.
            set_seed(model_seed)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            samples = autoregressive_sample_model(
                model=item["model"],
                batch=ar_batch,
                num_samples=num_ar_samples,
                target_order=target_order,  # type: ignore[arg-type]
                stochln_noise_mode=stochln_noise_mode,  # type: ignore[arg-type]
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            rows = metric_rows_for_batch(
                samples=samples,
                target=y_anchor,
                nc=eval_nc,
                source=item["name"],
                task_offset=task_offset,
            )

            runtime_per_task = (
                elapsed
                / float(batch_size)
            )

            for row in rows:
                row["runtime_s_per_task"] = (
                    runtime_per_task
                )

            all_rows.extend(rows)

        # Explicit climatological baseline on the same targets.
        set_seed(
            seed
            + 2_000_000
            + batch_index
        )

        trivial_samples = torch.rand(
            (
                num_ar_samples,
                *y_anchor.shape,
            ),
            device=device,
            dtype=y_anchor.dtype,
        )

        trivial_rows = metric_rows_for_batch(
            samples=trivial_samples,
            target=y_anchor,
            nc=eval_nc,
            source="Trivial U(0,1)",
            task_offset=task_offset,
        )

        for row in trivial_rows:
            row["runtime_s_per_task"] = 0.0

        all_rows.extend(trivial_rows)

        task_offset += batch_size

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {task_offset:,}/"
                f"{num_tasks:,} test tasks."
            )

    if task_offset != num_tasks:
        raise RuntimeError(
            f"Expected {num_tasks} tasks, processed {task_offset}."
        )

    per_task = pd.DataFrame(all_rows)
    summary = summarise(per_task)

    per_task_path = (
        output_dir
        / "sawtooth_ar_per_task.csv"
    )
    summary_path = (
        output_dir
        / "sawtooth_ar_summary.csv"
    )
    resolved_path = (
        output_dir
        / "sawtooth_ar_resolved.json"
    )

    per_task.to_csv(
        per_task_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    with open(resolved_path, "w") as file:
        json.dump(
            {
                "config": cfg,
                "cli": vars(args),
                "resolved": {
                    "evaluation_mode": "ar_anchors",
                    "metrics_on": (
                        "raw AR predictive samples at "
                        "the sampled anchor locations"
                    ),
                    "num_tasks": num_tasks,
                    "num_ar_samples": num_ar_samples,
                    "eval_batch_size": eval_batch_size,
                    "eval_nc": eval_nc,
                    "num_ar_anchors": num_anchors,
                    "final_context_size": (
                        eval_nc + num_anchors
                    ),
                    "target_order": target_order,
                    "stochln_noise_mode": (
                        stochln_noise_mode
                    ),
                    "seed": seed,
                    "anchor_seed": anchor_seed,
                },
            },
            file,
            indent=2,
        )


    display_columns = [
        "nc",
        "source",
        "rmse",
        "crps",
        "energy",
        "coverage90",
        "coverage95",
        "width90",
        "width95",
        "runtime_s_per_task",
    ]

    print("\n=== Sawtooth AR-anchor evaluation ===")

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved {per_task_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {resolved_path}")



@torch.inference_mode()
def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    if not isinstance(cfg, dict):
        raise TypeError(
            "Expected evaluation config to resolve to a dictionary, "
            f"got {type(cfg)}."
        )

    evaluation_mode = str(
        args.evaluation_mode
        or cfg.get(
            "evaluation_mode",
            "one_shot",
        )
    )

    if evaluation_mode == "ar_anchors":
        run_ar_anchor_evaluation(
            args=args,
            cfg=cfg,
        )
        return

    output_dir = Path(
        args.output_dir
        or cfg["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        args.device
        or cfg.get("device", "cuda")
    )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Requested CUDA but CUDA is unavailable."
        )

    device = torch.device(device_name)

    fixed_nc_values = [
        int(value)
        for value in (
            args.fixed_nc
            if args.fixed_nc is not None
            else cfg.get(
                "fixed_nc_values",
                [1, 2, 4, 8, 16, 30],
            )
        )
    ]

    num_tasks_per_nc = int(
        args.num_tasks_per_nc
        if args.num_tasks_per_nc is not None
        else cfg.get("num_tasks_per_nc", 4096)
    )

    num_eval_samples = int(
        args.num_eval_samples
        if args.num_eval_samples is not None
        else cfg.get("num_eval_samples", 64)
    )

    eval_batch_size = int(
        cfg.get("eval_batch_size", 16)
    )

    seed = int(
        cfg.get("seed", 20260715)
    )

    if num_eval_samples < 2:
        raise ValueError(
            "num_eval_samples must be at least two."
        )

    if num_tasks_per_nc % eval_batch_size != 0:
        raise ValueError(
            "num_tasks_per_nc must be divisible by eval_batch_size. "
            f"Got {num_tasks_per_nc} and {eval_batch_size}."
        )

    base_generator_config = str(
        cfg["base_generator_config"]
    )

    models = load_models(
        model_entries=cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    for item in models:
        item["model"].eval()

    base_generator_cfg = OmegaConf.load(
        base_generator_config
    )

    all_rows: List[Dict[str, Any]] = []

    for nc in fixed_nc_values:
        if nc < 1:
            raise ValueError(
                "This evaluator currently requires Nc >= 1."
            )

        print(
            f"\n=== Fixed Nc={nc}: "
            f"{num_tasks_per_nc} tasks ==="
        )

        # Clone the complete config so interpolations such as ${params.dim_x}
        # remain attached to their root before being resolved.
        fixed_generator_cfg = OmegaConf.create(
            OmegaConf.to_container(
                base_generator_cfg,
                resolve=False,
            )
        )

        fixed_generator_cfg.generators.test.min_nc = int(nc)
        fixed_generator_cfg.generators.test.max_nc = int(nc)
        fixed_generator_cfg.generators.test.min_nt = 100
        fixed_generator_cfg.generators.test.max_nt = 100
        fixed_generator_cfg.generators.test.samples_per_epoch = int(
            num_tasks_per_nc
        )
        fixed_generator_cfg.generators.test.batch_size = int(
            eval_batch_size
        )
        fixed_generator_cfg.generators.test.deterministic = True

        resolved_test_cfg = OmegaConf.to_container(
            fixed_generator_cfg.generators.test,
            resolve=True,
        )

        set_seed(
            seed + 100000 * int(nc)
        )

        generator = instantiate(
            resolved_test_cfg
        )

        loader = torch.utils.data.DataLoader(
            generator,
            batch_size=None,
            num_workers=0,
        )

        # Materialise the deterministic task batches before model sampling,
        # so model RNG use cannot affect later generated tasks.
        batches = list(loader)

        generated_tasks = sum(
            int(batch.yt.shape[0])
            for batch in batches
        )

        if generated_tasks != num_tasks_per_nc:
            raise RuntimeError(
                f"Expected {num_tasks_per_nc} tasks at Nc={nc}, "
                f"but generated {generated_tasks}."
            )

        task_offset = 0

        for batch_index, batch in enumerate(batches):
            batch = move_batch_to_device(
                batch,
                device,
            )

            target = batch.yt

            for source_index, item in enumerate(models):
                set_seed(
                    seed
                    + 100000 * int(nc)
                    + 1000 * batch_index
                    + 10 * source_index
                )

                samples = sample_model(
                    model=item["model"],
                    batch=batch,
                    num_eval_samples=num_eval_samples,
                ).to(
                    device=device,
                    dtype=target.dtype,
                )

                all_rows.extend(
                    metric_rows_for_batch(
                        samples=samples,
                        target=target,
                        nc=nc,
                        source=item["name"],
                        task_offset=task_offset,
                    )
                )

            # Explicit climatology baseline: independent Uniform(0,1)
            # samples at every target, using the same tasks and ensemble size.
            set_seed(
                seed
                + 100000 * int(nc)
                + 1000 * batch_index
                + 999
            )

            trivial_samples = torch.rand(
                (
                    num_eval_samples,
                    *target.shape,
                ),
                device=device,
                dtype=target.dtype,
            )

            all_rows.extend(
                metric_rows_for_batch(
                    samples=trivial_samples,
                    target=target,
                    nc=nc,
                    source="Trivial U(0,1)",
                    task_offset=task_offset,
                )
            )

            task_offset += int(target.shape[0])

    per_task = pd.DataFrame(all_rows)
    summary = summarise(per_task)

    per_task_path = (
        output_dir
        / "sawtooth_interpolation_gate_per_task.csv"
    )

    summary_path = (
        output_dir
        / "sawtooth_interpolation_gate_summary.csv"
    )

    resolved_path = (
        output_dir
        / "sawtooth_interpolation_gate_resolved.json"
    )

    per_task.to_csv(
        per_task_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    with open(resolved_path, "w") as f:
        json.dump(
            {
                "config": cfg,
                "cli": vars(args),
                "resolved": {
                    "device": str(device),
                    "fixed_nc_values": fixed_nc_values,
                    "num_tasks_per_nc": num_tasks_per_nc,
                    "num_eval_samples": num_eval_samples,
                    "eval_batch_size": eval_batch_size,
                    "seed": seed,
                    "trivial_baseline": (
                        "i.i.d. Uniform(0,1) samples at each target, "
                        "same tasks and ensemble size as trained models"
                    ),
                },
            },
            f,
            indent=2,
        )

    display_columns = [
        "nc",
        "source",
        "rmse",
        "crps",
        "energy",
        "coverage90",
        "coverage95",
        "width90",
        "width95",
        "delta_crps_vs_trivial",
        "delta_energy_vs_trivial",
    ]

    print(
        "\n=== Sawtooth interpolation one-shot gate ==="
    )

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print(f"\nSaved {per_task_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {resolved_path}")


if __name__ == "__main__":
    main()