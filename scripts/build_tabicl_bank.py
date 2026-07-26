"""Build a sharded raw numerical TabICL GraphSCM bank."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tnp_crps.data.tabular_tabicl import (
    TabICLGraphTaskSource,
)


TABICL_COMMIT = (
    "46b91961db4f8873dd049ec09990698a435e1e29"
)

PRIOR_CONFIG: dict[str, Any] = {
    "add_gaussian_noise": False,
    "allow_act_warping": False,
    "allow_kumaraswamy_warping": True,
    "disallow_y_warping": False,
    "filter_unpredictable_datasets": False,
    "filter_unpredictable_graphs": False,
    "min_n_nodes": 2,
    "max_n_nodes": 32,
    "ensure_iid": False,
    "remove_trivial_datasets": False,
}


def _build_shard(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)

    shard_index = int(job["shard_index"])
    shard_seed = int(job["seed"]) + 10_007 * shard_index

    random.seed(shard_seed)
    np.random.seed(shard_seed)
    torch.manual_seed(shard_seed)

    output_path = Path(job["output_path"])
    temporary_path = output_path.with_suffix(".tmp")

    source = TabICLGraphTaskSource(
        min_features=int(job["min_features"]),
        max_features=int(job["max_features"]),
        device="cpu",
        max_generation_attempts=100,
        prior_config=PRIOR_CONFIG,
        tabicl_commit=TABICL_COMMIT,
    )

    num_tasks = int(job["num_tasks"])
    seq_len = int(job["seq_len"])
    max_features = int(job["max_features"])

    x_bank = torch.zeros(
        num_tasks,
        seq_len,
        max_features,
        dtype=torch.float32,
    )
    y_bank = torch.zeros(
        num_tasks,
        seq_len,
        1,
        dtype=torch.float32,
    )
    num_features_bank = torch.zeros(
        num_tasks,
        dtype=torch.int64,
    )

    for task_index in range(num_tasks):
        x, y, metadata = source.sample_task(seq_len)

        num_features = int(
            metadata["active_num_features"]
        )

        x_bank[
            task_index,
            :,
            :num_features,
        ] = x

        y_bank[task_index] = y
        num_features_bank[task_index] = num_features

    payload = {
        "format_version": 1,
        "x": x_bank,
        "y": y_bank,
        "num_features": num_features_bank,
        "seq_len": seq_len,
        "max_features": max_features,
        "split": str(job["split"]),
        "shard_index": shard_index,
        "seed": shard_seed,
        "tabicl_commit": TABICL_COMMIT,
        "prior_config": PRIOR_CONFIG,
        "full_sequence_preprocessing": False,
        "categorical_features": False,
    }

    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)

    return {
        "shard_index": shard_index,
        "path": str(output_path),
        "num_tasks": num_tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "test"),
    )
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_tasks < 1:
        raise ValueError("num-tasks must be positive.")

    if args.shard_size < 1:
        raise ValueError("shard-size must be positive.")

    if args.num_workers < 1:
        raise ValueError("num-workers must be positive.")

    split_dir = (
        Path(args.output_dir).expanduser().resolve()
        / args.split
    )
    split_dir.mkdir(parents=True, exist_ok=True)

    num_shards = math.ceil(
        args.num_tasks / args.shard_size
    )

    jobs = []

    for shard_index in range(num_shards):
        start = shard_index * args.shard_size
        remaining = args.num_tasks - start
        tasks_in_shard = min(
            args.shard_size,
            remaining,
        )

        output_path = (
            split_dir
            / f"shard_{shard_index:06d}.pt"
        )

        if output_path.exists():
            print(
                f"Skipping existing {output_path}",
                flush=True,
            )
            continue

        jobs.append(
            {
                "output_path": str(output_path),
                "split": args.split,
                "shard_index": shard_index,
                "num_tasks": tasks_in_shard,
                "seq_len": args.seq_len,
                "min_features": args.min_features,
                "max_features": args.max_features,
                "seed": args.seed,
            }
        )

    context = multiprocessing.get_context("spawn")

    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_build_shard, job): job
                for job in jobs
            }

            completed = 0

            for future in as_completed(futures):
                result = future.result()
                completed += 1

                print(
                    f"completed shard "
                    f"{result['shard_index'] + 1}/{num_shards}: "
                    f"{result['path']}",
                    flush=True,
                )

    shard_paths = sorted(split_dir.glob("shard_*.pt"))

    if len(shard_paths) != num_shards:
        raise RuntimeError(
            f"Expected {num_shards} shards, found "
            f"{len(shard_paths)} in {split_dir}."
        )

    manifest = {
        "format_version": 1,
        "split": args.split,
        "num_tasks": args.num_tasks,
        "num_shards": num_shards,
        "shard_size": args.shard_size,
        "seq_len": args.seq_len,
        "min_features": args.min_features,
        "max_features": args.max_features,
        "seed": args.seed,
        "tabicl_commit": TABICL_COMMIT,
        "prior_config": PRIOR_CONFIG,
        "full_sequence_preprocessing": False,
        "categorical_features": False,
    }

    manifest_path = split_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()