"""Audit a raw numerical TabICL bank before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from tnp_crps.data.tabular_preprocessing import tabicl_preprocess_from_context


def load_shard(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def print_quantiles(
    name: str,
    values: list[float],
) -> None:
    tensor = torch.tensor(values, dtype=torch.float64)

    print(f"\n{name}")

    for quantile in (0.50, 0.90, 0.95, 0.99, 0.999, 1.0):
        value = torch.quantile(tensor, quantile)
        print(f"q={quantile:>5}: {float(value):.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-tasks", type=int, default=4096)
    parser.add_argument("--nc", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument(
        "--min-context-target-std",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--support-bound", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_dir = Path(args.bank_dir) / args.split
    shard_paths = sorted(split_dir.glob("shard_*.pt"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found under {split_dir}."
        )

    generator = torch.Generator().manual_seed(args.seed)

    rejection_reasons: Counter[str] = Counter()

    max_abs_x_values: list[float] = []
    max_abs_y_values: list[float] = []

    raw_x_tail_tasks = 0
    raw_y_tail_tasks = 0
    total_tasks = 0

    for shard_path in shard_paths:
        payload = load_shard(shard_path)

        for task_index in range(payload["x"].shape[0]):
            if total_tasks >= args.max_tasks:
                break

            num_features = int(
                payload["num_features"][task_index]
            )

            x = payload["x"][
                task_index,
                :,
                :num_features,
            ].float()

            y = payload["y"][task_index].float()

            if not torch.isfinite(x).all():
                rejection_reasons["nonfinite_raw_x"] += 1
                total_tasks += 1
                continue

            if not torch.isfinite(y).all():
                rejection_reasons["nonfinite_raw_y"] += 1
                total_tasks += 1
                continue

            # End-to-end diagnostic that raw shards have not been clipped.
            x_full_std = x.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            )
            x_valid = x_full_std > 1.0e-12

            if x_valid.any():
                x_full_z = torch.where(
                    x_valid,
                    (x - x.mean(dim=0, keepdim=True))
                    / x_full_std.clamp_min(1.0e-12),
                    torch.zeros_like(x),
                )

                if bool((x_full_z.abs() > 4.0).any()):
                    raw_x_tail_tasks += 1

            y_full_std = y.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            )

            if float(y_full_std.min()) > 1.0e-12:
                y_full_z = (
                    y - y.mean(dim=0, keepdim=True)
                ) / y_full_std

                if bool((y_full_z.abs() > 4.0).any()):
                    raw_y_tail_tasks += 1

            permutation = torch.randperm(
                x.shape[0],
                generator=generator,
            )
            x = x[permutation]
            y = y[permutation]

            nc = int(args.nc)

            xc_raw = x[:nc]
            xt_raw = x[nc:]
            yc_raw = y[:nc]
            yt_raw = y[nc:]

            context_y_std = yc_raw.std(
                dim=0,
                unbiased=False,
            )

            if (
                not torch.isfinite(context_y_std).all()
                or float(context_y_std.min())
                < args.min_context_target_std
            ):
                rejection_reasons["context_y_std"] += 1
                total_tasks += 1
                continue

            xc, xt, _, _ = tabicl_preprocess_from_context(
                xc_raw,
                xt_raw,
                epsilon=args.epsilon,
                outlier_threshold=4.0,
                standardized_clip=100.0,
                zero_constant_dimensions=True,
            )

            max_abs_x = float(
                torch.maximum(
                    xc.abs().max(),
                    xt.abs().max(),
                )
            )
            max_abs_x_values.append(max_abs_x)

            if max_abs_x > args.support_bound:
                rejection_reasons["standardized_x_bound"] += 1
                total_tasks += 1
                continue

            yc, yt, _, _ = tabicl_preprocess_from_context(
                yc_raw,
                yt_raw,
                epsilon=args.epsilon,
                outlier_threshold=4.0,
                standardized_clip=100.0,
                zero_constant_dimensions=False,
            )

            max_abs_y = float(
                torch.maximum(
                    yc.abs().max(),
                    yt.abs().max(),
                )
            )
            max_abs_y_values.append(max_abs_y)

            if max_abs_y > args.support_bound:
                rejection_reasons["standardized_y_bound"] += 1
                total_tasks += 1
                continue

            rejection_reasons["accepted"] += 1
            total_tasks += 1

        if total_tasks >= args.max_tasks:
            break

    rejected = total_tasks - rejection_reasons["accepted"]
    rejection_rate = rejected / total_tasks

    print(f"tasks audited: {total_tasks}")
    print(f"accepted: {rejection_reasons['accepted']}")
    print(f"rejected: {rejected}")
    print(f"rejection rate: {100.0 * rejection_rate:.4f}%")

    print("\nreasons:")
    for name, count in sorted(rejection_reasons.items()):
        print(f"  {name}: {count}")

    print(
        "\nraw tasks with at least one full-sequence |z| > 4:"
    )
    print(f"  x: {raw_x_tail_tasks}")
    print(f"  y: {raw_y_tail_tasks}")

    if max_abs_x_values:
        print_quantiles(
            "context-standardized max |x|",
            max_abs_x_values,
        )

    if max_abs_y_values:
        print_quantiles(
            "context-standardized max |y|",
            max_abs_y_values,
        )

    print()
    if rejection_rate <= 0.02:
        print(
            "DECISION: rejection rate <= 2%; "
            "proceed with the final raw bank."
        )
    elif rejection_rate <= 0.05:
        print(
            "DECISION: rejection rate is between 2% and 5%; "
            "inspect rejection causes before building the final bank."
        )
    else:
        print(
            "DECISION: rejection rate exceeds 5%; "
            "do not build or train yet."
        )


if __name__ == "__main__":
    main()