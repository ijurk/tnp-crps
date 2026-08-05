from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from omegaconf import OmegaConf

from evaluation.tabular_final_utils import (
    build_generator,
    prepare_nested_rung,
    raw_task_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--accepted_tasks", type=int, default=None)
    parser.add_argument("--max_scanned_tasks", type=int, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    task_cfg = dict(cfg["nested_tasks"])

    output_path = Path(args.output or task_cfg["cache_path"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_target = int(
        args.accepted_tasks
        if args.accepted_tasks is not None
        else task_cfg["accepted_tasks"]
    )
    max_scanned = args.max_scanned_tasks
    if max_scanned is None:
        max_scanned = task_cfg.get("max_scanned_tasks", None)
    max_scanned = int(max_scanned) if max_scanned is not None else None

    context_sizes = tuple(int(value) for value in task_cfg["context_sizes"])
    context_pool_size = int(task_cfg["context_pool_size"])
    num_targets = int(task_cfg["num_targets"])
    permutation_seed = int(task_cfg["row_permutation_seed"])
    feature_seed = int(task_cfg["feature_permutation_seed"])

    if max(context_sizes) > context_pool_size:
        raise ValueError("Largest context size exceeds context_pool_size.")

    generator = build_generator(
        base_generator_config=str(cfg["base_generator_config"]),
        overrides=list(task_cfg.get("generator_overrides", []) or []),
        samples_per_epoch=accepted_target,
        batch_size=1,
    )
    source = generator.source
    source_seq_len = int(source.seq_len)
    required_rows = context_pool_size + num_targets
    if source_seq_len < required_rows:
        raise ValueError(
            f"Stored tasks have {source_seq_len} rows; {required_rows} required."
        )

    max_features = int(generator.max_input_features)
    accepted_x: List[torch.Tensor] = []
    accepted_y: List[torch.Tensor] = []
    accepted_active_features: List[int] = []
    accepted_row_permutations: List[torch.Tensor] = []
    accepted_feature_permutations: List[torch.Tensor] = []
    accepted_metadata: List[Dict[str, Any]] = []
    task_fingerprints: List[str] = []
    target_fingerprints: List[str] = []

    rung_acceptance = Counter()
    rung_rejections: Dict[int, Counter] = defaultdict(Counter)
    intersection_rejections = Counter()
    support_rows: List[Dict[str, Any]] = []

    scanned_index = 0

    while len(accepted_x) < accepted_target:
        if max_scanned is not None and scanned_index >= max_scanned:
            raise RuntimeError(
                f"Reached max_scanned_tasks={max_scanned} with only "
                f"{len(accepted_x)} accepted intersection tasks."
            )

        try:
            x_raw, y_raw, metadata = source.sample_task(source_seq_len)
        except RuntimeError as exc:
            raise RuntimeError(
                "Exhausted the non-cycling test bank before collecting "
                f"{accepted_target} accepted intersection tasks. "
                f"accepted={len(accepted_x)}, scanned={scanned_index}."
            ) from exc

        x_raw = x_raw.detach().cpu().float()
        y_raw = y_raw.detach().cpu().float()
        active_features = int(metadata["active_num_features"])

        row_generator = torch.Generator(device="cpu")
        row_generator.manual_seed(permutation_seed + scanned_index)
        row_permutation = torch.randperm(source_seq_len, generator=row_generator)

        feature_generator = torch.Generator(device="cpu")
        feature_generator.manual_seed(feature_seed + scanned_index)
        feature_permutation = torch.randperm(
            max_features,
            generator=feature_generator,
        )

        context_pool_indices = row_permutation[:context_pool_size]
        target_indices = row_permutation[
            context_pool_size : context_pool_size + num_targets
        ]

        rung_batches = {}
        rung_diagnostics = {}
        failed = False
        first_failure = None

        for num_context in context_sizes:
            try:
                batch, diagnostics = prepare_nested_rung(
                    generator=generator,
                    x_raw=x_raw,
                    y_raw=y_raw,
                    context_pool_indices=context_pool_indices,
                    target_indices=target_indices,
                    feature_permutation=feature_permutation,
                    num_context=num_context,
                )
                rung_batches[num_context] = batch
                rung_diagnostics[num_context] = diagnostics
                rung_acceptance[num_context] += 1
            except Exception as exc:
                reason = getattr(exc, "reason", type(exc).__name__)
                rung_rejections[num_context][str(reason)] += 1
                if not failed:
                    first_failure = f"nc{num_context}:{reason}"
                failed = True

        if failed:
            intersection_rejections[str(first_failure)] += 1
            scanned_index += 1
            continue

        task_fingerprint, target_fingerprint = raw_task_fingerprints(
            x_raw=x_raw,
            y_raw=y_raw,
            context_pool_indices=context_pool_indices,
            target_indices=target_indices,
        )

        padded_x = torch.zeros(
            source_seq_len,
            max_features,
            dtype=torch.float32,
        )
        padded_x[:, :active_features] = x_raw

        accepted_index = len(accepted_x)
        accepted_x.append(padded_x)
        accepted_y.append(y_raw)
        accepted_active_features.append(active_features)
        accepted_row_permutations.append(row_permutation)
        accepted_feature_permutations.append(feature_permutation)
        accepted_metadata.append(
            {
                **dict(metadata),
                "accepted_index": accepted_index,
                "scanned_index": scanned_index,
            }
        )
        task_fingerprints.append(task_fingerprint)
        target_fingerprints.append(target_fingerprint)

        for num_context in context_sizes:
            diagnostics = rung_diagnostics[num_context]
            support_rows.append(
                {
                    "accepted_index": accepted_index,
                    "scanned_index": scanned_index,
                    "num_context": num_context,
                    "task_fingerprint": task_fingerprint,
                    "target_fingerprint": target_fingerprint,
                    **diagnostics,
                }
            )

        scanned_index += 1

        if len(accepted_x) % 256 == 0:
            print(
                f"accepted {len(accepted_x)}/{accepted_target}; "
                f"scanned={scanned_index}",
                flush=True,
            )

    payload = {
        "schema_version": "tabular_nested_ladder_tasks_v1",
        "x_raw_padded": torch.stack(accepted_x, dim=0),
        "y_raw": torch.stack(accepted_y, dim=0),
        "active_num_features": torch.tensor(
            accepted_active_features,
            dtype=torch.long,
        ),
        "row_permutations": torch.stack(accepted_row_permutations, dim=0),
        "feature_permutations": torch.stack(
            accepted_feature_permutations,
            dim=0,
        ),
        "metadata": accepted_metadata,
        "task_fingerprints": task_fingerprints,
        "target_fingerprints": target_fingerprints,
        "context_sizes": list(context_sizes),
        "context_pool_size": context_pool_size,
        "num_targets": num_targets,
        "source_seq_len": source_seq_len,
        "max_input_features": max_features,
        "accepted_tasks": accepted_target,
        "scanned_tasks": scanned_index,
        "row_permutation_seed": permutation_seed,
        "feature_permutation_seed": feature_seed,
    }
    torch.save(payload, output_path)

    diagnostics = {
        "schema_version": "tabular_nested_ladder_rejections_v1",
        "accepted_intersection_tasks": accepted_target,
        "scanned_tasks": scanned_index,
        "intersection_rejection_rate": (
            (scanned_index - accepted_target) / scanned_index
        ),
        "individually_accepted_by_rung": {
            str(key): int(value) for key, value in sorted(rung_acceptance.items())
        },
        "rejections_by_rung": {
            str(key): dict(sorted(value.items()))
            for key, value in sorted(rung_rejections.items())
        },
        "intersection_first_failure": dict(
            sorted(intersection_rejections.items())
        ),
        "cache_sha256": _sha256(output_path),
    }
    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    pd.DataFrame(support_rows).to_csv(
        output_path.with_name(output_path.stem + "_support.csv"),
        index=False,
    )

    print("TASK CACHE PASS")
    print(f"Wrote {output_path}")
    print(f"Wrote {diagnostics_path}")


if __name__ == "__main__":
    main()
