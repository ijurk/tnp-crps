from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from evaluate_synthetic_1d import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit old and mixed-context binary-fork checkpoint parity."
    )
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def as_dict(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def get_required(config: Any, dotted: str) -> Any:
    value = config
    for name in dotted.split("."):
        if not hasattr(value, name):
            raise KeyError(f"Resolved config is missing required field {dotted!r}.")
        value = getattr(value, name)
    return value


def main() -> None:
    args = parse_args()
    evaluation = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(evaluation, dict):
        raise TypeError("Evaluation config must resolve to a dictionary.")

    entries = [
        dict(entry)
        for entry in evaluation["sources"]
        if str(entry.get("kind", "model")) == "model"
    ]
    if len(entries) != 6:
        raise RuntimeError(
            "The conditioning audit expects six learned sources "
            f"(three old and three mixed), got {len(entries)}."
        )

    manifests: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        name = str(entry["name"])
        metadata = dict(entry.get("metadata", {}) or {})
        group = str(metadata.get("training_group"))
        generator_config = metadata.get("training_generator_config")
        if group not in {"ambiguous_only", "mixed_context"}:
            raise ValueError(f"Unexpected training_group={group!r} for {name}.")
        if generator_config is None:
            raise KeyError(f"{name} is missing metadata.training_generator_config.")

        checkpoint_path = Path(str(entry["checkpoint_path"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        config = load_merged_config(
            config_paths=[str(generator_config), str(entry["model_config"])],
            overrides=list(entry.get("overrides", []) or []),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        manifest = {
            "training_group": group,
            "training_generator_config": str(generator_config),
            "train_generator": as_dict(config.generators.train),
            "val_generator": as_dict(config.generators.val),
            "test_generator": as_dict(config.generators.test),
            "epochs": int(get_required(config, "params.epochs")),
            "optimiser": as_dict(get_required(config, "optimiser")),
            "scheduler": as_dict(get_required(config, "scheduler")),
            "gradient_clip_val": float(
                get_required(config, "misc.gradient_clip_val")
            ),
            "seed": int(get_required(config, "misc.seed")),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
        }
        manifests[name] = manifest
        groups[group].append(name)

        print("=" * 88)
        print(name)
        print(json.dumps(manifest, indent=2, default=str))

    if set(groups) != {"ambiguous_only", "mixed_context"}:
        raise RuntimeError(f"Unexpected training groups: {dict(groups)}")
    if any(len(names) != 3 for names in groups.values()):
        raise RuntimeError(f"Expected three models per training group: {dict(groups)}")

    common_fields = (
        "epochs",
        "optimiser",
        "scheduler",
        "gradient_clip_val",
        "seed",
        "checkpoint_epoch",
        "checkpoint_global_step",
    )
    reference_name = entries[0]["name"]
    reference = manifests[str(reference_name)]
    for name, manifest in manifests.items():
        for field in common_fields:
            if manifest[field] != reference[field]:
                raise RuntimeError(
                    f"Training-budget parity failed for {name}: field={field}\n"
                    f"reference={reference[field]}\nmodel={manifest[field]}"
                )

    for group, names in groups.items():
        group_reference = manifests[names[0]]["train_generator"]
        for name in names[1:]:
            if manifests[name]["train_generator"] != group_reference:
                raise RuntimeError(
                    f"Training generator differs within group={group}: {name}."
                )

    old_train = manifests[groups["ambiguous_only"][0]]["train_generator"]
    mixed_train = manifests[groups["mixed_context"][0]]["train_generator"]
    if not str(old_train["_target_"]).endswith("BinaryLatentForkGenerator"):
        raise RuntimeError(f"Unexpected old train generator: {old_train['_target_']}")
    if int(old_train["max_nc"]) != 32:
        raise RuntimeError(f"Old max_nc must be 32, got {old_train['max_nc']}.")
    if not str(mixed_train["_target_"]).endswith(
        "BinaryLatentForkGeneratorMixedContext"
    ):
        raise RuntimeError(
            f"Unexpected mixed train generator: {mixed_train['_target_']}"
        )
    if int(mixed_train["max_nc"]) != 64:
        raise RuntimeError(f"Mixed max_nc must be 64, got {mixed_train['max_nc']}.")
    if abs(float(mixed_train["p_revealing"]) - 0.5) > 1.0e-12:
        raise RuntimeError(
            f"Mixed p_revealing must be 0.5, got {mixed_train['p_revealing']}."
        )

    old_ref = manifests[groups["ambiguous_only"][0]]
    mixed_ref = manifests[groups["mixed_context"][0]]
    for split in ("val_generator", "test_generator"):
        if old_ref[split] != mixed_ref[split]:
            raise RuntimeError(
                f"Old and mixed {split} configurations differ. "
                "The acid test must remain unchanged."
            )

    print()
    print("PASS: binary-fork training audit")
    print("  - optimiser, scheduler, seed and training budget match across six runs")
    print("  - three checkpoints within each group use the same training generator")
    print("  - old training uses ambiguous-only contexts with max_nc=32")
    print("  - mixed training uses p_revealing=0.5 and max_nc=64")
    print("  - validation and test generators are identical across groups")
    print(
        "checkpoint_epoch:",
        {name: value["checkpoint_epoch"] for name, value in manifests.items()},
    )
    print(
        "checkpoint_global_step:",
        {name: value["checkpoint_global_step"] for name, value in manifests.items()},
    )


if __name__ == "__main__":
    main()
