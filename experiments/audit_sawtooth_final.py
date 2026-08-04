from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch
from omegaconf import OmegaConf

from tnp_crps.utils.crps import crps_loss


CONFIG_ROOT = Path("experiments/configs/evaluation/sawtooth")
CONTINUOUS_CONFIGS = (
    Path("experiments/configs/gaussian_fourier_continuous.yml"),
    Path("experiments/configs/dropout_fourier_continuous.yml"),
    Path("experiments/configs/stochln_fourier_continuous.yml"),
)
CURRICULUM_GENERATOR = Path(
    "experiments/configs/generators/sawtooth_1d_interpolation_curriculum.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the final sawtooth evaluation, trajectory and AR protocol."
    )
    parser.add_argument(
        "--marginals_config",
        default=str(CONFIG_ROOT / "sawtooth_marginals_final.yml"),
    )
    parser.add_argument(
        "--alpha_config",
        default=str(CONFIG_ROOT / "sawtooth_alpha_ablation_final.yml"),
    )
    parser.add_argument(
        "--ar_config",
        default=str(CONFIG_ROOT / "sawtooth_ar_final.yml"),
    )
    parser.add_argument(
        "--trajectories_config",
        default=str(CONFIG_ROOT / "sawtooth_trajectories_final.yml"),
    )
    parser.add_argument("--output", default=None, type=str)
    return parser.parse_args()


def _load(path: str | Path) -> Dict[str, Any]:
    resolved = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected {path!s} to resolve to a dictionary.")
    return resolved


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}.")


def _checkpoint_metadata(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }


def _checkpoint_rows(configs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for config in configs:
        entries: List[Mapping[str, Any]] = []
        entries.extend(config.get("sources", []) or [])
        entries.extend(config.get("models", []) or [])
        entries.extend(config.get("runs", []) or [])
        for entry in entries:
            path = entry.get("checkpoint_path")
            if not path or str(path) in seen:
                continue
            seen.add(str(path))
            row = _checkpoint_metadata(str(path))
            row["source_name"] = str(entry.get("name", "<unnamed>"))
            rows.append(row)
    return rows


def _audit_crps_parameterisation() -> Dict[str, float]:
    # A deterministic tensor verifies that alpha=1 is fair, alpha=0 ordinary,
    # and intermediate alpha is the stated convex combination.
    samples = torch.tensor([0.0, 0.4, 0.7, 1.0], dtype=torch.float64).view(4, 1, 1, 1)
    target = torch.tensor([[[0.25]]], dtype=torch.float64)
    fair = float(crps_loss(samples, target, alpha=1.0))
    ordinary = float(crps_loss(samples, target, alpha=0.0))
    alpha = 0.05
    almost = float(crps_loss(samples, target, alpha=alpha))
    expected = alpha * fair + (1.0 - alpha) * ordinary
    if not math.isclose(almost, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError("crps_alpha does not implement the documented interpolation.")
    retention = alpha + (1.0 - alpha) * (4.0 - 1.0) / 4.0
    return {
        "fair_example": fair,
        "ordinary_example": ordinary,
        "almost_fair_example": almost,
        "almost_fair_expected": expected,
        "m4_alpha005_spread_retention": retention,
    }


def main() -> None:
    args = parse_args()
    marginal = _load(args.marginals_config)
    alpha = _load(args.alpha_config)
    ar = _load(args.ar_config)
    trajectories = _load(args.trajectories_config)

    # Final one-shot protocol.
    _assert_equal(int(marginal["samples_per_eval_set"]), 80_000, "Marginal task count")
    _assert_equal(int(marginal["num_eval_samples"]), 256, "Marginal sample count")
    _assert_equal(int(marginal["sample_chunk_size"]), 64, "Marginal sample chunk")
    _assert_equal(int(marginal["test_generator"]["min_nc"]), 48, "Marginal min Nc")
    _assert_equal(int(marginal["test_generator"]["max_nc"]), 64, "Marginal max Nc")

    marginal_names = [str(item["name"]) for item in marginal["sources"]]
    _assert_equal(
        marginal_names,
        [
            "Trivial U(0,1)",
            "Gaussian curriculum",
            "Dropout curriculum",
            "StochLN curriculum",
            "Dropout from scratch Nc48-64",
        ],
        "Marginal source order",
    )

    # Alpha diagnostics: the deliberately absent comparison is load-bearing.
    _assert_equal(int(alpha["samples_per_eval_set"]), 4_096, "Alpha task count")
    _assert_equal(int(alpha["test_generator"]["min_nc"]), 14, "Alpha test Nc")
    _assert_equal(int(alpha["test_generator"]["max_nc"]), 14, "Alpha test Nc")
    alpha_names = [str(item["name"]) for item in alpha["sources"]]
    if any("variable alpha0.95" in name for name in alpha_names):
        raise AssertionError("A nonexistent variable-context alpha=0.95 run was configured.")
    required_alpha_names = {
        "Dropout variable alpha1",
        "Dropout variable alpha0.05",
        "Dropout pretrain-finetune alpha1",
        "Dropout pretrain-finetune alpha0.95",
    }
    if not required_alpha_names.issubset(alpha_names):
        raise AssertionError("Alpha config is missing a verified diagnostic source.")

    # AR protocol.
    _assert_equal(str(ar["evaluation_mode"]), "ar_anchors", "AR evaluation mode")
    _assert_equal(int(ar["num_tasks"]), 4_096, "AR task count")
    _assert_equal(int(ar["num_ar_samples"]), 50, "AR rollout count")
    _assert_equal(int(ar["eval_nc"]), 48, "AR initial Nc")
    _assert_equal(int(ar["num_ar_anchors"]), 16, "AR anchor count")
    _assert_equal(int(ar["training_max_nc"]), 64, "AR supported final Nc")
    _assert_equal(str(ar["target_order"]), "random", "AR target order")
    _assert_equal(str(ar["stochln_noise_mode"]), "refresh", "AR StochLN mode")

    # Curriculum stages and constant hard validation support.
    expected_stages = [
        {"name": "stage1_nc64", "start_epoch": 0, "min_nc": 64, "max_nc": 64},
        {"name": "stage2_nc56_64", "start_epoch": 200, "min_nc": 56, "max_nc": 64},
        {"name": "stage3_nc48_64", "start_epoch": 350, "min_nc": 48, "max_nc": 64},
    ]
    for path in CONTINUOUS_CONFIGS:
        config = _load(path)
        _assert_equal(int(config["params"]["epochs"]), 500, f"Epochs in {path}")
        stages = config["misc"]["context_curriculum"]["stages"]
        _assert_equal(stages, expected_stages, f"Curriculum stages in {path}")

    generator = _load(CURRICULUM_GENERATOR)
    for split in ("train", "val", "test"):
        spec = generator["generators"][split]
        _assert_equal(float(spec["min_freq"]), 2.0, f"{split} min frequency")
        _assert_equal(float(spec["max_freq"]), 4.0, f"{split} max frequency")
        _assert_equal(float(spec["noise_std"]), 0.0, f"{split} observation noise")
        _assert_equal(int(spec["min_nt"]), 100, f"{split} target count")
        _assert_equal(int(spec["max_nt"]), 100, f"{split} target count")
    _assert_equal(int(generator["generators"]["val"]["min_nc"]), 48, "Validation min Nc")
    _assert_equal(int(generator["generators"]["val"]["max_nc"]), 64, "Validation max Nc")

    trajectory_names = [str(item["name"]) for item in trajectories["runs"]]
    if "Gaussian raw x" not in trajectory_names or "Gaussian Fourier" not in trajectory_names:
        raise AssertionError("Representation controls are missing from trajectory config.")

    checkpoint_rows = _checkpoint_rows((marginal, alpha, ar, trajectories))
    if not checkpoint_rows:
        raise AssertionError("No sawtooth checkpoints were audited.")

    # All from-scratch and continuous-curriculum runs use 500 epochs and
    # 500,000 updates. The deterministic-pretrain -> CRPS fine-tune runs use
    # 50 fine-tuning epochs and 50,000 updates.
    for row in checkpoint_rows:
        path = str(row["path"])
        is_pretrain_finetune = (
            "sawtooth-dropout-fourier-curriculum/" in path
            or "sawtooth-dropout-fourier-curriculum-alpha095/" in path
        )
        expected_epoch = 49 if is_pretrain_finetune else 499
        expected_step = 50_000 if is_pretrain_finetune else 500_000
        _assert_equal(row["epoch"], expected_epoch, f"Checkpoint epoch for {path}")
        _assert_equal(
            row["global_step"], expected_step, f"Checkpoint global step for {path}"
        )

    crps_audit = _audit_crps_parameterisation()
    report = {
        "marginals_config": args.marginals_config,
        "alpha_config": args.alpha_config,
        "ar_config": args.ar_config,
        "trajectories_config": args.trajectories_config,
        "curriculum_stages": expected_stages,
        "constant_validation_support": [48, 64],
        "checkpoint_metadata": checkpoint_rows,
        "crps_parameterisation": crps_audit,
    }

    print("PASS: final sawtooth protocol audit")
    print("  - raw-coordinate and Fourier Gaussian controls are configured")
    print("  - final marginal evaluation is 80,000 paired tasks at M=256, chunk=64")
    print("  - alpha diagnostic contains no fabricated variable-context alpha=0.95 row")
    print("  - AR evaluation is 4,096 tasks, 50 paths, Nc 48 -> 64")
    print("  - curriculum transitions are epochs 200 and 350")
    print("  - validation support remains [48,64] throughout curriculum training")
    print("  - almost-fair CRPS parameterisation is verified")
    print(f"  - audited {len(checkpoint_rows)} unique checkpoints")
    print(json.dumps(crps_audit, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
