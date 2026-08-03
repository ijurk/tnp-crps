from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluate_synthetic_1d import load_merged_config, load_model_state


def git_value(args: List[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def runtime_metadata(device: torch.device) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "created_unix_time": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "repo_commit": git_value(["git", "rev-parse", "HEAD"]),
        "tnp_submodule_commit": git_value(
            ["git", "-C", "external/tnp", "rev-parse", "HEAD"]
        ),
        "git_status_short": git_value(["git", "status", "--short"]),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
            }
        )
    return metadata


def prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}. "
            "Use a new versioned path or pass --overwrite deliberately."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def source_metadata(entry: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(entry.get("metadata", {}) or {})
    metadata.setdefault("training_group", None)
    metadata.setdefault("training_alpha", None)
    metadata.setdefault("training_num_samples", None)
    metadata.setdefault("training_p_dropout", None)
    metadata.setdefault("training_layernorm_noise_dim", None)
    return metadata


def load_sources(
    *,
    entries: List[Dict[str, Any]],
    base_generator_config: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    for entry in entries:
        kind = str(entry.get("kind", "model"))
        if kind == "oracle":
            loaded.append({"entry": entry, "model": None})
            continue
        if kind != "model":
            raise ValueError(
                f"Unsupported source kind={kind!r} for {entry.get('name')!r}."
            )
        checkpoint_path = str(entry["checkpoint_path"])
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        config = load_merged_config(
            config_paths=[base_generator_config, str(entry["model_config"])],
            overrides=list(entry.get("overrides", []) or []),
        )
        pl.seed_everything(int(config.misc.seed), workers=False)
        model = instantiate(config.model)
        load_model_state(model, checkpoint_path)
        model.to(device)
        model.eval()
        loaded.append(
            {
                "entry": entry,
                "model": model,
                "resolved_model_config": OmegaConf.to_container(
                    config, resolve=True
                ),
            }
        )
        print(f"Loaded source={entry['name']} checkpoint={checkpoint_path}")
    return loaded


def validate_binary_fork_batch(
    *,
    batch: Any,
    expected_fork_x0: float,
    expected_delta: float,
    expected_noise_std: float,
    require_ambiguous_context: bool,
) -> None:
    predictor = getattr(batch, "gt_pred", None)
    required = (
        "posterior_marginal_components",
        "predictive_marginal_samples",
        "sample_paired_regime_observations",
    )
    if predictor is None or any(not hasattr(predictor, name) for name in required):
        raise TypeError(
            "Binary-fork evaluation requires the updated "
            "BinaryLatentForkGroundTruthPredictor."
        )
    fork = float(predictor.fork_locations.reshape(-1)[0])
    if abs(fork - float(expected_fork_x0)) > 1.0e-7:
        raise RuntimeError(f"Unexpected fork location: {fork}.")
    if abs(float(predictor.delta) - float(expected_delta)) > 1.0e-7:
        raise RuntimeError(f"Unexpected delta: {predictor.delta}.")
    if abs(float(predictor.noise_std) - float(expected_noise_std)) > 1.0e-7:
        raise RuntimeError(f"Unexpected noise_std: {predictor.noise_std}.")
    if require_ambiguous_context:
        max_context = float(batch.xc[..., 0].max().item())
        if max_context > float(expected_fork_x0) + 1.0e-7:
            raise RuntimeError(
                "Ambiguous-context evaluation received a post-fork context point: "
                f"max_xc={max_context}."
            )


def write_resolved_config(
    *,
    path: Path,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    resolved = dict(config)
    resolved.update(dict(overrides))
    resolved["runtime_metadata"] = dict(runtime)
    path.write_text(json.dumps(resolved, indent=2))
