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
        "repo_branch": git_value(["git", "branch", "--show-current"]),
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
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
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
    metadata.setdefault("training_schedule", None)
    metadata.setdefault("training_context", None)
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
    """Load learned sources; keep sampled baselines as model=None."""
    loaded: List[Dict[str, Any]] = []

    for entry in entries:
        kind = str(entry.get("kind", "model"))
        if kind == "uniform":
            loaded.append(
                {
                    "entry": entry,
                    "model": None,
                    "resolved_model_config": None,
                }
            )
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


def validate_sawtooth_batch(
    *,
    batch: Any,
    min_freq: float,
    max_freq: float,
    noise_std: float,
) -> None:
    predictor = getattr(batch, "gt_pred", None)
    required = ("freq", "direction", "offset", "latent_function")
    if predictor is None or any(not hasattr(predictor, name) for name in required):
        raise TypeError(
            "Sawtooth evaluation requires SawtoothGroundTruthPredictor metadata."
        )

    frequency = predictor.freq.detach().cpu().reshape(-1)
    direction = predictor.direction.detach().cpu().reshape(
        predictor.direction.shape[0], -1
    )

    tolerance = 1.0e-6
    if float(frequency.min()) < float(min_freq) - tolerance:
        raise RuntimeError("Sawtooth frequency fell below configured support.")
    if float(frequency.max()) > float(max_freq) + tolerance:
        raise RuntimeError("Sawtooth frequency exceeded configured support.")
    if not torch.all((direction == -1) | (direction == 1)):
        raise RuntimeError("Sawtooth direction is not restricted to {-1,+1}.")
    if abs(float(predictor.noise_std) - float(noise_std)) > tolerance:
        raise RuntimeError(
            f"Unexpected noise_std={predictor.noise_std}; expected {noise_std}."
        )

    values = torch.cat(
        [batch.yc.detach().cpu().reshape(-1), batch.yt.detach().cpu().reshape(-1)]
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Sawtooth observations contain non-finite values.")
    if float(values.min()) < -1.0e-5 or float(values.max()) > 1.0 + 1.0e-5:
        raise RuntimeError(
            "Noise-free sawtooth observations should lie in [0,1]. "
            f"Observed range=({float(values.min())}, {float(values.max())})."
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
