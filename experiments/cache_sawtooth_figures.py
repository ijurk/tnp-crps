from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp_crps.data.sawtooth import SawtoothGroundTruthPredictor

from evaluate_synthetic_1d import move_batch_to_device
from evaluation.autoregressive import autoregressive_sample_model
from evaluation.sawtooth_final_utils import load_sources, runtime_metadata
from plot_synthetic_1d_ar_functions import denoise_ar_samples_in_chunks


CHOICES = ("trajectories", "ar", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen CPU caches for the sawtooth trajectory and AR "
            "dissertation figures. The notebook performs no model inference."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--figure", default="all", choices=CHOICES)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: str | Path) -> Dict[str, Any]:
    resolved = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected {path!s} to resolve to a dictionary.")
    return resolved


def _set_seed(seed: int) -> None:
    pl.seed_everything(int(seed), workers=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def _save(cache: Dict[str, Any], path: Path) -> None:
    cache = _cpu_tree(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    sidecar = {
        "cache_path": str(path),
        "cache_sha256": _sha256(path),
        "schema_version": cache["schema_version"],
        "metadata": cache["metadata"],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def _output_path(
    *, output_dir: Path, name: str, smoke: bool, overwrite: bool
) -> Path:
    source = Path(name)
    if source.suffix != ".pt":
        raise ValueError("Cache output_name must end in .pt.")
    final_name = f"{source.stem}_smoke.pt" if smoke else source.name
    path = output_dir / final_name
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Cache already exists: {path}. Use a new version or --overwrite deliberately."
        )
    return path


def _trajectory_cache(spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    history_path = Path(str(spec["history_csv"]))
    summary_path = Path(str(spec["summary_csv"]))
    if not history_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Trajectory CSVs are missing. Run export_sawtooth_trajectories.py first."
        )
    history = pd.read_csv(history_path)
    summary = pd.read_csv(summary_path)

    required_history = {
        "model_name",
        "panel",
        "epoch_resolved",
        "val_rmse",
        "stage_name",
        "stage_min_nc",
        "stage_max_nc",
    }
    if not required_history.issubset(history.columns):
        raise RuntimeError(
            f"Trajectory history is missing {sorted(required_history.difference(history.columns))}."
        )
    if history.empty or summary.empty:
        raise RuntimeError("Trajectory inputs are empty.")

    records: Dict[str, Dict[str, Any]] = {}
    for model_name, group in history.groupby("model_name", sort=False):
        group = group.sort_values("epoch_resolved", kind="stable")
        records[str(model_name)] = {
            "panel": str(group["panel"].iloc[0]),
            "epoch": torch.as_tensor(group["epoch_resolved"].to_numpy(), dtype=torch.float32),
            "val_rmse": torch.as_tensor(group["val_rmse"].to_numpy(), dtype=torch.float32),
            "val_crps": torch.as_tensor(group["val_crps"].to_numpy(), dtype=torch.float32),
            "stage_name": group["stage_name"].astype(str).tolist(),
            "stage_min_nc": torch.as_tensor(group["stage_min_nc"].to_numpy(), dtype=torch.int64),
            "stage_max_nc": torch.as_tensor(group["stage_max_nc"].to_numpy(), dtype=torch.int64),
        }

    return {
        "schema_version": "sawtooth_trajectory_cache_v1",
        "metadata": {
            **dict(metadata),
            "history_csv": str(history_path),
            "summary_csv": str(summary_path),
            "stage_boundaries": [200, 350],
            "validation_support": [48, 64],
        },
        "records": records,
        "summary": summary.to_dict(orient="records"),
    }


def _slice_predictor(predictor: Any, local_index: int) -> SawtoothGroundTruthPredictor:
    return SawtoothGroundTruthPredictor(
        freq=predictor.freq.reshape(-1)[local_index : local_index + 1],
        direction=predictor.direction[local_index : local_index + 1],
        offset=predictor.offset.reshape(-1)[local_index : local_index + 1],
        noise_std=float(predictor.noise_std),
        jitter=float(predictor.jitter),
    )


def _slice_batch(batch: Any, local_index: int) -> Any:
    values: Dict[str, Any] = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            value = value[local_index : local_index + 1]
        values[field.name] = value
    values["gt_pred"] = _slice_predictor(batch.gt_pred, local_index)
    return type(batch)(**values)


def _selected_task_batches(
    *, ar_cfg: Mapping[str, Any], selected_tasks: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    task_ids = {
        str(label): int(item["task_id"])
        for label, item in dict(selected_tasks["tasks"]).items()
    }
    eval_batch_size = int(ar_cfg["eval_batch_size"])
    required_batches = {task_id // eval_batch_size for task_id in task_ids.values()}
    max_batch = max(required_batches)

    base_config = OmegaConf.load(str(ar_cfg["base_generator_config"]))
    fixed = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
    fixed.generators.test.min_nc = int(ar_cfg["eval_nc"])
    fixed.generators.test.max_nc = int(ar_cfg["eval_nc"])
    fixed.generators.test.min_nt = 1
    fixed.generators.test.max_nt = 1
    fixed.generators.test.samples_per_epoch = (max_batch + 1) * eval_batch_size
    fixed.generators.test.batch_size = eval_batch_size
    fixed.generators.test.deterministic = True

    generator = instantiate(OmegaConf.to_container(fixed.generators.test, resolve=True))
    loader = torch.utils.data.DataLoader(generator, batch_size=None, num_workers=0)
    result: Dict[str, Dict[str, Any]] = {}
    anchor_seed = int(ar_cfg["anchor_seed"])
    num_anchors = int(ar_cfg["num_ar_anchors"])
    anchor_min, anchor_max = [float(value) for value in ar_cfg["ar_anchor_range"]]

    for batch_index, batch in enumerate(loader):
        if batch_index > max_batch:
            break
        labels = [
            label
            for label, task_id in task_ids.items()
            if task_id // eval_batch_size == batch_index
        ]
        if not labels:
            continue

        anchor_generator = torch.Generator(device="cpu")
        anchor_generator.manual_seed(anchor_seed + batch_index)
        unit = torch.rand(
            eval_batch_size,
            num_anchors,
            1,
            generator=anchor_generator,
            dtype=torch.float32,
        )
        anchors = torch.sort(anchor_min + (anchor_max - anchor_min) * unit, dim=1).values

        for label in labels:
            task_id = task_ids[label]
            local_index = task_id % eval_batch_size
            selected = _slice_batch(batch, local_index)
            result[label] = {
                "task_id": task_id,
                "batch_index": batch_index,
                "local_index": local_index,
                "batch": selected,
                "anchors": anchors[local_index : local_index + 1],
            }

    if set(result) != set(task_ids):
        raise RuntimeError("Failed to reconstruct every selected AR task.")
    return result


def _release(sources: Iterable[Dict[str, Any]]) -> None:
    for item in sources:
        item["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ar_cache(
    *, spec: Mapping[str, Any], metadata: Mapping[str, Any], device: torch.device, smoke: bool
) -> Dict[str, Any]:
    ar_cfg = _load(str(spec["ar_config"]))
    selected = json.loads(Path(str(spec["selected_tasks_json"])).read_text())
    tasks = _selected_task_batches(ar_cfg=ar_cfg, selected_tasks=selected)

    num_paths = int(spec["num_paths_smoke"] if smoke else spec["num_paths"])
    dense_points = int(spec["num_dense_points_smoke"] if smoke else spec["num_dense_points"])
    denoise_samples = int(
        spec["num_denoise_samples_smoke"] if smoke else spec["num_denoise_samples"]
    )
    denoise_chunk = int(spec["denoise_chunk_size"])
    figure_seed = int(spec["sampling_seed"])

    sources = load_sources(
        entries=[
            {**dict(item), "kind": "model"}
            for item in ar_cfg["models"]
        ],
        base_generator_config=str(ar_cfg["base_generator_config"]),
        device=device,
    )

    x_dense = torch.linspace(
        float(spec["x_range"][0]),
        float(spec["x_range"][1]),
        dense_points,
        device=device,
        dtype=torch.float32,
    ).view(1, dense_points, 1)

    task_cache: Dict[str, Any] = {}
    for task_position, (label, item) in enumerate(tasks.items()):
        batch_cpu = item["batch"]
        batch = move_batch_to_device(batch_cpu, device)
        x_anchor = item["anchors"].to(device=device, dtype=batch.xc.dtype)
        y_anchor = batch.gt_pred.latent_function(x_anchor).to(device=device, dtype=batch.yc.dtype)
        ar_batch = dataclasses.replace(batch, xt=x_anchor, yt=y_anchor)
        query = x_dense.to(dtype=batch.xc.dtype)
        truth = batch.gt_pred.latent_function(query)
        models: Dict[str, Any] = {}

        for source_index, source in enumerate(sources):
            seed = figure_seed + 1_000_000 * (source_index + 1) + 10_000 * task_position
            _set_seed(seed)
            raw = autoregressive_sample_model(
                model=source["model"],
                batch=ar_batch,
                num_samples=num_paths,
                target_order=str(ar_cfg["target_order"]),
                stochln_noise_mode=str(ar_cfg["stochln_noise_mode"]),
            )
            denoised = denoise_ar_samples_in_chunks(
                model=source["model"],
                ar_batch=ar_batch,
                raw_samples=raw,
                query_xt=query,
                num_denoise_samples=denoise_samples,
                chunk_size=denoise_chunk,
            )
            models[str(source["entry"]["name"])] = {
                "raw_ar_samples": raw,
                "ar_denoised_paths": denoised,
            }

        task_cache[label] = {
            "task_id": int(item["task_id"]),
            "difficulty": selected["tasks"][label],
            "xc": batch.xc,
            "yc": batch.yc,
            "x_anchor": x_anchor,
            "y_anchor_truth": y_anchor,
            "x_plot": query,
            "latent_truth": truth,
            "frequency": batch.gt_pred.freq,
            "direction": batch.gt_pred.direction,
            "offset": batch.gt_pred.offset,
            "models": models,
        }

    _release(sources)
    return {
        "schema_version": "sawtooth_ar_figure_cache_v1",
        "metadata": {
            **dict(metadata),
            "ar_config": str(spec["ar_config"]),
            "selected_tasks_json": str(spec["selected_tasks_json"]),
            "smoke": bool(smoke),
            "num_paths": num_paths,
            "num_dense_points": dense_points,
            "num_denoise_samples": denoise_samples,
            "denoise_chunk_size": denoise_chunk,
            "target_order": str(ar_cfg["target_order"]),
            "stochln_noise_mode": str(ar_cfg["stochln_noise_mode"]),
            "num_ar_anchors": int(ar_cfg["num_ar_anchors"]),
            "initial_nc": int(ar_cfg["eval_nc"]),
            "final_nc": int(ar_cfg["eval_nc"]) + int(ar_cfg["num_ar_anchors"]),
        },
        "task_selection": selected,
        "tasks": task_cache,
    }


def main() -> None:
    args = parse_args()
    cfg = _load(args.config)
    output_dir = Path(args.output_dir or cfg["output_dir"])
    device_name = args.device or str(cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)
    runtime = runtime_metadata(device)
    metadata = {
        "config_path": str(args.config),
        "runtime_metadata": runtime,
    }
    selected: Sequence[str] = (
        ("trajectories", "ar") if args.figure == "all" else (args.figure,)
    )

    for figure in selected:
        spec = dict(cfg[figure])
        path = _output_path(
            output_dir=output_dir,
            name=str(spec["output_name"]),
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
        if figure == "trajectories":
            cache = _trajectory_cache(spec, metadata)
            cache["metadata"]["smoke"] = bool(args.smoke)
        else:
            cache = _ar_cache(
                spec=spec,
                metadata=metadata,
                device=device,
                smoke=args.smoke,
            )
        _save(cache, path)
        print(f"CACHE PASS [{figure.upper()}]: {path}")

    print("ALL SELECTED SAWTOOTH FIGURE CACHES COMPLETED.")


if __name__ == "__main__":
    main()
