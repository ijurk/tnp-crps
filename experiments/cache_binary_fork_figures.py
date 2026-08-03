from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.synthetic import SyntheticBatch

from evaluate_binary_fork_conditioning import _paired_variants
from evaluate_synthetic_1d import (
    apply_eval_dataset_overrides,
    load_merged_config,
    move_batch_to_device,
)
from evaluation.autoregressive import (
    autoregressive_sample_model,
    denoise_autoregressive_samples,
)
from evaluation.binary_fork_utils import (
    load_sources,
    runtime_metadata,
    validate_binary_fork_batch,
)
from evaluation.gaussian_controls_metrics import task_fingerprints
from evaluation.predictive_sampling import (
    sample_model_chunked,
    sampling_seed,
    validate_sampling_offsets,
)


FIGURE_CHOICES = (
    "marginals",
    "conditioning",
    "coherence",
    "all",
)

EXPECTED_MIXED_MODELS = (
    "Gaussian TNP",
    "Dropout CRPS-TNP",
    "StochLN CRPS-TNP",
)

EXPECTED_OLD_MODELS = (
    "Old Gaussian TNP",
    "Old Dropout CRPS-TNP",
    "Old StochLN CRPS-TNP",
)

EXPECTED_MIXED_CONDITIONING_MODELS = (
    "Mixed Gaussian TNP",
    "Mixed Dropout CRPS-TNP",
    "Mixed StochLN CRPS-TNP",
)


# -----------------------------------------------------------------------------
# CLI and configuration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen CPU caches for the three binary-fork dissertation "
            "figures. The notebook performs no model inference."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--figure",
        default="all",
        choices=FIGURE_CHOICES,
        help="Cache one figure or all three figures.",
    )
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use reduced sample/path/grid counts and write *_smoke.pt files. "
            "The final caches are unaffected."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deliberately replace an existing selected cache file.",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = OmegaConf.to_container(
        OmegaConf.load(str(path)),
        resolve=True,
    )
    if not isinstance(resolved, dict):
        raise TypeError(
            f"Expected {path!s} to resolve to a dictionary, got {type(resolved)}."
        )
    return resolved


def _selected_figures(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return (
            "marginals",
            "conditioning",
            "coherence",
        )
    return (requested,)


def _apply_smoke_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    out["marginals"].update(
        {
            "num_samples": 128,
            "sample_chunk_size": 64,
        }
    )

    out["conditioning"].update(
        {
            "num_grid_points": 41,
            "num_samples": 128,
            "sample_chunk_size": 64,
        }
    )

    out["coherence"].update(
        {
            "num_dense_points": 101,
            "num_direct_samples": 8,
            "direct_chunk_size": 8,
            "num_ar_paths": 8,
            "num_denoise_samples": 4,
            "denoise_chunk_size": 4,
        }
    )

    return out


def _output_name(spec: Mapping[str, Any], *, smoke: bool) -> str:
    name = str(spec["output_name"])
    path = Path(name)
    if path.suffix != ".pt":
        raise ValueError(
            f"Figure output_name must end in .pt, got {name!r}."
        )
    if smoke:
        return f"{path.stem}_smoke.pt"
    return name


def _preflight_outputs(
    *,
    output_dir: Path,
    cfg: Mapping[str, Any],
    selected: Sequence[str],
    smoke: bool,
    overwrite: bool,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        figure_name: output_dir
        / _output_name(cfg[figure_name], smoke=smoke)
        for figure_name in selected
    }

    existing = [
        path
        for path in outputs.values()
        if path.exists()
    ]
    if existing and not overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing
        )
        raise FileExistsError(
            "Selected cache file(s) already exist:\n"
            f"{formatted}\n"
            "Use new versioned names or pass --overwrite deliberately."
        )
    return outputs


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    pl.seed_everything(int(seed), workers=False)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"{name} contains non-finite values."
        )


def _release_sources(loaded_sources: Iterable[Dict[str, Any]]) -> None:
    for item in loaded_sources:
        item["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_cache(
    *,
    cache: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    digest = _sha256(output_path)

    sidecar = {
        "cache_path": str(output_path),
        "cache_sha256": digest,
        "schema_version": cache["schema_version"],
        "metadata": cache["metadata"],
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            sidecar,
            indent=2,
        )
    )

    print(f"Wrote cache: {output_path}")
    print(f"Wrote metadata: {sidecar_path}")
    print(f"SHA256: {digest}")


def _make_fixed_context_task(
    *,
    base_generator_config: str,
    context_size: int,
    deterministic_seed: int,
    qualifying_task_index: int,
    search_batches: int,
) -> tuple[SyntheticBatch, int]:
    if context_size < 1:
        raise ValueError(
            f"context_size must be positive, got {context_size}."
        )
    if qualifying_task_index < 0:
        raise ValueError(
            "qualifying_task_index cannot be negative."
        )
    if search_batches <= qualifying_task_index:
        raise ValueError(
            "search_batches must exceed qualifying_task_index."
        )

    generator_cfg = load_merged_config(
        config_paths=[base_generator_config]
    )
    apply_eval_dataset_overrides(
        generator_cfg,
        samples_per_eval_set=int(search_batches),
        eval_batch_size=1,
    )
    generator_cfg.generators.test.min_nc = int(context_size)
    generator_cfg.generators.test.max_nc = int(context_size)
    generator_cfg.generators.test.deterministic = True
    generator_cfg.generators.test.deterministic_seed = int(
        deterministic_seed
    )

    _set_seed(deterministic_seed)
    generator = instantiate(generator_cfg.generators.test)
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    for batch_index, batch in enumerate(loader):
        if batch_index < qualifying_task_index:
            continue
        if not isinstance(batch, SyntheticBatch):
            raise TypeError(
                f"Expected SyntheticBatch, got {type(batch)}."
            )
        if int(batch.xc.shape[0]) != 1:
            raise RuntimeError(
                "Figure-cache generation requires batch size one."
            )
        if int(batch.xc.shape[1]) != int(context_size):
            raise RuntimeError(
                "Generator did not honour the fixed context size: "
                f"expected {context_size}, got {batch.xc.shape[1]}."
            )
        return batch, batch_index

    raise RuntimeError(
        "Could not select the requested deterministic plotting task."
    )


def _validate_generator_identity(
    *,
    batch: SyntheticBatch,
    evaluation_cfg: Mapping[str, Any],
) -> None:
    validate_binary_fork_batch(
        batch=batch,
        expected_fork_x0=float(evaluation_cfg["fork_x0"]),
        expected_delta=float(evaluation_cfg["delta"]),
        expected_noise_std=float(evaluation_cfg["noise_std"]),
        require_ambiguous_context=True,
    )


def _model_entries(
    source_entries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        dict(entry)
        for entry in source_entries
        if str(entry.get("kind", "model")) == "model"
    ]


def _validate_exact_names(
    *,
    entries: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
    label: str,
) -> None:
    names = tuple(str(entry["name"]) for entry in entries)
    if names != tuple(expected):
        raise RuntimeError(
            f"Unexpected {label} source order.\n"
            f"Expected: {tuple(expected)}\n"
            f"Found:    {names}"
        )


def _validate_mixed_checkpoint_parity(
    *,
    marginal_cfg: Mapping[str, Any],
    coherence_cfg: Mapping[str, Any],
) -> None:
    marginal_entries = _model_entries(marginal_cfg["sources"])
    coherence_entries = _model_entries(coherence_cfg["sources"])
    _validate_exact_names(
        entries=marginal_entries,
        expected=EXPECTED_MIXED_MODELS,
        label="T1 learned",
    )
    _validate_exact_names(
        entries=coherence_entries,
        expected=EXPECTED_MIXED_MODELS,
        label="T3 learned",
    )

    for left, right in zip(marginal_entries, coherence_entries):
        fields = (
            "name",
            "model_config",
            "checkpoint_path",
            "overrides",
        )
        for field in fields:
            if left.get(field) != right.get(field):
                raise RuntimeError(
                    "T1 and T3 mixed-model definitions differ for "
                    f"source={left['name']!r}, field={field!r}."
                )


def _checkpoint_manifest(
    source_entries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(entry["name"]),
            "kind": str(entry.get("kind", "model")),
            "checkpoint_path": entry.get("checkpoint_path"),
            "model_config": entry.get("model_config"),
            "overrides": list(entry.get("overrides", []) or []),
            "sampling_seed_offset": int(
                entry["sampling_seed_offset"]
            ),
        }
        for entry in source_entries
    ]


def _exact_upper_probability(
    *,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    regime_weights: torch.Tensor,
) -> torch.Tensor:
    """Probability that Y lies above the component midpoint.

    component_means/component_scales: [B, 2, N, 1]
    regime_weights: [B, 2]
    returns: [B, N, 1]
    """
    midpoint = 0.5 * (
        component_means[:, 0]
        + component_means[:, 1]
    )
    z = (
        midpoint[:, None]
        - component_means
    ) / component_scales.clamp_min(1.0e-8)
    component_tail = 0.5 * torch.erfc(
        z / math.sqrt(2.0)
    )
    probability = (
        regime_weights[:, :, None, None]
        * component_tail
    ).sum(dim=1)
    _assert_finite(
        "exact upper-branch probability",
        probability,
    )
    return probability


@torch.no_grad()
def _denoise_in_chunks(
    *,
    model: torch.nn.Module,
    ar_batch: SyntheticBatch,
    raw_samples: torch.Tensor,
    query_xt: torch.Tensor,
    num_denoise_samples: int,
    chunk_size: int,
) -> torch.Tensor:
    if raw_samples.ndim != 4:
        raise ValueError(
            "Expected raw AR samples [M,B,K,D], "
            f"got {tuple(raw_samples.shape)}."
        )
    if chunk_size < 1:
        raise ValueError(
            f"denoise chunk_size must be positive, got {chunk_size}."
        )

    chunks = []
    for start in range(0, int(raw_samples.shape[0]), int(chunk_size)):
        end = min(
            start + int(chunk_size),
            int(raw_samples.shape[0]),
        )
        denoised = denoise_autoregressive_samples(
            model=model,
            batch=ar_batch,
            ar_samples=raw_samples[start:end],
            query_xt=query_xt,
            num_denoise_samples=int(num_denoise_samples),
        )
        chunks.append(denoised)

    out = torch.cat(chunks, dim=0).contiguous()
    _assert_finite("AR-conditioned mean paths", out)
    return out


# -----------------------------------------------------------------------------
# F1: ambiguous marginal recovery
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_marginal_cache(
    *,
    figure_cfg: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    smoke: bool,
) -> None:
    base_generator_config = str(
        evaluation_cfg["base_generator_config"]
    )
    source_entries = [
        dict(entry)
        for entry in evaluation_cfg["sources"]
    ]
    validate_sampling_offsets(source_entries)
    learned_entries = _model_entries(source_entries)
    _validate_exact_names(
        entries=learned_entries,
        expected=EXPECTED_MIXED_MODELS,
        label="F1 learned",
    )

    batch_cpu, selected_batch_index = _make_fixed_context_task(
        base_generator_config=base_generator_config,
        context_size=int(figure_cfg["context_size"]),
        deterministic_seed=int(figure_cfg["deterministic_seed"]),
        qualifying_task_index=int(
            figure_cfg["qualifying_task_index"]
        ),
        search_batches=int(figure_cfg["search_batches"]),
    )
    _validate_generator_identity(
        batch=batch_cpu,
        evaluation_cfg=evaluation_cfg,
    )

    probe_x_cpu = torch.tensor(
        list(figure_cfg["probe_x"]),
        dtype=batch_cpu.xc.dtype,
    ).view(1, -1, 1)
    probe_placeholder_cpu = torch.zeros(
        1,
        probe_x_cpu.shape[1],
        batch_cpu.yc.shape[-1],
        dtype=batch_cpu.yc.dtype,
    )
    probe_batch_cpu = dataclasses.replace(
        batch_cpu,
        xt=probe_x_cpu,
        yt=probe_placeholder_cpu,
    )
    probe_batch = move_batch_to_device(
        probe_batch_cpu,
        device,
    )

    predictor = batch_cpu.gt_pred
    if predictor is None:
        raise RuntimeError(
            "F1 task has no exact binary-fork predictor."
        )

    component_means, component_scales, regime_weights = (
        predictor.posterior_marginal_components(
            xc=probe_batch.xc,
            yc=probe_batch.yc,
            xt=probe_batch.xt,
            include_target_noise=True,
        )
    )
    _assert_finite("F1 component means", component_means)
    _assert_finite("F1 component scales", component_scales)
    _assert_finite("F1 regime weights", regime_weights)

    max_weight_deviation = float(
        (regime_weights - 0.5).abs().max().item()
    )
    if max_weight_deviation > 1.0e-7:
        raise RuntimeError(
            "The ambiguous F1 task does not have exact 0.5/0.5 regime "
            f"weights; max deviation={max_weight_deviation:.3e}."
        )

    loaded_sources = load_sources(
        entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    model_samples: Dict[str, torch.Tensor] = {}
    for loaded in loaded_sources:
        entry = loaded["entry"]
        if str(entry.get("kind", "model")) == "oracle":
            continue
        model = loaded["model"]
        if model is None:
            raise RuntimeError(
                f"F1 model did not load for {entry['name']!r}."
            )
        seed = sampling_seed(
            base_seed=int(figure_cfg["sampling_seed"]),
            source_offset=int(entry["sampling_seed_offset"]),
            batch_index=0,
            condition_index=0,
        )
        _set_seed(seed)
        samples = sample_model_chunked(
            model=model,
            batch=probe_batch,
            num_samples=int(figure_cfg["num_samples"]),
            chunk_size=int(figure_cfg["sample_chunk_size"]),
        )
        model_samples[str(entry["name"])] = (
            samples.detach().cpu()
        )
        print(
            f"F1 cached {entry['name']}: shape={tuple(samples.shape)}"
        )
        del samples

    metadata = {
        "figure_id": "F1",
        "figure_name": "ambiguous marginal recovery",
        "smoke": bool(smoke),
        "figure_config": dict(figure_cfg),
        "evaluation_config": str(
            Path(str(figure_cfg["evaluation_config"])).resolve()
        ),
        "base_generator_config": base_generator_config,
        "selected_generator_batch_index": int(
            selected_batch_index
        ),
        "task_fingerprint": task_fingerprints(batch_cpu)[0],
        "num_context": int(batch_cpu.xc.shape[1]),
        "max_ambiguous_weight_deviation": max_weight_deviation,
        "source_manifest": _checkpoint_manifest(source_entries),
        "runtime_metadata": runtime_metadata(device),
    }

    cache = {
        "schema_version": "binary_fork_marginals_cache_v1",
        "metadata": metadata,
        "task": {
            "xc": batch_cpu.xc.cpu(),
            "yc": batch_cpu.yc.cpu(),
            "probe_x": probe_x_cpu.cpu(),
        },
        "exact_marginal": {
            "component_means": component_means.detach().cpu(),
            "component_scales": component_scales.detach().cpu(),
            "regime_weights": regime_weights.detach().cpu(),
        },
        "models": model_samples,
    }

    _save_cache(
        cache=cache,
        output_path=output_path,
    )
    print(
        "CACHE PASS [F1]: exact 0.5/0.5 weights, finite oracle "
        "components, and all three learned sample tensors validated."
    )
    _release_sources(loaded_sources)


# -----------------------------------------------------------------------------
# F2: revealing-context conditioning
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_conditioning_cache(
    *,
    figure_cfg: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    smoke: bool,
) -> None:
    base_generator_config = str(
        evaluation_cfg["base_generator_config"]
    )
    source_entries = [
        dict(entry)
        for entry in evaluation_cfg["sources"]
    ]
    validate_sampling_offsets(source_entries)
    learned_entries = _model_entries(source_entries)
    _validate_exact_names(
        entries=learned_entries[:3],
        expected=EXPECTED_OLD_MODELS,
        label="F2 old",
    )
    _validate_exact_names(
        entries=learned_entries[3:],
        expected=EXPECTED_MIXED_CONDITIONING_MODELS,
        label="F2 mixed",
    )

    context_size = int(figure_cfg["context_size"])
    if context_size + 1 > int(
        evaluation_cfg["old_training_max_nc"]
    ):
        raise RuntimeError(
            "F2 context plus reveal exceeds the old checkpoints' "
            "training context-size support."
        )

    base_batch_cpu, selected_batch_index = _make_fixed_context_task(
        base_generator_config=base_generator_config,
        context_size=context_size,
        deterministic_seed=int(figure_cfg["deterministic_seed"]),
        qualifying_task_index=int(
            figure_cfg["qualifying_task_index"]
        ),
        search_batches=int(figure_cfg["search_batches"]),
    )
    _validate_generator_identity(
        batch=base_batch_cpu,
        evaluation_cfg=evaluation_cfg,
    )

    variants_cpu = _paired_variants(
        batch=base_batch_cpu,
        batch_index=0,
        counterfactual_seed=int(
            figure_cfg["counterfactual_seed"]
        ),
        inject_x=float(figure_cfg["inject_x"]),
    )
    condition_names = (
        "ambiguous",
        "upper_reveal",
    )

    if not torch.equal(
        variants_cpu["ambiguous"].xc,
        base_batch_cpu.xc,
    ):
        raise RuntimeError(
            "F2 ambiguous variant changed the base context inputs."
        )
    if not torch.allclose(
        variants_cpu["ambiguous"].yc,
        variants_cpu["upper_reveal"].yc[:, :-1],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError(
            "F2 paired ambiguous/revealing contexts do not share the "
            "same pre-fork observations."
        )

    x_grid_cpu = torch.linspace(
        float(figure_cfg["x_range"][0]),
        float(figure_cfg["x_range"][1]),
        int(figure_cfg["num_grid_points"]),
        dtype=base_batch_cpu.xc.dtype,
    ).view(1, -1, 1)

    loaded_sources = load_sources(
        entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    exact_curves: Dict[str, torch.Tensor] = {}
    exact_weights: Dict[str, torch.Tensor] = {}
    group_curves: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
        "old": {
            condition: {}
            for condition in condition_names
        },
        "mixed": {
            condition: {}
            for condition in condition_names
        },
    }

    for condition_index, condition in enumerate(condition_names):
        variant_cpu = variants_cpu[condition]
        placeholder_cpu = torch.zeros(
            1,
            x_grid_cpu.shape[1],
            variant_cpu.yc.shape[-1],
            dtype=variant_cpu.yc.dtype,
        )
        curve_batch_cpu = dataclasses.replace(
            variant_cpu,
            xt=x_grid_cpu,
            yt=placeholder_cpu,
        )
        curve_batch = move_batch_to_device(
            curve_batch_cpu,
            device,
        )

        predictor = variant_cpu.gt_pred
        if predictor is None:
            raise RuntimeError(
                f"F2 condition {condition!r} has no exact predictor."
            )
        component_means, component_scales, regime_weights = (
            predictor.posterior_marginal_components(
                xc=curve_batch.xc,
                yc=curve_batch.yc,
                xt=curve_batch.xt,
                include_target_noise=True,
            )
        )
        midpoint = 0.5 * (
            component_means[:, 0]
            + component_means[:, 1]
        )
        exact_probability = _exact_upper_probability(
            component_means=component_means,
            component_scales=component_scales,
            regime_weights=regime_weights,
        )
        exact_curves[condition] = (
            exact_probability.detach().cpu()
        )
        exact_weights[condition] = (
            regime_weights.detach().cpu()
        )

        for loaded in loaded_sources:
            entry = loaded["entry"]
            if str(entry.get("kind", "model")) == "oracle":
                continue
            source_name = str(entry["name"])
            if source_name.startswith("Old "):
                group = "old"
                display_name = source_name[len("Old ") :]
            elif source_name.startswith("Mixed "):
                group = "mixed"
                display_name = source_name[len("Mixed ") :]
            else:
                raise RuntimeError(
                    f"Unexpected F2 source name {source_name!r}."
                )

            model = loaded["model"]
            if model is None:
                raise RuntimeError(
                    f"F2 model did not load for {source_name!r}."
                )
            seed = sampling_seed(
                base_seed=int(figure_cfg["sampling_seed"]),
                source_offset=int(entry["sampling_seed_offset"]),
                batch_index=0,
                condition_index=condition_index,
            )
            _set_seed(seed)
            samples = sample_model_chunked(
                model=model,
                batch=curve_batch,
                num_samples=int(figure_cfg["num_samples"]),
                chunk_size=int(figure_cfg["sample_chunk_size"]),
            )
            probability = (
                samples > midpoint.unsqueeze(0)
            ).to(torch.float32).mean(dim=0)
            _assert_finite(
                f"F2 probability {source_name} {condition}",
                probability,
            )
            group_curves[group][condition][display_name] = (
                probability.detach().cpu()
            )
            print(
                f"F2 cached {source_name} | {condition}: "
                f"shape={tuple(probability.shape)}"
            )
            del samples
            del probability

    expected_display_names = set(EXPECTED_MIXED_MODELS)
    for group in ("old", "mixed"):
        for condition in condition_names:
            found = set(group_curves[group][condition])
            if found != expected_display_names:
                raise RuntimeError(
                    f"F2 cache keys mismatch for {group}/{condition}: "
                    f"expected={sorted(expected_display_names)}, "
                    f"found={sorted(found)}."
                )

    metadata = {
        "figure_id": "F2",
        "figure_name": "revealing-context conditioning",
        "smoke": bool(smoke),
        "figure_config": dict(figure_cfg),
        "evaluation_config": str(
            Path(str(figure_cfg["evaluation_config"])).resolve()
        ),
        "base_generator_config": base_generator_config,
        "selected_generator_batch_index": int(
            selected_batch_index
        ),
        "base_task_fingerprint": task_fingerprints(
            variants_cpu["ambiguous"]
        )[0],
        "condition_fingerprints": {
            condition: task_fingerprints(
                variants_cpu[condition]
            )[0]
            for condition in condition_names
        },
        "num_context_ambiguous": int(
            variants_cpu["ambiguous"].xc.shape[1]
        ),
        "num_context_revealing": int(
            variants_cpu["upper_reveal"].xc.shape[1]
        ),
        "conditions": list(condition_names),
        "source_manifest": _checkpoint_manifest(source_entries),
        "runtime_metadata": runtime_metadata(device),
    }

    cache = {
        "schema_version": "binary_fork_conditioning_cache_v1",
        "metadata": metadata,
        "x_grid": x_grid_cpu.cpu(),
        "contexts": {
            condition: {
                "xc": variants_cpu[condition].xc.cpu(),
                "yc": variants_cpu[condition].yc.cpu(),
            }
            for condition in condition_names
        },
        "exact": {
            "upper_probability": exact_curves,
            "regime_weights": exact_weights,
        },
        "groups": group_curves,
    }

    _save_cache(
        cache=cache,
        output_path=output_path,
    )
    print(
        "CACHE PASS [F2]: paired ambiguous/revealing task, exact curves, "
        "and all six learned probability curves validated."
    )
    _release_sources(loaded_sources)


# -----------------------------------------------------------------------------
# F3: direct versus AR-conditioned paths
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_coherence_cache(
    *,
    figure_cfg: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    smoke: bool,
) -> None:
    base_generator_config = str(
        evaluation_cfg["base_generator_config"]
    )
    source_entries = [
        dict(entry)
        for entry in evaluation_cfg["sources"]
    ]
    validate_sampling_offsets(source_entries)
    learned_entries = _model_entries(source_entries)
    _validate_exact_names(
        entries=learned_entries,
        expected=EXPECTED_MIXED_MODELS,
        label="F3 learned",
    )

    batch_cpu, selected_batch_index = _make_fixed_context_task(
        base_generator_config=base_generator_config,
        context_size=int(figure_cfg["context_size"]),
        deterministic_seed=int(figure_cfg["deterministic_seed"]),
        qualifying_task_index=int(
            figure_cfg["qualifying_task_index"]
        ),
        search_batches=int(figure_cfg["search_batches"]),
    )
    _validate_generator_identity(
        batch=batch_cpu,
        evaluation_cfg=evaluation_cfg,
    )

    num_anchors = int(figure_cfg["num_ar_anchors"])
    num_context = int(batch_cpu.xc.shape[1])
    training_max_nc = int(
        evaluation_cfg["training_max_nc"]
    )
    if num_context + num_anchors > training_max_nc:
        raise RuntimeError(
            f"F3 Nc+K={num_context + num_anchors} exceeds "
            f"training_max_nc={training_max_nc}."
        )

    x_dense_cpu = torch.linspace(
        float(figure_cfg["x_range"][0]),
        float(figure_cfg["x_range"][1]),
        int(figure_cfg["num_dense_points"]),
        dtype=batch_cpu.xc.dtype,
    ).view(1, -1, 1)

    predictor = batch_cpu.gt_pred
    if predictor is None or not hasattr(
        predictor,
        "sample_joint_observations_and_latent_function",
    ):
        raise RuntimeError(
            "F3 requires the binary-fork exact joint resampling API."
        )

    _set_seed(int(figure_cfg["resample_seed"]))
    observed_values, latent_truth = (
        predictor.sample_joint_observations_and_latent_function(
            x_observed=batch_cpu.x,
            x_plot=x_dense_cpu,
            regimes=getattr(
                predictor,
                "sampled_regimes",
                None,
            ),
            store=False,
        )
    )
    _assert_finite("F3 resampled observations", observed_values)
    _assert_finite("F3 latent truth", latent_truth)

    batch_cpu = dataclasses.replace(
        batch_cpu,
        y=observed_values,
        yc=observed_values[:, :num_context, :],
        yt=observed_values[:, num_context:, :],
    )
    batch = move_batch_to_device(batch_cpu, device)
    x_dense = x_dense_cpu.to(
        device=device,
        dtype=batch.xc.dtype,
    )
    dense_placeholder = torch.zeros(
        1,
        x_dense.shape[1],
        batch.yc.shape[-1],
        device=device,
        dtype=batch.yc.dtype,
    )
    dense_batch = dataclasses.replace(
        batch,
        xt=x_dense,
        yt=dense_placeholder,
    )

    x_anchor = torch.linspace(
        float(figure_cfg["ar_anchor_range"][0]),
        float(figure_cfg["ar_anchor_range"][1]),
        num_anchors,
        device=device,
        dtype=batch.xc.dtype,
    ).view(1, -1, 1)
    anchor_placeholder = torch.zeros(
        1,
        num_anchors,
        batch.yc.shape[-1],
        device=device,
        dtype=batch.yc.dtype,
    )
    ar_batch = dataclasses.replace(
        batch,
        xt=x_anchor,
        yt=anchor_placeholder,
    )

    component_means, component_scales, regime_weights = (
        predictor.posterior_marginal_components(
            xc=dense_batch.xc,
            yc=dense_batch.yc,
            xt=dense_batch.xt,
            include_target_noise=True,
        )
    )
    _assert_finite("F3 component means", component_means)
    _assert_finite("F3 component scales", component_scales)
    _assert_finite("F3 regime weights", regime_weights)

    loaded_sources = load_sources(
        entries=source_entries,
        base_generator_config=base_generator_config,
        device=device,
    )

    oracle_entry = source_entries[0]
    if str(oracle_entry.get("kind", "model")) != "oracle":
        raise RuntimeError(
            "F3 expects the exact joint oracle to be the first source."
        )
    oracle_seed = sampling_seed(
        base_seed=int(figure_cfg["sampling_seed"]),
        source_offset=int(oracle_entry["sampling_seed_offset"]),
        batch_index=0,
        condition_index=0,
    )
    _set_seed(oracle_seed)
    exact_direct = predictor.predictive_samples(
        xc=dense_batch.xc,
        yc=dense_batch.yc,
        xt=dense_batch.xt,
        num_samples=int(figure_cfg["num_direct_samples"]),
    )
    _assert_finite("F3 exact direct samples", exact_direct)

    model_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for loaded in loaded_sources:
        entry = loaded["entry"]
        if str(entry.get("kind", "model")) == "oracle":
            continue
        model_name = str(entry["name"])
        model = loaded["model"]
        if model is None:
            raise RuntimeError(
                f"F3 model did not load for {model_name!r}."
            )

        direct_seed = sampling_seed(
            base_seed=int(figure_cfg["sampling_seed"]),
            source_offset=int(entry["sampling_seed_offset"]),
            batch_index=0,
            condition_index=0,
        )
        _set_seed(direct_seed)
        direct_samples = sample_model_chunked(
            model=model,
            batch=dense_batch,
            num_samples=int(figure_cfg["num_direct_samples"]),
            chunk_size=int(figure_cfg["direct_chunk_size"]),
        )

        ar_seed = sampling_seed(
            base_seed=int(figure_cfg["sampling_seed"]),
            source_offset=int(entry["sampling_seed_offset"]),
            batch_index=0,
            condition_index=1,
        )
        _set_seed(ar_seed)
        raw_ar_samples = autoregressive_sample_model(
            model=model,
            batch=ar_batch,
            num_samples=int(figure_cfg["num_ar_paths"]),
            target_order=str(figure_cfg["target_order"]),
            stochln_noise_mode=str(
                figure_cfg["stochln_noise_mode"]
            ),
        )
        _assert_finite(
            f"F3 raw AR samples {model_name}",
            raw_ar_samples,
        )

        denoise_seed = sampling_seed(
            base_seed=int(figure_cfg["sampling_seed"]),
            source_offset=int(entry["sampling_seed_offset"]),
            batch_index=0,
            condition_index=2,
        )
        _set_seed(denoise_seed)
        ar_denoised_paths = _denoise_in_chunks(
            model=model,
            ar_batch=ar_batch,
            raw_samples=raw_ar_samples,
            query_xt=dense_batch.xt,
            num_denoise_samples=int(
                figure_cfg["num_denoise_samples"]
            ),
            chunk_size=int(figure_cfg["denoise_chunk_size"]),
        )

        model_cache[model_name] = {
            "direct_samples": direct_samples.detach().cpu(),
            "ar_support_x": x_anchor.detach().cpu(),
            "raw_ar_samples": raw_ar_samples.detach().cpu(),
            "ar_denoised_paths": ar_denoised_paths.detach().cpu(),
        }
        print(
            f"F3 cached {model_name}: "
            f"direct={tuple(direct_samples.shape)}, "
            f"raw_AR={tuple(raw_ar_samples.shape)}, "
            f"AR-conditioned={tuple(ar_denoised_paths.shape)}"
        )
        del direct_samples
        del raw_ar_samples
        del ar_denoised_paths
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(model_cache) != set(EXPECTED_MIXED_MODELS):
        raise RuntimeError(
            "F3 learned model cache keys are incomplete."
        )

    sampled_regimes = getattr(
        predictor,
        "sampled_regimes",
        None,
    )
    realised_regime_id = (
        int(sampled_regimes.reshape(-1)[0].item())
        if sampled_regimes is not None
        else None
    )
    realised_regime_name = (
        str(predictor.regime_name(realised_regime_id))
        if realised_regime_id is not None
        and hasattr(predictor, "regime_name")
        else None
    )

    metadata = {
        "figure_id": "F3",
        "figure_name": "direct versus AR-conditioned function paths",
        "smoke": bool(smoke),
        "figure_config": dict(figure_cfg),
        "evaluation_config": str(
            Path(str(figure_cfg["evaluation_config"])).resolve()
        ),
        "base_generator_config": base_generator_config,
        "selected_generator_batch_index": int(
            selected_batch_index
        ),
        "task_fingerprint": task_fingerprints(batch_cpu)[0],
        "num_context": num_context,
        "num_ar_anchors": num_anchors,
        "realised_regime_id": realised_regime_id,
        "realised_regime_name": realised_regime_name,
        "oracle_repeated_in_ar_row": True,
        "lower_row_semantics": (
            "raw sparse-anchor AR rollout followed by dense conditional "
            "predictive mean reconstruction"
        ),
        "source_manifest": _checkpoint_manifest(source_entries),
        "runtime_metadata": runtime_metadata(device),
    }

    cache = {
        "schema_version": "binary_fork_coherence_cache_v1",
        "metadata": metadata,
        "task": {
            "xc": batch_cpu.xc.cpu(),
            "yc": batch_cpu.yc.cpu(),
            "xt": batch_cpu.xt.cpu(),
            "yt": batch_cpu.yt.cpu(),
            "x_plot": x_dense_cpu.cpu(),
            "latent_truth": latent_truth.cpu(),
        },
        "exact_joint": {
            "direct_samples": exact_direct.detach().cpu(),
            "component_means": component_means.detach().cpu(),
            "component_scales": component_scales.detach().cpu(),
            "regime_weights": regime_weights.detach().cpu(),
            "oracle_repeated_in_ar_row": True,
        },
        "models": model_cache,
    }

    _save_cache(
        cache=cache,
        output_path=output_path,
    )
    print(
        "CACHE PASS [F3]: jointly resampled task, exact coherent paths, "
        "direct samples, raw AR supports, and dense AR-conditioned paths "
        "validated."
    )
    _release_sources(loaded_sources)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    cfg = _load_yaml(args.config)
    required_sections = {
        "marginals",
        "conditioning",
        "coherence",
    }
    missing_sections = required_sections.difference(cfg)
    if missing_sections:
        raise KeyError(
            "Figure config is missing sections: "
            f"{sorted(missing_sections)}"
        )

    selected = _selected_figures(args.figure)
    effective_cfg = (
        _apply_smoke_overrides(cfg)
        if args.smoke
        else copy.deepcopy(cfg)
    )

    device_name = args.device or str(
        effective_cfg.get("device", "cuda")
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )
    device = torch.device(device_name)

    output_dir = Path(
        args.output_dir
        or str(effective_cfg["output_dir"])
    )
    output_paths = _preflight_outputs(
        output_dir=output_dir,
        cfg=effective_cfg,
        selected=selected,
        smoke=bool(args.smoke),
        overwrite=bool(args.overwrite),
    )

    marginal_eval_cfg = _load_yaml(
        effective_cfg["marginals"]["evaluation_config"]
    )
    conditioning_eval_cfg = _load_yaml(
        effective_cfg["conditioning"]["evaluation_config"]
    )
    coherence_eval_cfg = _load_yaml(
        effective_cfg["coherence"]["evaluation_config"]
    )
    _validate_mixed_checkpoint_parity(
        marginal_cfg=marginal_eval_cfg,
        coherence_cfg=coherence_eval_cfg,
    )

    print("=" * 88)
    print("BINARY-FORK DISSERTATION FIGURE CACHE")
    print(f"Repository root: {repo_root}")
    print(f"Device: {device}")
    print(f"Selected figures: {', '.join(selected)}")
    print(f"Smoke mode: {bool(args.smoke)}")
    print(f"Output directory: {output_dir}")
    print("=" * 88)

    if "marginals" in selected:
        build_marginal_cache(
            figure_cfg=effective_cfg["marginals"],
            evaluation_cfg=marginal_eval_cfg,
            device=device,
            output_path=output_paths["marginals"],
            smoke=bool(args.smoke),
        )

    if "conditioning" in selected:
        build_conditioning_cache(
            figure_cfg=effective_cfg["conditioning"],
            evaluation_cfg=conditioning_eval_cfg,
            device=device,
            output_path=output_paths["conditioning"],
            smoke=bool(args.smoke),
        )

    if "coherence" in selected:
        build_coherence_cache(
            figure_cfg=effective_cfg["coherence"],
            evaluation_cfg=coherence_eval_cfg,
            device=device,
            output_path=output_paths["coherence"],
            smoke=bool(args.smoke),
        )

    print("=" * 88)
    print("ALL SELECTED BINARY-FORK FIGURE CACHES COMPLETED.")
    print("=" * 88)


if __name__ == "__main__":
    main()
