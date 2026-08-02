from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


DEFAULT_INTERVAL_LEVELS: Tuple[float, ...] = (0.90,)


def _level_suffix(level: float) -> str:
    return f"{int(round(100.0 * float(level))):02d}"


def _normal_interval_z(level: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"Interval level must lie in (0, 1), got {level}.")

    probability = torch.tensor(
        0.5 * (1.0 + float(level)),
        device=device,
        dtype=dtype,
    )
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    return standard_normal.icdf(probability)


def gaussian_crps_per_element(
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Closed-form CRPS for univariate Gaussian predictive marginals."""
    if loc.shape != target.shape or scale.shape != target.shape:
        raise ValueError(
            "loc, scale and target must have identical shapes. "
            f"Got loc={tuple(loc.shape)}, scale={tuple(scale.shape)}, "
            f"target={tuple(target.shape)}."
        )

    safe_scale = scale.clamp_min(float(epsilon))
    z = (target - loc) / safe_scale

    sqrt_two = math.sqrt(2.0)
    sqrt_pi = math.sqrt(math.pi)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)

    cdf = 0.5 * (1.0 + torch.erf(z / sqrt_two))
    pdf = torch.exp(-0.5 * z.square()) / sqrt_two_pi

    return safe_scale * (
        z * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / sqrt_pi
    )


def fair_crps_per_element_sorted(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Fair empirical CRPS without constructing an M x M tensor.

    Args:
        samples: [M, B, Nt, Dy]
        target:  [B, Nt, Dy]
    """
    if samples.ndim != target.ndim + 1 or samples.shape[1:] != target.shape:
        raise ValueError(
            "Expected samples [M, ...] with samples.shape[1:] == target.shape. "
            f"Got samples={tuple(samples.shape)}, target={tuple(target.shape)}."
        )

    num_samples = int(samples.shape[0])
    if num_samples < 2:
        raise ValueError("Fair CRPS requires at least two predictive samples.")

    sorted_samples = samples.sort(dim=0).values
    coefficients = torch.arange(
        1,
        num_samples + 1,
        device=samples.device,
        dtype=samples.dtype,
    )
    coefficients = (
        2.0 * coefficients - float(num_samples) - 1.0
    ).reshape(num_samples, *([1] * (samples.ndim - 1)))

    # Sum_{i<j} (x_(j) - x_(i)).
    weighted_order_sum = (coefficients * sorted_samples).sum(dim=0)
    target_term = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)

    return target_term - weighted_order_sum / (
        num_samples * (num_samples - 1)
    )


def fair_offdiag_diversity_per_element_sorted(samples: torch.Tensor) -> torch.Tensor:
    """Mean |X_i-X_j| over ordered off-diagonal ensemble pairs."""
    if samples.ndim < 2:
        raise ValueError(f"Expected samples [M, ...], got {tuple(samples.shape)}.")

    num_samples = int(samples.shape[0])
    if num_samples < 2:
        raise ValueError("Diversity requires at least two predictive samples.")

    sorted_samples = samples.sort(dim=0).values
    coefficients = torch.arange(
        1,
        num_samples + 1,
        device=samples.device,
        dtype=samples.dtype,
    )
    coefficients = (
        2.0 * coefficients - float(num_samples) - 1.0
    ).reshape(num_samples, *([1] * (samples.ndim - 1)))

    weighted_order_sum = (coefficients * sorted_samples).sum(dim=0)
    return 2.0 * weighted_order_sum / (
        num_samples * (num_samples - 1)
    )


def fingerprint_task(batch: Any, local_index: int) -> str:
    """Stable SHA-256 fingerprint of one generated context/target task."""
    digest = hashlib.sha256()

    for field_name in ("xc", "yc", "xt", "yt"):
        tensor = getattr(batch, field_name)[local_index].detach().cpu().contiguous()
        digest.update(field_name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))

    return digest.hexdigest()


def task_fingerprints(batch: Any) -> List[str]:
    return [fingerprint_task(batch, i) for i in range(int(batch.yt.shape[0]))]


def _context_bucket(num_context: int) -> str:
    if num_context <= 4:
        return "nc_001_004"
    if num_context <= 16:
        return "nc_005_016"
    return "nc_017_064"


def _region_masks(
    *,
    xt: torch.Tensor,
    target: torch.Tensor,
    context_range: Sequence[Sequence[float]],
) -> Mapping[str, torch.Tensor]:
    if xt.ndim != 3 or xt.shape[-1] != 1:
        raise ValueError(
            "Gaussian-control evaluation currently requires xt [B, Nt, 1]. "
            f"Got {tuple(xt.shape)}."
        )

    if len(context_range) != 1 or len(context_range[0]) != 2:
        raise ValueError(f"Expected one 1-D context range, got {context_range}.")

    context_min = float(context_range[0][0])
    context_max = float(context_range[0][1])

    interp = (
        (xt[..., 0] >= context_min)
        & (xt[..., 0] <= context_max)
    ).unsqueeze(-1).expand_as(target)

    return {
        "all": torch.ones_like(target, dtype=torch.bool),
        "interpolation": interp,
        "extrapolation": ~interp,
    }


def _normalise_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(metadata)
    out.setdefault("training_alpha", None)
    out.setdefault("training_num_samples", None)
    out.setdefault("training_p_dropout", None)
    out.setdefault("training_layernorm_noise_dim", None)
    return out


def _rows_from_elementwise_components(
    *,
    target: torch.Tensor,
    xt: torch.Tensor,
    context_range: Sequence[Sequence[float]],
    squared_error: torch.Tensor,
    crps: torch.Tensor,
    predictive_variance: torch.Tensor,
    diversity: torch.Tensor,
    intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    model_name: str,
    source_kind: str,
    metric_mode: str,
    checkpoint_path: str,
    eval_set: str,
    kernel_name: str,
    num_context: int,
    num_eval_samples: int,
    finite_ensemble: bool,
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not (
        target.shape
        == squared_error.shape
        == crps.shape
        == predictive_variance.shape
        == diversity.shape
    ):
        raise ValueError("All elementwise metric tensors must match target.shape.")

    batch_size = int(target.shape[0])
    num_targets = int(target.shape[1])
    output_dim = int(target.shape[2])

    if len(fingerprints) != batch_size:
        raise ValueError(
            f"Expected {batch_size} fingerprints, got {len(fingerprints)}."
        )

    region_masks = _region_masks(
        xt=xt,
        target=target,
        context_range=context_range,
    )

    metadata_dict = _normalise_metadata(metadata)
    rows: List[Dict[str, Any]] = []

    squared_flat = squared_error.reshape(batch_size, -1)
    crps_flat = crps.reshape(batch_size, -1)
    variance_flat = predictive_variance.reshape(batch_size, -1)
    diversity_flat = diversity.reshape(batch_size, -1)

    interval_flat: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level, (covered, width) in intervals.items():
        interval_flat[float(level)] = (
            covered.reshape(batch_size, -1),
            width.reshape(batch_size, -1),
        )

    finite_m_correction = (
        math.sqrt((float(num_eval_samples) + 1.0) / float(num_eval_samples))
        if finite_ensemble
        else 1.0
    )

    for region_name, region_mask in region_masks.items():
        mask_flat = region_mask.reshape(batch_size, -1)

        for local_index in range(batch_size):
            selected = mask_flat[local_index]
            numel = int(selected.sum().item())
            if numel == 0:
                continue

            sse = squared_flat[local_index][selected].sum()
            crps_sum = crps_flat[local_index][selected].sum()
            var_sum = variance_flat[local_index][selected].sum()
            diversity_sum = diversity_flat[local_index][selected].sum()

            rmse = torch.sqrt(sse / float(numel))
            spread = torch.sqrt(var_sum / float(numel))
            ssr = finite_m_correction * spread / (rmse + 1.0e-12)

            row: Dict[str, Any] = {
                "model_name": model_name,
                "source_kind": source_kind,
                "metric_mode": metric_mode,
                "checkpoint_path": checkpoint_path,
                "eval_set": eval_set,
                "kernel_name": kernel_name,
                "region": region_name,
                "context_bucket": _context_bucket(num_context),
                "task_index": int(task_index_start) + local_index,
                "generator_batch_index": int(generator_batch_index),
                "within_batch_index": local_index,
                "task_fingerprint": fingerprints[local_index],
                "num_context": int(num_context),
                "num_targets": num_targets,
                "output_dim": output_dim,
                "numel": numel,
                "num_eval_samples": int(num_eval_samples),
                "finite_ensemble": bool(finite_ensemble),
                "metric_alpha": 1.0,
                "sse": float(sse.item()),
                "rmse": float(rmse.item()),
                "crps_sum": float(crps_sum.item()),
                "crps": float((crps_sum / float(numel)).item()),
                "var_sum": float(var_sum.item()),
                "ensemble_spread": float(spread.item()),
                "spread_skill_ratio": float(ssr.item()),
                "diversity_sum": float(diversity_sum.item()),
                "sample_diversity_offdiag": float(
                    (diversity_sum / float(numel)).item()
                ),
            }

            for key, value in metadata_dict.items():
                row[key] = value

            for level, (covered_flat, width_flat) in interval_flat.items():
                suffix = _level_suffix(level)
                coverage_count = covered_flat[local_index][selected].sum()
                width_sum = width_flat[local_index][selected].sum()
                row[f"coverage_count_{suffix}"] = float(coverage_count.item())
                row[f"coverage_{suffix}"] = float(
                    (coverage_count / float(numel)).item()
                )
                row[f"width_sum_{suffix}"] = float(width_sum.item())
                row[f"width_{suffix}"] = float(
                    (width_sum / float(numel)).item()
                )

            rows.append(row)

    return rows


def per_task_rows_sampled(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    xt: torch.Tensor,
    context_range: Sequence[Sequence[float]],
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    model_name: str,
    source_kind: str,
    checkpoint_path: str,
    eval_set: str,
    kernel_name: str,
    num_context: int,
    metadata: Mapping[str, Any],
    interval_levels: Iterable[float] = DEFAULT_INTERVAL_LEVELS,
) -> List[Dict[str, Any]]:
    """Per-task rows for a finite predictive ensemble."""
    if samples.shape[1:] != target.shape:
        raise ValueError(
            "Expected samples.shape[1:] == target.shape. "
            f"Got samples={tuple(samples.shape)}, target={tuple(target.shape)}."
        )

    samples = samples.detach()
    target = target.detach()

    num_samples = int(samples.shape[0])
    if num_samples < 2:
        raise ValueError("Sample-based Gaussian controls require M >= 2.")

    pred_mean = samples.mean(dim=0)
    squared_error = (pred_mean - target).square()
    predictive_variance = samples.var(dim=0, unbiased=True)
    crps = fair_crps_per_element_sorted(samples=samples, target=target)
    diversity = fair_offdiag_diversity_per_element_sorted(samples)

    intervals: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level in interval_levels:
        level = float(level)
        lower_q = 0.5 * (1.0 - level)
        upper_q = 1.0 - lower_q
        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        covered = ((target >= lower) & (target <= upper)).to(target.dtype)
        intervals[level] = (covered, upper - lower)

    return _rows_from_elementwise_components(
        target=target,
        xt=xt,
        context_range=context_range,
        squared_error=squared_error,
        crps=crps,
        predictive_variance=predictive_variance,
        diversity=diversity,
        intervals=intervals,
        task_index_start=task_index_start,
        generator_batch_index=generator_batch_index,
        fingerprints=fingerprints,
        model_name=model_name,
        source_kind=source_kind,
        metric_mode="sampled",
        checkpoint_path=checkpoint_path,
        eval_set=eval_set,
        kernel_name=kernel_name,
        num_context=num_context,
        num_eval_samples=num_samples,
        finite_ensemble=True,
        metadata=metadata,
    )


def per_task_rows_gaussian(
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    xt: torch.Tensor,
    context_range: Sequence[Sequence[float]],
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    model_name: str,
    source_kind: str,
    checkpoint_path: str,
    eval_set: str,
    kernel_name: str,
    num_context: int,
    metadata: Mapping[str, Any],
    interval_levels: Iterable[float] = DEFAULT_INTERVAL_LEVELS,
) -> List[Dict[str, Any]]:
    """Per-task rows for analytic univariate Gaussian marginals."""
    if loc.shape != target.shape or scale.shape != target.shape:
        raise ValueError(
            "Analytic Gaussian loc/scale must match target.shape. "
            f"Got loc={tuple(loc.shape)}, scale={tuple(scale.shape)}, "
            f"target={tuple(target.shape)}."
        )

    loc = loc.detach()
    scale = scale.detach().clamp_min(1.0e-8)
    target = target.detach()

    squared_error = (loc - target).square()
    predictive_variance = scale.square()
    crps = gaussian_crps_per_element(loc=loc, scale=scale, target=target)

    # For X, X' iid N(mu, sigma^2), E|X-X'| = 2 sigma / sqrt(pi).
    diversity = 2.0 * scale / math.sqrt(math.pi)

    intervals: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level in interval_levels:
        level = float(level)
        z = _normal_interval_z(level, device=loc.device, dtype=loc.dtype)
        lower = loc - z * scale
        upper = loc + z * scale
        covered = ((target >= lower) & (target <= upper)).to(target.dtype)
        intervals[level] = (covered, upper - lower)

    return _rows_from_elementwise_components(
        target=target,
        xt=xt,
        context_range=context_range,
        squared_error=squared_error,
        crps=crps,
        predictive_variance=predictive_variance,
        diversity=diversity,
        intervals=intervals,
        task_index_start=task_index_start,
        generator_batch_index=generator_batch_index,
        fingerprints=fingerprints,
        model_name=model_name,
        source_kind=source_kind,
        metric_mode="analytic_gaussian",
        checkpoint_path=checkpoint_path,
        eval_set=eval_set,
        kernel_name=kernel_name,
        num_context=num_context,
        num_eval_samples=0,
        finite_ensemble=False,
        metadata=metadata,
    )
