from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from evaluation.gaussian_controls_metrics import (
    fair_crps_per_element_sorted,
    fair_offdiag_diversity_per_element_sorted,
)


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def oracle_shape_reference(
    *,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    regime_weights: torch.Tensor,
    gap_half_width_fraction: float,
) -> Dict[str, torch.Tensor]:
    """Exact marginal branch and central-gap probabilities.

    Components are ordered lower then upper and have shape [B, 2, N, 1].
    """
    if component_means.shape != component_scales.shape:
        raise ValueError("component_means and component_scales must match.")
    if component_means.ndim != 4 or component_means.shape[1] != 2:
        raise ValueError(
            "Expected components [B, 2, N, 1], got "
            f"{tuple(component_means.shape)}."
        )
    if regime_weights.shape != component_means.shape[:2]:
        raise ValueError(
            f"Expected regime_weights {component_means.shape[:2]}, "
            f"got {tuple(regime_weights.shape)}."
        )
    gap_half_width_fraction = float(gap_half_width_fraction)
    if not 0.0 < gap_half_width_fraction < 1.0:
        raise ValueError(
            "gap_half_width_fraction must be in (0, 1), got "
            f"{gap_half_width_fraction}."
        )

    lower_mean = component_means[:, 0]
    upper_mean = component_means[:, 1]
    lower_scale = component_scales[:, 0].clamp_min(1.0e-8)
    upper_scale = component_scales[:, 1].clamp_min(1.0e-8)

    centre = 0.5 * (lower_mean + upper_mean)
    half_separation = 0.5 * (upper_mean - lower_mean).abs()
    gap_half_width = gap_half_width_fraction * half_separation
    gap_lower = centre - gap_half_width
    gap_upper = centre + gap_half_width

    weights = regime_weights[:, :, None, None]
    component_mean = component_means
    component_scale = component_scales.clamp_min(1.0e-8)

    centre_z = (centre[:, None] - component_mean) / component_scale
    p_upper = (
        weights * (1.0 - _normal_cdf(centre_z))
    ).sum(dim=1)

    gap_lower_z = (gap_lower[:, None] - component_mean) / component_scale
    gap_upper_z = (gap_upper[:, None] - component_mean) / component_scale
    p_gap = (
        weights * (_normal_cdf(gap_upper_z) - _normal_cdf(gap_lower_z))
    ).sum(dim=1)

    return {
        "centre": centre,
        "gap_lower": gap_lower,
        "gap_upper": gap_upper,
        "oracle_upper_mass": p_upper,
        "oracle_gap_mass": p_gap,
        "half_separation": half_separation,
    }


def sample_marginal_mixture(
    *,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    regime_weights: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Draw exact marginal mixture samples from precomputed components.

    One latent branch is drawn per task and ensemble member. Conditional
    residuals are independent across targets because this helper is used only
    for marginal scoring.
    """
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if component_means.shape != component_scales.shape:
        raise ValueError("component_means and component_scales must match.")
    if component_means.ndim != 4 or int(component_means.shape[1]) != 2:
        raise ValueError(
            "Expected component tensors [B, 2, N, Dy], got "
            f"{tuple(component_means.shape)}."
        )
    if regime_weights.shape != component_means.shape[:2]:
        raise ValueError(
            f"Expected regime_weights {component_means.shape[:2]}, "
            f"got {tuple(regime_weights.shape)}."
        )

    batch_size = int(component_means.shape[0])
    upper_probability = regime_weights[:, 1].reshape(1, batch_size, 1, 1)
    choose_upper = torch.rand(
        num_samples,
        batch_size,
        1,
        1,
        device=component_means.device,
        dtype=component_means.dtype,
    ) < upper_probability

    lower_mean = component_means[:, 0].unsqueeze(0)
    upper_mean = component_means[:, 1].unsqueeze(0)
    lower_scale = component_scales[:, 0].unsqueeze(0)
    upper_scale = component_scales[:, 1].unsqueeze(0)
    selected_mean = torch.where(choose_upper, upper_mean, lower_mean)
    selected_scale = torch.where(choose_upper, upper_scale, lower_scale)

    samples = selected_mean + selected_scale * torch.randn(
        num_samples,
        batch_size,
        int(component_means.shape[2]),
        int(component_means.shape[3]),
        device=component_means.device,
        dtype=component_means.dtype,
    )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Exact marginal mixture samples are non-finite.")
    return samples


def _region_masks(
    *,
    xt: torch.Tensor,
    target: torch.Tensor,
    branch_start: float,
) -> Mapping[str, torch.Tensor]:
    x = xt[..., 0]
    post = (x >= float(branch_start)).unsqueeze(-1).expand_as(target)
    return {
        "all": torch.ones_like(target, dtype=torch.bool),
        "postfork": post,
    }


def per_task_marginal_rows(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    xt: torch.Tensor,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    regime_weights: torch.Tensor,
    branch_start: float,
    gap_half_width_fraction: float,
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    model_name: str,
    source_kind: str,
    checkpoint_path: str,
    eval_set: str,
    num_context: int,
    metadata: Mapping[str, Any],
    condition: str = "ambiguous",
    interval_levels: Iterable[float] = (0.90,),
) -> List[Dict[str, Any]]:
    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"samples.shape[1:]={samples.shape[1:]} != target={target.shape}."
        )
    num_samples = int(samples.shape[0])
    if num_samples < 2:
        raise ValueError("At least two predictive samples are required.")

    samples = samples.detach()
    target = target.detach()
    batch_size = int(target.shape[0])
    num_targets = int(target.shape[1])
    output_dim = int(target.shape[2])

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

    shape = oracle_shape_reference(
        component_means=component_means,
        component_scales=component_scales,
        regime_weights=regime_weights,
        gap_half_width_fraction=gap_half_width_fraction,
    )
    centre = shape["centre"]
    gap_lower = shape["gap_lower"]
    gap_upper = shape["gap_upper"]
    oracle_upper = shape["oracle_upper_mass"]
    oracle_gap = shape["oracle_gap_mass"]

    pred_upper = (samples > centre.unsqueeze(0)).to(samples.dtype).mean(dim=0)
    pred_gap = (
        (samples >= gap_lower.unsqueeze(0))
        & (samples <= gap_upper.unsqueeze(0))
    ).to(samples.dtype).mean(dim=0)
    branch_error = (pred_upper - oracle_upper).abs()
    gap_error = (pred_gap - oracle_gap).abs()

    masks = _region_masks(xt=xt, target=target, branch_start=branch_start)
    finite_m_correction = math.sqrt((num_samples + 1.0) / num_samples)
    metadata_dict = dict(metadata)
    rows: List[Dict[str, Any]] = []

    for region, mask in masks.items():
        flat_mask = mask.reshape(batch_size, -1)
        for local_index in range(batch_size):
            selected = flat_mask[local_index]
            numel = int(selected.sum().item())
            if numel == 0:
                continue

            def selected_sum(value: torch.Tensor) -> torch.Tensor:
                return value.reshape(batch_size, -1)[local_index][selected].sum()

            sse = selected_sum(squared_error)
            crps_sum = selected_sum(crps)
            var_sum = selected_sum(predictive_variance)
            diversity_sum = selected_sum(diversity)
            rmse = torch.sqrt(sse / float(numel))
            spread = torch.sqrt(var_sum / float(numel))

            row: Dict[str, Any] = {
                "model_name": model_name,
                "source_kind": source_kind,
                "metric_mode": "sampled",
                "checkpoint_path": checkpoint_path,
                "eval_set": eval_set,
                "condition": condition,
                "region": region,
                "task_index": int(task_index_start) + local_index,
                "generator_batch_index": int(generator_batch_index),
                "within_batch_index": local_index,
                "task_fingerprint": fingerprints[local_index],
                "num_context": int(num_context),
                "num_targets": num_targets,
                "output_dim": output_dim,
                "num_eval_samples": num_samples,
                "finite_ensemble": True,
                "numel": numel,
                "sse": float(sse.item()),
                "rmse": float(rmse.item()),
                "crps_sum": float(crps_sum.item()),
                "crps": float((crps_sum / float(numel)).item()),
                "var_sum": float(var_sum.item()),
                "ensemble_spread": float(spread.item()),
                "spread_skill_ratio": float(
                    (finite_m_correction * spread / (rmse + 1.0e-12)).item()
                ),
                "diversity_sum": float(diversity_sum.item()),
                "sample_diversity_offdiag": float(
                    (diversity_sum / float(numel)).item()
                ),
            }
            for key, value in metadata_dict.items():
                row[key] = value

            for level, (covered, width) in intervals.items():
                suffix = f"{int(round(100.0 * level)):02d}"
                coverage_count = selected_sum(covered)
                width_sum = selected_sum(width)
                row[f"coverage_count_{suffix}"] = float(coverage_count.item())
                row[f"coverage_{suffix}"] = float(
                    (coverage_count / float(numel)).item()
                )
                row[f"width_sum_{suffix}"] = float(width_sum.item())
                row[f"width_{suffix}"] = float((width_sum / float(numel)).item())

            if region == "postfork":
                row.update(
                    {
                        "shape_numel": numel,
                        "pred_upper_mass_sum": float(selected_sum(pred_upper).item()),
                        "oracle_upper_mass_sum": float(selected_sum(oracle_upper).item()),
                        "branch_mass_error_sum": float(selected_sum(branch_error).item()),
                        "pred_gap_mass_sum": float(selected_sum(pred_gap).item()),
                        "oracle_gap_mass_sum": float(selected_sum(oracle_gap).item()),
                        "gap_mass_error_sum": float(selected_sum(gap_error).item()),
                        "pred_upper_mass": float(
                            (selected_sum(pred_upper) / float(numel)).item()
                        ),
                        "oracle_upper_mass": float(
                            (selected_sum(oracle_upper) / float(numel)).item()
                        ),
                        "branch_mass_error": float(
                            (selected_sum(branch_error) / float(numel)).item()
                        ),
                        "pred_gap_mass": float(
                            (selected_sum(pred_gap) / float(numel)).item()
                        ),
                        "oracle_gap_mass": float(
                            (selected_sum(oracle_gap) / float(numel)).item()
                        ),
                        "gap_mass_error": float(
                            (selected_sum(gap_error) / float(numel)).item()
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "shape_numel": 0,
                        "pred_upper_mass_sum": 0.0,
                        "oracle_upper_mass_sum": 0.0,
                        "branch_mass_error_sum": 0.0,
                        "pred_gap_mass_sum": 0.0,
                        "oracle_gap_mass_sum": 0.0,
                        "gap_mass_error_sum": 0.0,
                        "pred_upper_mass": float("nan"),
                        "oracle_upper_mass": float("nan"),
                        "branch_mass_error": float("nan"),
                        "pred_gap_mass": float("nan"),
                        "oracle_gap_mass": float("nan"),
                        "gap_mass_error": float("nan"),
                    }
                )

            rows.append(row)

    return rows


def _path_metrics_one(
    side: torch.Tensor,
) -> Tuple[int, int, int, float]:
    """Return active count, switches, zero-switch and gap fraction."""
    active = side[side != 0]
    active_count = int(active.numel())
    if active_count >= 2:
        switches = int((active[1:] != active[:-1]).sum().item())
        zero_switch = int(switches == 0)
    else:
        switches = 0
        zero_switch = 0
    gap_fraction = float((side == 0).float().mean().item())
    return active_count, switches, zero_switch, gap_fraction


def _independence_baseline(
    p_lower: torch.Tensor,
    p_gap: torch.Tensor,
    p_upper: torch.Tensor,
) -> Tuple[float, float, float]:
    """Exact independence baseline under the observed marginal categories."""
    p_lower = p_lower.double()
    p_gap = p_gap.double()
    p_upper = p_upper.double()
    k = int(p_gap.numel())

    all_gap = torch.prod(p_gap)
    no_lower = torch.prod(p_upper + p_gap)
    no_upper = torch.prod(p_lower + p_gap)

    one_upper = torch.tensor(0.0, dtype=torch.float64)
    one_lower = torch.tensor(0.0, dtype=torch.float64)
    for index in range(k):
        others = torch.cat([p_gap[:index], p_gap[index + 1 :]])
        gap_others = torch.prod(others) if others.numel() else torch.tensor(1.0)
        one_upper += p_upper[index] * gap_others
        one_lower += p_lower[index] * gap_others

    zero_switch = (
        (no_lower - all_gap - one_upper)
        + (no_upper - all_gap - one_lower)
    ).clamp(0.0, 1.0)

    expected_switches = torch.tensor(0.0, dtype=torch.float64)
    for left in range(k - 1):
        for right in range(left + 1, k):
            between = p_gap[left + 1 : right]
            gap_between = (
                torch.prod(between)
                if between.numel()
                else torch.tensor(1.0, dtype=torch.float64)
            )
            expected_switches += (
                p_lower[left] * p_upper[right]
                + p_upper[left] * p_lower[right]
            ) * gap_between

    return (
        float(zero_switch.item()),
        float(expected_switches.item()),
        float(p_gap.mean().item()),
    )


def per_task_path_rows(
    *,
    samples: torch.Tensor,
    x_path: torch.Tensor,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    regime_weights: torch.Tensor,
    branch_start: float,
    gap_half_width_fraction: float,
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    model_name: str,
    source_kind: str,
    checkpoint_path: str,
    deployment: str,
    metadata: Mapping[str, Any],
    oracle_repeated: bool = False,
) -> List[Dict[str, Any]]:
    if samples.ndim != 4 or samples.shape[-1] != 1:
        raise ValueError(f"Expected samples [M, B, K, 1], got {samples.shape}.")
    if x_path.shape[:2] != samples.shape[1:3]:
        raise ValueError(
            f"x_path {x_path.shape} does not align with samples {samples.shape}."
        )

    shape = oracle_shape_reference(
        component_means=component_means,
        component_scales=component_scales,
        regime_weights=regime_weights,
        gap_half_width_fraction=gap_half_width_fraction,
    )
    centre = shape["centre"]
    gap_lower = shape["gap_lower"]
    gap_upper = shape["gap_upper"]

    post_mask = x_path[..., 0] >= float(branch_start)
    num_paths = int(samples.shape[0])
    batch_size = int(samples.shape[1])
    rows: List[Dict[str, Any]] = []

    for task_index in range(batch_size):
        selected = post_mask[task_index]
        if int(selected.sum().item()) < 2:
            raise ValueError("Need at least two post-fork path locations.")

        task_samples = samples[:, task_index, selected, 0].detach().cpu()
        task_centre = centre[task_index, selected, 0].detach().cpu()
        task_lower = gap_lower[task_index, selected, 0].detach().cpu()
        task_upper = gap_upper[task_index, selected, 0].detach().cpu()

        side = torch.zeros_like(task_samples, dtype=torch.int8)
        side[task_samples < task_lower.unsqueeze(0)] = -1
        side[task_samples > task_upper.unsqueeze(0)] = 1

        zero_switch_sum = 0
        switch_sum = 0
        gap_fraction_sum = 0.0
        active_path_sum = 0
        for path_index in range(num_paths):
            active_count, switches, zero_switch, gap_fraction = _path_metrics_one(
                side[path_index]
            )
            zero_switch_sum += zero_switch
            switch_sum += switches
            gap_fraction_sum += gap_fraction
            active_path_sum += int(active_count >= 2)

        p_lower = (side == -1).float().mean(dim=0)
        p_gap = (side == 0).float().mean(dim=0)
        p_upper = (side == 1).float().mean(dim=0)
        baseline_zero, baseline_switches, baseline_gap = _independence_baseline(
            p_lower=p_lower,
            p_gap=p_gap,
            p_upper=p_upper,
        )

        row: Dict[str, Any] = {
            "model_name": model_name,
            "source_kind": source_kind,
            "checkpoint_path": checkpoint_path,
            "deployment": deployment,
            "oracle_repeated": bool(oracle_repeated),
            "task_index": int(task_index_start) + task_index,
            "generator_batch_index": int(generator_batch_index),
            "within_batch_index": task_index,
            "task_fingerprint": fingerprints[task_index],
            "num_paths": num_paths,
            "num_path_points": int(selected.sum().item()),
            "zero_switch_sum": int(zero_switch_sum),
            "switch_count_sum": int(switch_sum),
            "gap_fraction_sum": float(gap_fraction_sum),
            "active_path_sum": int(active_path_sum),
            "zero_switch_rate": float(zero_switch_sum / num_paths),
            "mean_switch_count": float(switch_sum / num_paths),
            "mid_gap_rate": float(gap_fraction_sum / num_paths),
            "independence_zero_switch": baseline_zero,
            "independence_mean_switches": baseline_switches,
            "independence_mid_gap_rate": baseline_gap,
            "excess_zero_switch": float(zero_switch_sum / num_paths - baseline_zero),
        }
        for key, value in dict(metadata).items():
            row[key] = value
        rows.append(row)

    return rows
