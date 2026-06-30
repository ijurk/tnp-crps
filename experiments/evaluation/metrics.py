from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch


DEFAULT_INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def context_bucket(num_context: int) -> str:
    """Bucket context count for stratified reporting."""
    if num_context <= 4:
        return "nc_001_004"
    if num_context <= 16:
        return "nc_005_016"
    return "nc_017_064"


def level_suffix(level: float) -> str:
    return f"{int(round(level * 100)):02d}"


def expand_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Return boolean mask with shape matching target [B, Nt, Dy]."""
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)

    if mask.ndim == target.ndim - 1:
        mask = mask.unsqueeze(-1)

    return mask.expand_as(target).bool()


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].sum()


def crps_per_element(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Almost-fair marginal CRPS per target element.

    Args:
        samples: [M, B, Nt, Dy]
        target:  [B, Nt, Dy]
        alpha:   1.0 fair CRPS, 0.0 ordinary empirical CRPS.

    Returns:
        [B, Nt, Dy] CRPS values.
    """
    if samples.ndim != target.ndim + 1:
        raise ValueError(
            f"Expected samples [M, ...] and target [...]. "
            f"Got samples={samples.shape}, target={target.shape}."
        )

    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"Expected samples.shape[1:] == target.shape. "
            f"Got samples={samples.shape}, target={target.shape}."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]. Got {alpha}.")

    num_samples = samples.shape[0]
    if num_samples < 2:
        raise ValueError("CRPS evaluation requires at least 2 samples.")

    target_term = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)

    pairwise_dist = torch.abs(samples[:, None, ...] - samples[None, :, ...])

    ordinary_pairwise = pairwise_dist.mean(dim=(0, 1))
    fair_pairwise = pairwise_dist.sum(dim=(0, 1)) / (
        num_samples * (num_samples - 1)
    )

    combined_pairwise = alpha * fair_pairwise + (1.0 - alpha) * ordinary_pairwise

    return target_term - 0.5 * combined_pairwise

def energy_score_per_task(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fair finite-ensemble energy score per task.

    Args:
        samples: [M, B, Nt, Dy]
        target:  [B, Nt, Dy]
        mask:    optional [B, Nt] or [B, Nt, Dy]

    Returns:
        [B] energy scores, with NaN for tasks with no selected targets.
    """
    if samples.ndim != target.ndim + 1:
        raise ValueError(
            f"Expected samples [M, ...] and target [...]. "
            f"Got samples={samples.shape}, target={target.shape}."
        )

    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"Expected samples.shape[1:] == target.shape. "
            f"Got samples={samples.shape}, target={target.shape}."
        )

    num_samples = samples.shape[0]
    if num_samples < 2:
        raise ValueError("Energy score requires at least 2 samples.")

    bool_mask = expand_mask(mask, target)

    scores = []
    for b in range(target.shape[0]):
        flat_mask = bool_mask[b].reshape(-1)

        if int(flat_mask.sum().item()) == 0:
            scores.append(torch.tensor(float("nan"), device=target.device))
            continue

        sample_b = samples[:, b].reshape(num_samples, -1)[:, flat_mask]
        target_b = target[b].reshape(-1)[flat_mask]

        sample_to_target = torch.linalg.vector_norm(
            sample_b - target_b.unsqueeze(0),
            dim=-1,
        ).mean()

        pairwise = torch.cdist(sample_b, sample_b, p=2)
        pairwise_fair = pairwise.sum() / (num_samples * (num_samples - 1))

        scores.append(sample_to_target - 0.5 * pairwise_fair)

    return torch.stack(scores)

def metric_sums_for_mask(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    alpha: float = 1.0,
    interval_levels: Iterable[float] = DEFAULT_INTERVAL_LEVELS,
) -> Optional[Dict[str, float]]:
    """Compute additive metric components for one masked subset.

    These are summed across batches, then final metrics are computed once at the end.
    """
    samples = samples.detach()
    target = target.detach()

    num_samples = samples.shape[0]
    bool_mask = expand_mask(mask, target)
    numel = int(bool_mask.sum().item())

    if numel == 0:
        return None

    pred_mean = samples.mean(dim=0)
    squared_error = (pred_mean - target).pow(2)

    sample_var = samples.var(dim=0, unbiased=True)

    pairwise_dist = torch.abs(samples[:, None, ...] - samples[None, :, ...])
    offdiag_diversity = pairwise_dist.sum(dim=(0, 1)) / (
        num_samples * (num_samples - 1)
    )

    crps = crps_per_element(samples=samples, target=target, alpha=alpha)
    energy = energy_score_per_task(samples=samples, target=target, mask=mask)
    valid_energy = torch.isfinite(energy)

    row: Dict[str, float] = {
        "num_eval_samples": int(num_samples),
        "numel": int(numel),
        "sse": float(masked_sum(squared_error, bool_mask).item()),
        "crps_sum": float(masked_sum(crps, bool_mask).item()),
        "var_sum": float(masked_sum(sample_var, bool_mask).item()),
        "diversity_sum": float(masked_sum(offdiag_diversity, bool_mask).item()),
        "energy_score_sum": float(energy[valid_energy].sum().item()),
        "energy_score_num_tasks": int(valid_energy.sum().item()),
    }

    for level in interval_levels:
        suffix = level_suffix(level)
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q

        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)

        covered = ((target >= lower) & (target <= upper)).to(target.dtype)
        width = upper - lower

        row[f"coverage_count_{suffix}"] = float(
            masked_sum(covered, bool_mask).item()
        )
        row[f"width_sum_{suffix}"] = float(masked_sum(width, bool_mask).item())

    ranks = (samples <= target.unsqueeze(0)).sum(dim=0).long()
    selected_ranks = ranks[bool_mask].reshape(-1)
    rank_counts = torch.bincount(selected_ranks, minlength=num_samples + 1)

    for rank_idx, count in enumerate(rank_counts.tolist()):
        row[f"rank_{rank_idx:03d}"] = int(count)

    return row


def batch_metric_rows(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    xt: torch.Tensor,
    num_context: int,
    context_range: List[List[float]],
    model_name: str,
    checkpoint_path: str,
    eval_set: str,
    alpha: float = 1.0,
) -> List[Dict[str, float]]:
    """Return metric-sum rows for overall, region, and context buckets."""
    if xt.shape[-1] != 1:
        raise ValueError(
            "This evaluator currently assumes 1D inputs, i.e. xt.shape[-1] == 1."
        )

    x = xt[..., 0]

    context_min = float(context_range[0][0])
    context_max = float(context_range[0][1])

    interp_mask = (x >= context_min) & (x <= context_max)
    extrap_mask = ~interp_mask

    region_masks = {
        "all": None,
        "interpolation": interp_mask,
        "extrapolation": extrap_mask,
    }

    bucket = context_bucket(num_context)
    context_buckets = ["all", bucket]

    rows: List[Dict[str, float]] = []

    for region_name, region_mask in region_masks.items():
        for bucket_name in context_buckets:
            metric_row = metric_sums_for_mask(
                samples=samples,
                target=target,
                mask=region_mask,
                alpha=alpha,
            )

            if metric_row is None:
                continue

            metric_row.update(
                {
                    "model_name": model_name,
                    "checkpoint_path": checkpoint_path,
                    "eval_set": eval_set,
                    "region": region_name,
                    "context_bucket": bucket_name,
                    "num_context": int(num_context),
                }
            )
            rows.append(metric_row)

    return rows


def finalise_metric_rows(rows: List[Dict[str, float]]) -> pd.DataFrame:
    """Aggregate additive metric rows into final scalar metrics."""
    raw = pd.DataFrame(rows)

    if raw.empty:
        return raw

    group_cols = [
        "model_name",
        "checkpoint_path",
        "eval_set",
        "region",
        "context_bucket",
    ]

    sum_cols = [
        col
        for col in raw.columns
        if (
            col in {"numel", "sse", "crps_sum", "var_sum", "diversity_sum","energy_score_sum", "energy_score_num_tasks"}
            or col.startswith("coverage_count_")
            or col.startswith("width_sum_")
            or col.startswith("rank_")
        )
    ]

    summed = raw.groupby(group_cols, as_index=False)[sum_cols].sum()

    first_cols = raw.groupby(group_cols, as_index=False)["num_eval_samples"].first()
    out = summed.merge(first_cols, on=group_cols, how="left")

    out["rmse_pooled"] = (out["sse"] / out["numel"]).pow(0.5)
    out["crps"] = out["crps_sum"] / out["numel"]
    out["ensemble_spread"] = (out["var_sum"] / out["numel"]).pow(0.5)
    out["sample_diversity_offdiag"] = out["diversity_sum"] / out["numel"]
    out["energy_score"] = out["energy_score_sum"] / out["energy_score_num_tasks"].clip(lower=1)

    finite_m_correction = ((out["num_eval_samples"] + 1.0) / out["num_eval_samples"]).pow(
        0.5
    )
    out["spread_skill_ratio"] = (
        finite_m_correction * out["ensemble_spread"] / (out["rmse_pooled"] + 1e-12)
    )

    for level in DEFAULT_INTERVAL_LEVELS:
        suffix = level_suffix(level)
        out[f"coverage_{suffix}"] = out[f"coverage_count_{suffix}"] / out["numel"]
        out[f"width_{suffix}"] = out[f"width_sum_{suffix}"] / out["numel"]

    rank_cols = [col for col in out.columns if col.startswith("rank_")]
    for col in rank_cols:
        out[f"{col}_prob"] = out[col] / out["numel"]

    preferred_cols = [
        "model_name",
        "eval_set",
        "region",
        "context_bucket",
        "num_eval_samples",
        "numel",
        "rmse_pooled",
        "crps",
        "energy_score",
        "ensemble_spread",
        "spread_skill_ratio",
        "sample_diversity_offdiag",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "width_50",
        "width_80",
        "width_90",
        "width_95",
        "checkpoint_path",
    ]

    remaining_cols = [col for col in out.columns if col not in preferred_cols]
    return out[preferred_cols + remaining_cols]