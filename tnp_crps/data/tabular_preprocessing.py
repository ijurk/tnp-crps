"""Context-only preprocessing for raw tabular tasks."""

from __future__ import annotations

from typing import Tuple

import torch


def _sample_std(x: torch.Tensor) -> torch.Tensor:
    """Columnwise sample standard deviation."""
    return x.std(
        dim=0,
        unbiased=x.shape[0] > 1,
        keepdim=True,
    )


def tabicl_preprocess_from_context(
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    outlier_threshold: float,
    standardized_clip: float,
    zero_constant_dimensions: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply a context-fitted version of TabICL preprocessing.

    The procedure mirrors TabICL's two-stage outlier removal and
    standard scaling, but all fitted statistics and clipping bounds
    are calculated from context rows only.
    """
    if context.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Expected rank-two context and target tensors, got "
            f"{tuple(context.shape)} and {tuple(target.shape)}."
        )

    if context.shape[-1] != target.shape[-1]:
        raise ValueError(
            "Context and target dimensionalities differ: "
            f"{context.shape[-1]} versus {target.shape[-1]}."
        )

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    if outlier_threshold <= 0.0:
        raise ValueError(
            "outlier_threshold must be positive."
        )

    if standardized_clip <= 0.0:
        raise ValueError(
            "standardized_clip must be positive."
        )

    # First-pass context statistics.
    initial_mean = context.mean(
        dim=0,
        keepdim=True,
    )
    initial_std = _sample_std(context).clamp_min(
        epsilon
    )

    initial_lower = (
        initial_mean
        - outlier_threshold * initial_std
    )
    initial_upper = (
        initial_mean
        + outlier_threshold * initial_std
    )

    inlier_mask = (
        (context >= initial_lower)
        & (context <= initial_upper)
        & torch.isfinite(context)
    )

    counts = inlier_mask.sum(
        dim=0,
        keepdim=True,
    )

    safe_counts = counts.clamp_min(1).to(
        dtype=context.dtype
    )

    masked_values = torch.where(
        inlier_mask,
        context,
        torch.zeros_like(context),
    )

    robust_mean = (
        masked_values.sum(dim=0, keepdim=True)
        / safe_counts
    )

    centered = torch.where(
        inlier_mask,
        context - robust_mean,
        torch.zeros_like(context),
    )

    variance_denominator = (
        counts - 1
    ).clamp_min(1).to(dtype=context.dtype)

    robust_variance = (
        centered.square().sum(
            dim=0,
            keepdim=True,
        )
        / variance_denominator
    )

    robust_std = robust_variance.sqrt()

    # Match TabICL's fallback when too few inliers remain.
    robust_mean = torch.where(
        counts > 0,
        robust_mean,
        initial_mean,
    )
    robust_std = torch.where(
        counts > 1,
        robust_std,
        torch.zeros_like(robust_std),
    )

    lower = (
        robust_mean
        - outlier_threshold * robust_std
    )
    upper = (
        robust_mean
        + outlier_threshold * robust_std
    )

    context_clipped = torch.maximum(
        torch.minimum(context, upper),
        lower,
    )
    target_clipped = torch.maximum(
        torch.minimum(target, upper),
        lower,
    )

    mean = context_clipped.mean(
        dim=0,
        keepdim=True,
    )
    raw_std = _sample_std(context_clipped)

    constant = raw_std < epsilon
    safe_std = raw_std.clamp_min(epsilon)

    context_scaled = (
        context_clipped - mean
    ) / safe_std
    target_scaled = (
        target_clipped - mean
    ) / safe_std

    context_scaled = context_scaled.clamp(
        min=-standardized_clip,
        max=standardized_clip,
    )
    target_scaled = target_scaled.clamp(
        min=-standardized_clip,
        max=standardized_clip,
    )

    if zero_constant_dimensions:
        context_scaled = torch.where(
            constant,
            torch.zeros_like(context_scaled),
            context_scaled,
        )
        target_scaled = torch.where(
            constant,
            torch.zeros_like(target_scaled),
            target_scaled,
        )

    return (
        context_scaled,
        target_scaled,
        mean,
        safe_std,
    )