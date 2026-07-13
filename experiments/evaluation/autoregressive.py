from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import torch

from tnp.data.base import Batch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn


TargetOrder = Literal[
    "ascending",
    "descending",
    "given",
    "nearest_context",
    "random",
]

StochLNNoiseMode = Literal[
    "refresh",
    "fixed",
]


def _is_stochln_model(model) -> bool:
    """Detect StochLN CRPS-TNP without importing the subclass directly."""
    return (
        isinstance(model, DirectTNP)
        and hasattr(model, "_set_layernorm_noise")
        and hasattr(model, "_clear_layernorm_noise")
        and hasattr(model, "layernorm_noise_dim")
    )


def _repeat_for_samples(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Repeat [B, N, D] tensor into [M * B, N, D].

    The repeated dimension is sample-major:

        original [B, N, D]
        repeated [M * B, N, D]

    where row index r = sample_idx * B + batch_idx.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected tensor [B, N, D], got {tuple(x.shape)}.")

    batch_size, n_points, dim = x.shape

    return (
        x.unsqueeze(0)
        .expand(num_samples, batch_size, n_points, dim)
        .reshape(num_samples * batch_size, n_points, dim)
        .contiguous()
    )


def _target_order_indices(
    xt: torch.Tensor,
    *,
    target_order: TargetOrder,
    xc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return target traversal order [B_like, Nt].

    Args:
        xt: target inputs [B_like, Nt, Dx].
        target_order: traversal order.
        xc: context inputs [B_like, Nc, Dx], required for nearest_context.

    Returns:
        order: integer indices [B_like, Nt].
    """
    if xt.ndim != 3:
        raise ValueError(f"Expected xt [B, Nt, Dx], got {tuple(xt.shape)}.")

    batch_size, num_targets, _ = xt.shape

    if target_order == "given":
        return (
            torch.arange(num_targets, device=xt.device)
            .view(1, num_targets)
            .expand(batch_size, num_targets)
        )

    if target_order == "ascending":
        return torch.argsort(xt[..., 0], dim=1, descending=False)

    if target_order == "descending":
        return torch.argsort(xt[..., 0], dim=1, descending=True)

    if target_order == "random":
        # Independent random target order for every rollout path.
        return torch.rand(
            batch_size,
            num_targets,
            device=xt.device,
            dtype=xt.dtype,
        ).argsort(dim=1)

    if target_order == "nearest_context":
        if xc is None:
            raise ValueError("target_order='nearest_context' requires xc.")

        if xc.ndim != 3:
            raise ValueError(f"Expected xc [B, Nc, Dx], got {tuple(xc.shape)}.")

        # Distance from every target to its nearest original context point.
        # Shape: [B_like, Nt, Nc]
        distances = (xt[:, :, None, 0] - xc[:, None, :, 0]).abs()
        nearest_distance = distances.amin(dim=-1)  # [B_like, Nt]

        return torch.argsort(nearest_distance, dim=1, descending=False)

    raise ValueError(
        f"Unknown target_order={target_order!r}. "
        "Expected one of: 'ascending', 'descending', 'given', "
        "'nearest_context', 'random'."
    )


@torch.no_grad()
def _sample_one_step(
    *,
    model,
    step_batch: Batch,
    fixed_stochln_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample one target point for each replicated rollout context.

    Args:
        model: Gaussian TNP or DirectTNP/CRPS model.
        step_batch: batch with xt shape [R, 1, Dx].
        fixed_stochln_noise: optional [R, noise_dim] StochLN noise.

    Returns:
        y_step: [R, 1, Dy]
    """
    if fixed_stochln_noise is not None:
        if not _is_stochln_model(model):
            raise ValueError(
                "fixed_stochln_noise was provided, but model does not look like "
                "a StochasticLayerNormTNP."
            )

        if fixed_stochln_noise.shape[0] != step_batch.xc.shape[0]:
            raise ValueError(
                "fixed_stochln_noise batch dimension must match AR batch size. "
                f"Got fixed_stochln_noise={tuple(fixed_stochln_noise.shape)}, "
                f"xc={tuple(step_batch.xc.shape)}."
            )

        # Fixed-noise StochLN AR.
        # This is now an ablation mode, not the default.
        try:
            model._set_layernorm_noise(fixed_stochln_noise)
            y = model.forward(
                xc=step_batch.xc,
                yc=step_batch.yc,
                xt=step_batch.xt,
            )
        finally:
            model._clear_layernorm_noise()

    elif isinstance(model, DirectTNP):
        # Rich's preferred default for the current marginal-CRPS training setup:
        # call model.sample at every AR step, so StochLN/dropout stochasticity is
        # freshly sampled at each autoregressive target step.
        #
        # DirectTNP.sample currently requires num_samples >= 2 because it was
        # written for CRPS training/evaluation. AR rollout needs one sample per
        # replicated path, so request two and keep the first.
        y_all = model.sample(
            xc=step_batch.xc,
            yc=step_batch.yc,
            xt=step_batch.xt,
            num_samples=2,
        )

        # Expected DirectTNP shape: [S, R, 1, Dy].
        if y_all.ndim == 4:
            y = y_all[0]
        elif y_all.ndim == 3:
            # Defensive fallback if a future DirectTNP.sample returns [R, 1, Dy].
            y = y_all
        else:
            raise ValueError(
                "Unexpected DirectTNP one-step sample shape. "
                f"Got {tuple(y_all.shape)}."
            )

    else:
        pred_dist = np_pred_fn(
            model=model,
            batch=step_batch,
            num_samples=1,
        )
        y = pred_dist.sample()

    if y.ndim == 2:
        y = y.unsqueeze(-1)

    if y.ndim != 3:
        raise ValueError(f"Expected one-step sample [R, 1, Dy], got {tuple(y.shape)}.")

    if y.shape[1] != 1:
        raise ValueError(f"Expected one-step target dimension 1, got {tuple(y.shape)}.")

    return y.contiguous()


@torch.no_grad()
def autoregressive_sample_model(
    *,
    model,
    batch: Batch,
    num_samples: int,
    target_order: TargetOrder = "ascending",
    stochln_noise_mode: StochLNNoiseMode = "refresh",
) -> torch.Tensor:
    """Draw autoregressive rollout samples from a trained NP model.

    This is test-time AR deployment. The model is not retrained.

    Default behaviour:
        stochln_noise_mode="refresh"

    This means stochasticity is freshly sampled at every AR step. For the current
    marginal CRPS training objective, this is the main/default interpretation:
    coherence should arise from AR conditioning on previously sampled points.

    Optional ablation:
        stochln_noise_mode="fixed"

    This reuses one StochLN noise vector across each rollout path. This is useful
    for comparison only, not the default result.

    Args:
        model: trained Gaussian TNP or DirectTNP/CRPS model.
        batch: Batch with xc, yc, xt. Supports B >= 1.
        num_samples: number of AR rollout paths M.
        target_order: target traversal order.
        stochln_noise_mode: "refresh" or "fixed".

    Returns:
        samples: [M, B, Nt, Dy], aligned with the original batch.xt order.
    """
    if int(num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")

    if batch.xc.ndim != 3 or batch.yc.ndim != 3 or batch.xt.ndim != 3:
        raise ValueError(
            "Expected batch.xc, batch.yc, batch.xt to all be rank-3 tensors. "
            f"Got xc={tuple(batch.xc.shape)}, yc={tuple(batch.yc.shape)}, "
            f"xt={tuple(batch.xt.shape)}."
        )

    if stochln_noise_mode not in ("refresh", "fixed"):
        raise ValueError(
            f"Unknown stochln_noise_mode={stochln_noise_mode!r}. "
            "Expected 'refresh' or 'fixed'."
        )

    model.eval()

    num_samples = int(num_samples)
    batch_size, num_targets, _ = batch.xt.shape
    dy = batch.yc.shape[-1]

    # Each rollout path gets its own evolving context.
    # R = M * B replicated rollout paths.
    xc_ar = _repeat_for_samples(batch.xc, num_samples)
    yc_ar = _repeat_for_samples(batch.yc, num_samples)
    xt_all_ar = _repeat_for_samples(batch.xt, num_samples)

    num_rollout_paths = xc_ar.shape[0]

    order = _target_order_indices(
        xt_all_ar,
        target_order=target_order,
        xc=xc_ar,
    )  # [R, Nt]

    fixed_stochln_noise = None

    if stochln_noise_mode == "fixed" and _is_stochln_model(model):
        fixed_stochln_noise = torch.randn(
            num_rollout_paths,
            int(model.layernorm_noise_dim),
            device=batch.xc.device,
            dtype=batch.xc.dtype,
        )

    samples_flat = torch.empty(
        num_rollout_paths,
        num_targets,
        dy,
        device=batch.xt.device,
        dtype=batch.yc.dtype,
    )

    rollout_indices = torch.arange(num_rollout_paths, device=batch.xt.device)

    for step in range(num_targets):
        target_idx = order[:, step]  # [R]

        x_step_ar = xt_all_ar[
            rollout_indices,
            target_idx,
            :,
        ].unsqueeze(1)  # [R, 1, Dx]

        y_placeholder = torch.zeros(
            x_step_ar.shape[0],
            1,
            dy,
            device=batch.xt.device,
            dtype=batch.yc.dtype,
        )

        step_batch = dataclasses.replace(
            batch,
            xc=xc_ar,
            yc=yc_ar,
            xt=x_step_ar,
            yt=y_placeholder,
        )

        y_step_ar = _sample_one_step(
            model=model,
            step_batch=step_batch,
            fixed_stochln_noise=fixed_stochln_noise,
        )  # [R, 1, Dy]

        expected_shape = (num_rollout_paths, 1, dy)
        if tuple(y_step_ar.shape) != expected_shape:
            raise ValueError(
                f"One-step sample has wrong shape. Expected {expected_shape}, "
                f"got {tuple(y_step_ar.shape)}."
            )

        # Store in original target order, not AR traversal order.
        samples_flat[rollout_indices, target_idx, :] = y_step_ar[:, 0, :]

        # Append sampled target to each rollout path's evolving context.
        xc_ar = torch.cat([xc_ar, x_step_ar], dim=1)
        yc_ar = torch.cat([yc_ar, y_step_ar], dim=1)

    return samples_flat.reshape(
        num_samples,
        batch_size,
        num_targets,
        dy,
    ).contiguous()


@torch.no_grad()
def denoise_autoregressive_samples(
    *,
    model,
    batch: Batch,
    ar_samples: torch.Tensor,
    num_denoise_samples: int = 32,
) -> torch.Tensor:
    """Denoise AR samples by conditioning on them and returning predictive means.

    This follows the AR-CNP smooth-sample idea:

        1. Generate noisy AR sampled points.
        2. Append those sampled points to the original context.
        3. Query the model again and use the predictive mean as the smooth sample.

    Args:
        model: trained Gaussian TNP or DirectTNP/CRPS model.
        batch: Batch used for AR sampling.
        ar_samples: raw AR samples [M, B, Nt, Dy], aligned with batch.xt.
        num_denoise_samples: number of stochastic samples used to approximate
            the predictive mean for DirectTNP/CRPS models.

    Returns:
        denoised_samples: [M, B, Nt, Dy]
    """
    if ar_samples.ndim != 4:
        raise ValueError(
            "Expected ar_samples [M, B, Nt, Dy], "
            f"got {tuple(ar_samples.shape)}."
        )

    num_samples, batch_size, num_targets, dy = ar_samples.shape

    if batch.xt.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: ar_samples B={batch_size}, "
            f"batch.xt B={batch.xt.shape[0]}."
        )

    if batch.xt.shape[1] != num_targets:
        raise ValueError(
            f"Target size mismatch: ar_samples Nt={num_targets}, "
            f"batch.xt Nt={batch.xt.shape[1]}."
        )

    if batch.yc.shape[-1] != dy:
        raise ValueError(
            f"Output dimension mismatch: ar_samples Dy={dy}, "
            f"batch.yc Dy={batch.yc.shape[-1]}."
        )

    model.eval()

    x_support = (
        batch.xt.unsqueeze(0)
        .expand(num_samples, batch_size, *batch.xt.shape[1:])
        .reshape(num_samples * batch_size, num_targets, batch.xt.shape[-1])
        .contiguous()
    )

    y_support = ar_samples.reshape(
        num_samples * batch_size,
        num_targets,
        dy,
    ).contiguous()

    xc_rep = _repeat_for_samples(batch.xc, num_samples)
    yc_rep = _repeat_for_samples(batch.yc, num_samples)

    xc_denoise = torch.cat([xc_rep, x_support], dim=1)
    yc_denoise = torch.cat([yc_rep, y_support], dim=1)

    y_placeholder = torch.zeros_like(y_support)

    denoise_batch = dataclasses.replace(
        batch,
        xc=xc_denoise,
        yc=yc_denoise,
        xt=x_support,
        yt=y_placeholder,
    )

    if isinstance(model, DirectTNP):
        num_denoise_samples = max(2, int(num_denoise_samples))

        y_all = model.sample(
            xc=denoise_batch.xc,
            yc=denoise_batch.yc,
            xt=denoise_batch.xt,
            num_samples=num_denoise_samples,
        )

        if y_all.ndim != 4:
            raise ValueError(
                "Expected DirectTNP denoise samples [S, M*B, Nt, Dy], "
                f"got {tuple(y_all.shape)}."
            )

        mean_flat = y_all.mean(dim=0)

    else:
        pred_dist = np_pred_fn(
            model=model,
            batch=denoise_batch,
            num_samples=1,
        )
        mean_flat = pred_dist.mean

    expected_shape = (num_samples * batch_size, num_targets, dy)
    if tuple(mean_flat.shape) != expected_shape:
        raise ValueError(
            f"Denoised mean has wrong shape. Expected {expected_shape}, "
            f"got {tuple(mean_flat.shape)}."
        )

    return mean_flat.reshape(
        num_samples,
        batch_size,
        num_targets,
        dy,
    ).contiguous()