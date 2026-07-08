from __future__ import annotations

import dataclasses
from typing import Literal

import torch

from tnp.data.base import Batch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn


TargetOrder = Literal["ascending", "descending", "given"]


def _repeat_for_samples(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Repeat [B, N, D] tensor into [M*B, N, D]."""
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
) -> torch.Tensor:
    """Return target traversal order [B, Nt]."""
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

    raise ValueError(
        f"Unknown target_order={target_order!r}. "
        "Expected one of: 'ascending', 'descending', 'given'."
    )


@torch.no_grad()
def _sample_one_step(
    *,
    model,
    step_batch: Batch,
) -> torch.Tensor:
    """Sample one target point for each replicated rollout context.

    Args:
        model: Gaussian TNP or DirectTNP/CRPS model.
        step_batch: batch with xt shape [M*B, 1, Dx].

    Returns:
        y_step: [M*B, 1, Dy]
    """
    if isinstance(model, DirectTNP):
        # DirectTNP.sample currently enforces num_samples >= 2 because it was
        # written for CRPS training/evaluation. AR rollout only needs one
        # sample per replicated rollout path, so request two and keep one.
        y_all = model.sample(
            xc=step_batch.xc,
            yc=step_batch.yc,
            xt=step_batch.xt,
            num_samples=2,
        )

        # Expected DirectTNP shape: [S, B, Nt, Dy].
        if y_all.ndim == 4:
            y = y_all[0]
        elif y_all.ndim == 3:
            # Defensive fallback if a future DirectTNP.sample returns [B, Nt, Dy].
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
        raise ValueError(f"Expected one-step sample [B, 1, Dy], got {tuple(y.shape)}.")

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
) -> torch.Tensor:
    """Draw autoregressive rollout samples from a trained NP model.

    This is test-time AR deployment. The model is not retrained.

    Procedure:
        1. Choose an order over target locations.
        2. Replicate the original context once per rollout sample path.
        3. Predict and sample one target at a time.
        4. Append each sampled (x, y) to that rollout path's context.
        5. Return samples in the original target order.

    Args:
        model: trained Gaussian TNP or DirectTNP/CRPS model.
        batch: Batch with xc, yc, xt. Supports B >= 1.
        num_samples: number of AR rollout paths M.
        target_order: target traversal order. For 1D plots, use "ascending".

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

    model.eval()

    num_samples = int(num_samples)
    batch_size, num_targets, _ = batch.xt.shape
    dy = batch.yc.shape[-1]

    order = _target_order_indices(batch.xt, target_order=target_order)

    # Each rollout sample path gets its own evolving context.
    xc_ar = _repeat_for_samples(batch.xc, num_samples)
    yc_ar = _repeat_for_samples(batch.yc, num_samples)

    samples = torch.empty(
        num_samples,
        batch_size,
        num_targets,
        dy,
        device=batch.xt.device,
        dtype=batch.yc.dtype,
    )

    batch_indices = torch.arange(batch_size, device=batch.xt.device)

    for step in range(num_targets):
        target_idx = order[:, step]  # [B]

        # Select the next target x for each original task in the batch.
        x_step_base = batch.xt[batch_indices, target_idx, :].unsqueeze(1)  # [B, 1, Dx]
        x_step_ar = _repeat_for_samples(x_step_base, num_samples)  # [M*B, 1, Dx]

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
        )  # [M*B, 1, Dy]

        expected_shape = (num_samples * batch_size, 1, dy)
        if tuple(y_step_ar.shape) != expected_shape:
            raise ValueError(
                f"One-step sample has wrong shape. Expected {expected_shape}, "
                f"got {tuple(y_step_ar.shape)}."
            )

        y_step_mbd = y_step_ar.reshape(num_samples, batch_size, 1, dy)

        # Store in original target order, not AR traversal order.
        samples[:, batch_indices, target_idx, :] = y_step_mbd[:, :, 0, :]

        # Append sampled target to each rollout path's evolving context.
        xc_ar = torch.cat([xc_ar, x_step_ar], dim=1)
        yc_ar = torch.cat([yc_ar, y_step_ar], dim=1)

    return samples