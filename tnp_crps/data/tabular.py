"""Source-independent synthetic tabular regression generator."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

import torch
import math

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch, SyntheticGenerator


class TabularTaskSource(Protocol):
    """Protocol implemented by raw tabular task sources."""

    def sample_task(
        self,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return raw x [N, D], raw y [N, 1], and task metadata."""


def _standardize_from_context(
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    zero_constant_dimensions: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fit a z-score transform on context rows and apply it to both sets."""
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

    mean = context.mean(dim=0, keepdim=True)
    std = context.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    )

    constant = std < float(epsilon)
    safe_std = torch.where(
        constant,
        torch.ones_like(std),
        std,
    )

    context_scaled = (context - mean) / safe_std
    target_scaled = (target - mean) / safe_std

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

    return context_scaled, target_scaled, mean, safe_std


class TabularRegressionGenerator(SyntheticGenerator):
    """Generate standardized tabular regression tasks from a raw source.

    Processing order:

        sample raw task
        -> split context and targets
        -> fit x and y transforms on context only
        -> apply transforms to both sets
        -> pad and optionally permute features
        -> return SyntheticBatch
    """

    def __init__(
        self,
        *,
        source: TabularTaskSource,
        max_input_features: int = 20,
        epsilon: float = 1.0e-6,
        min_context_target_std: float = 1.0e-4,
        max_task_attempts: int = 32,
        max_abs_standardized_input: float = 50.0,
        max_abs_standardized_target: float = 50.0,
        permute_features: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.source = source
        self.max_input_features = int(max_input_features)
        self.epsilon = float(epsilon)
        self.min_context_target_std = float(
            min_context_target_std
        )
        self.max_task_attempts = int(max_task_attempts)
        self.max_abs_standardized_input = float(max_abs_standardized_input)
        self.max_abs_standardized_target = float(max_abs_standardized_target)
        self.permute_features = bool(permute_features)

        if not hasattr(self.source, "sample_task"):
            raise TypeError(
                "source must implement sample_task(seq_len)."
            )

        if self.max_input_features < 1:
            raise ValueError(
                "max_input_features must be positive."
            )

        if int(self.dim) != self.max_input_features:
            raise ValueError(
                "SyntheticGenerator.dim must equal max_input_features. "
                f"Got dim={self.dim} and "
                f"max_input_features={self.max_input_features}."
            )

        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        if self.min_context_target_std <= 0.0:
            raise ValueError(
                "min_context_target_std must be positive."
            )

        if self.max_task_attempts < 1:
            raise ValueError(
                "max_task_attempts must be at least one."
            )

        if self.max_abs_standardized_input <= 0.0:
            raise ValueError(
                "max_abs_standardized_input must be positive."
            )

        if self.max_abs_standardized_target <= 0.0:
            raise ValueError(
                "max_abs_standardized_target must be positive."
            )

    def _sample_one_task(
        self,
        *,
        nc: int,
        nt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total = nc + nt
        last_rejection = "unknown"

        for _ in range(self.max_task_attempts):
            x_raw, y_raw, _ = self.source.sample_task(total)

            x_raw = (
                x_raw.detach()
                .to(device="cpu", dtype=torch.float32)
            )
            y_raw = (
                y_raw.detach()
                .to(device="cpu", dtype=torch.float32)
            )

            if y_raw.ndim == 1:
                y_raw = y_raw.unsqueeze(-1)

            if x_raw.ndim != 2:
                last_rejection = (
                    f"x_raw has shape {tuple(x_raw.shape)}"
                )
                continue

            if y_raw.ndim != 2 or y_raw.shape[-1] != 1:
                last_rejection = (
                    f"y_raw has shape {tuple(y_raw.shape)}"
                )
                continue

            if x_raw.shape[0] != total or y_raw.shape[0] != total:
                last_rejection = (
                    "incorrect sequence length: "
                    f"x={tuple(x_raw.shape)}, "
                    f"y={tuple(y_raw.shape)}, "
                    f"expected {total}"
                )
                continue

            num_features = int(x_raw.shape[-1])

            if num_features > self.max_input_features:
                last_rejection = (
                    f"{num_features} features exceed "
                    f"maximum {self.max_input_features}"
                )
                continue

            if not torch.isfinite(x_raw).all():
                last_rejection = "x contains NaN or Inf"
                continue

            if not torch.isfinite(y_raw).all():
                last_rejection = "y contains NaN or Inf"
                continue

            xc_raw = x_raw[:nc]
            xt_raw = x_raw[nc:]

            yc_raw = y_raw[:nc]
            yt_raw = y_raw[nc:]

            context_target_std = yc_raw.std(
                dim=0,
                unbiased=False,
            )

            if (
                not torch.isfinite(context_target_std).all()
                or float(context_target_std.min().item())
                < self.min_context_target_std
            ):
                last_rejection = (
                    "context target standard deviation is too small: "
                    f"{context_target_std.tolist()}"
                )
                continue

            xc, xt, _, _ = _standardize_from_context(
                xc_raw,
                xt_raw,
                epsilon=self.epsilon,
                zero_constant_dimensions=True,
            )

            max_abs_input = float(
                torch.maximum(
                    xc.abs().max(),
                    xt.abs().max(),
                ).item()
            )

            if (
                not math.isfinite(max_abs_input)
                or max_abs_input > self.max_abs_standardized_input
            ):
                last_rejection = (
                    "standardized input magnitude exceeds support bound: "
                    f"max_abs_x={max_abs_input:.6g}, "
                    f"bound={self.max_abs_standardized_input:.6g}"
                )
                continue

            yc, yt, _, _ = _standardize_from_context(
                yc_raw,
                yt_raw,
                epsilon=self.epsilon,
                zero_constant_dimensions=False,
            )

            max_abs_target = float(
                torch.maximum(
                    yc.abs().max(),
                    yt.abs().max(),
                ).item()
            )

            if (
                not math.isfinite(max_abs_target)
                or max_abs_target > self.max_abs_standardized_target
            ):
                last_rejection = (
                    "standardized target magnitude exceeds support bound: "
                    f"max_abs_y={max_abs_target:.6g}, "
                    f"bound={self.max_abs_standardized_target:.6g}"
                )
                continue

            x = torch.cat([xc, xt], dim=0)
            y = torch.cat([yc, yt], dim=0)

            padding = self.max_input_features - num_features

            if padding > 0:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            x.shape[0],
                            padding,
                            dtype=x.dtype,
                            device=x.device,
                        ),
                    ],
                    dim=-1,
                )

            if self.permute_features:
                permutation = torch.randperm(
                    self.max_input_features,
                    device=x.device,
                )
                x = x[:, permutation]

            if not torch.isfinite(x).all():
                last_rejection = (
                    "standardized x contains NaN or Inf"
                )
                continue

            if not torch.isfinite(y).all():
                last_rejection = (
                    "standardized y contains NaN or Inf"
                )
                continue

            return x.contiguous(), y.contiguous()

        raise RuntimeError(
            "Unable to sample a valid tabular task after "
            f"{self.max_task_attempts} attempts. "
            f"Last rejection: {last_rejection}."
        )

    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        nc = int(nc)
        nt = int(nt)
        batch_size = int(batch_shape.numel())

        if nc < 1:
            raise ValueError(f"nc must be positive, got {nc}.")

        if nt < 1:
            raise ValueError(f"nt must be positive, got {nt}.")

        x_tasks = []
        y_tasks = []

        for _ in range(batch_size):
            x_task, y_task = self._sample_one_task(
                nc=nc,
                nt=nt,
            )
            x_tasks.append(x_task)
            y_tasks.append(y_task)

        x = torch.stack(x_tasks, dim=0)
        y = torch.stack(y_tasks, dim=0)

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=None,
        )

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "TabularRegressionGenerator overrides sample_batch directly."
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Optional[GroundTruthPredictor],
    ]:
        raise NotImplementedError(
            "TabularRegressionGenerator overrides sample_batch directly."
        )