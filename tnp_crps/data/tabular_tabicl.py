"""Raw numerical TabICL GraphSCM task source.

This module reuses the causal graph and mechanism generation from the
pinned TabICL repository while deliberately excluding:

- categorical feature generation;
- full-sequence outlier clipping;
- full-sequence standardisation;
- padding and feature permutation.

The source returns raw tasks. Context-only preprocessing, support checks,
padding and permutation are performed by TabularRegressionGenerator.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch

from tabicl.prior._dataset import should_filter
from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._dataset import RandomDataset


class TabICLGraphTaskSource:
    """Generate raw numerical regression tasks from TabICL GraphSCM."""

    def __init__(
        self,
        *,
        min_features: int = 2,
        max_features: int = 20,
        device: str = "cpu",
        max_generation_attempts: int = 100,
        prior_config: Optional[Mapping[str, Any]] = None,
        tabicl_commit: str = (
            "46b91961db4f8873dd049ec09990698a435e1e29"
        ),
    ) -> None:
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.device = str(device)
        self.max_generation_attempts = int(
            max_generation_attempts
        )
        self.tabicl_commit = str(tabicl_commit)

        config_values = dict(prior_config or {})
        self.config = PriorConfig(**config_values)

        if self.min_features < 1:
            raise ValueError("min_features must be positive.")

        if self.max_features < self.min_features:
            raise ValueError(
                "max_features must be greater than or equal to "
                "min_features."
            )

        if self.device != "cpu":
            raise ValueError(
                "TabICL GraphSCM generation is currently restricted "
                "to device='cpu'."
            )

        if self.max_generation_attempts < 1:
            raise ValueError(
                "max_generation_attempts must be positive."
            )

    def _sample_num_features(self) -> int:
        """Match GraphPrior's feature-count sampling convention."""
        return int(
            round(
                float(
                    np.random.uniform(
                        self.min_features,
                        self.max_features,
                    )
                )
            )
        )

    @staticmethod
    def _remove_constant_features(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Remove exactly constant columns without padding the result."""
        minimum = x.amin(dim=0)
        maximum = x.amax(dim=0)
        keep = maximum != minimum

        x_filtered = x[:, keep]
        active_features = int(keep.sum().item())

        return x_filtered, active_features

    @torch.no_grad()
    def sample_task(
        self,
        seq_len: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Return raw x [N, D], raw y [N, 1], and metadata."""
        seq_len = int(seq_len)

        if seq_len < 2:
            raise ValueError(
                f"seq_len must be at least two, got {seq_len}."
            )

        last_rejection = "unknown"

        for _ in range(self.max_generation_attempts):
            requested_features = self._sample_num_features()

            context = Context(
                config=self.config,
                device=self.device,
            )

            # cat_size=0 denotes a numerical feature in TabICL.
            properties = DatasetProperties(
                n_train=seq_len,
                n_test=0,
                cat_sizes={
                    "x": [0] * requested_features,
                    "y": [0],
                },
            )

            dataset = RandomDataset(context).sample(properties)
            data = dataset.get_concat_tensors()

            if "x_cat" in data or "y_cat" in data:
                raise RuntimeError(
                    "Categorical tensors were generated despite the "
                    "numerical-only DatasetProperties."
                )

            x = data.get("x_num")
            y = data.get("y_num")

            if x is None:
                last_rejection = "missing x_num"
                continue

            if y is None:
                last_rejection = "missing y_num"
                continue

            x = (
                x.detach()
                .to(device="cpu", dtype=torch.float32)
            )
            y = (
                y.detach()
                .to(device="cpu", dtype=torch.float32)
            )

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            if x.ndim != 2:
                last_rejection = (
                    f"x has unexpected shape {tuple(x.shape)}"
                )
                continue

            if y.ndim != 2 or y.shape[-1] != 1:
                last_rejection = (
                    f"y has unexpected shape {tuple(y.shape)}"
                )
                continue

            if x.shape[0] != seq_len:
                last_rejection = (
                    f"x has {x.shape[0]} rows, expected {seq_len}"
                )
                continue

            if y.shape[0] != seq_len:
                last_rejection = (
                    f"y has {y.shape[0]} rows, expected {seq_len}"
                )
                continue

            if not torch.isfinite(x).all():
                last_rejection = "x contains NaN or Inf"
                continue

            if not torch.isfinite(y).all():
                last_rejection = "y contains NaN or Inf"
                continue

            x, active_features = self._remove_constant_features(x)

            if active_features < 1:
                last_rejection = (
                    "all generated features were constant"
                )
                continue

            # This is effectively a no-op under the pinned defaults,
            # but it preserves TabICL behaviour if the filtering flags
            # are enabled explicitly in the future.
            if should_filter(
                x,
                y.squeeze(-1),
                self.config,
                is_classif=False,
            ):
                last_rejection = (
                    "task rejected by TabICL predictability filter"
                )
                continue

            metadata = {
                "source": "tabicl_graph_scm_raw_numeric",
                "tabicl_commit": self.tabicl_commit,
                "seq_len": seq_len,
                "requested_num_features": requested_features,
                "active_num_features": active_features,
                "prior_config": repr(self.config),
                "full_sequence_preprocessing": False,
                "categorical_features": False,
            }

            return (
                x.contiguous(),
                y.contiguous(),
                metadata,
            )

        raise RuntimeError(
            "Unable to generate a valid raw TabICL GraphSCM task "
            f"after {self.max_generation_attempts} attempts. "
            f"Last rejection: {last_rejection}."
        )