from __future__ import annotations

from typing import Tuple

import torch

from tnp.data.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorUniformInput,
)


class SawtoothGeneratorBase(SyntheticGenerator):
    """Random sawtooth process.

    This is based on the sawtooth synthetic process from
    Bruinsma et al. (2023), Autoregressive Conditional Neural Processes.

    For 1D:
        f(x) = (omega * (direction * x - offset)) mod 1

    where:
        omega     ~ Uniform(min_freq, max_freq)
        direction ~ {-1, +1}
        offset    chosen so phase is random

    The output is in approximately [0, 1], optionally with observation noise.
    """

    def __init__(
        self,
        *,
        min_freq: float,
        max_freq: float,
        noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.noise_std = float(noise_std)

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Sample sawtooth outputs.

        Args:
            x: [B, N, D]

        Returns:
            y: [B, N, 1]
            gt_pred: None
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, N, D], got {x.shape}.")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected final dim={self.dim}, got {x.shape[-1]}.")

        batch_size = x.shape[0]

        # Frequency per task.
        freq = (
            torch.rand(batch_size, device=x.device, dtype=x.dtype)
            * (self.max_freq - self.min_freq)
            + self.min_freq
        )

        # Direction per task and input dimension. In 1D this is ±1.
        direction = torch.where(
            torch.rand(batch_size, self.dim, device=x.device, dtype=x.dtype) < 0.5,
            -torch.ones(batch_size, self.dim, device=x.device, dtype=x.dtype),
            torch.ones(batch_size, self.dim, device=x.device, dtype=x.dtype),
        )

        # Random phase/offset per task.
        # External TNP used offset = U(0, 1/freq), which is equivalent to a
        # random phase after multiplying by freq.
        phase = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        offset = phase / freq

        projected = x @ direction[:, :, None]  # [B, N, 1]

        f = (
            freq[:, None, None]
            * (projected - offset[:, None, None])
        ) % 1.0

        if self.noise_std > 0.0:
            y = f + self.noise_std * torch.randn_like(f)
        else:
            y = f

        return y.detach(), None


class SawtoothGenerator(
    SawtoothGeneratorBase,
    SyntheticGeneratorUniformInput,
):
    pass