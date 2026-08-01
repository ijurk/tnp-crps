from __future__ import annotations

from typing import Tuple

import torch

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorUniformInput,
)


class SawtoothGroundTruthPredictor(GroundTruthPredictor):
    """Ground-truth helper for the realised sawtooth task.

    This stores the latent sawtooth parameters sampled for the task, so dense
    function plots can show the exact noiseless realised function.

    This is not a posterior over hidden sawtooth parameters given context.
    """

    # Avoid plotting this as an oracle posterior summary. The useful thing here
    # is the exact realised latent function from latent_function(...).
    plot_posterior_summary = False

    def __init__(
        self,
        *,
        freq: torch.Tensor,
        direction: torch.Tensor,
        offset: torch.Tensor,
        noise_std: float = 0.0,
        jitter: float = 1e-6,
    ):
        self.freq = freq.detach().cpu()
        self.direction = direction.detach().cpu()
        self.offset = offset.detach().cpu()
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)
        self.dense_ground_truth_label = "GT realised function"

    def _expanded_params(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latent sawtooth parameters matching requested batch size.

        Training/validation batches may store parameters for B=16 tasks, while
        plotting utilities often slice the tensors to B=1 but keep the same
        gt_pred object. In that case, use the first stored task's parameters.
        """
        freq = self.freq.to(device=device, dtype=dtype).reshape(-1)
        offset = self.offset.to(device=device, dtype=dtype).reshape(-1)

        direction = self.direction.to(device=device, dtype=dtype)
        if direction.ndim == 1:
            direction = direction[:, None]
        else:
            direction = direction.reshape(direction.shape[0], -1)

        stored_batch_size = int(freq.shape[0])

        if int(offset.shape[0]) != stored_batch_size:
            raise ValueError(
                "Stored sawtooth offset batch size does not match freq batch size. "
                f"freq={freq.shape}, offset={offset.shape}."
            )

        if int(direction.shape[0]) != stored_batch_size:
            raise ValueError(
                "Stored sawtooth direction batch size does not match freq batch size. "
                f"freq={freq.shape}, direction={direction.shape}."
            )

        if stored_batch_size == batch_size:
            return freq, direction, offset

        if stored_batch_size == 1:
            return (
                freq.expand(batch_size),
                direction.expand(batch_size, -1),
                offset.expand(batch_size),
            )

        if batch_size <= stored_batch_size:
            return (
                freq[:batch_size],
                direction[:batch_size],
                offset[:batch_size],
            )

        raise ValueError(
            "Stored sawtooth latent batch size does not match requested batch size. "
            f"Stored={stored_batch_size}, requested={batch_size}."
        )

    def latent_function(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate noiseless realised sawtooth f(x).

        Args:
            x: [B, N, D]

        Returns:
            f: [B, N, 1]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, N, D], got {x.shape}.")

        batch_size = x.shape[0]
        freq, direction, offset = self._expanded_params(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        projected = x @ direction[:, :, None]  # [B, N, 1]

        f = (
            freq[:, None, None]
            * (projected - offset[:, None, None])
        ) % 1.0

        return f

    def sample_outputs(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        if sample_shape != torch.Size():
            raise NotImplementedError(
                "SawtoothGroundTruthPredictor.sample_outputs currently supports "
                "only empty sample_shape."
            )

        f = self.latent_function(x)

        if self.noise_std > 0.0:
            return f + self.noise_std * torch.randn_like(f)

        return f

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor | None = None,
    ):
        """Old-style gt_pred contract: mean, std, gt_loglik.

        This is the hidden-parameter oracle, not the posterior over sawtooth
        parameters given context. Function plots should use latent_function(...).
        """
        mean = self.latent_function(xt)[..., 0]
        std = torch.full_like(mean, max(self.noise_std, self.jitter))

        gt_loglik = None
        if yt is not None:
            dist = torch.distributions.Normal(loc=mean, scale=std)
            gt_loglik = dist.log_prob(yt[..., 0])

        return mean, std, gt_loglik


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
        jitter: float = 1e-6,
        return_gt_pred: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)
        self.return_gt_pred = bool(return_gt_pred)

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, SawtoothGroundTruthPredictor | None]:
        """Sample sawtooth outputs.

        Args:
            x: [B, N, D]

        Returns:
            y: [B, N, 1]
            gt_pred: SawtoothGroundTruthPredictor or None
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
        phase = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        offset = phase / freq

        gt_pred = SawtoothGroundTruthPredictor(
            freq=freq,
            direction=direction,
            offset=offset,
            noise_std=self.noise_std,
            jitter=self.jitter,
        )

        f = gt_pred.latent_function(x)

        if self.noise_std > 0.0:
            y = f + self.noise_std * torch.randn_like(f)
        else:
            y = f

        return y.detach(), gt_pred if self.return_gt_pred else None


class SawtoothGenerator(
    SawtoothGeneratorBase,
    SyntheticGeneratorUniformInput,
):
    pass