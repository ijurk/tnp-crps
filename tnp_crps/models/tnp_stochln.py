from typing import Optional

import torch
from torch import nn

from tnp_crps.models.tnp_crps import DirectTNP


class StochasticLayerNormTNP(DirectTNP):
    """Direct-output CRPS TNP with conditional LayerNorm noise.

    This model is trained from scratch using CRPS. It uses the same direct
    sample-output head as other CRPS models, but predictive samples are produced
    by injecting sample-level latent noise into conditional LayerNorm layers.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_samples: int = 4,
        crps_alpha: float = 1.0,
        layernorm_noise_dim: int = 32,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            num_samples=num_samples,
            crps_alpha=crps_alpha,
            use_mc_dropout=False,
        )

        self.layernorm_noise_dim = layernorm_noise_dim

        transformer_encoder = getattr(self.encoder, "transformer_encoder", None)
        if transformer_encoder is None or not hasattr(transformer_encoder, "set_noise"):
            raise RuntimeError(
                "StochasticLayerNormTNP expects encoder.transformer_encoder "
                "to implement set_noise(...) and clear_noise(...)."
            )

    def _set_layernorm_noise(self, noise: torch.Tensor) -> None:
        self.encoder.transformer_encoder.set_noise(noise)

    def _clear_layernorm_noise(self) -> None:
        self.encoder.transformer_encoder.clear_noise()

    def sample(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Return predictive samples with shape [M, B, Nt, Dy]."""

        if num_samples is None:
            num_samples = self.num_samples

        if num_samples < 2:
            raise ValueError("CRPS training requires at least 2 samples.")

        batch_size = xc.shape[0]

        xc_rep = xc.repeat_interleave(num_samples, dim=0)
        yc_rep = yc.repeat_interleave(num_samples, dim=0)
        xt_rep = xt.repeat_interleave(num_samples, dim=0)

        noise = torch.randn(
            batch_size * num_samples,
            self.layernorm_noise_dim,
            device=xc.device,
            dtype=xc.dtype,
        )

        try:
            self._set_layernorm_noise(noise)

            pred_rep = self.forward(xc_rep, yc_rep, xt_rep)
            pred = pred_rep.reshape(
                batch_size,
                num_samples,
                *pred_rep.shape[1:],
            )

            return pred.permute(1, 0, 2, 3).contiguous()

        finally:
            self._clear_layernorm_noise()