from typing import Optional

import torch
from torch import nn


class DirectTNP(nn.Module):
    """Direct-output TNP for CRPS training.

    Unlike the original TNP, this model does not apply a Gaussian likelihood.
    It returns direct predictions with shape [B, Nt, Dy].

    Repeated stochastic forward passes produce samples for CRPS.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_samples: int = 2,
        crps_alpha: float = 1.0,
        use_mc_dropout: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_samples = num_samples
        self.crps_alpha = crps_alpha
        self.use_mc_dropout = use_mc_dropout

    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        zt = self.encoder(xc, yc, xt)
        return self.decoder(zt, xt)

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

        dropout_states = None

        if self.use_mc_dropout:
            dropout_states = self._enable_dropout_modules()

        try:
            samples = [self.forward(xc, yc, xt) for _ in range(num_samples)]
            return torch.stack(samples, dim=0)
        finally:
            if dropout_states is not None:
                self._restore_dropout_modules(dropout_states)

    def _enable_dropout_modules(self):
        """Enable dropout even if model is in eval mode.

        Needed for MC dropout validation/test sampling.
        """
        dropout_states = []

        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_states.append((module, module.training))
                module.train(True)

        return dropout_states

    @staticmethod
    def _restore_dropout_modules(dropout_states):
        for module, was_training in dropout_states:
            module.train(was_training)