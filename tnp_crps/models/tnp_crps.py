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

        # try:
        #     samples = [self.forward(xc, yc, xt) for _ in range(num_samples)]
        #     return torch.stack(samples, dim=0)
        
        # vectorise MC dropout sampling
        try:
            batch_size = xc.shape[0]

            xc_rep = xc.repeat_interleave(num_samples, dim=0)
            yc_rep = yc.repeat_interleave(num_samples, dim=0)
            xt_rep = xt.repeat_interleave(num_samples, dim=0)

            pred_rep = self.forward(xc_rep, yc_rep, xt_rep)
            # pred_rep: [B * M, Nt, Dy]

            pred = pred_rep.reshape(
                batch_size,
                num_samples,
                *pred_rep.shape[1:],
            )

            # [B, M, Nt, Dy] -> [M, B, Nt, Dy]
            return pred.permute(1, 0, 2, 3).contiguous()
        
        finally:
            if dropout_states is not None:
                self._restore_dropout_modules(dropout_states)

    def _enable_dropout_modules(self):
        """Enable dropout even if model is in eval mode.

        This matters during validation/test, where Lightning may set model.eval().
        We still need dropout active to generate MC-dropout samples.
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

            