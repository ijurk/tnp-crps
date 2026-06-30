from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConditionalLayerNorm(nn.Module):
    """LayerNorm with noise-conditioned affine parameters.

    The base LayerNorm normalises activations as usual. A per-task latent noise
    vector then produces an additive perturbation to the LayerNorm scale and
    shift parameters.

    Expected input shape:
        x: [B, ..., D]

    Expected noise shape:
        noise: [B, noise_dim]

    The same noise vector is broadcast across all token/target positions for
    each batch element, giving coherent sample-level perturbations.
    """

    def __init__(
        self,
        normalized_shape,
        noise_dim: int = 32,
        hidden_dim: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, torch.Size):
            normalized_shape = tuple(normalized_shape)
        else:
            normalized_shape = tuple(normalized_shape)

        if len(normalized_shape) != 1:
            raise ValueError(
                "ConditionalLayerNorm currently expects a 1D normalized_shape. "
                f"Got normalized_shape={normalized_shape}."
            )

        self.normalized_shape: Tuple[int, ...] = normalized_shape
        self.noise_dim = noise_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        feature_dim = normalized_shape[0]

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(feature_dim))
            self.bias = nn.Parameter(torch.zeros(feature_dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if hidden_dim is None:
            hidden_dim = max(64, 2 * noise_dim)

        self.noise_to_affine = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )

        # Start close to standard LayerNorm. The final layer can learn non-zero
        # stochastic perturbations during CRPS training.
        final_layer = self.noise_to_affine[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_layer.bias)

        self._noise: Optional[torch.Tensor] = None

    def set_noise(self, noise: torch.Tensor) -> None:
        self._noise = noise

    def clear_noise(self) -> None:
        self._noise = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.layer_norm(
            x,
            self.normalized_shape,
            weight=None,
            bias=None,
            eps=self.eps,
        )

        if self._noise is None:
            # Deterministic fallback for ordinary forward calls:
            # standard LayerNorm with the learned base affine parameters,
            # and no stochastic perturbation.
            if self.weight is None:
                return y
        
            view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
            weight = self.weight.view(*view_shape)
            bias = self.bias.view(*view_shape)
        
            return y * weight + bias
        
        noise = self._noise.to(device=x.device, dtype=x.dtype)

        if noise.shape[0] != x.shape[0]:
            raise ValueError(
                f"Noise batch dimension must match input batch dimension. "
                f"Got noise.shape={noise.shape}, x.shape={x.shape}."
            )

        delta_weight, delta_bias = self.noise_to_affine(noise).chunk(2, dim=-1)

        if self.weight is None:
            weight = 1.0 + delta_weight
            bias = delta_bias
        else:
            weight = self.weight.unsqueeze(0) + delta_weight
            bias = self.bias.unsqueeze(0) + delta_bias

        view_shape = [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1]]
        weight = weight.view(*view_shape)
        bias = bias.view(*view_shape)

        return y * weight + bias
