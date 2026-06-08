import copy
import warnings
from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn

from tnp_crps.networks.attention_layers import (
    StochasticLayerNormCrossAttentionLayer,
    StochasticLayerNormSelfAttentionLayer,
)


class StochasticLayerNormTNPTransformerEncoder(nn.Module):
    """TNP transformer encoder using conditional LayerNorm attention blocks."""

    def __init__(
        self,
        num_layers: int,
        mhca_layer: StochasticLayerNormCrossAttentionLayer,
        mhsa_layer: Optional[StochasticLayerNormSelfAttentionLayer] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

    def set_noise(self, noise: torch.Tensor) -> None:
        seen = set()

        for layer in list(self.mhsa_layers) + list(self.mhca_layers):
            layer_id = id(layer)
            if layer_id in seen:
                continue

            if hasattr(layer, "set_noise"):
                layer.set_noise(noise)

            seen.add(layer_id)

    def clear_noise(self) -> None:
        seen = set()

        for layer in list(self.mhsa_layers) + list(self.mhca_layers):
            layer_id = id(layer)
            if layer_id in seen:
                continue

            if hasattr(layer, "clear_noise"):
                layer.clear_noise()

            seen.add(layer_id)

    @check_shapes(
        "xc: [m, nc, d]",
        "xt: [m, nt, d]",
        "mask: [m, nt, nc]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, StochasticLayerNormSelfAttentionLayer):
                xc = mhsa_layer(xc)
            elif isinstance(mhsa_layer, StochasticLayerNormCrossAttentionLayer):
                xc = mhsa_layer(xc, xc)
            else:
                raise TypeError(f"Unknown layer type: {type(mhsa_layer)}")

            xt = mhca_layer(xt, xc)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])