import torch
import torch.distributions as td
from torch import nn

from tnp.data.base import Batch, ImageBatch
from tnp.models.base import (
    ARConditionalNeuralProcess,
    ConditionalNeuralProcess,
    LatentNeuralProcess,
)
from tnp.models.convcnp import GriddedConvCNP
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.crps import crps_loss


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.distributions.Distribution:
    if isinstance(model, DirectTNP):
        samples = model.sample(
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            num_samples=num_samples,
        )
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp_min(1e-6)
        return td.Normal(mean, std)
        
    if isinstance(model, GriddedConvCNP):
        assert isinstance(batch, ImageBatch)
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, LatentNeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    elif isinstance(model, ARConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    else:
        raise ValueError(f"Unsupported model type for np_pred_fn: {type(model)}")

    return pred_dist


def crps_pred_sample_fn(
    model: DirectTNP,
    batch: Batch,
    num_samples: int | None = None,
) -> torch.Tensor:
    return model.sample(
        xc=batch.xc,
        yc=batch.yc,
        xt=batch.xt,
        num_samples=num_samples,
    )


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Training loss.

    Standard NP/TNP models:
        negative log-likelihood.

    DirectTNP CRPS models:
        marginal fair/almost-fair CRPS.
    """

    if isinstance(model, DirectTNP):
        samples = crps_pred_sample_fn(
            model=model,
            batch=batch,
            num_samples=model.num_samples,
        )

        return crps_loss(
            samples=samples,
            target=batch.yt,
            alpha=model.crps_alpha,
        )

    pred_dist = np_pred_fn(model, batch, num_samples)
    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik