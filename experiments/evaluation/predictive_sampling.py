from __future__ import annotations

from typing import Iterable, Mapping

import torch

from tnp.data.base import Batch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn


def sampling_seed(
    *,
    base_seed: int,
    source_offset: int,
    batch_index: int,
    condition_index: int = 0,
) -> int:
    """Stable source/batch/condition-specific sampling seed."""
    source_offset = int(source_offset)
    if source_offset < 1:
        raise ValueError(
            f"source_offset must be a positive integer, got {source_offset}."
        )
    if int(batch_index) < 0 or int(condition_index) < 0:
        raise ValueError("batch_index and condition_index must be non-negative.")
    return (
        int(base_seed)
        + 10_000_000 * source_offset
        + 1_000_000 * int(condition_index)
        + int(batch_index)
    )


def validate_sampling_offsets(entries: Iterable[Mapping[str, object]]) -> None:
    offsets = []
    for entry in entries:
        if "sampling_seed_offset" not in entry:
            raise KeyError(
                f"Source {entry.get('name', '<unnamed>')!r} is missing "
                "sampling_seed_offset."
            )
        offsets.append(int(entry["sampling_seed_offset"]))
    if len(offsets) != len(set(offsets)):
        raise ValueError(f"sampling_seed_offset values must be unique: {offsets}")
    if any(value < 1 for value in offsets):
        raise ValueError(f"sampling_seed_offset values must be positive: {offsets}")


@torch.no_grad()
def sample_model_chunked(
    *,
    model: torch.nn.Module,
    batch: Batch,
    num_samples: int,
    chunk_size: int,
) -> torch.Tensor:
    """Return [M, B, Nt, Dy] predictive samples with bounded memory use."""
    num_samples = int(num_samples)
    chunk_size = int(chunk_size)
    if num_samples < 2:
        raise ValueError(f"num_samples must be at least two, got {num_samples}.")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be at least two, got {chunk_size}.")

    if isinstance(model, DirectTNP):
        chunks = []
        remaining = num_samples
        while remaining > 0:
            retained = min(chunk_size, remaining)
            requested = max(2, retained)
            chunk = model.sample(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                num_samples=requested,
            )
            chunks.append(chunk[:retained])
            remaining -= retained
        samples = torch.cat(chunks, dim=0)
    else:
        pred_dist = np_pred_fn(model=model, batch=batch, num_samples=num_samples)
        if not isinstance(pred_dist, torch.distributions.Normal):
            raise TypeError(
                "Gaussian baseline sampling requires torch.distributions.Normal; "
                f"got {type(pred_dist)}."
            )
        samples = pred_dist.sample((num_samples,))

    expected = (
        num_samples,
        int(batch.yt.shape[0]),
        int(batch.yt.shape[1]),
        int(batch.yt.shape[2]),
    )
    if tuple(samples.shape) != expected:
        raise ValueError(
            f"Predictive samples have wrong shape. Expected {expected}, "
            f"got {tuple(samples.shape)}."
        )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Predictive samples contain non-finite values.")
    return samples
