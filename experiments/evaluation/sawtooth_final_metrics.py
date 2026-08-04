from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from evaluation.metrics import (
    crps_per_element_sorted,
    energy_score_per_task,
)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C")


def task_fingerprints(batch: Any) -> List[str]:
    """Stable task fingerprints from the complete observed task tensors."""
    required = ("xc", "yc", "xt", "yt")
    if any(not hasattr(batch, name) for name in required):
        raise TypeError("Batch must expose xc, yc, xt and yt.")

    batch_size = int(batch.yt.shape[0])
    fingerprints: List[str] = []

    for task_index in range(batch_size):
        digest = hashlib.sha256()
        digest.update(str(int(batch.xc.shape[1])).encode("utf-8"))
        digest.update(str(int(batch.xt.shape[1])).encode("utf-8"))
        for tensor_name in required:
            digest.update(tensor_name.encode("utf-8"))
            digest.update(_tensor_bytes(getattr(batch, tensor_name)[task_index]))
        fingerprints.append(digest.hexdigest())

    return fingerprints


def trivial_uniform_samples(
    *,
    target: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """I.i.d. U(0,1) samples with shape [M, B, Nt, Dy]."""
    num_samples = int(num_samples)
    if num_samples < 2:
        raise ValueError("Uniform baseline requires at least two samples.")

    return torch.rand(
        (num_samples, *target.shape),
        device=target.device,
        dtype=target.dtype,
    )


def theoretical_uniform_reference(num_samples: int) -> Dict[str, float]:
    """Population sanity values for Y,X_m iid U(0,1).

    CRPS is the population score of U(0,1). RMSE includes the finite-M
    variance of the ensemble mean, matching the sampled headline estimator.
    Coverage and width are population central-90% values, not finite-order-
    statistic expectations.
    """
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    return {
        "population_mean_rmse": math.sqrt(1.0 / 12.0),
        "finite_ensemble_mean_rmse": math.sqrt(
            (num_samples + 1.0) / (12.0 * num_samples)
        ),
        "fair_crps": 1.0 / 6.0,
        "coverage_90_population": 0.90,
        "width_90_population": 0.90,
    }


def _latent_metadata(batch: Any, local_index: int) -> Dict[str, float]:
    predictor = getattr(batch, "gt_pred", None)
    metadata = {
        "frequency": float("nan"),
        "direction": float("nan"),
        "offset": float("nan"),
    }

    if predictor is None:
        return metadata

    for key, attribute in (
        ("frequency", "freq"),
        ("direction", "direction"),
        ("offset", "offset"),
    ):
        value = getattr(predictor, attribute, None)
        if value is None or not torch.is_tensor(value) or value.numel() == 0:
            continue

        flattened = value.detach().cpu().reshape(value.shape[0], -1)
        source_index = min(int(local_index), int(flattened.shape[0]) - 1)
        metadata[key] = float(flattened[source_index, 0].item())

    return metadata


def per_task_marginal_rows(
    *,
    samples: torch.Tensor,
    target: torch.Tensor,
    batch_cpu: Any,
    model_name: str,
    source_kind: str,
    checkpoint_path: str,
    eval_set: str,
    task_index_start: int,
    generator_batch_index: int,
    fingerprints: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
    compute_energy: bool = False,
) -> List[Dict[str, Any]]:
    """One additive metric row per sawtooth task.

    Metrics use the same finite-ensemble estimators for every source.
    Additive components are retained so nonlinear aggregate metrics can be
    recomputed correctly inside a cluster bootstrap.
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError(
            "Expected samples [M,B,Nt,Dy] and target [B,Nt,Dy]. "
            f"Got {tuple(samples.shape)} and {tuple(target.shape)}."
        )
    if samples.shape[1:] != target.shape:
        raise ValueError(
            "Predictive samples do not match target shape: "
            f"samples={tuple(samples.shape)}, target={tuple(target.shape)}."
        )
    if len(fingerprints) != int(target.shape[0]):
        raise ValueError("Fingerprint count does not match batch size.")
    if not torch.isfinite(samples).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Samples or targets contain non-finite values.")

    samples = samples.detach()
    target = target.detach()

    num_samples = int(samples.shape[0])
    batch_size = int(target.shape[0])
    num_targets = int(target.shape[1])
    output_dim = int(target.shape[2])
    numel = num_targets * output_dim

    predictive_mean = samples.mean(dim=0)
    squared_error = (predictive_mean - target).square().reshape(batch_size, -1)
    sse = squared_error.sum(dim=1)

    crps = crps_per_element_sorted(
        samples=samples,
        target=target,
        alpha=1.0,
    ).reshape(batch_size, -1)
    crps_sum = crps.sum(dim=1)

    sample_variance = samples.var(
        dim=0,
        unbiased=True,
    ).reshape(batch_size, -1)
    var_sum = sample_variance.sum(dim=1)

    lower = torch.quantile(samples, 0.05, dim=0)
    upper = torch.quantile(samples, 0.95, dim=0)
    covered = ((target >= lower) & (target <= upper)).to(target.dtype)
    interval_width = upper - lower
    coverage_count = covered.reshape(batch_size, -1).sum(dim=1)
    width_sum = interval_width.reshape(batch_size, -1).sum(dim=1)

    if compute_energy:
        energy = energy_score_per_task(samples=samples, target=target)
    else:
        energy = torch.full(
            (batch_size,),
            float("nan"),
            device=target.device,
            dtype=target.dtype,
        )

    finite_m_correction = math.sqrt((num_samples + 1.0) / num_samples)
    metadata_dict = dict(metadata or {})

    rows: List[Dict[str, Any]] = []
    for local_index in range(batch_size):
        rmse_task = math.sqrt(float(sse[local_index].item()) / numel)
        spread_task = math.sqrt(float(var_sum[local_index].item()) / numel)
        row: Dict[str, Any] = {
            "model_name": str(model_name),
            "source_kind": str(source_kind),
            "checkpoint_path": str(checkpoint_path),
            "eval_set": str(eval_set),
            "region": "all",
            "task_index": int(task_index_start) + local_index,
            "generator_batch_index": int(generator_batch_index),
            "task_fingerprint": str(fingerprints[local_index]),
            "num_context": int(batch_cpu.xc.shape[1]),
            "context_bucket": f"nc_{int(batch_cpu.xc.shape[1]):03d}",
            "num_targets": num_targets,
            "output_dim": output_dim,
            "num_eval_samples": num_samples,
            "numel": numel,
            "sse": float(sse[local_index].item()),
            "crps_sum": float(crps_sum[local_index].item()),
            "var_sum": float(var_sum[local_index].item()),
            "coverage_count_90": float(coverage_count[local_index].item()),
            "width_sum_90": float(width_sum[local_index].item()),
            "rmse_task": rmse_task,
            "crps_task": float(crps_sum[local_index].item()) / numel,
            "ensemble_spread_task": spread_task,
            "spread_skill_ratio_task": (
                finite_m_correction * spread_task / (rmse_task + 1.0e-12)
            ),
            "coverage_90_task": float(coverage_count[local_index].item()) / numel,
            "width_90_task": float(width_sum[local_index].item()) / numel,
            "energy_score_task": float(energy[local_index].item()),
        }
        row.update(_latent_metadata(batch_cpu, local_index))
        row.update(metadata_dict)
        rows.append(row)

    return rows
