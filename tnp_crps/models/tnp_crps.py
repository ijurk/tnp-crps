import torch


def crps_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Marginal almost-fair CRPS.

    samples: [M, B, Nt, Dy]
    target:  [B, Nt, Dy]

    alpha = 1.0 -> fair CRPS
    alpha = 0.0 -> ordinary empirical CRPS
    """

    if samples.ndim != target.ndim + 1:
        raise ValueError(
            f"Expected samples to have one extra sample dimension. "
            f"Got samples.shape={samples.shape}, target.shape={target.shape}."
        )

    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"Expected samples.shape[1:] == target.shape. "
            f"Got samples.shape={samples.shape}, target.shape={target.shape}."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]. Got {alpha}.")

    num_samples = samples.shape[0]

    if num_samples < 2:
        raise ValueError("Fair CRPS requires at least 2 samples.")

    target_term = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)

    pairwise_dist = torch.abs(samples[:, None, ...] - samples[None, :, ...])

    ordinary_pairwise = pairwise_dist.mean(dim=(0, 1))
    ordinary_crps = target_term - 0.5 * ordinary_pairwise

    fair_pairwise = pairwise_dist.sum(dim=(0, 1)) / (
        num_samples * (num_samples - 1)
    )
    fair_crps = target_term - 0.5 * fair_pairwise

    almost_fair_crps = alpha * fair_crps + (1.0 - alpha) * ordinary_crps

    return almost_fair_crps.mean()