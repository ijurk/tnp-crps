from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tnp.data.gp import GPRegressionModel
from tnp.data.synthetic import SyntheticBatch
from tnp_crps.models.tnp_crps import DirectTNP
from tnp_crps.utils.np_functions import np_pred_fn

from evaluate_gaussian_controls import _load_learned_sources
from evaluate_synthetic_1d import (
    apply_eval_dataset_overrides,
    apply_eval_kernel,
    load_merged_config,
    move_batch_to_device,
)
from evaluation.autoregressive import (
    autoregressive_sample_model,
    denoise_autoregressive_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def _find_eval_set(cfg: Mapping[str, Any], name: str) -> Dict[str, Any]:
    matches = [dict(item) for item in cfg["eval_sets"] if str(item["name"]) == name]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one eval_set named {name!r}, got {len(matches)}.")
    return matches[0]


def _select_task(
    *,
    generator,
    min_nc: int,
    max_nc: int,
    qualifying_task_index: int,
) -> tuple[SyntheticBatch, int]:
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    qualifying_seen = 0
    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, SyntheticBatch):
            raise TypeError(f"Expected SyntheticBatch, got {type(batch)}.")
        nc = int(batch.xc.shape[1])
        if nc < min_nc or nc > max_nc:
            continue
        if qualifying_seen == qualifying_task_index:
            return batch, batch_index
        qualifying_seen += 1

    raise RuntimeError(
        "Could not locate a plotting task satisfying the context-size constraints."
    )


@torch.no_grad()
def _resample_joint_plot_task(
    *,
    batch: SyntheticBatch,
    x_plot: torch.Tensor,
    seed: int,
) -> tuple[SyntheticBatch, torch.Tensor]:
    """Jointly resample the displayed finite task and dense latent truth."""
    if batch.x.shape[0] != 1:
        raise ValueError("Plot-cache generation expects batch size 1.")
    if batch.gt_pred is None:
        raise RuntimeError("Selected GP task has no gt_pred.")

    pl.seed_everything(seed, workers=False)

    predictor = batch.gt_pred
    predictor._result_cache = None

    x_all = torch.cat([batch.x, x_plot], dim=1)
    gp_model = GPRegressionModel(
        likelihood=predictor.likelihood,
        kernel=predictor.kernel,
    )
    gp_model.eval()
    gp_model.likelihood.eval()

    latent_dist = gp_model.forward(x_all[0])
    latent_values = latent_dist.sample()
    observation_dist = gp_model.likelihood(latent_values)
    observed_values = observation_dist.sample()

    num_task_points = int(batch.x.shape[1])
    y_task = observed_values[:num_task_points].view(1, num_task_points, 1)
    latent_dense = latent_values[num_task_points:].view(1, x_plot.shape[1], 1)
    nc = int(batch.xc.shape[1])

    resampled = dataclasses.replace(
        batch,
        y=y_task,
        yc=y_task[:, :nc, :],
        yt=y_task[:, nc:, :],
    )
    return resampled, latent_dense


@torch.no_grad()
def _exact_gp_posterior_cache(
    *,
    batch: SyntheticBatch,
    x_plot: torch.Tensor,
    num_samples: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if batch.gt_pred is None:
        raise RuntimeError("GP task has no ground-truth predictor.")

    pl.seed_everything(seed, workers=False)
    predictor = batch.gt_pred
    predictor._result_cache = None

    gp_model = GPRegressionModel(
        likelihood=predictor.likelihood,
        kernel=predictor.kernel,
        train_inputs=batch.xc[0],
        train_targets=batch.yc[0, :, 0],
    )
    gp_model.eval()
    gp_model.likelihood.eval()

    latent_posterior = gp_model(x_plot[0])
    observed_posterior = gp_model.likelihood.marginal(latent_posterior)

    samples = observed_posterior.sample(torch.Size([num_samples]))
    return {
        "mean": observed_posterior.mean.view(1, x_plot.shape[1], 1).cpu(),
        "scale": observed_posterior.stddev.view(1, x_plot.shape[1], 1).cpu(),
        "direct_samples": samples.view(num_samples, 1, x_plot.shape[1], 1).cpu(),
    }


@torch.no_grad()
def _direct_samples(
    *,
    model: torch.nn.Module,
    batch: SyntheticBatch,
    num_samples: int,
) -> torch.Tensor:
    if isinstance(model, DirectTNP):
        out = model.sample(
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            num_samples=num_samples,
        )
    else:
        pred_dist = np_pred_fn(model=model, batch=batch, num_samples=num_samples)
        out = pred_dist.sample((num_samples,))

    if out.ndim != 4:
        raise ValueError(f"Expected direct samples [M,B,N,D], got {tuple(out.shape)}.")
    return out


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    figure_cfg = dict(cfg["figure"])

    device_name = args.device or str(cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)

    eval_set = _find_eval_set(cfg, str(figure_cfg["eval_set"]))
    generator_config = load_merged_config(
        config_paths=[str(cfg["base_generator_config"])],
        overrides=list(eval_set.get("overrides", []) or []),
    )
    apply_eval_kernel(generator_config, str(eval_set["kernel"]))
    apply_eval_dataset_overrides(
        generator_config,
        samples_per_eval_set=int(figure_cfg["search_batches"]),
        eval_batch_size=1,
    )
    generator_config.generators.test.deterministic = True
    generator_config.generators.test.deterministic_seed = int(
        figure_cfg["deterministic_seed"]
    )

    generator = instantiate(generator_config.generators.test)
    batch, selected_batch_index = _select_task(
        generator=generator,
        min_nc=int(figure_cfg["min_nc"]),
        max_nc=int(figure_cfg["max_nc"]),
        qualifying_task_index=int(figure_cfg["qualifying_task_index"]),
    )

    x_min, x_max = (float(value) for value in figure_cfg["x_range"])
    x_plot = torch.linspace(
        x_min,
        x_max,
        int(figure_cfg["num_dense_points"]),
        dtype=batch.x.dtype,
    ).view(1, -1, 1)

    batch, latent_truth = _resample_joint_plot_task(
        batch=batch,
        x_plot=x_plot,
        seed=int(figure_cfg["resample_seed"]),
    )

    exact_cache = _exact_gp_posterior_cache(
        batch=batch,
        x_plot=x_plot,
        num_samples=int(figure_cfg["num_direct_samples"]),
        seed=int(figure_cfg["sampling_seed"]),
    )

    learned = _load_learned_sources(
        source_entries=list(cfg["sources"]),
        base_generator_config=str(cfg["base_generator_config"]),
        device=device,
    )

    dense_placeholder = torch.zeros(
        1,
        x_plot.shape[1],
        batch.yc.shape[-1],
        dtype=batch.yc.dtype,
    )
    dense_batch_cpu = dataclasses.replace(batch, xt=x_plot, yt=dense_placeholder)
    dense_batch = move_batch_to_device(dense_batch_cpu, device)

    nc = int(batch.xc.shape[1])
    requested_anchors = int(figure_cfg["num_ar_anchors"])
    max_anchors = int(figure_cfg["training_max_nc"]) - nc
    actual_anchors = min(requested_anchors, max_anchors)
    if actual_anchors < 1:
        raise RuntimeError(
            f"No valid AR anchors: Nc={nc}, training_max_nc={figure_cfg['training_max_nc']}."
        )

    anchor_min, anchor_max = (float(value) for value in figure_cfg["ar_anchor_range"])
    x_anchor = torch.linspace(
        anchor_min,
        anchor_max,
        actual_anchors,
        device=device,
        dtype=dense_batch.xc.dtype,
    ).view(1, actual_anchors, 1)
    anchor_placeholder = torch.zeros(
        1,
        actual_anchors,
        dense_batch.yc.shape[-1],
        device=device,
        dtype=dense_batch.yc.dtype,
    )
    ar_batch = dataclasses.replace(
        dense_batch,
        xt=x_anchor,
        yt=anchor_placeholder,
    )

    model_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    model_number = 0
    for item in learned:
        entry = item["entry"]
        if str(entry.get("kind", "model")) == "exact_gp":
            continue

        model = item["model"]
        assert model is not None
        model_number += 1

        direct_seed = int(figure_cfg["sampling_seed"]) + 10_000 * model_number
        pl.seed_everything(direct_seed, workers=False)
        direct = _direct_samples(
            model=model,
            batch=dense_batch,
            num_samples=int(figure_cfg["num_direct_samples"]),
        )

        ar_seed = direct_seed + 1_000
        pl.seed_everything(ar_seed, workers=False)
        raw_ar = autoregressive_sample_model(
            model=model,
            batch=ar_batch,
            num_samples=int(figure_cfg["num_ar_samples"]),
            target_order=str(figure_cfg["target_order"]),
            stochln_noise_mode=str(figure_cfg["stochln_noise_mode"]),
        )

        denoised = denoise_autoregressive_samples(
            model=model,
            batch=ar_batch,
            ar_samples=raw_ar,
            query_xt=dense_batch.xt,
            num_denoise_samples=int(figure_cfg["num_denoise_samples"]),
        )

        model_cache[str(entry["name"])] = {
            "direct_samples": direct.detach().cpu(),
            "ar_support_x": x_anchor.detach().cpu(),
            "raw_ar_samples": raw_ar.detach().cpu(),
            "ar_denoised_paths": denoised.detach().cpu(),
        }

    cache = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "eval_set": str(eval_set["name"]),
            "kernel": str(eval_set["kernel"]),
            "selected_generator_batch_index": selected_batch_index,
            "num_context": nc,
            "num_ar_anchors": actual_anchors,
            "target_order": str(figure_cfg["target_order"]),
            "stochln_noise_mode": str(figure_cfg["stochln_noise_mode"]),
            "lower_row_semantics": (
                "AR support samples followed by dense conditional predictive means"
            ),
        },
        "task": {
            "xc": batch.xc.cpu(),
            "yc": batch.yc.cpu(),
            "xt": batch.xt.cpu(),
            "yt": batch.yt.cpu(),
            "x_plot": x_plot.cpu(),
            "latent_truth": latent_truth.cpu(),
        },
        "exact_gp": exact_cache,
        "models": model_cache,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    (output.with_suffix(".json")).write_text(json.dumps(cache["metadata"], indent=2))
    print(f"Wrote figure cache: {output}")


if __name__ == "__main__":
    main()
