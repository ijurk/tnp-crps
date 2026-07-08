from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from tnp.data.base import Batch

from evaluate_synthetic_1d import (
    move_batch_to_device,
    sample_model,
)
from plot_synthetic_1d_functions import (
    get_plot_batch,
    load_models,
)
from evaluation.plotting import plot_predictive_histogram_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def dataclass_replace_batch(batch: Batch, **kwargs) -> Batch:
    return dataclasses.replace(batch, **kwargs)


@torch.no_grad()
def make_hist_prediction_rows(
    *,
    models: List[Dict[str, Any]],
    hist_batch: Batch,
    num_hist_samples: int,
) -> List[Dict[str, Any]]:
    rows = []

    for item in models:
        samples = sample_model(
            model=item["model"],
            batch=hist_batch,
            num_eval_samples=num_hist_samples,
        )

        rows.append(
            {
                "name": item["name"],
                "samples": samples,
            }
        )

        print(f"Sampled histogram predictions for {item['name']}")

    return rows


@torch.no_grad()
def make_one_histogram_plot(
    *,
    batch: Batch,
    models: List[Dict[str, Any]],
    hist_spec: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_hist_samples: int,
    num_oracle_samples: int,
    bins: int,
) -> None:
    batch = move_batch_to_device(batch, device)

    x_locations = hist_spec.get("x_locations", [1.0, 2.5, 3.5])
    x_hist = torch.tensor(
        x_locations,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]

    y_placeholder = torch.zeros_like(x_hist)

    hist_batch = dataclass_replace_batch(
        batch,
        xt=x_hist,
        yt=y_placeholder,
    )

    prediction_rows = make_hist_prediction_rows(
        models=models,
        hist_batch=hist_batch,
        num_hist_samples=num_hist_samples,
    )

    dataset_label = hist_spec.get(
        "kernel",
        hist_spec.get("dataset", "native_generator"),
    )

    title = (
        f"{hist_spec['name']} | dataset={dataset_label} | "
        f"Nc={batch.xc.shape[1]}"
    )

    output_path_base = output_dir / hist_spec["name"]

    plot_predictive_histogram_comparison(
        batch=batch,
        x_hist=x_hist,
        prediction_rows=prediction_rows,
        output_path_base=output_path_base,
        title=title,
        num_oracle_samples=num_oracle_samples,
        bins=bins,
    )


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "histogram_config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device_name = args.device or cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but CUDA is not available.")

    device = torch.device(device_name)

    base_generator_config = cfg["base_generator_config"]

    models = load_models(
        model_entries=cfg["models"],
        base_generator_config=base_generator_config,
        device=device,
    )

    for hist_spec in cfg["hist_specs"]:
        batch = get_plot_batch(
            base_generator_config=base_generator_config,
            plot_spec=hist_spec,
            search_batches=int(cfg.get("search_batches", 512)),
        )

        make_one_histogram_plot(
            batch=batch,
            models=models,
            hist_spec=hist_spec,
            output_dir=output_dir,
            device=device,
            num_hist_samples=int(cfg.get("num_hist_samples", 1024)),
            num_oracle_samples=int(cfg.get("num_oracle_samples", 2048)),
            bins=int(cfg.get("bins", 50)),
        )


if __name__ == "__main__":
    main()