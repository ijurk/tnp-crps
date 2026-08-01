from __future__ import annotations

"""Diagnostic: do trained models condition on a post-fork context point?

This diagnostic tests whether a trained model uses a single post-fork context
observation to infer and propagate the global binary regime. It can be used to
compare models trained with pre-fork-only contexts against models trained with
mixed ambiguous/revealing contexts. AR coherence requires conditioning on the model's
own post-fork samples. This script tests that capability directly, without any
AR machinery:

    Variant A: original pre-fork context (baseline).
    Variant B: original context + ONE fabricated post-fork point lying on the
               realised latent branch (default x = 1.0).
    Variant C: original context + ONE in-range point set control_zscore
               original-oracle standard deviations above the oracle predictive
               mean at control_x (default x = -1.0, z = 2.0). Control showing
               that in-range conditioning works.

Verified oracle behaviour (tnp_crps/data/binary_latent_fork.py):
    _conditional_for_regime subtracts the regime offset AT the context
    locations (y_base = yc - o_c) and returns the context log-evidence;
    predictive_samples draws the regime from softmax(log_evidence - log 2),
    i.e. the regime POSTERIOR. The oracle therefore collapses correctly when a
    post-fork context point is present. Variant B's oracle is valid.

Interpretation (decision rule):
    - Decisive columns are x = 2.0 and 3.0 (and 0.5 for backward propagation).
      The injected location x = 1.0 is secondary: a model could copy the
      context value there without propagating the regime.
    - Variant B: oracle p_upper_side collapses to ~1.0 (upper regime) or ~0.0
      (lower). A model near ~0.5 at x = 2, 3 has failed to use the post-fork
      observation to resolve and propagate the global regime.
    - Variant C: judge via the printed control readout (mean shift in units of
      the original oracle sigma, and the predictive-std ratio), NOT via branch
      probabilities.

Usage (from repo root):
    python -u experiments/diagnose_post_fork_conditioning.py \
        --config experiments/configs/evaluation/binary_latent_fork_histograms_mixctx.yml \
        --device cuda
"""

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from evaluate_synthetic_1d import move_batch_to_device, sample_model
from plot_synthetic_1d_functions import get_plot_batch, load_models
from evaluation.plotting import plot_predictive_histogram_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="experiments/configs/evaluation/binary_latent_fork_histograms.yml",
        type=str,
        help="Histogram eval config providing models + base_generator_config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=(
            "Optional output-directory override. If omitted, uses "
            "conditioning_diagnostic.output_dir from the YAML."
        ),
    )
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--task_index", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument(
        "--max_nc",
        default=None,
        type=int,
        help=(
            "Cap the selected task's context size. Keeps Nc+1 within the "
            "training context-size range."
        ),
    )
    parser.add_argument(
        "--training_max_nc",
        default=None,
        type=int,
        help="Training-time maximum context size (guard: Nc+1 must not exceed).",
    )

    parser.add_argument("--inject_x", default=None, type=float)
    parser.add_argument("--control_x", default=None, type=float)
    parser.add_argument(
        "--control_zscore",
        default=None,
        type=float,
        help=(
            "Set the in-range control observation this many original-oracle "
            "standard deviations above its predictive mean. 2.0 is plausible "
            "under the DGP and visible in histograms; 1.0 is detectable only "
            "in the numeric summary."
        ),
    )
    parser.add_argument(
        "--x_locations",
        nargs="+",
        type=float,
        default=None,
        help="Post-fork query locations. Decisive columns: 2.0 and 3.0.",
    )

    parser.add_argument("--num_hist_samples", default=None, type=int)
    parser.add_argument("--num_oracle_samples", default=None, type=int)
    parser.add_argument("--bins", default=None, type=int)
    parser.add_argument(
        "--noiseless_injection",
        action="store_true",
        help="Inject f(x*) exactly instead of f(x*) + observation noise.",
    )

    return parser.parse_args()


def _gt_row_to_1d(t: torch.Tensor) -> torch.Tensor:
    """gt_pred.__call__ returns mean/std of shape [B, Nt]; be shape-agnostic."""
    return t[0].reshape(-1)


def append_context_point(batch, x_new: float, y_new: torch.Tensor):
    """Return a copy of batch with one extra (x, y) context point."""
    x_new_t = torch.full(
        (1, 1, batch.xc.shape[-1]),
        float(x_new),
        device=batch.xc.device,
        dtype=batch.xc.dtype,
    )
    y_new_t = y_new.reshape(1, 1, 1).to(
        device=batch.yc.device,
        dtype=batch.yc.dtype,
    )

    return dataclasses.replace(
        batch,
        xc=torch.cat([batch.xc, x_new_t], dim=1),
        yc=torch.cat([batch.yc, y_new_t], dim=1),
    )


def summary_rows(
    *,
    variant: str,
    source: str,
    samples: torch.Tensor,
    x_values: List[float],
    branch_centres: torch.Tensor,
) -> List[Dict[str, Any]]:
    """samples: [M, 1, K, 1]; branch_centres: [K] (original-oracle mixture mean)."""
    if samples.ndim != 4 or samples.shape[1] != 1 or samples.shape[-1] != 1:
        raise ValueError(f"Expected samples [M, 1, K, 1], got {tuple(samples.shape)}.")

    rows = []
    for k, x_val in enumerate(x_values):
        vals = samples[:, 0, k, 0].detach().float().cpu()
        centre = float(branch_centres[k])
        rows.append(
            {
                "variant": variant,
                "source": source,
                "x": float(x_val),
                "branch_centre": centre,
                "mean": float(vals.mean()),
                "std": float(vals.std(unbiased=True)),
                "p_upper_side": float((vals > centre).float().mean()),
                "n": int(vals.numel()),
            }
        )
    return rows


def sparse_path_side_consistency_row(
    *,
    variant: str,
    source: str,
    samples: torch.Tensor,
    x_values: List[float],
    branch_centres: torch.Tensor,
    postfork_min: float = 0.1,
    exclude_x: float = 1.0,
    exclude_tolerance: float = 1.0e-6,
) -> Dict[str, Any]:
    """Measure per-sample branch-side consistency at held-out post-fork points.

    The injected coordinate is excluded so that Variant B tests whether branch
    information propagates to other locations, rather than whether the model
    reproduces a context observation at the same x.

    This is a sparse-column diagnostic, not a complete full-path coherence
    measure.
    """
    if samples.ndim != 4:
        raise ValueError(
            "Expected samples with shape [M, B, K, Dy], "
            f"got {tuple(samples.shape)}."
        )

    if samples.shape[1] != 1 or samples.shape[-1] != 1:
        raise ValueError(
            "Expected diagnostic batch size 1 and output dimension 1, "
            f"got {tuple(samples.shape)}."
        )

    if len(x_values) != samples.shape[2]:
        raise ValueError(
            "x_values length must match the sample location dimension. "
            f"Got len(x_values)={len(x_values)} and "
            f"samples.shape[2]={samples.shape[2]}."
        )

    if branch_centres.numel() != samples.shape[2]:
        raise ValueError(
            "branch_centres length must match the sample location dimension. "
            f"Got {branch_centres.numel()} and {samples.shape[2]}."
        )

    # Use only held-out, fully post-transition locations.
    # In the current diagnostic this should select x = 0.5, 2.0, 3.0 and
    # deliberately exclude the injected coordinate x = 1.0.
    cols = [
        k
        for k, x in enumerate(x_values)
        if (
            float(x) > float(postfork_min)
            and abs(float(x) - float(exclude_x)) > float(exclude_tolerance)
        )
    ]

    if len(cols) < 2:
        raise ValueError(
            "Need at least two held-out post-fork columns for a switch "
            f"diagnostic. Selected x values: {[x_values[k] for k in cols]}."
        )

    vals = samples[:, 0, cols, 0].detach().float().cpu()  # [M, Kheld]
    centres = (
        branch_centres[cols]
        .detach()
        .float()
        .cpu()
        .reshape(1, -1)
    )

    # True means that this particular predictive sample lies above the
    # original oracle mixture centre at this location.
    above = vals > centres  # [M, Kheld]

    # Number of upper/lower side changes along the ordered held-out columns.
    switches = (
        above[:, 1:] != above[:, :-1]
    ).float().sum(dim=1)

    zero_switch = switches == 0
    all_upper = above.all(dim=1)
    all_lower = (~above).all(dim=1)

    frac_zero_switch = float(zero_switch.float().mean())
    frac_all_upper = float(all_upper.float().mean())
    frac_all_lower = float(all_lower.float().mean())

    # Null comparison: expected probability of no switches if sample identity
    # were independent across columns but each column retained its observed
    # marginal upper-side probability.
    p_upper_by_col = above.float().mean(dim=0)

    independent_zero_switch_baseline = float(
        torch.prod(p_upper_by_col)
        + torch.prod(1.0 - p_upper_by_col)
    )

    return {
        "variant": variant,
        "source": source,
        "num_heldout_postfork_cols": len(cols),
        "heldout_x": ",".join(
            f"{float(x_values[k]):.3f}"
            for k in cols
        ),
        "frac_zero_switch": frac_zero_switch,
        "mean_switches": float(switches.mean()),
        "frac_all_upper_side": frac_all_upper,
        "frac_all_lower_side": frac_all_lower,
        "frac_mixed_side": float((~zero_switch).float().mean()),
        "independent_zero_switch_baseline": (
            independent_zero_switch_baseline
        ),
        "excess_zero_switch": (
            frac_zero_switch - independent_zero_switch_baseline
        ),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()

    cfg = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Expected YAML config to resolve to a dictionary, got {type(cfg)}."
        )

    diagnostic_cfg = cfg.get("conditioning_diagnostic", {})

    if not isinstance(diagnostic_cfg, dict):
        raise TypeError(
            "conditioning_diagnostic must be a YAML mapping, "
            f"got {type(diagnostic_cfg)}."
        )

    def resolve_setting(
        *,
        cli_value,
        key: str,
        fallback,
    ):
        """Resolve setting with priority: CLI > diagnostic YAML > fallback."""
        if cli_value is not None:
            return cli_value

        return diagnostic_cfg.get(key, fallback)

    args.output_dir = resolve_setting(
        cli_value=args.output_dir,
        key="output_dir",
        fallback=(
            "results/synthetic_1d/"
            "binary_latent_fork_conditioning_diagnostic"
        ),
    )

    args.seed = int(
        resolve_setting(
            cli_value=args.seed,
            key="seed",
            fallback=20260713,
        )
    )

    args.max_nc = int(
        resolve_setting(
            cli_value=args.max_nc,
            key="max_nc",
            fallback=24,
        )
    )

    args.training_max_nc = int(
        resolve_setting(
            cli_value=args.training_max_nc,
            key="training_max_nc",
            fallback=32,
        )
    )

    args.inject_x = float(
        resolve_setting(
            cli_value=args.inject_x,
            key="inject_x",
            fallback=1.0,
        )
    )

    args.control_x = float(
        resolve_setting(
            cli_value=args.control_x,
            key="control_x",
            fallback=-1.0,
        )
    )

    args.control_zscore = float(
        resolve_setting(
            cli_value=args.control_zscore,
            key="control_zscore",
            fallback=2.0,
        )
    )

    args.x_locations = [
        float(value)
        for value in resolve_setting(
            cli_value=args.x_locations,
            key="x_locations",
            fallback=[0.5, 2.0, 3.0],
        )
    ]

    args.num_hist_samples = int(
        resolve_setting(
            cli_value=args.num_hist_samples,
            key="num_hist_samples",
            fallback=1024,
        )
    )

    args.num_oracle_samples = int(
        resolve_setting(
            cli_value=args.num_oracle_samples,
            key="num_oracle_samples",
            fallback=1024,
        )
    )

    args.bins = int(
        resolve_setting(
            cli_value=args.bins,
            key="bins",
            fallback=55,
        )
    )


    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but CUDA is not available.")
    device = torch.device(device_name)

    # ---------------------------- guards ----------------------------
    if float(args.inject_x) <= 0.1:
        raise ValueError(
            "inject_x must be beyond the completed fork transition; "
            f"got inject_x={args.inject_x}."
        )
    if float(args.control_x) > -0.25:
        raise ValueError(
            "control_x must remain inside the original pre-fork context support"
            f"[-4.0, -0.25]; got {args.control_x}."
        )

    if int(args.num_hist_samples) < 2:
        raise ValueError(
            "num_hist_samples must be at least 2 because the diagnostic "
            "computes an unbiased sample standard deviation."
        )

    if int(args.num_oracle_samples) < 2:
        raise ValueError(
            "num_oracle_samples must be at least 2."
        )

    models = load_models(
        model_entries=cfg["models"],
        base_generator_config=cfg["base_generator_config"],
        device=device,
    )

    # Reuse the first hist spec as the task template.
    spec = dict(cfg["hist_specs"][0])
    spec["task_index"] = int(args.task_index)
    spec["max_nc"] = int(args.max_nc)
    spec["name"] = f"conditioning_diag_task{args.task_index}"

    batch = get_plot_batch(
        base_generator_config=cfg["base_generator_config"],
        plot_spec=spec,
        search_batches=int(cfg.get("search_batches", 512)),
    )
    batch = move_batch_to_device(batch, device)

    if batch.xc.shape[0] != 1:
        raise ValueError(
            f"Expected a single plotting task, got batch size {batch.xc.shape[0]}."
        )
    if batch.yc.shape[-1] != 1:
        raise ValueError(
            f"Expected one-dimensional outputs, got Dy={batch.yc.shape[-1]}."
        )

    nc = batch.xc.shape[1]
    if nc + 1 > int(args.training_max_nc):
        raise RuntimeError(
            "Injected context would exceed the training context-size range: "
            f"Nc+1={nc + 1}, training_max_nc={args.training_max_nc}."
        )

    gt = getattr(batch, "gt_pred", None)
    if gt is None or not hasattr(gt, "sample_joint_observations_and_latent_function"):
        raise RuntimeError(
            "This diagnostic requires the binary latent fork gt_pred with "
            "sample_joint_observations_and_latent_function."
        )
    if not hasattr(batch, "x"):
        raise RuntimeError("Expected a SyntheticBatch with joint inputs batch.x.")

    # ------------------------------------------------------------------
    # Jointly resample task observations together with the exact realised
    # latent function at the probe locations, so the injected y-value lies
    # exactly on the realised branch of THIS task. Mirrors
    # maybe_resample_batch_for_exact_dense_truth.
    # ------------------------------------------------------------------
    probe_values = sorted({float(args.inject_x), float(args.control_x)})
    x_probe = torch.tensor(
        probe_values,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]

    y_obs, f_probe = gt.sample_joint_observations_and_latent_function(
        x_observed=batch.x,
        x_plot=x_probe,
        regimes=getattr(gt, "sampled_regimes", None),
        store=False,
    )

    batch = dataclasses.replace(
        batch,
        y=y_obs,
        yc=y_obs[:, :nc, :],
        yt=y_obs[:, nc:, :],
    )

    probe_index = {v: i for i, v in enumerate(probe_values)}
    f_inject = f_probe[0, probe_index[float(args.inject_x)], 0]

    regime_label = "unknown"
    sampled_regimes = getattr(gt, "sampled_regimes", None)
    if sampled_regimes is not None:
        regime_label = gt.regime_name(int(sampled_regimes[0]))

    noise_std = float(getattr(gt, "noise_std", 0.0))
    if args.noiseless_injection or noise_std <= 0.0:
        y_inject = f_inject.clone()
    else:
        y_inject = f_inject + noise_std * torch.randn(
            (), device=device, dtype=batch.yc.dtype
        )

    # ------------------------------------------------------------------
    # In-range control value: original-oracle mean + z * std at control_x.
    # A plausible counterfactual rather than an extreme outlier.
    # ------------------------------------------------------------------
    x_control_query = torch.tensor(
        [[[float(args.control_x)]]],
        device=device,
        dtype=batch.xc.dtype,
    )
    control_mean_t, control_std_t, _ = gt(
        xc=batch.xc,
        yc=batch.yc,
        xt=x_control_query,
    )
    control_mean = _gt_row_to_1d(control_mean_t)[0]
    control_std = _gt_row_to_1d(control_std_t)[0]
    y_control = control_mean + float(args.control_zscore) * control_std

    print(
        f"Task: Nc={nc} | realised regime={regime_label} | "
        f"inject ({float(args.inject_x):+.2f}, {float(y_inject):+.3f}) "
        f"[f={float(f_inject):+.3f}] | "
        f"control ({float(args.control_x):+.2f}, {float(y_control):+.3f}) "
        f"[oracle mean={float(control_mean):+.3f}, std={float(control_std):.3f}, "
        f"z={args.control_zscore:+.1f}]"
    )

    variants = [
        ("A_original", batch),
        ("B_postfork_injected", append_context_point(batch, args.inject_x, y_inject)),
        ("C_inrange_zscore_control", append_context_point(batch, args.control_x, y_control)),
    ]

    # Shared histogram locations: control, injection point, post-fork queries.
    x_values = sorted(
        {float(args.control_x), float(args.inject_x)}
        | {float(v) for v in args.x_locations}
    )
    x_hist = torch.tensor(
        x_values,
        device=device,
        dtype=batch.xc.dtype,
    )[None, :, None]
    y_placeholder = torch.zeros(
        1,
        x_hist.shape[1],
        batch.yc.shape[-1],
        device=device,
        dtype=batch.yc.dtype,
    )

    # Branch centres: ORIGINAL-context oracle mixture mean at each location.
    # Under the 50/50 mixture the offsets cancel, so this is the posterior
    # mean of the shared base GP, i.e. the midpoint between the branches.
    branch_mean_t, branch_std_t, _ = gt(
        xc=batch.xc,
        yc=batch.yc,
        xt=x_hist,
    )
    branch_centres = _gt_row_to_1d(branch_mean_t).detach().float().cpu()
    original_oracle_std = _gt_row_to_1d(branch_std_t).detach().float().cpu()
    control_col = x_values.index(float(args.control_x))

    all_rows: List[Dict[str, Any]] = []
    coherence_rows: List[Dict[str, Any]] = []

    for variant_name, variant_batch in variants:
        hist_batch = dataclasses.replace(
            variant_batch,
            xt=x_hist,
            yt=y_placeholder,
        )

        prediction_rows = []
        for item in models:
            samples = sample_model(
                model=item["model"],
                batch=hist_batch,
                num_eval_samples=int(args.num_hist_samples),
            )
            prediction_rows.append({"name": item["name"], "samples": samples})
            all_rows.extend(
                summary_rows(
                    variant=variant_name,
                    source=item["name"],
                    samples=samples,
                    x_values=x_values,
                    branch_centres=branch_centres,
                )
            )
            coherence_rows.append(
                sparse_path_side_consistency_row(
                    variant=variant_name,
                    source=item["name"],
                    samples=samples,
                    x_values=x_values,
                    branch_centres=branch_centres,
                    postfork_min=0.1,
                    exclude_x=float(args.inject_x),
                )
            )
            print(f"[{variant_name}] sampled {item['name']}")

        # Oracle reference (conditions on the SAME variant context).
        oracle_samples = gt.predictive_samples(
            variant_batch.xc,
            variant_batch.yc,
            x_hist,
            num_samples=int(args.num_oracle_samples),
        )
        all_rows.extend(
            summary_rows(
                variant=variant_name,
                source="Oracle",
                samples=oracle_samples,
                x_values=x_values,
                branch_centres=branch_centres,
            )
        )
        coherence_rows.append(
            sparse_path_side_consistency_row(
                variant=variant_name,
                source="Oracle",
                samples=oracle_samples,
                x_values=x_values,
                branch_centres=branch_centres,
                postfork_min=0.1,
                exclude_x=float(args.inject_x),
            )
        )

        title = (
            f"{spec['name']} | {variant_name} | regime={regime_label} | "
            f"Nc={variant_batch.xc.shape[1]}"
        )
        plot_predictive_histogram_comparison(
            batch=hist_batch,
            x_hist=x_hist,
            prediction_rows=prediction_rows,
            output_path_base=output_dir / f"{spec['name']}_{variant_name}",
            title=title,
            num_oracle_samples=int(args.num_oracle_samples),
            bins=int(args.bins),
        )

    # ------------------------------------------------------------------
    # Numeric summary: CSV + console tables + decision rule.
    # ------------------------------------------------------------------
    csv_path = (output_dir / f"{spec['name']}_conditioning_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "source",
                "x",
                "branch_centre",
                "mean",
                "std",
                "p_upper_side",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    coherence_csv_path = (
        output_dir / f"{spec['name']}_sparse_path_side_consistency.csv"
    )
    
    coherence_fieldnames = [
        "variant",
        "source",
        "num_heldout_postfork_cols",
        "heldout_x",
        "frac_zero_switch",
        "mean_switches",
        "frac_all_upper_side",
        "frac_all_lower_side",
        "frac_mixed_side",
        "independent_zero_switch_baseline",
        "excess_zero_switch",
    ]
    
    with open(coherence_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=coherence_fieldnames,
        )
        writer.writeheader()
        writer.writerows(coherence_rows)
    
    print(
        "\n=== Sparse held-out post-fork side consistency ==="
    )
    
    print(
        "variant".ljust(28)
        + "source".ljust(28)
        + "zero-sw".rjust(10)
        + "indep-null".rjust(12)
        + "excess".rjust(10)
        + "mean-sw".rjust(10)
        + "all-up".rjust(10)
        + "all-low".rjust(10)
    )
    
    for row in coherence_rows:
        print(
            str(row["variant"]).ljust(28)
            + str(row["source"]).ljust(28)
            + f"{row['frac_zero_switch']:10.3f}"
            + f"{row['independent_zero_switch_baseline']:12.3f}"
            + f"{row['excess_zero_switch']:10.3f}"
            + f"{row['mean_switches']:10.3f}"
            + f"{row['frac_all_upper_side']:10.3f}"
            + f"{row['frac_all_lower_side']:10.3f}"
        )
    
    print(f"\nWrote {coherence_csv_path}")

    diagnostic_config_path = (
        output_dir
        / f"{spec['name']}_diagnostic_config_resolved.json"
    )
    
    with open(diagnostic_config_path, "w") as f:
        json.dump(
            {
                "config": cfg,
                "cli": vars(args),
                "regime": regime_label,
                "nc": nc,
                "f_inject": float(f_inject),
                "y_inject": float(y_inject),
                "control_oracle_mean": float(control_mean),
                "control_oracle_std": float(control_std),
                "y_control": float(y_control),
            },
            f,
            indent=2,
        )

    def _lookup(variant: str, source: str, key: str) -> List[float]:
        return [
            r[key]
            for r in all_rows
            if r["variant"] == variant and r["source"] == source
        ]

    sources = ["Oracle"] + [m["name"] for m in models]

    print("\n=== p_upper_side (mass above original-oracle branch centre) ===")
    header = "source".ljust(28) + "".join(f"x={v:+.2f}".rjust(10) for v in x_values)
    for variant_name, _ in variants:
        print(f"\n[{variant_name}]")
        print(header)
        for source in sources:
            vals = _lookup(variant_name, source, "p_upper_side")
            print(source.ljust(28) + "".join(f"{v:10.2f}" for v in vals))

    print(
        "\n=== In-range control readout at "
        f"x={float(args.control_x):+.2f} (A vs C) ==="
    )
    sigma_ref = float(original_oracle_std[control_col])
    print(
        "source".ljust(28)
        + "mean_A".rjust(10)
        + "mean_C".rjust(10)
        + "shift/sig".rjust(11)
        + "std_C/std_A".rjust(13)
    )
    for source in sources:
        mean_a = _lookup("A_original", source, "mean")[control_col]
        mean_c = _lookup("C_inrange_zscore_control", source, "mean")[control_col]
        std_a = _lookup("A_original", source, "std")[control_col]
        std_c = _lookup("C_inrange_zscore_control", source, "std")[control_col]
        shift_sig = (mean_c - mean_a) / sigma_ref if sigma_ref > 0 else float("nan")
        std_ratio = std_c / std_a if std_a > 0 else float("nan")
        print(
            source.ljust(28)
            + f"{mean_a:10.3f}"
            + f"{mean_c:10.3f}"
            + f"{shift_sig:11.2f}"
            + f"{std_ratio:13.2f}"
        )

    print(
        "\nDecision rule:\n"
        "  B (decisive columns x=2.0, 3.0; x=0.5 for backward propagation):\n"
        "     Oracle p_upper_side collapses to ~1.0 (upper regime) or ~0.0\n"
        "     (lower). A model remaining near ~0.5 there has not used the\n"
        "     injected point to resolve and propagate the global regime. The\n"
        f"     injected column x={float(args.inject_x):+.2f} is secondary:\n"
        "     copying the context value locally is not regime propagation.\n"
        "  C: judge via mean shift (in original-oracle sigmas) and std ratio\n"
        "     at the control column, not via branch probabilities.\n"
        f"\nWrote figures + {csv_path}"
    )


if __name__ == "__main__":
    main()