# Section 4.1 Gaussian Controls — reproducible evaluation bundle

This bundle adds **new files only**. It does not replace the existing generic evaluator or tabular metrics.

## Added files

- `experiments/evaluation/gaussian_controls_metrics.py`
- `experiments/evaluate_gaussian_controls.py`
- `experiments/analyse_gaussian_controls.py`
- `experiments/cache_gaussian_controls_figure.py`
- `experiments/render_gaussian_controls_figure.py`
- `experiments/configs/evaluation/gaussian_controls_final.yml`

## Evaluation design

- Five fixed-hyperparameter GP test sets: RBF, Matérn-1/2, Matérn-3/2, Matérn-5/2, periodic.
- Lengthscale is fixed at `0.5`; periodic period is fixed at `2.0`.
- Observation noise remains `0.1` through the existing generator.
- 8,000 tasks per kernel, batch size 16, 40,000 tasks total.
- Exact GP oracle and Gaussian TNP are each evaluated twice: analytically
  (closed-form Gaussian marginal metrics) and as an M=64 sampled twin scored
  with exactly the same finite-ensemble estimators as the CRPS models.
- Headline comparisons use the sampled rows for all four comparators; the
  analytic-minus-sampled twin deltas quantify the finite-M estimator bias
  in RMSE, coverage and width, and are reported as a diagnostic.
- Dropout and StochLN use 64 predictive samples and fair sample CRPS.
- Region masks are computed from each task's realised context support
  (min/max of that task's xc), not the generator's global sampling window.
- Sampled sources use explicit, unique seed offsets, so predictive Monte Carlo
  draws do not change when diagnostic sources are reordered or added.
- Figure-cache generation uses an explicit list of the three learned models and
  ignores analytic/sampled diagnostic twins.
- Main metrics: RMSE, CRPS, spread-skill ratio, 90% coverage, 90% width.
- Per-task rows retain additive components so pooled metrics and paired bootstrap differences are computed correctly.
- Bootstrap unit is the generator batch, because each batch shares context size and GP hyperparameters.
- The figure cache separates expensive inference from rendering.

## Install

From the repository root:

```bash
unzip -o gaussian_controls_bundle.zip -d .
python -m py_compile \
  experiments/evaluation/gaussian_controls_metrics.py \
  experiments/evaluate_gaussian_controls.py \
  experiments/analyse_gaussian_controls.py \
  experiments/cache_gaussian_controls_figure.py \
  experiments/render_gaussian_controls_figure.py
```

## Smoke test

```bash
export PYTHONPATH="$PWD/external/tnp:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=<GPU_ID>

python -u experiments/evaluate_gaussian_controls.py \
  --config experiments/configs/evaluation/gaussian_controls_final.yml \
  --output_dir results/synthetic_1d/gaussian_controls_smoke_20260802_v4 \
  --samples_per_eval_set 32 \
  --eval_batch_size 16 \
  --num_eval_samples 64

python -u experiments/analyse_gaussian_controls.py \
  --input results/synthetic_1d/gaussian_controls_smoke_20260802_v4/per_task_metrics.csv \
  --output_dir results/synthetic_1d/gaussian_controls_smoke_20260802_v4/analysis \
  --bootstrap_replicates 500 \
  --bootstrap_seed 20260831
```

Do not run the final evaluation unless the evaluator prints `Pairing PASS` for all five kernels and the analysis completes without an assertion.

## Final quantitative run

```bash
python -u experiments/evaluate_gaussian_controls.py \
  --config experiments/configs/evaluation/gaussian_controls_final.yml \
  2>&1 | tee logs/gaussian-controls-final-20260802-v1.log

python -u experiments/analyse_gaussian_controls.py \
  --input results/synthetic_1d/gaussian_controls_final_20260802_v2/per_task_metrics.csv \
  --output_dir results/synthetic_1d/gaussian_controls_final_20260802_v2/analysis \
  --bootstrap_replicates 10000 \
  --bootstrap_seed 20260831
```

## Figure cache and render

```bash
python -u experiments/cache_gaussian_controls_figure.py \
  --config experiments/configs/evaluation/gaussian_controls_final.yml \
  --output results/synthetic_1d/gaussian_controls_final_20260802_v2/figure/matern32_cache.pt

python -u experiments/render_gaussian_controls_figure.py \
  --cache results/synthetic_1d/gaussian_controls_final_20260802_v2/figure/matern32_cache.pt \
  --output results/synthetic_1d/gaussian_controls_final_20260802_v2/figure/gaussian_controls_direct_vs_ar \
  --visible_paths 8
```

The lower row is deliberately labelled **AR-conditioned mean paths**: each path is produced by sampling sparse AR supports, appending them to the context, and evaluating the resulting dense conditional predictive mean.
