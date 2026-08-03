# Section 4.2 Binary Latent Fork — final evaluation bundle

This bundle adds dedicated paired evaluators for the three claims in Section 4.2.
It does not alter the generic tabular evaluator.

## Files

- `tnp_crps/data/binary_latent_fork.py`
  - exact marginal component API;
  - efficient exact marginal samples;
  - paired lower/upper counterfactual task generation;
  - vectorised coherent oracle path sampling.
- `experiments/evaluation/predictive_sampling.py`
- `experiments/audit_binary_fork_training.py`
- `experiments/evaluation/binary_fork_utils.py`
- `experiments/evaluation/binary_fork_metrics.py`
- `experiments/evaluate_binary_fork_marginals.py`
- `experiments/analyse_binary_fork_marginals.py`
- `experiments/evaluate_binary_fork_conditioning.py`
- `experiments/analyse_binary_fork_conditioning.py`
- `experiments/evaluate_binary_fork_coherence.py`
- `experiments/analyse_binary_fork_coherence.py`
- three final YAML configs under `experiments/configs/evaluation/latent_fork/`.

## Locked definitions

- Fully post-fork targets: `x >= 0.5`.
- Branch boundary: midpoint of the exact lower/upper component means.
- Central gap: midpoint plus/minus `0.25` times the exact half-separation.
- Branch-mass error: absolute difference between model and exact posterior mass above the branch boundary.
- Gap-mass error: absolute difference between model and exact posterior mass inside the central gap.
- Coherence baseline: exact independence expectation computed from each source's empirical lower/gap/upper marginal probabilities at the 16 path anchors.
- Quantitative marginal evaluation: `M=256`, chunk size `64`.
- Coherence evaluation: 50 paths per task, 16 sparse anchors, random traversal, refreshed stochasticity.

## Evaluation budgets

- T1 ambiguous marginals: 80,000 paired tasks.
- T2 revealing-context intervention: 4,096 paired base tasks under ambiguous, upper-reveal and lower-reveal conditions.
- T3 direct versus AR coherence: 4,096 paired tasks.

## Important interpretation

The earlier approximately 0.25 no-switch independence value was computed over
three held-out post-fork locations. The final coherence evaluation uses 16
ordered anchors and computes the independence expectation from each source's
empirical lower/gap/upper probabilities, so its numerical baseline will be much
smaller when the marginals are close to balanced and gap-free.

T3 scores raw sparse-anchor paths. Dense denoised paths are reserved for the
qualitative dissertation figure and are not substituted for the quantitative
switch statistics.

## Main outputs

- T1 analysis: `summary_headline.csv`, `paired_cluster_bootstrap_deltas.csv`,
  `binary_fork_ambiguous_table.tex`.
- T2 analysis: `summary_by_source_condition.csv`,
  `summary_conditioning_table.csv`, `mixed_minus_old_deltas.csv`,
  `binary_fork_conditioning_table.tex`.
- T3 analysis: `summary_coherence.csv`, `ar_minus_direct_deltas.csv`,
  `binary_fork_coherence_table.tex`.
