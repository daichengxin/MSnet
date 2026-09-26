# Aggregate data included with this release

Only small, three-seed aggregate tables needed to reproduce the figures are
included here. Raw peptide records, model checkpoints, per-run predictions,
and proprietary/source reports are intentionally excluded.

- `dataset_scaling_aggregate.csv`: Phase 1, 12 training-set sizes.
- `compute_scaling_aggregate.csv`: Phase 2, 7 dataset sizes × 10 exact
  optimizer-step checkpoints.
- `parameter_scaling_aggregate.csv`: Phase 3, 6 hidden dimensions and their
  measured trainable parameter counts.

All reported test values are mean absolute errors on the fixed test set in the
original CCS scale. The three-seed mean and sample standard deviation are
stored separately; the released final figures show the means without error
bars.
