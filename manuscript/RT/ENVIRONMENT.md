# Runtime requirements

The benchmark was run with separate software environments.

| Software | Runtime |
| --- | --- |
| AutoRT | AutoRT v2.0.0-beta and its TensorFlow environment. |
| DeepLC | DeepLC v3.1.13 and its TensorFlow environment. |
| GPTime | The GPTime repository workflow with `GPTIME_OPTIMIZE_RESTARTS=1`. |

The benchmark runners require the repository locations and Python executables specified in `config.json`. AutoRT must expose the strict from-scratch mode used by `run_autort_benchmark.py`. DeepLC must support terminating retraining at `DEEPLC_TOTAL_STEPS` while retaining its three default models.

The scaling-law analysis requires step-level evaluation output. Each requested evaluation step must write one valid metric row containing `software`, `split_name`, `train_size`, `step`, `mae`, and `status`. AutoRT uses `AUTORT_STRICT_STEP_EVAL_CONFIG` to pass the step schedule. The `analyze_scaling.py` stage consumes these saved metric rows.
