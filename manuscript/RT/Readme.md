# MSnet RT benchmarking and scaling-law analysis

This directory contains the reproducible analysis workflow for the MSnet RT application demonstration.

The workflow starts from the merged MSnet peptide-level parquet input, normalizes and deduplicates RT records, creates group-disjoint benchmark and scaling-law splits, runs AutoRT, DeepLC, and GPTime, aggregates metrics, and produces the manuscript figures.

## Files

| File | Purpose |
| --- | --- |
| `config.example.json` | Paths, input-column candidates, group definitions, random seeds, training sizes, and step schedules. |
| `prepare_rt_input.py` | Builds normalized peptide-level RT pools and writes total peptide and PXD-project counts. |
| `make_benchmark_splits.py` | Creates the three-round, leave-one-group-out splits used for the AutoRT, DeepLC, and GPTime benchmark. |
| `run_autort_benchmark.py` | Runs AutoRT training and prediction tasks. |
| `run_deeplc_benchmark.py` | Runs DeepLC training and prediction tasks. |
| `run_gptime_benchmark.py` | Runs GPTime training and prediction tasks. |
| `aggregate_benchmark_results.py` | Aggregates model metrics and calculates mean and standard deviation across rounds. |
| `plot_benchmark_r2.py` | Produces the six-panel R2 benchmark figure. |
| `make_scaling_splits.py` | Creates held-out-species scaling-law splits. |
| `analyze_scaling.py` | Calculates converged MAE and log-log power-law fits from step-level metric files. |
| `rt_utils.py` | Shared split, metric, and smoothing functions. |

## Input processing

The input directory contains the raw project-level parquet files. `prepare_rt_input.py` scans the configured parquet pattern, retains unmodified peptides only, normalizes each RT observation as `100 * retention / effective_rt`, and averages normalized RT values for each peptide sequence, species, and first-character combination. It then writes the `by_first_char` and `by_species` parquet pools and `input_pool_counts.csv` below `prepared_root`.

Update the path values and, where needed, input-column candidates in a copy of `config.example.json` before running the workflow.

```bash
python prepare_rt_input.py --config config.json
```

## Three-software benchmark

The benchmark evaluates the five largest first-character groups (`L`, `A`, `E`, `S`, and `V`) and the five largest species groups (`Homo sapiens`, `Mus musculus`, `Arabidopsis thaliana`, `Gallus gallus`, and `Danio rerio`). In each of three independent rounds, 10,000 non-overlapping peptides from the held-out group form the test set. All peptides from remaining eligible groups form the training pool. The nested training sizes are 1,000, 5,000, 10,000, 15,000, 20,000, 25,000, 50,000, 75,000, and 100,000 peptides. GPTime is evaluated through 25,000 peptides because of its memory requirements.

```bash
python make_benchmark_splits.py --config config.json
python run_autort_benchmark.py --config config.json --shard-id 0 --n-shards 1
python run_deeplc_benchmark.py --config config.json --shard-id 0 --n-shards 1
python run_gptime_benchmark.py --config config.json --shard-id 0 --n-shards 1
python aggregate_benchmark_results.py --config config.json
python plot_benchmark_r2.py --config config.json --output benchmark_r2.pdf
```

The AutoRT runner uses the strict from-scratch training mode and the default 10-model ensemble. The DeepLC runner uses its default three-model ensemble. The GPTime runner uses the repository workflow with `GPTIME_OPTIMIZE_RESTARTS=1`. A reproducible environment must provide AutoRT v2.0.0-beta, DeepLC v3.1.13, and the configured GPTime repository.

## Scaling-law analysis

Scaling-law experiments are run for the five held-out species with AutoRT and DeepLC. For each species, 1,000 randomly sampled held-out peptides form the test set. Nested training subsets of the same nine sizes are sampled from the non-held-out-species pool. During training, the AutoRT and DeepLC workflows must emit step-level metrics containing at least `software`, `split_name`, `train_size`, `step`, `mae`, and `status`.

```bash
python make_scaling_splits.py --config config.json --software AutoRT
python make_scaling_splits.py --config config.json --software DeepLC
python analyze_scaling.py --config config.json --input step_metric_directory --output scaling_power_law_fits.csv
```

`analyze_scaling.py` applies the step-dependent moving-average procedure from step 100, selects the earliest point within the configured MAE tolerance of each trajectory minimum as the converged MAE, and fits `MAE = aN^-b` in log-log space.

## Outputs

`benchmark_root` contains split CSV files, runner-specific model and prediction files, raw metrics, and `benchmark_r2_summary.csv`. `scaling_root` contains scaling splits and the step-level metrics used by `analyze_scaling.py`. Model weights and raw parquet input should remain outside the source repository.
