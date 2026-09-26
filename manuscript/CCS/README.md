# CCS scaling laws for AlphaPeptDeep

This repository contains the cleaned code and compact aggregate results for
the ion-mobility / collisional-cross-section (CCS) scaling-law experiments.
It covers dataset scaling, compute scaling, parameter scaling, and the final
three-panel figure.

The released aggregate CSV files and figures can be used without retraining.
Raw source tables, peptide-level splits, model checkpoints, and per-run
predictions are not included because of their size and data-distribution
constraints.

## What is included

```text
.
├── README.md
├── LICENSE.txt
├── requirements.txt
├── data/
│   ├── dataset_scaling_aggregate.csv
│   ├── compute_scaling_aggregate.csv
│   └── parameter_scaling_aggregate.csv
├── figures/
│   ├── compute_scaling_final.{png,svg}
│   ├── dataset_scaling_final.{png,svg}
│   ├── parameter_scaling_final.{png,svg}
│   ├── three_panel_scaling_law_final.{png,svg}
│   └── compute_scaling_continuous_breakpoint_diagnostic.{png,svg}
└── scripts/
    ├── prepare_clean_data.py
    ├── training/
    │   ├── phase1_dataset_scaling.py
    │   ├── phase2_compute_scaling.py
    │   └── phase3_parameter_scaling.py
    ├── orchestration/
    │   ├── phase1_orchestrator.py
    │   ├── phase2_orchestrator.py
    │   └── phase3_orchestrator.py
    ├── audit/
    │   ├── phase2_independent_audit.py
    │   └── phase3_independent_audit.py
    └── plotting/
        ├── render_final_figures.py
        └── generate_full_data_clean_final*.py
```

`render_final_figures.py` is the public plotting entry point. The longer
`generate_full_data_clean_final*.py` files are internal helpers retained to
preserve the exact, audited evolution of the final figure.

## Experimental design

All phases use three fresh random-initialization seeds: `42`, `123`, and
`2026`. The test metric is

```text
Test MAE = mean(abs(CCS_prediction - CCS_target))
```

Prediction and target are evaluated in the original CCS scale (Å²). The test
set is used only for post-training reporting, never for model selection,
scheduling, early stopping, or hyperparameter tuning.

### Phase 1 — dataset scaling

- Fixed model: official `peptdeep.model.ccs.Model_CCS_LSTM`
- Hidden dimension: 256
- Trainable parameters: 712,428
- Dataset sizes: 500 to 465,356 examples (12 nested sizes)
- Optimizer: Adam, learning rate 1e-3
- Selection: minimum validation MAE only
- Output: one converged test MAE per dataset size and seed

### Phase 2 — compute scaling

- Fixed 712,428-parameter model
- Dataset sizes: 500, 2,500, 10,000, 50,000, 100,000, 280,000, 465,356
- Exact optimizer-step checkpoints: 10, 30, 50, 100, 300, 1,000,
  3,000, 10,000, 30,000, 50,000
- Constant learning rate; no scheduler and no early stopping
- Checkpoints are saved immediately after the exact optimizer step and are
  evaluated only after training completes

### Phase 3 — parameter scaling

- Full frozen training set: 465,356 examples
- Hidden dimensions: 32, 64, 128, 256, 512, 1,024
- Measured parameter counts: 49,164 to 9,081,324
- Fresh random initialization for every run
- Validation-only best-checkpoint selection

The final main Panel (c) displays hidden dimensions 32–512. The 1,024-hidden
model remains in `parameter_scaling_aggregate.csv` but is omitted from the
main five-point fit because it provides no additional performance gain under
the fixed dataset/training regime.

## Reproduce the released figures without training

Create an environment and install the plotting dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install matplotlib numpy pandas scipy pyarrow
python scripts/plotting/render_final_figures.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The command reads only the three CSV files under `data/` and writes PNG, SVG,
fit metadata, and QC files under `figures/`. It does not load a model or train
anything.

### Panel (a) convergence-onset analysis

For each three-seed mean learning curve, the code fits the continuous hinge
model below in log10(step)–log10(Test MAE) space:

```text
y = b0 + m1*x + delta*max(0, x - t)
```

The breakpoint `t` is optimized continuously rather than restricted to an
observed checkpoint. Both segments require at least three checkpoints, and a
clear transition requires `abs(post_slope) <= 0.5 * abs(pre_slope)`. The 50%
criterion is fixed before the aggregate power-law fit. The N=465,356 curve is
flagged weak/ambiguous rather than having its threshold adjusted.

The seven retained breakpoints are fitted in log-log space as
`L = a * C^(-k)`. The released fit is:

```text
a  = 86.153990687301
k  = 0.193832538887
R² = 0.593084663152
```

No curve is deleted and no breakpoint is selected to optimize the final R².

### Panels (b) and (c)

For consistent visualization, the final plotted power laws use ordinary least
squares in log10 space. The phase-specific training scripts additionally save
their original-space nonlinear fits and diagnostics.

## Run the full experiments

Full reruns require the AlphaPeptDeep source/package, a CUDA-capable PyTorch
environment, and the source data needed to construct the frozen parquet files.
Define paths through environment variables; no personal server path is baked
into the released code.

```bash
export CCS_SCALING_ROOT=/path/to/ccs_scaling_runs
export ALPHAPEPTDEEP_ROOT=/path/to/alphapeptdeep
export CCS_PYTHON=$(which python)
```

Optional data-preparation variables:

```bash
export CCS_PXD019086_REPORT=/path/to/20201028_ExperimentalLibrary-Report.csv
export CCS_PXD019086_ARCHIVE_ROOT=/path/to/PXD019086_archives
```

Place these official/source files under
`$CCS_SCALING_ROOT/phase0d/source_data/`:

- `SourceData_Figure_1.csv`
- `SourceData_Figure_4.csv`
- `CCS_Alignment.ipynb`

Then prepare and verify the clean-room artifacts:

```bash
python scripts/prepare_clean_data.py
```

Initialize each phase before starting its runs:

```bash
python scripts/training/phase1_dataset_scaling.py init
python scripts/training/phase2_compute_scaling.py init
python scripts/training/phase3_parameter_scaling.py init
python scripts/training/phase3_parameter_scaling.py preflight
```

Run a single task explicitly:

```bash
python scripts/training/phase1_dataset_scaling.py run --n 500 --seed 42
python scripts/training/phase2_compute_scaling.py run --n 500 --seed 42
python scripts/training/phase3_parameter_scaling.py run --hidden 32 --seed 42
```

After all tasks in a phase finish:

```bash
python scripts/training/phase1_dataset_scaling.py aggregate
python scripts/training/phase2_compute_scaling.py aggregate
python scripts/training/phase3_parameter_scaling.py aggregate
```

The optional scripts under `scripts/orchestration/` dispatch the full task
grids using `nvidia-smi`. They are convenience utilities for a local
multi-GPU/single-GPU Linux server, not a requirement for reproducing figures.

After a complete Phase 2 or Phase 3 rerun, the independent checks can be run
with:

```bash
python scripts/audit/phase2_independent_audit.py
python scripts/audit/phase3_independent_audit.py
```

## Reproducibility and safeguards

- Frozen input row counts and SHA-256 values are validated before training.
- Dataset-size subsets reuse nested, seed-specific permutations.
- Runs fail closed if expected checkpoints or evaluations are missing.
- Model checkpoints record optimizer steps, seed, selection rule, and hashes.
- PXD017703 is excluded from every training/validation/sampling pool.
- The aggregate CSV files are sufficient for plot-only reproduction.

## Data and repository hygiene

Do not commit raw reports, parquet datasets, model checkpoints, per-run logs,
or credentials. The supplied `.gitignore` excludes these common artifacts.
Review source-data redistribution terms before publishing any additional data.

## License

The code is distributed under the license in `LICENSE.txt`. AlphaPeptDeep and
its dependencies retain their respective licenses.
