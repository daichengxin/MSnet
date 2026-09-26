# MS2 Spectrum Prediction

This directory contains the code for MS/MS spectrum prediction (MS2) used in the manuscript (Figure 3): the training pipeline for the π-MSNet MS² model, the dataset-size and model-size scaling laws, and the PCC90 benchmark against AlphaPeptDeep, Prosit, and UniSpec.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `msnet_trainer.py` | Training pipeline: dataset construction, the BERT-style MS² model, and distributed training. |
| `Figure3.py` | Scaling laws of Figure 3 — test loss vs. dataset size and vs. parameter count. |
| `Figure3CD.py` | Figure 3C/3D — PCC90 benchmarking across eight instrument/fragmentation conditions. |
| `tests/` | Per-dataset prediction tables (PCC per PSM) and the training-precursor list used by the figure scripts. |
| `benchmark_tests.zip` | Archived copy of `tests/` (89 files, ~10.4 GB uncompressed). |

## Required Inputs

The training tensors, model checkpoints, and prediction tables are too large to ship with the repository. They are available on Zenodo: [https://zenodo.org/records/20792504](https://zenodo.org/records/20792504).

| Path | Contents | Source |
|------|----------|--------|
| `train_data/<project>_precursor_df[_top20].csv` | Precursor annotations per project (sequence, charge, modifications, NCE, instrument, fragment index range). | Zenodo |
| `train_data/<project>_fragment_intensity_df[_top20].csv` | Matching fragment intensities for the `b_z1`, `b_z2`, `y_z1`, `y_z2` channels. | Zenodo |
| `DDP/*.pt` | Model checkpoints (see below). | `msnet_trainer.py` or Zenodo |
| `tests/train_data_precursor_v2.csv` | Precursors seen during training, used as the "seen" reference for the unseen-precursor filter. | `msnet_trainer.py` |
| `tests/*_MSNet_PCC_latest.csv`, `*_Alphapeptdeep.csv`, `*_Prosit.csv`, `*_unispec.csv` | Predicted-vs-experimental PCC per PSM for the four models. | `benchmark_tests.zip` (Zenodo) |
| `PXD009737gluc_inputs.csv`, `PXD009737gluc_ground_truth_with_zero.csv` | Inputs and ground truth for `Figure3CD.test_ground_truth()`. | Zenodo |

`build_train_data()` resolves each of the 29 entries of the `train_data` list in `train()` against a hard-coded set of 27 project-level tables — e.g. `PXD004732`, `PXD010595`, `PXD012131`, `huiyan` / `Cohort_E480_DDAQC` (as `IPX0001804001`), and the four `PXD012636` species splits. Note that the suffixes are not uniform: most projects use `_precursor_df_top20.csv`, others (e.g. `PXD004732`, `PXD002767`) use `_precursor_df.csv`. A branch has to be added when a new project is included.

`tests/train_data_precursor_v2.csv` is **not** included in `benchmark_tests.zip` either, so both figure scripts fail on a fresh checkout until it is regenerated with `msnet_trainer.py` (step 3 below) or restored from Zenodo.

## Training

```bash
python msnet_trainer.py    # runs train()
```

`train()` performs four steps:

1. **Load the training data.** Concatenates the precursor and fragment-intensity tables of all projects listed in `train_data`, offsetting the fragment index ranges so that they index into one shared intensity matrix. Instrument names are normalized to `Lumos`/`QE`/`Exploris480` and NCEs are cast to integers.
2. **Write the scaling-law subsets.** The unique `(sequence, charge, mods, mod_sites)` combinations are halved four times with `random_state=42`, and each level is merged back onto the full PSM table to produce `train_data/scaling_law_P1.csv` (largest) through `scaling_law_P4.csv` (smallest). The header of `Figure3.py` relates these four levels to 2M / 1M / 0.5M / 0.25M precursors (40M / 20M / 10M / 5M PSMs); which checkpoint is evaluated at which reported dataset size is hard-coded in the `make_prediction_*` functions of `Figure3.py`, which load `DDP/final_model_7.pt` followed by `DDP/scaling_law_model_P1.pt` … `P4.pt`.
3. **Write the training-precursor list.** Unmodified peptides (`mods == ""`) are written to `tests/train_data_precursor_v2.csv`; the figure scripts use this file to exclude training precursors from the benchmark.
4. **Train.** `train_data/scaling_law_P4.csv` is loaded and passed to a PyTorch Lightning `Trainer` together with a `DataModule`.

### Configuration

| Setting | Value in the script |
|---------|---------------------|
| Device | `accelerator='gpu'`, `devices=[6, 7]` — edit this for your machine. |
| Strategy | `DDPStrategy(find_unused_parameters=False, static_graph=True)`. |
| Batch size | 1024, always full batches. |
| Batching | `GroupedBatchSampler` groups batches by peptide length (`nAA`) so that every sequence in a batch has the same length; batches are shuffled and split across DDP ranks. |
| Model | `MSNetMS2Model` (subclass of PeptDeep's `pDeepModel`) wrapping `ModelMS2Bert` with `nlayers=4`, `dropout=0.1`. |
| Fragment channels | `b_z1`, `b_z2`, `y_z1`, `y_z2`. |
| Epochs | `max_epochs=100`, `check_val_every_n_epoch=None` (no validation split is held out). |
| Checkpoints | `./DDP`, top 3 by `train_Loss`, named `{epoch}-{train_Loss:.4f}.pt`. |

The model takes the peptide sequence and modification features together with three acquisition parameters (`charge`, `nce`, `instrument`) and predicts the four fragment-intensity channels.

## Figure 3: Scaling Laws

```bash
python Figure3.py
```

`make_prediction_1()` … `make_prediction_5()` load the five checkpoints (`DDP/final_model_7.pt` and `DDP/scaling_law_model_P1.pt` … `P4.pt`); `run_on()` evaluates each of them on the precursors of `tests/train_data_precursor_v2.csv`, computes the L1 loss against the ground-truth intensities for the precursors that were **not** part of training, and returns the summed loss and the number of evaluated PSMs. These are the values quoted in `plot_scaling_laws()`, which writes:

- `scaling_law_test_loss_v3.svg` — test loss vs. dataset size, with the fitted power law `L = b · D^a`.
- `scaling_law_test_loss_parameters_v3.svg` — test loss vs. parameter count for the architecture-ablation checkpoints `MSNet_Epoch100_ratio_full_layer1/2/3/4.pt`, `..._layer3_128.pt`, `..._layer4_dp01.pt`, and `..._layer4_512.pt`.

`__main__` calls `plot_scaling_laws()`, which currently plots the values hard-coded in the function; call `run_on()` first if you want to recompute them from checkpoints.

## Figure 3C/3D: PCC90 Benchmark

```bash
python Figure3CD.py    # runs plot_latest_pcc()
```

`plot_latest_pcc()` reads the four prediction tables of each condition, computes the fraction of PSMs with PCC > 0.90 (PCC90), and repeats the calculation after removing every precursor present in `tests/train_data_precursor_v2.csv` to obtain the "unseen precursors" values. Two figures are written:

- `PCC90_latest_V4.svg` — PCC90 per condition for all PSMs.
- `PCC90_unseen_latest_V4.svg` — PCC90 restricted to precursors not seen during training.

Conditions evaluated (MSNet, AlphaPeptDeep, Prosit, UniSpec each):

| Condition label | Prediction files |
|-----------------|------------------|
| Q Exactive HF/HCD@28 | `PXD012636_Danio_rerio_*` |
| Orbitrap Fusion Lumos/HCD@30 | `IPX0004073001_*` |
| Q Exactive HF-X/Lysc/HCD@30 | `PXD009737_lysc_msnet_*` |
| Q Exactive HF-X/Gluc/HCD@30 | `PXD009737_gluc_msnet_*` |
| Q Exactive HF-X/HCD@27 | `PXD019483_msnet_*` |
| Q Exactive HF/HCD@27 | `PXD014877_Mus_musculus_*` + `PXD014877_Neurospora_*` + `PXD014877_Bacteroides_Fragilis_*` |
| Exploris480/HCD@28&30 | `huiyan_*` |
| Orbitrap Elite/HCD@32 | `PXD000561_*_32_*` |

The PCC column is named `pcc` in the MSNet tables and `PCC` in the AlphaPeptDeep, Prosit, and UniSpec tables.

Two further functions in the file are kept for reference but are not run by default:

- `plot_pcc()` — an earlier version of the benchmark that reads per-PSM PCC arrays from `.npy` files (e.g. `PXD000561_AlphaPeptDeep.npy`) that are not part of the released data.
- `test_ground_truth()` — an intensity-level comparison on the Glu-C dataset, requiring `PXD009737gluc_inputs.csv` and `PXD009737gluc_ground_truth_with_zero.csv`.

## Environment

The scripts were run with `peptdeep` (including `alphabase`), `pytorch-lightning`, `torch`, `pandas`, `numpy`, `seaborn`, `matplotlib`, and `pandarallel`. `Figure3.py` imports the model class from `msnet_trainer.py`, so both files must stay in the same directory.

Note that `Figure3CD.py` uses the private `DataFrame._append` API and was written against an older pandas release; on pandas ≥ 2.2 consider replacing those calls with `pandas.concat`.

## Original Data

The original trained models, prediction results, and benchmarking data are available on Zenodo:

[https://zenodo.org/records/20792504](https://zenodo.org/records/20792504)
