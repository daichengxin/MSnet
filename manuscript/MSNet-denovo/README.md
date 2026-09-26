# MSNet-de novo

This directory contains the code for de novo peptide sequencing (MSNet-de novo) used in the manuscript (Figure 5), together with the input tables needed to reproduce the three panels.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `figure5/` | Scripts, inputs, and the assembled figure for Figure 5. |
| `figure5/fig5a.py` | Figure 5a — peptide recall of π-HelixNovo-raw vs. π-HelixNovo-MSNet across species; mean ± sample s.d. over three independent MSNet runs. |
| `figure5/fig5b.py` | Figure 5b — π-MSNet vs. the nine-species dataset without ricebean, across data scale, sequence coverage, PTM richness, and species breadth. |
| `figure5/fig5c.py` | Figure 5c — peptide length distribution after stripping mass-shift modifications. |
| `figure5/MSNet_fig5.pptx` | Assembled Figure 5 as submitted. |
| `figure5/summary_helixraw.csv` | Peptide recall of π-HelixNovo-raw (baseline) per species. |
| `figure5/summary_helixmsnet_sampled1.csv`, `_sampled2.csv`, `_sampled3.csv` | Peptide recall of π-HelixNovo-MSNet for the three independent runs. |
| `figure5/fig5c_precursors.csv` | Precursor table (237 MB) used for the peptide length distribution. |
| `figure5/fig5a_summary_helixraw.csv`, `figure5/fig5a_summary_helixmsnet.csv` | Earlier single-run summaries, superseded by the files above; not read by the current scripts. |

## Inputs

The three result tables read by `fig5a.py` share the same layout — one row per dataset, with the columns `dataset`, `aa_precision`, `aa_recall`, and `pep_recall` (proportions between 0 and 1):

| File | Meaning |
|------|---------|
| `summary_helixraw.csv` | π-HelixNovo-raw baseline. |
| `summary_helixmsnet_sampled1.csv` … `_sampled3.csv` | Three independent π-HelixNovo-MSNet training runs. |

`fig5c.py` reads `fig5c_precursors.csv`, which contains an index column plus `Titles`, `Peptides`, and `Charges`; peptide sequences carry mass-shift annotations such as `+15.9949` or `+79.9663`, which the script strips with the pattern `+<digits>.<digits>` before measuring length.

`fig5b.py` requires no input file — the values of both datasets are defined inside the script, with the first value of each metric list corresponding to π-MSNet and the second to the nine-species dataset without ricebean.

## Running

```bash
cd figure5
python fig5a.py
python fig5b.py
python fig5c.py
```

| Script | Output |
|--------|--------|
| `fig5a.py` | `compare_pep_recall_line.svg` and `summary_pep_recall_comparison.csv` (baseline, MSNet mean, s.d., and relative improvement per species). |
| `fig5b.py` | `msnet_diversity_three_panels.svg` (percentage annotations give the increase of π-MSNet over the nine-species dataset). |
| `fig5c.py` | `peptide_length.svg` and console statistics on modifications, stripped peptides, length distribution, and length diversity. |

Dependencies: `pandas`, `numpy`, `matplotlib` (with Arial installed for consistent rendering; SVG text is kept editable where supported).

Note that `fig5a.py` excludes *Haloarcula marismortui* and sorts species alphabetically; relative improvement is reported as undefined where the baseline is zero.

## Original Data

The original trained models, prediction results, and benchmarking data are available on Zenodo:

[https://zenodo.org/records/20792504](https://zenodo.org/records/20792504)
