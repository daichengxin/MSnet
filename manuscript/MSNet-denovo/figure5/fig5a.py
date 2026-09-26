#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Title:
    Figure 5a - Peptide recall comparison between π-HelixNovo-raw and
    π-HelixNovo-MSNet across species

Description:
    This script generates Figure 5a for the manuscript. It compares the
    peptide recall of π-HelixNovo-raw with that of π-HelixNovo-MSNet across
    multiple species datasets.

    The π-HelixNovo-raw result is used as the baseline. The
    π-HelixNovo-MSNet results are obtained from three independent training
    runs. For each species, the mean peptide recall and sample standard
    deviation across the three runs are calculated and displayed as the
    point estimate and error bar, respectively.

    Peptide recall values are converted from proportions to percentages.
    The relative improvement of π-HelixNovo-MSNet over π-HelixNovo-raw is
    calculated as:

        (MSNet mean - baseline) / baseline * 100

    Positive and negative relative changes are annotated above the
    corresponding π-HelixNovo-MSNet data points.

Input:
    - summary_helixraw.csv:
        Peptide recall results for π-HelixNovo-raw.

    - summary_helixmsnet_sampled.csv:
        Peptide recall results from the first π-HelixNovo-MSNet run.

    - summary_helixmsnet_sampled2.csv:
        Peptide recall results from the second π-HelixNovo-MSNet run.

    - summary_helixmsnet_sampled3.csv:
        Peptide recall results from the third π-HelixNovo-MSNet run.

    Each input CSV file must contain:
        - dataset:
            Species or dataset name.
        - pep_recall:
            Peptide recall represented as a proportion between 0 and 1.

Output:
    - compare_pep_recall_line.svg:
        Publication-ready line plot showing peptide recall across species.
        Error bars represent the sample standard deviation across three
        independent π-HelixNovo-MSNet runs.

    - summary_pep_recall_comparison.csv:
        Numerical summary containing the baseline peptide recall, the mean
        π-HelixNovo-MSNet peptide recall, the corresponding sample standard
        deviation, and the relative improvement for each species.

Author:
    Tianze Ling, Ph.D. candidate @ Tsinghua University and
    National Center for Protein Sciences (Beijing)

Contact:
    tianzeling98@outlook.com

Usage:
    python fig5a.py

Dependencies:
    - Python >= 3.6
    - pandas
    - NumPy
    - Matplotlib

Notes:
    - "Haloarcula marismortui" is excluded from the analysis.
    - Species are sorted alphabetically before plotting.
    - Scientific names are displayed in italics on the x-axis.
    - Error bars indicate mean ± sample standard deviation (s.d., n = 3).
    - The sample standard deviation is calculated using one degree of
      freedom (ddof = 1).
    - Relative improvement is reported as undefined when the baseline value
      is zero.
    - Arial is used as the global font and must be installed on the system
      for consistent figure rendering.
    - SVG text is preserved as editable text when supported by the plotting
      environment.
===============================================================================
"""


from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================
# 1. Global plotting configuration
# ==============================
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 17,
        "figure.dpi": 150,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 15,
        # Preserve editable text when exporting SVG files.
        "svg.fonttype": "none",
    }
)


# ==============================
# 2. Input and output configuration
# ==============================
DATA_DIR = Path("./")

BASELINE_FILE = "summary_helixraw.csv"

MSNET_FILES = [
    "summary_helixmsnet_sampled1.csv",
    "summary_helixmsnet_sampled2.csv",
    "summary_helixmsnet_sampled3.csv",
]

METRIC = "pep_recall"

# Species excluded from the analysis.
EXCLUDE_SPECIES = [
    "Haloarcula marismortui",
]

FIGURE_FILE = "compare_pep_recall_line.svg"
SUMMARY_FILE = "summary_pep_recall_comparison.csv"


# ==============================
# 3. Data-loading functions
# ==============================
def load_result_table(file_path: Path, metric: str) -> pd.DataFrame:
    """
    Load a result table and set the dataset column as the index.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input CSV file.
    metric : str
        Name of the metric column required for the analysis.

    Returns
    -------
    pandas.DataFrame
        Loaded result table indexed by dataset name.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If required columns are missing or dataset names are duplicated.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    if "dataset" not in df.columns:
        raise ValueError(
            f"Required column 'dataset' is missing from: {file_path}"
        )

    if metric not in df.columns:
        raise ValueError(
            f"Required metric column '{metric}' is missing from: {file_path}"
        )

    if df["dataset"].duplicated().any():
        duplicated_datasets = (
            df.loc[df["dataset"].duplicated(), "dataset"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicated dataset names in {file_path}: "
            f"{duplicated_datasets}"
        )

    return df.set_index("dataset")


def validate_dataset_consistency(
    baseline_df: pd.DataFrame,
    msnet_dfs: List[pd.DataFrame],
) -> None:
    """
    Verify that all MSNet result tables contain the baseline datasets.

    Parameters
    ----------
    baseline_df : pandas.DataFrame
        Baseline result table.
    msnet_dfs : list of pandas.DataFrame
        Result tables from independent MSNet runs.

    Raises
    ------
    ValueError
        If an MSNet result table is missing one or more baseline datasets.
    """
    baseline_datasets = set(baseline_df.index)

    for run_index, df in enumerate(msnet_dfs, start=1):
        missing_datasets = sorted(baseline_datasets - set(df.index))

        if missing_datasets:
            raise ValueError(
                f"MSNet run {run_index} is missing the following datasets: "
                f"{missing_datasets}"
            )



# ==============================
# 4. Main analysis and plotting
# ==============================
def main() -> None:
    """Run the peptide-recall comparison analysis."""

    # Create the output directory if it does not already exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the baseline result table.
    baseline_path = DATA_DIR / BASELINE_FILE
    df_baseline = load_result_table(baseline_path, METRIC)

    # Load results from the three independent MSNet runs.
    msnet_dfs = [
        load_result_table(DATA_DIR / file_name, METRIC)
        for file_name in MSNET_FILES
    ]

    # Remove species excluded from the analysis.
    df_baseline = df_baseline.drop(
        index=EXCLUDE_SPECIES,
        errors="ignore",
    )
    msnet_dfs = [
        df.drop(index=EXCLUDE_SPECIES, errors="ignore")
        for df in msnet_dfs
    ]

    if df_baseline.empty:
        raise ValueError(
            "No datasets remain after applying the exclusion criteria."
        )

    # Verify that every MSNet table contains all baseline datasets.
    validate_dataset_consistency(df_baseline, msnet_dfs)

    # Sort species alphabetically and use the same order for all tables.
    species_list = sorted(df_baseline.index.astype(str).tolist())

    df_baseline = df_baseline.loc[species_list]
    msnet_dfs = [
        df.loc[species_list]
        for df in msnet_dfs
    ]

    # Convert the selected metric to numeric values.
    baseline_values = pd.to_numeric(
        df_baseline[METRIC],
        errors="raise",
    ).to_numpy(dtype=float)

    msnet_stack = np.stack(
        [
            pd.to_numeric(df[METRIC], errors="raise").to_numpy(dtype=float)
            for df in msnet_dfs
        ],
        axis=0,
    )

    # Calculate the mean and sample standard deviation across MSNet runs.
    msnet_mean = msnet_stack.mean(axis=0)
    msnet_std = msnet_stack.std(axis=0, ddof=1)

    # Convert recall values from proportions to percentages.
    baseline_pct = baseline_values * 100.0
    msnet_pct = msnet_mean * 100.0
    msnet_err_pct = msnet_std * 100.0

    # Calculate relative improvement over the baseline.
    # Undefined values are reported as NaN when the baseline is zero.
    relative_improvement = np.divide(
        msnet_pct - baseline_pct,
        baseline_pct,
        out=np.full_like(msnet_pct, np.nan, dtype=float),
        where=baseline_pct != 0,
    ) * 100.0

    # ==============================
    # 5. Generate the line plot
    # ==============================
    fig, ax = plt.subplots(figsize=(16, 7))

    x_positions = np.arange(len(species_list))

    ax.plot(
        x_positions,
        baseline_pct,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="#4A6FA5",
        label="π-HelixNovo-raw",
    )

    ax.errorbar(
        x_positions,
        msnet_pct,
        yerr=msnet_err_pct,
        marker="s",
        markersize=8,
        linewidth=2.2,
        capsize=8,
        capthick=2.2,
        elinewidth=2.2,
        color="#E07A5F",
        label="π-HelixNovo-MSNet (mean ± s.d., n = 3)",
    )

    # Annotate the relative improvement above each MSNet data point.
    for index, improvement in enumerate(relative_improvement):
        if np.isnan(improvement):
            annotation = "NA"
            annotation_color = "#666666"
        else:
            sign = "+" if improvement >= 0 else ""
            annotation = f"{sign}{improvement:.1f}%"
            annotation_color = (
                "black" if improvement >= 0 else "#D62828"
            )

        ax.annotate(
            annotation,
            (
                x_positions[index],
                msnet_pct[index] + msnet_err_pct[index],
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=13,
            color=annotation_color,
            fontweight="bold",
        )

    ax.set_xlabel("Datasets")
    ax.set_ylabel("Peptide recall (%)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        species_list,
        rotation=22,
        ha="right",
        fontstyle="italic",
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        frameon=False,
    )

    # Remove the top and right axis spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add sufficient vertical space for annotations and error bars.
    y_max = max(
        float(np.nanmax(baseline_pct)),
        float(np.nanmax(msnet_pct + msnet_err_pct)),
    )
    ax.set_ylim(0, y_max * 1.25)

    fig.tight_layout()

    figure_path = DATA_DIR / FIGURE_FILE
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()
    plt.close(fig)

    # ==============================
    # 6. Export the numerical summary
    # ==============================
    summary = pd.DataFrame(
        {
            "baseline_pep_recall(%)": baseline_pct,
            "msnet_mean_pep_recall(%)": msnet_pct,
            "msnet_std(%)": msnet_err_pct,
            "relative_improvement(%)": relative_improvement,
        },
        index=species_list,
    )

    summary.index.name = "dataset"
    summary = summary.round(1)

    summary_path = DATA_DIR / SUMMARY_FILE
    summary.to_csv(summary_path, index=True)

    print("\n=== Peptide recall comparison summary ===")
    print(summary.to_string())

    print("\nOutput files:")
    print(f"Figure: {figure_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
