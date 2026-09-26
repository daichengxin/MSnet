"""Render the current final CCS scaling figures with Y label ``Test MAE``.

The current final visual state (blue Panel A, no error bars, no grid, and
``Test L1 Loss`` axes) is preserved exactly except for the requested Y-axis
text. Existing plot-data CSV and fit JSON files are read verbatim. No metric is
computed, no model is trained, and no experimental data is written.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from generate_full_data_clean_final import (
    OUTPUT_DIR,
    plot_compute,
    plot_dataset,
    plot_parameter,
)
from generate_full_data_clean_final_blue import (
    reference_limits,
    requested_blue_colors,
)
from generate_full_data_clean_final_nogrid import remove_all_grid


Y_LABEL = "Test MAE"


def load_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    compute = pd.read_csv(OUTPUT_DIR / "plot_data_compute.csv")
    dataset = pd.read_csv(OUTPUT_DIR / "plot_data_dataset.csv")
    parameter = pd.read_csv(OUTPUT_DIR / "plot_data_parameter_main5.csv")
    dataset_fit = json.loads((OUTPUT_DIR / "plot_fit_dataset.json").read_text(encoding="utf-8"))
    parameter_fit = json.loads(
        (OUTPUT_DIR / "plot_fit_parameter_main5.json").read_text(encoding="utf-8")
    )
    return compute, dataset, parameter, dataset_fit, parameter_fit


def save(figure: plt.Figure, stem: str) -> list[str]:
    figure.tight_layout()
    outputs = []
    for extension in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{extension}"
        figure.savefig(path, format=extension, bbox_inches="tight", dpi=300)
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def finish_axis(axis: plt.Axes, limits) -> None:
    axis.set_xlim(limits[0])
    axis.set_ylim(limits[1])
    remove_all_grid(axis)


def main() -> None:
    compute, dataset, parameter, dataset_fit, parameter_fit = load_frozen_inputs()
    colors = requested_blue_colors()

    # Recover and preserve the exact axes limits of the current final version.
    compute_limits = reference_limits(plot_compute, compute)
    dataset_limits = reference_limits(plot_dataset, dataset, dataset_fit)
    parameter_limits = reference_limits(plot_parameter, parameter, parameter_fit)

    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(6, 5))
    plot_compute(
        axis,
        compute,
        show_errorbars=False,
        curve_colors=colors,
        ylabel=Y_LABEL,
    )
    finish_axis(axis, compute_limits)
    outputs.extend(save(figure, "compute_scaling_final"))

    figure, axis = plt.subplots(figsize=(6, 5))
    plot_dataset(axis, dataset, dataset_fit, show_errorbars=False, ylabel=Y_LABEL)
    finish_axis(axis, dataset_limits)
    outputs.extend(save(figure, "dataset_scaling_final"))

    figure, axis = plt.subplots(figsize=(6, 5))
    plot_parameter(axis, parameter, parameter_fit, show_errorbars=False, ylabel=Y_LABEL)
    finish_axis(axis, parameter_limits)
    outputs.extend(save(figure, "parameter_scaling_final"))

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_compute(
        axes[0],
        compute,
        show_errorbars=False,
        curve_colors=colors,
        ylabel=Y_LABEL,
    )
    plot_dataset(axes[1], dataset, dataset_fit, show_errorbars=False, ylabel=Y_LABEL)
    plot_parameter(
        axes[2], parameter, parameter_fit, show_errorbars=False, ylabel=Y_LABEL
    )
    for axis, limits in zip(
        axes, (compute_limits, dataset_limits, parameter_limits)
    ):
        finish_axis(axis, limits)
    outputs.extend(save(figure, "three_panel_scaling_law_final"))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
