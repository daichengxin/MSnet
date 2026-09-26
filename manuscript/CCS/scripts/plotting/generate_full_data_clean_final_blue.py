"""Render the frozen no-error-bar final figures with the requested blue Panel A.

Inputs are the existing final plot-data CSV and fit JSON files. This script does
not fit, train, split, or write experimental data. It only writes new ``*_blue``
PNG/SVG figure files.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

from generate_full_data_clean_final import (
    EXPECTED_COMPUTE_N,
    OUTPUT_DIR,
    plot_compute,
    plot_dataset,
    plot_parameter,
)


Y_LABEL = "Test L1 Loss"


def load_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    compute = pd.read_csv(OUTPUT_DIR / "plot_data_compute.csv")
    dataset = pd.read_csv(OUTPUT_DIR / "plot_data_dataset.csv")
    parameter = pd.read_csv(OUTPUT_DIR / "plot_data_parameter_main5.csv")
    dataset_fit = json.loads((OUTPUT_DIR / "plot_fit_dataset.json").read_text(encoding="utf-8"))
    parameter_fit = json.loads(
        (OUTPUT_DIR / "plot_fit_parameter_main5.json").read_text(encoding="utf-8")
    )
    return compute, dataset, parameter, dataset_fit, parameter_fit


def requested_blue_colors() -> np.ndarray:
    """Build the requested reversed Blues palette and map larger N darker."""
    n = len(EXPECTED_COMPUTE_N)
    blue_colors = cm.Blues(np.linspace(0.35, 0.90, n))[::-1]
    # The palette above is dark-to-light. Associate it with descending N, then
    # return colors in the unchanged ascending curve-drawing order.
    color_by_n = dict(zip(reversed(EXPECTED_COMPUTE_N), blue_colors))
    return np.asarray([color_by_n[n_value] for n_value in EXPECTED_COMPUTE_N])


def reference_limits(plotter, *args) -> tuple[tuple[float, float], tuple[float, float]]:
    """Recover axes limits from the current final figure, including old SD extent."""
    figure, axis = plt.subplots(figsize=(6, 5))
    plotter(axis, *args, show_errorbars=True)
    figure.canvas.draw()
    limits = axis.get_xlim(), axis.get_ylim()
    plt.close(figure)
    return limits


def save(figure: plt.Figure, stem: str) -> list[str]:
    figure.tight_layout()
    outputs = []
    for extension in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{extension}"
        figure.savefig(path, format=extension, bbox_inches="tight", dpi=300)
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def main() -> None:
    compute, dataset, parameter, dataset_fit, parameter_fit = load_frozen_inputs()
    colors = requested_blue_colors()

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
    axis.set_xlim(compute_limits[0])
    axis.set_ylim(compute_limits[1])
    outputs.extend(save(figure, "compute_scaling_final_blue"))

    figure, axis = plt.subplots(figsize=(6, 5))
    plot_dataset(axis, dataset, dataset_fit, show_errorbars=False, ylabel=Y_LABEL)
    axis.set_xlim(dataset_limits[0])
    axis.set_ylim(dataset_limits[1])
    outputs.extend(save(figure, "dataset_scaling_final_blue"))

    figure, axis = plt.subplots(figsize=(6, 5))
    plot_parameter(axis, parameter, parameter_fit, show_errorbars=False, ylabel=Y_LABEL)
    axis.set_xlim(parameter_limits[0])
    axis.set_ylim(parameter_limits[1])
    outputs.extend(save(figure, "parameter_scaling_final_blue"))

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
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
    outputs.extend(save(figure, "three_panel_scaling_law_final_blue"))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
