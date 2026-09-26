"""Add a reproducible convergence-onset power-law fit to Panel (a) only.

This is a plot-only workflow. It reads the frozen plotting tables and fit
metadata used by the current final figures. It does not train a model, alter
experimental data, or recompute the existing Panel (b)/(c) fits.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import shgo

from generate_full_data_clean_final import (
    ORANGE,
    OUTPUT_DIR,
    plot_compute,
    plot_dataset,
    plot_parameter,
    power_law_fit,
)
from generate_full_data_clean_final_blue import reference_limits, requested_blue_colors
from generate_full_data_clean_final_mae import Y_LABEL, finish_axis, save


REPORT_PATH = OUTPUT_DIR / "compute_scaling_convergence_fit.md"
QC_CSV_PATH = OUTPUT_DIR / "compute_scaling_continuous_breakpoints.csv"
MIN_POINTS_PER_SEGMENT = 3
MAX_POST_TO_PRE_SLOPE_RATIO = 0.50


def load_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    compute = pd.read_csv(OUTPUT_DIR / "plot_data_compute.csv")
    dataset = pd.read_csv(OUTPUT_DIR / "plot_data_dataset.csv")
    parameter = pd.read_csv(OUTPUT_DIR / "plot_data_parameter_main5.csv")
    dataset_fit = json.loads(
        (OUTPUT_DIR / "plot_fit_dataset.json").read_text(encoding="utf-8")
    )
    parameter_fit = json.loads(
        (OUTPUT_DIR / "plot_fit_parameter_main5.json").read_text(encoding="utf-8")
    )
    return compute, dataset, parameter, dataset_fit, parameter_fit


def select_convergence_onsets(compute: pd.DataFrame) -> pd.DataFrame:
    """Fit continuous two-segment regressions in log10-log10 space.

    For a continuous breakpoint ``t``, the hinge model is
    ``y = b0 + m1*x + delta*max(0, x-t)``. Thus the pre-break slope is ``m1``
    and the post-break slope is ``m1 + delta``. The primary solution minimizes
    total squared residual subject to at least three observed checkpoints on
    each side and ``abs(post_slope) <= 0.5 * abs(pre_slope)``.

    If the fixed 50% flattening constraint is infeasible, the curve is marked
    weak/ambiguous. For the requested seven-point aggregate fit, its diagnostic
    point is the minimum-residual solution under the weaker requirement that
    the post-break absolute slope is merely smaller than the pre-break slope.
    """

    def fit_at_breakpoint(
        breakpoint: float, log_step: np.ndarray, log_loss: np.ndarray
    ) -> tuple[float, float, float, np.ndarray]:
        breakpoint = float(np.atleast_1d(breakpoint)[0])
        design = np.column_stack(
            (
                np.ones_like(log_step),
                log_step,
                np.maximum(0.0, log_step - breakpoint),
            )
        )
        coefficients = np.linalg.lstsq(design, log_loss, rcond=None)[0]
        residual = log_loss - design @ coefficients
        pre_slope = float(coefficients[1])
        post_slope = float(coefficients[1] + coefficients[2])
        residual_sse = float(residual @ residual)
        return residual_sse, pre_slope, post_slope, coefficients

    def optimize(
        log_step: np.ndarray,
        log_loss: np.ndarray,
        max_ratio: float,
    ):
        lower = float(log_step[MIN_POINTS_PER_SEGMENT - 1])
        upper = float(log_step[-MIN_POINTS_PER_SEGMENT])

        def objective(value) -> float:
            return fit_at_breakpoint(value, log_step, log_loss)[0]

        def flattening_constraint(value) -> float:
            _, pre_slope, post_slope, _ = fit_at_breakpoint(
                value, log_step, log_loss
            )
            return max_ratio * abs(pre_slope) - abs(post_slope)

        result = shgo(
            objective,
            [(lower, upper)],
            constraints={"type": "ineq", "fun": flattening_constraint},
            n=512,
            iters=3,
            sampling_method="sobol",
        )
        return result, lower, upper

    selected_rows: list[dict] = []
    for n_value, group in compute.groupby("N", sort=True):
        group = group.sort_values("optimizer_step").reset_index(drop=True)
        log_step = np.log10(group["optimizer_step"].to_numpy(dtype=float))
        log_loss = np.log10(group["mean_test_L1"].to_numpy(dtype=float))
        result, lower, upper = optimize(
            log_step, log_loss, MAX_POST_TO_PRE_SLOPE_RATIO
        )
        if result.success:
            status = "clear"
        else:
            # Do not tune the fixed 50% criterion. Retain the curve, flag it,
            # and find the best point having any genuine slope flattening.
            result, lower, upper = optimize(log_step, log_loss, 1.0 - 1e-9)
            if not result.success:
                raise RuntimeError(
                    f"No continuous flattening breakpoint found for N={n_value}"
                )
            status = "weak/ambiguous"

        breakpoint_log_step = float(result.x[0])
        residual_sse, pre_slope, post_slope, coefficients = fit_at_breakpoint(
            breakpoint_log_step, log_step, log_loss
        )
        breakpoint_step = float(10**breakpoint_log_step)
        breakpoint_log_loss = float(
            coefficients[0] + coefficients[1] * breakpoint_log_step
        )
        breakpoint_loss = float(10**breakpoint_log_loss)
        at_support_boundary = bool(
            np.isclose(breakpoint_log_step, lower, atol=1e-6)
            or np.isclose(breakpoint_log_step, upper, atol=1e-6)
        )
        selected_rows.append(
            {
                "N": int(n_value),
                "continuous_breakpoint_step": breakpoint_step,
                "breakpoint_Test_MAE": breakpoint_loss,
                "pre_break_slope": pre_slope,
                "post_break_slope": post_slope,
                "post_to_pre_abs_slope_ratio": abs(post_slope) / abs(pre_slope),
                "fit_residual_log10_sse": residual_sse,
                "transition_status": status,
                "at_support_boundary": at_support_boundary,
            }
        )

    return pd.DataFrame(selected_rows)


def add_compute_fit(axis: plt.Axes, selected: pd.DataFrame, fit: dict) -> None:
    x = selected["continuous_breakpoint_step"].to_numpy(dtype=float)
    y = selected["breakpoint_Test_MAE"].to_numpy(dtype=float)

    axis.plot(
        x,
        y,
        linestyle="none",
        marker="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.2,
        zorder=5,
    )

    fit_x = np.logspace(np.log10(x.min() * 0.5), np.log10(x.max() * 2.0), 100)
    fit_y = fit["coefficient"] * fit_x ** (-fit["exponent"])
    axis.plot(
        fit_x,
        fit_y,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )

    label = (
        rf"Power law: $L = {fit['coefficient']:.2f} \cdot C^"
        rf"{{-{fit['exponent']:.4f}}}$" + "\n" + rf"$R^2 = {fit['R2']:.4f}$"
    )
    axis.text(
        0.98,
        0.98,
        label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
        zorder=6,
    )


def make_diagnostic(
    compute: pd.DataFrame,
    selected: pd.DataFrame,
    colors,
    compute_limits,
) -> list[str]:
    figure, axis = plt.subplots(figsize=(6, 5))
    plot_compute(
        axis,
        compute,
        show_errorbars=False,
        curve_colors=colors,
        ylabel=Y_LABEL,
    )
    x = selected["continuous_breakpoint_step"].to_numpy(dtype=float)
    y = selected["breakpoint_Test_MAE"].to_numpy(dtype=float)
    axis.plot(
        x,
        y,
        linestyle="none",
        marker="D",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.3,
        zorder=5,
    )
    label_offsets = {
        500: (8, 14),
        2500: (8, -15),
        10000: (8, -18),
        50000: (8, 8),
        100000: (-74, -20),
        280000: (10, 4),
        465356: (10, 14),
    }
    for row in selected.itertuples(index=False):
        suffix = " *" if row.transition_status == "weak/ambiguous" else ""
        axis.annotate(
            f"N={row.N}{suffix}",
            (row.continuous_breakpoint_step, row.breakpoint_Test_MAE),
            xytext=label_offsets[row.N],
            textcoords="offset points",
            fontsize=7,
            color=ORANGE,
            arrowprops={"arrowstyle": "-", "color": ORANGE, "alpha": 0.65, "lw": 0.7},
            zorder=6,
        )
    axis.text(
        0.98,
        0.03,
        "* weak/ambiguous transition",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=ORANGE,
    )
    finish_axis(axis, compute_limits)
    return save(figure, "compute_scaling_continuous_breakpoint_diagnostic")


def write_report(selected: pd.DataFrame, fit: dict) -> None:
    rule = (
        "Fit the continuous hinge model `y = b0 + m1*x + delta*max(0, x-t)` "
        "in log10(step)-log10(Test MAE) space. Optimize continuous breakpoint "
        "`t` for minimum total log10 residual SSE, require at least 3 formal "
        "checkpoints on each side, and require "
        "`abs(post_slope) <= 0.50 * abs(pre_slope)`. The 50% criterion was fixed "
        "before fitting and was not tuned against the aggregate R2. If it is "
        "infeasible, mark the curve weak/ambiguous and retain its best supported "
        "flattening point without deleting the curve."
    )
    lines = [
        "# Panel (a) convergence-onset fit",
        "",
        "## Selection rule",
        "",
        rule,
        "",
        "## Selected points",
        "",
        "| N | continuous_breakpoint_step | breakpoint_Test_MAE | pre_break_slope | post_break_slope | fit_residual_log10_sse | status |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.N} | {row.continuous_breakpoint_step:.6f} | "
            f"{row.breakpoint_Test_MAE:.12f} | {row.pre_break_slope:.6f} | "
            f"{row.post_break_slope:.6f} | {row.fit_residual_log10_sse:.9f} | "
            f"{row.transition_status}"
            f"{' (support-boundary)' if row.at_support_boundary else ''} |"
        )
    lines.extend(
        [
            "",
            "## Power-law fit",
            "",
            "Ordinary least squares in log10 space, matching Panels (b)/(c):",
            "",
            f"- Formula: `L = {fit['coefficient']:.12f} * C^(-{fit['exponent']:.12f})`",
            f"- a = `{fit['coefficient']:.12f}`",
            f"- k = `{fit['exponent']:.12f}`",
            f"- R2 (log10 loss) = `{fit['R2']:.12f}`",
            f"- Number of representative points = `{fit['n_points']}`",
            "- OLD discrete-breakpoint R2 = `0.5846`",
            f"- NEW continuous-breakpoint R2 = `{fit['R2']:.12f}`",
            "",
            "The same seven curves are retained. Any R2 change comes from "
            "continuous breakpoint estimation and hinge-model interpolation, "
            "not from data filtering or modification.",
            "",
            "No model was retrained and no existing experimental data or Panel "
            "(b)/(c) fit result was modified.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    compute, dataset, parameter, dataset_fit, parameter_fit = load_frozen_inputs()
    selected = select_convergence_onsets(compute)
    compute_fit = power_law_fit(
        selected["continuous_breakpoint_step"].to_numpy(dtype=float),
        selected["breakpoint_Test_MAE"].to_numpy(dtype=float),
        "C",
    )

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
    add_compute_fit(axis, selected, compute_fit)
    finish_axis(axis, compute_limits)
    outputs.extend(save(figure, "compute_scaling_final"))

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_compute(
        axes[0],
        compute,
        show_errorbars=False,
        curve_colors=colors,
        ylabel=Y_LABEL,
    )
    add_compute_fit(axes[0], selected, compute_fit)
    plot_dataset(
        axes[1], dataset, dataset_fit, show_errorbars=False, ylabel=Y_LABEL
    )
    plot_parameter(
        axes[2], parameter, parameter_fit, show_errorbars=False, ylabel=Y_LABEL
    )
    for axis, limits in zip(
        axes, (compute_limits, dataset_limits, parameter_limits)
    ):
        finish_axis(axis, limits)
    outputs.extend(save(figure, "three_panel_scaling_law_final"))

    outputs.extend(make_diagnostic(compute, selected, colors, compute_limits))
    selected.to_csv(QC_CSV_PATH, index=False)
    write_report(selected, compute_fit)
    for output in outputs:
        print(output)
    print(QC_CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
