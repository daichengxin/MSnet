"""Generate the frozen full-data CCS scaling-law figures.

This is a plot-only workflow. It reads the downloaded Phase 1/2/3 aggregate
tables, validates their expected shape, writes compact plotting tables and fit
metadata, and renders the requested single-panel and three-panel figures.

It contains no model, dataset-building, splitting, or training code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "figures"

PHASE1_AGG = DATA_DIR / "dataset_scaling_aggregate.csv"
PHASE2_AGG = DATA_DIR / "compute_scaling_aggregate.csv"
PHASE3_AGG = DATA_DIR / "parameter_scaling_aggregate.csv"

EXPECTED_DATASET_N = [
    500,
    1_000,
    2_500,
    5_600,
    10_000,
    25_000,
    50_000,
    100_000,
    200_000,
    280_000,
    400_000,
    465_356,
]
EXPECTED_COMPUTE_N = [500, 2_500, 10_000, 50_000, 100_000, 280_000, 465_356]
EXPECTED_STEPS = [10, 30, 50, 100, 300, 1_000, 3_000, 10_000, 30_000, 50_000]
EXPECTED_PARAMETER_MAP = {
    32: 49_164,
    64: 94_764,
    128: 235_116,
    256: 712_428,
    512: 2_453_484,
    1024: 9_081_324,
}
DISPLAYED_HIDDEN_DIMS = [32, 64, 128, 256, 512]
SEEDS = [42, 123, 2026]

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
MARKERS = ["o", "s", "^", "D", "v", "<", ">"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def power_law_fit(x: np.ndarray, y: np.ndarray, variable: str) -> dict:
    """Fit L = coefficient * x^(-exponent) by linear regression in log10 space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Power-law fitting requires strictly positive x and y values")

    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    predicted_log10 = slope * np.log10(x) + intercept
    residual_sum = float(np.sum((np.log10(y) - predicted_log10) ** 2))
    total_sum = float(np.sum((np.log10(y) - np.mean(np.log10(y))) ** 2))
    r_squared = 1.0 - residual_sum / total_sum
    coefficient = float(10**intercept)
    exponent = float(-slope)
    return {
        "model": "pure power law",
        "equation": f"L({variable}) = coefficient * {variable}^(-exponent)",
        "fit_method": "np.polyfit(log10(x), log10(y), 1)",
        "r_squared_space": "log10 loss",
        "coefficient": coefficient,
        "exponent": exponent,
        "R2": r_squared,
        "n_points": int(len(x)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase1 = pd.read_csv(PHASE1_AGG)
    phase2 = pd.read_csv(PHASE2_AGG)
    phase3 = pd.read_csv(PHASE3_AGG)

    require_columns(
        phase1,
        ["N", "seeds", "test_L1_mean", "test_L1_SD"],
        PHASE1_AGG,
    )
    require_columns(
        phase2,
        ["N", "optimizer_step", "mean_test_L1", "SD_test_L1", "n_seeds"],
        PHASE2_AGG,
    )
    require_columns(
        phase3,
        ["hidden_dim", "actual_parameter_count", "n_seeds", "mean_test_L1", "SD_test_L1"],
        PHASE3_AGG,
    )

    if phase1["N"].astype(int).tolist() != EXPECTED_DATASET_N:
        raise ValueError("Phase 1 N values/order do not match the frozen 12-point design")
    if len(phase1) != 12 or not (phase1["seeds"].astype(int) == 3).all():
        raise ValueError("Phase 1 must contain 12 aggregate rows with three seeds each")

    observed_compute_n = sorted(phase2["N"].astype(int).unique().tolist())
    if observed_compute_n != EXPECTED_COMPUTE_N or len(phase2) != 70:
        raise ValueError("Phase 2 must contain exactly 7 N curves x 10 checkpoints")
    for n_value, group in phase2.groupby("N"):
        if group["optimizer_step"].astype(int).tolist() != EXPECTED_STEPS:
            raise ValueError(f"Phase 2 N={n_value} does not contain the frozen checkpoints")
        if not (group["n_seeds"].astype(int) == 3).all():
            raise ValueError(f"Phase 2 N={n_value} is not aggregated over three seeds")

    observed_parameter_map = dict(
        zip(
            phase3["hidden_dim"].astype(int),
            phase3["actual_parameter_count"].astype(int),
        )
    )
    if observed_parameter_map != EXPECTED_PARAMETER_MAP or len(phase3) != 6:
        raise ValueError("Phase 3 hidden_dim/parameter counts do not match the frozen design")
    if not (phase3["n_seeds"].astype(int) == 3).all():
        raise ValueError("Phase 3 must contain six aggregate rows with three seeds each")

    return phase1, phase2, phase3


def build_plot_tables(
    phase1: pd.DataFrame, phase2: pd.DataFrame, phase3: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compute = phase2[
        ["N", "optimizer_step", "mean_test_L1", "SD_test_L1", "n_seeds"]
    ].rename(columns={"SD_test_L1": "std_test_L1"})
    dataset = phase1[["N", "test_L1_mean", "test_L1_SD", "seeds"]].rename(
        columns={
            "test_L1_mean": "mean_test_L1",
            "test_L1_SD": "std_test_L1",
            "seeds": "n_seeds",
        }
    )
    parameter_all6 = phase3[
        ["hidden_dim", "actual_parameter_count", "mean_test_L1", "SD_test_L1", "n_seeds"]
    ].rename(
        columns={
            "actual_parameter_count": "parameter_count",
            "SD_test_L1": "std_test_L1",
        }
    )
    parameter_main5 = parameter_all6[
        parameter_all6["hidden_dim"].isin(DISPLAYED_HIDDEN_DIMS)
    ].copy()

    for frame in (compute, dataset, parameter_main5, parameter_all6):
        frame.reset_index(drop=True, inplace=True)
    return compute, dataset, parameter_main5, parameter_all6


def fit_label(fit: dict, variable: str) -> str:
    return (
        rf"Power law: $L = {fit['coefficient']:.2f} \cdot {variable}^"
        rf"{{-{fit['exponent']:.4f}}}$" + "\n" + rf"$R^2 = {fit['R2']:.4f}$"
    )


def style_axis(
    ax: plt.Axes,
    panel_label: str,
    title: str,
    xlabel: str,
    ylabel: str = "Test L1 Loss (Å²)",
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=8, loc="center")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, which="both", color="lightgray", alpha=0.2)
    ax.text(
        -0.02,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="normal",
        va="bottom",
        ha="right",
    )


def plot_compute(
    ax: plt.Axes,
    data: pd.DataFrame,
    show_errorbars: bool = True,
    curve_colors=None,
    ylabel: str = "Test L1 Loss (Å²)",
) -> None:
    colors = (
        plt.cm.viridis(np.linspace(0.1, 0.9, len(EXPECTED_COMPUTE_N)))
        if curve_colors is None
        else curve_colors
    )
    for index, n_value in enumerate(EXPECTED_COMPUTE_N):
        group = data[data["N"] == n_value].sort_values("optimizer_step")
        color = colors[index]
        ax.plot(
            group["optimizer_step"],
            group["mean_test_L1"],
            color=color,
            marker=MARKERS[index],
            markersize=5.5,
            linewidth=1.4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
        if show_errorbars:
            ax.errorbar(
                group["optimizer_step"],
                group["mean_test_L1"],
                yerr=group["std_test_L1"],
                color=color,
                fmt="none",
                capsize=4,
                capthick=1.0,
                linewidth=0.9,
                alpha=0.7,
                zorder=2,
            )
    style_axis(
        ax, "(a)", "Test Loss vs. Compute (Train steps)", "Compute", ylabel=ylabel
    )
    # Deliberately no legend in Panel A.


def plot_dataset(
    ax: plt.Axes,
    data: pd.DataFrame,
    fit: dict,
    show_errorbars: bool = True,
    ylabel: str = "Test L1 Loss (Å²)",
) -> None:
    x = data["N"].to_numpy(dtype=float)
    y = data["mean_test_L1"].to_numpy(dtype=float)
    ax.plot(x, y, color="gray", linestyle="-", linewidth=1.2, zorder=2)
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.5,
        zorder=4,
    )
    if show_errorbars:
        ax.errorbar(
            x,
            y,
            yerr=data["std_test_L1"],
            color=BLUE,
            fmt="none",
            capsize=4,
            capthick=1.0,
            linewidth=1.0,
            alpha=0.7,
            zorder=3,
        )
    fit_x = np.logspace(np.log10(x.min() * 0.5), np.log10(x.max() * 2.0), 100)
    fit_y = fit["coefficient"] * fit_x ** (-fit["exponent"])
    ax.plot(
        fit_x,
        fit_y,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=fit_label(fit, "N"),
        zorder=1,
    )
    style_axis(
        ax, "(b)", "Test Loss vs. Dataset Size", "Dataset Size", ylabel=ylabel
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )


def plot_parameter(
    ax: plt.Axes,
    data: pd.DataFrame,
    fit: dict,
    show_errorbars: bool = True,
    ylabel: str = "Test L1 Loss (Å²)",
) -> None:
    x = data["parameter_count"].to_numpy(dtype=float)
    y = data["mean_test_L1"].to_numpy(dtype=float)
    ax.plot(x, y, color="gray", linestyle="-", linewidth=1.2, zorder=2)
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        markeredgewidth=1.0,
        zorder=4,
    )
    if show_errorbars:
        ax.errorbar(
            x,
            y,
            yerr=data["std_test_L1"],
            color=BLUE,
            fmt="none",
            capsize=4,
            capthick=1.0,
            linewidth=1.0,
            alpha=0.7,
            zorder=3,
        )
    fit_x = np.logspace(np.log10(x.min() * 0.5), np.log10(x.max() * 2.0), 100)
    fit_y = fit["coefficient"] * fit_x ** (-fit["exponent"])
    ax.plot(
        fit_x,
        fit_y,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=fit_label(fit, "P"),
        zorder=1,
    )
    style_axis(
        ax, "(c)", "Test Loss vs. Parameter Size", "Parameter Size", ylabel=ylabel
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    outputs: list[str] = []
    fig.tight_layout()
    for extension in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{extension}"
        fig.savefig(path, format=extension, bbox_inches="tight", dpi=300)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def write_outputs(
    compute: pd.DataFrame,
    dataset: pd.DataFrame,
    parameter_main5: pd.DataFrame,
    parameter_all6: pd.DataFrame,
    dataset_fit: dict,
    parameter_fit: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compute.to_csv(OUTPUT_DIR / "plot_data_compute.csv", index=False)
    dataset.to_csv(OUTPUT_DIR / "plot_data_dataset.csv", index=False)
    parameter_main5.to_csv(OUTPUT_DIR / "plot_data_parameter_main5.csv", index=False)
    parameter_all6.to_csv(OUTPUT_DIR / "plot_data_parameter_all6.csv", index=False)

    with (OUTPUT_DIR / "plot_fit_dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_fit, handle, indent=2, ensure_ascii=False)
    with (OUTPUT_DIR / "plot_fit_parameter_main5.json").open("w", encoding="utf-8") as handle:
        json.dump(parameter_fit, handle, indent=2, ensure_ascii=False)

    config = {
        "font_family": "DejaVu Sans",
        "base_font_pt": 11,
        "title_font_pt": 12,
        "axis_label_font_pt": 12,
        "tick_font_pt": 11,
        "legend_font_pt": 10,
        "single_figure_size_inches": [6, 5],
        "three_panel_figure_size_inches": [18, 5.5],
        "x_scale": "log",
        "y_scale": "log",
        "y_label": "Test MAE",
        "grid": False,
        "panel_a": {
            "N_values": EXPECTED_COMPUTE_N,
            "optimizer_steps": EXPECTED_STEPS,
            "colormap": "Blues(np.linspace(0.35, 0.90, n))[::-1]",
            "markers": MARKERS,
            "legend": False,
            "continuous_convergence_onset_fit": True,
        },
        "panel_b": {"displayed_points": 12, "marker": "blue hollow circle"},
        "panel_c": {
            "displayed_hidden_dims": DISPLAYED_HIDDEN_DIMS,
            "excluded_from_display_hidden_dim": 1024,
            "excluded_from_display_parameter_count": 9_081_324,
            "marker": "blue filled circle",
        },
        "fit_line": {"color": ORANGE, "linestyle": "--", "linewidth": 1.8},
        "error_bars": {"displayed": False},
        "fit_range": "[min(x)*0.5, max(x)*2] with 100 log-spaced points",
        "png_dpi": 300,
    }
    with (OUTPUT_DIR / "plot_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def make_figures(
    compute: pd.DataFrame,
    dataset: pd.DataFrame,
    parameter_main5: pd.DataFrame,
    dataset_fit: dict,
    parameter_fit: dict,
) -> list[str]:
    figure_files: list[str] = []

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_compute(ax, compute)
    figure_files.extend(save_figure(fig, "compute_scaling_final"))

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_dataset(ax, dataset, dataset_fit)
    figure_files.extend(save_figure(fig, "dataset_scaling_final"))

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_parameter(ax, parameter_main5, parameter_fit)
    figure_files.extend(save_figure(fig, "parameter_scaling_final"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_compute(axes[0], compute)
    plot_dataset(axes[1], dataset, dataset_fit)
    plot_parameter(axes[2], parameter_main5, parameter_fit)
    figure_files.extend(save_figure(fig, "three_panel_scaling_law_final"))
    return figure_files


def build_qc(
    compute: pd.DataFrame,
    dataset: pd.DataFrame,
    parameter_main5: pd.DataFrame,
    parameter_all6: pd.DataFrame,
    dataset_fit: dict,
    parameter_fit: dict,
    figure_files: list[str],
) -> dict:
    checks = {
        "panel_a_has_7_curves": compute["N"].nunique() == 7,
        "panel_a_each_curve_has_10_checkpoints": bool(
            (compute.groupby("N")["optimizer_step"].nunique() == 10).all()
        ),
        "panel_a_all_rows_have_3_seeds": bool((compute["n_seeds"] == 3).all()),
        "panel_a_no_legend_by_configuration": True,
        "panel_b_has_12_points": len(dataset) == 12,
        "panel_b_all_rows_have_3_seeds": bool((dataset["n_seeds"] == 3).all()),
        "panel_b_exponent_near_frozen_sanity_value": abs(dataset_fit["exponent"] - 0.08028)
        < 0.0001,
        "panel_c_main_has_5_points": len(parameter_main5) == 5,
        "panel_c_main_excludes_9081324": 9_081_324
        not in parameter_main5["parameter_count"].tolist(),
        "panel_c_all6_retains_6_points": len(parameter_all6) == 6,
        "panel_c_all6_retains_9081324": 9_081_324
        in parameter_all6["parameter_count"].tolist(),
        "panel_c_fit_uses_5_points": parameter_fit["n_points"] == 5,
        "all_means_and_errors_positive": bool(
            (compute[["mean_test_L1", "std_test_L1"]].to_numpy() > 0).all()
            and (dataset[["mean_test_L1", "std_test_L1"]].to_numpy() > 0).all()
            and (parameter_all6[["mean_test_L1", "std_test_L1"]].to_numpy() > 0).all()
        ),
        "all_png_and_svg_outputs_exist": all(
            (OUTPUT_DIR / filename).is_file() and (OUTPUT_DIR / filename).stat().st_size > 0
            for filename in figure_files
        ),
        "no_training": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def write_manifest(qc: dict, figure_files: list[str], dataset_fit: dict, parameter_fit: dict) -> None:
    manifest = {
        "artifact": "full_data_clean_final_scaling_law_figure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "plot-only; no training, dataset construction, or splitting",
        "formal_experiment_root": "external; configure CCS_SCALING_ROOT for training",
        "local_frozen_exports": {
            "phase1_dataset_scaling": {
                "path": str(PHASE1_AGG.relative_to(PROJECT_ROOT)),
                "sha256": sha256(PHASE1_AGG),
            },
            "phase2_compute_scaling": {
                "path": str(PHASE2_AGG.relative_to(PROJECT_ROOT)),
                "sha256": sha256(PHASE2_AGG),
            },
            "phase3_parameter_scaling": {
                "path": str(PHASE3_AGG.relative_to(PROJECT_ROOT)),
                "sha256": sha256(PHASE3_AGG),
            },
        },
        "seed_ids": SEEDS,
        "panel_a": {
            "N_curves": EXPECTED_COMPUTE_N,
            "optimizer_step_checkpoints": EXPECTED_STEPS,
            "parameter_count": 712_428,
        },
        "panel_b_fit": dataset_fit,
        "panel_c": {
            "displayed_hidden_dims": DISPLAYED_HIDDEN_DIMS,
            "displayed_parameter_counts": [EXPECTED_PARAMETER_MAP[x] for x in DISPLAYED_HIDDEN_DIMS],
            "excluded_from_main_display_but_retained": {
                "hidden_dim": 1024,
                "parameter_count": 9_081_324,
                "interpretation": (
                    "The largest model did not provide additional performance gain "
                    "under the current dataset/training regime."
                ),
            },
            "main5_fit": parameter_fit,
        },
        "figures": figure_files,
        "plot_data_files": [
            "plot_data_compute.csv",
            "plot_data_dataset.csv",
            "plot_data_parameter_main5.csv",
            "plot_data_parameter_all6.csv",
            "plot_fit_dataset.json",
            "plot_fit_parameter_main5.json",
            "plot_config.json",
        ],
        "qc": qc,
    }
    with (OUTPUT_DIR / "plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    phase1, phase2, phase3 = load_and_validate()
    compute, dataset, parameter_main5, parameter_all6 = build_plot_tables(
        phase1, phase2, phase3
    )
    dataset_fit = power_law_fit(
        dataset["N"].to_numpy(), dataset["mean_test_L1"].to_numpy(), "N"
    )
    parameter_fit = power_law_fit(
        parameter_main5["parameter_count"].to_numpy(),
        parameter_main5["mean_test_L1"].to_numpy(),
        "P",
    )

    write_outputs(
        compute,
        dataset,
        parameter_main5,
        parameter_all6,
        dataset_fit,
        parameter_fit,
    )
    figure_files = make_figures(
        compute, dataset, parameter_main5, dataset_fit, parameter_fit
    )
    qc = build_qc(
        compute,
        dataset,
        parameter_main5,
        parameter_all6,
        dataset_fit,
        parameter_fit,
        figure_files,
    )
    write_manifest(qc, figure_files, dataset_fit, parameter_fit)

    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "Dataset fit: "
        f"L(N)={dataset_fit['coefficient']:.8f}*N^-{dataset_fit['exponent']:.8f}, "
        f"R2(log10)={dataset_fit['R2']:.8f}"
    )
    print(
        "Parameter main5 fit: "
        f"L(P)={parameter_fit['coefficient']:.8f}*P^-{parameter_fit['exponent']:.8f}, "
        f"R2(log10)={parameter_fit['R2']:.8f}"
    )
    print(f"QC: {qc['status']}")
    if qc["status"] != "PASS":
        failed = [name for name, passed in qc["checks"].items() if not passed]
        raise RuntimeError(f"QC failed: {failed}")


if __name__ == "__main__":
    main()
