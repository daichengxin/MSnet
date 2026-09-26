#!/usr/bin/env python3
"""Rebuild the released CCS scaling-law figures from aggregate CSV files.

The helper modules preserve the exact evolution of the final plotting code.
This single entry point prepares the plotting tables, applies the final clean
style, and finally adds the continuous convergence-onset fit to Panel (a).
No model training is performed.
"""

from generate_full_data_clean_final import main as prepare_plot_inputs
from generate_full_data_clean_final_mae import main as render_clean_panels
from generate_full_data_clean_final_compute_fit import main as render_compute_fit


def main() -> None:
    prepare_plot_inputs()
    render_clean_panels()
    render_compute_fit()


if __name__ == "__main__":
    main()
