"""Trivariate gap analysis (State + District + Age Group).

Produces:
- outputs/gap_summary.csv : aggregated metrics per state,district,age_group
- outputs/gap_top_districts.png : visualization of top gap rates

This script is deterministic, modular, and intended for review by judges.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_data, validate_df, aggregate_trivariate


def run_gap_analysis(input_csv: str, out_dir: str = "outputs", top_n: int = 20):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(input_csv)
    df = validate_df(df)

    summary = aggregate_trivariate(df)
    summary = summary.sort_values("gap_rate", ascending=False)

    out_csv = os.path.join(out_dir, "gap_summary.csv")
    summary.to_csv(out_csv, index=False)

    # Top districts by gap rate for visualization
    top = summary.head(top_n).copy()

    plt.figure(figsize=(10, max(4, top_n * 0.25)))
    sns.barplot(data=top, x="gap_rate", y="district", hue="age_group", dodge=False)
    plt.title(f"Top {top_n} district-age groups by MBU gap rate")
    plt.xlabel("Gap rate (mbu_pending / eligible_for_mbu)")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "gap_top_districts.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    return out_csv, out_png


def _parse_args():
    p = argparse.ArgumentParser(description="Trivariate MBU gap analysis")
    p.add_argument("--input", default="dataset/mbu_synthetic_data.csv", help="Path to input CSV")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--top", type=int, default=20, help="Top N rows to plot")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path, png_path = run_gap_analysis(args.input, args.out, args.top)
    print(f"Wrote summary to: {csv_path}")
    print(f"Wrote visualization to: {png_path}")