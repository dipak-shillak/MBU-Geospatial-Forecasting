from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prophet import Prophet

from src.data_loader import load_data, validate_df

np.random.seed(42)


@dataclass
class ForecastResult:
    model: Prophet
    forecast: pd.DataFrame
    ts: pd.DataFrame

    
def prepare_district_series(
    df: pd.DataFrame,
    state: str,
    district: str,
    age_group: int | None = None,
    target: str = "mbu_pending"
) -> pd.DataFrame:
    """Prepare monthly time series (ds,y) for a district + optional age group."""
    df = df.copy()
    mask = (df["state"] == state) & (df["district"] == district)
    if age_group is not None:
        mask &= df["age_group"] == int(age_group)

    sub = df.loc[mask, :]
    if sub.empty:
        raise ValueError(f"No data for {state} / {district} / {age_group}")

    # Ensure 'date' column exists
    if "date" not in sub.columns:
        sub["date"] = pd.to_datetime(sub["year"].astype(str) + "-" + sub["month"].astype(str))

    ts = sub.groupby("date")[target].sum().reset_index()
    ts = ts.set_index("date").asfreq("MS").fillna(0).reset_index()
    ts = ts.rename(columns={"date": "ds", target: "y"})
    return ts


def fit_prophet(ts: pd.DataFrame, periods: int = 12) -> ForecastResult:
    """Fit Prophet and produce forecast for `periods` months."""
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(ts)

    future = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)
    return ForecastResult(model=m, forecast=forecast, ts=ts)


def calculate_risk_score(forecast: pd.DataFrame, ts: pd.DataFrame) -> float:
    """Simple risk score: future sum / historical sum * 100"""
    next12_sum = forecast["yhat"].tail(12).sum()
    hist_sum = ts["y"].sum()
    score = min(max((next12_sum / hist_sum) * 100, 0), 100)
    return score


def explain_and_save(result: ForecastResult, out_dir: str, fname_base: str):
    """Save forecast CSV, plots, and spike reasoning."""
    os.makedirs(out_dir, exist_ok=True)

    # Save forecast CSV
    forecast_csv = os.path.join(out_dir, f"{fname_base}_forecast.csv")
    result.forecast.to_csv(forecast_csv, index=False)

    # Time series + forecast plot
    fig1 = result.model.plot(result.forecast)
    fig1.suptitle(f"Forecast - {fname_base}")
    fig1_path = os.path.join(out_dir, f"{fname_base}_forecast.png")
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Components (trend/seasonality)
    fig2 = result.model.plot_components(result.forecast)
    fig2.suptitle(f"Components - {fname_base}")
    fig2_path = os.path.join(out_dir, f"{fname_base}_components.png")
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    # Spike detection
    hist = result.ts.copy()
    mean = hist["y"].mean()
    std = hist["y"].std(ddof=0)
    threshold = mean + 2 * std
    spikes = hist[hist["y"] > threshold]

    spike_lines = [f"mean={mean:.2f}, std={std:.2f}, threshold={threshold:.2f}"]
    spike_lines.append("Historical spikes (date, y):")
    for _, r in spikes.iterrows():
        spike_lines.append(f"{r['ds'].date()} -> {int(r['y'])}")

    # Calculate risk score
    risk = calculate_risk_score(result.forecast, result.ts)
    spike_lines.append(f"\nPredicted 12-month risk score: {risk:.2f}/100")

    spike_txt_path = os.path.join(out_dir, f"{fname_base}_spikes.txt")
    with open(spike_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(spike_lines))

    return {
        "forecast_csv": forecast_csv,
        "forecast_plot": fig1_path,
        "components_plot": fig2_path,
        "spike_summary": spike_txt_path,
        "risk_score": risk
    }


def run_for_district(
    input_csv: str,
    state: str,
    district: str,
    age_group: int | None = None,
    out_root: str = "outputs",
    periods: int = 12
):
    df = load_data(input_csv)
    df = validate_df(df)

    ts = prepare_district_series(df, state=state, district=district, age_group=age_group, target="mbu_pending")
    res = fit_prophet(ts, periods=periods)

    name_parts = [state.replace(" ", "_"), district.replace(" ", "_")]
    if age_group is not None:
        name_parts.append(str(age_group))
    fname_base = "__".join(name_parts)
    out_dir = os.path.join(out_root, "forecasts")
    outputs = explain_and_save(res, out_dir=out_dir, fname_base=fname_base)

    print(f"Saved forecast outputs for {state} / {district} to {out_dir}")
    print(f"Risk Score: {outputs['risk_score']:.2f}/100")
    return outputs


def _parse_args():
    p = argparse.ArgumentParser(description="District-level MBU forecasting using Prophet")
    p.add_argument("--input", default="dataset/mbu_synthetic_data.csv", help="Path to dataset CSV")
    p.add_argument("--state", required=True, help="State name")
    p.add_argument("--district", required=True, help="District name")
    p.add_argument("--age_group", type=int, required=False, help="Age group (5 or 15)")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--periods", type=int, default=12, help="Forecast horizon in months")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_for_district(args.input, args.state, args.district, age_group=args.age_group, out_root=args.out, periods=args.periods)