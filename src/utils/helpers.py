"""
helpers.py
----------
General utility functions used across the project.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Ensure all output sub-directories exist ────────────────────────────────────

def ensure_output_dirs(base: str = "outputs") -> None:
    """Create outputs/models, outputs/predictions, outputs/plots if absent."""
    for sub in ("models", "predictions", "plots"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)


# ── Load any JSON result file ─────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load and return a JSON file as a Python dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Quick summary stats ────────────────────────────────────────────────────────

def describe_series(series: pd.Series) -> dict:
    """Return basic statistics for a price series."""
    return {
        "count": int(len(series)),
        "min":   round(float(series.min()), 2),
        "max":   round(float(series.max()), 2),
        "mean":  round(float(series.mean()), 2),
        "std":   round(float(series.std()), 2),
        "first": str(series.index[0].date()),
        "last":  str(series.index[-1].date()),
    }


# ── Combine historical + future into one clean plot ───────────────────────────

def plot_combined(
    series: pd.Series,
    arima_future: pd.Series,
    lstm_future:  pd.Series,
    save_path: str = "outputs/plots/combined_forecast.png",
    history_days: int = 120,
) -> None:
    """
    Plot the last `history_days` of actual prices plus both model forecasts.
    Saves to `save_path`.
    """
    history = series.iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.fill_between(history.index, history.values,
                    alpha=0.08, color="#1f77b4")
    ax.plot(history.index, history.values,
            label="Historical", color="#1f77b4", linewidth=2)
    ax.plot(arima_future.index, arima_future.values,
            label="ARIMA Forecast", color="#d62728",
            linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.plot(lstm_future.index, lstm_future.values,
            label="LSTM Forecast",  color="#ff7f0e",
            linewidth=2, linestyle="--", marker="s", markersize=4)

    ax.axvline(series.index[-1], color="gray", linestyle=":",
               linewidth=1.5, label="Forecast Start")
    ax.set_title("Combined Forecast – Historical vs ARIMA vs LSTM", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Helpers] Combined plot saved → {save_path}")


# ── RMSE helper ───────────────────────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


# ── Format signal for display ─────────────────────────────────────────────────

def format_signal(signal: str) -> str:
    """Return a coloured emoji-prefixed signal string for CLI display."""
    if signal.upper() == "BUY":
        return "📈  BUY  – Predicted price is HIGHER than current price."
    return "📉  SELL – Predicted price is LOWER than current price."
