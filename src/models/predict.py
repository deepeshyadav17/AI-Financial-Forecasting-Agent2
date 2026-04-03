"""
predict.py
----------
Generates 30-day future forecasts using both ARIMA and LSTM models,
plots them side-by-side against historical data, and saves results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predict_future(
    series: pd.Series,
    arima_results: dict,
    lstm_results: dict,
    n_days: int = 30,
    output_dir: str = "outputs",
) -> dict:
    """
    Run future forecasts for both models and produce a combined plot.

    Parameters
    ----------
    series        : full historical Close price series
    arima_results : dict returned by train_arima()  (must contain '_model')
    lstm_results  : dict returned by train_lstm()   (must contain '_model', '_scaler', '_scaled_data')
    n_days        : forecast horizon (default 30 business days)
    output_dir    : root output folder

    Returns
    -------
    dict  {arima_future (list), lstm_future (list), future_dates (list)}
    """
    from src.models.arima_model import forecast_future_arima
    from src.models.lstm_model  import forecast_future_lstm

    last_date = series.index[-1]

    # ── ARIMA future ──────────────────────────────────────────────────────────
    print(f"[Predict] Forecasting {n_days} days with ARIMA …")
    arima_future = forecast_future_arima(arima_results["_model"], last_date, n_days)

    # ── LSTM future ───────────────────────────────────────────────────────────
    print(f"[Predict] Forecasting {n_days} days with LSTM …")
    lstm_future = forecast_future_lstm(
        model       = lstm_results["_model"],
        scaler      = lstm_results["_scaler"],
        scaled_data = lstm_results["_scaled_data"],
        last_date   = last_date,
        n_days      = n_days,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Show last 90 days of history + 30 days forecast
    history_window = series.iloc[-90:]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history_window.index, history_window.values,
            label="Historical", color="#1f77b4", linewidth=1.5)
    ax.plot(arima_future.index, arima_future.values,
            label="ARIMA Forecast", color="#d62728",
            linewidth=1.5, linestyle="--", marker="o", markersize=3)
    ax.plot(lstm_future.index, lstm_future.values,
            label="LSTM Forecast",  color="#ff7f0e",
            linewidth=1.5, linestyle="--", marker="s", markersize=3)
    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1.2,
               label="Forecast Start")
    ax.set_title(f"30-Day Future Price Forecast – ARIMA vs LSTM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "future_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[Predict] Future forecast plot saved → {plot_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    future_results = {
        "future_dates": [str(d.date()) for d in arima_future.index],
        "arima_future": arima_future.values.tolist(),
        "lstm_future":  lstm_future.values.tolist(),
    }
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    result_path = os.path.join(pred_dir, "future_predictions.json")
    with open(result_path, "w") as f:
        json.dump(future_results, f, indent=2)
    print(f"[Predict] Future predictions saved → {result_path}")

    return future_results
