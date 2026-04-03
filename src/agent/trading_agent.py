"""
trading_agent.py
----------------
A simple rule-based AI Trading Agent.

Logic
-----
  • Uses the BEST model (lowest RMSE) to obtain the 30-day forecast.
  • Compares the forecast mean against the current (last known) price.
  • Issues a BUY signal if the average forecasted price is higher
    than the current price; otherwise issues a SELL signal.

Returns a structured recommendation dict with supporting statistics.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd


def run_agent(
    series: pd.Series,
    future_results: dict,
    comparison: dict,
    output_dir: str = "outputs",
) -> dict:
    """
    Analyse future forecasts and issue a trading recommendation.

    Parameters
    ----------
    series         : full historical Close price series
    future_results : dict returned by predict_future()
    comparison     : dict returned by compare_models()
    output_dir     : root output folder

    Returns
    -------
    dict
        {
          current_price      : float,
          predicted_avg      : float,
          predicted_30d_end  : float,
          signal             : "BUY" | "SELL",
          confidence_pct     : float,   # how far (%) predicted_avg is from current
          best_model         : str,
          forecast_dates     : list[str],
          forecast_prices    : list[float],
        }
    """
    best_model = comparison["best_model"]

    # Pick forecast series from the winning model
    if best_model == "ARIMA":
        forecast_prices = future_results["arima_future"]
    else:
        forecast_prices = future_results["lstm_future"]

    forecast_dates  = future_results["future_dates"]
    current_price   = float(series.iloc[-1])
    predicted_avg   = float(np.mean(forecast_prices))
    predicted_30d   = float(forecast_prices[-1])        # price at end of 30 days

    # Signal logic
    signal = "BUY" if predicted_avg > current_price else "SELL"

    # Confidence = percentage difference between avg forecast and current price
    confidence_pct = round(abs(predicted_avg - current_price) / current_price * 100, 2)

    recommendation = {
        "current_price":     round(current_price, 2),
        "predicted_avg":     round(predicted_avg, 2),
        "predicted_30d_end": round(predicted_30d, 2),
        "signal":            signal,
        "confidence_pct":    confidence_pct,
        "best_model":        best_model,
        "forecast_dates":    forecast_dates,
        "forecast_prices":   [round(p, 2) for p in forecast_prices],
    }

    # ── Pretty print ──────────────────────────────────────────────────────────
    emoji = "📈 BUY" if signal == "BUY" else "📉 SELL"
    print("\n" + "=" * 55)
    print("          AI TRADING AGENT RECOMMENDATION")
    print("=" * 55)
    print(f"  Best Model         : {best_model}")
    print(f"  Current Price      : {current_price:.2f}")
    print(f"  Avg Forecast (30d) : {predicted_avg:.2f}")
    print(f"  Price at 30th Day  : {predicted_30d:.2f}")
    print(f"  Confidence         : {confidence_pct:.2f} %")
    print(f"  ➡  Signal          : {emoji}")
    print("=" * 55 + "\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, "agent_recommendation.json")
    with open(out_path, "w") as f:
        json.dump(recommendation, f, indent=2)
    print(f"[Agent] Recommendation saved → {out_path}")

    return recommendation
