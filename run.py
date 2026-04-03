"""
run.py
------
End-to-end pipeline runner.

Usage
-----
    # With a real CSV:
    python run.py --csv data/raw/INFY.csv --symbol INFY

    # With auto-generated demo data:
    python run.py --demo

    # Extra options:
    python run.py --demo --epochs 30 --forecast 30
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Make project root importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_csv(path: str) -> None:
    """Write a synthetic 3-year OHLCV CSV to `path`."""
    dates = pd.bdate_range("2021-01-04", periods=756)
    np.random.seed(42)
    price = 1500.0
    closes = []
    for _ in dates:
        price *= np.exp(np.random.normal(0.0003, 0.015))
        closes.append(round(price, 2))

    pd.DataFrame({
        "Date":   dates.strftime("%Y-%m-%d"),
        "Open":   [round(c * np.random.uniform(0.99, 1.01), 2) for c in closes],
        "High":   [round(c * np.random.uniform(1.00, 1.02), 2) for c in closes],
        "Low":    [round(c * np.random.uniform(0.98, 1.00), 2) for c in closes],
        "Close":  closes,
        "Volume": np.random.randint(500_000, 5_000_000, size=len(dates)),
    }).to_csv(path, index=False)
    print(f"[run.py] Demo CSV written → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Financial Forecasting Pipeline")
    parser.add_argument("--csv",       type=str, default=None,
                        help="Path to stock CSV file.")
    parser.add_argument("--symbol",    type=str, default=None,
                        help="Stock symbol to filter (optional).")
    parser.add_argument("--demo",      action="store_true",
                        help="Generate and use synthetic demo data.")
    parser.add_argument("--train_ratio", type=float, default=0.80,
                        help="Train/test split ratio (default 0.80).")
    parser.add_argument("--epochs",    type=int, default=20,
                        help="LSTM training epochs (default 20).")
    parser.add_argument("--forecast",  type=int, default=30,
                        help="Number of future days to forecast (default 30).")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Root directory for all outputs.")
    args = parser.parse_args()

    # ── Resolve CSV path ──────────────────────────────────────────────────────
    if args.demo:
        os.makedirs("data/raw", exist_ok=True)
        csv_path = "data/raw/DEMO_STOCK.csv"
        _make_demo_csv(csv_path)
    elif args.csv:
        csv_path = args.csv
    else:
        print("[run.py] ERROR: Provide --csv <path> or use --demo flag.")
        sys.exit(1)

    # Ensure output structure
    from src.utils.helpers import ensure_output_dirs
    ensure_output_dirs(args.output_dir)

    # ── STEP 1 – Load data ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 1 — Loading Data")
    print("─" * 60)
    from src.data.data_loader import load_stock_data, get_close_series
    df     = load_stock_data(csv_path, stock_symbol=args.symbol)
    series = get_close_series(df)

    # ── STEP 2 – Train/test split ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 2 — Train / Test Split")
    print("─" * 60)
    from src.features.preprocess import train_test_split_ts
    train, test, split_idx = train_test_split_ts(series, train_ratio=args.train_ratio)

    # ── STEP 3 – ARIMA ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 3 — ARIMA Model")
    print("─" * 60)
    from src.models.arima_model import train_arima
    arima_results = train_arima(train, test, output_dir=args.output_dir)

    # ── STEP 4 – LSTM ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 4 — LSTM Model")
    print("─" * 60)
    from src.models.lstm_model import train_lstm
    lstm_results = train_lstm(
        series, split_idx,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )

    # ── STEP 5 – Model comparison ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 5 — Model Comparison")
    print("─" * 60)
    from src.models.evaluate import compare_models
    comparison = compare_models(output_dir=args.output_dir)

    # ── STEP 6 – Future forecast ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f" STEP 6 — {args.forecast}-Day Future Forecast")
    print("─" * 60)
    from src.models.predict import predict_future
    future_results = predict_future(
        series, arima_results, lstm_results,
        n_days=args.forecast,
        output_dir=args.output_dir,
    )

    # ── STEP 7 – Trading Agent ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 7 — AI Trading Agent")
    print("─" * 60)
    from src.agent.trading_agent import run_agent
    recommendation = run_agent(
        series, future_results, comparison,
        output_dir=args.output_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(" ✅  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  All outputs saved in:  {os.path.abspath(args.output_dir)}/")
    print(f"  Best Model   : {comparison['best_model']}")
    print(f"  Signal       : {recommendation['signal']}")
    print(f"  Plots        : {args.output_dir}/plots/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
