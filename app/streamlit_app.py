"""
streamlit_app.py
----------------
AI Agent for Financial Forecasting — Streamlit User Interface

Run with:
    streamlit run app/streamlit_app.py

Features
--------
  • Upload a stock CSV or use the bundled demo data
  • Select a stock symbol (if CSV contains a Symbol column)
  • Configure LSTM epochs
  • View ARIMA & LSTM forecasts side-by-side
  • Model comparison (RMSE table)
  • 30-day future forecast chart
  • AI Trading Agent Buy/Sell signal
"""

import os
import sys
import json
import tempfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Make project root importable regardless of CWD ───────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Forecasting Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.5px;
}
.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: #f1f5f9;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #38bdf8;
}
.signal-buy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #d1fae5;
}
.signal-sell {
    background: linear-gradient(135deg, #4c0519, #7f1d1d);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #fee2e2;
}
.signal-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 4px;
}
.info-box {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_csv(tmp_dir: str) -> str:
    """Generate a synthetic 3-year daily stock CSV for demo purposes."""
    dates = pd.bdate_range("2021-01-04", periods=756)
    np.random.seed(42)
    price = 1500.0
    closes = []
    for _ in dates:
        price *= np.exp(np.random.normal(0.0003, 0.015))
        closes.append(round(price, 2))

    df = pd.DataFrame({
        "Date":   dates.strftime("%Y-%m-%d"),
        "Open":   [round(c * np.random.uniform(0.99, 1.01), 2) for c in closes],
        "High":   [round(c * np.random.uniform(1.00, 1.02), 2) for c in closes],
        "Low":    [round(c * np.random.uniform(0.98, 1.00), 2) for c in closes],
        "Close":  closes,
        "Volume": np.random.randint(500_000, 5_000_000, size=len(dates)),
    })
    path = os.path.join(tmp_dir, "DEMO_STOCK.csv")
    df.to_csv(path, index=False)
    return path


def _figure_to_streamlit(fig):
    """Render a matplotlib Figure in Streamlit and close it."""
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load(filepath: str, symbol: str):
    from src.data.data_loader import load_stock_data, get_close_series
    df = load_stock_data(filepath, stock_symbol=symbol if symbol else None)
    return df, get_close_series(df)


@st.cache_data(show_spinner=False)
def cached_arima(_series_key, train_vals, train_idx, test_vals, test_idx):
    """Wrapper that accepts hashable primitives for caching."""
    train = pd.Series(train_vals, index=pd.DatetimeIndex(train_idx))
    test  = pd.Series(test_vals,  index=pd.DatetimeIndex(test_idx))
    train.index.freq = train.index.inferred_freq or "B"
    test.index.freq  = test.index.inferred_freq  or "B"
    with tempfile.TemporaryDirectory() as tmp:
        from src.models.arima_model import train_arima
        results = train_arima(train, test, output_dir=tmp)
        # Serialise only what we need (model object can't be cached)
        return {
            "model":       "ARIMA",
            "order":       results["order"],
            "rmse":        results["rmse"],
            "predictions": results["predictions"],
            "test_dates":  results["test_dates"],
            "_model_obj":  results["_model"],   # pmdarima object — picklable
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 📈  AI Agent for Financial Forecasting")
    st.markdown(
        "<div class='info-box'>Upload your stock CSV → configure → click <b>Run Forecast</b>. "
        "The agent will train ARIMA + LSTM, compare them, and issue a trading signal.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️  Configuration")

        uploaded = st.file_uploader(
            "Upload Stock CSV",
            type=["csv"],
            help="Must contain Date and Close columns.",
        )
        use_demo = st.checkbox("Use demo data (no upload needed)", value=True)

        stock_symbol = st.text_input(
            "Stock Symbol (optional)",
            value="",
            help="Fill only if your CSV has a 'Symbol' column.",
        ).strip().upper()

        train_ratio = st.slider(
            "Train/Test split",
            min_value=0.60, max_value=0.90, value=0.80, step=0.05,
            help="Fraction of data used for training.",
        )
        lstm_epochs = st.slider(
            "LSTM epochs",
            min_value=5, max_value=100, value=20, step=5,
        )
        forecast_days = st.slider(
            "Future forecast days",
            min_value=7, max_value=60, value=30, step=1,
        )

        run_btn = st.button("🚀  Run Forecast", use_container_width=True, type="primary")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not run_btn:
        st.info("👈  Configure settings in the sidebar and click **Run Forecast** to begin.")
        return

    with st.spinner("Loading data …"):
        try:
            tmp_dir = tempfile.mkdtemp()

            if uploaded:
                csv_path = os.path.join(tmp_dir, uploaded.name)
                with open(csv_path, "wb") as f:
                    f.write(uploaded.read())
            elif use_demo:
                csv_path = _make_demo_csv(tmp_dir)
                st.caption("ℹ️  Using synthetic demo data (3 years of simulated prices).")
            else:
                st.error("Please upload a CSV file or enable demo mode.")
                return

            df, series = cached_load(csv_path, stock_symbol)

        except Exception as e:
            st.error(f"Data loading failed: {e}")
            return

    # ── Data overview ─────────────────────────────────────────────────────────
    with st.expander("📋  Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows",   len(series))
        col2.metric("Start Date",   str(series.index[0].date()))
        col3.metric("End Date",     str(series.index[-1].date()))
        col4.metric("Current Price", f"{series.iloc[-1]:.2f}")
        st.line_chart(series.rename("Close Price"))

    # ── Train/Test split ──────────────────────────────────────────────────────
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test  = series.iloc[split_idx:]

    out_dir = os.path.join(tmp_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    st.markdown("### 📊  Model Training")
    arima_col, lstm_col = st.columns(2)

    with arima_col:
        with st.spinner("Training ARIMA model …"):
            try:
                from src.models.arima_model import train_arima, forecast_future_arima
                arima_results = train_arima(train, test, output_dir=out_dir)
                st.success(f"✅  ARIMA trained — RMSE: **{arima_results['rmse']:.2f}**")
            except Exception as e:
                st.error(f"ARIMA failed: {e}")
                return

    # ── LSTM ──────────────────────────────────────────────────────────────────
    with lstm_col:
        with st.spinner(f"Training LSTM ({lstm_epochs} epochs) …"):
            try:
                from src.models.lstm_model import train_lstm, forecast_future_lstm
                lstm_results = train_lstm(
                    series, split_idx, output_dir=out_dir,
                    epochs=lstm_epochs,
                )
                st.success(f"✅  LSTM trained — RMSE: **{lstm_results['rmse']:.2f}**")
            except Exception as e:
                st.error(f"LSTM failed: {e}")
                return

    # ── RMSE Comparison ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏆  Model Comparison")

    arima_rmse = arima_results["rmse"]
    lstm_rmse  = lstm_results["rmse"]
    best_model = "ARIMA" if arima_rmse <= lstm_rmse else "LSTM"
    best_rmse  = min(arima_rmse, lstm_rmse)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>ARIMA RMSE</div>"
            f"<div class='value'>{arima_rmse:.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>LSTM RMSE</div>"
            f"<div class='value'>{lstm_rmse:.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Best Model</div>"
            f"<div class='value' style='color:#4ade80'>{best_model}</div></div>",
            unsafe_allow_html=True,
        )

    # ── Test-period forecast charts ───────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍  Test-Period Predictions")

    tab_arima, tab_lstm = st.tabs(["ARIMA", "LSTM"])

    with tab_arima:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index[-60:], train.values[-60:],
                label="Train (last 60d)", color="#1f77b4", linewidth=1.2)
        ax.plot(test.index, test.values,
                label="Actual", color="#2ca02c", linewidth=1.2)
        ax.plot(
            pd.to_datetime(arima_results["test_dates"]),
            arima_results["predictions"],
            label=f"ARIMA (RMSE={arima_rmse:.2f})",
            color="#d62728", linewidth=1.2, linestyle="--",
        )
        ax.set_title("ARIMA – Test Period Forecast")
        ax.legend(); ax.grid(alpha=0.25)
        plt.tight_layout()
        _figure_to_streamlit(fig)

    with tab_lstm:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index[-60:], train.values[-60:],
                label="Train (last 60d)", color="#1f77b4", linewidth=1.2)
        ax.plot(test.index, test.values,
                label="Actual", color="#2ca02c", linewidth=1.2)
        n_pred = len(lstm_results["predictions"])
        ax.plot(
            test.index[:n_pred],
            lstm_results["predictions"],
            label=f"LSTM (RMSE={lstm_rmse:.2f})",
            color="#ff7f0e", linewidth=1.2, linestyle="--",
        )
        ax.set_title("LSTM – Test Period Forecast")
        ax.legend(); ax.grid(alpha=0.25)
        plt.tight_layout()
        _figure_to_streamlit(fig)

    # ── Future Forecast ───────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 🔮  {forecast_days}-Day Future Forecast")

    with st.spinner("Generating future forecasts …"):
        last_date = series.index[-1]

        arima_future = forecast_future_arima(
            arima_results["_model"], last_date, forecast_days
        )
        lstm_future = forecast_future_lstm(
            model       = lstm_results["_model"],
            scaler      = lstm_results["_scaler"],
            scaled_data = lstm_results["_scaled_data"],
            last_date   = last_date,
            n_days      = forecast_days,
        )

    fig, ax = plt.subplots(figsize=(14, 5))
    history = series.iloc[-90:]
    ax.fill_between(history.index, history.values, alpha=0.07, color="#1f77b4")
    ax.plot(history.index, history.values,
            label="Historical (last 90d)", color="#1f77b4", linewidth=1.8)
    ax.plot(arima_future.index, arima_future.values,
            label=f"ARIMA Forecast ({forecast_days}d)",
            color="#d62728", linewidth=1.8, linestyle="--", marker="o", markersize=3)
    ax.plot(lstm_future.index, lstm_future.values,
            label=f"LSTM Forecast ({forecast_days}d)",
            color="#ff7f0e", linewidth=1.8, linestyle="--", marker="s", markersize=3)
    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1.2)
    ax.set_title(f"{forecast_days}-Day Future Price Forecast")
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    _figure_to_streamlit(fig)

    # Table of future values
    with st.expander("📄  View forecast numbers"):
        future_df = pd.DataFrame({
            "Date":  arima_future.index.strftime("%Y-%m-%d"),
            "ARIMA": arima_future.values.round(2),
            "LSTM":  lstm_future.values.round(2),
        })
        st.dataframe(future_df, use_container_width=True, hide_index=True)

    # ── Trading Agent ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖  AI Trading Agent")

    best_forecast = arima_future if best_model == "ARIMA" else lstm_future
    current_price = float(series.iloc[-1])
    predicted_avg = float(best_forecast.mean())
    predicted_end = float(best_forecast.iloc[-1])
    signal        = "BUY" if predicted_avg > current_price else "SELL"
    confidence    = round(abs(predicted_avg - current_price) / current_price * 100, 2)

    if signal == "BUY":
        st.markdown(
            f"<div class='signal-buy'>"
            f"<div class='signal-text'>📈  BUY</div>"
            f"<p style='margin-top:0.8rem; font-size:1rem;'>"
            f"The <b>{best_model}</b> model predicts the average price over the next "
            f"{forecast_days} days ({predicted_avg:.2f}) will be "
            f"<b>{confidence:.1f}%</b> above the current price ({current_price:.2f}).</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='signal-sell'>"
            f"<div class='signal-text'>📉  SELL</div>"
            f"<p style='margin-top:0.8rem; font-size:1rem;'>"
            f"The <b>{best_model}</b> model predicts the average price over the next "
            f"{forecast_days} days ({predicted_avg:.2f}) will be "
            f"<b>{confidence:.1f}%</b> below the current price ({current_price:.2f}).</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price",       f"₹ {current_price:.2f}")
    m2.metric("Avg Forecast",        f"₹ {predicted_avg:.2f}",
              delta=f"{predicted_avg - current_price:+.2f}")
    m3.metric(f"Price at Day {forecast_days}", f"₹ {predicted_end:.2f}",
              delta=f"{predicted_end - current_price:+.2f}")
    m4.metric("Signal Confidence",   f"{confidence:.1f} %")

    st.markdown(
        "<div class='info-box'>⚠️  <b>Disclaimer:</b> This is a demonstration project. "
        "Predictions are generated by statistical/ML models and should <b>not</b> be "
        "used as financial advice. Always consult a qualified financial advisor.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
