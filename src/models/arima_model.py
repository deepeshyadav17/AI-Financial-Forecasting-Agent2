"""
arima_model.py
--------------
ARIMA model: auto-tunes (p, d, q), trains on 80 % of data,
evaluates on the remaining 20 %, and saves results.

Uses pmdarima.auto_arima for automatic parameter selection.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in all envs
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")   # suppress convergence warnings during search


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_arima(
    train: pd.Series,
    test: pd.Series,
    output_dir: str = "outputs",
    max_p: int = 5,
    max_q: int = 5,
) -> dict:
    """
    Fit an ARIMA model, generate test predictions, and persist results.

    Parameters
    ----------
    train      : pd.Series   Training close prices (DatetimeIndex, freq='B')
    test       : pd.Series   Test close prices
    output_dir : str         Root output folder
    max_p, max_q : int       Search boundaries for auto_arima

    Returns
    -------
    dict  {rmse, order, predictions (list), test_dates (list)}
    """
    try:
        import pmdarima as pm
    except ImportError as e:
        raise ImportError("pmdarima is required. Run: pip install pmdarima") from e

    print("[ARIMA] Auto-searching best (p,d,q) parameters …")
    model = pm.auto_arima(
        train,
        start_p=1, start_q=1,
        max_p=max_p, max_q=max_q,
        d=None,                  # let the test determine d
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    order = model.order
    print(f"[ARIMA] Best order: {order}")

    # ── Forecast test-length steps ahead ─────────────────────────────────────
    n_test = len(test)
    forecast, conf_int = model.predict(n_periods=n_test, return_conf_int=True)
    forecast = np.array(forecast)

    # ── RMSE ──────────────────────────────────────────────────────────────────
    from sklearn.metrics import mean_squared_error
    rmse = float(np.sqrt(mean_squared_error(test.values, forecast)))
    print(f"[ARIMA] Test RMSE: {rmse:.4f}")

    # ── Persist plot ──────────────────────────────────────────────────────────
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, label="Train", color="#1f77b4", linewidth=1.2)
    ax.plot(test.index,  test.values,  label="Actual", color="#2ca02c", linewidth=1.2)
    ax.plot(test.index,  forecast,     label="ARIMA Forecast",
            color="#d62728", linewidth=1.2, linestyle="--")
    ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1],
                    alpha=0.15, color="#d62728", label="95 % CI")
    ax.set_title(f"ARIMA {order} – Train vs Actual vs Forecast  (RMSE={rmse:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "arima_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[ARIMA] Plot saved → {plot_path}")

    # ── Persist JSON results ──────────────────────────────────────────────────
    results = {
        "model": "ARIMA",
        "order": list(order),
        "rmse": rmse,
        "predictions": forecast.tolist(),
        "test_dates": [str(d.date()) for d in test.index],
    }
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    result_path = os.path.join(pred_dir, "arima_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ARIMA] Results saved → {result_path}")

    # keep the fitted model on the results dict so predict.py can reuse it
    results["_model"] = model
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Future Forecast
# ──────────────────────────────────────────────────────────────────────────────

def forecast_future_arima(model, last_date: pd.Timestamp, n_days: int = 30) -> pd.Series:
    """
    Extend the fitted ARIMA model to predict n_days beyond last_date.

    Parameters
    ----------
    model     : fitted pmdarima ARIMA object  (stored in results['_model'])
    last_date : the last date in the historical data
    n_days    : number of future business days to forecast

    Returns
    -------
    pd.Series  with DatetimeIndex (business-day freq)
    """
    future_preds, _ = model.predict(n_periods=n_days, return_conf_int=True)
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=n_days)
    return pd.Series(future_preds, index=future_dates, name="ARIMA_Future")
