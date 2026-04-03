"""
lstm_model.py
-------------
LSTM model built with TensorFlow / Keras.

Architecture:
  Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)

Workflow:
  1. Normalise full series with MinMaxScaler
  2. Create overlapping windows (look_back days → next day)
  3. Train on windows that fall in the training period
  4. Predict on the test period
  5. Save model weights and results JSON
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

LOOK_BACK = 60      # how many past days feed into each prediction


# ──────────────────────────────────────────────────────────────────────────────
# Helper – build Keras model
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(look_back: int) -> "tf.keras.Model":
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 1)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm(
    series: pd.Series,
    train_size: int,
    output_dir: str = "outputs",
    epochs: int = 30,
    batch_size: int = 32,
) -> dict:
    """
    Train LSTM on the training portion of `series`, evaluate on the test portion.

    Parameters
    ----------
    series     : full Close price series (DatetimeIndex)
    train_size : number of rows belonging to the training period
    output_dir : root output folder
    epochs     : training epochs
    batch_size : mini-batch size

    Returns
    -------
    dict  {rmse, predictions (list), test_dates (list), _scaler, _model, _scaled_data}
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except ImportError as e:
        raise ImportError("TensorFlow is required. Run: pip install tensorflow") from e

    from src.features.preprocess import normalise_series, inverse_transform, create_sequences

    # ── Scale the ENTIRE series (scaler fitted on train, applied to all) ──────
    train_series = series.iloc[:train_size]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.values.reshape(-1, 1))
    scaled_all = scaler.transform(series.values.reshape(-1, 1))

    # ── Build sequences from training data only ───────────────────────────────
    X_train, y_train = _make_sequences(scaled_all[:train_size], LOOK_BACK)

    print(f"[LSTM] Training on {len(X_train)} sequences | epochs={epochs}")
    model = _build_model(LOOK_BACK)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    print(f"[LSTM] Training complete. Final val_loss: "
          f"{history.history['val_loss'][-1]:.6f}")

    # ── Predict on test section ───────────────────────────────────────────────
    test_series = series.iloc[train_size:]
    n_test = len(test_series)

    # Build test sequences: each test prediction needs look_back days before it
    test_inputs = scaled_all[train_size - LOOK_BACK: train_size + n_test]
    X_test, _ = _make_sequences(test_inputs, LOOK_BACK)

    preds_scaled = model.predict(X_test, verbose=0)
    predictions  = scaler.inverse_transform(preds_scaled).flatten()

    # ── RMSE ──────────────────────────────────────────────────────────────────
    from sklearn.metrics import mean_squared_error
    actuals = test_series.values[:len(predictions)]
    rmse    = float(np.sqrt(mean_squared_error(actuals, predictions)))
    print(f"[LSTM] Test RMSE: {rmse:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    test_dates = test_series.index[:len(predictions)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index[:train_size], series.values[:train_size],
            label="Train", color="#1f77b4", linewidth=1.2)
    ax.plot(test_dates, actuals,     label="Actual",          color="#2ca02c", linewidth=1.2)
    ax.plot(test_dates, predictions, label="LSTM Prediction", color="#ff7f0e",
            linewidth=1.2, linestyle="--")
    ax.set_title(f"LSTM – Train vs Actual vs Predicted  (RMSE={rmse:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "lstm_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[LSTM] Plot saved → {plot_path}")

    # ── Save model weights ────────────────────────────────────────────────────
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lstm_model.keras")
    model.save(model_path)
    print(f"[LSTM] Model saved → {model_path}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model": "LSTM",
        "look_back": LOOK_BACK,
        "rmse": rmse,
        "predictions": predictions.tolist(),
        "test_dates": [str(d.date()) for d in test_dates],
    }
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    result_path = os.path.join(pred_dir, "lstm_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[LSTM] Results saved → {result_path}")

    results["_model"]       = model
    results["_scaler"]      = scaler
    results["_scaled_data"] = scaled_all
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Future Forecast
# ──────────────────────────────────────────────────────────────────────────────

def forecast_future_lstm(
    model,
    scaler,
    scaled_data: np.ndarray,
    last_date: pd.Timestamp,
    n_days: int = 30,
) -> pd.Series:
    """
    Iteratively predict n_days into the future using the trained LSTM.

    Strategy: feed the model its own prediction as the next input (autoregressive).
    """
    last_window = scaled_data[-LOOK_BACK:].flatten().tolist()
    future_preds = []

    for _ in range(n_days):
        x = np.array(last_window[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        future_preds.append(pred)
        last_window.append(pred)

    # Inverse-transform back to price space
    future_prices = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=n_days)
    return pd.Series(future_prices, index=future_dates, name="LSTM_Future")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _make_sequences(data: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)
