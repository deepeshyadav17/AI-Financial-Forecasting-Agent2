"""
preprocess.py
-------------
Feature engineering and data preprocessing utilities shared by both models.

Responsibilities:
  • Train / test split
  • MinMax normalisation (for LSTM)
  • Sequence creation (for LSTM)
  • Stationarity check (for ARIMA)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split_ts(series: pd.Series, train_ratio: float = 0.80):
    """
    Split a time-series into train and test portions.

    Parameters
    ----------
    series : pd.Series
        The full Close-price series with DatetimeIndex.
    train_ratio : float
        Fraction of data to use for training (default 80 %).

    Returns
    -------
    train : pd.Series
    test  : pd.Series
    split_idx : int   – the integer position where the split occurs
    """
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test  = series.iloc[split_idx:]
    print(f"[Preprocess] Train size: {len(train)} | Test size: {len(test)}")
    return train, test, split_idx


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Stationarity Check  (used by ARIMA auto-tuning)
# ──────────────────────────────────────────────────────────────────────────────

def check_stationarity(series: pd.Series, significance: float = 0.05) -> bool:
    """
    Augmented Dickey-Fuller test.

    Returns True if the series is stationary at the given significance level.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    is_stationary = p_value < significance
    print(f"[Preprocess] ADF p-value: {p_value:.4f} → "
          f"{'Stationary ✓' if is_stationary else 'Non-stationary ✗'}")
    return is_stationary


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MinMax Normalisation  (LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def normalise_series(series: pd.Series):
    """
    Scale series values to [0, 1] using MinMaxScaler.

    Returns
    -------
    scaled_values : np.ndarray  shape (n, 1)
    scaler        : MinMaxScaler  (keep this to inverse-transform predictions)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def inverse_transform(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """
    Reverse the MinMax scaling back to original price space.

    Parameters
    ----------
    values : np.ndarray  shape (n,) or (n, 1)
    """
    values = values.reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Sequence Creation  (LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(scaled_data: np.ndarray, look_back: int = 60):
    """
    Build (X, y) pairs for LSTM training.

    Each sample X[i] is a window of `look_back` consecutive scaled prices;
    y[i] is the price immediately following that window.

    Parameters
    ----------
    scaled_data : np.ndarray  shape (n, 1)
    look_back   : int          number of past days used as input features

    Returns
    -------
    X : np.ndarray  shape (samples, look_back, 1)
    y : np.ndarray  shape (samples,)
    """
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back: i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, look_back, 1)   # LSTM expects 3-D input
    y = np.array(y)
    print(f"[Preprocess] Sequences created — X: {X.shape} | y: {y.shape}")
    return X, y
