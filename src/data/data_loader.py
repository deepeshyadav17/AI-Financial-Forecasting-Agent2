"""
data_loader.py
--------------
Handles loading and initial preparation of stock price CSV data.

Expected CSV format:
    Date, Open, High, Low, Close, Volume
    (Date column will be parsed and set as the DataFrame index)
"""

import pandas as pd
import os


def load_stock_data(filepath: str, stock_symbol: str = None) -> pd.DataFrame:
    """
    Load stock data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing stock data.
    stock_symbol : str, optional
        If the CSV contains multiple stocks (a 'Symbol' column), filter by this symbol.

    Returns
    -------
    pd.DataFrame
        A clean DataFrame with Date as index and a 'Close' column at minimum.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[DataLoader] Loaded {len(df)} rows from '{filepath}'")

    # ── Normalise column names (strip whitespace, title-case) ─────────────────
    df.columns = [c.strip().title() for c in df.columns]

    # ── Parse Date column ─────────────────────────────────────────────────────
    date_col = None
    for col in df.columns:
        if col.lower() == "date":
            date_col = col
            break

    if date_col is None:
        raise ValueError("No 'Date' column found in the CSV file.")

    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    df.index.name = "Date"

    # ── Optional: filter by stock symbol ─────────────────────────────────────
    if stock_symbol and "Symbol" in df.columns:
        df = df[df["Symbol"].str.upper() == stock_symbol.upper()]
        if df.empty:
            raise ValueError(f"No data found for symbol '{stock_symbol}'.")
        print(f"[DataLoader] Filtered to symbol: {stock_symbol} — {len(df)} rows remain")

    # ── Keep only numeric columns + ensure 'Close' exists ────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "Close" not in numeric_cols:
        raise ValueError("CSV must contain a 'Close' price column.")

    df = df[numeric_cols]

    # ── Enforce business-day frequency (fill gaps with forward-fill) ──────────
    df = df.asfreq("B")          # reindex to every business day
    df = df.ffill()              # forward-fill any gaps
    df = df.dropna(subset=["Close"])

    print(f"[DataLoader] Final dataset: {len(df)} business-day rows | "
          f"Range: {df.index.min().date()} → {df.index.max().date()}")

    return df


def get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract the 'Close' price series from the loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_stock_data().

    Returns
    -------
    pd.Series
        Close price series with DatetimeIndex.
    """
    series = df["Close"].copy()
    series.index.freq = series.index.inferred_freq or "B"
    return series
