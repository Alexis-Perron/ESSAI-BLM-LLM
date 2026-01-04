#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baselines (Equal-Weight + Mean-Variance) computed ONLY from filtered_sp500_data.csv.

This script:
- Loads yfinance/filtered_sp500_data.csv
- Detects whether the dataset is LONG (columns: date, tic, stock_ret) or WIDE (date + many ticker columns)
- Builds a return matrix R[t, i] (index=date, columns=tickers) in decimal returns
- For each monthly step:
    train = lookback window ending at train_end (default: 12 months lookback)
    test  = next calendar month
  Computes:
    - equal-weight portfolio return over test window
    - mean-variance (long-only) portfolio weights from train window and returns over test window
- Saves:
    results/baselines_results.csv
    results/weights_<YYYY-MM>.csv  (weights for the MV portfolio per train month)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

try:
    from scipy.optimize import minimize
except Exception as e:
    raise ImportError("scipy is required for mean-variance optimization. Install with: pip install scipy") from e


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("yfinance/filtered_sp500_data.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Months you want to run (inclusive). If None, uses full range from data.
RUN_START = "2024-06-01"
RUN_END   = "2024-12-31"

# Mean-variance settings
LOOKBACK_MONTHS = 12          # <-- IMPORTANT if your data is monthly (EOM). Needs >=2 observations.
RIDGE_LAMBDA = 1e-6           # covariance ridge for stability
RISK_AVERSION = 1.0           # objective: maximize mu'w - (gamma/2) w'Σw
LONG_ONLY = True              # w >= 0
WEIGHT_CAP = None             # e.g., 0.05 for 5% cap, or None


@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# -----------------------------
# Helpers
# -----------------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    """Parse many date formats (YYYYMMDD int, YYYY-MM-DD string, datetime)."""
    if np.issubdtype(s.dtype, np.number):
        # e.g., 20240628
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the month-end date for timestamp ts."""
    return (ts + pd.offsets.MonthEnd(0))


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the month-start date for timestamp ts."""
    return ts.replace(day=1)


def build_monthly_windows(run_start: str, run_end: str) -> list[Window]:
    """
    Build monthly train/test windows:
      train_month = current month (used only to define train_end)
      test_month  = next month
    Training sample actually uses LOOKBACK_MONTHS ending at train_end.
    """
    start = pd.Timestamp(run_start)
    end = pd.Timestamp(run_end)

    # We'll iterate over month starts for the TRAIN month
    train_month_starts = pd.date_range(month_start(start), month_start(end), freq="MS")

    windows: list[Window] = []
    for tms in train_month_starts:
        te = month_end(tms)
        ts = month_start(tms)

        next_ms = tms + pd.offsets.MonthBegin(1)
        test_s = month_start(next_ms)
        test_e = month_end(next_ms)

        windows.append(Window(train_start=ts, train_end=te, test_start=test_s, test_end=test_e))
    return windows


def detect_and_build_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a wide matrix of returns with:
      index = datetime date
      columns = tickers
      values = stock returns (float)
    Supports:
      - LONG format: ['date','tic','stock_ret'] (case-insensitive variants)
      - WIDE format: 'date' + many numeric columns (tickers)
    """
    cols = {c.lower(): c for c in df.columns}

    # Locate date column
    date_col = None
    for cand in ["date", "Date", "DATE", "date_key"]:
        if cand.lower() in cols:
            date_col = cols[cand.lower()]
            break
    if date_col is None:
        raise ValueError(f"Could not find a date column in {list(df.columns)[:20]}...")

    df = df.copy()
    df[date_col] = parse_date_series(df[date_col])
    df = df.dropna(subset=[date_col])

    # LONG format?
    tic_col = cols.get("tic")
    ret_col = cols.get("stock_ret") or cols.get("ret") or cols.get("return")

    if tic_col is not None and ret_col is not None:
        df[tic_col] = df[tic_col].astype(str).str.upper()
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

        R = (df.pivot_table(index=date_col, columns=tic_col, values=ret_col, aggfunc="mean")
               .sort_index())
        return R

    # Otherwise, assume WIDE: all non-date columns that are numeric-ish are tickers
    other_cols = [c for c in df.columns if c != date_col]
    # Try to coerce to numeric and keep those with at least one numeric value
    wide = df[[date_col] + other_cols].copy()
    for c in other_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    wide = wide.set_index(date_col).sort_index()

    # Drop columns with all NaN
    wide = wide.dropna(axis=1, how="all")
    return wide


def safe_cov(mat: pd.DataFrame, ridge: float = 1e-6) -> np.ndarray:
    """Compute covariance with NaN handling + ridge."""
    X = mat.to_numpy(dtype=float)
    # If there are NaNs, pandas cov handles pairwise, but can yield NaNs. We'll use pandas then sanitize.
    S = mat.cov().to_numpy(dtype=float)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    # Ridge
    if ridge and ridge > 0:
        S = S + ridge * np.eye(S.shape[0])
    return S


def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, gamma: float = 1.0,
                          long_only: bool = True, cap: float | None = None) -> np.ndarray:
    """
    Solve:
        max_w  mu'w - (gamma/2) w' Sigma w
        s.t.   sum(w)=1, and (if long_only) w>=0, and (if cap) w<=cap
    """
    n = len(mu)

    def objective(w):
        return -(mu @ w - 0.5 * gamma * (w @ Sigma @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = []
    if long_only:
        lo = 0.0
    else:
        lo = None
    hi = cap if cap is not None else None
    for _ in range(n):
        bounds.append((lo, hi))

    w0 = np.full(n, 1.0 / n)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success or np.any(~np.isfinite(res.x)):
        raise RuntimeError(res.message)

    w = res.x
    # Numerical cleanup
    w = np.clip(w, 0.0, None) if long_only else w
    w = w / w.sum() if w.sum() != 0 else np.full(n, 1.0 / n)
    return w


def portfolio_return(R: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Compute portfolio return time series for returns matrix R over its index."""
    return pd.Series(R.to_numpy(dtype=float) @ w, index=R.index, name="port_ret")


# -----------------------------
# Main
# -----------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run your data prep first.")

    raw = pd.read_csv(DATA_PATH)
    R = detect_and_build_returns(raw)

    # Show coverage (helps debug "No data for this window")
    print(f"Loaded returns matrix: {R.shape[0]} dates x {R.shape[1]} tickers")
    print(f"Date range: {R.index.min().date()} -> {R.index.max().date()}")

    windows = build_monthly_windows(RUN_START, RUN_END)

    rows = []
    weights_rows = []

    for w in windows:
        print(f"\nProcessing: Training {w.train_start.date()} to {w.train_end.date()}, "
              f"Testing {w.test_start.date()} to {w.test_end.date()}")

        # Training sample uses lookback months ending at train_end
        train_lb_start = (w.train_end - pd.DateOffset(months=LOOKBACK_MONTHS)) + pd.offsets.MonthBegin(0)
        train_slice = R.loc[train_lb_start:w.train_end]
        test_slice = R.loc[w.test_start:w.test_end]

        if train_slice.empty or test_slice.empty:
            print("No data for this window (train or test slice empty). Skipping.")
            continue

        # Keep assets that exist in BOTH train and test
        assets = train_slice.columns.intersection(test_slice.columns)

        train_slice = train_slice[assets]
        test_slice = test_slice[assets]

        # Drop assets with all NaN in either slice
        train_slice = train_slice.dropna(axis=1, how="all")
        test_slice = test_slice.dropna(axis=1, how="all")
        assets = train_slice.columns.intersection(test_slice.columns)
        train_slice = train_slice[assets]
        test_slice = test_slice[assets]

        if len(assets) == 0:
            print("No overlapping assets with non-missing returns. Skipping.")
            continue

        print(f"Assets used: {len(assets)} | Train obs: {train_slice.shape[0]} | Test obs: {test_slice.shape[0]}")

        # Equal-weight
        w_eq = np.full(len(assets), 1.0 / len(assets))
        eq_ret = portfolio_return(test_slice.fillna(0.0), w_eq)  # fill missing with 0 for stability
        eq_total = (1.0 + eq_ret).prod() - 1.0

        # Mean-Variance
        # Need at least 2 observations for covariance; if not, fallback to equal-weight
        mv_ok = True
        if train_slice.shape[0] < 2:
            mv_ok = False

        if mv_ok:
            mu = train_slice.mean(skipna=True).to_numpy(dtype=float)
            Sigma = safe_cov(train_slice, ridge=RIDGE_LAMBDA)

            try:
                w_mv = mean_variance_weights(mu, Sigma, gamma=RISK_AVERSION,
                                             long_only=LONG_ONLY, cap=WEIGHT_CAP)
            except Exception as e:
                print(f"Optimization failed ({e}). Falling back to equal weights.")
                w_mv = w_eq
        else:
            print("Not enough training observations for covariance. Falling back to equal weights.")
            w_mv = w_eq

        mv_ret = portfolio_return(test_slice.fillna(0.0), w_mv)
        mv_total = (1.0 + mv_ret).prod() - 1.0

        rows.append({
            "train_start": w.train_start.date().isoformat(),
            "train_end": w.train_end.date().isoformat(),
            "train_lookback_start": train_lb_start.date().isoformat(),
            "test_start": w.test_start.date().isoformat(),
            "test_end": w.test_end.date().isoformat(),
            "n_assets": int(len(assets)),
            "n_train_obs": int(train_slice.shape[0]),
            "n_test_obs": int(test_slice.shape[0]),
            "eq_total_return": float(eq_total),
            "mv_total_return": float(mv_total),
        })

        # Save weights for the train month end (tag by train_end month)
        tag = pd.Timestamp(w.train_end).strftime("%Y-%m")
        w_df = pd.DataFrame({"tic": assets, "w_mv": w_mv, "w_eq": w_eq})
        w_df["train_end"] = w.train_end.date().isoformat()
        w_df["test_start"] = w.test_start.date().isoformat()
        w_df["test_end"] = w.test_end.date().isoformat()

        weights_path = OUT_DIR / f"weights_{tag}.csv"
        w_df.to_csv(weights_path, index=False)

        print(f"Saved weights: {weights_path.as_posix()}")

    if not rows:
        print("\nNo windows produced results. Check your data coverage and RUN_START/RUN_END.")
        return

    results = pd.DataFrame(rows)
    results_path = OUT_DIR / "baselines_results.csv"
    results.to_csv(results_path, index=False)

    print(f"\nSaved results: {results_path.as_posix()}")


if __name__ == "__main__":
    main()
