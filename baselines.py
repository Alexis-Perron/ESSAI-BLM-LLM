#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baselines portfolios computed from ONE master file: yfinance/filtered_sp500_data.csv

Fixes vs your current baselines.py:
1) Windows are filtered using (year, month) from the master file (more robust than comparing dates).
2) Mean-variance uses a LOOKBACK window (default 12 months) so covariance is computable with monthly data.
3) Universe comes directly from master columns:
   - tic (ticker)
   - market_equity (keeps only tickers with a valid market cap in the TRAIN-END month)

Outputs (same style as before):
  responses_portfolios/equal_weighted_portfolio_<train_start>_<train_end>.csv
  responses_portfolios/optimized_portfolio_<train_start>_<train_end>.csv
"""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------
# Config
# ----------------------------
MASTER_PATH = Path("yfinance/filtered_sp500_data.csv")
OUT_DIR = Path("responses_portfolios")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEAR = 2021
TRAIN_MONTHS = range(1, 7)      # June..November => 6 windows
LOOKBACK_MONTHS = 12             # <-- KEY FIX for monthly data
LAMBDA_PARAM = 0.1               # same spirit as original
LONG_ONLY = True                 # no short-selling


# ----------------------------
# Helpers
# ----------------------------
def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures columns are in the right dtype and adds:
      - Date: datetime parsed from 'date' (YYYYMMDD int or string)
      - ym: monthly Period (YYYY-MM) built from (year, month)
    """
    df = df.copy()

    required = {"date", "year", "month", "tic", "stock_ret"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"filtered_sp500_data.csv is missing required columns: {sorted(missing)}")

    # Parse date (month-end trading date)
    if np.issubdtype(df["date"].dtype, np.number):
        df["Date"] = pd.to_datetime(df["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Normalize ticker and returns
    df["tic"] = df["tic"].astype(str).str.upper().str.strip()
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")

    # year/month as ints
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    # Monthly period index (robust filtering)
    ym_dt = pd.to_datetime(
        dict(year=df["year"].astype("Int64"),
             month=df["month"].astype("Int64"),
             day=1),
        errors="coerce"
    )
    df["ym"] = ym_dt.dt.to_period("M")
    df = df.dropna(subset=["ym"])

    if "market_equity" in df.columns:
        df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")

    return df


def build_returns_matrix(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot (Date, tic, stock_ret) into wide returns:
      index  = Date (one row per month in your file)
      cols   = tickers
      values = stock_ret
    """
    R = (df_slice.pivot_table(index="Date", columns="tic", values="stock_ret", aggfunc="mean")
                  .sort_index())
    return R


def optimize_mean_variance(train_R: pd.DataFrame, lambda_param: float = 0.1, long_only: bool = True) -> np.ndarray:
    """
    Minimize: w' Σ w - lambda * (μ' w)
    s.t. sum(w)=1 and w>=0 (if long_only)
    """
    mu = train_R.mean(skipna=True)
    Sigma = train_R.cov()

    mu_v = mu.to_numpy(dtype=float)
    S = Sigma.to_numpy(dtype=float)

    if not np.isfinite(S).all():
        raise RuntimeError("Covariance contains NaN/inf (too few obs or too many missing values).")

    n = len(mu_v)

    def objective(w: np.ndarray) -> float:
        port_ret = float(mu_v @ w)
        port_risk = float(w.T @ S @ w)
        return port_risk - (lambda_param * port_ret)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)] if long_only else [(None, None) for _ in range(n)]
    w0 = np.full(n, 1.0 / n)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if (not res.success) or (not np.isfinite(res.x).all()):
        raise RuntimeError(res.message)

    w = res.x
    if long_only:
        w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s != 0 else np.full(n, 1.0 / n)


# ----------------------------
# Main
# ----------------------------
def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Cannot find {MASTER_PATH}. Make sure the path is correct.")

    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df = parse_master(df)

    for train_month in TRAIN_MONTHS:
        train_year = TRAIN_YEAR

        # test month/year
        if train_month == 12:
            test_month, test_year = 1, train_year + 1
        else:
            test_month, test_year = train_month + 1, train_year

        train_start = pd.Timestamp(train_year, train_month, 1)
        train_end = pd.Timestamp(train_year, train_month, get_last_day_of_month(train_year, train_month))
        test_start = pd.Timestamp(test_year, test_month, 1)
        test_end = pd.Timestamp(test_year, test_month, get_last_day_of_month(test_year, test_month))

        print(f"\nProcessing: Training {train_start.date()} to {train_end.date()}, Testing {test_start.date()} to {test_end.date()}")

        # Periods
        train_end_p = pd.Period(f"{train_year}-{train_month:02d}", freq="M")
        train_start_p = train_end_p - (LOOKBACK_MONTHS - 1)
        test_p = train_end_p + 1

        # Train = lookback months up to train_end; Test = next month
        train_df = df[(df["ym"] >= train_start_p) & (df["ym"] <= train_end_p)]
        test_df = df[df["ym"] == test_p]

        # Force a single month-end Date for the test month (use the latest date present)
        test_last_date = test_df["Date"].max()
        test_df = test_df[test_df["Date"] == test_last_date]

        print("Train months expected:", train_start_p, "->", train_end_p)
        print("Train months present :", sorted(train_df["ym"].unique()))
        print("Nb rows train_df:", len(train_df))
        print("Nb unique dates:", train_df["Date"].nunique())


        if train_df.empty or test_df.empty:
            print("No data for this window. Skipping.")
            continue

        # Universe: tickers with market_equity in TRAIN-END month AND present in TEST month
        train_end_df = df[df["ym"] == train_end_p]
        if "market_equity" in train_end_df.columns:
            train_tickers = set(train_end_df.loc[train_end_df["market_equity"].notna(), "tic"].unique())
        else:
            train_tickers = set(train_end_df["tic"].unique())

        test_tickers = set(test_df["tic"].unique())
        universe = sorted(train_tickers.intersection(test_tickers))

        if not universe:
            print("No overlapping tickers between train-end and test. Skipping.")
            continue

        train_df = train_df[train_df["tic"].isin(universe)]
        test_df = test_df[test_df["tic"].isin(universe)]

        # Build returns matrices
        train_R = build_returns_matrix(train_df)
        test_R = build_returns_matrix(test_df)

        # Align columns
        cols = train_R.columns.intersection(test_R.columns)
        train_R = train_R[cols]
        test_R = test_R[cols]

        # Drop assets with no data
        train_R = train_R.dropna(axis=1, how="all")
        test_R = test_R[train_R.columns]

        # Covariance needs >=2 observations; also drop assets with <2 non-NA points
        train_R = train_R.dropna(axis=1, thresh=2)
        test_R = test_R[train_R.columns]

        asset_columns = list(train_R.columns)
        n_assets = len(asset_columns)
        if n_assets == 0 or test_R.empty:
            print("No usable assets after cleaning. Skipping.")
            continue

        # Equal-weight
        w_eq = np.full(n_assets, 1.0 / n_assets)
        eq_returns = (test_R.fillna(0.0).to_numpy(dtype=float) @ w_eq)

        equal_weighted_portfolio = pd.DataFrame({"Date": test_R.index, "Portfolio_Return": eq_returns})
        equal_weighted_portfolio.to_csv(
            OUT_DIR / f"equal_weighted_portfolio_{train_start.date()}_{train_end.date()}.csv",
            index=False
        )

        # Optimized
        if train_R.shape[0] < 2:
            print("Not enough training observations for covariance. Using equal weights for optimized portfolio.")
            w_opt = w_eq
        else:
            try:
                w_opt = optimize_mean_variance(train_R, lambda_param=LAMBDA_PARAM, long_only=LONG_ONLY)
            except Exception as e:
                print(f"Optimization failed ({e}). Falling back to equal weights.")
                w_opt = w_eq

        opt_returns = (test_R.fillna(0.0).to_numpy(dtype=float) @ w_opt)
        optimized_portfolio = pd.DataFrame({"Date": test_R.index, "Portfolio_Return": opt_returns})
        optimized_portfolio.to_csv(
            OUT_DIR / f"optimized_portfolio_{train_start.date()}_{train_end.date()}.csv",
            index=False
        )

        print(f"Saved portfolio results for training period: {train_start.date()} to {train_end.date()} | "
              f"Assets: {n_assets} | Train obs: {train_R.shape[0]}")

if __name__ == "__main__":
    main()
