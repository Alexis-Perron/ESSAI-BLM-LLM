#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baselines portfolios computed from ONE master file: yfinance/filtered_sp500_data.csv

Key changes vs prior version:
- Uses MONTHLY panel: last observation per (tic, ym) from filtered_sp500_data.csv
- Adds min_train_rows (default 12) and will expand history backward to reach it if possible
- Does NOT reduce the universe by market cap (no top-N filtering)
- Still outputs one CSV per training month:
    responses_portfolios/equal_weighted_portfolio_<train_start>_<train_end>.csv
    responses_portfolios/optimized_portfolio_<train_start>_<train_end>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------
# Helpers
# ----------------------------
def _normalize_ticker_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace("-", ".", regex=False)
    return s


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

    # Parse date (YYYYMMDD)
    if np.issubdtype(df["date"].dtype, np.number):
        df["Date"] = pd.to_datetime(df["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Normalize ticker and returns
    df["tic"] = _normalize_ticker_series(df["tic"])
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")

    # year/month as ints
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    ym_dt = pd.to_datetime(
        dict(
            year=df["year"].astype("Int64"),
            month=df["month"].astype("Int64"),
            day=1,
        ),
        errors="coerce",
    )
    df["ym"] = ym_dt.dt.to_period("M")
    df = df.dropna(subset=["ym"]).copy()

    if "market_equity" in df.columns:
        df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")

    return df


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a monthly panel with ONE row per (tic, ym):
      - keeps the last available Date in that month for each ticker
      - keeps stock_ret for that snapshot

    Output columns include: [ym (Period), ym_dt (Timestamp month-start), tic, stock_ret]
    """
    tmp = df.dropna(subset=["ym", "tic"]).copy()
    tmp = tmp.sort_values(["tic", "ym", "Date"])
    tmp = tmp.drop_duplicates(subset=["tic", "ym"], keep="last")  # last obs of the month

    tmp["ym_dt"] = tmp["ym"].dt.to_timestamp()  # month-start Timestamp
    tmp = tmp[["ym", "ym_dt", "tic", "stock_ret"]].copy()
    return tmp


def pivot_monthly_returns(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot monthly panel to wide returns matrix:
      index = ym_dt (month start)
      columns = tickers
      values = stock_ret
    """
    R = (
        monthly_panel.pivot_table(index="ym_dt", columns="tic", values="stock_ret", aggfunc="mean")
        .sort_index()
    )
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


def _month_starts_inclusive(start: str, end: str) -> pd.DatetimeIndex:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return pd.date_range(s, e, freq="MS")


def _debug_opt_diagnostics(train_R: pd.DataFrame, debug_top: int = 15) -> None:
    obs_per_asset = train_R.notna().sum(axis=0).sort_values()
    na_ratio = float(train_R.isna().mean().mean()) if train_R.size > 0 else np.nan

    print(f"[DEBUG_OPT] train_R shape={train_R.shape} | avg NaN ratio={na_ratio:.3%}")
    if len(obs_per_asset) > 0:
        print("[DEBUG_OPT] obs per asset (min/median/max):",
              int(obs_per_asset.min()), float(obs_per_asset.median()), int(obs_per_asset.max()))
        low = obs_per_asset[obs_per_asset < 3]
        if len(low) > 0:
            show = low.head(int(debug_top))
            print(f"[DEBUG_OPT] assets with <3 obs (show {len(show)}/{len(low)}): {show.index.tolist()}")

    Sigma = train_R.cov()
    n_nan_cov = int(Sigma.isna().sum().sum())
    print(f"[DEBUG_OPT] covariance NaN entries={n_nan_cov}")

    S = np.nan_to_num(Sigma.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        rank = int(np.linalg.matrix_rank(S))
        cond = float(np.linalg.cond(S)) if S.shape[0] else np.nan
        print(f"[DEBUG_OPT] covariance rank={rank}/{S.shape[0]} | cond={cond:.3e}")
    except Exception as e:
        print(f"[DEBUG_OPT] rank/cond failed: {e}")

    if n_nan_cov > 0:
        nan_by_col = Sigma.isna().sum(axis=0).sort_values(ascending=False)
        print("[DEBUG_OPT] top covariance NaN columns:", nan_by_col.head(int(debug_top)).to_dict())


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_path", type=str, default="yfinance/filtered_sp500_data.csv")
    ap.add_argument("--out_dir", type=str, default="responses_portfolios")

    ap.add_argument("--start", type=str, default="2021-01-01")
    ap.add_argument("--end", type=str, default="2022-06-30")

    ap.add_argument("--lookback_months", type=int, default=12)
    ap.add_argument("--min_train_rows", type=int, default=12, help="Minimum months required to run MVO optimization.")
    ap.add_argument("--lambda_param", type=float, default=0.1)
    ap.add_argument("--long_only", action="store_true", help="Long-only optimization (default True).")
    ap.add_argument("--allow_short", action="store_true", help="Allow shorting (overrides long_only).")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--debug_opt", action="store_true")
    ap.add_argument("--debug_opt_top", type=int, default=15)

    args = ap.parse_args()

    master_path = Path(args.master_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not master_path.exists():
        raise FileNotFoundError(f"Cannot find {master_path}.")

    lookback = max(1, int(args.lookback_months))
    min_train_rows = max(1, int(args.min_train_rows))

    long_only = True
    if args.allow_short:
        long_only = False
    elif args.long_only:
        long_only = True

    df = pd.read_csv(master_path, low_memory=False)
    df = parse_master(df)

    monthly_panel = build_monthly_panel(df)

    # Available month range in the file
    min_ym = monthly_panel["ym"].min()
    max_ym = monthly_panel["ym"].max()
    if not args.quiet:
        print(f"Data months available in CSV: {min_ym} -> {max_ym}")

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    # Because test month is M+1, last training month must be <= end_dt - 1 month
    last_train_month_start = (end_dt - pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

    month_starts = _month_starts_inclusive(args.start, args.end)
    month_starts = month_starts[month_starts <= last_train_month_start]

    if len(month_starts) == 0:
        raise ValueError(
            f"No training months generated. Because test is next month, training stops at {last_train_month_start.date()}."
        )

    for train_start_dt in month_starts:
        train_end_dt = (train_start_dt + pd.offsets.MonthEnd(1))
        train_end_p = train_start_dt.to_period("M")
        test_p = train_end_p + 1

        test_start_dt = test_p.to_timestamp()
        test_end_dt = (test_start_dt + pd.offsets.MonthEnd(1))

        out_eq = out_dir / f"equal_weighted_portfolio_{train_start_dt.date()}_{train_end_dt.date()}.csv"
        out_opt = out_dir / f"optimized_portfolio_{train_start_dt.date()}_{train_end_dt.date()}.csv"

        if (not args.overwrite) and out_eq.exists() and out_opt.exists():
            if not args.quiet:
                print(f"Skip (exists): {train_start_dt.date()} -> {train_end_dt.date()}")
            continue

        # Base lookback window (we may expand backward to reach min_train_rows)
        train_start_p = train_end_p - (lookback - 1)

        # Expand backward if needed to reach min_train_rows months (if data exists)
        cur_start_p = train_start_p

        # Build test slice (single month)
        test_slice = monthly_panel[monthly_panel["ym"] == test_p].copy()
        if test_slice.empty:
            if not args.quiet:
                print(f"\nProcessing: {train_end_p} -> test {test_p} : No test data. Skipping.")
            continue

        test_tickers = set(test_slice["tic"].unique())

        # Iteratively expand training window if too short
        train_R = None
        test_R = None
        used_train_start_p = None

        while True:
            train_slice = monthly_panel[(monthly_panel["ym"] >= cur_start_p) & (monthly_panel["ym"] <= train_end_p)].copy()

            # Universe: tickers present in TEST month, and present at least once in TRAIN slice
            train_tickers = set(train_slice["tic"].unique())
            universe = sorted(test_tickers.intersection(train_tickers))

            if len(universe) == 0:
                train_R = None
                used_train_start_p = cur_start_p
                break

            train_slice_u = train_slice[train_slice["tic"].isin(universe)].copy()
            test_slice_u = test_slice[test_slice["tic"].isin(universe)].copy()

            train_R_tmp = pivot_monthly_returns(train_slice_u)
            test_R_tmp = pivot_monthly_returns(test_slice_u)

            # Align columns
            cols = train_R_tmp.columns.intersection(test_R_tmp.columns)
            train_R_tmp = train_R_tmp[cols]
            test_R_tmp = test_R_tmp[cols]

            # Drop assets with no data
            train_R_tmp = train_R_tmp.dropna(axis=1, how="all")
            test_R_tmp = test_R_tmp[train_R_tmp.columns]

            # Drop assets with <2 months of observations (needed for cov)
            train_R_tmp = train_R_tmp.dropna(axis=1, thresh=2)
            test_R_tmp = test_R_tmp[train_R_tmp.columns]

            # Check rows in training (months)
            if train_R_tmp.shape[0] >= min_train_rows:
                train_R = train_R_tmp
                test_R = test_R_tmp
                used_train_start_p = cur_start_p
                break

            # If not enough rows, expand backward if possible
            if cur_start_p <= min_ym:
                train_R = train_R_tmp
                test_R = test_R_tmp
                used_train_start_p = cur_start_p
                break

            cur_start_p = cur_start_p - 1  # one more month back

        if not args.quiet:
            print(
                f"\nProcessing: Training window {used_train_start_p} -> {train_end_p} "
                f"(train_end={train_end_dt.date()}), Testing {test_start_dt.date()} -> {test_end_dt.date()}"
            )

        if train_R is None or train_R.empty or test_R is None or test_R.empty:
            if not args.quiet:
                print("No usable assets after cleaning. Skipping.")
            continue

        n_assets = train_R.shape[1]

        # Equal-weight (always)
        w_eq = np.full(n_assets, 1.0 / n_assets)
        eq_ret = float((test_R.fillna(0.0).to_numpy(dtype=float) @ w_eq).ravel()[0])
        equal_weighted_portfolio = pd.DataFrame({"Date": [test_R.index[0]], "Portfolio_Return": [eq_ret]})
        equal_weighted_portfolio.to_csv(out_eq, index=False)

        # Optimized: only if we have enough train rows
        if train_R.shape[0] < min_train_rows:
            if not args.quiet:
                print(
                    f"Not enough training months for MVO (have {train_R.shape[0]}, need {min_train_rows}). "
                    "Using equal weights for optimized portfolio."
                )
            w_opt = w_eq

        else:
            # === NEW: completeness filter for MVO (not a market-cap filter) ===
            obs = train_R.notna().sum(axis=0)

            keep_cols = obs[obs >= min_train_rows].index
            dropped = obs[obs < min_train_rows].sort_values()

            if not args.quiet:
                print(f"[MVO] Keeping {len(keep_cols)}/{train_R.shape[1]} assets with >= {min_train_rows} train months.")
                if len(dropped) > 0:
                    show = dropped.head(int(args.debug_opt_top))
                    print(f"[MVO] Dropping (worst {len(show)}/{len(dropped)}): {show.to_dict()}")

            train_R_mvo = train_R[keep_cols].copy()
            test_R_mvo = test_R[keep_cols].copy()

            if train_R_mvo.shape[1] < 2:
                print("[MVO] Not enough assets after completeness filter. Falling back to equal weights.")
                w_opt = w_eq
            else:
                try:
                    w_opt_mvo = optimize_mean_variance(
                        train_R_mvo,
                        lambda_param=float(args.lambda_param),
                        long_only=long_only
                    )

                    # Map back to full universe: dropped assets get 0 weight
                    w_opt = np.zeros(train_R.shape[1], dtype=float)
                    col_to_pos = {c: i for i, c in enumerate(train_R.columns)}
                    for c, w in zip(train_R_mvo.columns, w_opt_mvo):
                        w_opt[col_to_pos[c]] = float(w)

                    # Renormalize to sum to 1 (if needed)
                    s = float(w_opt.sum())
                    if s > 0:
                        w_opt = w_opt / s
                    else:
                        w_opt = w_eq

                except Exception as e:
                    print(f"Optimization failed ({e}). Falling back to equal weights.")
                    if args.debug_opt:
                        _debug_opt_diagnostics(train_R_mvo, debug_top=int(args.debug_opt_top))
                    w_opt = w_eq

        opt_ret = float((test_R.fillna(0.0).to_numpy(dtype=float) @ w_opt).ravel()[0])
        optimized_portfolio = pd.DataFrame({"Date": [test_R.index[0]], "Portfolio_Return": [opt_ret]})
        optimized_portfolio.to_csv(out_opt, index=False)

        if not args.quiet:
            print(
                f"Saved: {out_eq.name} & {out_opt.name} | "
                f"Assets: {n_assets} | Train months: {train_R.shape[0]} | Test month: {test_p}"
            )


if __name__ == "__main__":
    main()
