"""
Baselines portfolios computed from ONE master file: data/filtered_sp500_data.csv

This script creates, for each monthly test period:
  1. an equal-weight portfolio return
  2. a mean-variance optimized portfolio return

Important timing convention:
  - The training window ends at month M.
  - The portfolio return is evaluated on month M+1.
  - The universe is restricted to names available in the test month and with enough
    training observations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    parse_master,
    build_monthly_panel,
    pivot_monthly_returns,
    _month_starts_inclusive,
    optimize_mean_variance,
)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--master_path", type=str, default="data/filtered_sp25_data.csv")
    ap.add_argument("--out_dir", type=str, default="responses_portfolios")

    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default="2025-06-30")

    ap.add_argument("--lookback_months", type=int, default=12)
    ap.add_argument(
        "--min_train_rows",
        type=int,
        default=12,
        help="Minimum number of monthly observations required to run MVO optimization.",
    )
    ap.add_argument("--lambda_param", type=float, default=0.1)

    ap.add_argument(
        "--allow_short",
        action="store_true",
        help="Allow short positions in MVO. Default is long-only.",
    )

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    master_path = Path(args.master_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback = max(1, int(args.lookback_months))
    min_train_rows = max(1, int(args.min_train_rows))
    long_only = not args.allow_short

    df = pd.read_csv(master_path, low_memory=False)
    df = parse_master(df)

    monthly_panel = build_monthly_panel(df)

    # Available month range in the file
    min_ym = monthly_panel["ym"].min()
    max_ym = monthly_panel["ym"].max()

    print(f"Data months available in CSV: {min_ym} -> {max_ym}")
    print(f"MVO mode: {'long-only' if long_only else 'shorting allowed'}")

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    # Because test month is M+1, last training month must be <= end_dt - 1 month
    last_train_month_start = (end_dt - pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

    month_starts = _month_starts_inclusive(args.start, args.end)
    month_starts = month_starts[month_starts <= last_train_month_start]

    if len(month_starts) == 0:
        raise ValueError(
            f"No training months generated. Because test is next month, "
            f"training stops at {last_train_month_start.date()}."
        )

    for train_start_dt in month_starts:
        train_end_dt = train_start_dt + pd.offsets.MonthEnd(1)
        train_end_p = train_start_dt.to_period("M")
        test_p = train_end_p + 1

        test_start_dt = test_p.to_timestamp()
        test_end_dt = test_start_dt + pd.offsets.MonthEnd(1)

        out_eq = out_dir / f"equal_weighted_portfolio_{train_start_dt.date()}_{train_end_dt.date()}.csv"
        out_opt = out_dir / f"optimized_portfolio_{train_start_dt.date()}_{train_end_dt.date()}.csv"

        if (not args.overwrite) and out_eq.exists() and out_opt.exists():
            print(f"Skip (exists): {train_start_dt.date()} -> {train_end_dt.date()}")
            continue

        # Base lookback window. We may expand backward to reach min_train_rows.
        train_start_p = train_end_p - (lookback - 1)
        cur_start_p = train_start_p

        # Build test slice: single month M+1.
        test_slice = monthly_panel[monthly_panel["ym"] == test_p].copy()
        if test_slice.empty:
            print(f"\nProcessing: {train_end_p} -> test {test_p}: No test data. Skipping.")
            continue

        test_tickers = set(test_slice["tic"].unique())

        train_R = None
        test_R = None
        used_train_start_p = None

        # Iteratively expand the training window backward if it is too short.
        while True:
            train_slice = monthly_panel[
                (monthly_panel["ym"] >= cur_start_p)
                & (monthly_panel["ym"] <= train_end_p)
            ].copy()

            # Universe: tickers present in test month and at least once in training.
            train_tickers = set(train_slice["tic"].unique())
            universe = sorted(test_tickers.intersection(train_tickers))

            if len(universe) == 0:
                train_R = None
                test_R = None
                used_train_start_p = cur_start_p
                break

            train_slice_u = train_slice[train_slice["tic"].isin(universe)].copy()
            test_slice_u = test_slice[test_slice["tic"].isin(universe)].copy()

            train_R_tmp = pivot_monthly_returns(train_slice_u)
            test_R_tmp = pivot_monthly_returns(test_slice_u)

            # Align columns between training and test.
            cols = train_R_tmp.columns.intersection(test_R_tmp.columns)
            train_R_tmp = train_R_tmp[cols]
            test_R_tmp = test_R_tmp[cols]

            # Drop assets with no training data.
            train_R_tmp = train_R_tmp.dropna(axis=1, how="all")
            test_R_tmp = test_R_tmp[train_R_tmp.columns]

            # Drop assets with fewer than 2 months of observations, needed for covariance.
            train_R_tmp = train_R_tmp.dropna(axis=1, thresh=2)
            test_R_tmp = test_R_tmp[train_R_tmp.columns]

            if train_R_tmp.shape[0] >= min_train_rows:
                train_R = train_R_tmp
                test_R = test_R_tmp
                used_train_start_p = cur_start_p
                break

            # If there are still not enough rows, expand backward if possible.
            if cur_start_p <= min_ym:
                train_R = train_R_tmp
                test_R = test_R_tmp
                used_train_start_p = cur_start_p
                break

            cur_start_p = cur_start_p - 1

        print(
            f"\nProcessing: Training window {used_train_start_p} -> {train_end_p} "
            f"(train_end={train_end_dt.date()}), "
            f"Testing {test_start_dt.date()} -> {test_end_dt.date()}"
        )

        if train_R is None or train_R.empty or test_R is None or test_R.empty:
            print("No usable assets after cleaning. Skipping.")
            continue

        n_assets = train_R.shape[1]

        # ----------------------------
        # Equal-weight portfolio
        # ----------------------------
        w_eq = np.full(n_assets, 1.0 / n_assets)
        eq_ret = float((test_R.fillna(0.0).to_numpy(dtype=float) @ w_eq).ravel()[0])

        equal_weighted_portfolio = pd.DataFrame(
            {
                "Date": [test_R.index[0]],
                "Portfolio_Return": [eq_ret],
            }
        )
        equal_weighted_portfolio.to_csv(out_eq, index=False)

        # ----------------------------
        # Mean-variance optimized portfolio
        # ----------------------------
        if train_R.shape[0] < min_train_rows:
            print(
                f"Not enough training months for MVO "
                f"(have {train_R.shape[0]}, need {min_train_rows}). "
                "Using equal weights for optimized portfolio."
            )
            w_opt = w_eq

        else:
            obs = train_R.notna().sum(axis=0)
            keep_cols = obs[obs >= min_train_rows].index

            print(f"[MVO] Keeping {len(keep_cols)}/{train_R.shape[1]} assets with >= {min_train_rows} train months.")

            train_R_mvo = train_R[keep_cols].copy()

            if train_R_mvo.shape[1] < 2:
                print("[MVO] Not enough assets after completeness filter. Falling back to equal weights.")
                w_opt = w_eq

            else:
                try:
                    w_opt_mvo = optimize_mean_variance(
                        train_R_mvo,
                        lambda_param=float(args.lambda_param),
                        long_only=long_only,
                    )

                    # Map optimized weights back to full test universe.
                    # Assets dropped from MVO due to incomplete history get 0 weight.
                    w_opt = np.zeros(train_R.shape[1], dtype=float)
                    col_to_pos = {c: i for i, c in enumerate(train_R.columns)}

                    for c, w in zip(train_R_mvo.columns, w_opt_mvo):
                        w_opt[col_to_pos[c]] = float(w)

                    # Re-normalize after mapping back.
                    s = float(w_opt.sum())
                    if abs(s) > 1e-12:
                        w_opt = w_opt / s
                    else:
                        print("[MVO] Optimized weights sum to zero. Falling back to equal weights.")
                        w_opt = w_eq

                except Exception as e:
                    print(f"Optimization failed ({e}). Falling back to equal weights.")
                    w_opt = w_eq

        opt_ret = float((test_R.fillna(0.0).to_numpy(dtype=float) @ w_opt).ravel()[0])

        optimized_portfolio = pd.DataFrame(
            {
                "Date": [test_R.index[0]],
                "Portfolio_Return": [opt_ret],
            }
        )
        optimized_portfolio.to_csv(out_opt, index=False)

        print(
            f"Saved: {out_eq.name} & {out_opt.name} | "
            f"Assets: {n_assets} | Train months: {train_R.shape[0]} | Test month: {test_p}"
        )


if __name__ == "__main__":
    main()
