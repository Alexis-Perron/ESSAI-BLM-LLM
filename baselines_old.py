import os
import calendar

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_date_col(s: pd.Series) -> pd.Series:
    """Parse a date column that may be YYYYMMDD ints or ISO strings."""
    if np.issubdtype(s.dtype, np.number):
        # ex: 20240628 -> 2024-06-28
        return pd.to_datetime(
            s.astype('Int64').astype(str),
            format='%Y%m%d',
            errors='coerce',
        )
    return pd.to_datetime(s, errors='coerce')


def norm_tic(s: pd.Series) -> pd.Series:
    """Normalize tickers to improve matching (e.g., BRK-B -> BRK.B)."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .str.replace('-', '.', regex=False)
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    # Months to process (June to December 2024)
    months = range(6, 13)  # 6 = June, 12 = December

    in_path = 'yfinance/filtered_sp500_data.csv'
    out_dir = 'responses_portfolios'
    ensure_dir(out_dir)

    df = pd.read_csv(in_path)

    # --- Parse and normalize ---
    if 'date' not in df.columns:
        raise KeyError("Input CSV must contain a 'date' column")
    if 'tic' not in df.columns:
        raise KeyError("Input CSV must contain a 'tic' column")
    if 'stock_ret' not in df.columns:
        raise KeyError("Input CSV must contain a 'stock_ret' column")

    df['date'] = parse_date_col(df['date'])
    df = df.dropna(subset=['date']).copy()
    df['tic'] = norm_tic(df['tic'])
    df['stock_ret'] = pd.to_numeric(df['stock_ret'], errors='coerce')

    # --- Build wide returns matrix: index=date, columns=ticker, values=stock_ret ---
    returns_wide = (
        df.pivot_table(index='date', columns='tic', values='stock_ret', aggfunc='mean')
          .sort_index()
    )

    lambda_param = 0.1

    for month in months:
        train_year = 2024
        train_month = month

        # Handle year transition for December to January
        if month == 12:
            test_month = 1
            test_year = 2025
        else:
            test_month = month + 1
            test_year = train_year

        train_start = pd.Timestamp(train_year, train_month, 1)
        train_end = pd.Timestamp(train_year, train_month, get_last_day_of_month(train_year, train_month))
        test_start = pd.Timestamp(test_year, test_month, 1)
        test_end = pd.Timestamp(test_year, test_month, get_last_day_of_month(test_year, test_month))

        train_start_s = train_start.strftime('%Y-%m-%d')
        train_end_s = train_end.strftime('%Y-%m-%d')
        test_start_s = test_start.strftime('%Y-%m-%d')
        test_end_s = test_end.strftime('%Y-%m-%d')

        print(f"\nProcessing: Training {train_start_s} to {train_end_s}, Testing {test_start_s} to {test_end_s}")

        train = returns_wide.loc[train_start:train_end]
        test = returns_wide.loc[test_start:test_end]

        if train.empty or test.empty:
            print("No data for this window. Skipping.")
            continue

        # Keep only assets present in both windows
        common_cols = train.columns.intersection(test.columns)
        train = train[common_cols]
        test = test[common_cols]

        # Drop assets with all-NaN in either window
        non_empty = train.columns[train.notna().any(axis=0) & test.notna().any(axis=0)]
        train = train[non_empty]
        test = test[non_empty]

        n_assets = train.shape[1]
        if n_assets == 0:
            print("No overlapping assets with valid data. Skipping.")
            continue

        asset_columns = list(train.columns)
        print(f"Assets used: {n_assets}")

        # --------------------------
        # Equal-weighted portfolio
        # --------------------------
        w_eq = np.full(n_assets, 1.0 / n_assets, dtype=float)
        test_mat = np.nan_to_num(test.to_numpy(dtype=float), nan=0.0)
        eq_returns = test_mat @ w_eq

        equal_weighted_portfolio = pd.DataFrame({
            'date': test.index,
            'Portfolio_Return': eq_returns,
        })

        equal_weighted_portfolio.to_csv(
            f'{out_dir}/equal_weighted_portfolio_{train_start_s}_{train_end_s}.csv',
            index=False,
        )

        # --------------------------
        # Mean-variance optimized portfolio
        # objective: w'Σw - λ μ'w
        # constraints: sum(w)=1, w>=0
        # --------------------------
        mean_returns = train.mean(skipna=True).to_numpy(dtype=float)
        cov_matrix = train.cov().to_numpy(dtype=float)

        def objective(weights: np.ndarray) -> float:
            port_ret = float(np.dot(mean_returns, weights))
            port_risk = float(weights.T @ cov_matrix @ weights)
            return port_risk - (lambda_param * port_ret)

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: x},
        )
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        x0 = w_eq.copy()

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9},
        )

        if not result.success or result.x is None:
            print(f"Optimization failed ({result.message}). Falling back to equal weights.")
            w_opt = w_eq
        else:
            w_opt = result.x
            # numerical cleanup
            w_opt = np.clip(w_opt, 0.0, 1.0)
            s = w_opt.sum()
            w_opt = w_opt / s if s != 0 else w_eq

        opt_returns = test_mat @ w_opt

        optimized_portfolio = pd.DataFrame({
            'date': test.index,
            'Portfolio_Return': opt_returns,
        })

        optimized_portfolio.to_csv(
            f'{out_dir}/optimized_portfolio_{train_start_s}_{train_end_s}.csv',
            index=False,
        )

        # Optional: save weights for inspection
        weights_df = pd.DataFrame({'tic': asset_columns, 'weight': w_opt})
        weights_df.to_csv(
            f'{out_dir}/optimized_weights_{train_start_s}_{train_end_s}.csv',
            index=False,
        )

        print(f"Saved portfolio results for training period: {train_start_s} to {train_end_s}")


if __name__ == '__main__':
    main()
