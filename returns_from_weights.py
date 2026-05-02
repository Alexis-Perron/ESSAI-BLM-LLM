import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from utils import (
    _clean_ticker,
    _month_starts_between,
    _normalize_weights_columns,
    _parse_dataset_dates,
    _get_monthly_returns_series,
)

# -------------------------
# Core computation
# -------------------------
def calculate_model_returns(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    dataset_path: str = "data/filtered_sp500_data.csv",
    results_dir: str = "results",
    weights_path: Optional[str] = None,
    out_path: Optional[str] = None,
    apply_next_month: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute portfolio returns for a model from BL weights + realized returns,
    WITHOUT using per-month returns_*.csv files.

    Realized returns are taken directly from `filtered_sp500_data.csv`:
      - ticker column: auto-detected (prefers 'tic')
      - date column: auto-detected (prefers 'date')
      - return column: 'stock_ret'
      - for each month, we use the last observation per ticker as that month's return.

    apply_next_month:
      - True  => weights month M applied to returns of month M+1
      - False => weights month M applied to returns of month M
    """
    model_raw = model_name.strip()
    model = model_raw.lower()

    if weights_path is None:
        weights_path = str(Path(results_dir) / f"{model}_black_litterman_weights_tau_{tau}.csv")
    weights_path = str(weights_path)

    if out_path is None:
        out_path = str(Path(results_dir) / f"{model}_black_litterman_returns_tau_{tau}_{start_date}_{end_date}.csv")
    out_path = str(out_path)

    candidates = [Path(weights_path)]
    if model_raw and model_raw.lower() != model_raw:
        candidates.append(Path(results_dir) / f"{model_raw}_black_litterman_weights_tau_{tau}.csv")

    weights_file = next((c for c in candidates if c.exists()), None)
    if weights_file is None:
        return None

    weights_df = pd.read_csv(weights_file)

    weights_df['Date'] = pd.to_datetime(weights_df['Date'], errors="coerce")

    if weights_df['Date'].isna().all():
        raise ValueError(f"[{model}] Could not parse dates in weights file column 'Date'.")

    # Normalize weights dates to month-start
    weights_df['Date'] = weights_df['Date'].dt.to_period("M").dt.to_timestamp()
    # Normalize / aggregate asset columns by cleaned ticker
    weights_df, asset_columns = _normalize_weights_columns(weights_df, date_col="Date")

    # Load dataset once
    ds = pd.read_csv(dataset_path)

    ret_col = "stock_ret"
    if ret_col not in ds.columns:
        raise ValueError(f"'{ret_col}' column not found in dataset. Columns={list(ds.columns)[:30]}...")

    ds['date'] = _parse_dataset_dates(ds['date'])
    ds = ds.dropna(subset=['date']).copy()
    ds['tic'] = ds['tic'].astype(str).map(_clean_ticker)
    ds = ds[ds['tic'] != ""].copy()

    # Precompute month key
    ds["ym"] = ds['date'].dt.to_period("M").astype(str)

    # Group by month once (fast lookups)
    month_groups = dict(tuple(ds.groupby("ym", sort=False)))

    month_starts = _month_starts_between(start_date, end_date)
    out_rows = []

    for m0 in month_starts:
        weight_date = m0.to_period("M").to_timestamp()

        if apply_next_month:
            target = (m0 + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
        else:
            target = m0.to_period("M").to_timestamp()

        target_ym = pd.Period(target, freq="M").strftime("%Y-%m")

        if target_ym not in month_groups:
            msg = f"[{model}] No data in dataset for target month {target_ym}."
            continue

        month_df = month_groups[target_ym]
        r_series, month_end_date = _get_monthly_returns_series(
            month_df=month_df,
            ticker_col='tic',
            date_col='date',
            ret_col=ret_col,
        )

        if r_series.empty or not np.isfinite(r_series.to_numpy(dtype=float)).any():
            msg = f"[{model}] No valid numeric returns in dataset for target month {target_ym}."
            continue

        # Select weights row for weight month
        row = weights_df.loc[weights_df['Date'] == weight_date]
        if row.empty:
            msg = f"[{model}] No weights found for {weight_date.date()} in {weights_file.name}"
            continue

        w = row[asset_columns].iloc[0].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # Align assets between weights and available returns
        r = r_series.reindex(asset_columns).to_numpy(dtype=float)
        finite = np.isfinite(r)

        if not finite.any():
            msg = f"[{model}] No finite returns for target month {target_ym} after alignment."
            continue

        # Use only assets with finite returns
        w_f = w[finite]
        r_f = r[finite]

        # Renormalization:
        # - normally: normalize by sum(w_f)
        # - if sum is ~0 (e.g., all zero weights or long/short cancels), fallback:
        #     1) if sum(abs(w_f)) > eps: normalize by sum(abs(w_f)) (keeps relative exposure)
        #     2) else: equal weight on available assets (strategy "does something" rather than skipping)
        eps = 1e-12
        denom = float(np.sum(w_f))
        if not np.isfinite(denom) or abs(denom) < eps:
            abs_denom = float(np.sum(np.abs(w_f)))
            if np.isfinite(abs_denom) and abs_denom >= eps:
                denom = abs_denom
                w_use = np.abs(w_f)
            else:
                denom = float(len(w_f))
                w_use = np.ones_like(w_f, dtype=float)
        else:
            w_use = w_f

        port_ret = float(np.sum(w_use * r_f) / denom)

        out_rows.append(
            {
                "date_key": month_end_date if pd.notna(month_end_date) else target,
                "Portfolio_Return": port_ret,
            }
        )

    if not out_rows:
        return None

    out = pd.DataFrame(out_rows).sort_values("date_key").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    return out


# -------------------------
# CLI
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute portfolio returns from BL weights + dataset returns.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names (e.g., gpt gemma3).")
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2025-06-30")
    parser.add_argument("--dataset_path", type=str, default="data/filtered_sp500_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--apply_next_month", action="store_true", help="Apply weights at month M to returns of month M+1.")
    parser.add_argument("--no_apply_next_month", action="store_true", help="Apply weights at month M to returns of month M.")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    apply_next_month = True
    if args.no_apply_next_month:
        apply_next_month = False
    elif args.apply_next_month:
        apply_next_month = True


    for m in args.models:
        calculate_model_returns(
            model_name=m,
            tau=args.tau,
            start_date=args.start_date,
            end_date=args.end_date,
            dataset_path=args.dataset_path,
            results_dir=args.results_dir,
            apply_next_month=apply_next_month,
        )


if __name__ == "__main__":
    main()
