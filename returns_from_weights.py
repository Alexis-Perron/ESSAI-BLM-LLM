import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import re


def _month_starts_between(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Return month-start timestamps between start_date and end_date inclusive."""
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    return pd.date_range(start=s, end=e, freq="MS")


def _parse_dataset_dates(s: pd.Series) -> pd.Series:
    """
    Robustly parse dates from the dataset.

    Handles common formats:
      - int/float like 20141230 interpreted as YYYYMMDD
      - strings like '2014-12-30' or '20141230'
      - already-datetime values

    Returns datetime64[ns] with NaT for unparsable values.
    """
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        if np.isfinite(v).any():
            v_nonnull = v[np.isfinite(v)]
            if len(v_nonnull) and (np.nanmedian(v_nonnull) > 1_000_000):  # ~YYYYMMDD
                ss = v.astype("Int64").astype(str).str.zfill(8)
                return pd.to_datetime(ss, format="%Y%m%d", errors="coerce")
        return pd.to_datetime(v, errors="coerce")

    ss = s.astype(str).str.strip()
    looks_yyyymmdd = ss.str.match(r"^\d{8}$")
    out = pd.Series(pd.NaT, index=s.index)

    if looks_yyyymmdd.any():
        out.loc[looks_yyyymmdd] = pd.to_datetime(ss.loc[looks_yyyymmdd], format="%Y%m%d", errors="coerce")
    out.loc[~looks_yyyymmdd] = pd.to_datetime(ss.loc[~looks_yyyymmdd], errors="coerce")
    return out


def _clean_ticker(x: str) -> str:
    """
    Normalize tickers so weights columns and dataset tickers align.
    - strip whitespace
    - uppercase
    - convert '-' -> '.' (e.g., BRK-B -> BRK.B)
    - drop surrounding quotes
    """
    if x is None:
        return ""
    s = str(x).strip().strip('"').strip("'").upper()
    if not s or s in {"NAN", "NONE"}:
        return ""
    if "-" in s and "." not in s:
        s = s.replace("-", ".")
    # Collapse multiple spaces
    s = re.sub(r"\s+", "", s)
    return s


def _normalize_weights_columns(
    weights_df: pd.DataFrame,
    date_col: str,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rename / aggregate asset columns in weights_df using _clean_ticker.

    If multiple original columns map to the same cleaned ticker, we sum them.
    Returns (new_weights_df, cleaned_asset_columns).
    """
    raw_assets = [c for c in weights_df.columns if c != date_col]
    mapping: Dict[str, List[str]] = {}
    for c in raw_assets:
        cc = _clean_ticker(c)
        if not cc:
            continue
        mapping.setdefault(cc, []).append(c)

    if not mapping:
        raise ValueError("No usable asset columns found after ticker normalization.")

    cols_series: List[pd.Series] = []
    for cc, cols in mapping.items():
        if len(cols) == 1:
            s = pd.to_numeric(weights_df[cols[0]], errors="coerce")
        else:
            # Sum duplicates (e.g., multiple share-class spellings)
            s = pd.to_numeric(weights_df[cols], errors="coerce").sum(axis=1, min_count=1)
        cols_series.append(s.rename(cc))

    # Build all columns at once to avoid DataFrame fragmentation warnings / slow inserts
    new_df = pd.concat([weights_df[[date_col]].copy()] + cols_series, axis=1)

    cleaned_assets = list(mapping.keys())
    if verbose:
        dropped = len(raw_assets) - sum(len(v) for v in mapping.values())
        print(f"[normalize] weights: {len(raw_assets)} raw asset cols -> {len(cleaned_assets)} cleaned cols (dropped {dropped})")
    return new_df, cleaned_assets


def _get_monthly_returns_series(
    month_df: pd.DataFrame,
    ticker_col: str,
    date_col: str,
    ret_col: str,
) -> Tuple[pd.Series, pd.Timestamp]:
    """
    From a slice of the dataset for ONE month, return:
      - Series indexed by *cleaned* ticker with that month's return (float)
      - month_end_date: max date observed in that month slice

    We use the last available observation per (month, ticker) to represent that month's return.
    """
    if month_df.empty:
        return pd.Series(dtype=float), pd.NaT

    month_df = month_df.sort_values(date_col)

    last_obs = (
        month_df.groupby(ticker_col, observed=True, sort=False)
        .tail(1)
        [[ticker_col, ret_col, date_col]]
    )

    r = pd.to_numeric(last_obs[ret_col], errors="coerce").to_numpy()
    t = last_obs[ticker_col].astype(str).map(_clean_ticker).to_numpy()
    out = pd.Series(r, index=t, dtype=float)
    out = out[~out.index.duplicated(keep="last")]

    month_end_date = pd.to_datetime(last_obs[date_col], errors="coerce").max()
    return out, month_end_date


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
    strict: bool = False,
    verbose: bool = True,
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
    model_is_none = model in {"none", "null"}

    if weights_path is None:
        weights_path = str(Path(results_dir) / f"{model}_black_litterman_weights_tau_{tau}.csv")
    weights_path = str(weights_path)

    if out_path is None:
        out_path = str(Path(results_dir) / f"{model}_black_litterman_returns_tau_{tau}_{start_date}_{end_date}.csv")
    out_path = str(out_path)

    candidates = [Path(weights_path)]
    if model_raw and model_raw.lower() != model_raw:
        candidates.append(Path(results_dir) / f"{model_raw}_black_litterman_weights_tau_{tau}.csv")
    if model_is_none:
        candidates.extend([
            Path(results_dir) / f"None_black_litterman_weights_tau_{tau}.csv",
            Path(results_dir) / f"none_black_litterman_weights_tau_{tau}.csv",
        ])

    weights_file = next((c for c in candidates if c.exists()), None)
    if weights_file is None:
        msg = f"[{model}] Missing weights file. Tried: " + ", ".join(str(c) for c in candidates)
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(msg, "-> SKIP model")
        return None

    weights_df = pd.read_csv(weights_file)

    weights_df['Date'] = pd.to_datetime(weights_df['Date'], errors="coerce")

    if weights_df['Date'].isna().all():
        raise ValueError(f"[{model}] Could not parse dates in weights file column 'Date'.")

    # Normalize weights dates to month-start
    weights_df['Date'] = weights_df['Date'].dt.to_period("M").dt.to_timestamp()
    # Normalize / aggregate asset columns by cleaned ticker
    weights_df, asset_columns = _normalize_weights_columns(weights_df, date_col="Date", verbose=False)

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
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month", str(weight_date.date()))
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
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month", str(weight_date.date()))
            continue

        # Select weights row for weight month
        row = weights_df.loc[weights_df['Date'] == weight_date]
        if row.empty:
            msg = f"[{model}] No weights found for {weight_date.date()} in {weights_file.name}"
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month")
            continue

        w = row[asset_columns].iloc[0].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # Align assets between weights and available returns
        r = r_series.reindex(asset_columns).to_numpy(dtype=float)
        finite = np.isfinite(r)

        if not finite.any():
            msg = f"[{model}] No finite returns for target month {target_ym} after alignment."
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month", str(weight_date.date()))
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
            if verbose:
                print(f"[{model}] Warning: weights not investable for {weight_date.date()} (sum≈0). "
                      f"Using fallback normalization for target {target_ym}.")
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
        msg = f"[{model}] No returns computed for the requested period."
        if strict:
            raise ValueError(msg)
        if verbose:
            print(msg)
        return None

    out = pd.DataFrame(out_rows).sort_values("date_key").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[{model}] Saved: {out_path} (rows={len(out)})")
    return out


# -------------------------
# CLI
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute portfolio returns from BL weights + dataset returns.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names (e.g., gpt gemma3 None).")
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2025-06-30")
    parser.add_argument("--dataset_path", type=str, default="data/filtered_sp500_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--apply_next_month", action="store_true", help="Apply weights at month M to returns of month M+1.")
    parser.add_argument("--no_apply_next_month", action="store_true", help="Apply weights at month M to returns of month M.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    apply_next_month = True
    if args.no_apply_next_month:
        apply_next_month = False
    elif args.apply_next_month:
        apply_next_month = True

    verbose = not args.quiet

    for m in args.models:
        calculate_model_returns(
            model_name=m,
            tau=args.tau,
            start_date=args.start_date,
            end_date=args.end_date,
            dataset_path=args.dataset_path,
            results_dir=args.results_dir,
            apply_next_month=apply_next_month,
            strict=args.strict,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
