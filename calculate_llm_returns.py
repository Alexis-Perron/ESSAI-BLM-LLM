import argparse
import calendar
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# -------------------------
# Date helpers
# -------------------------
def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _detect_date_col(df: pd.DataFrame) -> str:
    """
    Detects a date-like column in a returns/weights dataframe.
    Accepts: 'date_key', 'Date', or monthly 'ym'.
    Also handles: first column like 'Unnamed: 0' produced by to_csv(index=True).
    """
    if "date_key" in df.columns:
        return "date_key"
    if "Date" in df.columns:
        return "Date"
    if "ym" in df.columns:
        return "ym"

    # Common case: index written to CSV -> first column 'Unnamed: 0'
    if len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        df.rename(columns={df.columns[0]: "date_key"}, inplace=True)
        return "date_key"

    raise ValueError(f"Could not find a date column. Columns are: {list(df.columns)}")


def _to_month_start(s: pd.Series) -> pd.Series:
    """
    Convert a column to month start timestamps:
      - if values look like 'YYYY-MM' -> interpret as YYYY-MM-01
      - else -> pd.to_datetime directly
    """
    ss = s.astype(str).str.strip()
    looks_yyyymm = ss.str.match(r"^\d{4}-\d{2}$")
    out = pd.Series(pd.NaT, index=s.index)

    if looks_yyyymm.any():
        out.loc[looks_yyyymm] = pd.to_datetime(ss.loc[looks_yyyymm] + "-01", errors="coerce")
    out.loc[~looks_yyyymm] = pd.to_datetime(ss.loc[~looks_yyyymm], errors="coerce")
    return out


def _month_starts_between(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Return month-start timestamps between start_date and end_date inclusive.
    """
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    return pd.date_range(start=s, end=e, freq="MS")


# -------------------------
# Core computation
# -------------------------
def calculate_model_returns(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    returns_dir: str = "yfinance",
    results_dir: str = "results",
    weights_path: Optional[str] = None,
    out_path: Optional[str] = None,
    apply_next_month: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute portfolio returns for a given model from BL weights + realized returns.

    Assumptions / conventions:
    - weights file default:
        results/{model_name}_black_litterman_weights_tau_{tau}.csv
    - returns files:
        {returns_dir}/returns_<YYYY-MM-01>_<YYYY-MM-lastday>.csv
      Each returns file may contain:
        - 'ym' (YYYY-MM) + tickers, OR
        - 'date_key'/'Date' (dates) + tickers
      We standardize and filter to the target month only.

    apply_next_month:
      - True  => weights month M applied to returns of month M+1 (your previous convention)
      - False => weights month M applied to returns of month M (same month)

    strict:
      - If True, any missing month/file raises immediately.
      - If False, missing items are skipped (recommended for multi-model batch runs).
    """
    model = model_name.strip().lower()

    if weights_path is None:
        weights_path = str(Path(results_dir) / f"{model}_black_litterman_weights_tau_{tau}.csv")
    weights_path = str(weights_path)

    if out_path is None:
        out_path = str(Path(results_dir) / f"{model}_black_litterman_returns_tau_{tau}_{start_date}_{end_date}.csv")
    out_path = str(out_path)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        msg = f"[{model}] Missing weights file: {weights_file}"
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(msg, "-> SKIP model")
        return None

    weights_df = pd.read_csv(weights_file)

    # Detect date column in weights and parse to datetime (month-start)
    w_date_col = _detect_date_col(weights_df)
    if w_date_col == "ym":
        weights_df[w_date_col] = _to_month_start(weights_df[w_date_col])
    else:
        weights_df[w_date_col] = pd.to_datetime(weights_df[w_date_col], errors="coerce")

    if weights_df[w_date_col].isna().all():
        raise ValueError(f"[{model}] Could not parse dates in weights file column '{w_date_col}'.")

    # Ensure weights dates are normalized to month-start
    weights_df[w_date_col] = weights_df[w_date_col].dt.to_period("M").dt.to_timestamp()

    # Asset columns = all except date col
    asset_columns = [c for c in weights_df.columns if c != w_date_col]
    if not asset_columns:
        raise ValueError(f"[{model}] No asset columns found in weights CSV (only date column present).")

    month_starts = _month_starts_between(start_date, end_date)
    out_parts = []

    for m0 in month_starts:
        weight_date = m0.to_period("M").to_timestamp()  # month-start

        # Determine target month for realized returns
        if apply_next_month:
            target = (m0 + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
        else:
            target = m0.to_period("M").to_timestamp()

        target_year = int(target.year)
        target_month = int(target.month)

        returns_start = f"{target_year}-{target_month:02d}-01"
        returns_end = f"{target_year}-{target_month:02d}-{get_last_day_of_month(target_year, target_month):02d}"
        returns_file = Path(returns_dir) / f"returns_{returns_start}_{returns_end}.csv"

        if not returns_file.exists():
            msg = f"[{model}] Missing returns file: {returns_file}"
            if strict:
                raise FileNotFoundError(msg)
            if verbose:
                print(msg, "-> SKIP month", str(weight_date.date()))
            continue

        future_data = pd.read_csv(returns_file)

        # Standardize returns date info
        r_date_col = _detect_date_col(future_data)

        if r_date_col == "ym":
            future_data["date_key"] = _to_month_start(future_data["ym"])
            future_data["ym"] = future_data["date_key"].dt.to_period("M").astype(str)
        else:
            future_data["date_key"] = pd.to_datetime(future_data[r_date_col], errors="coerce")
            future_data["ym"] = future_data["date_key"].dt.to_period("M").astype(str)

        # Filter to ONLY the target month
        target_ym = pd.Period(target, freq="M").strftime("%Y-%m")
        future_data = future_data.loc[future_data["ym"] == target_ym].copy()

        if future_data.empty:
            msg = f"[{model}] Returns file has no rows for target month {target_ym}: {returns_file}"
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month", str(weight_date.date()))
            continue

        # Select weights row for weight month
        row = weights_df.loc[weights_df[w_date_col] == weight_date]
        if row.empty:
            msg = f"[{model}] No weights found for {weight_date.date()} in {weights_file.name}"
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month")
            continue

        w = row[asset_columns].iloc[0].astype(float)

        # Align assets
        common_assets = [c for c in asset_columns if c in future_data.columns]
        if not common_assets:
            msg = (
                f"[{model}] No common assets between weights and returns for {target_ym}.\n"
                f"  weights sample: {asset_columns[:10]}\n"
                f"  returns sample: {list(future_data.columns)[:10]}"
            )
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month")
            continue

        w_common = w[common_assets].to_numpy(dtype=float)
        r_common = future_data[common_assets].astype(float).to_numpy()

        # Renormalize weights over available assets
        s = np.nansum(w_common)
        if not np.isfinite(s) or abs(s) < 1e-12:
            msg = f"[{model}] Weights sum invalid (~0/NaN) for {weight_date.date()} after alignment."
            if strict:
                raise ValueError(msg)
            if verbose:
                print(msg, "-> SKIP month")
            continue
        w_common = w_common / s

        # Portfolio return (for monthly data, this should be a single row)
        port_ret = np.nansum(r_common * w_common.reshape(1, -1), axis=1)

        part = pd.DataFrame(
            {
                "date_key": future_data["date_key"],
                "Portfolio_Return": port_ret,
            }
        )
        out_parts.append(part)

    if not out_parts:
        msg = f"[{model}] No returns computed for the requested period."
        if strict:
            raise RuntimeError(msg)
        if verbose:
            print(msg)
        return None

    out_df = pd.concat(out_parts, ignore_index=True).sort_values("date_key")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if verbose:
        print(f"[{model}] Saved: {out_path} (rows={len(out_df)})")

    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gpt gemma3_1b qwen"], help="List of model names (e.g., gpt gemma qwen).")
    p.add_argument("--tau", type=float, default=0.025)
    p.add_argument("--start", type=str, default="2021-01-01")
    p.add_argument("--end", type=str, default="2023-06-30")
    p.add_argument("--returns_dir", type=str, default="yfinance")
    p.add_argument("--results_dir", type=str, default="results")

    # Convention control:
    # True: weights month M -> returns month M+1
    # False: weights month M -> returns month M
    p.add_argument("--apply_next_month", action="store_true", help="Apply weights M to returns of M+1 (default True).")
    p.add_argument("--apply_same_month", action="store_true", help="Apply weights M to returns of M (overrides).")

    p.add_argument("--strict", action="store_true", help="Fail fast on missing files/months.")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")

    args = p.parse_args()

    apply_next = True
    if args.apply_same_month:
        apply_next = False
    elif args.apply_next_month:
        apply_next = True  # explicit

    verbose = not args.quiet

    # Run each model sequentially
    for model_name in args.models:
        calculate_model_returns(
            model_name=model_name,
            tau=float(args.tau),
            start_date=args.start,
            end_date=args.end,
            returns_dir=args.returns_dir,
            results_dir=args.results_dir,
            apply_next_month=apply_next,
            strict=bool(args.strict),
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
