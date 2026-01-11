import pandas as pd
import numpy as np
import calendar
from pathlib import Path


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


def calculate_gpt_returns(
    tau: float = 0.025,
    weights_year: int = 2021,
    start_month: int = 1,
    end_month: int = 6,
    weights_path: str | None = None,
    returns_dir: str = "yfinance",
    out_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute GPT portfolio returns from Black-Litterman weights.

    Convention:
      - weights are for month M (YYYY-MM-01)
      - applied to returns of month M+1

    Robust to date column names:
      - weights: 'Date' or 'date_key' (must be parseable as datetime)
      - returns: 'ym' (YYYY-MM), or 'date_key'/'Date', or unnamed index col

    Also aligns tickers by intersection and renormalizes weights over common tickers.
    """
    if weights_path is None:
        weights_path = f"results/gpt_black_litterman_weights_tau_{tau}.csv"

    weights_df = pd.read_csv(weights_path)

    # Detect date column in weights and parse to datetime
    w_date_col = _detect_date_col(weights_df)
    if w_date_col == "ym":
        # if ever weights are monthly keys like YYYY-MM
        weights_df[w_date_col] = _to_month_start(weights_df[w_date_col])
    else:
        weights_df[w_date_col] = pd.to_datetime(weights_df[w_date_col], errors="coerce")

    if weights_df[w_date_col].isna().all():
        raise ValueError(f"Could not parse dates in weights file column '{w_date_col}'.")

    # Asset columns = all except date col
    asset_columns = [c for c in weights_df.columns if c != w_date_col]
    if not asset_columns:
        raise ValueError("No asset columns found in weights CSV (only date column present).")

    gpt_monthly_returns = []

    for month in range(start_month, end_month + 1):
        weight_date = pd.Timestamp(year=weights_year, month=month, day=1)

        # Returns are for next month
        returns_month = month + 1
        returns_year = weights_year
        if returns_month == 13:
            returns_month = 1
            returns_year += 1

        returns_start = f"{returns_year}-{returns_month:02d}-01"
        returns_end = f"{returns_year}-{returns_month:02d}-{get_last_day_of_month(returns_year, returns_month):02d}"
        returns_file = Path(returns_dir) / f"returns_{returns_start}_{returns_end}.csv"

        if not returns_file.exists():
            raise FileNotFoundError(f"Missing returns file: {returns_file}")

        future_data = pd.read_csv(returns_file)

        # Detect date column in returns and standardize to 'date_key' month-start timestamps
        r_date_col = _detect_date_col(future_data)

        if r_date_col == "ym":
            # ym is like '2021-02' -> convert to 2021-02-01
            future_data["date_key"] = _to_month_start(future_data["ym"])
            future_data["ym"] = future_data["date_key"].dt.to_period("M").astype(str)
        else:
            future_data["date_key"] = pd.to_datetime(future_data[r_date_col], errors="coerce")
            future_data["ym"] = future_data["date_key"].dt.to_period("M").astype(str)

        # Filter to ONLY the target month (avoid repeated window problem)
        target_ym = pd.Period(pd.to_datetime(returns_start), freq="M").strftime("%Y-%m")
        future_data = future_data.loc[future_data["ym"] == target_ym].copy()

        if future_data.empty:
            raise ValueError(
                f"Returns file has no rows for target month {target_ym}. File: {returns_file}"
            )

        # Find the weights row for this month
        row = weights_df.loc[weights_df[w_date_col] == weight_date]
        if row.empty:
            near = weights_df[w_date_col].dropna().sort_values().tail(8).astype(str).tolist()
            raise ValueError(
                f"No weights found for {weight_date.date()} in '{weights_path}'. "
                f"Date col='{w_date_col}'. Example dates in file: {near}"
            )

        w = row[asset_columns].iloc[0].astype(float)

        # Align assets (intersection)
        common_assets = [c for c in asset_columns if c in future_data.columns]
        if not common_assets:
            raise ValueError(
                f"No common asset columns between weights and returns.\n"
                f"Weights assets sample: {asset_columns[:10]}\n"
                f"Returns columns sample: {list(future_data.columns)[:10]}"
            )

        w_common = w[common_assets].to_numpy(dtype=float)
        r_common = future_data[common_assets].astype(float).to_numpy()

        # Renormalize weights over common assets
        s = np.nansum(w_common)
        if not np.isfinite(s) or abs(s) < 1e-12:
            raise ValueError(f"Weights sum to ~0 or invalid for {weight_date.date()} after alignment.")
        w_common = w_common / s

        # Portfolio returns
        port_ret = np.nansum(r_common * w_common.reshape(1, -1), axis=1)

        gpt_portfolio = pd.DataFrame({
            "date_key": future_data["date_key"],
            "Portfolio_Return": port_ret
        })

        gpt_monthly_returns.append(gpt_portfolio)

    gpt_all_returns = pd.concat(gpt_monthly_returns, ignore_index=True).sort_values("date_key")

    if out_path is None:
        out_path = f"results/gpt_black_litterman_returns_tau_{tau}_{weights_year}_{start_month:02d}-{end_month:02d}.csv"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gpt_all_returns.to_csv(out_path, index=False)
    return gpt_all_returns


if __name__ == "__main__":
    tau = 0.025
    gpt_returns = calculate_gpt_returns(
        tau=tau,
        weights_year=2021,
        start_month=1,
        end_month=6,
        returns_dir="yfinance",
        # weights_path="results/gpt_black_litterman_weights_tau_0.025.csv",  # optionnel
    )
    print(gpt_returns.head(10))
