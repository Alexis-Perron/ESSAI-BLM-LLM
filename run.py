import argparse
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from keys import gpt_key


class ResearchPaperExtraction(BaseModel):
    """Structured output schema returned by the model."""
    expected_return: float


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    # Standardize tickers. yfinance usually uses "." instead of "-" for class shares.
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("-", ".", regex=False)
    return s


def parse_yyyymmdd_int_to_datetime(s: pd.Series) -> pd.Series:
    """
    Robustly parse integer-like YYYYMMDD in a column that might be read as float/int/str.
    Example: 20210129 -> 2021-01-29.
    """
    ss = s.astype("Int64").astype(str).str.replace(r"\D+", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(ss, format="%Y%m%d", errors="coerce")


def make_system_prompt() -> str:
    return (
        "You are a model designed to predict stock returns. "
        "Given a time-series of PAST MONTHLY returns (decimal returns, e.g. 0.02 = +2%), "
        "company metadata, and optionally a recent summarized filing payload (summary_json), "
        "predict the expected MONTHLY return (decimal) for the NEXT month. "
        "Return ONLY valid JSON that matches this schema: {\"expected_return\": number}. "
        "Do not include any extra keys or text."
    )


def make_user_prompt(ticker: str, row: dict) -> str:
    """Build the user prompt from metadata + return history + optional filing summary."""
    filing_json = row.get("summary_json", "")
    filing_json = "" if filing_json is None else str(filing_json)
    filing_json = "" if filing_json.strip().lower() in {"nan", "none"} else filing_json

    return (
        f"Ticker: {ticker}\n"
        f"Company: {row.get('company_name', '')}\n"
        f"Sector (GICS): {row.get('gics_sector_name', '')}\n"
        f"GICS code: {row.get('gics', '')}\n"
        f"SIC: {row.get('sic', '')}\n"
        f"NAICS: {row.get('naics', '')}\n"
        f"Market equity: {row.get('market_equity', '')}\n"
        f"past_monthly_returns: {row.get('past_returns', [])}\n"
        f"summary_json: {filing_json}"
    )


# =========================
# NEW: returns CSV builder
# =========================
def build_returns_matrix(
    sp500_table: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a MONTHLY returns matrix (1 row per month).
      - index = ym (YYYY-MM)
      - columns = tickers
      - values = stock_ret (monthly returns)
    We keep the last available date within each month per ticker.
    """
    df = sp500_table.loc[
        (sp500_table["date_key"] >= window_start) & (sp500_table["date_key"] <= window_end),
        ["date_key", "tic", "stock_ret"],
    ].copy()

    df = df.dropna(subset=["date_key", "tic"])
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")

    # Month key
    df["ym"] = df["date_key"].dt.to_period("M")

    # Keep last row within each (ticker, month)
    df = df.sort_values(["tic", "date_key"]).drop_duplicates(subset=["tic", "ym"], keep="last")

    # Pivot: one row per month
    mat = df.pivot(index="ym", columns="tic", values="stock_ret").sort_index()

    # Optional: write index as string "YYYY-MM" in CSV
    mat.index = mat.index.astype(str)

    return mat



def ensure_returns_csv_for_period(
    sp500_table: pd.DataFrame,
    period_start: str,
    period_end: str,
    out_dir: str,
    window_mode: str,
    global_start: str,
    global_end: str,
    lookback_months: int,
) -> str:
    """
    Create the file expected by evaluate_multiple_updated.py:
        yfinance/returns_<period_start>_<period_end>.csv

    window_mode:
      - "full": use [global_start, global_end] for ALL periods (recommended to avoid variance=0 early)
      - "expanding": use [global_start, period_end]
      - "rolling": use last `lookback_months` months ending at period_end
      - "period": only [period_start, period_end] (NOT recommended; may give variance=0)
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"returns_{period_start}_{period_end}.csv")
    if os.path.exists(out_path):
        return out_path

    p_start = pd.to_datetime(period_start)
    p_end = pd.to_datetime(period_end)

    data_min = sp500_table["date_key"].min()
    data_max = sp500_table["date_key"].max()

    g_start = pd.to_datetime(global_start)
    g_end = pd.to_datetime(global_end)

    if window_mode == "full":
        w_start, w_end = g_start, g_end
    elif window_mode == "expanding":
        w_start, w_end = g_start, p_end
    elif window_mode == "rolling":
        # rolling months window ending at p_end
        w_end = p_end
        w_start = (p_end.to_period("M") - (max(1, int(lookback_months)) - 1)).to_timestamp()
    elif window_mode == "period":
        w_start, w_end = p_start, p_end
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    # clamp to available data
    w_start = max(w_start, data_min)
    w_end = min(w_end, data_max)

    mat = build_returns_matrix(sp500_table, w_start, w_end)

    # If extremely short window (e.g., January 2021 expanding/rolling), you can still end with too few rows.
    # To keep evaluate_multiple_updated.py from crashing (market_var==0), we do a pragmatic fallback:
    # if <2 rows, switch to FULL sample window inside [global_start, global_end].
    if mat.shape[0] < 2:
        w_start2, w_end2 = max(g_start, data_min), min(g_end, data_max)
        mat = build_returns_matrix(sp500_table, w_start2, w_end2)

    mat.to_csv(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt")  # used for output filename prefix
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input_csv", type=str, default="yfinance/filtered_sp500_data.csv")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2021-07-30")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--summary_json_max_chars",
        type=int,
        default=0,
        help="Max characters of summary_json to include in prompt (0 disables).",
    )
    parser.add_argument("--lookback_months", type=int, default=1, help="How many past months of returns to include.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute months even if output json already exists.")

    # NEW: returns csv generation controls
    parser.add_argument("--returns_out_dir", type=str, default="yfinance")
    parser.add_argument(
        "--returns_window_mode",
        type=str,
        default="full",
        choices=["full", "expanding", "rolling", "period"],
        help="How to build returns_*.csv from filtered_sp500_data.csv.",
    )
    parser.add_argument("--returns_lookback_months", type=int, default=24, help="Used only for rolling mode.")

    args = parser.parse_args()

    if args.model_name != "gpt":
        raise ValueError("This script currently supports only --model_name=gpt")

    os.makedirs("responses", exist_ok=True)

    client = OpenAI(api_key=gpt_key)

    sp500_table = pd.read_csv(args.input_csv, low_memory=False)

    required_cols = {"date", "year", "month", "tic", "stock_ret"}
    missing = required_cols - set(sp500_table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {sorted(missing)}")

    # Robust dtypes
    sp500_table["year"] = pd.to_numeric(sp500_table["year"], errors="coerce").astype("Int64")
    sp500_table["month"] = pd.to_numeric(sp500_table["month"], errors="coerce").astype("Int64")
    sp500_table["stock_ret"] = pd.to_numeric(sp500_table["stock_ret"], errors="coerce")

    # Parse date (YYYYMMDD int)
    sp500_table["date_key"] = parse_yyyymmdd_int_to_datetime(sp500_table["date"])
    sp500_table = sp500_table.dropna(subset=["date_key", "year", "month", "tic"])

    # Monthly period
    sp500_table["ym"] = pd.to_datetime(
        dict(year=sp500_table["year"].astype(int), month=sp500_table["month"].astype(int), day=1)
    ).dt.to_period("M")

    # Normalize tickers
    sp500_table["tic"] = normalize_ticker_series(sp500_table["tic"])

    # Optional metadata columns
    company_col = "conm" if "conm" in sp500_table.columns else None
    sector_name_col = "gics_sector_name" if "gics_sector_name" in sp500_table.columns else None
    gics_col = "gics" if "gics" in sp500_table.columns else None
    sic_col = "sic" if "sic" in sp500_table.columns else None
    naics_col = "naics" if "naics" in sp500_table.columns else None
    market_equity_col = "market_equity" if "market_equity" in sp500_table.columns else None
    summary_json_col = "summary_json" if "summary_json" in sp500_table.columns else None

    # Build month iterator from args.start to args.end
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    month_starts = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    system_prompt = make_system_prompt()

    for month_start_dt in tqdm(month_starts, total=len(month_starts)):
        month_end_dt = (month_start_dt + pd.offsets.MonthEnd(1)).to_pydatetime()
        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = pd.Timestamp(month_end_dt).strftime("%Y-%m-%d")

        out_path = f"responses/{args.model_name}_{month_start}_{month_end}.json"
        if (not args.overwrite) and os.path.exists(out_path):
            # Still ensure returns csv exists (so evaluate_multiple_updated.py won’t crash)
            ensure_returns_csv_for_period(
                sp500_table=sp500_table,
                period_start=month_start,
                period_end=month_end,
                out_dir=args.returns_out_dir,
                window_mode=args.returns_window_mode,
                global_start=args.start,
                global_end=args.end,
                lookback_months=args.returns_lookback_months,
            )
            continue

        # NEW: create required returns csv for this period
        ensure_returns_csv_for_period(
            sp500_table=sp500_table,
            period_start=month_start,
            period_end=month_end,
            out_dir=args.returns_out_dir,
            window_mode=args.returns_window_mode,
            global_start=args.start,
            global_end=args.end,
            lookback_months=args.returns_lookback_months,
        )

        # Select month rows for metadata snapshot
        current_p = month_start_dt.to_period("M")
        mdf = sp500_table.loc[sp500_table["ym"] == current_p].copy()
        if mdf.empty:
            # No data for that month
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        # Lookback slice for returns history
        lb = max(int(args.lookback_months), 1)
        hist_start_p = current_p - (lb - 1)
        hdf = sp500_table.loc[
            (sp500_table["ym"] >= hist_start_p) & (sp500_table["ym"] <= current_p),
            ["ym", "tic", "stock_ret"],
        ].copy()

        hdf = hdf.dropna(subset=["stock_ret"])
        hdf = hdf.sort_values(["tic", "ym"])
        hist_map = hdf.groupby("tic")["stock_ret"].apply(list).to_dict()

        # Build a dictionary per ticker: returns list + metadata
        data_dict: dict[str, dict] = {}
        for tic, g in mdf.groupby("tic", sort=False):
            past_returns = hist_map.get(tic, [])
            summary_json_val = g[summary_json_col].iloc[0] if summary_json_col else ""
            if pd.isna(summary_json_val):
                summary_json_val = ""
            else:
                summary_json_val = str(summary_json_val)

            maxc = int(getattr(args, "summary_json_max_chars", 0) or 0)
            if maxc > 0 and len(summary_json_val) > maxc:
                half = maxc // 2
                summary_json_val = summary_json_val[:half] + "\n...\n" + summary_json_val[-half:]

            row = {
                "company_name": g[company_col].iloc[0] if company_col else "",
                "gics_sector_name": g[sector_name_col].iloc[0] if sector_name_col else "",
                "gics": g[gics_col].iloc[0] if gics_col else "",
                "sic": g[sic_col].iloc[0] if sic_col else "",
                "naics": g[naics_col].iloc[0] if naics_col else "",
                "market_equity": float(g[market_equity_col].iloc[0]) if market_equity_col else "",
                "past_returns": [float(x) for x in past_returns if pd.notna(x)],
                "summary_json": summary_json_val,
            }
            data_dict[tic] = row

        # Call LLM per ticker
        for ticker in tqdm(list(data_dict.keys()), desc=f"LLM {month_start}->{month_end}", leave=False):
            user_prompt = make_user_prompt(ticker, data_dict[ticker])
            answers = []
            for _ in range(int(args.n_samples)):
                try:
                    resp = client.chat.completions.create(
                        model=args.openai_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=float(args.temperature),
                    )
                    content = resp.choices[0].message.content
                    parsed = json.loads(content)
                    # Validate schema
                    obj = ResearchPaperExtraction(**parsed)
                    answers.append(float(obj.expected_return))
                except Exception:
                    continue

            data_dict[ticker]["expected_return"] = answers
            data_dict[ticker]["n_success"] = int(len(answers))
            data_dict[ticker]["expected_return_mean"] = float(np.mean(answers)) if answers else None

        with open(out_path, "w") as f:
            json.dump(data_dict, f, default=json_default)


if __name__ == "__main__":
    main()
