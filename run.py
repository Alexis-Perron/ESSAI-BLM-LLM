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
    """JSON serializer for numpy / pandas types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (pd.Period,)):
        return str(o)
    return str(o)


def make_system_prompt() -> str:
    """
    Prompt for structured numeric output.

    NOTE: In this project, 'stock_ret' is a MONTHLY return (decimal, e.g. 0.02 = +2%).
    We provide a short history of monthly returns and ask for an expected NEXT-month return.
    """
    return (
        "You are a model designed to predict stock returns. "
        "Given a time-series of PAST MONTHLY returns (decimal returns, e.g. 0.02 = +2%), "
        "and the company metadata provided, predict the expected MONTHLY return (decimal) for the NEXT month. "
        "Return ONLY valid JSON that matches this schema: {\"expected_return\": number}. "
        "Do not include any extra keys or text."
    )


def make_user_prompt(ticker: str, row: dict) -> str:
    """Build the user prompt from metadata + return history."""
    return (
        f"Ticker: {ticker}\n"
        f"Company: {row.get('company_name', '')}\n"
        f"Sector (GICS): {row.get('gics_sector_name', '')}\n"
        f"GICS code: {row.get('gics', '')}\n"
        f"SIC: {row.get('sic', '')}\n"
        f"NAICS: {row.get('naics', '')}\n"
        f"Market equity: {row.get('market_equity', '')}\n"
        f"past_monthly_returns: {row.get('past_returns', [])}"
    )


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    """Normalize tickers to uppercase, trim, and unify separators."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", ".", regex=False)
    )


def parse_yyyymmdd_int_to_datetime(s: pd.Series) -> pd.Series:
    """Parse an int-like YYYYMMDD series into datetime."""
    # Keep only digits, then parse with fixed format.
    ss = s.astype("Int64").astype(str).str.replace(r"\D+", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(ss, format="%Y%m%d", errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt")  # used for output filename prefix
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input_csv", type=str, default="yfinance/filtered_sp500_data.csv")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2021-06-30")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lookback_months", type=int, default=1, help="How many past months of returns to include (per ticker).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute months even if output json already exists.")
    args = parser.parse_args()

    if args.model_name != "gpt":
        raise ValueError("This script currently supports only --model_name=gpt")

    os.makedirs("responses", exist_ok=True)

    client = OpenAI(api_key=gpt_key)

    sp500_table = pd.read_csv(args.input_csv, low_memory=False)

    # Required columns
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
    # Prefer descriptive sector name if available
    sector_name_col = "gics_sector_name" if "gics_sector_name" in sp500_table.columns else None
    gics_col = "gics" if "gics" in sp500_table.columns else None
    sic_col = "sic" if "sic" in sp500_table.columns else None
    naics_col = "naics" if "naics" in sp500_table.columns else None
    me_col = "market_equity" if "market_equity" in sp500_table.columns else None

    system_prompt = make_system_prompt()

    # Month iteration (month starts)
    date_range = pd.date_range(start=args.start, end=args.end, freq="MS")

    for current_date in tqdm(date_range):
        # current month label
        y = current_date.year
        m = current_date.month
        current_p = pd.Period(f"{y}-{m:02d}", freq="M")

        month_start_dt = pd.Timestamp(current_date).normalize()
        month_end_dt = (month_start_dt + pd.DateOffset(months=1) - timedelta(days=1)).normalize()
        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = month_end_dt.strftime("%Y-%m-%d")

        out_path = f"responses/{args.model_name}_{month_start}_{month_end}.json"
        if (not args.overwrite) and os.path.exists(out_path):
            # Skip if already computed
            continue

        print(f"Processing data for {month_start} - {month_end}")

        # Rows in the current month
        mdf = sp500_table.loc[sp500_table["ym"] == current_p].copy()
        if mdf.empty:
            print(f"[WARN] No rows found for {current_p}. Writing empty file.")
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        # Lookback slice for returns history
        lb = max(int(args.lookback_months), 1)
        hist_start_p = current_p - (lb - 1)
        hdf = sp500_table.loc[(sp500_table["ym"] >= hist_start_p) & (sp500_table["ym"] <= current_p),
                              ["ym", "tic", "stock_ret"]].copy()

        # Build per-ticker history dict: list of returns ordered by month
        hdf = hdf.dropna(subset=["stock_ret"])
        hdf = hdf.sort_values(["tic", "ym"])
        hist_map = hdf.groupby("tic")["stock_ret"].apply(list).to_dict()

        # Build a dictionary per ticker: returns list + metadata
        data_dict: dict[str, dict] = {}
        for tic, g in mdf.groupby("tic", sort=False):
            past_returns = hist_map.get(tic, [])
            row = {
                "company_name": g[company_col].iloc[0] if company_col else "",
                "gics_sector_name": g[sector_name_col].iloc[0] if sector_name_col else "",
                "gics": float(g[gics_col].iloc[0]) if (gics_col and pd.notna(g[gics_col].iloc[0])) else "",
                "sic": float(g[sic_col].iloc[0]) if (sic_col and pd.notna(g[sic_col].iloc[0])) else "",
                "naics": float(g[naics_col].iloc[0]) if (naics_col and pd.notna(g[naics_col].iloc[0])) else "",
                "market_equity": float(g[me_col].iloc[0]) if (me_col and pd.notna(g[me_col].iloc[0])) else "",
                "past_returns": past_returns,
            }
            data_dict[tic] = row

        tickers = sorted(data_dict.keys())

        # Inference per ticker
        for ticker in tqdm(tickers, leave=False):
            user_prompt = make_user_prompt(ticker, data_dict[ticker])

            answers: list[float] = []
            for _ in range(int(args.n_samples)):
                try:
                    response = client.responses.parse(
                        model=args.openai_model,
                        instructions=system_prompt,
                        input=user_prompt,
                        text_format=ResearchPaperExtraction,
                        temperature=float(args.temperature),
                    )
                    parsed = response.output_parsed
                    answers.append(float(parsed.expected_return))
                except Exception:
                    # If parsing/validation fails, skip this sample
                    continue

            data_dict[ticker]["expected_return"] = answers  # keep raw samples for analysis
            data_dict[ticker]["n_success"] = int(len(answers))
            data_dict[ticker]["expected_return_mean"] = float(np.mean(answers)) if answers else None

        # Save responses for this month
        with open(out_path, "w") as f:
            json.dump(data_dict, f, default=json_default)


if __name__ == "__main__":
    main()
