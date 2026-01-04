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
    """Make numpy / pandas scalars JSON-serializable."""
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def make_system_prompt() -> str:
    """System prompt aligned with the Pydantic schema used by structured outputs."""
    return (
        "You are a model designed to predict stock returns. "
        "Given a time-series of daily returns from the past month (as decimal returns, e.g. 0.01 = 1%), "
        "and the company metadata provided, predict the average daily return (decimal) for the next month. "
        "Return ONLY valid JSON that matches this schema: {\"expected_return\": number}. "
        "Do not include any extra keys or text."
    )


def make_user_prompt(ticker: str, row: dict) -> str:
    """Build the user prompt from metadata + return history."""
    return (
        f"Ticker: {ticker}\n"
        f"Security: {row.get('Security', '')}\n"
        f"Sector (GICS): {row.get('GICS Sector', '')}\n"
        f"Sub-Industry: {row.get('GICS Sub-Industry', '')}\n"
        f"pct_change: {row.get('pct_change', [])}"
    )


def _parse_yyyymmdd_series_to_datetime(s: pd.Series) -> pd.Series:
    """Parse YYYYMMDD-like numeric/string series to datetime."""
    s_num = pd.to_numeric(s, errors="coerce")
    # If numeric looks like 8-digit yyyymmdd, parse with that format
    if s_num.notna().any() and (s_num.dropna().astype("int64").astype(str).str.len() == 8).all():
        return pd.to_datetime(s_num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    # Fallback to generic parsing
    return pd.to_datetime(s, errors="coerce")


def add_date_key(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a datetime64[ns] column named 'date_key'."""
    # Common date columns we might see
    candidates = ["date", "Date", "datetime", "timestamp", "date_key"]
    src = next((c for c in candidates if c in df.columns), None)
    if src is None:
        raise ValueError(
            "Input table must contain a date column among: " + ", ".join(candidates)
        )

    df = df.copy()
    # Always (re)build date_key from the best available source to avoid dtype issues
    df["date_key"] = _parse_yyyymmdd_series_to_datetime(df[src])

    # Defensive check: if parsing failed entirely, surface a helpful error early
    if df["date_key"].isna().all():
        sample = df[src].head(5).tolist()
        raise ValueError(
            f"Could not parse '{src}' into datetimes. Sample values: {sample}"
        )
    return df


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    """Normalize tickers to uppercase, trim, and unify common separators."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        # Optional: make BRK-B comparable to BRK.B style
        .str.replace("-", ".", regex=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt")
    parser.add_argument("--input_csv", type=str, default="yfinance/filtered_sp500_data.csv")
    parser.add_argument("--start", type=str, default="2024-06-01")
    parser.add_argument("--end", type=str, default="2024-12-01")
    parser.add_argument("--n_samples", type=int, default=30)
    args = parser.parse_args()

    if args.model_name != "gpt":
        raise ValueError("This script currently supports only --model_name=gpt")

    client = OpenAI(api_key=gpt_key)

    os.makedirs("yfinance", exist_ok=True)
    os.makedirs("responses", exist_ok=True)

    # Monthly start dates
    date_range = pd.date_range(start=args.start, end=args.end, freq="MS")

    # --- load data ---
    sp500_table = pd.read_csv(args.input_csv, low_memory=False)
    sp500_table = add_date_key(sp500_table)

    # Identify columns robustly
    ticker_col = "tic" if "tic" in sp500_table.columns else (
        "Ticker" if "Ticker" in sp500_table.columns else (
            "Symbol" if "Symbol" in sp500_table.columns else None
        )
    )
    if ticker_col is None:
        raise ValueError("Could not find a ticker column (expected 'tic' or 'Ticker' or 'Symbol').")

    ret_col = "stock_ret" if "stock_ret" in sp500_table.columns else None

    if ret_col is None:
        raise ValueError("Could not find a return column ('stock_ret').")

    # Metadata columns (optional)
    security_col = "Security" if "Security" in sp500_table.columns else ("conm" if "conm" in sp500_table.columns else None)
    sector_col = "GICS Sector" if "GICS Sector" in sp500_table.columns else ("gics" if "gics" in sp500_table.columns else None)
    subind_col = "GICS Sub-Industry" if "GICS Sub-Industry" in sp500_table.columns else ("sic" if "sic" in sp500_table.columns else None)

    # Normalize tickers + returns
    sp500_table[ticker_col] = normalize_ticker_series(sp500_table[ticker_col])
    sp500_table[ret_col] = pd.to_numeric(sp500_table[ret_col], errors="coerce")

    system_prompt = make_system_prompt()

    # --- loop months ---
    for current_date in tqdm(date_range):
        month_start_dt = pd.Timestamp(current_date).normalize()
        month_end_dt = (month_start_dt + pd.DateOffset(months=1) - timedelta(days=1)).normalize()

        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = month_end_dt.strftime("%Y-%m-%d")
        print(f"Processing data for {month_start} - {month_end}")

        mask = (sp500_table["date_key"] >= month_start_dt) & (sp500_table["date_key"] <= month_end_dt)
        mdf = sp500_table.loc[mask].copy()
        if mdf.empty:
            # still write an empty response file for traceability
            out_path = f"responses/{args.model_name}_{month_start}_{month_end}.json"
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        # Build a dictionary per ticker: returns list + metadata
        data_dict: dict[str, dict] = {}

        for tic, g in mdf.groupby(ticker_col, sort=False):
            # returns history for the month
            pct_series = g[ret_col].dropna().tolist()

            row = {
                "Security": g[security_col].iloc[0] if security_col else "",
                "GICS Sector": g[sector_col].iloc[0] if sector_col else "",
                "GICS Sub-Industry": g[subind_col].iloc[0] if subind_col else "",
                "pct_change": pct_series,
            }
            data_dict[tic] = row

        sp500_tickers = sorted(data_dict.keys())

        # Only do inference on AAPL for debugging / credit saving
        for ticker in tqdm(sp500_tickers, leave=False):
            if ticker != "AAPL":
                continue

            user_prompt = make_user_prompt(ticker, data_dict[ticker])

            answers: list[float] = []
            for _ in range(args.n_samples):
                try:
                    response = client.responses.parse(
                        model="gpt-4o-2024-08-06",
                        instructions=system_prompt,
                        input=user_prompt,
                        text_format=ResearchPaperExtraction,
                        temperature=1.0,
                    )
                    parsed = response.output_parsed
                    answers.append(float(parsed.expected_return))
                except Exception:
                    # If parsing/validation fails, skip this sample
                    continue

            data_dict[ticker]["expected_return"] = answers

        # Save responses for this month
        out_path = f"responses/{args.model_name}_{month_start}_{month_end}.json"
        with open(out_path, "w") as f:
            json.dump(data_dict, f, default=json_default)


if __name__ == "__main__":
    main()
