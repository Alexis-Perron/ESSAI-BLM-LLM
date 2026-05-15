"""utils_returns_metadata_only.py

Utilities for a returns + metadata only LLM inference pipeline.

This file intentionally excludes any `summary_json` / filing-text input from the
prompt. It is meant to be used with `run_returns_metadata_only.py` so that the
outputs can be compared against the original text-enhanced pipeline.
"""
from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd


MODEL_MAP = {
    "gemma3": "gemma3",
    "qwen": "qwen2.5:1.5b",
    "gpt": "gpt-4o-mini",
    "llama": "llama3.2",
}


def json_default(o: Any):
    """JSON serializer for NumPy / pandas-friendly objects."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    """Normalize tickers so prompts and output keys are consistent."""
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("-", ".", regex=False)
    return s


def parse_yyyymmdd_int_to_datetime(s: pd.Series) -> pd.Series:
    """Robust parse for YYYYMMDD stored as int, float, or string."""
    ss = s.astype("Int64").astype(str).str.replace(r"\D+", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(ss, format="%Y%m%d", errors="coerce")


def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


# -------------------------
# Prompts: returns + metadata only
# -------------------------
def make_system_prompt_returns_metadata_only() -> str:
    return (
        "You are a model designed to predict stock returns. "
        "Given a time-series of PAST MONTHLY returns (decimal returns, e.g. 0.02 = +2%) "
        "and company metadata, predict the expected MONTHLY return (decimal) for the NEXT month "
        "using only the provided data. "
        "Do not use or assume any filing text, news text, management discussion, risk factors, "
        "or summarized textual payload. "
        "If you would output a percent (e.g., 2 for 2%), convert it to decimal (0.02). "
        "Expected return should be a decimal in the range [-1, 1]. "
        'Return ONLY valid JSON that matches this schema: {"expected_return": number}. '
        "Do not include any extra keys or text."
    )


def make_user_prompt_returns_metadata_only(ticker: str, row: dict) -> str:
    """Build the user prompt without `summary_json` or any textual filing content."""
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
