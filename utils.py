from __future__ import annotations

import numpy as np
import pandas as pd
import calendar
import json
from pydantic import BaseModel, Field
import random
import sys
import time
import hashlib
import re
from typing import Any, Optional

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from scipy.optimize import minimize
from typing import Dict, Tuple, List

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
run.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

MODEL_MAP = {
    "gemma3": "gemma3",
    "qwen": "qwen2.5:1.5b",
    "gpt": "gpt-4o-mini",
    "llama": "llama3.2",
}



# -------------------------
# Utils
# -------------------------
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def normalize_ticker_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("-", ".", regex=False)
    return s

def parse_yyyymmdd_int_to_datetime(s: pd.Series) -> pd.Series:
    """
    This function was written by ChatGPT 5.2
    Robust parse for YYYYMMDD stored as int/float/str.
    """
    ss = s.astype("Int64").astype(str).str.replace(r"\D+", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(ss, format="%Y%m%d", errors="coerce")

def get_last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]



# -------------------------
# Prompts
# -------------------------
def make_system_prompt() -> str:
    return (
        "You are a model designed to predict stock returns. "
        "Given a time-series of PAST MONTHLY returns (decimal returns, e.g. 0.02 = +2%), "
        "company metadata, and optionally a recent summarized filing payload (summary_json), "
        "predict the expected MONTHLY return (decimal) for the NEXT month using only the provided data."
        "If you would output a percent (e.g., 2 for 2%), convert it to decimal (0.02). "
        "Expected return should be a decimal in the range [-1, 1]. "
        'Return ONLY valid JSON that matches this schema: {"expected_return": number}. '
        "Do not include any extra keys or text."
    )

def make_user_prompt(ticker: str, row: dict) -> str:
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



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
summarize_text_reports.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# -----------------------------
# Compatibility for some pickles
# -----------------------------
def _ensure_numpy_core_alias() -> None:
    """
    Some pickles reference numpy._core (NumPy 2.x layout). If running NumPy 1.x,
    create aliases so unpickling works.
    """
    try:
        import numpy.core as npcore  # noqa
        sys.modules.setdefault("numpy._core", npcore)
        try:
            import numpy.core._multiarray_umath as mau  # noqa
            sys.modules.setdefault("numpy._core._multiarray_umath", mau)
        except Exception:
            pass
    except Exception:
        pass


# -----------------------------
# Structured output schema
# -----------------------------
class StructuredSummary(BaseModel):
    summary: str = Field(..., description="150-200 word factual summary of the most material points.")
    bullish_points: list[str] = Field(..., description="2-5 short bullets (<= 18 words) that could be positive for returns.")
    bearish_points: list[str] = Field(..., description="2-5 short bullets (<= 18 words) that could be negative for returns.")
    guidance_change: str = Field(..., description="One of: up, down, none, unknown")
    risk_level: int = Field(..., ge=1, le=5, description="1 (low) to 5 (high) risk based on disclosed risks.")


def _normalize_whitespace(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def _trim_text(x: Any, max_chars: int, trim_mode: str = "headtail") -> str:
    """
    Trim large text to control token usage.

    trim_mode:
      - "head": keep first max_chars
      - "headtail": keep first half + last half (better context)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = _normalize_whitespace(str(x))
    if not max_chars or max_chars <= 0 or len(s) <= max_chars:
        return s

    if trim_mode.lower() == "head":
        return s[:max_chars] + "\n...[TRUNCATED]..."

    # headtail
    half = max_chars // 2
    head = s[:half]
    tail = s[-(max_chars - half):]
    return head + "\n...[MIDDLE TRUNCATED]...\n" + tail


def _system_prompt() -> str:
    return (
        "You are a careful financial analyst.\n"
        "Summarize SEC filing text sections into a compact, structured form.\n"
        "Rules:\n"
        "- Use ONLY the content provided.\n"
        "- Be concise, factual, and avoid speculation.\n"
        "- Keep concrete numbers if explicitly stated (revenue, margins, guidance, debt, cash flows).\n"
        "- If a section is empty, do not invent details.\n"
        "- guidance_change must be exactly one of: up, down, none, unknown.\n"
        "- risk_level must be an integer 1-5.\n"
        "Return ONLY JSON matching the schema."
    )


def _user_prompt(*, date: Any, gvkey: Any, file_type: Any, mgmt: str, rf: str) -> str:
    return (
        f"Metadata:\n"
        f"- date: {date}\n"
        f"- gvkey: {gvkey}\n"
        f"- file_type: {file_type}\n\n"
        "Section: Management Discussion & Analysis (mgmt)\n"
        "-----\n"
        f"{mgmt}\n"
        "-----\n\n"
        "Section: Risk Factors (rf)\n"
        "-----\n"
        f"{rf}\n"
        "-----\n\n"
        "Produce JSON:\n"
        "- summary: 150-200 words\n"
        "- bullish_points: 2-5 bullets\n"
        "- bearish_points: 2-5 bullets\n"
        "- guidance_change: up/down/none/unknown\n"
        "- risk_level: 1-5\n"
    )


# -----------------------------
# Retry / backoff
# -----------------------------
@dataclass
class RetryConfig:
    max_retries: int = 8
    base_sleep: float = 1.0
    max_sleep: float = 30.0


def _sleep_backoff(attempt: int, cfg: RetryConfig) -> None:
    delay = min(cfg.max_sleep, cfg.base_sleep * (2 ** attempt))
    delay *= random.uniform(0.7, 1.3)
    time.sleep(delay)


def _call_openai_summary(
    client: OpenAI,
    model: str,
    temperature: float,
    date: Any,
    gvkey: Any,
    file_type: Any,
    mgmt_text: str,
    rf_text: str,
    retry_cfg: RetryConfig,
) -> StructuredSummary:
    instructions = _system_prompt()
    prompt = _user_prompt(date=date, gvkey=gvkey, file_type=file_type, mgmt=mgmt_text, rf=rf_text)

    last_err: Optional[Exception] = None
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            resp = client.responses.parse(
                model=model,
                instructions=instructions,
                input=prompt,
                text_format=StructuredSummary,
                temperature=temperature,
            )
            return resp.output_parsed
        except Exception as e:
            last_err = e
            if attempt >= retry_cfg.max_retries:
                break
            _sleep_backoff(attempt, retry_cfg)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}") from last_err


# -----------------------------
# Caching
# -----------------------------
def _hash_payload(mgmt: str, rf: str, file_type: str, schema_version: str) -> str:
    h = hashlib.sha256()
    h.update(schema_version.encode("utf-8", errors="ignore"))
    h.update(b"\n---\n")
    h.update(str(file_type).encode("utf-8", errors="ignore"))
    h.update(b"\n---\n")
    h.update(mgmt.encode("utf-8", errors="ignore"))
    h.update(b"\n---\n")
    h.update(rf.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _compact_rf_block(bull: list[str], bear: list[str], guidance: str, risk: int) -> str:
    out = []
    if bull:
        out.append("Bullish points: " + " | ".join(map(str, bull)))
    if bear:
        out.append("Bearish points: " + " | ".join(map(str, bear)))
    out.append(f"Guidance change: {guidance}")
    out.append(f"Risk level: {risk}/5")
    return "\n".join(out)

# -----------------------------
# S&P 500 filtering via gvkey (data/filtered_sp500_data.csv)
# -----------------------------
def _normalize_gvkey(x: Any) -> str:
    """
    Normalize gvkey identifiers for matching across datasets.

    The .pkl files often store gvkey as float (e.g., 1234.0). We convert to a
    clean string key (e.g., "1234").
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    # Try numeric normalization first
    try:
        # Handles floats like 1234.0 and strings like "1234.0"
        xi = int(float(x))
        return str(xi)
    except Exception:
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s


def _infer_year_series(df: pd.DataFrame, *, year_col: Optional[str] = None, date_col: Optional[str] = None) -> pd.Series:
    """
    Infer a year Series from a dataframe using either an explicit year column,
    or a date-like column whose first 4 digits represent the year.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Explicit year column
    if year_col and year_col in df.columns:
        y = pd.to_numeric(df[year_col], errors="coerce")
        return y.astype("Int64")

    for cand in ["year", "fyear", "fiscal_year"]:
        if cand in cols_lower:
            y = pd.to_numeric(df[cols_lower[cand]], errors="coerce")
            return y.astype("Int64")

    # Date column
    if date_col and date_col in df.columns:
        s = df[date_col]
    else:
        for cand in ["date", "datadate", "char_date", "rdq", "adate", "caldt"]:
            if cand in cols_lower:
                s = df[cols_lower[cand]]
                break
        else:
            raise ValueError(
                "Cannot infer year from filtered_sp500_data.csv: no year/date column found. "
                f"Columns: {list(df.columns)}"
            )

    # Parse year from common formats
    s = s.astype(str).str.strip()
    # If it's like 20050104, year = first 4 chars
    year = pd.to_numeric(s.str.slice(0, 4), errors="coerce")
    return year.astype("Int64")


def _load_sp500_gvkeys_for_year(
    year: int,
    csv_path: Path,
    *,
    gvkey_col: Optional[str] = None,
    year_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> set[str]:
    """
    Load the set of gvkeys that are in the S&P 500 (as represented by the
    data filtered dataset) for the given calendar year.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing S&P500 gvkey dataset: {csv_path}")

    dfx = pd.read_csv(csv_path)

    cols_lower = {c.lower(): c for c in dfx.columns}
    if gvkey_col and gvkey_col in dfx.columns:
        gcol = gvkey_col
    elif "gvkey" in cols_lower:
        gcol = cols_lower["gvkey"]
    else:
        raise ValueError(
            "filtered_sp500_data.csv must contain a gvkey column. "
            f"Columns: {list(dfx.columns)}"
        )

    years = _infer_year_series(dfx, year_col=year_col, date_col=date_col)
    mask = years == int(year)

    gv = dfx.loc[mask, gcol].map(_normalize_gvkey)
    return {x for x in gv.tolist() if x}



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
baselines.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# ----------------------------
# Helpers
# ----------------------------
def _normalize_ticker_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("-", ".", regex=False)
    return s


def parse_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures columns are in the right dtype and adds:
      - Date: datetime parsed from 'date' (YYYYMMDD int or string)
      - ym: monthly Period (YYYY-MM) built from (year, month)
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

    df = df.dropna(subset=["Date"]).copy()

    # Normalize ticker and returns
    df["tic"] = _normalize_ticker_series(df["tic"])
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")

    # year/month as ints
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    ym_dt = pd.to_datetime(
        dict(
            year=df["year"].astype("Int64"),
            month=df["month"].astype("Int64"),
            day=1,
        ),
        errors="coerce",
    )
    df["ym"] = ym_dt.dt.to_period("M")
    df = df.dropna(subset=["ym"]).copy()
    df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")

    return df


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a monthly panel with ONE row per (tic, ym):
      - keeps the last available Date in that month for each ticker
      - keeps stock_ret for that snapshot

    Output columns include: [ym (Period), ym_dt (Timestamp month-start), tic, stock_ret]
    """
    tmp = df.dropna(subset=["ym", "tic"]).copy()
    tmp = tmp.sort_values(["tic", "ym", "Date"])
    tmp = tmp.drop_duplicates(subset=["tic", "ym"], keep="last")  # last obs of the month

    tmp["ym_dt"] = tmp["ym"].dt.to_timestamp()  # month-start Timestamp
    tmp = tmp[["ym", "ym_dt", "tic", "stock_ret"]].copy()
    return tmp


def pivot_monthly_returns(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot monthly panel to wide returns matrix:
      index = ym_dt (month start)
      columns = tickers
      values = stock_ret
    """
    R = (
        monthly_panel.pivot_table(index="ym_dt", columns="tic", values="stock_ret", aggfunc="mean")
        .sort_index()
    )
    return R


def optimize_mean_variance(
    train_R: pd.DataFrame,
    lambda_param: float = 0.1,
    long_only: bool = True,
) -> np.ndarray:
    """
    Minimize: w' Σ w - lambda * (μ' w)

    Constraints:
      - sum(w) = 1
      - if long_only=True: 0 <= w_i <= 1
      - if long_only=False: shorting allowed, with loose bounds
    """
    mu = train_R.mean(skipna=True)
    Sigma = train_R.cov()

    mu_v = mu.to_numpy(dtype=float)
    S = Sigma.to_numpy(dtype=float)

    if not np.isfinite(mu_v).all():
        raise RuntimeError("Mean returns contain NaN/inf.")

    if not np.isfinite(S).all():
        raise RuntimeError("Covariance contains NaN/inf.")

    n = len(mu_v)

    def objective(w: np.ndarray) -> float:
        port_ret = float(mu_v @ w)
        port_risk = float(w.T @ S @ w)
        return port_risk - (lambda_param * port_ret)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if long_only:
        bounds = [(0.0, 1.0) for _ in range(n)]
    else:
        bounds = [(-1.0, 1.0) for _ in range(n)]

    w0 = np.full(n, 1.0 / n)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if (not res.success) or (not np.isfinite(res.x).all()):
        raise RuntimeError(res.message)

    w = res.x.astype(float)

    if long_only:
        w = np.clip(w, 0.0, None)

    s = w.sum()
    if abs(s) < 1e-12:
        return np.full(n, 1.0 / n)

    return w / s


def _month_starts_inclusive(start: str, end: str) -> pd.DatetimeIndex:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return pd.date_range(s, e, freq="MS")



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
blacklitterman_weights.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


LOOKBACK_MONTHS = 60  # trailing months used to estimate cov/betas (including current month)
MIN_RET_ROWS = 2      # minimum months required to run estimation

# View calibration guardrails
OMEGA_FLOOR = 1e-4
MIN_VIEW_SAMPLES = 10

# -------------------------
# Ticker / date utils
# -------------------------
def _normalize_ticker(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("-", ".")
    return s

def _parse_yyyymmdd_int(x):
    """

    This function was written by ChatGPT 5.2
    filtered_sp500_data.csv often has 'date' like 20210129 (int).

    """
    try:
        xi = int(x)
        s = str(xi)
    except Exception:
        s = str(x)
    s_digits = "".join(ch for ch in s if ch.isdigit())
    s_digits = s_digits[:8]
    
    return pd.to_datetime(s_digits, format="%Y%m%d", errors="coerce")

def month_pairs(start: str, end: str) -> List[Tuple[str, str]]:
    """Return list of (month_start, month_end) covering [start, end]."""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    ms = pd.date_range(s, e, freq="MS")
    out = []
    for d in ms:
        out.append((d.strftime("%Y-%m-%d"), (d + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")))
    return out

def _project_to_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    A = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T

def _robust_covariance_psd(returns_used: pd.DataFrame) -> np.ndarray:
    """
    Robust covariance estimation (straightforward):
      - Pairwise covariance (no imputation)
      - PSD projection
      - Ridge for numerical stability
    """
    X = returns_used.to_numpy(dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 2:
        # Too small; return tiny diagonal
        n = X.shape[1]
        return np.eye(n, dtype=float) * 1e-6

    # Pairwise covariance (min_periods=2)
    sigma = returns_used.cov(min_periods=2).to_numpy(dtype=float)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = _project_to_psd(sigma, eps=1e-12)

    # Ridge regularization (scale-aware)
    n = sigma.shape[0]
    tr = float(np.trace(sigma))
    ridge = (1e-6 * (tr / n)) if (np.isfinite(tr) and tr > 0) else 1e-6
    sigma = sigma + np.eye(n, dtype=float) * ridge
    return sigma

def _clip_posterior(x: np.ndarray) -> np.ndarray:
    """
    Robustly clip posterior returns to avoid numerical explosions dominating the optimizer.
    Uses MAD-based bounds; falls back to std if MAD ~ 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if np.isfinite(mad) and mad > 1e-12:
        scale = 1.4826 * mad
    else:
        sd = float(np.std(x))
        scale = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0

    lo = med - 10.0 * scale
    hi = med + 10.0 * scale
    return np.clip(x, lo, hi)

def _robust_view_stats(samples: np.ndarray) -> tuple[float, float, int]:
    """
    Robust mean/variance for LLM view samples with small-n protection.
    - Winsorize around median using MAD/STD scale
    - Inflate variance when n is small to avoid overconfidence
    """
    s = np.asarray(samples, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        raise ValueError("No valid samples.")

    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    if np.isfinite(mad) and mad > 1e-12:
        scale = 1.4826 * mad
    else:
        sd = float(np.std(s))
        scale = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0

    lo = med - 5.0 * scale
    hi = med + 5.0 * scale
    s2 = np.clip(s, lo, hi)

    q = float(np.mean(s2))
    v = float(np.var(s2, ddof=1)) if n > 1 else OMEGA_FLOOR

    if n < MIN_VIEW_SAMPLES:
        v *= (MIN_VIEW_SAMPLES / max(1, n))

    v = max(v, OMEGA_FLOOR)
    return q, v, n

# -------------------------
# Data loaders
# -------------------------
def prepare_dataset(dataset_csv_path: str) -> pd.DataFrame:
    """
    Load and pre-process data/filtered_sp500_data.csv once.

    Expected columns (minimum):
      - date (often int like 20210129)
      - tic  (ticker)
      - stock_ret (realized return for that date)
      - market_equity (market cap proxy)

    Returns
    -------
    pd.DataFrame
        Columns: date_dt (datetime), tic (normalized), stock_ret (float), market_equity (float)
    """
    df = pd.read_csv(dataset_csv_path, low_memory=False)

    required = {"date", "tic", "stock_ret", "market_equity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filtered_sp500_data.csv missing columns: {sorted(missing)}")

    df["date_dt"] = df["date"].apply(_parse_yyyymmdd_int)
    df["tic"] = df["tic"].astype(str).map(_normalize_ticker)

    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")
    df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")

    df = df.dropna(subset=["date_dt", "tic"]).copy()
    df = df.sort_values(["date_dt", "tic"])
    return df

def load_market_caps_from_dataset(dataset: pd.DataFrame, period_end_date: str) -> Dict[str, float]:
    """
    Build a {tic: market_equity} snapshot for the month of period_end_date.

    Keeps the last available row per ticker in that month.
    """
    p = pd.Period(pd.to_datetime(period_end_date), freq="M")
    d = dataset[dataset["date_dt"].dt.to_period("M") == p].copy()

    if d.empty:
        return {}

    d = d.sort_values(["tic", "date_dt"]).drop_duplicates(subset=["tic"], keep="last")

    caps = (
        d[["tic", "market_equity"]]
        .dropna(subset=["tic", "market_equity"])
        .set_index("tic")["market_equity"]
        .to_dict()
    )
    return caps

def load_returns_window_from_dataset(
    dataset: pd.DataFrame,
    window_end_date: str,
    lookback_months: int = LOOKBACK_MONTHS,
    tickers: List[str] | None = None,
) -> pd.DataFrame:
    """
    Build a *monthly* returns matrix ending at window_end_date directly from filtered_sp500_data.csv.

    Note: filtered_sp500_data.csv is (mostly) month-end data (often 1 date per month),
    so a "within-month daily matrix" would have only 1 row. To estimate covariances/betas,
    we instead use a trailing window of monthly returns.

    Parameters
    ----------
    dataset : pd.DataFrame
        Pre-processed dataset from prepare_filtered_sp500_dataset().
    window_end_date : str
        Month end date for the evaluation period (e.g., '2015-01-31'). The window includes this month.
    lookback_months : int
        Number of trailing months to include (including current month).
    tickers : list[str] | None
        Optional restriction on tickers.

    Returns
    -------
    pd.DataFrame
        Index: month-end datetime, Columns: tickers, Values: stock_ret (monthly return)
    """
    lookback_months = int(lookback_months)
    if lookback_months < 2:
        raise ValueError("lookback_months must be >= 2")

    end_p = pd.Period(pd.to_datetime(window_end_date), freq="M")
    start_p = end_p - (lookback_months - 1)

    d = dataset.copy()
    d["ym"] = d["date_dt"].dt.to_period("M")
    d = d[(d["ym"] >= start_p) & (d["ym"] <= end_p)].copy()

    if tickers is not None:
        tick_set = set(map(_normalize_ticker, tickers))
        d = d[d["tic"].isin(tick_set)].copy()

    if d.empty:
        return pd.DataFrame()

    # Keep the last available row per (month, ticker)
    d = d.sort_values(["ym", "tic", "date_dt"]).drop_duplicates(subset=["ym", "tic"], keep="last")

    mat = d.pivot(index="ym", columns="tic", values="stock_ret").sort_index()

    # Ensure numeric & drop all-NaN columns
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")
    mat = mat.dropna(axis=1, how="all")

    # Month-end timestamps for compatibility
    mat.index = mat.index.to_timestamp(how="end")
    mat = mat[~mat.index.duplicated(keep="last")].sort_index()
    return mat

def load_llm_responses(responses_path: str) -> Dict[str, dict]:
    """Load responses/{model}_{start}_{end}.json and normalize ticker keys."""
    with open(responses_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        out[_normalize_ticker(k)] = v
    return out


# -------------------------
# Risk-free rate loader (dynamic)
# -------------------------
def load_risk_free_monthly_annual(rf_csv_path: str) -> pd.Series:
    """
    Load a risk-free rate CSV (e.g., FRED DGS10) and return a *monthly* series of
    annualized yields in decimal form, indexed by month-end timestamps.

    The CSV is expected to contain:
      - a date column named 'DATE' (typical FRED export), and
      - a value column (e.g., 'DGS10') representing an annualized yield in percent.

    Notes
    -----
    - FRED DGS10 values are in percent (e.g., 4.25 means 4.25%).
    - We convert to decimal (0.0425) and resample to month-end using the last available
      observation in each month.
    """
    df = pd.read_csv(rf_csv_path)
    # Robustly identify columns
    val_cols = [c for c in df.columns if c != 'observation_date']
    val_col = val_cols[0]

    df['observation_date'] = pd.to_datetime(df['observation_date'], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=['observation_date', val_col]).sort_values('observation_date')
    if df.empty:
        raise ValueError("Risk-free CSV contains no valid (date, value) rows.")

    # Convert percent -> decimal
    s = df.set_index('observation_date')[val_col].astype(float) / 100.0

    # Month-end (annualized) yield, last observation in month
    s_m = s.resample("M").last()
    # Ensure month-end timestamps (no timezone)
    s_m.index = s_m.index.to_period("M").to_timestamp("M")
    return s_m


def align_monthly_rf_to_returns_index(rf_monthly_annual: pd.Series, returns_index: pd.Index) -> pd.Series:
    """
    Align a monthly *annualized* risk-free series to the index of a monthly returns matrix,
    and convert it to a monthly risk-free return approximation (annual/12).

    Parameters
    ----------
    rf_monthly_annual : pd.Series
        Monthly annualized yields in decimal form, indexed by month-end timestamps.
    returns_index : pd.Index
        Index of the monthly returns matrix (month-end timestamps, may include time component).

    Returns
    -------
    pd.Series
        Monthly risk-free rate (decimal), indexed exactly like returns_index.
    """
    # Convert returns index to month-end dates (normalize time component)
    idx_m = pd.to_datetime(returns_index).to_period("M").to_timestamp("M")

    rf = rf_monthly_annual.copy()
    rf.index = pd.to_datetime(rf.index).to_period("M").to_timestamp("M")

    aligned_annual = rf.reindex(idx_m).ffill().bfill()
    # Monthly approximation
    aligned_monthly = aligned_annual / 12.0
    aligned_monthly.index = returns_index
    return aligned_monthly


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
returns_from_weights.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



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