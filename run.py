"""run.py

Run monthly inference of expected returns using an LLM.

Supported model_name values:
  - gpt   -> OpenAI via gpt_query.py (requires keys.py with gpt_key)
  - gemma3 -> Ollama via gemma_query.py
  - qwen  -> Ollama via qwen_query.py (internally uses qwen2.5:1.5b)

To add future local models later, extend the _LOCAL_MODEL_MAP dict.
"""

import argparse
import json
import os
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
from tqdm import tqdm

# NOTE: Backend-specific imports are intentionally lazy (inside main),
# so users can run Ollama models without having OpenAI installed/configured.


# -------------------------
# Model routing
# -------------------------

# Local model aliases -> actual Ollama model ids.
# Keep the alias short because it also becomes the output prefix.
_LOCAL_MODEL_MAP = {
    "gemma3": "gemma3",
    "qwen": "qwen2.5:1.5b",
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
        "predict the expected MONTHLY return (decimal) for the NEXT month. "
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


# -------------------------
# Returns CSV builder (from filtered_sp500_data.csv)
# -------------------------
def build_returns_matrix_monthly(
    sp500_table: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a MONTHLY returns matrix (1 row per month):
      - index = ym (YYYY-MM as string)
      - columns = tickers
      - values = stock_ret (monthly returns)
    Keep the last available date within each (ticker, month).
    """
    df = sp500_table.loc[
        (sp500_table["date_key"] >= window_start) & (sp500_table["date_key"] <= window_end),
        ["date_key", "tic", "stock_ret"],
    ].copy()

    df = df.dropna(subset=["date_key", "tic"])
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")

    # month key
    df["ym"] = df["date_key"].dt.to_period("M")

    # keep last obs within month per ticker
    df = df.sort_values(["tic", "date_key"]).drop_duplicates(subset=["tic", "ym"], keep="last")

    mat = df.pivot(index="ym", columns="tic", values="stock_ret").sort_index()
    mat.index = mat.index.astype(str)  # "YYYY-MM"
    mat = mat.reset_index().rename(columns={"ym": "ym"})  # make 'ym' a column
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
    overwrite: bool = False,
) -> str:
    """
    Create the file expected by evaluate_multiple_updated.py:
        yfinance/returns_<period_start>_<period_end>.csv

    window_mode:
      - "full": use [global_start, global_end] for ALL periods (recommended for stable cov)
      - "expanding": use [global_start, period_end]
      - "rolling": use last `lookback_months` months ending at period_end
      - "period": only [period_start, period_end] (not recommended for cov; too few rows)
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"returns_{period_start}_{period_end}.csv")

    if (not overwrite) and os.path.exists(out_path):
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
        w_end = p_end
        lb = max(1, int(lookback_months))
        w_start = (p_end.to_period("M") - (lb - 1)).to_timestamp()
    elif window_mode == "period":
        w_start, w_end = p_start, p_end
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    # clamp to available data
    w_start = max(w_start, data_min)
    w_end = min(w_end, data_max)

    mat = build_returns_matrix_monthly(sp500_table, w_start, w_end)

    # fallback: if too short window, use full range inside [global_start, global_end]
    if mat.shape[0] < 2:
        w_start2, w_end2 = max(g_start, data_min), min(g_end, data_max)
        mat = build_returns_matrix_monthly(sp500_table, w_start2, w_end2)

    mat.to_csv(out_path, index=False)
    return out_path


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    # Model selector (also used as output filename prefix).
    # Examples:
    #   --model_name gpt    (OpenAI; uses --openai_model)
    #   --model_name gemma3 (Ollama; uses local Gemma3)
    #   --model_name qwen   (Ollama; uses qwen2.5:1.5b)
    parser.add_argument("--model_name", type=str, default="gpt")

    # OpenAI model id (only used when --model_name=gpt)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")

    # Ollama host (only used for local models like gemma3/qwen)
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL (default: http://localhost:11434)",
    )
    parser.add_argument("--input_csv", type=str, default="yfinance/filtered_sp500_data.csv")

    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2022-06-30")

    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--summary_json_max_chars", type=int, default=0)
    parser.add_argument("--lookback_months", type=int, default=12)

    parser.add_argument("--overwrite", action="store_true", help="Recompute months even if output json exists.")

    # returns csv generation
    parser.add_argument("--returns_out_dir", type=str, default="yfinance")
    parser.add_argument(
        "--returns_window_mode",
        type=str,
        default="full",
        choices=["full", "expanding", "rolling", "period"],
    )
    parser.add_argument("--returns_lookback_months", type=int, default=24)

    args = parser.parse_args()

    model_name = str(args.model_name).strip().lower()

    # Output prefix = model_name (keeps file names stable and short)
    out_prefix = model_name

    os.makedirs("responses", exist_ok=True)

    # -------------------------
    # Build LLM client (lazy imports)
    # -------------------------
    if model_name == "gpt":
        try:
            from keys import gpt_key  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Impossible d'importer keys.gpt_key. "
                "Assure-toi d'avoir un fichier keys.py avec gpt_key='...'."
            ) from e

        try:
            try:
                from models_query.gpt_query import GPTQuery  # type: ignore
            except Exception:
                from gpt_query import GPTQuery  # type: ignore
        except Exception as e:
            raise RuntimeError("Impossible d'importer GPTQuery. Fais: pip install openai") from e

        llm = GPTQuery(
            api_key=gpt_key,
            model=str(args.openai_model),
            max_retries=5,
            retry_backoff_s=1.0,
        )

        llm_backend = "openai"
        llm_model_id = str(args.openai_model)

    elif model_name in _LOCAL_MODEL_MAP:
        # Local Ollama model
        ollama_model_id = _LOCAL_MODEL_MAP[model_name]

        if model_name == "qwen":
            try:
                try:
                    from models_query.qwen_query import QwenQuery  # type: ignore
                except Exception:
                    from qwen_query import QwenQuery  # type: ignore
            except Exception as e:
                raise RuntimeError("Impossible d'importer QwenQuery. Fais: pip install ollama") from e

            llm = QwenQuery(
                model=ollama_model_id,
                host=str(args.ollama_host),
                max_retries=5,
                retry_backoff_s=1.0,
            )
        else:
            # Default local wrapper (GemmaQuery)
            try:
                try:
                    from models_query.gemma_query import GemmaQuery  # type: ignore
                except Exception:
                    from gemma_query import GemmaQuery  # type: ignore
            except Exception as e:
                raise RuntimeError("Impossible d'importer GemmaQuery. Fais: pip install ollama") from e

            llm = GemmaQuery(
                model=ollama_model_id,
                host=str(args.ollama_host),
                max_retries=5,
                retry_backoff_s=1.0,
            )

        llm_backend = "ollama"
        llm_model_id = ollama_model_id

    else:
        raise ValueError(
            f"Unknown --model_name={args.model_name!r}. "
            f"Supported: 'gpt' + {sorted(_LOCAL_MODEL_MAP.keys())}"
        )

    # Load data
    sp500_table = pd.read_csv(args.input_csv, low_memory=False)

    required_cols = {"date", "year", "month", "tic", "stock_ret"}
    missing = required_cols - set(sp500_table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {sorted(missing)}")

    # Robust types
    sp500_table["year"] = pd.to_numeric(sp500_table["year"], errors="coerce").astype("Int64")
    sp500_table["month"] = pd.to_numeric(sp500_table["month"], errors="coerce").astype("Int64")
    sp500_table["stock_ret"] = pd.to_numeric(sp500_table["stock_ret"], errors="coerce")

    sp500_table["date_key"] = parse_yyyymmdd_int_to_datetime(sp500_table["date"])
    sp500_table = sp500_table.dropna(subset=["date_key", "year", "month", "tic"]).copy()

    sp500_table["ym"] = pd.to_datetime(
        dict(year=sp500_table["year"].astype(int), month=sp500_table["month"].astype(int), day=1),
        errors="coerce",
    ).dt.to_period("M")

    sp500_table["tic"] = normalize_ticker_series(sp500_table["tic"])

    # Optional metadata columns
    company_col = "conm" if "conm" in sp500_table.columns else None
    sector_name_col = "gics_sector_name" if "gics_sector_name" in sp500_table.columns else None
    gics_col = "gics" if "gics" in sp500_table.columns else None
    sic_col = "sic" if "sic" in sp500_table.columns else None
    naics_col = "naics" if "naics" in sp500_table.columns else None
    market_equity_col = "market_equity" if "market_equity" in sp500_table.columns else None
    summary_json_col = "summary_json" if "summary_json" in sp500_table.columns else None

    # Iterate months
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    month_starts = pd.date_range(start=start_dt, end=end_dt, freq="MS")

    system_prompt = make_system_prompt()

    for month_start_dt in tqdm(month_starts, total=len(month_starts)):
        month_end_dt = (month_start_dt + pd.offsets.MonthEnd(1))
        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = month_end_dt.strftime("%Y-%m-%d")

        out_path = f"responses/{out_prefix}_{month_start}_{month_end}.json"
        already = os.path.exists(out_path)

        # Always ensure returns csv exists for eval script
        ensure_returns_csv_for_period(
            sp500_table=sp500_table,
            period_start=month_start,
            period_end=month_end,
            out_dir=args.returns_out_dir,
            window_mode=args.returns_window_mode,
            global_start=args.start,
            global_end=args.end,
            lookback_months=args.returns_lookback_months,
            overwrite=args.overwrite,  # if overwriting, also overwrite returns file
        )

        if already and (not args.overwrite):
            continue

        # Current month snapshot (metadata)
        current_p = month_start_dt.to_period("M")
        mdf = sp500_table.loc[sp500_table["ym"] == current_p].copy()

        if mdf.empty:
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        # Lookback history for returns list per ticker
        lb = max(int(args.lookback_months), 1)
        hist_start_p = current_p - (lb - 1)
        hdf = sp500_table.loc[
            (sp500_table["ym"] >= hist_start_p) & (sp500_table["ym"] <= current_p),
            ["ym", "tic", "stock_ret"],
        ].copy()

        hdf = hdf.dropna(subset=["stock_ret"])
        hdf = hdf.sort_values(["tic", "ym"])
        hist_map = hdf.groupby("tic")["stock_ret"].apply(list).to_dict()

        data_dict: dict[str, dict] = {}

        for tic, g in mdf.groupby("tic", sort=False):
            past_returns = hist_map.get(tic, [])

            summary_json_val = g[summary_json_col].iloc[0] if summary_json_col else ""
            if pd.isna(summary_json_val):
                summary_json_val = ""
            else:
                summary_json_val = str(summary_json_val)

            maxc = int(args.summary_json_max_chars or 0)
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

        # Query LLM per ticker
        for ticker in tqdm(
            list(data_dict.keys()),
            desc=f"{llm_backend}:{llm_model_id} {month_start}->{month_end}",
            leave=False,
        ):
            user_prompt = make_user_prompt(ticker, data_dict[ticker])

            res = llm.sample_expected_return(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                n_samples=int(args.n_samples),
                temperature=float(args.temperature),
            )

            data_dict[ticker]["expected_return"] = res.samples
            data_dict[ticker]["n_success"] = int(res.n_success)
            data_dict[ticker]["expected_return_mean"] = res.mean
            # If you want debug later:
            # data_dict[ticker]["errors"] = res.errors

        # Save month results
        with open(out_path, "w") as f:
            json.dump(data_dict, f, default=json_default)


if __name__ == "__main__":
    main()
