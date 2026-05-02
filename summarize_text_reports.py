"""
This python script: has largely been written by ChatGPT-5.2

summarize_text_reports.py

Creates summarized versions of the annual "text_us_{year}.pkl" files using OpenAI,
keeping the original DataFrame structure and replacing large text with compact summaries.

Input (example columns from text_us_2005.pkl):
- date (object, often YYYYMMDD like "20050104")
- gvkey (float)
- file_type (e.g., 10Q/10K/10KSB)
- mgmt (large text)
- rf (large text)
- plus metadata/returns columns (cik, cusip, year, ret_*, mgmt_size, rf_size, ...)

Output:
- Same rows + all original columns preserved.
- mgmt is replaced with the structured "summary" text (150-200 words).
- rf is replaced with a compact block containing bullish/bearish bullets + guidance_change + risk_level.
- mgmt_size and rf_size are recalculated.
- Adds optional convenience columns:
    summary, bullish_points, bearish_points, guidance_change, risk_level, summary_json

Defaults you asked for:
- temperature = 0.0
- max_chars = 20000 per section
- trim_mode = headtail (keeps beginning and end; better context)

Restartability:
- Per-year checkpoint pickle written every --save_every rows (default 250)
- Global JSON cache by text hash to avoid re-paying if you rerun

Usage (Windows example):
python summarize_text_reports.py ^
  --input_template "C:\\...\\TEXT DATA US by YEAR\\{year}\\text_us_{year}.pkl" ^
  --output_root "C:\\...\\TEXT DATA US SUMMARIZED" ^
  --start_year 2005 --end_year 2025 ^
  --model "gpt-4o-mini" --temperature 0.0 ^
  --max_chars 20000 --trim_mode headtail ^
  --save_every 250 --cache_file "summaries_cache.json"

Requires:
pip install openai pydantic pandas tqdm
and keys.py containing gpt_key
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from keys import gpt_key


from utils import (
    _compact_rf_block,
    _ensure_numpy_core_alias,
    _hash_payload,
    _load_cache,
    _load_sp500_gvkeys_for_year,
    _save_cache,
    _trim_text,
    _normalize_gvkey,
    _call_openai_summary,
    RetryConfig
)


# -----------------------------
# Main pipeline
# -----------------------------
def summarize_year(
    year: int,
    input_template: str,
    output_root: Path,
    model: str,
    temperature: float,
    max_chars: int,
    trim_mode: str,
    save_every: int,
    cache_path: Path,
    retry_cfg: RetryConfig,
    sp500_csv: Path,
    sp500_gvkey_col: Optional[str] = None,
    sp500_year_col: Optional[str] = None,
    sp500_date_col: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Path:
    _ensure_numpy_core_alias()

    in_path = Path(input_template.format(year=year))
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_pickle(in_path)

    for c in ["mgmt", "rf", "file_type", "date", "gvkey"]:
        if c not in df.columns:
            raise ValueError(f"Input file {in_path} missing required column: {c}")

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    cache = _load_cache(cache_path)
    client = OpenAI(api_key=gpt_key)

    out_dir = output_root / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"text_us_{year}.pkl"
    ckpt_path = out_dir / f"text_us_{year}.checkpoint.pkl"

    if ckpt_path.exists():
        df = pd.read_pickle(ckpt_path)

    # Filter to S&P 500 constituents (by gvkey) for this year
    allowed_gvkeys = _load_sp500_gvkeys_for_year(
        year,
        csv_path=sp500_csv,
        gvkey_col=sp500_gvkey_col,
        year_col=sp500_year_col,
        date_col=sp500_date_col,
    )

    before = len(df)
    df_gv = df["gvkey"].map(_normalize_gvkey)
    df = df.loc[df_gv.isin(allowed_gvkeys)].copy()
    after = len(df)
    print(f"[INFO] {year}: filtered using S&P500 gvkeys: {before} -> {after} rows")

    # Ensure size columns exist
    if "mgmt_size" not in df.columns:
        df["mgmt_size"] = pd.NA
    if "rf_size" not in df.columns:
        df["rf_size"] = pd.NA

    # Convenience structured columns (added)
    for col in ["summary", "bullish_points", "bearish_points", "guidance_change", "risk_level", "summary_json"]:
        if col not in df.columns:
            df[col] = pd.NA

    schema_version = "structured_summary"

    it = tqdm(range(len(df)), desc=f"Summarizing {year}", unit="row")
    for i in it:
        idx = df.index[i]
        mgmt_raw = df.at[idx, "mgmt"]
        rf_raw = df.at[idx, "rf"]
        file_type = df.at[idx, "file_type"]
        date = df.at[idx, "date"]
        gvkey = df.at[idx, "gvkey"]

        mgmt_text = _trim_text(mgmt_raw, max_chars=max_chars, trim_mode=trim_mode)
        rf_text = _trim_text(rf_raw, max_chars=max_chars, trim_mode=trim_mode)

        key = _hash_payload(mgmt_text, rf_text, str(file_type), schema_version)
        cached = cache.get(key)

        if cached is None:
            summary_obj = _call_openai_summary(
                client=client,
                model=model,
                temperature=temperature,
                date=date,
                gvkey=gvkey,
                file_type=file_type,
                mgmt_text=mgmt_text,
                rf_text=rf_text,
                retry_cfg=retry_cfg,
            )
            cached = summary_obj.model_dump()
            cache[key] = cached

            if len(cache) % 100 == 0:
                _save_cache(cache_path, cache)

        # Store structured fields
        df.at[idx, "summary"] = cached.get("summary", "")
        df.at[idx, "bullish_points"] = cached.get("bullish_points", [])
        df.at[idx, "bearish_points"] = cached.get("bearish_points", [])
        df.at[idx, "guidance_change"] = cached.get("guidance_change", "unknown")
        df.at[idx, "risk_level"] = int(cached.get("risk_level", 3))
        df.at[idx, "summary_json"] = json.dumps(cached, ensure_ascii=False)

        # Replace full text columns (as requested)
        df.at[idx, "mgmt"] = df.at[idx, "summary"]
        df.at[idx, "rf"] = _compact_rf_block(
            bull=df.at[idx, "bullish_points"] if isinstance(df.at[idx, "bullish_points"], list) else [],
            bear=df.at[idx, "bearish_points"] if isinstance(df.at[idx, "bearish_points"], list) else [],
            guidance=str(df.at[idx, "guidance_change"]),
            risk=int(df.at[idx, "risk_level"]),
        )

        df.at[idx, "mgmt_size"] = len(str(df.at[idx, "mgmt"]))
        df.at[idx, "rf_size"] = len(str(df.at[idx, "rf"]))

        if save_every and save_every > 0 and (i + 1) % save_every == 0:
            df.to_pickle(ckpt_path)

    df.to_pickle(out_path)

    if ckpt_path.exists():
        ckpt_path.unlink()
    _save_cache(cache_path, cache)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_template",
        type=str,
        default=r"C:\Users\alexi\OneDrive\Documents\école\McGill-FIAM\2025\Hackathon-Final-2025\DATA ASSET MANAGEMENT HACKATHON 2025 FINALS\TEXT DATA US by YEAR\{year}\text_us_{year}.pkl"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=r"C:\Users\alexi\OneDrive\Documents\école\McGill-FIAM\2025\Hackathon-Final-2025\DATA ASSET MANAGEMENT HACKATHON 2025 FINALS\TEXT DATA US SUMMARIZED"
    )

    # Filter universe using gvkey membership from data/filtered_sp500_data.csv
    parser.add_argument(
        "--sp500_gvkey_csv",
        type=str,
        default=r"data/filtered_sp500_data.csv",
        help="CSV containing S&P 500 (filtered) universe with a gvkey column; used to filter per year."
    )
    parser.add_argument(
        "--sp500_gvkey_col",
        type=str,
        default="",
        help="Optional name of the gvkey column in --sp500_gvkey_csv (auto-detected if empty)."
    )
    parser.add_argument(
        "--sp500_year_col",
        type=str,
        default="",
        help="Optional name of the year column in --sp500_gvkey_csv (auto-detected if empty)."
    )
    parser.add_argument(
        "--sp500_date_col",
        type=str,
        default="",
        help="Optional name of the date column (YYYYMMDD/ISO) in --sp500_gvkey_csv to infer year (auto-detected if empty)."
    )
    parser.add_argument("--start_year", type=int, default=2015)
    parser.add_argument("--end_year", type=int, default=2016)

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--max_chars", type=int, default=20000)
    parser.add_argument("--trim_mode", type=str, default="headtail", choices=["head", "headtail"])
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--cache_file", type=str, default="summaries_cache.json")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="For testing: only process first N rows of each year (0 = all).")

    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--base_sleep", type=float, default=1.0)
    parser.add_argument("--max_sleep", type=float, default=30.0)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    sp500_csv = Path(args.sp500_gvkey_csv)
    if not sp500_csv.is_absolute():
        sp500_csv = (script_dir / sp500_csv).resolve()

    sp500_gvkey_col = str(args.sp500_gvkey_col).strip() or None
    sp500_year_col = str(args.sp500_year_col).strip() or None
    sp500_date_col = str(args.sp500_date_col).strip() or None

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    retry_cfg = RetryConfig(max_retries=args.max_retries, base_sleep=args.base_sleep, max_sleep=args.max_sleep)
    cache_path = Path(args.cache_file)

    max_rows = None if args.max_rows <= 0 else int(args.max_rows)

    for year in range(args.start_year, args.end_year + 1):
        out_path = summarize_year(
            year=year,
            input_template=args.input_template,
            output_root=output_root,
            model=args.model,
            temperature=args.temperature,
            max_chars=args.max_chars,
            trim_mode=args.trim_mode,
            save_every=args.save_every,
            cache_path=cache_path,
            retry_cfg=retry_cfg,
            sp500_csv=sp500_csv,
            sp500_gvkey_col=sp500_gvkey_col,
            sp500_year_col=sp500_year_col,
            sp500_date_col=sp500_date_col,
            max_rows=max_rows,
        )
        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
