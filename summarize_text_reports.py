"""
summarize_text_reports.py (v3)

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
- mgmt is replaced with the structured "summary" text (150–200 words).
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
python summarize_text_reports_v3.py ^
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
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from keys import gpt_key


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

    # Ensure size columns exist
    if "mgmt_size" not in df.columns:
        df["mgmt_size"] = pd.NA
    if "rf_size" not in df.columns:
        df["rf_size"] = pd.NA

    # Convenience structured columns (added)
    for col in ["summary", "bullish_points", "bearish_points", "guidance_change", "risk_level", "summary_json"]:
        if col not in df.columns:
            df[col] = pd.NA

    schema_version = "v3_structured_summary_2026-01-07"

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
    parser.add_argument("--input_template", type=str, required=True,
                        help=r"Path template with {year}, e.g. C:\...\{year}\text_us_{year}.pkl")
    parser.add_argument("--output_root", type=str, required=True, default="\TEXT DATA US SUMMARIZED",
                        help="Output root directory. Will create one folder per year.")
    parser.add_argument("--start_year", type=int, default=2005)
    parser.add_argument("--end_year", type=int, default=2025)

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
            max_rows=max_rows,
        )
        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
