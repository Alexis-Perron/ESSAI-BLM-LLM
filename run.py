"""
Run monthly LLM expected-return inference for the original text-enhanced pipeline.

Supported model names:
  - gpt, gpt54mini: OpenAI through GPTQuery
  - gemma3, qwen, llama: local Ollama models
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    MODEL_MAP,
    json_default,
    make_system_prompt,
    make_user_prompt,
    normalize_ticker_series,
    parse_yyyymmdd_int_to_datetime,
)

OPENAI_MODELS = {"gpt", "gpt54mini"}
LOCAL_MODELS = {"gemma3", "qwen", "llama"}
MODEL_IDS = {
    "gpt": MODEL_MAP.get("gpt", "gpt-4o-mini"),
    "gpt54mini": "gpt-5.4-mini",
    "gemma3": MODEL_MAP.get("gemma3", "gemma3"),
    "qwen": MODEL_MAP.get("qwen", "qwen"),
    "llama": MODEL_MAP.get("llama", "llama"),
}


def resolve_model(model_name: str) -> tuple[str, str]:
    model = str(model_name).strip().lower()
    if model not in MODEL_IDS:
        supported = ", ".join(sorted(MODEL_IDS))
        raise ValueError(f"Unknown model: {model}. Supported models: {supported}")
    return model, MODEL_IDS[model]


def build_llm_client(model_name: str, model_id: str, ollama_host: str):
    if model_name in OPENAI_MODELS:
        from keys import gpt_key
        from models_query.gpt_query import GPTQuery

        return GPTQuery(
            api_key=gpt_key,
            model=model_id,
            max_retries=5,
            retry_backoff_s=1.0,
        )

    if model_name == "gemma3":
        from models_query.gemma_query import GemmaQuery
        client_class = GemmaQuery
    elif model_name == "qwen":
        from models_query.qwen_query import QwenQuery
        client_class = QwenQuery
    elif model_name == "llama":
        from models_query.llama_query import Llama_Query
        client_class = Llama_Query
    else:
        raise ValueError(f"Unknown local model: {model_name}")

    return client_class(
        model=model_id,
        host=str(ollama_host),
        max_retries=5,
        retry_backoff_s=1.0,
    )


def load_dataset(input_csv: str | Path) -> pd.DataFrame:
    data = pd.read_csv(input_csv, low_memory=False)
    required_cols = {
        "year",
        "month",
        "stock_ret",
        "char_date",
        "tic",
        "conm",
        "gics_sector_name",
        "gics",
        "sic",
        "naics",
        "market_equity",
        "summary_json",
    }
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["month"] = pd.to_numeric(data["month"], errors="coerce").astype("Int64")
    data["stock_ret"] = pd.to_numeric(data["stock_ret"], errors="coerce")
    data["market_equity"] = pd.to_numeric(data["market_equity"], errors="coerce")
    data["decision_date"] = parse_yyyymmdd_int_to_datetime(data["char_date"])
    data = data.dropna(subset=["decision_date", "tic"]).copy()
    data["ym"] = data["decision_date"].dt.to_period("M")
    data["tic"] = normalize_ticker_series(data["tic"])
    data = data.sort_values(["tic", "ym", "decision_date"])
    data["realized_ret"] = data.groupby("tic")["stock_ret"].shift(1)

    return data


def build_month_payload(data: pd.DataFrame, month_start: pd.Timestamp, lookback_months: int) -> dict[str, dict]:
    current_period = month_start.to_period("M")
    current_data = data.loc[data["ym"] == current_period].copy()
    if current_data.empty:
        return {}

    lookback = max(int(lookback_months), 1)
    hist_start = current_period - (lookback - 1)
    history = data.loc[
        (data["ym"] >= hist_start) & (data["ym"] <= current_period),
        ["ym", "tic", "realized_ret"],
    ].copy()
    history = history.dropna(subset=["realized_ret"]).sort_values(["tic", "ym"])
    returns_by_ticker = history.groupby("tic")["realized_ret"].apply(list).to_dict()

    payload = {}
    for ticker, group in current_data.groupby("tic", sort=False):
        last = group.sort_values("decision_date").iloc[-1]
        summary_json = last.get("summary_json", "")
        summary_json = "" if pd.isna(summary_json) else str(summary_json)

        market_equity = last.get("market_equity")
        market_equity = float(market_equity) if pd.notna(market_equity) else None

        payload[ticker] = {
            "company_name": last.get("conm", ""),
            "gics_sector_name": last.get("gics_sector_name", ""),
            "gics": last.get("gics", ""),
            "sic": last.get("sic", ""),
            "naics": last.get("naics", ""),
            "market_equity": market_equity,
            "past_returns": [float(x) for x in returns_by_ticker.get(ticker, []) if pd.notna(x)],
            "summary_json": summary_json,
        }

    return payload


def run_model(
    model_name: str,
    model_id: str,
    llm,
    data: pd.DataFrame,
    month_starts: pd.DatetimeIndex,
    output_dir: Path,
    lookback_months: int,
    n_samples: int,
    temperature: float,
    overwrite: bool,
) -> None:
    system_prompt = make_system_prompt()

    for month_start_dt in tqdm(month_starts, desc=model_name):
        month_end_dt = month_start_dt + pd.offsets.MonthEnd(1)
        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = month_end_dt.strftime("%Y-%m-%d")
        out_path = output_dir / f"{model_name}_{month_start}_{month_end}.json"

        if out_path.exists() and not overwrite:
            continue

        data_dict = build_month_payload(data, month_start_dt, lookback_months)
        if not data_dict:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            continue

        for ticker in tqdm(
            list(data_dict),
            desc=f"{model_id} {month_start}->{month_end}",
            leave=False,
        ):
            user_prompt = make_user_prompt(ticker, data_dict[ticker])
            result = llm.sample_expected_return(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                n_samples=int(n_samples),
                temperature=float(temperature),
            )
            data_dict[ticker]["expected_return"] = result.samples
            data_dict[ticker]["n_success"] = int(result.n_success)
            data_dict[ticker]["expected_return_mean"] = result.mean

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, default=json_default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monthly LLM return forecasts.")
    parser.add_argument("--models", nargs="+", default=["gpt"])
    parser.add_argument("--ollama_host", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--input_csv", type=str, default="data/filtered_sp25_data.csv")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lookback_months", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="responses")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.input_csv)
    month_starts = pd.date_range(start=pd.to_datetime(args.start), end=pd.to_datetime(args.end), freq="MS")

    for requested_model in args.models:
        model_name, model_id = resolve_model(requested_model)
        llm = build_llm_client(model_name, model_id, args.ollama_host)
        run_model(
            model_name=model_name,
            model_id=model_id,
            llm=llm,
            data=data,
            month_starts=month_starts,
            output_dir=output_dir,
            lookback_months=int(args.lookback_months),
            n_samples=int(args.n_samples),
            temperature=float(args.temperature),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
