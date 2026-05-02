"""run.py

Run monthly inference of expected returns using an LLM.

Supported model_name values:
  - gpt   -> OpenAI via gpt_query.py
  - gemma3 -> Ollama via gemma_query.py
  - qwen  -> Ollama via qwen_query.py
  - llama -> Ollama via llama_query.py

"""
from utils import (json_default, normalize_ticker_series, parse_yyyymmdd_int_to_datetime, make_system_prompt, make_user_prompt, MODEL_MAP)
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm

# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    # Model selector
    parser.add_argument("--models", type=str, default="gpt")

    # Ollama host (only used for local models like gemma3/qwen/llama)
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    parser.add_argument("--input_csv", type=str, default="data/filtered_sp500_data.csv")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lookback_months", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true", help="Recompute months even if output json exists.")

    args = parser.parse_args()

    model_name = str(args.models).strip().lower()

    model_id = MODEL_MAP[model_name]

    os.makedirs("responses", exist_ok=True)

    # ----------------
    # Build LLM client
    # ----------------
    if model_name == "gpt":
        try:
            from keys import gpt_key
        except Exception as e:
            raise RuntimeError(
                "Can't import gpt_key from keys.py."
            ) from e

        try:
            from models_query.gpt_query import GPTQuery
        except Exception as e:
            raise RuntimeError("Impossible d'importer GPTQuery. Fais: pip install openai") from e

        llm = GPTQuery(
            api_key=gpt_key,
            model=model_id,
            max_retries=5,
            retry_backoff_s=1.0,
        )

    elif model_name == "qwen":
        try:
            from models_query.qwen_query import QwenQuery
        except Exception as e:
            raise RuntimeError("Impossible d'importer QwenQuery.") from e

        llm = QwenQuery(
            model=model_id,
            host=str(args.ollama_host),
            max_retries=5,
            retry_backoff_s=1.0,
        )

    elif model_name == "gemma3":
        try:
            from models_query.gemma_query import GemmaQuery
        except Exception as e:
            raise RuntimeError("Impossible d'importer GemmaQuery.") from e

        llm = GemmaQuery(
            model=model_id,
            host=str(args.ollama_host),
            max_retries=5,
            retry_backoff_s=1.0,
        )

    elif model_name == "llama":
        try:
            from models_query.llama_query import Llama_Query
        except Exception as e:
            raise RuntimeError("Impossible d'importer Llama_Query.") from e

        llm = Llama_Query(
            model=model_id,
            host=args.ollama_host,
            max_retries=5,
            retry_backoff_s=1.0,
        )


    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Load data
    sp500_table = pd.read_csv(args.input_csv, low_memory=False)

    # Robust types
    sp500_table["year"] = pd.to_numeric(sp500_table["year"], errors="coerce").astype("Int64")
    sp500_table["month"] = pd.to_numeric(sp500_table["month"], errors="coerce").astype("Int64")
    sp500_table["stock_ret"] = pd.to_numeric(sp500_table["stock_ret"], errors="coerce")


    sp500_table["decision_date"] = parse_yyyymmdd_int_to_datetime(sp500_table["char_date"])

    sp500_table = sp500_table.dropna(subset=["decision_date", "tic"]).copy()

    # Month key for decision month
    sp500_table["ym"] = sp500_table["decision_date"].dt.to_period("M")

    sp500_table["tic"] = normalize_ticker_series(sp500_table["tic"])

    # Shift by 1 month per ticker so past_returns are strictly < decision month.
    sp500_table = sp500_table.sort_values(["tic", "ym", "decision_date"])
    sp500_table["realized_ret"] = sp500_table.groupby("tic")["stock_ret"].shift(1)

    # Iterate months
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    month_starts = pd.date_range(start=start_dt, end=end_dt, freq="MS")

    system_prompt = make_system_prompt()

    for month_start_dt in tqdm(month_starts, total=len(month_starts)):
        month_end_dt = (month_start_dt + pd.offsets.MonthEnd(1))
        month_start = month_start_dt.strftime("%Y-%m-%d")
        month_end = month_end_dt.strftime("%Y-%m-%d")

        out_path = f"responses/{model_name}_{month_start}_{month_end}.json"
        already = os.path.exists(out_path)

        if already and (not args.overwrite):
            continue

        # Current month snapshot
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
            ["ym", "tic", "realized_ret"],
        ].copy()

        hdf = hdf.dropna(subset=["realized_ret"])
        hdf = hdf.sort_values(["tic", "ym"])
        hist_map = hdf.groupby("tic")["realized_ret"].apply(list).to_dict()

        data_dict: dict[str, dict] = {}

        for tic, g in mdf.groupby("tic", sort=False):
            g = g.sort_values("decision_date")
            last = g.iloc[-1]
            past_returns = hist_map.get(tic, [])

            summary_json_val = last["summary_json"]
            if pd.isna(summary_json_val):
                summary_json_val = ""
            else:
                summary_json_val = str(summary_json_val)

            row = {
                "company_name": last["conm"],
                "gics_sector_name": last["gics_sector_name"],
                "gics": last["gics"],
                "sic": last["sic"],
                "naics": last["naics"],
                "market_equity": float(last["market_equity"]),
                "past_returns": [float(x) for x in past_returns if pd.notna(x)],
                "summary_json": summary_json_val,
            }

            data_dict[tic] = row

        # Query LLM per ticker
        for ticker in tqdm(
            list(data_dict.keys()),
            desc=f"{model_id} {month_start}->{month_end}",
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

        # Save month results
        with open(out_path, "w") as f:
            json.dump(data_dict, f, default=json_default)


if __name__ == "__main__":
    main()
