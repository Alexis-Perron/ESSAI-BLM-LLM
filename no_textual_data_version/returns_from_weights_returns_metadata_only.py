import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# This script is intended to live in:
#   ESSAI-BLM-LLM/no_textual_data_version/
#
# The original utils.py is expected to live one level above:
#   ESSAI-BLM-LLM/utils.py
#
# Adding ROOT_DIR to sys.path allows:
#   from utils import ...
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils import (
    _clean_ticker,
    _month_starts_between,
    _normalize_weights_columns,
    _parse_dataset_dates,
    _get_monthly_returns_series,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _candidate_weight_paths(
    model_name: str,
    tau: float,
    results_dir: str | Path,
    weights_path: Optional[str] = None,
) -> list[Path]:
    """
    Build candidate paths for the weights file.

    This no-textual-data version expects weights created by:
      blacklitterman_weights_returns_metadata_only.py

    Default expected format:
      {model}_black_litterman_weights_returns_metadata_only_tau_{tau}.csv

    The function also keeps a fallback to the old naming convention, which is
    useful if you manually renamed files or want backward compatibility.
    """
    results_dir = Path(results_dir)

    model_raw = str(model_name).strip()
    model_lower = model_raw.lower()

    candidates: list[Path] = []

    if weights_path is not None:
        candidates.append(Path(weights_path))

    # Preferred no-text naming convention
    candidates.append(
        results_dir / f"{model_lower}_black_litterman_weights_returns_metadata_only_tau_{tau}.csv"
    )

    if model_raw != model_lower:
        candidates.append(
            results_dir / f"{model_raw}_black_litterman_weights_returns_metadata_only_tau_{tau}.csv"
        )

    # Backward-compatible fallback
    candidates.append(
        results_dir / f"{model_lower}_black_litterman_weights_tau_{tau}.csv"
    )

    if model_raw != model_lower:
        candidates.append(
            results_dir / f"{model_raw}_black_litterman_weights_tau_{tau}.csv"
        )

    # Remove duplicates while preserving order
    seen = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)

    return unique


def _default_out_path(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    results_dir: str | Path,
) -> Path:
    """
    Default output path for no-textual-data realized returns.
    """
    model = str(model_name).strip().lower()
    return Path(results_dir) / (
        f"{model}_black_litterman_returns_returns_metadata_only_tau_"
        f"{tau}_{start_date}_{end_date}.csv"
    )


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------
def calculate_model_returns_returns_metadata_only(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    dataset_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    weights_path: Optional[str] = None,
    out_path: Optional[str] = None,
    apply_next_month: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute realized portfolio returns from Black-Litterman weights produced by
    the no-textual-data pipeline.

    This version is adapted for the current project layout:

        ESSAI-BLM-LLM/
        |-- data/
        |   |-- filtered_sp500_data.csv
        |-- utils.py
        |-- no_textual_data_version/
            |-- blacklitterman_weights_returns_metadata_only.py
            |-- returns_from_weights_returns_metadata_only.py
            |-- results_returns_metadata_only/

    Parameters
    ----------
    model_name:
        Model name used in the weights file, e.g. "gpt", "gemma3", "qwen", "llama".
    tau:
        Tau value used in the Black-Litterman weights file.
    start_date, end_date:
        Period over which to compute realized portfolio returns.
    dataset_path:
        Path to filtered_sp500_data.csv. If None, defaults to ROOT_DIR/data/filtered_sp500_data.csv.
    results_dir:
        Directory containing the no-textual-data weights and where returns are saved.
        If None, defaults to the local no_textual_data_version/results_returns_metadata_only.
    weights_path:
        Optional explicit path to a weights CSV.
    out_path:
        Optional explicit path for the output returns CSV.
    apply_next_month:
        True means weights at month M are applied to realized returns of month M+1.
        This is the financially natural setting if the weights are formed using
        information available at month M.
    verbose:
        Print skipped periods and save path.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns [date_key, Portfolio_Return], or None if nothing
        could be produced.
    """
    model_raw = str(model_name).strip()
    model = model_raw.lower()

    if dataset_path is None:
        dataset_path = ROOT_DIR / "data" / "filtered_sp500_data.csv"
    dataset_path = Path(dataset_path)

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent / "results_returns_metadata_only"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Use --dataset_path ..\\data\\filtered_sp500_data.csv or check the project layout."
        )

    candidates = _candidate_weight_paths(
        model_name=model_raw,
        tau=tau,
        results_dir=results_dir,
        weights_path=weights_path,
    )

    weights_file = next((p for p in candidates if p.exists()), None)
    if weights_file is None:
        if verbose:
            print(f"[{model}] No weights file found. Tried:")
            for p in candidates:
                print(f"  - {p}")
        return None

    if out_path is None:
        out_path = _default_out_path(
            model_name=model,
            tau=tau,
            start_date=start_date,
            end_date=end_date,
            results_dir=results_dir,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load and normalize weights
    # -------------------------------------------------------------------------
    weights_df = pd.read_csv(weights_file)

    if "Date" not in weights_df.columns:
        raise ValueError(
            f"[{model}] Weights file must contain a 'Date' column. "
            f"Columns={list(weights_df.columns)[:20]}"
        )

    weights_df["Date"] = pd.to_datetime(weights_df["Date"], errors="coerce")

    if weights_df["Date"].isna().all():
        raise ValueError(f"[{model}] Could not parse dates in weights file column 'Date'.")

    # Normalize weights dates to month-start because weights files store month-start dates.
    weights_df["Date"] = weights_df["Date"].dt.to_period("M").dt.to_timestamp()

    # Normalize / aggregate asset columns by cleaned ticker.
    weights_df, asset_columns = _normalize_weights_columns(weights_df, date_col="Date")

    if len(asset_columns) == 0:
        raise ValueError(f"[{model}] No asset columns found in weights file: {weights_file}")

    # -------------------------------------------------------------------------
    # Load realized returns dataset
    # -------------------------------------------------------------------------
    ds = pd.read_csv(dataset_path, low_memory=False)

    required_cols = {"date", "tic", "stock_ret"}
    missing = required_cols - set(ds.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Columns={list(ds.columns)[:30]}..."
        )

    ret_col = "stock_ret"

    ds["date"] = _parse_dataset_dates(ds["date"])
    ds = ds.dropna(subset=["date"]).copy()

    ds["tic"] = ds["tic"].astype(str).map(_clean_ticker)
    ds = ds[ds["tic"] != ""].copy()

    ds[ret_col] = pd.to_numeric(ds[ret_col], errors="coerce")

    # Precompute month key for fast lookups.
    ds["ym"] = ds["date"].dt.to_period("M").astype(str)
    month_groups = dict(tuple(ds.groupby("ym", sort=False)))

    # -------------------------------------------------------------------------
    # Compute realized portfolio returns
    # -------------------------------------------------------------------------
    month_starts = _month_starts_between(start_date, end_date)
    out_rows = []

    for m0 in month_starts:
        weight_date = m0.to_period("M").to_timestamp()

        if apply_next_month:
            target = (m0 + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
        else:
            target = m0.to_period("M").to_timestamp()

        target_ym = pd.Period(target, freq="M").strftime("%Y-%m")

        if target_ym not in month_groups:
            if verbose:
                print(f"[{model}] No dataset rows for target month {target_ym}. Skipping.")
            continue

        # Select weights row for the decision month.
        row = weights_df.loc[weights_df["Date"] == weight_date]
        if row.empty:
            if verbose:
                print(f"[{model}] No weights found for {weight_date.date()} in {weights_file.name}. Skipping.")
            continue

        month_df = month_groups[target_ym]
        r_series, month_end_date = _get_monthly_returns_series(
            month_df=month_df,
            ticker_col="tic",
            date_col="date",
            ret_col=ret_col,
        )

        if r_series.empty or not np.isfinite(r_series.to_numpy(dtype=float)).any():
            if verbose:
                print(f"[{model}] No valid returns for target month {target_ym}. Skipping.")
            continue

        w = row[asset_columns].iloc[0].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # Align returns to the exact asset columns in the weights file.
        r = r_series.reindex(asset_columns).to_numpy(dtype=float)
        finite = np.isfinite(r)

        if not finite.any():
            if verbose:
                print(f"[{model}] No finite returns after alignment for target month {target_ym}. Skipping.")
            continue

        w_f = w[finite]
        r_f = r[finite]

        # Renormalize on assets with available realized returns.
        # This avoids mechanically penalizing the portfolio when a return is missing
        # for one ticker in the dataset.
        eps = 1e-12
        denom = float(np.sum(w_f))

        if not np.isfinite(denom) or abs(denom) < eps:
            abs_denom = float(np.sum(np.abs(w_f)))
            if np.isfinite(abs_denom) and abs_denom >= eps:
                w_use = np.abs(w_f)
                denom = abs_denom
            else:
                w_use = np.ones_like(w_f, dtype=float)
                denom = float(len(w_f))
        else:
            w_use = w_f

        port_ret = float(np.sum(w_use * r_f) / denom)

        out_rows.append(
            {
                "date_key": month_end_date if pd.notna(month_end_date) else target,
                "Portfolio_Return": port_ret,
                "weight_date": weight_date,
                "target_month": target_ym,
                "n_assets_in_weights": int(len(asset_columns)),
                "n_assets_with_returns": int(finite.sum()),
            }
        )

    if not out_rows:
        if verbose:
            print(f"[{model}] No portfolio returns produced.")
        return None

    out = pd.DataFrame(out_rows).sort_values("date_key").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[{model}] Saved returns: {out_path} | shape={out.shape}")
        print(f"[{model}] Weights source: {weights_file}")

    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute realized portfolio returns from no-textual-data "
            "Black-Litterman weights + filtered_sp500_data.csv."
        )
    )

    parser.add_argument("--models", nargs="+", required=True, help="List of model names, e.g. gpt gemma3 qwen llama.")
    parser.add_argument("--tau", type=float, default=0.025)

    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2025-06-30")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(ROOT_DIR / "data" / "filtered_sp500_data.csv"),
        help="Path to filtered_sp500_data.csv. Default points to ../data/filtered_sp500_data.csv.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results_returns_metadata_only"),
        help="Directory containing no-textual-data weights and where realized returns are saved.",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Optional explicit path to one weights file. Usually not needed.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional explicit output path. Usually not needed.",
    )

    parser.add_argument(
        "--apply_next_month",
        action="store_true",
        help="Apply weights at month M to realized returns of month M+1. This is the default.",
    )

    parser.add_argument(
        "--no_apply_next_month",
        action="store_true",
        help="Apply weights at month M to realized returns of month M instead.",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce printed output.")

    args = parser.parse_args()

    apply_next_month = True
    if args.no_apply_next_month:
        apply_next_month = False
    elif args.apply_next_month:
        apply_next_month = True

    # If multiple models are passed, an explicit weights_path or out_path would be ambiguous.
    if len(args.models) > 1 and args.weights_path is not None:
        raise ValueError("--weights_path should only be used with a single model.")
    if len(args.models) > 1 and args.out_path is not None:
        raise ValueError("--out_path should only be used with a single model.")

    for m in args.models:
        calculate_model_returns_returns_metadata_only(
            model_name=m,
            tau=args.tau,
            start_date=args.start_date,
            end_date=args.end_date,
            dataset_path=args.dataset_path,
            results_dir=args.results_dir,
            weights_path=args.weights_path,
            out_path=args.out_path,
            apply_next_month=apply_next_month,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
