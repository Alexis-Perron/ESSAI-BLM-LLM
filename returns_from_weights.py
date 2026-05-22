import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

from utils import (
    _clean_ticker,
    _month_starts_between,
    _normalize_weights_columns,
    _parse_dataset_dates,
    _get_monthly_returns_series,
)


WEIGHTS_MIDDLE = "_black_litterman_market_implied_weights_tau_"
RETURNS_MIDDLE = "_black_litterman_implied_weights_returns_tau_"


# -------------------------
# File discovery helpers
# -------------------------
def _weight_suffix(tau: float | str) -> str:
    return f"{WEIGHTS_MIDDLE}{tau}.csv"


def _returns_filename(portfolio_id: str, tau: float | str, start_date: str, end_date: str) -> str:
    return f"{portfolio_id}{RETURNS_MIDDLE}{tau}_{start_date}_{end_date}.csv"


def _portfolio_id_from_weights_file(path: Path, tau: float | str) -> str:
    """Extract portfolio id from '{portfolio_id}_black_litterman_market_implied_weights_tau_{tau}.csv'."""
    name = path.name
    suffix = _weight_suffix(tau)
    if name.endswith(suffix):
        return name[: -len(suffix)]
    # fallback: remove extension only
    return path.stem


def discover_weight_files_for_model(
    model_name: str,
    tau: float | str,
    results_dir: str = "results",
    include_variants: bool = True,
) -> List[Path]:
    """
    Discover all BL weight files compatible with a requested model name.

    Examples if model_name='gpt':
      - old/original: results/gpt_black_litterman_market_implied_weights_tau_0.025.csv
      - new variants: results/gpt_omega_x25_black_litterman_market_implied_weights_tau_0.025.csv
                      results/gpt_blend_alpha_0.5_omega_1_black_litterman_market_implied_weights_tau_0.025.csv

    Matching is case-insensitive, but returned paths preserve actual file names.
    """
    results_path = Path(results_dir)
    suffix = _weight_suffix(tau)
    model_clean = str(model_name).strip()
    model_lower = model_clean.lower()

    if not model_clean:
        return []

    all_weight_files = sorted(results_path.glob(f"*{suffix}"))
    matched: List[Path] = []

    for p in all_weight_files:
        pid = _portfolio_id_from_weights_file(p, tau)
        pid_lower = pid.lower()

        # Exact old/original file: gpt_... or exact variant if user passes full id
        is_exact = pid_lower == model_lower

        # Variant files: gpt_omega_x25_..., gpt_blend_alpha_..., etc.
        is_variant = pid_lower.startswith(model_lower + "_")

        if is_exact or (include_variants and is_variant):
            matched.append(p)

    # Deduplicate while keeping sorted order
    seen = set()
    out = []
    for p in matched:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# -------------------------
# Dataset loading
# -------------------------
def load_dataset_month_groups(dataset_path: str) -> Dict[str, pd.DataFrame]:
    """Load realized returns dataset once and return {YYYY-MM: dataframe for that month}."""
    ds = pd.read_csv(dataset_path, low_memory=False)

    ret_col = "stock_ret"
    required = {"date", "tic", ret_col}
    missing = required - set(ds.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}. Columns={list(ds.columns)[:30]}...")

    ds["date"] = _parse_dataset_dates(ds["date"])
    ds = ds.dropna(subset=["date"]).copy()
    ds["tic"] = ds["tic"].astype(str).map(_clean_ticker)
    ds = ds[ds["tic"] != ""].copy()
    ds[ret_col] = pd.to_numeric(ds[ret_col], errors="coerce")
    ds["ym"] = ds["date"].dt.to_period("M").astype(str)

    return dict(tuple(ds.groupby("ym", sort=False)))


# -------------------------
# Core computation
# -------------------------
def calculate_returns_from_weights_file(
    weights_file: str | Path,
    portfolio_id: str,
    tau: float,
    start_date: str,
    end_date: str,
    month_groups: Dict[str, pd.DataFrame],
    results_dir: str = "results",
    out_path: Optional[str] = None,
    apply_next_month: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute portfolio returns from one BL weights file + realized returns.

    Output format:
      date_key, Portfolio_Return

    apply_next_month:
      - True  => weights month M applied to returns of month M+1
      - False => weights month M applied to returns of month M
    """
    weights_file = Path(weights_file)
    portfolio_id = str(portfolio_id).strip()

    if out_path is None:
        out_path = str(Path(results_dir) / _returns_filename(portfolio_id, tau, start_date, end_date))
    out_path = str(out_path)

    if not weights_file.exists():
        return None

    weights_df = pd.read_csv(weights_file)

    if "Date" not in weights_df.columns:
        raise ValueError(f"[{portfolio_id}] Weights file has no 'Date' column: {weights_file}")

    weights_df["Date"] = pd.to_datetime(weights_df["Date"], errors="coerce")

    if weights_df["Date"].isna().all():
        raise ValueError(f"[{portfolio_id}] Could not parse dates in weights file column 'Date'.")

    # Normalize weights dates to month-start
    weights_df["Date"] = weights_df["Date"].dt.to_period("M").dt.to_timestamp()

    # Normalize / aggregate asset columns by cleaned ticker
    weights_df, asset_columns = _normalize_weights_columns(weights_df, date_col="Date")

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
            continue

        month_df = month_groups[target_ym]
        r_series, month_end_date = _get_monthly_returns_series(
            month_df=month_df,
            ticker_col="tic",
            date_col="date",
            ret_col="stock_ret",
        )

        if r_series.empty or not np.isfinite(r_series.to_numpy(dtype=float)).any():
            continue

        # Select weights row for weight month
        row = weights_df.loc[weights_df["Date"] == weight_date]
        if row.empty:
            continue

        w = row[asset_columns].iloc[0].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # Align assets between weights and available returns
        r = r_series.reindex(asset_columns).to_numpy(dtype=float)
        finite = np.isfinite(r)

        if not finite.any():
            continue

        w_f = w[finite]
        r_f = r[finite]

        eps = 1e-12
        denom = float(np.sum(w_f))
        if not np.isfinite(denom) or abs(denom) < eps:
            abs_denom = float(np.sum(np.abs(w_f)))
            if np.isfinite(abs_denom) and abs_denom >= eps:
                denom = abs_denom
                w_use = np.abs(w_f)
            else:
                denom = float(len(w_f))
                w_use = np.ones_like(w_f, dtype=float)
        else:
            w_use = w_f

        port_ret = float(np.sum(w_use * r_f) / denom)

        out_rows.append(
            {
                "date_key": month_end_date if pd.notna(month_end_date) else target,
                "Portfolio_Return": port_ret,
            }
        )

    if not out_rows:
        return None

    out = pd.DataFrame(out_rows).sort_values("date_key").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def calculate_model_returns(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    dataset_path: str = "data/filtered_sp25_data.csv",
    results_dir: str = "results",
    weights_path: Optional[str] = None,
    out_path: Optional[str] = None,
    apply_next_month: bool = True,
    include_variants: bool = True,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Backward-compatible wrapper.

    If weights_path is provided, computes returns for that one file.
    Otherwise, discovers every matching old/new BL weights file for model_name.

    Returns a dict: {portfolio_id: dataframe_or_None}.
    """
    month_groups = load_dataset_month_groups(dataset_path)

    if weights_path is not None:
        weights_file = Path(weights_path)
        portfolio_id = _portfolio_id_from_weights_file(weights_file, tau)
        if not portfolio_id or portfolio_id == weights_file.stem:
            portfolio_id = str(model_name).strip().lower()
        return {
            portfolio_id: calculate_returns_from_weights_file(
                weights_file=weights_file,
                portfolio_id=portfolio_id,
                tau=tau,
                start_date=start_date,
                end_date=end_date,
                month_groups=month_groups,
                results_dir=results_dir,
                out_path=out_path,
                apply_next_month=apply_next_month,
            )
        }

    weight_files = discover_weight_files_for_model(
        model_name=model_name,
        tau=tau,
        results_dir=results_dir,
        include_variants=include_variants,
    )

    outputs: Dict[str, Optional[pd.DataFrame]] = {}
    for wf in weight_files:
        portfolio_id = _portfolio_id_from_weights_file(wf, tau)
        outputs[portfolio_id] = calculate_returns_from_weights_file(
            weights_file=wf,
            portfolio_id=portfolio_id,
            tau=tau,
            start_date=start_date,
            end_date=end_date,
            month_groups=month_groups,
            results_dir=results_dir,
            out_path=None,
            apply_next_month=apply_next_month,
        )

    return outputs


# -------------------------
# CLI
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute portfolio returns from BL weights + dataset returns. "
            "For each model, the script now discovers both old files and new variant files."
        )
    )
    parser.add_argument("--models", nargs="+", required=True, help="List of model names or portfolio prefixes, e.g. gpt gemma3 llama qwen none.")
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2025-06-30")
    parser.add_argument("--dataset_path", type=str, default="data/filtered_sp25_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--weights_path", type=str, default=None, help="Optional explicit path to one weights CSV. Use only with one model.")
    parser.add_argument("--out_path", type=str, default=None, help="Optional explicit output CSV path. Use only with one explicit weights file.")
    parser.add_argument("--apply_next_month", action="store_true", help="Apply weights at month M to returns of month M+1. This is the default.")
    parser.add_argument("--no_apply_next_month", action="store_true", help="Apply weights at month M to returns of month M.")
    parser.add_argument("--exact_only", action="store_true", help="Only compute the exact old-style file for each model, not model_* variants.")
    parser.add_argument("--list_only", action="store_true", help="Only list discovered weights files; do not compute returns.")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    apply_next_month = True
    if args.no_apply_next_month:
        apply_next_month = False
    elif args.apply_next_month:
        apply_next_month = True

    include_variants = not args.exact_only

    if (args.weights_path is not None or args.out_path is not None) and len(args.models) != 1:
        raise ValueError("--weights_path and --out_path can only be used when exactly one model is provided.")
    if args.out_path is not None and args.weights_path is None:
        raise ValueError("--out_path requires --weights_path, because multiple discovered files would otherwise map to one output path.")

    # Discover files first, so the user can verify exactly what will be run.
    explicit_mode = args.weights_path is not None

    if args.list_only:
        if explicit_mode:
            print(Path(args.weights_path))
            return
        for m in args.models:
            files = discover_weight_files_for_model(
                model_name=m,
                tau=args.tau,
                results_dir=args.results_dir,
                include_variants=include_variants,
            )
            print(f"\n[{m}] {len(files)} matching weights file(s):")
            for f in files:
                print(f"  - {f}")
        return

    month_groups = load_dataset_month_groups(args.dataset_path)

    total_saved = 0

    if explicit_mode:
        wf = Path(args.weights_path)
        portfolio_id = _portfolio_id_from_weights_file(wf, args.tau)
        if portfolio_id == wf.stem:
            portfolio_id = str(args.models[0]).strip().lower()

        out = calculate_returns_from_weights_file(
            weights_file=wf,
            portfolio_id=portfolio_id,
            tau=args.tau,
            start_date=args.start_date,
            end_date=args.end_date,
            month_groups=month_groups,
            results_dir=args.results_dir,
            out_path=args.out_path,
            apply_next_month=apply_next_month,
        )
        if not args.quiet:
            if out is None:
                print(f"[{portfolio_id}] No returns produced from {wf.name}.")
            else:
                print(f"[{portfolio_id}] Saved returns with {len(out)} rows from {wf.name}.")
        return

    # Normal multi-model / multi-variant mode
    seen_files = set()
    for m in args.models:
        weight_files = discover_weight_files_for_model(
            model_name=m,
            tau=args.tau,
            results_dir=args.results_dir,
            include_variants=include_variants,
        )

        if not weight_files:
            if not args.quiet:
                print(f"[{m}] No matching weights files found in {args.results_dir}.")
            continue

        if not args.quiet:
            print(f"\n[{m}] Found {len(weight_files)} matching weights file(s).")

        for wf in weight_files:
            key = str(wf.resolve())
            if key in seen_files:
                continue
            seen_files.add(key)

            portfolio_id = _portfolio_id_from_weights_file(wf, args.tau)
            out = calculate_returns_from_weights_file(
                weights_file=wf,
                portfolio_id=portfolio_id,
                tau=args.tau,
                start_date=args.start_date,
                end_date=args.end_date,
                month_groups=month_groups,
                results_dir=args.results_dir,
                out_path=None,
                apply_next_month=apply_next_month,
            )

            if not args.quiet:
                if out is None:
                    print(f"  - [{portfolio_id}] No returns produced from {wf.name}.")
                else:
                    total_saved += 1
                    out_file = Path(args.results_dir) / _returns_filename(portfolio_id, args.tau, args.start_date, args.end_date)
                    print(f"  - [{portfolio_id}] Saved {len(out)} rows -> {out_file.name}")

    if not args.quiet:
        print(f"\nDone. Return files saved: {total_saved}")


if __name__ == "__main__":
    main()
