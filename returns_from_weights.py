"""
Compute monthly portfolio returns from Black-Litterman weight CSV files.

Default convention:
  - weights at month M are applied to returns in month M+1
  - no transaction cost is charged on the first allocation
  - transaction costs are charged on later rebalances at 0.001 * turnover
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    _clean_ticker,
    _get_monthly_returns_series,
    _month_starts_between,
    _normalize_weights_columns,
    _parse_dataset_dates,
)

WEIGHTS_MIDDLE = "_black_litterman_market_implied_weights_tau_"
RETURNS_MIDDLE = "_black_litterman_implied_weights_returns_tau_"
DEFAULT_TRANSACTION_COST_RATE = 0.001


def _format_portfolio_risk_aversion(value: float | str) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value).strip().replace(" ", "_")


def _resolve_results_dir(results_dir: str | Path, portfolio_risk_aversion: float | str | None) -> Path:
    base = Path(results_dir)
    if portfolio_risk_aversion is None:
        return base

    subdir = f"portfolio_risk_aversion_{_format_portfolio_risk_aversion(portfolio_risk_aversion)}"
    return base if base.name == subdir else base / subdir


def _weight_suffix(tau: float | str) -> str:
    return f"{WEIGHTS_MIDDLE}{tau}.csv"


def _returns_filename(portfolio_id: str, tau: float | str, start_date: str, end_date: str) -> str:
    return f"{portfolio_id}{RETURNS_MIDDLE}{tau}_{start_date}_{end_date}.csv"


def _portfolio_id_from_weights_file(path: Path, tau: float | str) -> str:
    suffix = _weight_suffix(tau)
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected weights filename: {path.name}")
    return path.name[: -len(suffix)]


def discover_weight_files_for_model(model_name: str, tau: float | str, results_dir: str | Path) -> list[Path]:
    model = str(model_name).strip().lower()
    if not model:
        return []

    results_path = Path(results_dir)
    suffix = _weight_suffix(tau)

    if model in {"none", "null", "no_views", "noviews"}:
        model = "none"

    return sorted(results_path.glob(f"{model}*{suffix}"))


def load_dataset_month_groups(dataset_path: str | Path) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(dataset_path, low_memory=False)
    required = {"date", "tic", "stock_ret"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data["date"] = _parse_dataset_dates(data["date"])
    data = data.dropna(subset=["date"]).copy()
    data["tic"] = data["tic"].astype(str).map(_clean_ticker)
    data = data[data["tic"] != ""].copy()
    data["stock_ret"] = pd.to_numeric(data["stock_ret"], errors="coerce")
    data["ym"] = data["date"].dt.to_period("M").astype(str)

    return dict(tuple(data.groupby("ym", sort=False)))


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(w.sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.zeros_like(w, dtype=float)
    return w / total


def _calculate_transaction_cost(
    current_weights: np.ndarray,
    previous_weights: np.ndarray | None,
    transaction_cost_rate: float,
) -> tuple[float, float]:
    rate = float(transaction_cost_rate or 0.0)
    if rate <= 0.0 or previous_weights is None:
        return 0.0, 0.0

    current = _normalize_weights(current_weights)
    previous = _normalize_weights(previous_weights)
    turnover = float(np.sum(np.abs(current - previous)))
    return turnover, float(rate * turnover)


def calculate_returns_from_weights_file(
    weights_file: str | Path,
    portfolio_id: str,
    tau: float,
    start_date: str,
    end_date: str,
    month_groups: dict[str, pd.DataFrame],
    results_dir: str | Path,
    apply_next_month: bool = True,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
) -> pd.DataFrame | None:
    weights_file = Path(weights_file)
    if not weights_file.exists():
        return None

    weights_df = pd.read_csv(weights_file)
    if "Date" not in weights_df.columns:
        raise ValueError(f"[{portfolio_id}] Weights file has no Date column: {weights_file}")

    weights_df["Date"] = pd.to_datetime(weights_df["Date"], errors="coerce")
    if weights_df["Date"].isna().all():
        raise ValueError(f"[{portfolio_id}] Could not parse dates in Date column.")

    weights_df["Date"] = weights_df["Date"].dt.to_period("M").dt.to_timestamp()
    weights_df, asset_columns = _normalize_weights_columns(weights_df, date_col="Date")

    rows = []
    previous_weights = None

    for weight_month in _month_starts_between(start_date, end_date):
        weight_date = weight_month.to_period("M").to_timestamp()
        target_month = weight_month + pd.offsets.MonthBegin(1) if apply_next_month else weight_month
        target_ym = pd.Period(target_month, freq="M").strftime("%Y-%m")

        if target_ym not in month_groups:
            continue

        row = weights_df.loc[weights_df["Date"] == weight_date]
        if row.empty:
            continue

        returns_series, month_end_date = _get_monthly_returns_series(
            month_df=month_groups[target_ym],
            ticker_col="tic",
            date_col="date",
            ret_col="stock_ret",
        )
        if returns_series.empty:
            continue

        weights = row[asset_columns].iloc[0].to_numpy(dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        returns = returns_series.reindex(asset_columns).to_numpy(dtype=float)
        finite = np.isfinite(returns)
        if not finite.any():
            continue

        weights_f = weights[finite]
        returns_f = returns[finite]
        denom = float(weights_f.sum())
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            continue

        gross_return = float(np.sum(weights_f * returns_f) / denom)
        turnover, transaction_cost = _calculate_transaction_cost(
            current_weights=weights,
            previous_weights=previous_weights,
            transaction_cost_rate=transaction_cost_rate,
        )
        net_return = gross_return - transaction_cost

        rows.append(
            {
                "date_key": month_end_date if pd.notna(month_end_date) else target_month,
                "Portfolio_Return": net_return,
                "Gross_Portfolio_Return": gross_return,
                "Turnover": turnover,
                "Transaction_Cost": transaction_cost,
            }
        )
        previous_weights = weights.copy()

    if not rows:
        return None

    output = pd.DataFrame(rows).sort_values("date_key").reset_index(drop=True)
    out_path = Path(results_dir) / _returns_filename(portfolio_id, tau, start_date, end_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def calculate_model_returns(
    model_name: str,
    tau: float,
    start_date: str,
    end_date: str,
    dataset_path: str = "data/filtered_sp25_data.csv",
    results_dir: str = "results",
    apply_next_month: bool = True,
    portfolio_risk_aversion: float | str | None = None,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
) -> dict[str, pd.DataFrame | None]:
    resolved_results_dir = _resolve_results_dir(results_dir, portfolio_risk_aversion)
    month_groups = load_dataset_month_groups(dataset_path)
    outputs = {}

    for weights_file in discover_weight_files_for_model(model_name, tau, resolved_results_dir):
        portfolio_id = _portfolio_id_from_weights_file(weights_file, tau)
        outputs[portfolio_id] = calculate_returns_from_weights_file(
            weights_file=weights_file,
            portfolio_id=portfolio_id,
            tau=tau,
            start_date=start_date,
            end_date=end_date,
            month_groups=month_groups,
            results_dir=resolved_results_dir,
            apply_next_month=apply_next_month,
            transaction_cost_rate=transaction_cost_rate,
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute portfolio returns from BL weights.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2025-06-30")
    parser.add_argument("--dataset_path", type=str, default="data/filtered_sp25_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--portfolio_risk_aversion", type=str, default=None)
    parser.add_argument("--transaction_cost_rate", type=float, default=DEFAULT_TRANSACTION_COST_RATE)
    parser.add_argument("--same_month", action="store_true", help="Apply weights to same-month returns instead of next-month returns.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results_dir = _resolve_results_dir(args.results_dir, args.portfolio_risk_aversion)
    month_groups = load_dataset_month_groups(args.dataset_path)
    apply_next_month = not args.same_month

    if not args.quiet:
        print(f"Results directory: {results_dir}")
        print(f"Transaction cost rate: {float(args.transaction_cost_rate):g}")
        print("Initial transaction cost: 0")

    seen_files = set()
    saved = 0

    for model in args.models:
        weight_files = discover_weight_files_for_model(model, args.tau, results_dir)
        if not weight_files:
            if not args.quiet:
                print(f"[{model}] No matching weights files found.")
            continue

        if not args.quiet:
            print(f"\n[{model}] Found {len(weight_files)} weights file(s).")

        for weights_file in weight_files:
            key = str(weights_file.resolve())
            if key in seen_files:
                continue
            seen_files.add(key)

            portfolio_id = _portfolio_id_from_weights_file(weights_file, args.tau)
            output = calculate_returns_from_weights_file(
                weights_file=weights_file,
                portfolio_id=portfolio_id,
                tau=args.tau,
                start_date=args.start_date,
                end_date=args.end_date,
                month_groups=month_groups,
                results_dir=results_dir,
                apply_next_month=apply_next_month,
                transaction_cost_rate=float(args.transaction_cost_rate),
            )

            if not args.quiet:
                if output is None:
                    print(f"  - [{portfolio_id}] No returns produced.")
                else:
                    saved += 1
                    out_file = results_dir / _returns_filename(portfolio_id, args.tau, args.start_date, args.end_date)
                    print(f"  - [{portfolio_id}] Saved {len(output)} rows -> {out_file.name}")

    if not args.quiet:
        print(f"\nDone. Return files saved: {saved}")


if __name__ == "__main__":
    main()
