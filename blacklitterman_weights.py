"""
Compute Black-Litterman portfolio weights for the original text-enhanced pipeline.

Outputs:
  results/portfolio_risk_aversion_{value}/
      {portfolio_id}_black_litterman_market_implied_weights_tau_{tau}.csv

Model names are intentionally strict. Use `gpt54mini` for GPT-5.4-mini.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from utils import (
    LOOKBACK_MONTHS,
    MIN_RET_ROWS,
    _clip_posterior,
    _normalize_ticker,
    _robust_covariance_psd,
    _robust_view_stats,
    load_llm_responses,
    load_market_caps_from_dataset,
    load_returns_window_from_dataset,
    month_pairs,
    prepare_dataset,
)

OMEGA_FLOOR = 1e-4
MARKET_PRIOR_DELTA = 1.0
BASE_SUFFIX = "base_omega_1"
VALID_MODELS = {"gemma3", "gpt", "gpt54mini", "llama", "qwen"}
NO_VIEW_MODELS = {"none", "null", "no_views", "noviews"}


def _clean_model_name(model: str) -> str:
    return str(model).strip().lower()


def _portfolio_name(model: str) -> str:
    return f"{model}_{BASE_SUFFIX}"


def _risk_aversion_results_subdir(portfolio_risk_aversion: float) -> str:
    return f"portfolio_risk_aversion_{float(portfolio_risk_aversion):g}"


def _safe_normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)

    total = float(w.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.ones_like(w, dtype=float) / max(1, w.size)
    return w / total


def _mean_variance_optimize(
    exp_rets: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    max_weight: float,
) -> np.ndarray:
    exp_rets = np.asarray(exp_rets, dtype=float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    max_weight = float(max_weight)

    if not np.isfinite(max_weight) or max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError(f"max_weight must be in (0, 1]. Got {max_weight}.")

    n = exp_rets.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    if max_weight + 1e-12 < 1.0 / n:
        raise ValueError(f"max_weight={max_weight:g} is infeasible for n={n} assets.")

    def objective(w: np.ndarray) -> float:
        return float(w.T @ cov_matrix @ w) - float(risk_aversion) * float(w @ exp_rets)

    x0 = np.ones(n, dtype=float) / n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000},
    )

    if not res.success or not np.isfinite(res.x).all():
        return x0
    return _safe_normalize_weights(res.x)


def _load_responses_for_period(
    responses_dir: Path,
    model: str,
    start_date: str,
    end_date: str,
) -> dict[str, dict]:
    path = responses_dir / f"{model}_{start_date}_{end_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM response file: {path}")

    raw = load_llm_responses(str(path))
    responses = {}
    for ticker, value in raw.items():
        clean_ticker = _normalize_ticker(ticker)
        if isinstance(value, dict):
            responses[clean_ticker] = value

    if len(responses) < 2:
        raise ValueError(f"Response file contains <2 usable tickers: {path}")
    return responses


def _build_absolute_view_maps(
    responses: dict[str, dict],
    tickers: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    q_map = {}
    omega_map = {}

    for ticker in tickers:
        vals = responses.get(ticker, {}).get("expected_return")
        if vals is None:
            continue
        if not isinstance(vals, list):
            vals = [vals]

        try:
            q, variance, n = _robust_view_stats(np.asarray(vals, dtype=float))
        except Exception:
            continue

        if np.isfinite(q) and np.isfinite(variance) and int(n) > 0:
            q_map[ticker] = float(q)
            omega_map[ticker] = float(max(OMEGA_FLOOR, variance))

    return q_map, omega_map


def _compute_market_prior(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    caps = pd.Series({_normalize_ticker(k): v for k, v in market_caps.items()}, dtype=float)
    caps = caps.replace([np.inf, -np.inf], np.nan).dropna()
    caps = caps[caps > 0]

    tickers = [t for t in returns_df.columns if t in caps.index]
    tickers = [t for t in tickers if int(returns_df[t].notna().sum()) >= int(MIN_RET_ROWS)]
    if len(tickers) < 2:
        raise ValueError("Not enough tickers after intersecting returns and positive market caps.")

    returns = returns_df[tickers].copy()
    market_weights = (caps.loc[tickers] / caps.loc[tickers].sum()).to_numpy(dtype=float)
    sigma = _robust_covariance_psd(returns)
    pi = _clip_posterior(float(MARKET_PRIOR_DELTA) * (sigma @ market_weights))

    if not np.isfinite(pi).all():
        raise ValueError("Market-implied prior contains NaN/inf.")

    return tickers, pi, sigma, _safe_normalize_weights(market_weights)


def _bl_weights_absolute_views(
    responses: dict[str, dict],
    returns_df: pd.DataFrame,
    tickers: list[str],
    pi: np.ndarray,
    tau: float,
    portfolio_risk_aversion: float,
    max_weight: float,
) -> np.ndarray:
    q_map, omega_map = _build_absolute_view_maps(responses, tickers)
    keep = [t for t in tickers if t in q_map and int(returns_df[t].notna().sum()) >= int(MIN_RET_ROWS)]

    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    idx = [tickers.index(t) for t in keep]
    pi_k = np.asarray(pi, dtype=float)[idx]
    sigma_k = _robust_covariance_psd(returns_df[keep].copy())

    q = np.array([q_map[t] for t in keep], dtype=float)
    omega = np.diag([omega_map[t] for t in keep])

    inv_tau_sigma = np.linalg.pinv(float(tau) * sigma_k)
    inv_omega = np.linalg.pinv(omega)
    posterior_cov = np.linalg.pinv(inv_tau_sigma + inv_omega)
    posterior_mean = posterior_cov @ (inv_tau_sigma @ pi_k + inv_omega @ q)
    posterior_mean = _clip_posterior(posterior_mean)

    w_keep = _mean_variance_optimize(
        exp_rets=posterior_mean,
        cov_matrix=sigma_k,
        risk_aversion=portfolio_risk_aversion,
        max_weight=max_weight,
    )

    weights = np.zeros(len(tickers), dtype=float)
    for ticker, weight in zip(keep, w_keep):
        weights[tickers.index(ticker)] = float(weight)

    return _safe_normalize_weights(weights)


def _process_no_view_period(
    end_date: str,
    dataset_df: pd.DataFrame,
) -> pd.Series:
    returns_df = load_returns_window_from_dataset(
        dataset_df,
        window_end_date=end_date,
        lookback_months=LOOKBACK_MONTHS,
        tickers=None,
    )
    if returns_df.shape[0] < int(MIN_RET_ROWS):
        raise ValueError(f"Returns window too short: rows={returns_df.shape[0]}")
    if returns_df.shape[1] < 2:
        raise ValueError("Returns window contains <2 usable tickers.")

    market_caps = load_market_caps_from_dataset(dataset_df, end_date)
    tickers, _pi, _sigma, market_weights = _compute_market_prior(returns_df, market_caps)
    return pd.Series(_safe_normalize_weights(market_weights), index=tickers)


def _process_period(
    model: str,
    start_date: str,
    end_date: str,
    tau: float,
    dataset_df: pd.DataFrame,
    responses_dir: Path,
    portfolio_risk_aversion: float,
    max_weight: float,
) -> pd.Series:
    responses = _load_responses_for_period(responses_dir, model, start_date, end_date)
    returns_df = load_returns_window_from_dataset(
        dataset_df,
        window_end_date=end_date,
        lookback_months=LOOKBACK_MONTHS,
        tickers=list(responses),
    )

    if returns_df.shape[0] < int(MIN_RET_ROWS):
        raise ValueError(f"Returns window too short: rows={returns_df.shape[0]}")
    if returns_df.shape[1] < 2:
        raise ValueError("Returns window contains <2 usable tickers.")

    common = [c for c in returns_df.columns if c in responses]
    if len(common) < 2:
        raise ValueError("Not enough overlapping tickers between returns and responses.")

    returns_df = returns_df[common].copy()
    market_caps = load_market_caps_from_dataset(dataset_df, end_date)
    tickers, pi, _sigma, _market_weights = _compute_market_prior(returns_df, market_caps)
    returns_df = returns_df[tickers].copy()

    weights = _bl_weights_absolute_views(
        responses=responses,
        returns_df=returns_df,
        tickers=tickers,
        pi=pi,
        tau=tau,
        portfolio_risk_aversion=portfolio_risk_aversion,
        max_weight=max_weight,
    )
    return pd.Series(weights, index=tickers)


def _write_results_csv(results: dict[tuple[str, str], pd.Series], out_path: Path) -> None:
    df = pd.DataFrame(results).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["start_date", "end_date"])
    df = df.reset_index()
    df["Date"] = df["start_date"]
    df = df.drop(columns=["start_date", "end_date"])
    df = df[["Date"] + [c for c in df.columns if c != "Date"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved weights: {out_path} | shape={df.shape}")


def _run_model_periods(
    model: str,
    periods: list[tuple[str, str]],
    out_path: Path,
    dataset_df: pd.DataFrame,
    responses_dir: Path,
    tau: float,
    portfolio_risk_aversion: float,
    max_weight: float,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"Skip existing: {out_path}")
        return

    results = {}
    desc = "none" if model in NO_VIEW_MODELS else _portfolio_name(model)

    for start_date, end_date in tqdm(periods, desc=desc):
        try:
            if model in NO_VIEW_MODELS:
                weights = _process_no_view_period(end_date=end_date, dataset_df=dataset_df)
            else:
                weights = _process_period(
                    model=model,
                    start_date=start_date,
                    end_date=end_date,
                    tau=tau,
                    dataset_df=dataset_df,
                    responses_dir=responses_dir,
                    portfolio_risk_aversion=portfolio_risk_aversion,
                    max_weight=max_weight,
                )
            results[(start_date, end_date)] = weights
        except Exception as exc:
            print(f"[{desc}] Error period {start_date} -> {end_date}: {exc}")

    if results:
        _write_results_csv(results, out_path)
    else:
        print(f"[{desc}] No results produced.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BL-LLM portfolio weights.")
    parser.add_argument("--models", nargs="+", default=["none", "gemma3", "gpt", "gpt54mini", "llama", "qwen"])
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")
    parser.add_argument("--responses_dir", type=str, default="responses")
    parser.add_argument("--dataset_csv", type=str, default="data/filtered_sp25_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--portfolio_risk_aversion", type=float, default=0.1)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    max_weight = float(args.max_weight)
    if not np.isfinite(max_weight) or max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError(f"--max_weight must be in (0, 1]. Got {max_weight}.")

    requested_models = [_clean_model_name(m) for m in args.models]
    valid_requested = []
    for model in requested_models:
        if model in NO_VIEW_MODELS:
            if "none" not in valid_requested:
                valid_requested.append("none")
        elif model in VALID_MODELS:
            valid_requested.append(model)
        else:
            print(f"Warning: unknown model '{model}'. Skipping.")

    if not valid_requested:
        raise ValueError("No valid model selected.")

    results_dir = Path(args.results_dir) / _risk_aversion_results_subdir(args.portfolio_risk_aversion)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset_df = prepare_dataset(args.dataset_csv)
    periods = month_pairs(args.start, args.end)
    responses_dir = Path(args.responses_dir)

    print(f"Market prior delta: {MARKET_PRIOR_DELTA}")
    print(f"Portfolio risk aversion: {float(args.portfolio_risk_aversion):g}")
    print(f"Max weight: {max_weight:g}")
    print(f"Results directory: {results_dir}")
    print(f"Periods: {args.start} -> {args.end} | tau={args.tau}")

    for model in valid_requested:
        if model == "none":
            portfolio_id = "none"
        else:
            portfolio_id = _portfolio_name(model)

        out_path = results_dir / f"{portfolio_id}_black_litterman_market_implied_weights_tau_{args.tau}.csv"
        print(f"\n=== {portfolio_id} ===")
        _run_model_periods(
            model=model,
            periods=periods,
            out_path=out_path,
            dataset_df=dataset_df,
            responses_dir=responses_dir,
            tau=float(args.tau),
            portfolio_risk_aversion=float(args.portfolio_risk_aversion),
            max_weight=max_weight,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
