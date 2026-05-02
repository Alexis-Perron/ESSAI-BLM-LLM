import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from utils import (
    load_market_caps_from_dataset,
    load_returns_window_from_dataset,
    load_llm_responses,
    prepare_dataset,
    load_risk_free_monthly_annual,
    align_monthly_rf_to_returns_index,
    month_pairs,
    _robust_view_stats,
    _robust_covariance_psd,
    _clip_posterior,
    _normalize_ticker,
    MIN_RET_ROWS,
    LOOKBACK_MONTHS,
)

# -------------------------
# Black-Litterman core
# -------------------------
def _mean_variance_optimize(
    exp_rets: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.1,
) -> np.ndarray:
    """
    Solve long-only, fully-invested mean-variance optimisation:

        minimize_w   w' Σ w  -  risk_aversion * (w' μ)
        s.t.         sum(w)=1,  w>=0

    Returns
    -------
    w : np.ndarray
        Optimal weights (shape: [n_assets,]).
    """
    exp_rets = np.asarray(exp_rets, dtype=float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = int(exp_rets.shape[0])
    if n == 0:
        return np.array([], dtype=float)

    def portfolio_variance(w):
        return float(w.T @ cov_matrix @ w)

    def objective(w):
        return portfolio_variance(w) - float(risk_aversion) * float(w @ exp_rets)

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": lambda x: x},
    )
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.ones(n, dtype=float) / max(1, n)

    result = minimize(
        objective,
        x0,
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 2000},
    )
    return result.x if result.success else x0

def black_litterman_LLM(
    data_dict: Dict[str, dict],
    returns_df: pd.DataFrame,
    tickers: List[str],
    market_equilibrium_return: np.ndarray,
    tau: float,
    risk_aversion: float = 0.1,
) -> np.ndarray:
    """
    Black-Litterman with LLM views (Q) from data_dict[ticker]["expected_return"] list.
    Patch:
      - Robust PSD covariance (Ledoit-Wolf + ridge), fallback to PSD projection
      - Omega floor to avoid overconfidence
      - Robust clip of posterior returns
    """
    q_map = {}
    omega_map = {}

    # Build views per ticker
    for t in tickers:
        samples = data_dict.get(t, {}).get("expected_return", None)
        if not isinstance(samples, list) or len(samples) == 0:
            continue
        s = pd.to_numeric(pd.Series(samples), errors="coerce").dropna().to_numpy(dtype=float)
        if s.size == 0:
            continue

        q, v, _n = _robust_view_stats(s)
        q_map[t] = q
        omega_map[t] = v

    keep = [t for t in tickers if t in q_map]

    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    # Optionally: drop tickers with basically no return history in this period
    obs = returns_df[keep].notna().sum(axis=0)
    keep = [t for t in keep if int(obs.get(t, 0)) >= 2]
    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    returns_used = returns_df[keep].copy()

    # Robust PSD covariance
    sigma = _robust_covariance_psd(returns_used)

    # Align pi, Q, Omega with keep order
    eq_map = dict(zip(tickers, market_equilibrium_return))
    pi = np.array([eq_map.get(t, 0.0) for t in keep], dtype=float)

    Q = np.array([q_map[t] for t in keep], dtype=float)
    Omega = np.diag([omega_map[t] for t in keep]).astype(float)

    # Identity P: each view on one asset
    P = np.eye(len(keep), dtype=float)

    # BL posterior
    tau_sigma = tau * sigma
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_Omega = np.linalg.pinv(Omega)
    M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_Omega @ P)

    posterior_returns = M @ (inv_tau_sigma @ pi + P.T @ inv_Omega @ Q)
    posterior_returns = _clip_posterior(posterior_returns)
    # Mean-variance optimisation (long-only, fully invested)
    w_keep = _mean_variance_optimize(
        exp_rets=posterior_returns,
        cov_matrix=sigma,
        risk_aversion=risk_aversion,
    )

    # Expand to full tickers
    w_full = np.zeros(len(tickers), dtype=float)
    idx_map = {t: i for i, t in enumerate(tickers)}
    for t, w in zip(keep, w_keep):
        w_full[idx_map[t]] = float(w)

    s = w_full.sum()
    if s > 0:
        w_full = w_full / s
    return w_full

def black_litterman_no_views(
    returns_df: pd.DataFrame,
    tickers: List[str],
    market_equilibrium_return: np.ndarray,
    tau: float,
    risk_aversion: float = 0.1,
) -> np.ndarray:
    """
    Black-Litterman *without* views.

    In the BL model, if you provide no views (or views equal to the equilibrium, Q = pi),
    the posterior expected returns reduce to the equilibrium returns (pi). This function
    therefore:
      - estimates a robust PSD covariance Σ
      - uses μ = pi as expected returns (no view tilt)
      - runs the same long-only mean-variance optimisation as the LLM-view version
    """
    # Drop tickers with basically no return history in this period
    obs = returns_df[tickers].notna().sum(axis=0)
    keep = [t for t in tickers if int(obs.get(t, 0)) >= 2]

    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    returns_used = returns_df[keep].copy()
    sigma = _robust_covariance_psd(returns_used)

    eq_map = dict(zip(tickers, market_equilibrium_return))
    pi = np.array([eq_map.get(t, 0.0) for t in keep], dtype=float)
    pi = _clip_posterior(pi)

    w_keep = _mean_variance_optimize(
        exp_rets=pi,
        cov_matrix=sigma,
        risk_aversion=risk_aversion,
    )

    # Expand to full tickers
    w_full = np.zeros(len(tickers), dtype=float)
    idx_map = {t: i for i, t in enumerate(tickers)}
    for t, w in zip(keep, w_keep):
        w_full[idx_map[t]] = float(w)

    s = w_full.sum()
    if s > 0:
        w_full = w_full / s
    return w_full

# -------------------------
# Equilibrium returns (robust beta calc)
# -------------------------
def compute_market_equilibrium_returns(
    returns_df: pd.DataFrame,
    market_caps: Dict[str, float],
    risk_free_monthly: pd.Series | None = None,
    debug_tag: str = "",
) -> Tuple[List[str], np.ndarray]:
    """
    CAPM-like equilibrium:
      - cap-weighted market return from returns_df using market_caps
      - betas vs market computed robustly (requires >=2 observations per ticker)
      - pi = beta * market_risk_premium

    Avoids RuntimeWarnings (ddof<=0, divide by zero).
    Optionally prints tickers with insufficient data.
    """
    caps_s = pd.Series({_normalize_ticker(k): v for k, v in market_caps.items()}, dtype=float).dropna()
    common = [t for t in returns_df.columns if t in caps_s.index]
    if len(common) < 2:
        raise ValueError("Not enough tickers after intersecting returns with market caps.")

    caps_s = caps_s.loc[common]
    w_mkt = caps_s / caps_s.sum()

    # cap-weighted market return
    mkt = (returns_df[common].mul(w_mkt, axis=1)).sum(axis=1)

    # Dynamic risk-free (monthly) aligned to market series.
    # If not provided, we fallback to 0 (should not happen in normal runs).
    if risk_free_monthly is None:
        rf_series = pd.Series(0.0, index=mkt.index, dtype=float)
    else:
        rf_series = risk_free_monthly.reindex(mkt.index).astype(float).ffill().bfill()

    mkt_valid = mkt.dropna()
    if mkt_valid.shape[0] < 2:
        msg = f"{debug_tag} Market series has <2 valid observations."
        raise ValueError(msg)

    mkt_var = float(mkt_valid.var(ddof=1))
    if (not np.isfinite(mkt_var)) or mkt_var <= 1e-18:
        msg = f"{debug_tag} Market variance is zero/NaN; cannot compute betas."
        raise ValueError(msg)

    betas = pd.Series(index=returns_df.columns, dtype=float)
    insufficient_tickers: List[str] = []

    for t in returns_df.columns:
        x = returns_df[t]
        xy = pd.concat([x, mkt], axis=1).dropna()
        if xy.shape[0] < 2:
            betas.loc[t] = 0.0
            insufficient_tickers.append(t)
            continue

        cov_tm = float(xy.iloc[:, 0].cov(xy.iloc[:, 1], ddof=1))
        if not np.isfinite(cov_tm):
            betas.loc[t] = 0.0
            insufficient_tickers.append(t)
            continue

        betas.loc[t] = cov_tm / mkt_var

    # Report tickers with insufficient data
    if insufficient_tickers:
        print(
            f"{debug_tag} beta: insufficient data for {len(insufficient_tickers)} tickers -> betas set to 0."
        )
    mkt_rp = float((mkt_valid - rf_series.loc[mkt_valid.index]).mean())
    pi = (betas.fillna(0.0) * mkt_rp).to_numpy(dtype=float)
    return list(returns_df.columns), pi

# -------------------------
# Period processing
# -------------------------
def process_period_for_model(
    model_name: str,
    start_date: str,
    end_date: str,
    tau: float,
    dataset_df: pd.DataFrame,
    responses_dir: str,
    rf_monthly_annual: pd.Series,
    min_tickers: int = 25,
) -> pd.Series:
    """
    For a given period and model:
      - build *monthly* trailing-window returns matrix directly from filtered_sp500_data.csv (stock_ret)
      - load LLM responses json (unless model is None)
      - load market caps snapshot from filtered_sp500_data.csv (month=end_date)
      - compute equilibrium returns (robust)
      - compute BL weights (LLM views) or BL weights with no views for model=None
    """
    market_caps = load_market_caps_from_dataset(dataset_df, end_date)

    # Special case: model == None  -> Black-Litterman without views (posterior mean = equilibrium pi)
    if str(model_name).strip().lower() in {"none", "null"}:
        returns_df = load_returns_window_from_dataset(dataset_df, window_end_date=end_date, lookback_months=LOOKBACK_MONTHS, tickers=None)
        # No LLM responses needed; use the full returns universe as-is.
    else:
        resp_path = Path(responses_dir) / f"{model_name}_{start_date}_{end_date}.json"
        if not resp_path.exists():
            raise FileNotFoundError(f"Missing LLM response file: {resp_path}")

        model_dict = load_llm_responses(str(resp_path))
        good_resp = {t: v for t, v in model_dict.items() if isinstance(v, dict)}
        if len(good_resp) < 2:
            raise ValueError("Model responses contain <2 usable tickers.")

        returns_df = load_returns_window_from_dataset(dataset_df, window_end_date=end_date, lookback_months=LOOKBACK_MONTHS, tickers=list(good_resp.keys()))

    if returns_df.shape[0] < int(MIN_RET_ROWS):
        raise ValueError(
            f"Returns window too short (rows={returns_df.shape[0]}) for {start_date}->{end_date}. "
            f"Increase LOOKBACK_MONTHS or check dataset coverage."
        )

    if str(model_name).strip().lower() not in {"none", "null"}:
        # Restrict returns to tickers available in responses (already filtered),
        # but also ensure we have at least 2 overlapping tickers with data.
        common_cols = [c for c in returns_df.columns if c in good_resp]
        if len(common_cols) < 2:
            raise ValueError("Not enough overlapping tickers between returns and model responses.")
        if len(common_cols) < int(min_tickers):
            # permissive: keep going
            pass
        returns_df = returns_df[common_cols].copy()

    # Align dynamic risk-free to the monthly returns index (convert annual yield -> monthly rate)
    rf_monthly = align_monthly_rf_to_returns_index(rf_monthly_annual, returns_df.index)

    tickers_all, pi = compute_market_equilibrium_returns(
        returns_df=returns_df,
        market_caps=market_caps,
        risk_free_monthly=rf_monthly,
        debug_tag=f"[{model_name}] {start_date}->{end_date}",
    )

    if str(model_name).strip().lower() in {"none", "null"}:
        w = black_litterman_no_views(
            returns_df=returns_df,
            tickers=tickers_all,
            market_equilibrium_return=pi,
            tau=tau,
            risk_aversion=0.1,
        )
    else:
        w = black_litterman_LLM(
            data_dict=good_resp,
            returns_df=returns_df,
            tickers=tickers_all,
            market_equilibrium_return=pi,
            tau=tau,
            risk_aversion=0.1,
        )

    return pd.Series(w, index=tickers_all)

# -------------------------
# Main (multi-model)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt"], help="e.g. --models gpt gemma3 qwen")
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")
    parser.add_argument("--responses_dir", type=str, default="responses")
    parser.add_argument("--dataset_csv", type=str, default="data/filtered_sp500_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--risk_free_csv", type=str, default="data/DGS1.csv", help="CSV with risk-free yield series (e.g., FRED DGS1 export).")
    parser.add_argument("--min_tickers", type=int, default=25)

    args = parser.parse_args()

    # Load filtered_sp500_data.csv once (we use it for both returns and market caps)
    dataset_df = prepare_dataset(args.dataset_csv)
    # Load dynamic risk-free series (monthly annualized yield in decimal)
    rf_monthly_annual = load_risk_free_monthly_annual(args.risk_free_csv)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    periods = month_pairs(args.start, args.end)

    for model in args.models:
        model = model.strip()
        print(f"\n=== Evaluating model: {model} | tau={args.tau} | {args.start} -> {args.end} ===")

        results = {}  # (start,end) -> Series weights
        for start_date, end_date in tqdm(periods, desc=f"{model} periods"):
            try:
                w = process_period_for_model(
                    model_name=model,
                    start_date=start_date,
                    end_date=end_date,
                    tau=float(args.tau),
                    dataset_df=dataset_df,
                    responses_dir=args.responses_dir,
                    rf_monthly_annual=rf_monthly_annual,
                    min_tickers=int(args.min_tickers),
                )
                results[(start_date, end_date)] = w
            except Exception as e:
                msg = f"[{model}] Error period {start_date} -> {end_date}: {e}"
                print(msg)

        if not results:
            print(f"[{model}] No results produced. (Check missing files / overlap / dates.)")
            continue

        # Build DataFrame: rows=periods, cols=tickers
        df = pd.DataFrame(results).T

        # Turn tuple index into columns cleanly
        df = df.reset_index()
        if "index" in df.columns:
            df[["start_date", "end_date"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
            df = df.drop(columns=["index"])
        else:
            # in case pandas named them level_0/level_1
            if "level_0" in df.columns and "level_1" in df.columns:
                df = df.rename(columns={"level_0": "start_date", "level_1": "end_date"})
            else:
                raise ValueError("Unexpected index format after reset_index; cannot recover period dates.")

        df["Date"] = df["start_date"]  # keep compatibility with downstream scripts
        df = df.drop(columns=["start_date", "end_date"])

        out_path = Path(args.results_dir) / f"{model}_black_litterman_weights_tau_{args.tau}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{model}] Saved weights: {out_path} | shape={df.shape}")

if __name__ == "__main__":
    main()
