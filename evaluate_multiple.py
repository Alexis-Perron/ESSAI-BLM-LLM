import argparse
import calendar
import json
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm


# -------------------------
# Ticker / date utils
# -------------------------
def _normalize_ticker(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("-", ".")
    return s


def _parse_yyyymmdd_int(x):
    """

    This function was written by ChatGPT 5.2
    filtered_sp500_data.csv often has 'date' like 20210129 (int).

    """
    if pd.isna(x):
        return pd.NaT
    try:
        xi = int(x)
        s = str(xi)
    except Exception:
        s = str(x)
    s_digits = "".join(ch for ch in s if ch.isdigit())
    if len(s_digits) >= 8:
        s_digits = s_digits[:8]
        return pd.to_datetime(s_digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def month_pairs(start: str, end: str) -> List[Tuple[str, str]]:
    """Return list of (month_start, month_end) covering [start, end]."""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    ms = pd.date_range(s, e, freq="MS")
    out = []
    for d in ms:
        out.append((d.strftime("%Y-%m-%d"), (d + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")))
    return out
def _project_to_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    A = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def _robust_covariance_psd(returns_used: pd.DataFrame) -> np.ndarray:
    """
    Robust covariance estimation:
      - Fill NaNs with column means (simple, stable)
      - Ledoit-Wolf shrinkage -> PSD
      - Ridge for numerical stability
      - Fallback: pairwise cov -> PSD projection
    """
    X = returns_used.to_numpy(dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 2:
        # Too small; return tiny diagonal
        n = X.shape[1]
        return np.eye(n, dtype=float) * 1e-6

    # Impute NaNs with column means
    col_means = np.nanmean(X, axis=0)
    # If a column is all-NaN, nanmean gives NaN -> set to 0
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_means, inds[1])

    try:
        lw = LedoitWolf().fit(X)
        sigma = lw.covariance_.astype(float)
        sigma = 0.5 * (sigma + sigma.T)
    except Exception:
        # Fallback: pairwise covariance then PSD projection
        sigma = returns_used.cov(min_periods=2).to_numpy(dtype=float)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = _project_to_psd(sigma, eps=1e-12)

    # Ridge regularization (scale-aware)
    n = sigma.shape[0]
    tr = float(np.trace(sigma))
    ridge = (1e-6 * (tr / n)) if (np.isfinite(tr) and tr > 0) else 1e-6
    sigma = sigma + np.eye(n, dtype=float) * ridge
    return sigma


def _clip_posterior(x: np.ndarray) -> np.ndarray:
    """
    Robustly clip posterior returns to avoid numerical explosions dominating the optimizer.
    Uses MAD-based bounds; falls back to std if MAD ~ 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if np.isfinite(mad) and mad > 1e-12:
        scale = 1.4826 * mad
    else:
        sd = float(np.std(x))
        scale = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0

    lo = med - 10.0 * scale
    hi = med + 10.0 * scale
    return np.clip(x, lo, hi)


# -------------------------
# Data loaders
# -------------------------
def load_market_caps_from_dataset(dataset_csv_path: str, period_end_date: str) -> Dict[str, float]:
    """
    Build a {tic: market_equity} snapshot using filtered_sp500_data.csv for the month of period_end_date.
    Keeps last available row per ticker in that month.
    """
    df = pd.read_csv(dataset_csv_path, low_memory=False)

    required = {"date", "tic", "market_equity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filtered_sp500_data.csv missing columns: {sorted(missing)}")

    df["date_dt"] = df["date"].apply(_parse_yyyymmdd_int)
    df["tic"] = df["tic"].astype(str).map(_normalize_ticker)
    df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")

    p = pd.Period(pd.to_datetime(period_end_date), freq="M")
    df = df[df["date_dt"].dt.to_period("M") == p].copy()

    # Keep last available row per ticker in that month
    df = df.sort_values(["tic", "date_dt"]).drop_duplicates(subset=["tic"], keep="last")

    caps = (
        df[["tic", "market_equity"]]
        .dropna(subset=["tic", "market_equity"])
        .set_index("tic")["market_equity"]
        .to_dict()
    )
    return caps


def load_returns_matrix(returns_path: str) -> pd.DataFrame:
    """
    Load returns_{start}_{end}.csv robustly.
    Supports:
      - monthly matrix with column 'ym' + tickers
      - matrix saved with an index column (Unnamed: 0)
      - also accepts 'date_key' / 'Date'
    Output index is month-start datetime (YYYY-MM-01).
    """
    df = pd.read_csv(returns_path, low_memory=False)

    # If first col is unnamed index, drop it
    if len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Identify date column
    if "ym" in df.columns:
        dt = pd.to_datetime(df["ym"].astype(str) + "-01", errors="coerce")
        df = df.drop(columns=["ym"])
        df.index = dt.dt.to_period("M").dt.to_timestamp()
    elif "date_key" in df.columns:
        dt = pd.to_datetime(df["date_key"], errors="coerce")
        df = df.drop(columns=["date_key"])
        df.index = dt.dt.to_period("M").dt.to_timestamp()
    elif "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df = df.drop(columns=["Date"])
        df.index = dt.dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError(f"Returns file has no 'ym'/'date_key'/'Date' column: {returns_path}")

    # Normalize tickers columns & numeric
    df.columns = [_normalize_ticker(c) for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    # Drop duplicated months if any
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_llm_responses(responses_path: str) -> Dict[str, dict]:
    """Load responses/{model}_{start}_{end}.json and normalize ticker keys."""
    with open(responses_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        out[_normalize_ticker(k)] = v
    return out


# -------------------------
# Black-Litterman core
# -------------------------
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

        q = float(np.mean(s))
        v = float(np.var(s)) if s.size > 1 else 1e-4

        # IMPORTANT: floor Omega to avoid near-zero confidence -> explosions
        omega_floor = 1e-4  # ~1% monthly std floor (variance)
        q_map[t] = q
        omega_map[t] = max(v, omega_floor)

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
    def portfolio_variance(w, cov_matrix):
        return float(w.T @ cov_matrix @ w)

    def objective_function(w, exp_rets, cov_matrix):
        return portfolio_variance(w, cov_matrix) - float(risk_aversion) * float(w @ exp_rets)

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": lambda x: x},
    )

    bounds = tuple((0.0, 1.0) for _ in range(len(keep)))
    x0 = np.ones(len(keep), dtype=float) / len(keep)

    result = minimize(
        objective_function,
        x0,
        args=(posterior_returns, sigma),
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 2000},
    )
    w_keep = result.x if result.success else x0

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
    risk_free_rate_annual: float = 0.02,
    debug_tag: str = "",
    returns_file: str = "",
    debug_beta: bool = False,
    debug_beta_max: int = 10,
) -> Tuple[List[str], np.ndarray]:
    """
    CAPM-like equilibrium:
      - cap-weighted market return from returns_df using market_caps
      - betas vs market computed robustly (requires >=2 observations per ticker)
      - pi = beta * market_risk_premium

    Avoids RuntimeWarnings (ddof<=0, divide by zero).
    Prints which returns file has insufficient data, and optional tickers.
    """
    caps_s = pd.Series({_normalize_ticker(k): v for k, v in market_caps.items()}, dtype=float).dropna()
    common = [t for t in returns_df.columns if t in caps_s.index]
    if len(common) < 2:
        raise ValueError("Not enough tickers after intersecting returns with market caps.")

    caps_s = caps_s.loc[common]
    w_mkt = caps_s / caps_s.sum()

    # cap-weighted market return
    mkt = (returns_df[common].mul(w_mkt, axis=1)).sum(axis=1)

    rf = risk_free_rate_annual / 12.0

    mkt_valid = mkt.dropna()
    if mkt_valid.shape[0] < 2:
        msg = f"{debug_tag} Market series has <2 valid observations."
        if returns_file:
            msg += f" returns_file={returns_file}"
        raise ValueError(msg)

    mkt_var = float(mkt_valid.var(ddof=1))
    if (not np.isfinite(mkt_var)) or mkt_var <= 1e-18:
        msg = f"{debug_tag} Market variance is zero/NaN; cannot compute betas."
        if returns_file:
            msg += f" returns_file={returns_file}"
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

    if insufficient_tickers and returns_file:
        print(
            f"{debug_tag} beta: insufficient data for {len(insufficient_tickers)} tickers -> betas set to 0. "
            f"returns_file={returns_file}"
        )
        if debug_beta:
            show = insufficient_tickers[: max(1, int(debug_beta_max))]
            print(f"{debug_tag} tickers (first {len(show)}): {show}")

    mkt_rp = float((mkt_valid - rf).mean())
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
    returns_dir: str,
    responses_dir: str,
    dataset_csv_path: str,
    risk_free_rate_annual: float = 0.02,
    min_tickers: int = 25,
    debug_beta: bool = False,
    debug_beta_max: int = 10,
) -> pd.Series:
    """
    For a given period and model:
      - load returns file
      - load LLM responses json
      - load market caps snapshot from filtered_sp500_data.csv (month=end_date)
      - compute equilibrium returns (robust)
      - compute BL weights using LLM views
    """
    market_caps = load_market_caps_from_dataset(dataset_csv_path, end_date)

    returns_path = Path(returns_dir) / f"returns_{start_date}_{end_date}.csv"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing returns file: {returns_path}")

    returns_df = load_returns_matrix(str(returns_path))
    if returns_df.shape[0] < 2:
        raise ValueError(f"Returns matrix too short (rows={returns_df.shape[0]}): {returns_path}")

    resp_path = Path(responses_dir) / f"{model_name}_{start_date}_{end_date}.json"
    if not resp_path.exists():
        raise FileNotFoundError(f"Missing LLM response file: {resp_path}")

    model_dict = load_llm_responses(str(resp_path))
    good_resp = {t: v for t, v in model_dict.items() if isinstance(v, dict)}

    # Restrict returns to tickers available in responses
    common_cols = [c for c in returns_df.columns if c in good_resp]
    if len(common_cols) < 2:
        raise ValueError("Not enough overlapping tickers between returns and model responses.")
    if len(common_cols) < int(min_tickers):
        # permissive: keep going, but you'll see the debug prints if betas are unstable
        pass

    returns_df = returns_df[common_cols].copy()

    tickers_all, pi = compute_market_equilibrium_returns(
        returns_df=returns_df,
        market_caps=market_caps,
        risk_free_rate_annual=risk_free_rate_annual,
        debug_tag=f"[{model_name}] {start_date}->{end_date}",
        returns_file=str(returns_path),
        debug_beta=debug_beta,
        debug_beta_max=debug_beta_max,
    )

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
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")

    parser.add_argument("--returns_dir", type=str, default="yfinance")
    parser.add_argument("--responses_dir", type=str, default="responses")
    parser.add_argument("--dataset_csv", type=str, default="yfinance/filtered_sp500_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--risk_free_rate", type=float, default=0.02)
    parser.add_argument("--min_tickers", type=int, default=25)

    parser.add_argument("--fail_fast", action="store_true", help="Stop on first error.")

    # NEW: debug betas
    parser.add_argument(
        "--debug_beta",
        action="store_true",
        help="Print tickers with insufficient data when computing betas (also prints the returns file).",
    )
    parser.add_argument("--debug_beta_max", type=int, default=10, help="How many tickers to print when debug_beta is on.")

    args = parser.parse_args()

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
                    returns_dir=args.returns_dir,
                    responses_dir=args.responses_dir,
                    dataset_csv_path=args.dataset_csv,
                    risk_free_rate_annual=float(args.risk_free_rate),
                    min_tickers=int(args.min_tickers),
                    debug_beta=bool(args.debug_beta),
                    debug_beta_max=int(args.debug_beta_max),
                )
                results[(start_date, end_date)] = w
            except Exception as e:
                msg = f"[{model}] Error period {start_date} -> {end_date}: {e}"
                if args.fail_fast:
                    raise
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
