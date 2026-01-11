import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
from pathlib import Path
from tqdm import tqdm


def black_litterman_LLM(data_dict, returns, tickers, market_equilibrium_return, tau):
    """
    Black-Litterman with LLM views (Q) as posterior expected returns.
    - data_dict[ticker]["expected_return"] is assumed to be a list of samples.
    """
    Q = np.array([np.mean(data_dict[ticker]["expected_return"]) for ticker in tickers], dtype=float)
    P = np.eye(len(tickers))
    Omega = np.diag([np.var(data_dict[ticker]["expected_return"]) for ticker in tickers]).astype(float)

    sigma = np.cov(returns.T)
    tau_sigma = tau * sigma

    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_Omega = np.linalg.pinv(Omega)
    M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_Omega @ P)

    posterior_returns = M @ (inv_tau_sigma @ market_equilibrium_return + P.T @ inv_Omega @ Q)

    def portfolio_variance(w, cov_matrix):
        return float(w.T @ cov_matrix @ w)

    def objective_function(w, exp_rets, cov_matrix, risk_aversion=0.1):
        # mean-variance (min var - lambda*mu)
        return portfolio_variance(w, cov_matrix) - float(risk_aversion) * float(w @ exp_rets)

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # fully invested
        {"type": "ineq", "fun": lambda x: x},              # long-only
    )
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))

    x0 = np.ones(len(tickers), dtype=float) / len(tickers)
    result = minimize(
        objective_function,
        x0,
        args=(posterior_returns, sigma),
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 1000},
    )

    if not result.success:
        # fallback: equal weights
        return x0

    return result.x


def _parse_yyyymmdd_int(x):
    """
    filtered_sp500_data.csv uses 'date' like 20210129 (int).
    """
    if pd.isna(x):
        return pd.NaT
    s = str(int(x)) if isinstance(x, (int, np.integer, float, np.floating)) and not pd.isna(x) else str(x)
    s_digits = "".join(ch for ch in s if ch.isdigit())
    if len(s_digits) == 8:
        return pd.to_datetime(s_digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_market_caps_from_filtered(filtered_csv_path: str, period_end_date: str) -> dict:
    """
    Build a {tic: market_equity} snapshot using filtered_sp500_data.csv for the month of period_end_date.
    """
    df = pd.read_csv(filtered_csv_path)
    if "date" not in df.columns or "tic" not in df.columns or "market_equity" not in df.columns:
        raise ValueError("filtered_sp500_data.csv must contain columns: date, tic, market_equity")

    df["date_dt"] = df["date"].apply(_parse_yyyymmdd_int)
    df["tic"] = df["tic"].astype(str)

    p = pd.Period(pd.to_datetime(period_end_date), freq="M")
    df = df[df["date_dt"].dt.to_period("M") == p].copy()

    # Keep last available row per ticker (should already be 1 per month)
    df = df.sort_values(["tic", "date_dt"]).drop_duplicates(subset=["tic"], keep="last")

    # market_equity may have NaNs; drop them
    caps = (
        df[["tic", "market_equity"]]
        .dropna(subset=["tic", "market_equity"])
        .set_index("tic")["market_equity"]
        .to_dict()
    )
    return caps


def process_period(start_date, end_date, tau, filtered_csv_path="yfinance/filtered_sp500_data.csv"):
    """
    For a given period, load:
      - returns_{start}_{end}.csv (daily returns matrix)
      - responses/gpt_{start}_{end}.json (LLM expected returns)
      - market caps snapshot from filtered_sp500_data.csv (month = end_date)
    Then compute BL weights.
    """
    # --- Market caps (from filtered_sp500_data.csv) ---
    market_caps = load_market_caps_from_filtered(filtered_csv_path, end_date)

    # --- Returns data ---
    returns = pd.read_csv(f"yfinance/returns_{start_date}_{end_date}.csv", index_col=0)

    # Restrict to tickers with market caps
    returns = returns[returns.columns.intersection(market_caps.keys())]

    # Drop columns with any NaN
    nan_cols = returns.columns[returns.isna().any()]
    returns = returns.dropna(axis=1)

    tickers = returns.columns.tolist()
    if len(tickers) < 2:
        raise ValueError("Not enough tickers after filtering/dropping NaNs to compute covariance.")

    # Also restrict market_caps to remaining tickers
    market_caps_series = pd.Series(market_caps, dtype=float)
    market_caps_series = market_caps_series.loc[market_caps_series.index.intersection(tickers)].dropna()

    if market_caps_series.empty:
        raise ValueError("No valid market caps after intersecting with returns columns.")

    market_cap_weights = market_caps_series / market_caps_series.sum()
    valid_tickers = market_cap_weights.index.tolist()

    # Market cap weighted market return
    market_return_weighted = (returns[valid_tickers] * market_cap_weights).sum(axis=1)

    # --- Market equilibrium returns (CAPM-like) ---
    risk_free_rate = 0.02
    market_var = market_return_weighted.var()
    if market_var == 0 or np.isnan(market_var):
        raise ValueError("Market variance is zero/NaN; cannot compute betas.")

    market_beta = returns[valid_tickers].apply(lambda x: x.cov(market_return_weighted)) / market_var
    market_risk_premium = (market_return_weighted - risk_free_rate).mean()
    market_equilibrium_return = market_beta * market_risk_premium
    market_equilibrium_return = market_equilibrium_return.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    # --- Load LLM responses (GPT) ---
    resp_path = Path(f"responses/gpt_{start_date}_{end_date}.json")
    if not resp_path.exists():
        raise FileNotFoundError(f"Missing LLM response file: {resp_path}")

    with resp_path.open("r", encoding="utf-8") as f:
        gpt_dict = json.load(f)

    # Remove nan columns and stocks not in returns data
    gpt_dict = {k: v for k, v in gpt_dict.items() if k in tickers and k not in nan_cols}

    # Some response files might contain extra tickers not in returns
    tickers_used = [t for t in tickers if t in gpt_dict]
    if len(tickers_used) < 2:
        raise ValueError("Not enough overlapping tickers between returns and GPT responses.")

    # Subset returns matrix to tickers_used (same order)
    returns_used = returns[tickers_used].to_numpy(dtype=float)

    # Subset equilibrium returns to tickers_used (same order)
    # market_equilibrium_return was built aligned to `tickers` earlier
    # Recompute for tickers_used
    eq_map = dict(zip(tickers, market_equilibrium_return))
    mkt_eq_used = np.array([eq_map.get(t, 0.0) for t in tickers_used], dtype=float)

    weights = black_litterman_LLM(gpt_dict, returns_used, tickers_used, mkt_eq_used, tau)
    return pd.Series(weights, index=tickers_used)


def main():
    tau = 0.025  # hyperparameter

    date_pairs = [
        ("2021-01-01", "2021-01-31"),
        ("2021-02-01", "2021-02-28"),
        ("2021-03-01", "2021-03-31"),
        ("2021-04-01", "2021-04-30"),
        ("2021-05-01", "2021-05-31"),
        ("2021-06-01", "2021-06-30"),
        ("2021-07-01", "2021-07-31")
    ]

    gpt_results = {}

    for start_date, end_date in tqdm(date_pairs):
        print(f"Processing period: {start_date} to {end_date}")
        try:
            gpt_weights = process_period(start_date, end_date, tau, filtered_csv_path="yfinance/filtered_sp500_data.csv")
            gpt_results[(start_date, end_date)] = gpt_weights
        except Exception as e:
            print(f"Error processing period {start_date} to {end_date}: {str(e)}")

    gpt_results_df = pd.DataFrame(gpt_results).T
    gpt_results_df = gpt_results_df.reset_index()
    gpt_results_df["Date"] = gpt_results_df["level_0"]
    gpt_results_df = gpt_results_df.drop(["level_0", "level_1"], axis=1)

    Path("results").mkdir(parents=True, exist_ok=True)
    gpt_results_df.to_csv(f"results/gpt_black_litterman_weights_tau_{tau}.csv", index=False)

    print("\nResults shape:")
    print(f"GPT results: {gpt_results_df.shape}")


if __name__ == "__main__":
    main()
