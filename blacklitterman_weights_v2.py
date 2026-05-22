"""
Black-Litterman weights with recommended LLM-view calibrations.

This script is designed to be compatible with the original
blacklitterman_weights.py output format:

    results/{portfolio_name}_black_litterman_market_implied_weights_tau_{tau}.csv

Each output file has:
    Date, <ticker_1>, <ticker_2>, ...

The portfolio_name includes the base LLM model and the tested calibration, e.g.:
    gemma3_blend_alpha_0.25_omega_1_black_litterman_market_implied_weights_tau_0.025.csv

You can then pass that portfolio_name as a model name to returns_from_weights.py, e.g.:
    python returns_from_weights.py --models gemma3_blend_alpha_0.25_omega_1 --tau 0.025

Expected inputs:
    data/filtered_sp25_data.csv
    responses/{model}_{start_date}_{end_date}.json
    data/DGS1.csv

Author: generated for Alexis Perron's BL-LLM project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

OMEGA_FLOOR = 1e-4


# -----------------------------------------------------------------------------
# Recommended specifications from the robustness notebook
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantSpec:
    """One BL-LLM calibration to run for a base model."""
    suffix: str
    view_mode: str = "absolute"  # "absolute" or "rank"
    omega_multiplier: float = 1.0
    blend_alpha: float = 1.0
    rank_quantile: Optional[float] = None
    rank_view_strength: Optional[float] = None

    def portfolio_name(self, model: str) -> str:
        return f"{model}_{self.suffix}"


# Main specifications to carry into analysis notebooks.
# gemma3: prudent / balanced / aggressive + rank robustness
# gpt: balanced / aggressive + rank robustness
# llama & qwen: conservative blend, because diagnostics showed weak active signal
RECOMMENDED_VARIANTS: Dict[str, List[VariantSpec]] = {
    "gemma3": [
        VariantSpec(suffix="blend_alpha_0.25_omega_1", view_mode="absolute", omega_multiplier=1.0, blend_alpha=0.25),
        VariantSpec(suffix="blend_alpha_0.5_omega_1", view_mode="absolute", omega_multiplier=1.0, blend_alpha=0.50),
        VariantSpec(suffix="omega_x25", view_mode="absolute", omega_multiplier=25.0, blend_alpha=1.0),
        VariantSpec(suffix="rank_q0.3_strength_1_omega_10", view_mode="rank", omega_multiplier=10.0, blend_alpha=1.0, rank_quantile=0.30, rank_view_strength=1.0),
    ],
    "gpt": [
        VariantSpec(suffix="blend_alpha_0.5_omega_1", view_mode="absolute", omega_multiplier=1.0, blend_alpha=0.50),
        VariantSpec(suffix="omega_x25", view_mode="absolute", omega_multiplier=25.0, blend_alpha=1.0),
        VariantSpec(suffix="rank_q0.2_strength_1_omega_10", view_mode="rank", omega_multiplier=10.0, blend_alpha=1.0, rank_quantile=0.20, rank_view_strength=1.0),
    ],
    "llama": [
        VariantSpec(suffix="blend_alpha_0.1_omega_1", view_mode="absolute", omega_multiplier=1.0, blend_alpha=0.10),
    ],
    "qwen": [
        VariantSpec(suffix="blend_alpha_0.1_omega_1", view_mode="absolute", omega_multiplier=1.0, blend_alpha=0.10),
    ],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 1e-12:
        return np.ones_like(w, dtype=float) / max(1, w.size)
    return w / s


def _mean_variance_optimize(
    exp_rets: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """
    Long-only fully-invested optimization:

        min_w  w'Σw - risk_aversion * w'μ
        s.t.   sum(w)=1, 0<=w<=1

    The robustness notebook used risk_aversion = 2 / market_risk_aversion
    because the objective is written without the usual 1/2 before variance.
    """
    exp_rets = np.asarray(exp_rets, dtype=float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = int(exp_rets.shape[0])
    if n == 0:
        return np.array([], dtype=float)

    if n == 1:
        return np.ones(1, dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(w.T @ cov_matrix @ w) - float(risk_aversion) * float(w @ exp_rets)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.ones(n, dtype=float) / n

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000},
    )

    if (not res.success) or (not np.isfinite(res.x).all()):
        return x0

    return _safe_normalize_weights(res.x)


def _load_responses_for_period(responses_dir: Path, model: str, start_date: str, end_date: str) -> Dict[str, dict]:
    path = responses_dir / f"{model}_{start_date}_{end_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM response file: {path}")

    raw = load_llm_responses(str(path))
    out: Dict[str, dict] = {}
    for k, v in raw.items():
        kk = _normalize_ticker(k)
        if isinstance(v, dict):
            out[kk] = v
    if len(out) < 2:
        raise ValueError(f"Response file contains <2 usable tickers: {path}")
    return out


def _build_absolute_view_maps(
    responses: Dict[str, dict],
    tickers: List[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    q_map: Dict[str, float] = {}
    omega_map: Dict[str, float] = {}
    n_map: Dict[str, int] = {}

    for t in tickers:
        vals = responses.get(t, {}).get("expected_return", None)
        if vals is None:
            continue
        if not isinstance(vals, list):
            vals = [vals]

        try:
            q, v, n = _robust_view_stats(np.asarray(vals, dtype=float))
        except Exception:
            continue

        if np.isfinite(q) and np.isfinite(v) and int(n) > 0:
            q_map[t] = float(q)
            omega_map[t] = float(max(OMEGA_FLOOR, v))
            n_map[t] = int(n)

    return q_map, omega_map, n_map


def _compute_market_prior(
    returns_df: pd.DataFrame,
    market_caps: Dict[str, float],
    risk_free_monthly: Optional[pd.Series],
    market_risk_aversion: float,
    estimate_market_risk_aversion: bool,
    debug_tag: str = "",
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute classical market-implied prior:

        pi = delta * Sigma @ w_mkt

    Returns aligned tickers, pi, sigma, and w_mkt.
    """
    caps_s = pd.Series({_normalize_ticker(k): v for k, v in market_caps.items()}, dtype=float)
    caps_s = caps_s.replace([np.inf, -np.inf], np.nan).dropna()
    caps_s = caps_s[caps_s > 0]

    common = [t for t in returns_df.columns if t in caps_s.index]
    common = [t for t in common if int(returns_df[t].notna().sum()) >= int(MIN_RET_ROWS)]
    if len(common) < 2:
        raise ValueError("Not enough tickers after intersecting returns and positive market caps.")

    R = returns_df[common].copy()
    caps_s = caps_s.loc[common].astype(float)
    w_mkt = (caps_s / caps_s.sum()).to_numpy(dtype=float)
    sigma = _robust_covariance_psd(R)

    delta = float(market_risk_aversion)
    if estimate_market_risk_aversion:
        mkt = R.mul(caps_s / caps_s.sum(), axis=1).sum(axis=1).dropna()
        if risk_free_monthly is None:
            rf = pd.Series(0.0, index=mkt.index, dtype=float)
        else:
            rf = risk_free_monthly.reindex(mkt.index).astype(float).ffill().bfill()
        mkt_var = float(mkt.var(ddof=1))
        mkt_excess_mean = float((mkt - rf.loc[mkt.index]).mean())
        delta_hat = mkt_excess_mean / mkt_var if mkt_var > 1e-18 else np.nan
        if np.isfinite(delta_hat) and delta_hat > 0:
            delta = float(delta_hat)
        else:
            print(f"{debug_tag} estimated delta invalid ({delta_hat}); fallback to fixed delta={market_risk_aversion}.")

    pi = float(delta) * (sigma @ w_mkt)
    pi = _clip_posterior(pi)
    if not np.isfinite(pi).all():
        raise ValueError("Market-implied prior contains NaN/inf.")

    return common, pi, sigma, _safe_normalize_weights(w_mkt)


def _bl_weights_absolute_views(
    responses: Dict[str, dict],
    returns_df: pd.DataFrame,
    tickers: List[str],
    pi: np.ndarray,
    tau: float,
    omega_multiplier: float,
    portfolio_risk_aversion: float,
) -> np.ndarray:
    q_map, omega_map, _ = _build_absolute_view_maps(responses, tickers)

    keep = [t for t in tickers if t in q_map and int(returns_df[t].notna().sum()) >= int(MIN_RET_ROWS)]
    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    idx = [tickers.index(t) for t in keep]
    pi_k = np.asarray(pi, dtype=float)[idx]
    sigma_k = _robust_covariance_psd(returns_df[keep].copy())

    Q = np.array([q_map[t] for t in keep], dtype=float)
    Omega = np.diag([max(OMEGA_FLOOR, omega_map[t]) * float(omega_multiplier) for t in keep])
    P = np.eye(len(keep), dtype=float)

    inv_tau_sigma = np.linalg.pinv(float(tau) * sigma_k)
    inv_omega = np.linalg.pinv(Omega)
    M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_omega @ P)

    posterior = M @ (inv_tau_sigma @ pi_k + P.T @ inv_omega @ Q)
    posterior = _clip_posterior(posterior)

    w_k = _mean_variance_optimize(posterior, sigma_k, portfolio_risk_aversion)

    w = np.zeros(len(tickers), dtype=float)
    for t, wt in zip(keep, w_k):
        w[tickers.index(t)] = float(wt)
    return _safe_normalize_weights(w)


def _bl_weights_rank_views(
    responses: Dict[str, dict],
    returns_df: pd.DataFrame,
    tickers: List[str],
    pi: np.ndarray,
    tau: float,
    omega_multiplier: float,
    rank_quantile: float,
    rank_view_strength: float,
    portfolio_risk_aversion: float,
) -> np.ndarray:
    q_map, omega_map, _ = _build_absolute_view_maps(responses, tickers)

    scores = pd.Series({t: q_map[t] for t in tickers if t in q_map}, dtype=float).dropna()
    scores = scores[[t for t in scores.index if t in tickers and int(returns_df[t].notna().sum()) >= int(MIN_RET_ROWS)]]
    if len(scores) < 5:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    lo = scores.quantile(float(rank_quantile))
    hi = scores.quantile(1.0 - float(rank_quantile))
    selected = scores[(scores <= lo) | (scores >= hi)].copy()
    keep = list(selected.index)
    if len(keep) < 2:
        return np.ones(len(tickers), dtype=float) / max(1, len(tickers))

    idx = [tickers.index(t) for t in keep]
    pi_k = np.asarray(pi, dtype=float)[idx]
    sigma_k = _robust_covariance_psd(returns_df[keep].copy())

    # Rank-only transformation from robustness notebook.
    ranks = selected.rank(pct=True).reindex(keep).to_numpy(dtype=float)
    z = (ranks - 0.5) / 0.5  # roughly [-1, 1]

    pi_scale = float(np.nanstd(pi))
    if not np.isfinite(pi_scale) or pi_scale < 1e-6:
        pi_scale = 0.01

    Q = pi_k + float(rank_view_strength) * pi_scale * z
    Omega = np.diag([max(OMEGA_FLOOR, omega_map.get(t, OMEGA_FLOOR)) * float(omega_multiplier) for t in keep])
    P = np.eye(len(keep), dtype=float)

    inv_tau_sigma = np.linalg.pinv(float(tau) * sigma_k)
    inv_omega = np.linalg.pinv(Omega)
    M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_omega @ P)

    posterior = M @ (inv_tau_sigma @ pi_k + P.T @ inv_omega @ Q)
    posterior = _clip_posterior(posterior)

    w_k = _mean_variance_optimize(posterior, sigma_k, portfolio_risk_aversion)

    w = np.zeros(len(tickers), dtype=float)
    for t, wt in zip(keep, w_k):
        w[tickers.index(t)] = float(wt)
    return _safe_normalize_weights(w)


def _apply_variant(
    spec: VariantSpec,
    responses: Dict[str, dict],
    returns_df: pd.DataFrame,
    tickers: List[str],
    pi: np.ndarray,
    w_mkt: np.ndarray,
    tau: float,
    portfolio_risk_aversion: float,
) -> np.ndarray:
    if spec.view_mode == "absolute":
        w_bl = _bl_weights_absolute_views(
            responses=responses,
            returns_df=returns_df,
            tickers=tickers,
            pi=pi,
            tau=tau,
            omega_multiplier=spec.omega_multiplier,
            portfolio_risk_aversion=portfolio_risk_aversion,
        )
    elif spec.view_mode == "rank":
        if spec.rank_quantile is None or spec.rank_view_strength is None:
            raise ValueError(f"Rank variant missing rank parameters: {spec}")
        w_bl = _bl_weights_rank_views(
            responses=responses,
            returns_df=returns_df,
            tickers=tickers,
            pi=pi,
            tau=tau,
            omega_multiplier=spec.omega_multiplier,
            rank_quantile=spec.rank_quantile,
            rank_view_strength=spec.rank_view_strength,
            portfolio_risk_aversion=portfolio_risk_aversion,
        )
    else:
        raise ValueError(f"Unknown view_mode: {spec.view_mode}")

    w_mkt = _safe_normalize_weights(w_mkt)
    alpha = float(spec.blend_alpha)
    w_final = (1.0 - alpha) * w_mkt + alpha * w_bl
    return _safe_normalize_weights(w_final)


def _process_period(
    model: str,
    spec: VariantSpec,
    start_date: str,
    end_date: str,
    tau: float,
    dataset_df: pd.DataFrame,
    responses_dir: Path,
    rf_monthly_annual: pd.Series,
    market_risk_aversion: float,
    portfolio_risk_aversion: float,
    estimate_market_risk_aversion: bool,
) -> pd.Series:
    responses = _load_responses_for_period(responses_dir, model, start_date, end_date)

    # Use response tickers to keep the same investable universe convention as the original LLM BL script.
    returns_df = load_returns_window_from_dataset(
        dataset_df,
        window_end_date=end_date,
        lookback_months=LOOKBACK_MONTHS,
        tickers=list(responses.keys()),
    )
    if returns_df.shape[0] < int(MIN_RET_ROWS):
        raise ValueError(f"Returns window too short: rows={returns_df.shape[0]}")
    if returns_df.shape[1] < 2:
        raise ValueError("Returns window contains <2 usable tickers.")

    common_cols = [c for c in returns_df.columns if c in responses]
    if len(common_cols) < 2:
        raise ValueError("Not enough overlapping tickers between returns and responses.")
    returns_df = returns_df[common_cols].copy()

    market_caps = load_market_caps_from_dataset(dataset_df, end_date)
    rf_monthly = align_monthly_rf_to_returns_index(rf_monthly_annual, returns_df.index)

    tickers, pi, _sigma, w_mkt = _compute_market_prior(
        returns_df=returns_df,
        market_caps=market_caps,
        risk_free_monthly=rf_monthly,
        market_risk_aversion=market_risk_aversion,
        estimate_market_risk_aversion=estimate_market_risk_aversion,
        debug_tag=f"[{model}:{spec.suffix}] {start_date}->{end_date}",
    )

    returns_df = returns_df[tickers].copy()
    w = _apply_variant(
        spec=spec,
        responses=responses,
        returns_df=returns_df,
        tickers=tickers,
        pi=pi,
        w_mkt=w_mkt,
        tau=tau,
        portfolio_risk_aversion=portfolio_risk_aversion,
    )

    return pd.Series(w, index=tickers)


def _write_results_csv(results: Dict[Tuple[str, str], pd.Series], out_path: Path) -> None:
    df = pd.DataFrame(results).T
    df = df.reset_index()

    if "index" in df.columns:
        df[["start_date", "end_date"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
        df = df.drop(columns=["index"])
    elif "level_0" in df.columns and "level_1" in df.columns:
        df = df.rename(columns={"level_0": "start_date", "level_1": "end_date"})
    else:
        raise ValueError("Unexpected DataFrame index format after reset_index.")

    # Same compatibility convention as original script: Date is the weight month start date.
    df["Date"] = df["start_date"]
    df = df.drop(columns=["start_date", "end_date"])

    # Put Date first.
    cols = ["Date"] + [c for c in df.columns if c != "Date"]
    df = df[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved weights: {out_path} | shape={df.shape}")


def _selected_variants(models: List[str], include_rank: bool) -> Dict[str, List[VariantSpec]]:
    selected: Dict[str, List[VariantSpec]] = {}
    for m in models:
        m_clean = str(m).strip()
        if m_clean not in RECOMMENDED_VARIANTS:
            print(f"Warning: no recommended variants configured for model '{m_clean}'. Skipping.")
            continue
        specs = RECOMMENDED_VARIANTS[m_clean]
        if not include_rank:
            specs = [s for s in specs if s.view_mode != "rank"]
        selected[m_clean] = specs
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recommended BL-LLM calibrations and export original-format weight CSV files.")
    parser.add_argument("--models", nargs="+", default=["gemma3", "gpt", "llama", "qwen"],
                        help="Base LLM models to run. Defaults: gemma3 gpt llama qwen.")
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-06-30")
    parser.add_argument("--responses_dir", type=str, default="responses")
    parser.add_argument("--dataset_csv", type=str, default="data/filtered_sp25_data.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--risk_free_csv", type=str, default="data/DGS1.csv")
    parser.add_argument("--market_risk_aversion", type=float, default=1.0,
                        help="Delta in pi = delta * Sigma @ w_mkt.")
    parser.add_argument("--portfolio_risk_aversion", type=float, default=None,
                        help="Final optimizer lambda. Default is 2 / market_risk_aversion, matching robustness notebook.")
    parser.add_argument("--estimate_market_risk_aversion", action="store_true")
    parser.add_argument("--include_rank", action="store_true",
                        help="Also export the rank-based robustness variants for gemma3 and gpt.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute files even if the output CSV already exists.")

    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    market_risk_aversion = float(args.market_risk_aversion)
    if args.portfolio_risk_aversion is None:
        portfolio_risk_aversion = 2.0 / market_risk_aversion
    else:
        portfolio_risk_aversion = float(args.portfolio_risk_aversion)

    print("Loading dataset and risk-free series...")
    dataset_df = prepare_dataset(args.dataset_csv)
    rf_monthly_annual = load_risk_free_monthly_annual(args.risk_free_csv)
    periods = month_pairs(args.start, args.end)

    variant_map = _selected_variants(args.models, include_rank=bool(args.include_rank))
    if not variant_map:
        raise ValueError("No variants selected. Check --models.")

    print(f"Market risk aversion delta: {market_risk_aversion}")
    print(f"Portfolio optimizer risk_aversion lambda: {portfolio_risk_aversion}")
    print(f"Periods: {args.start} -> {args.end} | tau={args.tau}")

    for model, specs in variant_map.items():
        for spec in specs:
            portfolio_name = spec.portfolio_name(model)
            out_path = results_dir / f"{portfolio_name}_black_litterman_market_implied_weights_tau_{args.tau}.csv"

            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
                continue

            print(
                f"\n=== {portfolio_name} | mode={spec.view_mode} | "
                f"omega={spec.omega_multiplier:g} | alpha={spec.blend_alpha:g} ==="
            )

            results: Dict[Tuple[str, str], pd.Series] = {}
            for start_date, end_date in tqdm(periods, desc=portfolio_name):
                try:
                    w = _process_period(
                        model=model,
                        spec=spec,
                        start_date=start_date,
                        end_date=end_date,
                        tau=float(args.tau),
                        dataset_df=dataset_df,
                        responses_dir=responses_dir,
                        rf_monthly_annual=rf_monthly_annual,
                        market_risk_aversion=market_risk_aversion,
                        portfolio_risk_aversion=portfolio_risk_aversion,
                        estimate_market_risk_aversion=bool(args.estimate_market_risk_aversion),
                    )
                    results[(start_date, end_date)] = w
                except Exception as e:
                    print(f"[{portfolio_name}] Error period {start_date} -> {end_date}: {e}")

            if not results:
                print(f"[{portfolio_name}] No results produced.")
                continue

            _write_results_csv(results, out_path)


if __name__ == "__main__":
    main()
