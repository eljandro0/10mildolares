from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.sort_values(["date", "ticker"]).copy()
    out["ret"] = out.groupby("ticker")["close"].pct_change()
    return out


def momentum_score(ret_df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    piv = ret_df.pivot(index="date", columns="ticker", values="ret")
    score = (1.0 + piv).tail(lookback).prod() - 1.0
    return score.replace([np.inf, -np.inf], np.nan)


def mean_reversion_score(ret_df: pd.DataFrame, lookback: int = 15) -> pd.Series:
    piv = ret_df.pivot(index="date", columns="ticker", values="ret")
    return -piv.tail(lookback).mean()


def factor_proxy_score(
    latest_instruments: pd.DataFrame,
    factor_weights: dict[str, float] | None = None,
) -> pd.Series:
    w = factor_weights or {"quality_score": 1.0}
    score = pd.Series(0.0, index=latest_instruments["ticker"])
    for col, weight in w.items():
        if col in latest_instruments.columns:
            z = _zscore(latest_instruments[col].astype(float))
            score = score + weight * z.values
    return score


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

