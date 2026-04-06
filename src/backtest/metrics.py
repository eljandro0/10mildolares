from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float((excess.mean() / vol) * np.sqrt(252))


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    roll_max = curve.cummax()
    dd = (curve / roll_max) - 1.0
    return float(dd.min())


def win_rate(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    if aligned.empty:
        return 0.0
    return float((aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean())


def rolling_alpha(port_ret: pd.Series, bench_ret: pd.Series, window: int = 30) -> pd.Series:
    aligned = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    alpha = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return alpha.rolling(window=window).mean()

