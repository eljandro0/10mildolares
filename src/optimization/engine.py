from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    risk_aversion: float
    alpha_weight: float
    max_weight: float


SCENARIOS: dict[str, ScenarioConfig] = {
    "agresivo": ScenarioConfig("agresivo", risk_aversion=1.0, alpha_weight=1.5, max_weight=0.45),
    "balanceado": ScenarioConfig("balanceado", risk_aversion=2.0, alpha_weight=1.0, max_weight=0.30),
    "defensivo": ScenarioConfig("defensivo", risk_aversion=3.5, alpha_weight=0.6, max_weight=0.22),
}


def optimize_weights(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    scenario: str = "balanceado",
) -> pd.Series:
    cfg = SCENARIOS.get(scenario)
    if cfg is None:
        raise ValueError(f"Unsupported scenario: {scenario}")

    aligned_b = benchmark_returns.reindex(asset_returns.index).fillna(0.0)
    mu = asset_returns.mean().values
    cov = asset_returns.cov().values
    n = asset_returns.shape[1]

    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, cfg.max_weight) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    bmu = aligned_b.mean()

    def objective(w: np.ndarray) -> float:
        p_ret = float(w @ mu)
        p_vol = float(np.sqrt(w @ cov @ w + 1e-12))
        alpha = p_ret - bmu
        score = p_ret + cfg.alpha_weight * alpha - cfg.risk_aversion * p_vol
        return -score

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return pd.Series(x0, index=asset_returns.columns, name="weight")
    return pd.Series(res.x, index=asset_returns.columns, name="weight")

