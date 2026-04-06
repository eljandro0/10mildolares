from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.io import read_csv_required
from portfolio.signals import (
    compute_daily_returns,
    factor_proxy_score,
    mean_reversion_score,
    momentum_score,
)


def load_racional_catalog(path: str) -> pd.DataFrame:
    df = read_csv_required(Path(path))
    expected = {"ticker", "name", "instrument_type", "sector", "region", "quality_score", "tradable"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"Catalog missing required columns: {sorted(missing)}")
    return df[df["tradable"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()


def select_assets_independent(
    prices: pd.DataFrame,
    catalog: pd.DataFrame,
    n_assets: int = 8,
    strategy_mix: dict[str, float] | None = None,
) -> pd.DataFrame:
    # Benchmark data is not an input here by design.
    mix = strategy_mix or {"momentum": 0.45, "mean_rev": 0.20, "factor": 0.25, "tactical": 0.10}

    ret_df = compute_daily_returns(prices)
    mom = momentum_score(ret_df)
    mr = mean_reversion_score(ret_df)
    fct = factor_proxy_score(catalog)
    tactical = _tactical_score(catalog)

    score_df = pd.DataFrame({"momentum": mom, "mean_rev": mr}).join(
        pd.DataFrame({"factor": fct, "tactical": tactical}),
        how="outer",
    )
    score_df = score_df.fillna(0.0)
    for col in score_df.columns:
        std = score_df[col].std(ddof=0)
        if std and std > 0:
            score_df[col] = (score_df[col] - score_df[col].mean()) / std
        else:
            score_df[col] = 0.0
    score_df["total_score"] = (
        mix["momentum"] * score_df["momentum"]
        + mix["mean_rev"] * score_df["mean_rev"]
        + mix["factor"] * score_df["factor"]
        + mix["tactical"] * score_df["tactical"]
    )
    selected = score_df.sort_values("total_score", ascending=False).head(n_assets)
    return selected.reset_index(names="ticker")


def _tactical_score(catalog: pd.DataFrame) -> pd.Series:
    cyclical = catalog["sector"].astype(str).str.lower().isin(
        ["technology", "energy", "financials", "industrials"]
    )
    regional_boost = catalog["region"].astype(str).str.lower().isin(["us", "global"])
    return pd.Series(
        0.6 * cyclical.astype(float).values + 0.4 * regional_boost.astype(float).values,
        index=catalog["ticker"],
    )
