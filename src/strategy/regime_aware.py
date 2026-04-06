from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import requests


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    asof: date
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    spy_drawdown_252: float
    spy_mom_63d: float
    vix_close: float
    vix_pctile_252: float
    regime: str


@dataclass(frozen=True)
class BacktestComparison:
    metrics: pd.DataFrame
    equity_curves: pd.DataFrame
    event_study: pd.DataFrame
    deployment_log: pd.DataFrame


def fetch_price_panel(
    tickers: Iterable[str],
    start: date,
    end: date,
    include_vix: bool = True,
    timeout_sec: int = 25,
) -> pd.DataFrame:
    universe = list(dict.fromkeys(tickers))
    if "SPY" not in universe:
        universe.append("SPY")
    if include_vix and "^VIX" not in universe:
        universe.append("^VIX")

    rows: list[pd.DataFrame] = []
    for tk in universe:
        rows.append(_fetch_yahoo_daily(tk, start, end, timeout_sec))
    out = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"])
    return out.reset_index(drop=True)


def compute_market_indicators(price_df: pd.DataFrame) -> MarketRegimeSnapshot:
    piv = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
    if "SPY" not in piv.columns:
        raise ValueError("SPY is required to compute market indicators")
    if "^VIX" not in piv.columns:
        raise ValueError("^VIX is required to compute market indicators")
    spy = piv["SPY"].dropna()
    vix = piv["^VIX"].dropna()
    common = spy.index.intersection(vix.index)
    spy = spy.reindex(common).dropna()
    vix = vix.reindex(common).dropna()
    if len(spy) < 220:
        raise ValueError("Need at least 220 sessions for robust regime metrics")

    sma50 = spy.rolling(50).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    rolling_max_252 = spy.rolling(252).max().iloc[-1]
    drawdown_252 = spy.iloc[-1] / rolling_max_252 - 1.0
    mom_63 = spy.iloc[-1] / spy.iloc[-64] - 1.0
    vix_hist = vix.iloc[-252:]
    vix_pctile = float((vix_hist <= vix.iloc[-1]).mean())

    regime = classify_market_regime(
        spy_close=float(spy.iloc[-1]),
        sma50=float(sma50),
        sma200=float(sma200),
        drawdown_252=float(drawdown_252),
        mom_63d=float(mom_63),
        vix=float(vix.iloc[-1]),
        vix_pctile=vix_pctile,
    )
    return MarketRegimeSnapshot(
        asof=spy.index[-1].date(),
        spy_close=float(spy.iloc[-1]),
        spy_sma50=float(sma50),
        spy_sma200=float(sma200),
        spy_drawdown_252=float(drawdown_252),
        spy_mom_63d=float(mom_63),
        vix_close=float(vix.iloc[-1]),
        vix_pctile_252=float(vix_pctile),
        regime=regime,
    )


def classify_market_regime(
    spy_close: float,
    sma50: float,
    sma200: float,
    drawdown_252: float,
    mom_63d: float,
    vix: float,
    vix_pctile: float,
) -> str:
    bearish_trend = spy_close < sma200 and sma50 < sma200
    stressed = vix >= 24 or vix_pctile >= 0.80
    deep_drawdown = drawdown_252 <= -0.10
    positive_trend = spy_close > sma50 > sma200 and mom_63d > 0

    if deep_drawdown and (mom_63d > -0.03 or vix > 26):
        return "oportunidad"
    if bearish_trend and stressed:
        return "riesgo_alto"
    if positive_trend and not stressed:
        return "neutro"
    if drawdown_252 <= -0.06 and stressed:
        return "riesgo_alto"
    return "neutro"


def build_recommended_weights(event_study_df: pd.DataFrame) -> pd.Series:
    latest = (
        event_study_df.groupby("ticker")[["ret_6m", "ret_12m"]]
        .mean()
        .assign(score=lambda d: 0.6 * d["ret_6m"] + 0.4 * d["ret_12m"])
    )
    conv = latest["score"].sort_values(ascending=False)
    top = conv.head(8)
    min_w = 0.06
    raw = top.clip(lower=top.min() * 0.6)
    scaled = raw / raw.sum()
    scaled = scaled * (1.0 - min_w * len(scaled)) + min_w
    scaled = scaled / scaled.sum()
    return scaled.round(6)


def run_event_study(price_df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    piv = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
    events = [
        ("post_invasion_ukr", pd.Timestamp("2022-02-24")),
        ("energy_shock_pivot", pd.Timestamp("2021-10-01")),
        ("post_chatgpt_launch", pd.Timestamp("2022-11-30")),
    ]
    rows: list[dict[str, object]] = []
    for event_name, event_dt in events:
        entry_dt = _next_trading_date(piv.index, event_dt)
        for tk in tickers:
            if tk not in piv.columns:
                continue
            px = piv[tk].dropna()
            if entry_dt not in px.index:
                continue
            entry_px = float(px.loc[entry_dt])
            r3 = _forward_return(px, entry_dt, 63)
            r6 = _forward_return(px, entry_dt, 126)
            r12 = _forward_return(px, entry_dt, 252)
            rows.append(
                {
                    "event": event_name,
                    "event_date": event_dt.date().isoformat(),
                    "entry_date": entry_dt.date().isoformat(),
                    "ticker": tk,
                    "ret_3m": r3,
                    "ret_6m": r6,
                    "ret_12m": r12,
                    "entry_px": entry_px,
                }
            )
    return pd.DataFrame(rows)


def compare_entry_methods(
    price_df: pd.DataFrame,
    investable_weights: pd.Series,
    market_regime: str,
    start: date,
    end: date,
    initial_capital: float = 100_000.0,
) -> BacktestComparison:
    piv = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
    piv = piv.loc[(piv.index.date >= start) & (piv.index.date <= end)]
    if len(piv) < 200:
        raise ValueError("Need at least ~200 sessions for 12m backtest window")
    piv = piv.dropna(subset=["SPY"])
    investable = [t for t in investable_weights.index if t in piv.columns]
    if not investable:
        raise ValueError("No investable tickers available in price panel")

    schedule = _tranche_schedule(piv.index, market_regime, start)
    im_eq, im_log = _simulate_strategy(
        prices=piv[investable + ["SPY"]],
        weights=investable_weights.reindex(investable).fillna(0.0),
        schedule_dates=[piv.index[0]],
        schedule_weights=[1.0],
        initial_capital=initial_capital,
        asymmetric=True,
    )
    st_eq, st_log = _simulate_strategy(
        prices=piv[investable + ["SPY"]],
        weights=investable_weights.reindex(investable).fillna(0.0),
        schedule_dates=schedule["date"],
        schedule_weights=schedule["weight"],
        initial_capital=initial_capital,
        asymmetric=True,
    )
    bench = piv["SPY"] / piv["SPY"].iloc[0] * initial_capital
    eq = pd.DataFrame(
        {
            "date": piv.index,
            "immediate": im_eq.values,
            "staggered": st_eq.values,
            "spy": bench.values,
        }
    )
    metrics = pd.DataFrame(
        [
            _compute_metrics(eq["immediate"], eq["spy"], "entrada_inmediata"),
            _compute_metrics(eq["staggered"], eq["spy"], "entrada_escalonada"),
        ]
    )
    logs = pd.concat([im_log.assign(strategy="immediate"), st_log.assign(strategy="staggered")], ignore_index=True)
    return BacktestComparison(metrics=metrics, equity_curves=eq, event_study=pd.DataFrame(), deployment_log=logs)


def _simulate_strategy(
    prices: pd.DataFrame,
    weights: pd.Series,
    schedule_dates: Iterable[pd.Timestamp],
    schedule_weights: Iterable[float],
    initial_capital: float,
    asymmetric: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    idx = prices.index
    trade_dates = [d for d in schedule_dates if d in idx]
    trade_w = list(schedule_weights)
    if len(trade_dates) != len(trade_w):
        raise ValueError("schedule_dates and schedule_weights must align")
    date_to_weight = {d: w for d, w in zip(trade_dates, trade_w, strict=True)}

    rebalance_dates = set(prices.resample("ME").last().index)
    holdings = pd.Series(0.0, index=weights.index)
    avg_cost = pd.Series(np.nan, index=weights.index)
    locked_cash = initial_capital
    liquid_cash = 0.0
    logs: list[dict[str, object]] = []
    eq = pd.Series(index=idx, dtype=float)
    last_tactical_unlock: pd.Timestamp | None = None
    spy = prices["SPY"].copy()
    spy_roll_max = spy.cummax()

    for dt in idx:
        px = prices.loc[dt, weights.index]
        portfolio_value = float((holdings * px).sum() + liquid_cash + locked_cash)
        eq.loc[dt] = portfolio_value

        # New tranche unlock on scheduled dates.
        if dt in date_to_weight:
            tranche = initial_capital * float(date_to_weight[dt])
            unlocked = min(tranche, locked_cash)
            locked_cash -= unlocked
            liquid_cash += unlocked
            budget = liquid_cash
            _buy_to_target(
                holdings=holdings,
                avg_cost=avg_cost,
                prices=px,
                buy_budget=budget,
                target_weights=weights,
            )
            liquid_cash -= budget
            logs.append({"date": dt.date().isoformat(), "event": "tranche", "amount": round(unlocked, 2)})

        # Tactical acceleration: deploy more when market sells off hard.
        dd_now = float(spy.loc[dt] / spy_roll_max.loc[dt] - 1.0)
        cooldown_ok = last_tactical_unlock is None or (dt - last_tactical_unlock).days >= 10
        if locked_cash > 0 and cooldown_ok and dd_now <= -0.10:
            extra_unlock = min(locked_cash, initial_capital * (0.35 if dd_now <= -0.15 else 0.20))
            locked_cash -= extra_unlock
            liquid_cash += extra_unlock
            _buy_to_target(holdings, avg_cost, px, liquid_cash, weights)
            liquid_cash = 0.0
            last_tactical_unlock = dt
            logs.append(
                {
                    "date": dt.date().isoformat(),
                    "event": "drawdown_accelerator",
                    "amount": round(extra_unlock, 2),
                    "drawdown": round(dd_now, 4),
                }
            )

        if dt in rebalance_dates:
            if asymmetric:
                sells = _asymmetric_sells(holdings, avg_cost, px, weights, portfolio_value)
                for tk, amount in sells.items():
                    if amount <= 0:
                        continue
                    qty = min(holdings[tk], amount / px[tk])
                    liquid_cash += qty * px[tk]
                    holdings[tk] -= qty
                    logs.append({"date": dt.date().isoformat(), "event": "sell_winner", "ticker": tk, "amount": round(float(qty * px[tk]), 2)})
            # Buy underweight + best momentum sleeve.
            if liquid_cash > 0:
                _buy_to_target(holdings, avg_cost, px, liquid_cash, weights)
                logs.append({"date": dt.date().isoformat(), "event": "rebalance_buy", "amount": round(float(liquid_cash), 2)})
                liquid_cash = 0.0

    return eq, pd.DataFrame(logs)


def _asymmetric_sells(
    holdings: pd.Series,
    avg_cost: pd.Series,
    prices: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
) -> dict[str, float]:
    sells: dict[str, float] = {}
    current_values = holdings * prices
    current_weights = current_values / max(portfolio_value, 1e-12)
    for tk in holdings.index:
        if holdings[tk] <= 0 or np.isnan(avg_cost[tk]) or avg_cost[tk] <= 0:
            continue
        pnl = float(prices[tk] / avg_cost[tk] - 1.0)
        if pnl <= 0:
            continue
        excess_w = float(current_weights[tk] - target_weights.get(tk, 0.0))
        if excess_w <= 0:
            continue
        sells[tk] = portfolio_value * excess_w
    return sells


def _buy_to_target(
    holdings: pd.Series,
    avg_cost: pd.Series,
    prices: pd.Series,
    buy_budget: float,
    target_weights: pd.Series,
) -> None:
    if buy_budget <= 0:
        return
    alloc = target_weights / target_weights.sum()
    for tk in alloc.index:
        amount = buy_budget * float(alloc[tk])
        if prices[tk] <= 0:
            continue
        qty = amount / float(prices[tk])
        old_qty = holdings[tk]
        new_qty = old_qty + qty
        if new_qty <= 0:
            continue
        if old_qty <= 0 or np.isnan(avg_cost[tk]):
            avg_cost[tk] = float(prices[tk])
        else:
            avg_cost[tk] = float((old_qty * avg_cost[tk] + qty * prices[tk]) / new_qty)
        holdings[tk] = new_qty


def _compute_metrics(port_curve: pd.Series, bench_curve: pd.Series, label: str) -> dict[str, object]:
    port_ret = port_curve.pct_change().fillna(0.0)
    bench_ret = bench_curve.pct_change().fillna(0.0)
    alpha = port_ret - bench_ret
    sharpe = 0.0 if port_ret.std(ddof=0) == 0 else (port_ret.mean() / port_ret.std(ddof=0)) * np.sqrt(252)
    dd = (port_curve / port_curve.cummax() - 1.0).min()
    return {
        "strategy": label,
        "retorno_total": float(port_curve.iloc[-1] / port_curve.iloc[0] - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "alpha_vs_spy": float(alpha.mean() * 252),
    }


def _tranche_schedule(index: pd.DatetimeIndex, regime: str, start: date) -> pd.DataFrame:
    start_ts = _next_trading_date(index, pd.Timestamp(start))
    if regime == "riesgo_alto":
        offsets = [0, 21, 42, 63, 84]
        w = [0.15, 0.15, 0.20, 0.25, 0.25]
    elif regime == "oportunidad":
        offsets = [0, 14, 28]
        w = [0.50, 0.30, 0.20]
    else:
        offsets = [0, 14, 35]
        w = [0.45, 0.35, 0.20]
    dates = [_next_trading_date(index, start_ts + timedelta(days=off)) for off in offsets]
    out = pd.DataFrame({"date": dates, "weight": w}).drop_duplicates(subset=["date"]).reset_index(drop=True)
    out["weight"] = out["weight"] / out["weight"].sum()
    return out


def _forward_return(px: pd.Series, start_dt: pd.Timestamp, horizon: int) -> float:
    loc = px.index.get_indexer([start_dt], method="nearest")[0]
    end_loc = min(loc + horizon, len(px) - 1)
    return float(px.iloc[end_loc] / px.iloc[loc] - 1.0)


def _next_trading_date(index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    if d in index:
        return d
    future = index[index >= d]
    if len(future) == 0:
        return index[-1]
    return future[0]


def _fetch_yahoo_daily(ticker: str, start: date, end: date, timeout_sec: int) -> pd.DataFrame:
    params = {
        "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    url = YAHOO_CHART_URL.format(ticker=ticker)
    resp = requests.get(url, params=params, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    data = resp.json()["chart"]["result"][0]
    ts = data["timestamp"]
    quotes = data["indicators"]["quote"][0]
    close = quotes["close"]
    df = pd.DataFrame({"timestamp": ts, "close": close})
    df["date"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    df = df.dropna(subset=["close"])
    df["ticker"] = ticker
    return df[["date", "ticker", "close"]]
