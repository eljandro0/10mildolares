from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


DEFAULT_TARGET_WEIGHTS: dict[str, float] = {
    "NVDA": 0.16,
    "MSFT": 0.12,
    "SMH": 0.08,
    "XLE": 0.12,
    "LMT": 0.06,
    "ITA": 0.06,
    "SPY": 0.20,
    "VXUS": 0.20,
}


@dataclass(frozen=True)
class RuleConfig:
    min_profit_to_trim: float = 0.20
    high_profit_to_trim_more: float = 0.40
    overweight_threshold: float = 0.03
    monthly_drawdown_pause: float = -0.08


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly Asymmetric Rebalancing Advisor")
    p.add_argument(
        "--positions",
        default="data/inputs/positions.csv",
        help="CSV with columns: ticker,quantity,avg_cost,target_weight(optional)",
    )
    p.add_argument(
        "--outdir",
        default="data/processed/monthly_advice",
        help="Output folder for recommendations",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    positions_path = Path(args.positions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    positions = pd.read_csv(positions_path)
    _validate_positions(positions)
    positions["ticker"] = positions["ticker"].str.upper()
    if "target_weight" not in positions.columns:
        positions["target_weight"] = positions["ticker"].map(DEFAULT_TARGET_WEIGHTS)

    tickers = positions["ticker"].tolist()
    px_hist = fetch_history(tickers, lookback_days=220)
    latest_date = px_hist["date"].max()
    latest_prices = (
        px_hist.loc[px_hist["date"] == latest_date, ["ticker", "close"]]
        .set_index("ticker")["close"]
        .to_dict()
    )
    px_21 = price_n_days_ago(px_hist, n_sessions=21)
    mom_63 = momentum_63d(px_hist)

    pos = positions.copy()
    pos["price"] = pos["ticker"].map(latest_prices)
    pos["value"] = pos["quantity"] * pos["price"]
    port_value = float(pos["value"].sum())
    if port_value <= 0:
        raise ValueError("Portfolio value must be > 0")

    pos["current_weight"] = pos["value"] / port_value
    pos["pnl_pct"] = pos["price"] / pos["avg_cost"] - 1.0
    pos["overweight"] = pos["current_weight"] - pos["target_weight"]
    pos["mom_63d"] = pos["ticker"].map(mom_63)

    prev_value = float((pos["quantity"] * pos["ticker"].map(px_21)).sum())
    monthly_return = np.nan if prev_value <= 0 else float(port_value / prev_value - 1.0)

    cfg = RuleConfig()
    paused = (not np.isnan(monthly_return)) and (monthly_return <= cfg.monthly_drawdown_pause)

    sells = build_sell_recommendations(pos, cfg, paused)
    sell_cash = float(sells["sell_value_usd"].sum()) if not sells.empty else 0.0
    buys = build_buy_recommendations(pos, sell_cash, paused)

    actions = build_action_table(pos, sells, buys, paused, monthly_return)
    summary = pd.DataFrame(
        [
            {
                "asof": latest_date.date().isoformat(),
                "portfolio_value_usd": round(port_value, 2),
                "monthly_return_est": None if np.isnan(monthly_return) else round(monthly_return, 6),
                "risk_pause_active": paused,
                "cash_from_sells_usd": round(sell_cash, 2),
                "planned_buys_usd": round(float(buys["buy_value_usd"].sum()) if not buys.empty else 0.0, 2),
            }
        ]
    )

    summary_path = outdir / "summary.csv"
    actions_path = outdir / "actions.csv"
    snapshot_path = outdir / "portfolio_snapshot.csv"
    summary.to_csv(summary_path, index=False)
    actions.to_csv(actions_path, index=False)
    pos.sort_values("ticker").to_csv(snapshot_path, index=False)

    print("Monthly advice generated:")
    print(f"- {summary_path}")
    print(f"- {actions_path}")
    print(f"- {snapshot_path}")
    print(summary.to_string(index=False))


def build_sell_recommendations(pos: pd.DataFrame, cfg: RuleConfig, paused: bool) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, r in pos.iterrows():
        sell_value = 0.0
        reason = "hold"
        if paused:
            reason = "risk_pause_no_rebalance"
        else:
            pnl = float(r["pnl_pct"])
            over = float(r["overweight"])
            excess_value = max(over, 0.0) * float(pos["value"].sum())
            if pnl >= cfg.high_profit_to_trim_more and over >= cfg.overweight_threshold:
                sell_value = excess_value
                reason = "trim_100_excess_high_profit"
            elif pnl >= cfg.min_profit_to_trim and over >= cfg.overweight_threshold:
                sell_value = 0.5 * excess_value
                reason = "trim_50_excess_profit"
            else:
                reason = "hold_not_enough_profit_or_overweight"
        qty = 0.0 if r["price"] <= 0 else min(float(r["quantity"]), sell_value / float(r["price"]))
        rows.append(
            {
                "ticker": r["ticker"],
                "sell_qty": qty,
                "sell_value_usd": qty * float(r["price"]),
                "sell_reason": reason,
            }
        )
    out = pd.DataFrame(rows)
    return out[out["sell_value_usd"] > 1e-6].copy()


def build_buy_recommendations(pos: pd.DataFrame, cash: float, paused: bool) -> pd.DataFrame:
    if paused or cash <= 0:
        return pd.DataFrame(columns=["ticker", "buy_qty", "buy_value_usd", "buy_reason"])
    under = pos.copy()
    under["underweight"] = (under["target_weight"] - under["current_weight"]).clip(lower=0.0)
    under = under[under["underweight"] > 0].copy()
    if under.empty:
        return pd.DataFrame(columns=["ticker", "buy_qty", "buy_value_usd", "buy_reason"])

    # Prioritize best momentum among underweight names; if all <=0 use plain underweight.
    raw_score = under["underweight"] * np.where(under["mom_63d"] > 0, 1.0 + under["mom_63d"], 0.0)
    if raw_score.sum() <= 0:
        alloc = under["underweight"] / under["underweight"].sum()
    else:
        alloc = raw_score / raw_score.sum()

    under["buy_value_usd"] = cash * alloc
    under["buy_qty"] = under["buy_value_usd"] / under["price"]
    under["buy_reason"] = "allocate_to_underweight_with_momentum"
    return under[["ticker", "buy_qty", "buy_value_usd", "buy_reason"]].copy()


def build_action_table(
    pos: pd.DataFrame,
    sells: pd.DataFrame,
    buys: pd.DataFrame,
    paused: bool,
    monthly_return: float,
) -> pd.DataFrame:
    sell_map = sells.set_index("ticker") if not sells.empty else pd.DataFrame()
    buy_map = buys.set_index("ticker") if not buys.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, r in pos.sort_values("ticker").iterrows():
        tk = r["ticker"]
        sell_qty = float(sell_map.loc[tk, "sell_qty"]) if (not sell_map.empty and tk in sell_map.index) else 0.0
        sell_val = float(sell_map.loc[tk, "sell_value_usd"]) if (not sell_map.empty and tk in sell_map.index) else 0.0
        buy_qty = float(buy_map.loc[tk, "buy_qty"]) if (not buy_map.empty and tk in buy_map.index) else 0.0
        buy_val = float(buy_map.loc[tk, "buy_value_usd"]) if (not buy_map.empty and tk in buy_map.index) else 0.0

        if paused:
            action = "HOLD"
            reason = "portfolio_monthly_return_below_pause_threshold"
        elif sell_val > 0 and buy_val > 0:
            action = "SELL_AND_BUY"
            reason = "trim_winner_and_reallocate"
        elif sell_val > 0:
            action = "SELL"
            reason = str(sell_map.loc[tk, "sell_reason"])
        elif buy_val > 0:
            action = "BUY"
            reason = "underweight_and_positive_momentum"
        else:
            action = "HOLD"
            reason = "no_trigger"

        rows.append(
            {
                "ticker": tk,
                "action": action,
                "quantity_current": float(r["quantity"]),
                "avg_cost": float(r["avg_cost"]),
                "price": float(r["price"]),
                "pnl_pct": float(r["pnl_pct"]),
                "target_weight": float(r["target_weight"]),
                "current_weight": float(r["current_weight"]),
                "mom_63d": float(r["mom_63d"]),
                "sell_qty": sell_qty,
                "sell_value_usd": sell_val,
                "buy_qty": buy_qty,
                "buy_value_usd": buy_val,
                "reason": reason,
                "monthly_return_est": monthly_return,
            }
        )
    return pd.DataFrame(rows)


def fetch_history(tickers: Iterable[str], lookback_days: int = 220) -> pd.DataFrame:
    end = datetime.now(UTC).date()
    start = end - timedelta(days=lookback_days * 2)
    rows: list[pd.DataFrame] = []
    for tk in tickers:
        rows.append(_fetch_yahoo_daily(tk, start, end))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def momentum_63d(price_df: pd.DataFrame) -> dict[str, float]:
    mom: dict[str, float] = {}
    for tk, g in price_df.groupby("ticker"):
        s = g.sort_values("date")["close"]
        if len(s) < 64:
            mom[tk] = 0.0
        else:
            mom[tk] = float(s.iloc[-1] / s.iloc[-64] - 1.0)
    return mom


def price_n_days_ago(price_df: pd.DataFrame, n_sessions: int = 21) -> dict[str, float]:
    out: dict[str, float] = {}
    for tk, g in price_df.groupby("ticker"):
        s = g.sort_values("date")["close"].reset_index(drop=True)
        if len(s) <= n_sessions:
            out[tk] = float(s.iloc[0])
        else:
            out[tk] = float(s.iloc[-(n_sessions + 1)])
    return out


def _fetch_yahoo_daily(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    data = resp.json()["chart"]["result"][0]
    ts = data["timestamp"]
    close = data["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"ts": ts, "close": close}).dropna()
    df["date"] = (
        pd.to_datetime(df["ts"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    df["ticker"] = ticker
    return df[["date", "ticker", "close"]]


def _validate_positions(df: pd.DataFrame) -> None:
    required = {"ticker", "quantity", "avg_cost"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"positions csv missing columns: {sorted(missing)}")
    if (df["quantity"] < 0).any():
        raise ValueError("quantity must be >= 0")
    if (df["avg_cost"] <= 0).any():
        raise ValueError("avg_cost must be > 0")


if __name__ == "__main__":
    main()
