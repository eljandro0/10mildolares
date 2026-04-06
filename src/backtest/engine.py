from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backtest.metrics import annualized_sharpe, max_drawdown, rolling_alpha, win_rate
from optimization.engine import optimize_weights
from portfolio.selector import select_assets_independent


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: pd.DataFrame
    orders: pd.DataFrame
    weights: pd.DataFrame


def run_backtest(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    initial_capital: float,
    rebalance: str = "monthly",
    scenario: str = "balanceado",
) -> BacktestResult:
    if rebalance not in {"weekly", "monthly"}:
        raise ValueError("rebalance must be weekly or monthly")

    prices = price_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    benchmark = benchmark_df.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date")
    benchmark["bench_ret"] = benchmark["valor_cuota"].pct_change().fillna(0.0)

    pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = pivot.pct_change().fillna(0.0)

    freq = "W-FRI" if rebalance == "weekly" else "M"
    rebal_dates = returns.resample(freq).last().index
    if len(rebal_dates) == 0:
        raise ValueError("Insufficient price data for rebalancing")

    current_weights = pd.Series(0.0, index=returns.columns)
    orders_rows: list[dict] = []
    weights_rows: list[dict] = []
    port_ret = pd.Series(0.0, index=returns.index)

    for dt in rebal_dates:
        hist = returns.loc[:dt].iloc[-252:]
        if hist.empty:
            continue

        selected = select_assets_independent(
            prices=prices[prices["date"] <= dt],
            catalog=catalog_df,
            n_assets=min(8, len(returns.columns)),
        )
        selected_tk = [t for t in selected["ticker"] if t in hist.columns]
        if not selected_tk:
            continue
        hist_sel = hist[selected_tk]
        b_slice = benchmark.set_index("date").reindex(hist_sel.index)["bench_ret"].fillna(0.0)
        new_weights_local = optimize_weights(hist_sel, b_slice, scenario=scenario)

        full_new_weights = pd.Series(0.0, index=returns.columns)
        full_new_weights.loc[new_weights_local.index] = new_weights_local.values

        for ticker, target_w in full_new_weights.items():
            delta = float(target_w - current_weights.get(ticker, 0.0))
            if abs(delta) < 1e-6:
                continue
            amount = delta * initial_capital
            px = float(pivot.loc[dt, ticker]) if ticker in pivot.columns else np.nan
            qty = amount / px if px and not np.isnan(px) else 0.0
            orders_rows.append(
                {
                    "date": dt.date().isoformat(),
                    "ticker": ticker,
                    "side": "BUY" if delta > 0 else "SELL",
                    "amount_clp": round(abs(amount), 2),
                    "quantity": round(abs(qty), 6),
                }
            )
        current_weights = full_new_weights
        row = {"date": dt}
        row.update({k: float(v) for k, v in current_weights.items()})
        weights_rows.append(row)

    weights_df = pd.DataFrame(weights_rows).set_index("date") if weights_rows else pd.DataFrame(index=returns.index)
    weights_df = weights_df.reindex(returns.index).ffill().fillna(0.0)
    port_ret = (weights_df * returns).sum(axis=1)

    eq_port = (1 + port_ret).cumprod() * initial_capital
    b_ret_aligned = benchmark.set_index("date").reindex(port_ret.index)["bench_ret"].fillna(0.0)
    eq_bench = (1 + b_ret_aligned).cumprod() * initial_capital

    curve = pd.DataFrame(
        {
            "date": port_ret.index,
            "portfolio_value": eq_port.values,
            "benchmark_value": eq_bench.values,
            "portfolio_ret": port_ret.values,
            "benchmark_ret": b_ret_aligned.values,
        }
    )
    curve["alpha_daily"] = curve["portfolio_ret"] - curve["benchmark_ret"]
    curve["alpha_cum"] = (1 + curve["alpha_daily"]).cumprod() - 1
    curve["rolling_alpha_30d"] = rolling_alpha(
        curve.set_index("date")["portfolio_ret"],
        curve.set_index("date")["benchmark_ret"],
        window=30,
    ).values

    metrics = pd.DataFrame(
        [
            {
                "retorno_acumulado": float(eq_port.iloc[-1] / initial_capital - 1),
                "retorno_benchmark": float(eq_bench.iloc[-1] / initial_capital - 1),
                "alpha_total": float((eq_port.iloc[-1] - eq_bench.iloc[-1]) / initial_capital),
                "drawdown": max_drawdown(eq_port),
                "sharpe": annualized_sharpe(port_ret),
                "win_rate_vs_benchmark": win_rate(port_ret, b_ret_aligned),
                "tracking_error_ref": float((port_ret - b_ret_aligned).std(ddof=0) * np.sqrt(252)),
            }
        ]
    )
    orders = pd.DataFrame(orders_rows)
    return BacktestResult(equity_curve=curve, metrics=metrics, orders=orders, weights=weights_df.reset_index())

